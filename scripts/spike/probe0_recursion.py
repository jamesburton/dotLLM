"""Probe 0 - zero-training spike: is recursive re-feeding of activations through a
slab of BitNet's ternary decoder layers numerically stable?

We load the bf16 BitNet master (microsoft/bitnet-b1.58-2B-4T-bf16) via HF
transformers. Its BitLinear applies ternary absmean weight quant + int8 activation
quant with STE on EVERY forward, so looping the decoder layers genuinely exercises
the ternary re-feed path (train==serve form).

Manual forward:
  embed -> layers[0:p) -> LOOP layers[p:q) N times -> layers[q:L) -> norm -> lm_head

For N=1 the loop runs the slab once => identical to the stock model (sanity check).

Metrics over a few fixed prompts and N in {1,2,4,8}:
  (a) residual-stream L2 norm at slab output (mean over positions) per loop iter
  (b) next-token cross-entropy / perplexity (teacher forcing on the prompt)
  (c) final-token output entropy + a greedy-decode degeneration / repetition check

Simplifications (see .docs/probe0_recursion.md "Caveats"):
  - single full-sequence forward, NO KV cache; positions are the natural [0..seq)
    on every looped pass (the slab re-sees the same position ids each loop).
  - eager attention with an explicit additive causal mask.
"""

import os
import sys
import math
import json

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HOME", "E:/.cache/huggingface")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitNetForCausalLM

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"

PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is",
    "In a shocking discovery, scientists found that the moon is slowly",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Once upon a time there was a small dragon who loved to collect",
    "The three laws of thermodynamics describe how energy behaves. The first law states that",
]

N_VALUES = [1, 2, 4, 8]
GEN_TOKENS = 24


def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1e9
    # bf16 2B weights ~5GB; need comfortable headroom. If something else is
    # hogging the card, fall back to CPU.
    if free_gb < 6.0:
        print(f"[device] only {free_gb:.1f} GB free on GPU -> using CPU", flush=True)
        return "cpu"
    print(f"[device] {free_gb:.1f} GB free on GPU -> using cuda", flush=True)
    return "cuda"


def build_causal_mask(seq, dtype, device):
    min_val = torch.finfo(dtype).min
    mask = torch.full((seq, seq), min_val, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


@torch.no_grad()
def manual_forward(model, input_ids, p, q, n_loops, capture_norms=False):
    """Run embed -> [0:p) -> loop[p:q) n_loops times -> [q:L) -> norm -> lm_head.

    Returns (logits, slab_norms) where slab_norms is a list of mean-over-position
    residual L2 norms measured at the slab output after each loop iteration.
    """
    base = model.model
    device = input_ids.device
    seq = input_ids.shape[1]

    hidden = base.embed_tokens(input_ids)
    dtype = hidden.dtype

    position_ids = torch.arange(seq, device=device).unsqueeze(0)
    position_embeddings = base.rotary_emb(hidden, position_ids)
    attn_mask = build_causal_mask(seq, dtype, device)

    layer_kwargs = dict(
        attention_mask=attn_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        cache_position=torch.arange(seq, device=device),
        position_embeddings=position_embeddings,
    )

    layers = base.layers

    # prefix
    for layer in layers[:p]:
        hidden = layer(hidden, **layer_kwargs)

    # looped slab
    slab_norms = []
    for _ in range(n_loops):
        for layer in layers[p:q]:
            hidden = layer(hidden, **layer_kwargs)
        if capture_norms:
            # mean over positions of per-token L2 norm
            tok_norm = hidden.float().norm(dim=-1)  # (1, seq)
            slab_norms.append(tok_norm.mean().item())

    # suffix
    for layer in layers[q:]:
        hidden = layer(hidden, **layer_kwargs)

    hidden = base.norm(hidden)
    logits = model.lm_head(hidden)
    return logits, slab_norms


@torch.no_grad()
def teacher_forcing_ppl(logits, input_ids):
    """CE / perplexity of predicting token t+1 from position t over the prompt."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )
    return ce.item(), math.exp(min(ce.item(), 20.0))


@torch.no_grad()
def final_token_entropy(logits):
    probs = F.softmax(logits[0, -1, :].float(), dim=-1)
    ent = -(probs * torch.log(probs + 1e-12)).sum().item()
    return ent


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, p, q, n_loops, gen_tokens):
    """Greedy decode by recomputing the full sequence each step (no cache)."""
    ids = input_ids.clone()
    new_tokens = []
    for _ in range(gen_tokens):
        logits, _ = manual_forward(model, ids, p, q, n_loops, capture_norms=False)
        nxt = int(logits[0, -1, :].argmax().item())
        new_tokens.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if tokenizer.eos_token_id is not None and nxt == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return new_tokens, text


def repetition_metrics(tokens):
    """Return (distinct_ratio, max_run, looped_flag).

    looped_flag: heuristic degeneration detector - low distinct-token ratio OR a
    long immediate run OR a short repeating cycle dominating the tail.
    """
    if not tokens:
        return 1.0, 0, False
    distinct_ratio = len(set(tokens)) / len(tokens)

    max_run = 1
    cur = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1

    # detect short repeating cycle (period 1..4) covering the last >=8 tokens
    cycle_flag = False
    tail = tokens[-12:]
    for period in range(1, 5):
        if len(tail) >= 2 * period:
            seg = tail[-period:]
            reps = 0
            j = len(tail) - period
            while j - period >= 0 and tail[j - period:j] == seg:
                reps += 1
                j -= period
            if reps >= 2:
                cycle_flag = True
                break

    looped = distinct_ratio < 0.5 or max_run >= 5 or cycle_flag
    return distinct_ratio, max_run, looped


def main():
    device = pick_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[load] {MODEL_ID} dtype={dtype} device={device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = BitNetForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    L = len(model.model.layers)
    p = L // 4
    q = (3 * L) // 4
    print(f"[arch] L={L} layers; slab = [{p}:{q}) ({q - p} layers looped)", flush=True)

    # ---- sanity: N=1 manual forward must match stock model ----
    print("\n[sanity] comparing N=1 manual forward vs stock model.forward ...", flush=True)
    sanity_ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        stock = model(sanity_ids).logits
        manual, _ = manual_forward(model, sanity_ids, p, q, 1)
    max_abs = (stock.float() - manual.float()).abs().max().item()
    argmax_match = bool((stock[0, -1].argmax() == manual[0, -1].argmax()).item())
    print(f"[sanity] max_abs_logit_diff={max_abs:.4e}  final_argmax_match={argmax_match}", flush=True)

    # ---- main sweep ----
    rows = []  # (N, mean_slab_norm_last, norm_growth_ratio, mean_ppl, mean_entropy, any_loop_flag)
    per_prompt_detail = []

    for N in N_VALUES:
        norm_lasts = []
        norm_firsts = []
        ppls = []
        ents = []
        loop_flags = []
        detail = {"N": N, "prompts": []}
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            logits, slab_norms = manual_forward(model, ids, p, q, N, capture_norms=True)
            ce, ppl = teacher_forcing_ppl(logits, ids)
            ent = final_token_entropy(logits)
            gen_toks, gen_text = greedy_generate(model, tokenizer, ids, p, q, N, GEN_TOKENS)
            dr, mr, looped = repetition_metrics(gen_toks)

            norm_firsts.append(slab_norms[0])
            norm_lasts.append(slab_norms[-1])
            ppls.append(ppl)
            ents.append(ent)
            loop_flags.append(looped)
            detail["prompts"].append({
                "prompt": prompt[:50],
                "slab_norms": [round(x, 2) for x in slab_norms],
                "ce": round(ce, 4),
                "ppl": round(ppl, 3),
                "entropy": round(ent, 4),
                "distinct_ratio": round(dr, 3),
                "max_run": mr,
                "looped": looped,
                "gen": gen_text[:120],
            })
        mean_norm_last = sum(norm_lasts) / len(norm_lasts)
        mean_norm_first = sum(norm_firsts) / len(norm_firsts)
        growth = mean_norm_last / mean_norm_first if mean_norm_first else float("nan")
        rows.append((
            N,
            round(mean_norm_first, 2),
            round(mean_norm_last, 2),
            round(growth, 3),
            round(sum(ppls) / len(ppls), 3),
            round(sum(ents) / len(ents), 4),
            any(loop_flags),
            sum(loop_flags),
        ))
        per_prompt_detail.append(detail)
        print(f"[sweep] N={N}: slab_norm {mean_norm_first:.1f}->{mean_norm_last:.1f} "
              f"(x{growth:.2f})  ppl={sum(ppls)/len(ppls):.2f}  "
              f"ent={sum(ents)/len(ents):.3f}  degenerate={sum(loop_flags)}/{len(loop_flags)}",
              flush=True)

    # ---- print table ----
    print("\n================ RESULTS TABLE ================")
    header = f"{'N':>3} | {'slab_norm(1st)':>14} | {'slab_norm(last)':>15} | {'growthx':>8} | {'ppl':>9} | {'entropy':>8} | {'degen(prompts)':>14}"
    print(header)
    print("-" * len(header))
    for (N, nf, nl, g, ppl, ent, anyflag, nflag) in rows:
        print(f"{N:>3} | {nf:>14} | {nl:>15} | {g:>8} | {ppl:>9} | {ent:>8} | {str(nflag)+'/'+str(len(PROMPTS)):>14}")

    # dump json for the .md writer / reproducibility
    out = {
        "model": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "L": L, "p": p, "q": q, "slab_layers": q - p,
        "sanity": {"max_abs_logit_diff": max_abs, "final_argmax_match": argmax_match},
        "N_values": N_VALUES,
        "gen_tokens": GEN_TOKENS,
        "table": [
            {"N": N, "slab_norm_first": nf, "slab_norm_last": nl, "growth": g,
             "ppl": ppl, "entropy": ent, "degenerate_prompts": nflag}
            for (N, nf, nl, g, ppl, ent, anyflag, nflag) in rows
        ],
        "detail": per_prompt_detail,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(here, "probe0_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {json_path}")


if __name__ == "__main__":
    main()
