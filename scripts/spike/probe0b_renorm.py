"""Probe 0b - zero-training follow-up to Probe 0.

Probe 0 found that looping a ternary BitNet layer-slab is numerically stable and
N=2 *improves* perplexity, BUT the residual-stream norm inflates ~linearly per loop
(~+123k/pass), which overwhelms the final RMSNorm + ternary activation quant and
breaks greedy decode by N=8.

Question: does inserting a per-loop residual RENORMALIZATION (applied to the hidden
state between slab passes, before re-feeding) hold the loop flat-or-improving out to
N=4, N=8 (and N=16)?

We reuse the exact Probe-0 setup:
  embed -> layers[0:p) -> LOOP layers[p:q) xN -> layers[q:L) -> norm -> lm_head
on microsoft/bitnet-b1.58-2B-4T-bf16 (HF BitNetForCausalLM, ternary BitLinear forward).

Renorm variants tested (renorm is active ONLY between passes, so N=1 == stock for all):
  (none)    baseline, no renorm (reproduces Probe 0, extended to N=16)
  (a)rescale  norm-preserving: after each non-final pass, rescale h per-token so its
              L2 norm equals the slab-INPUT norm of that pass -> residual-stream norm
              held constant across loops.
  (b)rmsnorm  weightless (unit-weight) RMSNorm applied between passes.
  (c)damped   under-relaxed residual: for passes >=2, h <- h_in + gamma*(slab(h_in)-h_in),
              gamma=0.5.

Metrics per (variant, N): slab-output L2 norm growth, mean teacher-forcing PPL,
final-token entropy, greedy-decode degenerate count. Same 5 prompts as Probe 0.
"""

import os
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

N_VALUES = [1, 2, 4, 8, 16]
GEN_TOKENS = 24
GAMMA = 0.5
VARIANTS = ["none", "rescale", "rmsnorm", "damped"]
EPS = 1e-12


def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1e9
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
def manual_forward(model, input_ids, p, q, n_loops, variant, rms_eps,
                   capture_norms=False):
    """embed -> [0:p) -> loop[p:q) xN (with between-pass renorm) -> [q:L) -> norm -> lm_head.

    renorm is applied AFTER each non-final pass (i.e. strictly between passes), so for
    n_loops==1 the result is identical to the stock model for every variant.
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

    def run_slab(h):
        for layer in layers[p:q]:
            h = layer(h, **layer_kwargs)
        return h

    # prefix
    for layer in layers[:p]:
        hidden = layer(hidden, **layer_kwargs)

    # looped slab with between-pass renorm
    slab_norms = []
    for i in range(n_loops):
        h_in = hidden
        if variant == "damped" and i >= 1:
            h_out = run_slab(h_in)
            hidden = h_in + GAMMA * (h_out - h_in)
        else:
            hidden = run_slab(h_in)

        is_last = (i == n_loops - 1)
        if not is_last:
            if variant == "rescale":
                in_norm = h_in.float().norm(dim=-1, keepdim=True)
                cur_norm = hidden.float().norm(dim=-1, keepdim=True)
                hidden = (hidden.float() * (in_norm / (cur_norm + EPS))).to(dtype)
            elif variant == "rmsnorm":
                h = hidden.float()
                rms = h.pow(2).mean(-1, keepdim=True).add(rms_eps).sqrt()
                hidden = (h / rms).to(dtype)
            # "none" and "damped" apply no extra between-pass renorm
        if capture_norms:
            slab_norms.append(hidden.float().norm(dim=-1).mean().item())

    # suffix
    for layer in layers[q:]:
        hidden = layer(hidden, **layer_kwargs)

    hidden = base.norm(hidden)
    logits = model.lm_head(hidden)
    return logits, slab_norms


@torch.no_grad()
def teacher_forcing_ppl(logits, input_ids):
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
def greedy_generate(model, tokenizer, input_ids, p, q, n_loops, variant, rms_eps,
                    gen_tokens):
    ids = input_ids.clone()
    new_tokens = []
    for _ in range(gen_tokens):
        logits, _ = manual_forward(model, ids, p, q, n_loops, variant, rms_eps,
                                   capture_norms=False)
        last = logits[0, -1, :]
        # guard: if the residual stream has inflated enough to produce non-finite
        # logits, decode is meaningless -> stop (repetition_metrics will flag it).
        if not torch.isfinite(last).all():
            break
        nxt = int(last.argmax().item())
        new_tokens.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if tokenizer.eos_token_id is not None and nxt == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return new_tokens, text


def repetition_metrics(tokens):
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
    rms_eps = float(getattr(model.config, "rms_norm_eps", 1e-5))

    L = len(model.model.layers)
    p = L // 4
    q = (3 * L) // 4
    print(f"[arch] L={L} layers; slab = [{p}:{q}) ({q - p} looped); rms_eps={rms_eps}",
          flush=True)

    # ---- sanity: N=1 manual forward must match stock model, for EVERY variant ----
    print("\n[sanity] N=1 manual forward vs stock model.forward (per variant)...", flush=True)
    sanity_ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        stock = model(sanity_ids).logits
    sanity = {}
    for v in VARIANTS:
        with torch.no_grad():
            manual, _ = manual_forward(model, sanity_ids, p, q, 1, v, rms_eps)
        max_abs = (stock.float() - manual.float()).abs().max().item()
        match = bool((stock[0, -1].argmax() == manual[0, -1].argmax()).item())
        sanity[v] = {"max_abs_logit_diff": max_abs, "final_argmax_match": match}
        print(f"  [{v:>8}] max_abs_logit_diff={max_abs:.4e} argmax_match={match}", flush=True)

    # ---- sweep variants x N ----
    all_tables = {}
    all_detail = {}
    for variant in VARIANTS:
        rows = []
        detail = []
        for N in N_VALUES:
            norm_firsts, norm_lasts, ppls, ents, loop_flags = [], [], [], [], []
            pdet = {"N": N, "prompts": []}
            for prompt in PROMPTS:
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                logits, slab_norms = manual_forward(model, ids, p, q, N, variant,
                                                    rms_eps, capture_norms=True)
                ce, ppl = teacher_forcing_ppl(logits, ids)
                ent = final_token_entropy(logits)
                gen_toks, gen_text = greedy_generate(model, tokenizer, ids, p, q, N,
                                                     variant, rms_eps, GEN_TOKENS)
                dr, mr, looped = repetition_metrics(gen_toks)
                norm_firsts.append(slab_norms[0])
                norm_lasts.append(slab_norms[-1])
                ppls.append(ppl)
                ents.append(ent)
                loop_flags.append(looped)
                pdet["prompts"].append({
                    "prompt": prompt[:50],
                    "slab_norms": [round(x, 1) for x in slab_norms],
                    "ce": round(ce, 4), "ppl": round(ppl, 3),
                    "entropy": round(ent, 4), "distinct_ratio": round(dr, 3),
                    "max_run": mr, "looped": looped, "gen": gen_text[:120],
                })
            mnf = sum(norm_firsts) / len(norm_firsts)
            mnl = sum(norm_lasts) / len(norm_lasts)
            growth = mnl / mnf if mnf else float("nan")
            mppl = sum(ppls) / len(ppls)
            ment = sum(ents) / len(ents)
            ndeg = sum(loop_flags)
            rows.append({"N": N, "slab_norm_first": round(mnf, 1),
                         "slab_norm_last": round(mnl, 1), "growth": round(growth, 3),
                         "ppl": round(mppl, 3), "entropy": round(ment, 4),
                         "degenerate_prompts": ndeg})
            detail.append(pdet)
            print(f"[{variant:>8} N={N:>2}] norm {mnf:.0f}->{mnl:.0f} (x{growth:.2f}) "
                  f"ppl={mppl:.2f} ent={ment:.3f} degen={ndeg}/{len(PROMPTS)}", flush=True)
        all_tables[variant] = rows
        all_detail[variant] = detail

    # ---- print tables ----
    for variant in VARIANTS:
        print(f"\n================ VARIANT: {variant} ================")
        hdr = f"{'N':>3} | {'norm(1st)':>11} | {'norm(last)':>11} | {'growthx':>8} | {'ppl':>9} | {'entropy':>8} | {'degen':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in all_tables[variant]:
            print(f"{r['N']:>3} | {r['slab_norm_first']:>11} | {r['slab_norm_last']:>11} | "
                  f"{r['growth']:>8} | {r['ppl']:>9} | {r['entropy']:>8} | "
                  f"{str(r['degenerate_prompts'])+'/'+str(len(PROMPTS)):>7}")

    out = {
        "model": MODEL_ID, "device": device, "dtype": str(dtype),
        "L": L, "p": p, "q": q, "slab_layers": q - p, "rms_eps": rms_eps,
        "gamma": GAMMA, "N_values": N_VALUES, "gen_tokens": GEN_TOKENS,
        "sanity": sanity, "variants": VARIANTS,
        "tables": all_tables, "detail": all_detail,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(here, "probe0b_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {json_path}")


if __name__ == "__main__":
    main()
