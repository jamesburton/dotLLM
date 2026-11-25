#!/usr/bin/env python
"""HF-transformers oracle for the dotLLM Olala CPU forward (issue #1970).

Two modes (run with a Python that has torch (CPU), transformers>=5, einops, safetensors, numpy, gguf):

  tiny  - builds a tiny random Olala (same modeling_olala.py as the real model, every flag the 7A1B uses, tiny dims), writes it as a
          BF16 GGUF with the OVH llama.cpp tensor names, and runs the oracle on it.  Output: <out>/tiny.gguf, <out>/tiny.meta.json,
          <out>/tiny.logits.f32.  Seconds to run; used by the unit tests (mutant proofs) -- generated, never committed.
  real  - runs the real OVHaiLLM/Olala-v1.0-mini-7A1B checkpoint in float32 on CPU on a few prompts.
          With --gguf the four zero-centred mixer norm gains are replaced by (GGUF value - 1) so that the oracle uses the same
          (bf16-rounded) 1 + w the GGUF carries, isolating engine differences from the converter's rounding.

The oracle records, per case: prompt ids, full no-cache logits [L, V], then a greedy continuation of N tokens produced through the
HF *cache* path (prefill with cache, then one-token steps with explicit position_ids) and the logits of every greedy step [N, V].
"""
import argparse, glob, json, math, os, struct, sys
import numpy as np
import torch

HF_DIR_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--OVHaiLLM--Olala-v1.0-mini-7A1B/snapshots/*/")


def snapshot_dir():
    d = sorted(glob.glob(HF_DIR_GLOB))
    if not d:
        sys.exit("OVHaiLLM/Olala-v1.0-mini-7A1B is not in the HF cache")
    return d[0]


def bf16_round(t):
    return t.to(torch.bfloat16).to(torch.float32)


# ----------------------------------------------------------------------------------------------------------------------
# oracle
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_oracle(model, ids, n_greedy):
    L = len(ids)
    ids_t = torch.tensor([ids], dtype=torch.long)
    pos = torch.arange(L).unsqueeze(0)
    full = model(input_ids=ids_t, position_ids=pos, use_cache=False).logits[0].float()

    # cache path: prefill with a cache, then greedy one-token steps with explicit positions
    out = model(input_ids=ids_t, position_ids=pos, use_cache=True)
    cache = out.past_key_values
    prefill_cache_last = out.logits[0, -1].float()
    cur = int(torch.argmax(prefill_cache_last))
    greedy_ids, greedy_logits = [], [prefill_cache_last]
    for j in range(n_greedy):
        greedy_ids.append(cur)
        if j == n_greedy - 1:
            break
        p = L + j
        o = model(input_ids=torch.tensor([[cur]]), position_ids=torch.tensor([[p]]), past_key_values=cache,
                  use_cache=True, cache_position=torch.tensor([p]))
        cache = o.past_key_values
        row = o.logits[0, -1].float()
        greedy_logits.append(row)
        cur = int(torch.argmax(row))
    greedy_logits = torch.stack(greedy_logits)
    # self-consistency of HF: the one-token STEP path (decode kernels) against the chunked no-cache forward, teacher-forced over
    # the second half of the prompt. Both paths are mathematically the same recurrence; their agreement is what lets the single
    # oracle serve prefill and decode comparisons.
    split = max(2, L // 2)   # HF treats a 1-token prefill as a decode step with no state
    o = model(input_ids=ids_t[:, :split], position_ids=pos[:, :split], use_cache=True)
    cache = o.past_key_values
    drift = 0.0
    for i in range(split, L):
        o = model(input_ids=ids_t[:, i:i + 1], position_ids=torch.tensor([[i]]), past_key_values=cache, use_cache=True,
                  cache_position=torch.tensor([i]))
        cache = o.past_key_values
        drift = max(drift, float((o.logits[0, -1].float() - full[i]).abs().max()))
    return full.numpy(), greedy_ids, greedy_logits.numpy(), drift


def write_case(out_dir, name, vocab, ids, full, greedy_ids, greedy_logits, extra=None):
    meta = {"name": name, "vocab": vocab, "ids": ids, "n_full": len(ids), "greedy_ids": greedy_ids, "n_greedy": len(greedy_ids)}
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, f"{name}.meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(out_dir, f"{name}.logits.f32"), "wb") as f:
        f.write(np.ascontiguousarray(full, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(greedy_logits, dtype="<f4").tobytes())


# ----------------------------------------------------------------------------------------------------------------------
# tiny model
# ----------------------------------------------------------------------------------------------------------------------
TINY = dict(
    hidden_size=64, num_hidden_layers=5, layers_config="MVMMV", vocab_size=96, mamba_headdim=16, mamba_d_state=16,
    mamba_mimo_dim=2, mamba_ngroups=1, head_dim=16, num_attention_heads=8, num_signal_heads_diff=6, tpa_rank=2,
    moe_num_routed_experts=8, moe_num_active_experts=3, moe_routed_input_dim=32, moe_routed_intermediate_size=32,
    moe_shared_intermediate_size=48, slw_wsize=5, sliding_window_size=64, max_position_embeddings=128,
    bos_token_id=1, eos_token_id=2, pad_token_id=0,
)


def model_dir(variant):
    """Directory holding the HF code. 'hf' = the shipped snapshot. 'strict' = a copy whose attention query is NOT rounded to bf16.

    modeling_olala.py casts the query to bf16 before attention and the (bf16) attention output is returned in bf16 (eager_attention_forward
    keeps query.dtype). That is an HF/flash-attn artifact worth ~1e-3 in logits; dotLLM computes the query and the attention output in F32
    (K and V stay BF16 like HF's cache), so 'strict' removes only that one cast and gives an oracle that a correct F32 engine can match to
    float rounding, while 'hf' measures how far the shipped numerics are from it.
    """
    snap = snapshot_dir()
    if variant == "hf":
        return snap
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".docs", "olala-hf-strict")
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    import shutil
    for f in ("config.json", "configuration_olala.py", "generation_config.json"):
        shutil.copy(os.path.join(snap, f), os.path.join(out, f))
    src = open(os.path.join(snap, "modeling_olala.py"), encoding="utf-8").read()
    needle = "            query_states.bfloat16(),\n"
    assert src.count(needle) == 1, "modeling_olala.py changed; update the strict patch"
    open(os.path.join(out, "modeling_olala.py"), "w", encoding="utf-8").write(src.replace(needle, "            query_states.float(),\n"))
    return out


def make_model(variant, overrides=None, seed=0):
    from transformers import AutoConfig, AutoModelForCausalLM
    d = model_dir(variant)
    cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.float32), cfg


def build_tiny(seed, variant='strict'):
    model, cfg = make_model(variant, TINY, seed)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("geodesic_mixer.scale", "geodesic_mlp.scale")):
                p.copy_(torch.empty_like(p).uniform_(0.6, 1.6, generator=g))
            elif name.endswith(("geodesic_mixer.bias", "geodesic_mlp.bias")):
                p.copy_(torch.randn(p.shape, generator=g) * 0.15)
            elif name.endswith("softmax_scaler"):
                p.copy_(torch.empty_like(p).uniform_(0.5, 2.0, generator=g))
            elif name.endswith(".norm.weight"):          # zero-centred gains
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            elif name.endswith(("B_bias", "C_bias")):
                p.copy_(torch.randn(p.shape, generator=g) * 0.4)
            elif name.endswith(("in_proj_mimo_x", "in_proj_mimo_z", "out_proj_mimo")):
                p.copy_(torch.randn(p.shape, generator=g) * 0.6 + 0.3)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.6 - 1.0)
            elif name.endswith(".D"):
                p.copy_(torch.randn(p.shape, generator=g))
            elif name.endswith("moe_gate.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif p.ndim >= 2:
                std = 0.5 if "embedding" in name or "lm_head" in name else 0.3
                p.copy_(torch.randn(p.shape, generator=g) * std)
        for name, b in model.named_buffers():
            if name.endswith("expert_bias"):
                b.copy_(torch.randn(b.shape, generator=g) * 0.15)
        # bf16-representable weights so the BF16 GGUF carries exactly what HF computes with
        for name, p in model.named_parameters():
            if p.ndim >= 2 and "moe_gate" not in name and not name.endswith(("B_bias", "C_bias", "in_proj_mimo_x", "in_proj_mimo_z", "out_proj_mimo")):
                p.copy_(bf16_round(p))
            elif name.endswith(".norm.weight"):
                p.copy_(bf16_round(p))
    model.eval()
    return model, cfg


def gguf_name_map(il, is_ssm):
    base = f"model.layers.{il}."
    m = {
        base + "geodesic_mixer.scale": f"blk.{il}.geo_mixer_scale", base + "geodesic_mixer.bias": f"blk.{il}.geo_mixer_bias",
        base + "geodesic_mlp.scale": f"blk.{il}.geo_mlp_scale", base + "geodesic_mlp.bias": f"blk.{il}.geo_mlp_bias",
        base + "mixer_proj.weight": f"blk.{il}.attn_output.weight",
        base + "mlp.moe_gate.weight": f"blk.{il}.ffn_gate_inp.weight", base + "mlp.expert_bias": f"blk.{il}.exp_probs_b",
        base + "mlp.down_proj.weight": f"blk.{il}.ffn_latent_down.weight", base + "mlp.up_proj.weight": f"blk.{il}.ffn_latent_up.weight",
        base + "mlp.experts.experts.weight": f"blk.{il}.ffn_up_exps.weight",
        base + "mlp.experts.output_experts.weight": f"blk.{il}.ffn_down_exps.weight",
        base + "mlp.shared_experts.fc_1.weight": f"blk.{il}.ffn_up_shexp.weight",
        base + "mlp.shared_experts.fc_2.weight": f"blk.{il}.ffn_down_shexp.weight",
    }
    mx = base + "mixer."
    if is_ssm:
        m.update({
            mx + "in_proj.weight": f"blk.{il}.ssm_in.weight", mx + "in_proj_dyn.weight": f"blk.{il}.ssm_in_dyn.weight",
            mx + "B_norm.norm.weight": f"blk.{il}.ssm_b_norm.weight", mx + "C_norm.norm.weight": f"blk.{il}.ssm_c_norm.weight",
            mx + "B_bias": f"blk.{il}.ssm_b_bias", mx + "C_bias": f"blk.{il}.ssm_c_bias", mx + "dt_bias": f"blk.{il}.ssm_dt_bias",
            mx + "D": f"blk.{il}.ssm_d", mx + "in_proj_mimo_x": f"blk.{il}.ssm_mimo_x", mx + "in_proj_mimo_z": f"blk.{il}.ssm_mimo_z",
            mx + "out_proj_mimo": f"blk.{il}.ssm_mimo_o",
        })
    else:
        m.update({
            mx + "c_q.weight": f"blk.{il}.attn_q.weight", mx + "W_A_k.weight": f"blk.{il}.attn_wa_k.weight",
            mx + "W_A_v.weight": f"blk.{il}.attn_wa_v.weight", mx + "W_B_k.weight": f"blk.{il}.attn_wb_k.weight",
            mx + "W_B_v.weight": f"blk.{il}.attn_wb_v.weight", mx + "shift_proj_k.weight": f"blk.{il}.attn_shift_k.weight",
            mx + "shift_proj_v.weight": f"blk.{il}.attn_shift_v.weight", mx + "lambda_proj.weight": f"blk.{il}.attn_lambda.weight",
            mx + "softmax_scaler": f"blk.{il}.attn_softmax_scaler", mx + "q_norm.norm.weight": f"blk.{il}.attn_q_norm.weight",
            mx + "k_norm.norm.weight": f"blk.{il}.attn_k_norm.weight", base + "gate_proj.weight": f"blk.{il}.attn_gate.weight",
        })
    return m


def write_tiny_gguf(model, cfg, path, bf16=True):
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType as Q
    from gguf.quants import quantize
    w = GGUFWriter(path, "olala")
    L = cfg.num_hidden_layers
    H = cfg.hidden_size
    d_inner = 2 * H
    nh_ssm = d_inner // cfg.mamba_headdim
    split = int(cfg.mamba_d_state * 0.5)
    split -= split % 2
    angles = split // 2
    signal = cfg.num_signal_heads_diff
    noise = cfg.num_attention_heads - signal
    w.add_block_count(L); w.add_context_length(int(cfg.max_position_embeddings)); w.add_embedding_length(H)
    w.add_feed_forward_length(0); w.add_head_count(cfg.num_attention_heads); w.add_head_count_kv(noise)
    w.add_key_length(cfg.head_dim); w.add_value_length(cfg.head_dim); w.add_layer_norm_rms_eps(float(cfg.norm_epsilon))
    w.add_file_type(32)
    w.add_sliding_window(int(cfg.slw_wsize)); w.add_attn_logit_softcapping(float(cfg.softcap_attn))
    w.add_expert_count(cfg.moe_num_routed_experts); w.add_expert_used_count(cfg.moe_num_active_experts)
    w.add_expert_feed_forward_length(cfg.moe_routed_intermediate_size)
    w.add_expert_shared_feed_forward_length(cfg.moe_shared_intermediate_size); w.add_expert_shared_count(1)
    w.add_expert_weights_scale(float(cfg.moe_routed_scaling_factor)); w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_moe_latent_size(cfg.moe_routed_input_dim)
    w.add_ssm_inner_size(d_inner); w.add_ssm_state_size(cfg.mamba_d_state); w.add_ssm_group_count(cfg.mamba_ngroups)
    w.add_ssm_time_step_rank(cfg.mamba_headdim)
    w.add_uint32("olala.mamba_mimo_dim", cfg.mamba_mimo_dim); w.add_uint32("olala.mamba_headdim", cfg.mamba_headdim)
    w.add_uint32("olala.num_signal_heads", signal); w.add_uint32("olala.num_noise_heads", noise)
    w.add_uint32("olala.tpa_rank", cfg.tpa_rank); w.add_uint32("olala.num_rope_angles", angles)
    w.add_string("olala.layer_types", cfg.layers_config)
    w.add_bool("olala.zero_centered_gamma", True); w.add_float32("olala.gate_bias", 1.15 if cfg.zero_centered_gate else 0.0)
    w.add_bool("olala.gate_attn", True); w.add_bool("olala.gate_gdn", True); w.add_bool("olala.geodesic_update", True)
    w.add_bool("olala.final_norm", False)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("qwen2")
    w.add_token_list([f"t{i}" for i in range(cfg.vocab_size)]); w.add_token_types([1] * cfg.vocab_size); w.add_token_merges([])

    sd = {k: v for k, v in model.state_dict().items()}
    for k, v in dict(model.named_buffers()).items():
        sd[k] = v

    def add(name, t, force_f32=False):
        t = t.detach().to(torch.float32).contiguous()
        if bf16 and not force_f32 and t.ndim >= 2:
            q = quantize(t.numpy(), Q.BF16)
            w.add_tensor(name, q, raw_dtype=Q.BF16)
        else:
            w.add_tensor(name, t.numpy())

    add("token_embd.weight", sd["model.embedding.weight"])
    add("output.weight", sd["lm_head.weight"])
    for il in range(L):
        is_ssm = cfg.layers_config[il] == "M"
        for hf, gg in gguf_name_map(il, is_ssm).items():
            t = sd[hf]
            if hf.endswith(".norm.weight") and ".mixer." in hf:
                t = 1.0 + t.to(torch.float32)              # zero-centred gamma folded by the converter
            if t.ndim == 0:
                t = t.reshape(1)
            small_f32 = hf.endswith(("moe_gate.weight", "B_bias", "C_bias", "in_proj_mimo_x", "in_proj_mimo_z", "out_proj_mimo"))
            add(gg, t, force_f32=small_f32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def cmd_tiny(a):
    os.makedirs(a.out, exist_ok=True)
    model, cfg = build_tiny(a.seed, "strict")
    write_tiny_gguf(model, cfg, os.path.join(a.out, "tiny.gguf"))
    hf_model, _ = make_model("hf", TINY, a.seed)
    hf_model.load_state_dict(model.state_dict())
    hf_model.eval()
    rng = np.random.RandomState(a.seed)
    seqs = {"": [int(x) for x in rng.randint(3, cfg.vocab_size, size=a.length)],
            "_short": [int(x) for x in rng.randint(3, cfg.vocab_size, size=3)]}      # shorter than the window: unclamped log(pos+1)
    for suffix, ids in seqs.items():
        for tag, m in (("tiny", model), ("tiny-hfbf16", hf_model)):
            full, gids, glog, drift = run_oracle(m, ids, a.greedy)
            print(f"{tag}{suffix}: HF step-vs-chunk max|dlogit| = {drift:.3e}")
            write_case(a.out, tag + suffix, cfg.vocab_size, ids, full, gids, glog, {"hf_step_drift": drift, "window": int(cfg.slw_wsize)})
    print("wrote", a.out)


# ----------------------------------------------------------------------------------------------------------------------
# real model
# ----------------------------------------------------------------------------------------------------------------------
PROMPTS = {
    "capital": "The capital of France is",
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "chat": None,   # chat template
}


def patch_norms_from_gguf(model, gguf_path):
    import gguf
    r = gguf.GGUFReader(gguf_path)
    by = {t.name: t for t in r.tensors}
    n = 0
    with torch.no_grad():
        for il, layer in enumerate(model.model.layers):
            mx = layer.mixer
            pairs = []
            if hasattr(mx, "B_norm"):
                pairs = [(mx.B_norm.norm.weight, f"blk.{il}.ssm_b_norm.weight"), (mx.C_norm.norm.weight, f"blk.{il}.ssm_c_norm.weight")]
            else:
                pairs = [(mx.q_norm.norm.weight, f"blk.{il}.attn_q_norm.weight"), (mx.k_norm.norm.weight, f"blk.{il}.attn_k_norm.weight")]
            for p, name in pairs:
                g = torch.from_numpy(np.array(by[name].data, dtype=np.float32)).reshape(p.shape)
                p.copy_(g - 1.0)
                n += 1
    print(f"patched {n} norm gains from the GGUF (1 + w as stored)")


def load_real(variant):
    """Real checkpoint in float32 on CPU: build on the meta device, materialise, stream the bf16 safetensors tensors into it."""
    from safetensors import safe_open
    with torch.device("meta"):
        model, _ = make_model(variant)
    model = model.to_empty(device="cpu")
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    seen = set()
    with safe_open(os.path.join(snapshot_dir(), "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k not in params:
                print("  (unused checkpoint tensor)", k)
                continue
            with torch.no_grad():
                params[k].copy_(f.get_tensor(k).to(torch.float32))
            seen.add(k)
    missing = [k for k in params if k not in seen and not k.endswith(("tokens_per_expert",))]
    if missing:
        sys.exit(f"checkpoint lacks {len(missing)} tensors, e.g. {missing[:4]}")
    model.eval()
    return model


def cmd_real(a):
    from transformers import AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    os.environ.setdefault("OLALA_TORCH_MAMBA3", "1")
    tok = AutoTokenizer.from_pretrained(snapshot_dir())
    model = load_real(a.variant)
    suffix = a.variant
    if a.gguf:
        patch_norms_from_gguf(model, a.gguf)
        suffix += "-gguf-norms"
    for name, text in PROMPTS.items():
        if name == "chat":
            ids = tok.apply_chat_template([{"role": "user", "content": "Say hello in French."}], add_generation_prompt=True, tokenize=True)
            if isinstance(ids, dict) or hasattr(ids, "keys"):
                ids = ids["input_ids"]
        else:
            ids = tok(text, add_special_tokens=False)["input_ids"]
        ids = [int(i) for i in ids]
        print(name, len(ids), "tokens", flush=True)
        full, gids, glog, drift = run_oracle(model, ids, a.greedy)
        print(f"  HF step-vs-chunk max|dlogit| = {drift:.3e}; greedy: {tok.decode(gids)!r}", flush=True)
        write_case(a.out, f"real-{suffix}-{name}", model.config.vocab_size, ids, full, gids, glog, {"hf_step_drift": drift, "text": tok.decode(gids)})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiny"); t.add_argument("--out", required=True); t.add_argument("--seed", type=int, default=7)
    t.add_argument("--length", type=int, default=12); t.add_argument("--greedy", type=int, default=10)
    r = sub.add_parser("real"); r.add_argument("--out", required=True); r.add_argument("--gguf"); r.add_argument("--greedy", type=int, default=24)
    r.add_argument("--threads", type=int, default=16); r.add_argument("--variant", choices=["strict", "hf"], default="strict")
    a = ap.parse_args()
    {"tiny": cmd_tiny, "real": cmd_real}[a.cmd](a)


if __name__ == "__main__":
    main()
