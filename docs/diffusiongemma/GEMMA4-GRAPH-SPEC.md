# Gemma-4 / DiffusionGemma forward-graph spec (source-confirmed)

Authoritative source: `danielhanchen/llama.cpp@diffusion-visual-updates` (PR ggml-org/llama.cpp#24423,
the branch Unsloth's GGUFs were converted from). Files: `src/models/gemma4.cpp`,
`gemma4-common.h`, `diffusion-gemma.cpp`, `src/llama-arch.cpp`, `src/llama-hparams.cpp`,
`src/llama-graph.cpp`, `conversion/{gemma,diffusion_gemma,base}.py`.

Real 26B-A4B shape: 30 layers, n_embd=2816, n_head=16, n_expert=128, top-8, expert_ff=704,
dense_ff=2112, final_softcap=30, eps=`attention.layernorm_rms_epsilon`.
Global (full-attn) layers = [5,11,17,23,29] (sliding_window_pattern=False); rest are sliding/SWA.

## RMSNorm convention — NO +1
Gemma4 overrides Gemma3's norm_shift to 0.0 (gemma.py:621-623). Inference op is plain
`rms_norm(x, eps) * w_gguf` with NO +1 offset. (Gemma3 baked +1 into weights; Gemma4 does NOT.)
**For GGUF gemma4/diffusion-gemma load, the norm weights must NOT get +1 added.**

## Per-layer head dims / rope (hparams.cpp:85-124)
- Global (non-SWA): head_dim 512, n_head_kv 2, n_rot 512 but PARTIAL rope = first 64 pairs (128 dims)
  rotate (partial_rotary_factor 0.25), freq_base 1e6, freq_factors=rope_freqs([1.0]×64,1e30×192).
- Sliding (SWA): head_dim 256, n_head_kv 8, n_rot 256 (full rotation), freq_base 1e4, no freq_factors.
- n_head=16 (Q) all layers. Equivalent: global RoPE PartialRotaryFactor=0.25 over head_dim 512.

## Attention — softmax scale = 1.0 (NOT 1/sqrt(d))
f_attention_scale=1.0 (gemma4.cpp:11). q_norm/k_norm make Q,K unit so scale is 1.0.

## V-from-K on V-less layers + weight-less V-norm (gemma4.cpp:241-272, gemma4-common.h:35-66)
Per layer:
```
Qcur = wq·cur; reshape(hd,16,T); Qcur = rms(Qcur)*attn_q_norm; Qcur = rope(Qcur, n_rot, base, ff)
Kcur = wk·cur
Vcur = wv ? (wv·cur) : Kcur          # V branches off RAW K projection when wv absent (global layers)
reshape K,V to (hd, n_head_kv, T)
Kcur = rms(Kcur)*attn_k_norm
Vcur = rms(Vcur)                     # WEIGHT-LESS rms norm (no scale), all layers; V NOT roped
Kcur = rope(Kcur, n_rot, base, ff)
attn = softmax(Qᵀ·K * 1.0 + mask) · V ; GQA broadcast 16 over (2|8)
cur = attn_output · concat_heads
cur = rms(cur)*attn_post_norm        # post_attention_norm
attn_out = cur + inpL                 # post-attn residual
```

## DUAL FFN — dense MLP and MoE in PARALLEL on attn_out, summed (gemma4.cpp:289-356)
Both branches read attn_out (NOT each other). All layers in 26B are MoE layers.
```
# Branch A (dense "shared expert"):
cur_mlp = rms(attn_out)*ffn_norm
cur_mlp = down( gelu_tanh(gate·cur_mlp) * (up·cur_mlp) )   # GeGLU, dense ff=2112
cur_mlp = rms(cur_mlp)*post_ffw_norm_1
# Branch B (128-expert MoE):
cur_moe = rms(attn_out)*pre_ffw_norm_2
# custom router (on attn_out, not cur_moe):
tmp = rms(attn_out); tmp *= 1/sqrt(n_embd); tmp *= ffn_gate_inp_s   # ffn_gate_inp.scale [n_embd]
logits = ffn_gate_inp · tmp                                         # [128,T]
cur_moe = MoE(cur_moe, logits): softmax→top8→RENORM weights(sum1,clamp 6.1e-5)
          → per expert gelu_tanh(gate)*up → down → *ffn_down_exps_s[expert] → *weight → sum
cur_moe = rms(cur_moe)*post_ffw_norm_2
# combine:
cur = cur_mlp + cur_moe
cur = rms(cur)*post_ffw_norm          # wraps the SUM, pre-residual
cur = cur + attn_out
# per-layer scale (LAST op):
cur = cur * layer_output_scale        # scalar [1], broadcast over n_embd
inpL = cur
```

## Fused gate_up experts split — CONCATENATED HALVES, gate first (base.py:626-627, graph:1634-38)
ffn_gate_up_exps[2816,1408,128]: rows [0,704)=gate, [704,1408)=up. n_ff_exp=704. NOT interleaved.
ffn_down_exps[704,2816,128]; ffn_down_exps.scale[128] applied per-expert after down-proj.

## Activation: tanh-approx GeGLU (ggml_geglu, GGML_GLU_OP_GEGLU) both dense + experts.

## Post-stack (gemma4.cpp:401-441)
cur = rms(inpL)*output_norm ; logits = output·cur (tied to token_embd) ; softcap 30·tanh(logits/30).
Input embed: tok_embd[ids]·sqrt(n_embd) once.

## DiffusionGemma UNIFIED single-forward (diffusion-gemma.cpp build graph) — IMPLEMENTATION TARGET

The reference no-cache UNIFIED forward (ignore the PKV prefill/decode cache optimization — that
just reproduces this). Split `P = n_tokens - canvas_length`; canvas = last `canvas_length` rows.
Backbone is the EXACT gemma4 layer (attn_norm, gemma4 Q/K/V with V-from-K + weight-less V-norm +
attn scale 1.0 + partial rope, attn_post_norm, residual, dual dense+MoE FFN, ffn_post_norm,
residual). Only THREE things are region-aware + (optional) self-conditioning:

1. **Region embedding** (before layers): `inpL = tok_embd[ids] * sqrt(n_embd)`, then
   - prompt rows [0,P): unchanged (scaled embedding).
   - canvas rows [P,n_tokens): `rms_norm(canvas, eps)` **no scale** (zero-SC path).  [diffusion-gemma.cpp:363-365,378-386]
   With self-conditioning ON (NOT step 0): `canvas = rms_noscale(canvas + sc_signal)` where
   `sc_signal = sc_down(gelu_tanh(sc_gate·n)*(sc_up·n))`, `n = rms(softmax(prev_logits/sc_temp)·tok_embd·sqrt(n_embd))*sc_pre_norm`, gated by `sc_use∈{0,1}` (0 on first step). DEFER — first step / single forward is zero-SC.

2. **Region per-layer scalar** (replaces gemma4's single `cur * layer_output_scale`, applied at the
   SAME position = after ffn residual, last per-layer op): prompt rows `× enc_layer_output_scale`,
   canvas rows `× layer_output_scale`. Same backbone weights; encoder only contributes the scalar.  [diffusion-gemma.cpp:475-489]

3. **Mask**: region-aware additive. Prompt query: causal over earlier prompt, never canvas. Canvas
   query: bidirectional over all (global) / last (n_swa-1) prompt + all canvas (sliding). For
   n_tokens < n_swa (=1024) the sliding clip is a no-op ⇒ identical to **Hybrid(prefixLength=P)**
   which dotLLM already supports. Use Hybrid(P) for validation (short canvas).  [diffusion-gemma.cpp:34-66]

Final: output_norm, lm_head (tied to token_embd), softcap 30 — same as gemma4. attention.causal=False.
New tensors to load: per-layer `enc_layer_output_scale` [1]; model-level `self_cond_pre_norm/gate/up/down` (defer use).
P comes from the diffusion canvas split = AttentionMaskSpec.Hybrid prefixLength (prompt length).

## DiffusionGemma deltas (diffusion-gemma.cpp) — backbone identical to gemma4
- attention.causal=False: unified [prompt|canvas] forward, custom additive mask. Prompt queries causal
  (SWA-clipped on sliding); canvas queries bidirectional over all (global) / last n_swa-1 prompt+canvas
  (sliding). Split P = n_tokens - canvas_length(256).
- Region embed: prompt rows = embed·sqrt(n_embd); canvas rows = rms_noscale(embed·sqrt(n_embd)[+SC]).
- Region per-layer scalar: prompt rows *= enc_layer_output_scale; canvas rows *= layer_output_scale.
  SAME weights — encoder contributes ONLY its layer_scalar buffers. No separate weight set.
- Self-conditioning (self_cond_pre_norm/gate/up/down, model-level, OFF on step 0): gated GeGLU MLP on
  softmax(prev_canvas_logits)·tok_embd·sqrt(n_embd), rms_pre_norm*self_cond_pre_norm, added to canvas
  embed, then rms_noscale. sc_use∈{0,1} gate.
- Final softcap 30 same.

## Implementation deltas vs current dotLLM Gemma path
NEW: (1) V-from-K when wv missing; (2) weight-less V-norm; (3) attn scale=1.0; (4) dual dense+MoE
FFN parallel with 5 norms; (5) fused gate_up expert split; (6) custom router (1/sqrt(n_embd)+gate_inp_s);
(7) per-expert down scale; (8) layer_output_scale; (9) global partial rope 0.25 over head_dim 512;
(10) NO +1 on gemma4 GGUF norm weights. EXISTING-reusable: GeGLU, softcap, dual head dim, per-attn rope,
GQA, sliding pattern. DiffusionGemma additionally needs region-aware embed/scalar + self-cond (later).
