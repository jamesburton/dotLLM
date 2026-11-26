# Olala (OVHcloud) — architecture notes

**Status:** CPU reference forward (issue #1970, epic #1969). Vulkan forward: issue #1983 (section 8). CUDA refuses the architecture explicitly until its port lands.
Models: `OVHaiLLM/Olala-v1.0-mini-7A1B` (HF `model_type: olala`, `OlalaForCausalLM`, custom code) and the later translate / summarise / SFT /
base variants that share the arch. GGUF: `OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF` (`general.architecture = olala`).

**Sources.** Semantics: HF `modeling_olala.py` (authoritative; `_mamba3_mimo_torch`, `_mamba3_step_torch`, `OlalaGeodesicNorm`,
`OlalaDifferentialTensorProductAttentionV2`, `OlalaMoE`). GGUF layout and the converter's folds: the OVH llama.cpp fork
(`OVHai-LLM/llama.cpp`, branch `olala-master`: `conversion/olala.py`, `src/models/olala.cpp`, `docs/olala.md`). Upstream llama.cpp,
Ollama and LM Studio cannot load these files; the fork is the only GGUF reference implementation.

## 1. Layer layout (7A1B)

36 blocks, `olala.layer_types = MMMMVMMMMVMMMMVMMMMVMMMMVMMMMVMMMMVM` = **29 M + 7 V**. Hidden 1536, vocab 151936 (Qwen BPE, `pre: qwen2`),
`context_length` 100000, tokenizer chat template embedded. **No final norm.** Token embeddings are **L2-normalised** (`normalize_embeddings_ngpt`),
`output.weight` is untied. Every block has the same shape:

```
x_{in} (unit norm)
  y = mixer(x_in)                      # no pre-norm: the mixer reads the raw residual
  x   = geodesic_mixer(x_in, y)
  y = MoE(x) + shared_expert(x)        # no post-mixer norm
  x_out = geodesic_mlp(x, y)
```

### Geodesic residual (`OlalaGeodesicNorm`, `geodesic_update`)
Rotation of the residual towards the orthogonal component of the update, so the residual norm stays 1 after every block (asserted in
Debug builds):

```
g_perp = g - (x.g / |x|^2) x;   R = |x|;  t = min(|g_perp| / R, pi/4)
theta  = min((t * scale + bias) / (layer + 1), pi/4)          # scale, bias: 4 scalars per block (geo_mixer_*, geo_mlp_*)
out    = x cos(theta) + (g_perp / |g_perp|) R sin(theta)
```
Reductions run in double (`OlalaOps.GeodesicUpdate`): the M-mixer outputs reach ~1e14 (see §3).

### M layer — Mamba-3 MIMO (`OlalaMamba3MimoFast`)
H = 48 heads, P = 64, N = 128, R = 4 (MIMO rank), 32 rotary angle pairs, G = 1 group, no conv (`remove_conv`), trapezoid, data-dependent A.
- `ssm_in` (1536 -> 48 x 131) is **per-head interleaved** `[z(64), x(64), dt, A, trap]`; `ssm_in_dyn` (1536 -> 1056) = `B[R][N]`, `C[R][N]`, 32 raw angles
  (shared by all heads). With G = 1 the `(G r n)` and `(r G n)` layouts coincide, so no test on this model can discriminate them.
- `A = min(-softplus(a), -1e-4)`, `dt = softplus(dt_raw + dt_bias)`, `trap = sigmoid`, `alpha = exp(A dt)`, `beta = (1 - trap) dt alpha`, `gamma = trap dt`.
- B and C: RMSNorm over N (gain `ssm_b_norm`/`ssm_c_norm`, **already `1 + w`** in the GGUF), + per-head bias `[H][R][N]`, then rotated: angle
  `cum += tanh(raw) * dt * pi` per head, pairs `(i, i + N/2)` for `i < 32`.
- Recurrence per head: `h_t = alpha h_{t-1} + beta u_{t-1} + gamma u_t`, `u_t = sum_r k_{t,r} (x_t * MimoX_r)^T`;
  `o_r = h_t q_r + D (x_t * MimoX_r)`; `y = sum_r MimoO_r * o_r * silu(z * MimoZ_r)`. No post-gate norm.
- State (F32, §4): `h[H][P][N]`, angle `[H][32]`, previous token's rotated K `[R][H][N]` and raw x `[H][P]`.
- Output `attn_output` (3072 -> 1536).

### V layer — differential tensor-product attention (not MLA)
48 query heads, 12 KV heads (= noise heads), head_dim 128, TPA rank 4, **no RoPE** (`rope_theta = 0`).
- `K[h] = (A_k[h] @ B_k) / 4`, `V[h] = (A_v[h] @ B_v) / 4` with `A: 1536 -> 12 x 4`, `B: 1536 -> 4 x 128` (so K/V are 12 x 128 per token).
- **Token shift** (position 0 masked): `K~ = a K_prev + (1 - a) K`, `a = sigmoid(shift_proj_k(x))` per KV head, same for V; `K_prev`/`V_prev`
  are the previous token's **raw (pre-shift, pre-norm)** K and V.
- `K~` and `Q` are RMS-normed per head (`attn_k_norm` / `attn_q_norm`, gain already `1 + w`). **Scalable softmax:**
  `Q *= softmax_scaler[h] * log(min(pos + 1, window))` with `window = attention.sliding_window` (HF `slw_wsize`, 32768).
- Scores `Q.K * (1/head_dim)` (completed-P scale, not `1/sqrt(d)`), soft cap `150 tanh(s/150)`, causal window of `window + 1` keys.
- **Differential combine** per KV group (query heads `4g .. 4g+3`): heads 0..2 are signal, head 3 is noise;
  `out[3g + s] = attn[4g + s] - sigmoid(lambda_proj(x)[g]) * attn[4g + 3]` -> 36 heads x 128.
- Output gate `silu(attn_gate(x) + 1.15)` (elementwise, 4608), then `attn_output` (4608 -> 1536).
- HF casts the query and the K/V to bf16 before the kernel and returns the attention output in bf16; dotLLM keeps the query and the output
  F32 and stores K/V as BF16 (see §5).

### MoE (every block, `OlalaMoE`)
`logits = x W_router^T` (F32 router, `ffn_gate_inp`), `s = sigmoid(logits)`, **top-6 of `s + exp_probs_b`**, the selected *un-biased* `s`
normalised to sum 1 (`+1e-20`) and multiplied by 2.45. Routed experts run in a **384-wide latent**: `z = ffn_latent_down(x)`;
per expert `relu(W_up[e] z)^2` (768) then `W_down[e]` (384); weighted sum; `ffn_latent_up` back to 1536. Shared expert:
`ffn_down_shexp(relu(ffn_up_shexp(x))^2)`. Result = routed + shared. ReLU^2 everywhere (no SwiGLU, no gate matrix).

## 2. GGUF tensor map

| GGUF | HF | shape (GGUF order) |
|---|---|---|
| `token_embd.weight`, `output.weight` | `model.embedding`, `lm_head` | 1536 x 151936 |
| `blk.N.geo_{mixer,mlp}_{scale,bias}` | `geodesic_{mixer,mlp}.{scale,bias}` | 1 (F32) |
| `blk.N.attn_output.weight` | `mixer_proj` | M: 3072 x 1536, V: 4608 x 1536 |
| M: `ssm_in`, `ssm_in_dyn` | `mixer.in_proj`, `in_proj_dyn` | 1536 x 6288, 1536 x 1056 |
| M: `ssm_b_norm`, `ssm_c_norm` (`.weight`) | `B_norm.norm`, `C_norm.norm` | 128 (F32, `1 + w`) |
| M: `ssm_b_bias`, `ssm_c_bias` | `B_bias`, `C_bias` | 128 x 4 x 48 (F32) |
| M: `ssm_dt_bias`, `ssm_d` | `dt_bias`, `D` | 48 |
| M: `ssm_mimo_x`, `ssm_mimo_z`, `ssm_mimo_o` | `in_proj_mimo_x`, `_z`, `out_proj_mimo` | 64 x 4 x 48 (F32) |
| V: `attn_q`, `attn_wa_k/v`, `attn_wb_k/v` | `c_q`, `W_A_k/v`, `W_B_k/v` | 1536 x 6144 / 48 / 512 |
| V: `attn_shift_k/v`, `attn_lambda` | `shift_proj_k/v`, `lambda_proj` | 1536 x 12 |
| V: `attn_gate` | block `gate_proj` | 1536 x 4608 |
| V: `attn_q_norm`, `attn_k_norm`, `attn_softmax_scaler` | `q_norm.norm`, `k_norm.norm`, `softmax_scaler` | 128, 128, 48 (F32) |
| `ffn_gate_inp.weight`, `exp_probs_b` | `moe_gate`, `expert_bias` | 1536 x 256, 256 (F32) |
| `ffn_latent_down` / `ffn_latent_up` | `mlp.down_proj` / `up_proj` | 1536 x 384 / 384 x 1536 |
| `ffn_up_exps` / `ffn_down_exps` | `experts.experts` / `output_experts` | 384 x 768 x 256 / 768 x 384 x 256 |
| `ffn_up_shexp` / `ffn_down_shexp` | `shared_experts.fc_1` / `fc_2` | 1536 x 1536 |
| `blk.N.ssm_m3_{bias,mxz,norms,misc}` | (none) | packed duplicates for the fork's GPU kernels — **ignored** |

`OlalaTensors.FindProblems` checks every shape and rejects unknown tensors (cosnet branches, value embeddings, a final norm, skip-layer
pruning, `layer_idx_map` variants all fail loudly instead of running a different model).

## 3. Quantisation: `safe2`, and why

`Olala-v1.0-mini-7A1B-q8_0-safe2.gguf` is produced by the fork's `llama-quantize` from the bf16 file:

```
--tensor-type attn_output=bf16  --tensor-type ffn_down_exps=q6_k  --tensor-type ffn_down_shexp=bf16
--tensor-type ffn_latent_up=bf16 --tensor-type ffn_up_exps=q8_0   <file> q8_0
```
Everything else is Q8_0 (embeddings, lm_head, `ssm_in*`, attention projections, router is F32, `ffn_latent_down`, `ffn_up_shexp`); biases, MIMO
projections, norms, geodesic scalars and the router are F32 and never quantised. The four protected tensors are exactly the ones whose
**inputs are unbounded**: `attn_output` (M-mixer outputs reach ~1e14), `ffn_down_shexp` (ReLU^2 of the shared expert, ~1e9), `ffn_down_exps` (ReLU^2 of the routed
experts, ~1e9, the one that stays quantised: Q6_K uses Q8_K activations with an F32 scale) and `ffn_latent_up`. Type-0 quants (Q8_0/Q5_0/Q4_0) quantise
activations to Q8_0 **with an FP16 scale**, which overflows to NaN on those inputs; `ffn_up_exps` and `ffn_latent_up` have K = 384 which K-quants cannot encode,
so llama-quantize silently falls back to type-0 for them — the reason for the explicit `ffn_up_exps=q8_0` / `ffn_latent_up=bf16` overrides. The 6/5-bit
files add `--imatrix` and the same two overrides. OVH report KLD vs bf16 of 0.0040 (q8_0-safe2, 96.9% same top-1), 0.0106 (q6_k), 0.0194 (q5_k_m) and withdrew the Q4_K_M.

**Consequences for dotLLM (all backends).**
- FP16 activation storage or an FP16 scale on the activation path is **not viable** for this model (ceiling 65,504 vs 1e9-1e14). Activations are F32 on
  CPU; any GPU port must keep BF16 on the mixer-output -> `attn_output` and ReLU^2 -> down paths (BF16 has the F32 exponent range; F32 storage there still needs sign-off under precision rule 1) (the fork forces `GGML_PREC_BF16` on every quantised matmul for this reason).
  This is a sign-off item for the Vulkan/CUDA slices (their F16 activation defaults, #1138, do not apply here).
- Weights are read in place (rule 2): BF16 tensors go through a row-streaming dequantise-and-dot with F32 activations, Q6_K through the Q8_K path, Q8_0 through
  the Q8_0 path (all q8_0 tensors of safe2 have bounded inputs).

## 4. Precision policy: F32 recurrent state (tracked exception)

CLAUDE.md precision rule 1 forbids FP32 storage on hot paths. For Olala the Mamba recurrent state is **F32 by explicit user sign-off (2026-11-25)**, as a
per-architecture, tracked exception (`OlalaStatePolicy`, a compile-time constant of this arch; no env var or global setting narrows it):
OVH serve with vLLM `--mamba-ssm-cache-dtype float32`; the fork's own docs state a reduced-precision state "degrades long generations into repetition loops"
(`OLALA_BF16_STATE=1` "not recommended", `OLALA_STATE_Q8=1` an unvalidated experiment). Covered buffers and their cost per sequence (7A1B):

**Extension, user ruling 2026-11-25: the token-shift last K/V and the trapezoid carry (previous token's rotated K and raw x) are F32 as well, until #1971 shows a narrower type is safe.** HF keeps both in the model dtype (bf16 in the shipped checkpoint), so this is stricter than the reference stack; the OVH llama.cpp fork also keeps them F32 (bf16 K/V sections are opt-in and "not recommended"). The guard test covers all four buffer kinds plus the token-shift byte total.


| buffer | shape | bytes / M layer | x 29 layers |
|---|---|---|---|
| SSM matrix `h` | 48 x 64 x 128 | 1,572,864 | 44.0 MiB |
| rotary phase | 48 x 32 | 6,144 | 174 KiB |
| trapezoid K (rotated) | 4 x 48 x 128 | 98,304 | 2.7 MiB |
| trapezoid V (raw x) | 48 x 64 | 12,288 | 348 KiB |
| token-shift last K/V (per V layer, x 7) | 2 x 12 x 128 | 12,288 | 84 KiB |

Evidence plan for a smaller dtype, and the exit criteria: issue #1971. Listed in `docs/PRECISION.md` (Keep-FP32 by policy). It is **not** precedent for any other
storage: K/V history stays BF16 (HF casts K and V to bf16 too), activations F32 only because of the outliers above.

## 5. Oracle and numerics (HF transformers, CPU)

`scripts/olala-oracle/olala_oracle.py` runs HF `modeling_olala.py` on CPU (`OLALA_TORCH_MAMBA3=1` pure-PyTorch scan/step, eager attention, torch experts) and records
full no-cache logits plus a greedy continuation produced through HF's *cache* (one-token step) path with explicit `position_ids`.
HF quirks that matter when building an oracle: (a) `OlalaModel.forward` defaults `position_ids` to 0 on every cached step — pass them (the token shift and
scalable softmax read them); (b) a 1-token prefill into an empty cache is treated as a decode step and crashes; (c) multi-token continuation with a cache
restarts the SSM from zero, so chunked-prefill continuation cannot be oracle-tested against HF (the dotLLM state continues correctly).

Two HF variants are recorded: **strict** (the query is not rounded to bf16) and **hf** (as shipped). Shipped HF rounds the query to bf16 and returns the attention output
in bf16; this alone is ~1e-3 in logits (measured: it vanishes to 5e-7 when the same two casts are added to dotLLM). dotLLM computes the query and the attention output in F32,
so the strict variant is the one a correct F32 engine matches to float rounding. The GGUF converter folds `1 + w` for B/C/Q/K norms **in the checkpoint dtype (bf16)**,
so the GGUF carries bf16-rounded gains (error up to ~4e-3 per gain) that HF-from-safetensors does not have; the real-model oracle can patch HF's gains from the GGUF to isolate engine differences from
that conversion rounding (`--gguf`).

Tiny model (`olala_oracle.py tiny`): 5 layers `MVMMV`, hidden 64, every 7A1B flag, a 5-token window with 12-token and 3-token prompts, BF16 GGUF; the unit tests compare
all prefill rows and every teacher-forced decode step (KL, max|dlogit|, argmax) and the free-running greedy continuation. Mutation record: §5.2.


### 5.1 Measured parity (2026-11-25, Zen 5 CPU, F32 activations)

**Tiny model** (strict oracle): every prefill row and every teacher-forced decode step agrees with HF to float rounding (max |dlogit| ~1e-6, KL ~1e-14), greedy
continuations identical. Against HF as shipped: max |dlogit| ~2e-3, KL ~2e-7 (the bf16 query / attention-output casts; same argmax everywhere).

**Real 7A1B, bf16 GGUF** vs HF float32 with the GGUF's own norm gains (`real-strict-gguf-norms`):

| prompt | prefill rows (worst KL / top-1) | decode 24 steps teacher-forced (worst KL / top-1) | median KL | greedy 24 tokens |
|---|---|---|---|---|
| "The capital of France is" (5) | 2.7e-7 / 5 of 5 | 1.05e-3 (one step, 0.7 logit) / 24 of 24 | 2.1e-7 | identical |
| `def fibonacci(n):` docstring (13) | 1.5e-6 / 13 of 13 | 1.1e-4 / 24 of 24 | 7.8e-8 | identical |
| chat template, "Say hello in French." (300) | 5.7 (worst row) / 294 of 300 | 4.9e-5 / 24 of 24 | 2.8e-5 | identical |

(Chat row re-measured after the #1973 review's double-precision softplus; before it: 0.45 / 297 of 300, 8.0e-4, 3.9e-5. The short prompts did not move.)

The isolated KL spikes are single rows caused by MoE top-6 flips between near-tied experts. **Verified on the "capital" decode step 8** (KL 1.06e-3, 0.71 logit)
in the #1973 review. At that step the first selection that differs is block 29: HF picks expert 52 over 163 by a margin of 1.46e-5 in `sigmoid + exp_probs_b`;
dotLLM picks 163. Blocks 30, 34 and 35 then differ as a consequence. Forcing only block 29's choice in HF (strict, GGUF norms) brings HF to KL 5.9e-7 against dotLLM, and
blocks 30, 34 and 35 then match without being forced. Near-ties of 0 to 7e-7 in blocks 7-10 and 14 resolved the same way on both sides. The 300-token chat prompt is numerically chaotic even for HF: its own one-token-step path differs from its chunked
forward by up to 2.9 logits (strict) / 5.7 (as shipped), because activation outliers of 1e9-1e14 amplify reassociation noise; it is gated at the noise floor (median KL, top-1) only.
Against HF **as shipped** (bf16 query/attention output, exact `1 + w` gains instead of the GGUF's bf16-rounded ones): short prompts median KL 7e-6 .. 2e-5, top-1 and greedy identical;
the 300-token prompt median KL 4.7e-3, top-1 274 of 300 (informational).

**q8_0-safe2 vs bf16** (dotLLM CPU both sides, teacher-forced on the HF greedy tokens, no NaN anywhere): "capital" mean KL 8.2e-3 / top-1 26 of 28, "code" 1.8e-3 / 36 of 36,
chat 0.15 / 281 of 323 (the long prompt amplifies the 8-bit noise). OVH's published numbers (16 wikitext chunks): KLD 0.0040, 96.9% same top-1 — consistent for the short prompts; a wikitext
KLD run is the proper like-for-like measurement and is not done yet.

### 5.2 Mutation record (tiny oracle, `OlalaOracleTests`)
Each row is a deliberately wrong implementation; all fail the oracle tests except the one noted (the soft-cap mutant needed the tiny cap of 3.0 to die).

| mutant | result |
|---|---|
| drop the `beta u_{t-1}` trapezoid carry; V-state not carried; angle not accumulated | killed |
| fold zero-centred gamma twice; swap B/C norm gains | killed |
| attention scale `1/sqrt(d)`; no soft cap; window off by one; `log(pos+1)` not clamped to the window | killed (soft cap needed a tiny cap of 3.0: at 150 the first oracle could not see it) |
| swap signal/noise heads; kv head `h % n` instead of `h / group`; TPA not divided by rank; gate bias dropped | killed |
| token shift: post-norm K carried; not masked at position 0 | killed |
| routing: no `expert_bias` in top-k; probabilities not renormalised; ReLU instead of ReLU^2; shared expert dropped | killed |
| embedding not L2-normalised; geodesic divisor `layer` instead of `layer + 1`; K cache perturbed by 1% | killed |
| **SSM state rounded to bf16 every step** | **survived** — on a 12-token sequence the narrowing is below the tolerance. This is why #1971 needs long-generation evidence; the short oracle cannot decide the state dtype. |

## 6. Differences from the existing code and what is reused

| existing | Olala |
|---|---|
| `Mamba3*` (safetensors-first, canonical in_proj `[z, x, B, C, dt, A, trap, angles]`, F32 spans, `qk_pre_dot` + shifted-gamma chunk kernels) | per-head interleaved `ssm_in` + separate `ssm_in_dyn`, recurrence evaluated token by token (`OlalaOps.Mamba3MimoStep`); MIMO tensor shapes (`[H,R,P]`, `[H,R,N]`) match, so `Mamba3CanonicalSsd.ExecuteMimoStreaming` is the candidate for chunked prefill once its input layout is adapted (follow-up) |
| MLA (`MlaConfig`, latent KV) | **Not MLA.** TPA factors are re-expanded to 12 x 128 K/V per token; the factored cache (12 x 4 + 4 x 128 per token vs 2 x 12 x 128) is a later memory optimisation (the fork's "2.6x less KV") |
| qwen4exp latent MoE (`MoeSwiGluMlp`, softmax router) | sigmoid router + selection bias, renormalised x 2.45, ReLU^2 experts without a gate matrix, latent down/up around the routed bank |
| pre-norm blocks, `final_norm` | none of either: geodesic rotation, unit-norm residual, lm_head straight on the residual |

## 7. Not done yet (tracked in #1969)
CUDA forward (refuses with `OlalaConfig.UnsupportedMessage`), chunked/parallel prefill of the M layers, a factored TPA KV cache, prefix-cache/snapshot support,
serving (chat template reasoning-effort, tool-call parser), `lazy` expert paging for small-RAM machines.

## 8. Vulkan forward (issue #1983)

`VulkanOlalaTransformerModel` + `native/vulkan/shaders/olala_*.glsl` (wrappers generated by `native/vulkan/gen_olala_wrappers.py`). One command buffer per chunk
of 256 tokens (decode: one submission per token); the model owns its sequence state like the CPU model.

**Precision (user decisions 2026-11-25).** Stored activations are **BF16** (`uint16_t` words); every kernel accumulates and reduces in F32 registers / shared
memory and rounds once on store. Recurrent state (SSM `[H][N][P]`, angle, trapezoid K/V, token-shift K/V) is F32 (`OlalaStatePolicy`; guard
`OlalaVulkanTests.StatePolicy_Guard_AllRecurrentStateIsF32` checks element bytes and the allocated device bytes against the section 4 table). KV cache BF16. Weights are read in
place by the shader (Q8_0 / Q6_K / BF16; F32 router): no quantized-activation (FP16-scaled Q8_1) matmul, no widening. The router scores, top-6 and
renormalisation run in one workgroup per token and never leave shared memory; only the expert ids and the BF16 routing weights are stored.
`olala_*_f32act` shaders build the F32-activation-storage **reference arm** (a measurement control, `OlalaVulkanOptions.ActivationF32`, never the default).

**Kernels.** `olala_gemv_{q8_0,q6_k,bf16}` (1 or 8 columns per workgroup, epilogue none / relu^2 / add, F32 logits out), `olala_embed_{q8_0,bf16}` (gather + L2),
`olala_geodesic`, `olala_mamba3_step` (one workgroup per head, loops over the chunk's tokens; B/C RMSNorm, bias, rotation, trapezoid recurrence, gated MIMO output),
`olala_tpa_kv` (TPA reconstruction, token shift, K RMSNorm, BF16 cache store, F32 carry), `olala_tpa_attn` (scalable softmax, soft cap, window, online softmax,
differential combine, silu gate; no score tensor in memory), `olala_moe_route`, `olala_moe_up` (relu^2) and `olala_moe_down` (routing-weighted sum) over the expert bank in place.
bf16 cooperative matrices: `STRIX_HALO.md` records `bf16:0` for this device and f16 coopmat would overflow on the unbounded inputs, so prefill is the multi-column GEMV, not coopmat.

**Parity.** Tiny model vs the CPU (F32 activations): F32-activation arm max KL 7e-14 (every kernel exact); BF16 arm max KL 1.4e-5, max |dlogit| 1.6e-2, decode KL <= 1.6e-5, greedy identical.
Real q8_0-safe2 (teacher-forced on the HF greedy tokens, vs HF float32 with the GGUF norm gains; the F32-activation arm of the same engine is the control):

| prompt | BF16-act vs HF: median KL / top-1 | F32-act vs HF: median KL / top-1 | BF16-act vs F32-act: median KL / top-1 | greedy 24 tokens vs HF |
|---|---|---|---|---|
| capital (28 rows) | 1.4e-3 / 27 of 28 | 3.2e-3 / 26 of 28 | 2.3e-3 / 27 of 28 | diverges at token 12 |
| code (36 rows) | 2.3e-4 / 36 of 36 | 1.0e-4 / 36 of 36 | 6.7e-5 / 36 of 36 | identical |
| chat (323 rows) | 1.8e-2 / 281 of 323 | 1.2e-2 / 280 of 323 | 1.0e-2 / 292 of 323 | identical |

The BF16 activation storage costs no more than the q8_0 weight noise already in the file (compare the CPU numbers of section 5.1); no NaN anywhere (the outlier tensors are the BF16 / Q6_K ones). The 300-token chat prompt is chaotic (section 5.1).

**The "capital" greedy divergence is a near tie, not a bug** (`Diag_GreedyDivergenceAttribution_Capital`, q8_0-safe2). At step 12 HF itself has top-1 374 vs top-2 1265 by a margin of **0.046** logits;
Vulkan BF16-act has the same two tokens swapped by 0.047 (1265 ahead), the F32-activation control also picks 1265 (margin 0.153) and its free-running greedy diverges even EARLIER (token 4: 2038 vs 2550, step-4 near-tie
introduced by the q8 weights). The bf16 GGUF on the same engine matches HF top-1 28 of 28 on this prompt. So the cause is q8_0 weight digitisation on a 0.05-logit tie, present in the F32-activation arm, not the BF16 storage.
The CPU q8_0-safe2 greedy confirms it (see "Greedy attribution, closed" below); the F32-activation arm is bit-equivalent to the CPU on the tiny oracle (KL 7e-14).

**Router.** The F32 `ffn_gate_inp` tensor goes through the same `UploadRawQuantMatrix` path as every other weight, in its own type (no managed F32 copy, never narrowed). Measured on Strix Halo in the #1988 review: with the defaults, all 411 weight tensors (6.5 GiB, routers included) are **staged** through the host, none imported, because the driver rejects importing the read-only GGUF mapping (`vkAllocateMemory` -1000072003, #1190); `DOTLLM_VULKAN_HOST_IMPORT_WRITABLE=1` is the opt-in zero-copy route. That is the box-wide Vulkan default, not Olala-specific; it is a host bounce under precision rule 3, not a widening. "BF16 routing weights" means the six per-token routing coefficients (the renormalised, x2.45 sigmoid scores), not the tensor.

**Mutants** (tiny oracle + kernel tests, `native/vulkan/shaders` edited, SPIR-V rebuilt, tests rerun): killed - trapezoid carry dropped, angle not accumulated, previous-x carry, dt bias,
scale 1/sqrt(d), no soft cap, window off by one, unclamped log, signal/noise swap, gate bias, kv-head mapping, V token shift unmasked at position 0, token-shift carry across batches,
TPA not divided by rank, router bias dropped, no renormalisation, x2.45 dropped, ReLU for ReLU^2 (routed and shared), routing weight ignored, geodesic divisor, theta clamp, embedding not normalised,
**SSM state truncated to bf16 each step (killed here by the strict F32-activation prefill gate; it survived the CPU oracle)**, **F16-overflow activation storage (killed only by the 1e12 / 1e14 kernel tests; the tiny model cannot see it)**.
Not equivalent (corrected in the #1988 review): leaving the K token shift unmasked at position 0 scales K by 1 - sigmoid(shift) there, which the K RMSNorm cancels only up to its eps (1e-6). It is measurable: tiny F32-act vs CPU max KL 7e-14 -> 1.4e-8, max |dlogit| 1e-6 -> 4.4e-4; real q8_0-safe2 median KL vs HF 1.4e-3 -> 4.4e-3 (capital), 2.3e-4 -> 9.3e-4 (code). It survived the original F32-act gate (KL 1e-6, |dlogit| 1e-3); that gate is now 1e-9 / 1e-4 and kills it (V masking was already killed).

**Review additions (#1988).** (1) A second survivor, converted into a kill: dropping the cross-block online-softmax rescale (`acc *= corr` in `olala_tpa_attn`) moved only the 323-token chat prompt, the one case past 128 keys (median KL 1.8e-2 -> 0.84, top-1 281 -> 165 of 323), and that prompt was ungated; it is now gated at median KL < 0.1 and top-1 >= 75%. (2) New coverage, not a survivor: `QuantRowDecoders_OnRealSafe2Rows_MatchDequantizeReference` checks the in-place Q8_0 / Q6_K row decoders (shared by the GEMV and both MoE kernels; the tiny model is BF16 and never reaches them) on real safe2 rows at scale 1 and 1e9 against `Dequantize` + a double dot (measured 1e-8 to 7e-8 relative, gate 1e-5). A swapped Q6_K sub-block scale (already killed by the slow q8_0-safe2 parity run, median KL 0.87) and a Q8_0 element-order mutant each fail only their own type here. Untested: KV-cache growth past 1024 tokens (the device-copy path in `EnsureKvCapacity`).

**Greedy attribution, closed (#1988 review).** The CPU q8_0-safe2 run diverges from HF at token 4 (2038 over 2550, margin 0.09; HF 2550 by 0.23), exactly like the F32-activation control (margin 0.03), and at step 12 also ranks 1265 above HF's 374. Both near-tie flips come from the q8_0 weights; BF16 activation storage keeps step 4 on HF's token.

**Throughput** (Strix Halo, Vulkan, q8_0-safe2, warm, same session, 300-token chat prompt prefill then 48 greedy decode steps; UMA swing and a shared GPU make runs differ):
BF16 activations prefill 180-307 tok/s, decode 27-31 tok/s; F32-activation control prefill 326-330, decode 30.3-30.4 (storage dtype is throughput-neutral within noise); bf16 GGUF prefill 289-296, decode 24.7-25.4.
Levers not taken: split-KV attention (12 workgroups per decode token), 128-thread Mamba workgroups (48 workgroups of 64 lanes), fusing the geodesic into the following GEMV, a chunked prefill scan, the 36 x ~25 dispatches per token.

**Stage profile and the router lever (#2000).** Opt-in `OlalaVulkanTests.RealModel_StageProfile` (`DOTLLM_OLALA_VK_PROFILE=1`) submits and waits after every stage (about 50 us sync per mark, subtracted).
q8_0-safe2, Strix 8060S, warm. Decode net ~29 ms/step: **MoE router 51-54%** (430 us/call x 36), M out_proj 9%, mamba3_step 8%, tpa_attn 5%, ssm_in 5%, shared_down 5%, lm_head 4%.
Prefill 300 tokens net ~943 ms: **mamba3_step 46%** (7.5 ms/call x 58), ssm_in GEMV 13%, moe up 8%, moe down 6%, V projection GEMVs 6%, route 5%, the rest under 4% each.
The router was one 256-thread workgroup walking the 1.5 MB F32 router scalar-wise, one expert row at a time per subgroup (3.6 GB/s). It now reads the F32 router in place as vec4 (hidden % 4 == 0)
with four expert rows in flight per subgroup; the scores still never leave shared memory (no new F32 storage). Same-session interleaved A/B (new/old/new/old, 3 warm rounds each): decode 50.3-53.1 vs 30.0-30.3 tok/s, prefill 328-335 vs 317-324.
Router mutants (tiny gates): row index, vec4 component, killed. The E % 4 != 0 tail is covered by `RouteKernel_ExpertCountNotMultipleOf4_MatchesDoubleReference` (isolated kernel, E in {3,5,6,7,8,13}, both activation arms, bias forces the last two experts): a guard one too strict and a clamp to E-2 are killed (E = 6); removing every clamp and guard survives because it is output-equivalent (it only drops the out-of-bounds read protection). Not bit-identical to the scalar form (lanes own four consecutive k): F32-act control vs the old shader KL <= 1e-6 on capital/code; on the long chat case one near-tie top-k flip cascades, KL vs HF unchanged (median 1.71e-2 vs 1.82e-2, top-1 280 vs 281 of 323). Review re-measure (four arms resident, interleaved x 8): decode 48.2-50.5 vs 22.6-29.7 (median 49.5 vs 29.4).


**Mamba-3 step state in registers (#2001).** `olala_mamba3_step` (46% of 300-token prefill, 7.5 ms/call) read-modify-wrote the 32 KB F32 state of each head in global memory on every token. Each lane (one `p`) now holds its N-long
state column in registers for the whole chunk (load once, store once; `N` and `R` are specialization constants so the loops unroll; loader requires head dim <= 64). Same F32 state and the same source arithmetic, but not bit-identical (review: F32-act control vs the old shader KL <= 6e-6 on the real capital/code cases, quality vs HF unchanged; at N = 128 the shader uses 256 VGPRs and spills 144 B per lane to scratch, a lever left for a 128-thread split). Same-session interleaved A/B
(new/old/new/old, 3 warm rounds each): prefill 471-486 vs 315-324 tok/s (+49%), decode 31.6-31.8 vs 30.1 (neutral). Review re-measure (2026-11-25, same session, interleaved ABAB/BABA x 8, both arms resident): prefill 428-467 vs 303-316 (+42%), decode 30.3-30.5 vs 29.6-30.1; with four 6.9 GB arms resident prefill was 376-413 vs 299-315. Mutants killed by the tiny gates: state store dropped, state load dropped (decode gate only), alpha decay dropped, previous-K swap.
