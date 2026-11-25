# Olala (OVHcloud) — architecture notes

**Status:** CPU reference forward (issue #1970, epic #1969). CUDA runs it (section 8); Vulkan refuses until its port lands.
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
Vulkan and CUDA forwards (both refuse with `OlalaConfig.UnsupportedMessage`), chunked/parallel prefill of the M layers, a factored TPA KV cache, prefix-cache/snapshot support,
serving (chat template reasoning-effort, tool-call parser), `lazy` expert paging for small-RAM machines.

## 8. CUDA forward (issue #1984, epic #1969)

`CudaOlalaTransformerModel` (`src/DotLLM.Cuda/Architectures/`) with `native/kernels/olala.cu` -> `native/ptx/olala.ptx` (CUDA 12.8, `.version 8.7`). Reached through
`CudaModelLoader.CreateFromGguf`; the generic `CudaTransformerModel` still refuses `olala` (it would run a different model). No change to Models or Core.

**Precision contract (James, 2026-11-25).** Stored activations are **BF16** (`act_t`), accumulation F32 in registers, geodesic reductions and the angle wrap in double (as on the CPU).
The recurrent state (SSM matrix, rotary phase, trapezoid K/V, token-shift K/V) is **F32**, sized from `OlalaStatePolicy` by the pure function `CudaOlalaStateLayout`
(`CudaOlalaStateLayoutTests` runs without a GPU). Weights are read **quantized in place**: Q8_0, Q6_K, BF16 and the F32 router are dequantized in registers inside the dot
loops (no activation quantisation at all, so the FP16-scale overflow of type-0 matmuls cannot occur). Router logits / sigmoid / top-6 / renormalised weights stay F32 in one
fused kernel (near-tied experts, margins ~1e-5, must not see BF16 rounding); the 6 routing weights per token are the only F32 inter-kernel storage besides the state. Logits are F32 output.
F32 activation storage is **not** shipped; it needs sign-off.

**Kernels.** `olala_gemv_{q8_0,q6_k,bf16,f32}_{t1,t8}` (warp per row, 1 or 8 tokens per weight read; epilogues store / ReLU^2 / add), `olala_moe_up_*` / `olala_moe_down_*`
(per (token, slot) indirect expert GEMV; down sums the 6 weighted experts in registers), `olala_route`, `olala_geodesic`, `olala_embed_*` (+L2 normalise),
`olala_mamba3_mimo` (one block per head, SSM rows in registers across the tokens of a chunk), `olala_tpa_kv` / `olala_tpa_carry` / `olala_attn` (online softmax, soft cap,
window) / `olala_attn_combine`. Limits (validated at load): N <= 128, P <= 64, R <= 4, head_dim <= 128, <= 16 experts per token. Prefill runs in chunks of 256 tokens
(the recurrence and the K/V history continue across chunks; `Tiny_ChunkedPrefill_IsBitIdenticalToOneShot`). K/V history grows on demand (BF16, 6 KiB per token per V layer x 7) and is admitted against free VRAM, never spilled.

**Known gaps.** Weight upload is `cuMemcpyHtoD` from the mmap (the same route as the other CUDA loaders; CLAUDE.md rule 3 direct route not added). No CUDA graph, no
factored TPA cache, no per-sequence state / prefix cache (single model-owned sequence, like the CPU model), `--gpu-layers` cannot split this architecture.

### 8.1 Measured on the RTX 3060 (framework, 2026-11-25, PCIe x4)

**Tiny HF oracle** (CUDA BF16 activations): worst KL vs HF 1.2e-5 (`tiny`) / 1.3e-5 (`tiny-hfbf16`), worst |dlogit| 1.7e-2, top-1 11/12 rows (the miss is a near tie), greedy identical over 10 tokens; vs the in-process CPU: prefill KL 1.2e-5, teacher-forced decode KL 1.5e-5, 9/9 top-1. Chunked prefill (chunk 1 / 3 / 5) is bit-identical to one shot.

**F32-activation control arm** (`olala_f32act.ptx`, internal measurement control, never the default): same kernels with F32 stored activations (K/V still rounded to BF16, as the CPU does) vs the CPU: worst KL 3.5e-14, worst |dlogit| 9e-7. This is the float-rounding floor of the kernel logic and the reference for what BF16 storage costs on the tiny model (KL ~1e-5, 5 orders above it).

**Real q8_0-safe2 vs the CPU forward** on the CPU PR's prompts (BF16 activations on CUDA, F32 on CPU):

| prompt | prefill median KL / top-1 | teacher-forced decode median KL / top-1 | free-running greedy (24 tokens) |
|---|---|---|---|
| capital (5) | 5.9e-4 / 5 of 5 | 1.5e-3 / 22 of 23 | first divergence at token 4 |
| code (13) | 4.8e-4 / 13 of 13 | 2.8e-5 / 23 of 23 | identical |
| chat (300) | 2.2e-2 / 262 of 300 | 5.6e-4 / 23 of 23 | identical |

The chat prompt is numerically chaotic even for HF (see 5.1); the CPU F32 path has 294 of 300 top-1 against HF there, the BF16-activation CUDA path 262 of 300 against the CPU. These two gaps (capital greedy divergence, chat prefill top-1) are the measured cost of BF16 activation storage: input for the F32-activation sign-off decision.

**Throughput** (RTX 3060, PCIe x4, 300-token chat prompt, 5 repetitions after warm-up, unoptimised kernels, one logits D2H per step; framework is a shared box, treat as ranges): prefill 297-298 tok/s, decode 44.4-47.7 tok/s.

### 8.2 Mutation record (CUDA tiny oracle, `CudaOlalaParityTests`)
BF16 arm, tightened gates (KL < 5e-5, |dlogit| < 4e-2): **17 of 19 mutants killed.** Killed: trapezoid K carry dropped, trapezoid V carry dropped, angle not accumulated, window off by one,
signal/noise heads swapped, router top-k without `exp_probs_b`, probabilities not renormalised, ReLU instead of ReLU^2, geodesic divisor `layer` instead of `layer + 1`, token-shift carry dropped,
kv head `h % nk`, gate bias dropped, TPA not divided by rank, embedding not L2-normalised, scale `1/sqrt(d)`, soft cap removed (KL 4.7e-3), token shift not masked at position 0 on V (KL 2.6e-3).
Survivors in the BF16 arm: (1) token shift not masked at position 0 on **K**: not equivalent (the K RMSNorm cancels the uniform scale only up to its eps), but its effect (KL 1.4e-8) is far below BF16 noise; the F32 control arm kills it (KL 1.35e-8 against the 1e-11 gate).
(2) SSM state rounded to BF16 every step: below the tolerance on a 12-token sequence, the documented #1971 evidence gap.
F32 control arm (gates KL < 1e-11, |dlogit| < 1e-5): the first five mutants run (trapezoid K carry, angle, V-side and K-side token-shift mask, soft cap) were all killed; the run was cut short when framework rebooted (see the PR), the rest of the control-arm set is not yet recorded.
