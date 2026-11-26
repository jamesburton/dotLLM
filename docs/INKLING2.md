# Inkling 2 (Thinking Machines) — architecture notes

**Status:** CPU reference forward (issue #2026, epic #2025). Vulkan, CUDA, the vision tower and the DSpark drafter are later slices (section 9).
Model: `thinkingmachines/Inkling-2-Mini` (HF `model_type: inkling2_loop`, `Inkling2LoopForConditionalGeneration`, Apache-2.0, 2026-11-20).
GGUF: `ggml-org/Inkling-2-Mini-GGUF` (`general.architecture = inkling2_loop`; a plain `inkling2` file is the same graph with one pass).

**Sources.** Semantics are llama.cpp's (master, validated against b12780 / `09c141064`): `src/models/inkling2.cpp` (the graph),
`src/models/inkling2-loop.cpp` (the loop), `src/llama-graph.cpp` (`build_moe_ffn`, `build_attn_mha`, `build_inp_mem_hybrid_iswa_rel`),
`src/llama-kv-cache.cpp` (`set_input_relative_pos`), `ggml/src/ggml-cpu/ops.cpp` (`l2_norm`, `ssm_conv`), `ggml/src/ggml-cuda/inkling2.cu`
(fused forms of the same ops: a readable statement of the order of operations), `src/llama-vocab.cpp` (pre-tokenizer `inkling2`),
`conversion/inkling2.py` (tensor layout and the folds), `gguf-py/gguf/constants.py` (names). HF `modeling` code was not needed; no HF weights are on this box.

## 1. Shape of the model (Inkling-2-Mini)

24 physical layers, hidden 2304, 24 query heads / 8 KV heads, head_dim 128, vocab 201,024 rows (200,058 tokens), final logit softcap 30. **No RoPE.**
Local (sliding-window 1024) layers `0-4, 6-10, 12-16, 18-22`; global layers `5, 11, 17, 23` (`attention.sliding_window_pattern`, true = local).
128 routed experts (8 used, intermediate 1152), no shared expert. Layers 6..17 (`loop.start`/`loop.end`) run **3 passes** with shared weights, so a token
costs 48 layer visits; every visit has its own KV history and conv state (**cache slot** `il`: physical layers `0..23`, then pass 1 = `24..35`,
pass 2 = `36..47`; `inkling2-loop.cpp: layer_phys`).

## 2. Residual stream: depth ("attention-residual") mixing

`attn_res.block_size = 6`. A *block* is the input of a layer `ip` with `ip % 6 == 0` (`ip` = physical index). Per layer visit:

- The layer that starts a block (`ip % 6 == 0`) pushes its **input** (`inpSA`) onto `blocks` and does **not** add a residual: its sub-layer input for the MoE is the
  attention output alone. Every other layer adds `inpSA + attn_out`.
- `mix(blocks, cur, query)` (`graph::mix`): states = RMS-normalised (gamma-free, eps 1e-6) block inputs plus the RMS-normalised `cur`; `w = softmax(states . q / sqrt(2304))`;
  result = `sum_i w_i * state_i` (the mix is over the NORMALISED states). One query per call site: `attn_res_query` (layer 0 has none: it reads the embedding; local
  layers 1 query, global layers 3 queries = Q, K, V inputs), `ffn_res_query` (1), `output_res_query` (1, before the output norm).
- Block inputs are immutable once pushed; the normalised copy is cached (`mix_stack`).

## 3. Layer visit, in order

```
inputs[3] = layer 0 ? {cur,cur,cur} : mix(blocks, cur, attn_res_query)      # local: the one result x3; global: result q/k/v
if pass > 0: inputs[j] = control(inputs[j], message_j)  for j in Q,K,V       # section 5
local : inputs = attn_norm(inputs[0]) (x3 after control: attn_norm applied to each)
global: inputs[0..2] = {q,k,v}_inp_norm(inputs[0..2])
Q = wq . in0;  K = conv(wk . in1; dynamic taps = k_conv_proj . in1 on global layers);  V = conv(wv . in2; dynamic = v_conv_proj . in2);  R = attn_rel . in0
Q = clip(Q * q_clip[h,:], 22.627);  K = clip(K, 22.627);  V = rms_norm(V, eps 1e-6) per head, NO gamma;  R = clip(R * rel_clip[h,:], 8.0)
global layers: Q *= tau;  R *= tau          tau = 1 + 0.1 * ln(max(1, (pos+1)/8192))   (exactly 1 below position 8191)
sink_h = sink_proj . in0 ;  gate = sigmoid(attn_gate . in0)
out_h  = attention (section 4);  out *= gate;  attn_out = post_conv(wo . out)
ip % 6 == 0 ? (blocks.push(norm(inpSA)); x = attn_out) : x = inpSA + attn_out
m = mix(blocks, x, ffn_res_query);  if pass > 0: m = control(m, message_FFN);  m = ffn_norm(m)
x = x + ffn_post_conv(MoE(m))                                                     # section 6
```

`clip(x, scale, t)` (`graph::clip`): multiply by the optional per-channel scale, then `x * t / max(|x|, t)` (`l2_norm(x, eps=t)` followed by `* t`): a row is only ever shrunk.
RMS norms with gamma use `rms_norm_eps` (1e-6); the gamma-free ones inside mix/control hard-code `1e-6f`.

## 4. Attention

`score(q, k) = (q . k) / sqrt(128) + bias[h][d] - sink[h]`, `d = pos_q - pos_k`, then softmax over the keys **plus one extra zero logit** (the "sink": the stored
`attn_sink_proj` output is subtracted from every logit of the head, and a fixed zero sink is added to the softmax; `exp(l - t) / (1 + sum exp(l - t))`). The sink contributes no value.

- `bias[h][d] = attn_rel_bias[d] . R[h]` for `d < extent`, **zero for `d >= extent`** (llama.cpp pads the table with a zero row; distances outside the extent and empty cells map to it).
  `attn_rel_bias` is `[16][extent]`; extent = 1024 for local layers (= the window) and for global layers (`attention.relative_extent`).
- The bias enters through the mask, i.e. **after** the `1/sqrt(d)` scaling and unscaled. (llama.cpp's FA path casts `bias + mask` to F16 — an oracle artefact, section 8.)
- Local layers: keys with `pos_q - pos_k < 1024` (`LLAMA_SWA_TYPE_STANDARD`); global layers see every earlier key. GQA 3:1. KV is the post-conv, post-clip K and post-norm V.

## 5. Loop core and the rank-64 controllers

For passes 1 and 2 each core layer `ip in 6..17` mixes the previous pass into its inputs. **Every pass restarts from the prelude output** (layers 0-5) and the prelude
blocks. The *bank* of the previous pass = its blocks **after the first core block** (the blocks pushed at `ip = 12`, i.e. all but `ip = 6`'s) plus its final output `cur`.

```
messages = mix(bank, bank_out, loop_msg_query[ip])            # 4 queries -> messages Q, K, V, FFN  (loop_msg_query rows: [q, k, v, ffn])
control(cur, message, ctrl):
    g      = ctrl_down . [ rms(cur) ; rms(message) ]           # [4608] -> rank 64, gamma-free rms, eps 1e-6
    gain   = tanh(ctrl_gain   . g)        forget = tanh(ctrl_forget . g)    # [64] -> 2304
    return (cur - forget * cur) + gain * message
```

Pass 0 runs the core with no messages. `inpSA` (the residual) is the pre-control input. Controllers exist per core layer and per kind (`loop_{q,k,v,ffn}_ctrl_{down,gain,forget}`).

## 6. MoE and the short convolutions

Router: `p = sigmoid(ffn_gate_inp . m)` (F32); top-8 of `p + exp_probs_b` (selection only); the **unbiased** `p` of the winners normalised to sum 1 (`+ 1e-20`) then `* route_scale (8)`.
Expert bank `ffn_gate_up_exps` is merged, **gate rows first** then up rows (`[2304][2304]` per expert); activation **SITU**:
`beta*tanh(g/beta) * sigmoid(g) * lb*tanh(u/lb)`, `beta = 4` (`situ_beta`), `lb = 25` (`situ_linear_beta`); then `ffn_down_exps`. NVFP4 banks (layers 0-19) carry a per-expert `.scale`
(`F32 [128]`, one for gate_up and one for down) that multiplies the matmul **output** of that expert (`build_lora_mm_id`); layers 20-23 are Q8_0 without scales. Ignoring a scale
changes the logits catastrophically (mutant `NoExpertScale`: KL 13, top-1 2%).

Short convolution (`graph::convolve`, `ggml_ssm_conv`), on K, V (`k_conv`, `v_conv`: 1024 channels), the attention output and the FFN output (`attn_post_conv`, `ffn_post_conv`: 2304 channels),
`K = 4` taps, per channel with a carried window of the last 3 inputs: `acc = sum_i x[t-3+i] * w[i]` (**tap 0 = the oldest input**; the converter flips the HF kernel) plus, on global layers' K and V,
input-dependent taps `sum_lag x[t-lag] * d[head][lag]` (`d` = `*_conv_proj . input`, 4 taps per KV head, **lag 0 = the current input**, channel c belongs to head `c / 128`);
`out = (x_t + acc) / sqrt((dynamic ? 2 : 1) * K + 1)`.

## 7. Tokenizer, template, output

- `tokenizer.ggml.model = gpt2`, `pre = inkling2`: llama.cpp maps this to the GPT-4o pre-tokenizer, **no BOS**, `clean_spaces = false`, `ignore_merges = true`. dotLLM routes `inkling2` to its GPT-4o pipeline.
  200,058 tokens; EOS/EOT 200006; `add_bos_token = false`.
- Chat template (identical in HF and the GGUF): turns are `<|message_user|><|content_text|>...<|end_message|>`, the assistant is `<|message_model|>`; a `<|message_system|><|content_text|>Thinking effort level: X<|end_message|>`
  is emitted before the first non-system message (or at the end if there is none). `reasoning_effort` is a name (`none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh/max 0.99`) or a number in [0, 0.99]; default 0.9;
  anything else raises. Tools are declared as sorted compact JSON; tool calls render as `{"name":...,"args":{...}}`. The template renders no generation prompt of its own.
- Output: `mix(blocks, cur, output_res_query)` -> `output_norm` -> `output.weight` (Q8_0; only the 200,058 unpadded rows are computed) -> `30 * tanh(x / 30)`.
- Embeddings: `token_embd` (Q8_0 rows) then `token_embd_norm` (RMS with gamma) — also for the layer-0 input.

## 8. Storage, precision and the oracle

| tensor | GGUF type | dotLLM |
|---|---|---|
| routed experts, layers 0-19 | NVFP4 (+ per-expert `.scale` F32) | read in place by `MatMul.GemmNvfp4`, scale applied to the expert output; no expansion |
| routed experts, layers 20-23 | Q8_0 | in place |
| attention / embedding / output matrices | Q8_0 | in place |
| router `ffn_gate_inp` | F32 (1.2 MB/layer, 28 MB) | in place (`expert_router_f32`) |
| norms, `*_res_query`, `*_conv`, `*_clip`, `attn_rel_bias`, `exp_probs_b` | F32 (~5 MB total) | in place; the converter forces these F32 (`tensor_force_quant`) |
| loop controllers (`loop_msg_query`, `ctrl_down/gain/forget`) | **F32, 113.7 MB** | **BF16 by default** (user ruling 2026-11-26): converted once at load (56.8 MB). `Inkling2Options.LoopController` / `DOTLLM_INKLING2_LOOP_CONTROLLER` keep F32 (in place), Q8_0 and F16 as A/B overrides. `loop_msg_query` is widened to F32 per visit into native scratch (4 x 2304 floats) |
| KV history | — | F16 (llama.cpp default KV type); K is L2-clipped to 22.6 and V RMS-normed, so the range is safe |
| conv state | — | **BF16 by default** (user ruling 2026-11-26; F16 is ruled out and removed). F32 stays as an A/B override (`DOTLLM_INKLING2_CONV_STATE`); 3.8 MB at F32 for all 48 slots |
| activations | — | F32 transient scratch inside a forward call (the CPU kernels take F32 activations), held in **native** 64-byte-aligned arenas (`Inkling2Scratch`; bump allocation, rewound per layer visit, pooled per model, freed on `Dispose`); nothing wider than the GGUF tensors is stored |

**Known bounded expansion.** `loop_msg_query` is stored BF16 but widened to F32 per core visit (pass > 0) into native scratch for the depth-mix dot: 4 x 2304 x 4 B = 36,864 B transient per visit, 24 visits per call, never resident. Reading the BF16 rows directly in the dot is a follow-up.

**Managed allocation (CLAUDE.md: no managed arrays for tensor data).** The first CPU reference allocated a fresh `new float[]` for almost every activation of every layer visit (48 visits per token), `ToArray()` on the residual queries, and a managed F32 copy of the loop message query per visit. Measured with `GC.GetTotalAllocatedBytes(precise: true)` on the NVFP4 file, warm, `Inkling2AllocationTests`: **before** 449.12 MB per 32-token prefill call and 15.98 MB (15,980,752 B) per decoded token; **after** 0.00 MB per prefill call and 104 B per decoded token (the returned logits tensor object). The test budget is 32 KB per decoded token; reintroducing one `new float[2304]` per layer visit measures 443,624 B per token and fails it (mutant killed). The test runs in a non-parallel collection because the measurement is process-wide.

**Oracle.** llama.cpp b12780 CPU (`-dev none -ngl 0`; llama.cpp's Vulkan backend is not an oracle), model `Inkling-2-Mini-NVFP4.gguf`. Gate data in `~/.dotllm/test-cache/token-oracles/inkling2/`:
`passage.txt` (866 tokens), `nvfp4-wikitext-c1280-fa-off.kld` (1280-token windows, section 10), `nvfp4-passage-c256-fa-off-logscale-floor1.kld` (llama.cpp run with `--override-kv inkling2_loop.attention.log_scale_floor=int:1`, section 10), `nvfp4-passage-c256*.kld` (`llama-perplexity --kl-divergence-base`, `-c 256`, 2 chunks, 254 scored positions, flash attention on/off),
`greedy-nvfp4*.json` (8 prompts x 48 greedy tokens with the oracle's own token ids and top-1/top-2 margins). **dotLLM is fed the oracle's token ids**, so tokenizer differences cannot contaminate the model KL.
The KL file keeps log-probs as uint16 over `[max-16, max]`; floor ~1e-6.

**Noise floor.** llama.cpp's own flash-attention on/off runs differ from each other by KL 4.0e-3 (the FA path rounds `bias + mask` to F16) and diverge in greedy decoding on 2 of 8 prompts, so
neither a KL below ~4e-3 nor free-running greedy identity is a meaningful bar against a single llama.cpp configuration. The gate therefore uses (a) mean KL against the F32-mask (`-fa off`) oracle,
(b) top-1 agreement, and (c) **teacher-forced** greedy agreement, counting only steps where the oracle's top-1/top-2 margin is >= 0.3 nats (free-running prefix identity is reported, not gated).

## 9. Not in this slice

- **Vision**: implemented on the CPU in issue #2035, see section 11 (Vulkan vision is a later slice).
  images enter the text stream as `<|content_image|><|unused_200054|>` placeholders whose embeddings are injected *without* `token_embd_norm` (llama.cpp: "do not normalize raw embeddings input"). The loader must accept an embedding-input forward.
- **DSpark drafter** (`dspark-Inkling-2-Mini-Q8_0.gguf`, `arch = dflash`): 3 layers (intermediate 9216, local layers 0 and 2), block size 8, a Markov head (`markov_rank 256`, `rnn_hidden`, a folded `markov_w2 = unembed @ hidden_out`),
  `target_layer_ids = [5, 23, 41, 48]` counted in **layer visits** (the target exposes the input of visit `n` and the final norm output as id 48: `res->t_layer_inp[visit]` in `inkling2-loop.cpp`). The CPU model keeps the trace hook points
  (`pass{n}.l{ip}`, `result_norm`) where those features would be tapped.
- Vulkan / CUDA text decoders (NVFP4 expert kernels must read the blocks in place), optimisation, llama.cpp Vulkan comparison.

## 10. Results (CPU reference vs llama.cpp b12780 CPU, `Inkling-2-Mini-NVFP4.gguf`, Strix Halo, 2026-11-26)

**Defaults since the user rulings of 2026-11-26: loop controller BF16, conv state BF16.** Gate (`Inkling2RealModelTests`, 254 scored positions; dotLLM sub-chunks the prefill by 32 tokens, so carried KV/conv state is exercised; chunk size does not change the KL with F32 state):

| arm | mean KL | max KL | top-1 | teacher-forced greedy mismatches (of 384) |
|---|---|---|---|---|
| **dotLLM, defaults (BF16 controller + BF16 conv state)** vs llama.cpp `-fa off` (the gate) | **4.10e-3** | 9.0e-2 | 98.8% | 2 (0 of 345 at margin >= 0.3; margins 0.032, 0.053); free-running prefixes 7/48/48/48/48/48/48/48 |
| dotLLM, all-F32 A/B arm (`DOTLLM_INKLING2_LOOP_CONTROLLER=f32 DOTLLM_INKLING2_CONV_STATE=f32`; the pre-refactor configuration) | 3.96e-3 | 6.6e-2 | 98.8% | 6 (0 of 345 at margin >= 0.3; the six are at margins 0.015-0.141) |
| dotLLM vs llama.cpp default (flash attention on) | 5.39e-3 | 3.1e-1 | 98.0% | — |
| llama.cpp `-fa on` vs `-fa off` (its own noise floor) | 4.05e-3 | 1.9e-1 | 99.2% | greedy diverges on 2 of 8 prompts |

The all-F32 arm is bit-for-bit the pre-refactor result (mean KL 3.959e-3, max 6.589e-2, free-running prefixes below), i.e. the native-scratch refactor changed no numerics. Free-running greedy prefix identity of the all-F32 arm vs the `-fa off` oracle (of 48 tokens, 8 prompts): 10 / 3 / 48 / 48 / 48 / 27 / 12 / 48; every divergence is at an oracle top-1/top-2 margin below 0.15.

**Mutants** (same gate; re-measured with the BF16 defaults, unmutated mean KL 4.10e-3; all killed):

| mutant | mean KL | top-1 |
|---|---|---|
| 2 loop passes instead of 3 | 0.635 | 69.7% |
| no final logit softcap | 2.338 | 98.8% (KL is the signal here: top-1 alone would miss it) |
| attention sink = 0 | 0.354 | 74.4% |
| no relative-position bias | 4.876 | 13.8% |
| NVFP4 per-expert `.scale` ignored | 13.33 | 1.6% |
| loop controllers off (no messages) | 1.018 | 63.0% |
| **no log scaling** (256-token passage, shipped floor 8192) | 3.96e-3 (identical) | 98.8% | survives this gate by construction: the factor is exactly 1 below position 8191; see the floor-1 test below |

**Log scaling, end to end (floor overridden to 1).** The factor is `1 + 0.1 ln(max(1, (pos+1)/floor))`; with the shipped floor of 8192 it is exactly 1 below position 8191, but it does NOT need a run longer than 8K: llama.cpp can be run with `--override-kv inkling2_loop.attention.log_scale_floor=int:1` (oracle `nvfp4-passage-c256-fa-off-logscale-floor1.kld`) and dotLLM with `LogScaleFloor = 1`, which puts the scaling in play from position 1 of the 256-token passage. `Inkling2RealModelTests.LogScale_Floor1_MatchesOverriddenLlamaCpp`: correct (BF16 defaults) mean KL **4.52e-3** (max 6.6e-2, top-1 95.7%), the `NoLogScale` mutant **3.37e-2** (max 1.04, top-1 92.9%): killed (gate: correct < 8e-3, mutant >= 1.5e-2). The reviewer measured 5.26e-3 vs 3.48e-2 on the same oracle at `2e40d71e` with the F32 defaults.

**Window / extent beyond 1024 (reviewer evidence).** Oracle `nvfp4-wikitext-c1280-fa-off.kld` (1280-token windows): mean KL **1.76e-3** below position 1024 and **1.59e-3** at or above it, so the 1024-key local window and the 1024-row relative-bias extent are exercised and agree with llama.cpp. Known gate limit: the one-key-wider-window mutant (1.69e-3) and the bias-clamp mutant (1.83e-3) survive against 1.59e-3 for the correct implementation; the gate cannot see them (tracked with the local KV ring in #2033).

**Loop-controller storage (user ruling 2026-11-26: BF16 default; first table measured with F32 conv state, second with the BF16 conv-state default).** Same gate; bytes are the stored size of `loop_msg_query` + 12 layers x 4 x (`down` 4608x64, `gain` 64x2304, `forget` 64x2304):

| storage | bytes | mean KL | max KL | top-1 | teacher-forced mismatches | free-running prefixes |
|---|---|---|---|---|---|---|
| F32 (GGUF, in place) | 113.7 MB | 3.96e-3 | 6.6e-2 | 98.8% | 6 | 10/3/48/48/48/27/12/48 |
| Q8_0 | 30.2 MB | 3.52e-3 | 6.9e-2 | 98.8% | 6 | 7/3/48/48/48/27/48/48 |
| F16 | 56.8 MB | 3.55e-3 | 7.6e-2 | 98.4% | 5 | 10/18/48/48/48/27/12/48 |
| BF16 | 56.8 MB | 3.37e-3 | 6.9e-2 | 99.2% | 4 | 10/18/48/48/48/27/48/48 |

Re-measured with the BF16 conv-state default (2026-11-26, after the refactor):

| storage | mean KL | max KL | top-1 | teacher-forced mismatches (confident) |
|---|---|---|---|---|
| F32 | 3.91e-3 | 1.12e-1 | 98.8% | 4 (0) |
| Q8_0 | 3.94e-3 | 1.11e-1 | 99.6% | 4 (0) |
| F16 | 4.21e-3 | 1.10e-1 | 99.2% | 3 (0) |
| **BF16 (default)** | 4.10e-3 | 9.0e-2 | 98.8% | 2 (0) |

The four are indistinguishable at this resolution (differences of 5e-4 in KL are below the oracle noise floor; no mismatch at a confident margin in any arm). Q8_0 saves 83.5 MB.

**Conv-state storage** (3.8 MB at F32 for all 48 slots; the sweep is only meaningful because the harness carries state across calls): F32 3.96e-3 / max 6.6e-2; **BF16 3.91e-3 / max 0.11** (4 mismatches, 0 confident); **F16 5.65e-3 / max 0.24, top-1 97.6%** — F16 is visibly worse than BF16 (a value range the 5-bit exponent cannot hold), BF16 is on par with F32 on mean KL. With the loop controller at BF16 the conv-state arms are F32 3.37e-3 (4 mismatches) and BF16 4.10e-3 (2 mismatches), 0 confident in both.

**Throughput** (32 threads, `cpu-bench` lock held, NVFP4 file, F32 conv state; one measurement, llama.cpp b12780 CPU measures ~10.5 tok/s decode on the same file): prefill 128 tokens 21.4 tok/s, decode 9.1 tok/s (110 ms/token) before the refactor; 22.8 / 9.8 tok/s (102 ms/token) after, BF16 defaults. Not tuned: attention, depth mixing and the convs are scalar loops, MoE is bucketed per expert.

## 11. Vision (CPU, issue #2035)

`mmproj-Inkling-2-Mini-Q8_0.gguf` (`general.architecture = clip`, projector `inkling2v`; llama.cpp b12780 `tools/mtmd/models/inkling2v.cpp`, `mtmd-graph.cpp` `build_vit`,
`mtmd-image.cpp` `mtmd_image_preprocessor_inkling2v`). Code: `src/DotLLM.Models/Vision/` (`Inkling2VisionEncoder`, `Inkling2ImagePreprocessor`, `ImageDecoder`, `VisionSupport`),
contracts in `src/DotLLM.Core/Vision/` (`IImageEncoder`, `ImageEmbeddings`, `IImageInputModel`, `ImagePlaceholders`).

**Preprocessing.** No fixed-size resize: the image keeps its size and is padded right/bottom to a multiple of `patch_size * merge = 40`. The padded buffer starts at **-1.0 in raw pixel units**
and is then normalised with `(v - 127.5) / 127.5` (mean = std = 0.5), so padding is -1.0078, not -1 (a reference quirk, kept). Over `4096 tokens * 1600 px` the reference resizes (bicubic); we
area-average instead, so over-limit images are not parity-gated. Tokens = `(W/40) * (H/40)`.

**Tower.** Patch conv 20x20x3 -> 1024 (F32, no bias; the temporal-2 kernel is already folded into one frame; the reference computes it through an **F16 im2col**, so normalised pixels are rounded to
F16 first, replicated). The 32x32 position embeddings are interpolated with ggml's bilinear upscale (half-pixel centres, clamped, no antialias; used as-is at a 32x32 grid). CLS (`class_embd + class_position_embd`)
is prepended at RoPE position (0, 0); patches are (x+1, y+1). LayerNorm (eps 1e-5, `pre_ln`, no post-LN), then 23 x [LN, QKV+bias, 2D RoPE theta 1e4 (head 64: dims 0-31 rotate adjacent pairs by the column position,
dims 32-63 by the row position), full non-causal attention (scale 1/8), o-proj+bias, **LayerScale ls1**, residual, LN, up+bias, erf GELU, down+bias, **ls2**, residual]. CLS is dropped, patches are merged
2x2 (a merged token is `[(2i,2j), (2i+1,2j), (2i,2j+1), (2i+1,2j+1)]`, x fastest, row-major over the merged grid), then `mm.1` (4096x4096, no bias), erf GELU, `mm.2` (4096 -> 2304, no bias).

**Memory / precision.** Q8_0 matrices are read in place through the Q8_0 GEMM (never expanded). `patch_embd` (4.9 MB), `position_embd` (4 MB), biases and layer scales stay F32 in the file and are read in
place (listed for sign-off, not converted). Activations are per-call F32 transients in native scratch (`Inkling2Arena`; a warm 640x640 encode allocates < 256 KB of managed memory,
`Inkling2VisionAllocationTests`). The stored result is **F16** (`ImageEmbeddings`, `tokens x 2304 x 2` bytes: 69 KB for a 15-token image, 2 MB at 640x640, 19 MB at the 4096-token limit).
Attention runs per (head, 16-query block) with a native score tile, so memory is O(N), not O(N^2).

**Prompt side.** The template renders ONE `<|content_image|><|unused_200054|>` per image part; the engine widens the placeholder to the image's token count (HF `Inkling2Processor`) and
`Inkling2TransformerModel.ForwardWithImages` replaces those positions' token embeddings with the image rows, **bypassing `token_embd_norm`** (llama.cpp: "do not normalize raw embeddings input").
llama.cpp's mtmd feeds `<|content_image|>` as text, then the image rows (no end marker); the token streams are identical (verified: oracle chunk 1 ends `... 200000 200005`, no doubling).
Attention over image tokens is causal (`mtmd_decode_use_non_causal` is false for inkling2v). A request with images reuses and retains no prefix cache (token ids alone cannot tell two images apart).

**Plumbing.** `ChatMessage.ContentParts` (text / image_url) is rendered to the template as the OpenAI part list; `InferenceOptions.Images` + `ImagePlaceholderTokenId` carry the encoded images into
`TextGenerator`; CLI `dotllm run --image <png|ppm> [--mmproj <gguf>]` (mmproj defaults to the `mmproj*.gguf` next to the model); server `/v1/chat/completions` accepts `image_url` parts
(`data:` URIs only, projector next to the model or `DOTLLM_MMPROJ`, loaded on first use; the continuous-batch scheduler is bypassed for such requests). PNG (all colour types, non-interlaced) and binary
PPM are decoded without dependencies; JPEG is refused by name. A backend plugs in by implementing `IImageEncoder` (host F16 output) and `IImageInputModel`.

**Oracle.** `tools/inkling2-vision-oracle/`: `oracle.c` (a C program over `llama.dll` + `mtmd.dll` of b12780: `mtmd_tokenize` / `mtmd_encode_chunk` / `mtmd_get_output_embd`, then
`mtmd_helper_eval_chunks` + greedy teacher forcing, `-fa off`, CPU; dumps the full F32 embeddings and the full-vocabulary logits of every step), `gen_inputs.py` (test images and prompts),
`trim_trace.py` (trims `llama-mtmd-debug -p encode` dumps to per-checkpoint corner values + sums). Cached under `~/.dotllm/test-cache/token-oracles/inkling2/vision/`.

**Results (CPU, vs llama.cpp b12780 CPU).**

| gate | result |
|---|---|
| per-op checkpoints (`llama-mtmd-debug`, rainbow 80 = interpolated, 640 = identity), early stages | equal to the printed precision (<= 5e-5 = 4 decimals); later stages drift to ~1e-2 of the corner scale, tensor sums within 3e-4 of N*rms |
| embeddings on the real path (F32 oracle vs our F16), relRMS / min row cosine | 190x110: 2.1e-2 / 0.9992; 80: 1.8e-2 / 0.9998; 640: 3.3e-2 / 0.9950 |
| end-to-end teacher-forced KL of text logits (full vocabulary), 13 rows per image | 80: 1.8e-3; 190x110: 1.07e-2; 640: 9.6e-3; top-1 13/13 on all, 0 greedy mismatches of 12 (text-only floor on this model: 4e-3) |

The embedding gap is a **noise floor, not an error**: perturbing our own input by ~1e-4 relative (F16 im2col rounding on/off) moves the embeddings by the same 2.1% relRMS, because 23 layers of Q8_0
activation quantisation amplify boundary flips. So the embedding gate (relRMS < 6%, cosine > 0.99) catches gross errors only; the sharp gates are the early checkpoints and the end-to-end KL.

**Mutants** (all in the committed tests; `Inkling2VisionMutant`, `Inkling2Mutant.NormalizeImageEmbeddings`). Checkpoint gate at n = 80: align_corners posemb, transposed posemb, row/col RoPE swapped, NEOX RoPE,
merge order, no merge, no LayerScale, no CLS all killed (tanh GELU only marginally, pixel_shuffle rel 0.108 vs 0.08; it is separated by the op-level GELU test: erf error < 1e-6 vs 4.7e-4 for tanh).
Embedding gate: BGR (relRMS 0.41) and pad = 0 after normalisation killed. End-to-end mean KL (gate 0.015): normalise image embeddings 0.53, NEOX RoPE 0.51, merge skipped 0.22, no LayerScale 1.72, no CLS 1.12,
embeddings not injected 1.38, BGR 0.37, RoPE axes swapped 0.95, merge order 0.062, align_corners 0.032, transposed posemb 0.0195.

**Known gate limits (survivors).** The reference's pad quirk (-1.0 raw vs 0.0 raw, a 0.0078 difference on pad pixels) and the F16 im2col rounding are below the 2% noise floor at the embedding level and are
not separately gated (the early checkpoints pin the F16 rounding: patch_embd matches to 5e-5). Over-limit images (> 4096 tokens) use a different resize than the reference. The position-embedding mutants are only
2-3x the end-to-end noise floor (sharp at the checkpoint gate).
