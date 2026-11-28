# Inkling 2 (Thinking Machines) — architecture notes

**Status:** CPU reference forward (issue #2026), Vulkan text decoder (issue #2034, section 11), the CPU DSpark drafter (issue #2039, sections 13-15) and the Vulkan DSpark drafter (issue #2046, section 16), epic #2025. The CPU vision tower is section 12 (issue #2035). CUDA and Vulkan vision are later slices (section 9).
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

- **Vision**: implemented on the CPU in issue #2035 (section 12); the Vulkan text model accepts image embeddings since #2062 (section 12a) and the Vulkan ViT encoder runs the tower on the GPU (section 12b, both #2062).
  images enter the text stream as `<|content_image|><|unused_200054|>` placeholders whose embeddings are injected *without* `token_embd_norm` (llama.cpp: "do not normalize raw embeddings input"). The loader must accept an embedding-input forward.
- **DSpark drafter**: implemented on the CPU in issue #2039, see sections 13-15; the Vulkan drafter (#2046) is section 16. `dspark-Inkling-2-Mini-Q8_0.gguf`, `arch = dflash`: 3 layers (intermediate 9216, local layers 0 and 2), block size 8, a Markov head (`markov_rank 256`, `rnn_hidden`, a folded `markov_w2 = unembed @ hidden_out`),
  `target_layer_ids = [5, 23, 41, 48]` counted in **layer visits** (the target exposes the input of visit `n` and the final norm output as id 48: `res->t_layer_inp[visit]` in `inkling2-loop.cpp`). The CPU model keeps the trace hook points
  (`pass{n}.l{ip}`, `result_norm`) where those features would be tapped.
- CUDA text decoder (NVFP4 expert kernels must read the blocks in place), optimisation, llama.cpp Vulkan comparison. (The Vulkan text decoder landed in #2034, section 11.)

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

## 11. Vulkan forward (issue #2034)

`VulkanInkling2TransformerModel` + `native/vulkan/shaders/inkling2_*.glsl` (wrappers from `native/vulkan/gen_inkling2_wrappers.py`; the dense projections, the router and the
LM head reuse `olala_gemv_{q8_0,bf16}` and `olala_moe_route`). One command buffer per chunk of 128 tokens; the model owns its sequence state like the CPU model.

**Precision (user rulings 2026-11-26).** Activations BF16 (rounded once per store, F32 in registers / shared memory: norms, softmax, attention sums, the cross-key-group
combine in shared memory). KV F16 (`Inkling2VulkanOptions.Kv`, `DOTLLM_INKLING2_KV=q8_0` for the q8_0 evaluation). Conv state BF16, loop-controller matrices BF16 (57 MB, converted once at load).
`_f32act` shaders build the F32-activation reference arm (a control, never the default). No split-KV, hence no stored F32 partials. The only F32 on the device are the GGUF's own F32 weight
vectors (norm gains, depth queries, clips, conv taps, router, expert scales: ~33 MB, uploaded verbatim) and the logits.

**In place, no expansion.** NVFP4 expert banks (layers 0-19, with the per-expert `.scale` applied to the matmul output) and Q8_0 banks (20-23) are decoded in registers by
`inkling2_moe_gate_up_{nvfp4,q8_0}` (gate row j and up row ff+j in one workgroup, scale, SITU, one BF16 store) and `inkling2_moe_down_{nvfp4,q8_0}` (scale, then routing weight, summed over the 8 slots).
Existing NVFP4 kernels were not reused: `moe_indexed_matmul_nvfp4_f32` takes F32 activations and `matmul_nvfp4_mmvq` Q8_1-quantised activations, neither fits the BF16 ruling. One workgroup per (row, token-slot pair),
so decode and prefill share the kernel; an expert-grouped prefill GEMM is slice 5.
Weights load through the existing path: on the Strix box 250 tensors / 17,497 MB went DirectStorage (no host bounce); 338 small tensors (5.4 MB: the F32 vectors) are staged, plus the 56.8 MB BF16 controller copy
(a narrowing transcode through a transient native buffer, then staged). Reported by `WeightLoadReport`.

**KV.** Position-major `[position][kv head][128]`. Local slots (40 of 48) are a ring of `window + MaxChunk` = 1024 + 128 = 1152 positions, flat in the sequence length: a chunk of T tokens writes all its K/V before attending,
so a ring of exactly 1024 positions would let a later token overwrite a key an earlier token of the same chunk still reads (`window + T - 1` slots are needed). Global slots (8 of 48) are linear and double from 1024 positions
(context length is 1,048,576). Bytes: F16 4,096 B per token per global slot (32,768 B per token over 8 slots), q8_0 2,176 B (17,408 B); local rings 40 x 1152 x 4,096 B = 189 MB (F16) / 100 MB (q8_0); conv state 1.92 MB.

**Gates** (`Inkling2VulkanGateTests`, vs the same llama.cpp b12780 CPU `-fa off` oracles as the CPU reference; chunk 128, 32-token sub-chunks):

| gate | CPU reference | Vulkan F16 KV | Vulkan q8_0 KV |
|---|---|---|---|
| passage, mean KL / top-1 | 4.10e-3 / 98.8% | **3.61e-3** / 98.8% | 4.53e-3 / 97.6% |
| passage, F32-activation control | 3.96e-3 (F32 CPU arm) | 3.12e-3 | |
| floor-1 log scale | 4.52e-3 | **4.31e-3** / 97.2% | |
| 1280-token windows, pos < 1024 / >= 1024 | 1.76e-3 / 1.59e-3 | **1.61e-3 / 2.01e-3** | 2.16e-3 / 2.32e-3 |
| teacher-forced mismatches (confident) | 2/384 (0) | 3/384 (0) | 6/384 (0) |

The 1280 run was repeated with chunk 100 / sub-chunk 97 (not dividing 1024): identical (1.768e-3). Vulkan vs the CPU reference on the same ids, first prompt: F32-activation arm mean KL 2.0e-4 (the residual is the CPU's own activation quantisation), BF16 arm below the oracle noise floor.
Op-level parity (`Inkling2VulkanKernelTests`): every new shader against a double reference built from `Inkling2Ops` / `Inkling2ConvState` / `Dequantize`; 37 mutants applied to a scratch copy of the shaders (tracked `.spv` untouched, loaded via `DOTLLM_INKLING2_SPV_OVERRIDE`): 36 killed, 1 equivalent (initialising the attention max at -inf instead of 0 is shift-invariant).

**Throughput** (Strix 8060S, `Inkling-2-Mini-NVFP4.gguf`, 300-token prefill then 32 decode steps, 3 warm rounds after a warm-up, not tuned): F16 KV prefill 81-95 tok/s, decode 26.5-28.8 tok/s; q8_0 KV prefill 68-88, decode 25.4-28.5. llama.cpp b12780 (CPU-only build: `llama-bench -dev none`, 16 threads, run back-to-back, same file): pp300 48.3 +- 0.3, tg32 13.9 +- 0.2 tok/s. The b12780 build has no Vulkan backend.
dotLLM CPU reference: 22.8 / 9.8 tok/s. Levers for slice 5: ~1,900 dispatches per decoded token (controllers are 5 dispatches each x 4 x 24 core visits), expert-grouped prefill, split-submit profiling, dp4a MMVQ for NVFP4.

## 12. Vision (CPU, issue #2035)

`mmproj-Inkling-2-Mini-Q8_0.gguf` (`general.architecture = clip`, projector `inkling2v`; llama.cpp b12780 `tools/mtmd/models/inkling2v.cpp`, `mtmd-graph.cpp` `build_vit`,
`mtmd-image.cpp` `mtmd_image_preprocessor_inkling2v`). Code: `src/DotLLM.Models/Vision/` (`Inkling2VisionEncoder`, `Inkling2ImagePreprocessor`, `ImageDecoder`, `VisionSupport`),
contracts in `src/DotLLM.Core/Vision/` (`IImageEncoder`, `ImageEmbeddings`, `IImageInputModel`, `ImagePlaceholders`).

**Preprocessing.** No fixed-size resize: the image keeps its size and is padded right/bottom to a multiple of `patch_size * merge = 40`. The padded buffer starts at **-1.0 in raw pixel units**
and is then normalised with `(v - 127.5) / 127.5` (mean = std = 0.5), so padding is -1.0078, not -1 (a reference quirk, kept). Over `4096 tokens * 1600 px` the reference resizes (a Pillow-compatible
bicubic, `img_tool::resize_pillow`); dotLLM **refuses** such an image by default and, with the over-limit option `downscale`, resizes it with a port of that same bicubic
(`ImageResampler`, byte-identical to `PIL.Image.resize(BICUBIC)` on the pinned cases, `tools/inkling2-vision-oracle/pillow_bicubic_golden.py`). The over-limit path is
gated against Pillow, not against an llama.cpp encode of a 4,096-token image. Tokens = `(W/40) * (H/40)`.

**Tower.** Patch conv 20x20x3 -> 1024 (F32, no bias; the temporal-2 kernel is already folded into one frame; the reference computes it through an **F16 im2col**, so normalised pixels are rounded to
F16 first, replicated). The 32x32 position embeddings are interpolated with ggml's bilinear upscale (half-pixel centres, clamped, no antialias; used as-is at a 32x32 grid). CLS (`class_embd + class_position_embd`)
is prepended at RoPE position (0, 0); patches are (x+1, y+1). LayerNorm (eps 1e-5, `pre_ln`, no post-LN), then 23 x [LN, QKV+bias, 2D RoPE theta 1e4 (head 64: dims 0-31 rotate adjacent pairs by the column position,
dims 32-63 by the row position), full non-causal attention (scale 1/8), o-proj+bias, **LayerScale ls1**, residual, LN, up+bias, erf GELU, down+bias, **ls2**, residual]. CLS is dropped, patches are merged
2x2 (a merged token is `[(2i,2j), (2i+1,2j), (2i,2j+1), (2i+1,2j+1)]`, x fastest, row-major over the merged grid), then `mm.1` (4096x4096, no bias), erf GELU, `mm.2` (4096 -> 2304, no bias).

**Memory / precision.** Q8_0 matrices are read in place through the Q8_0 GEMM (never expanded). `patch_embd` (4.9 MB), `position_embd` (4 MB), biases and layer scales are F32 in the file. **Biases and layer scales stay F32** (tiny vectors, read in place).
`patch_embd` and `position_embd` are an A/B arm (user ruling 2026-11-27): `DOTLLM_VISION_EMBD_PRECISION=f32|f16|bf16` (or `Inkling2VisionEncoder.Load(..., embedPrecision)`, default `bf16`)
converts them once at load to F16 or BF16 (2.45 + 2.0 MB resident instead of reading 4.9 + 4.0 MB F32; the GEMM reads them directly, `position_embd` is converted in registers per
element). **The default is BF16** (user ruling 2026-11-27, after the table below); F32 and F16 are the A/B arms. Activations are per-call F32 transients in native scratch (`Inkling2Arena`; a warm 640x640 encode allocates 48 managed bytes,
`Inkling2VisionAllocationTests`). The scratch is **released after each call down to 128 MB** (review of #2037: a 4,096-token image needs a 2,517 MB working set, which used to stay
resident for the life of the encoder; measured on a 2560x2560 noise image: peak 2,517 MB, 0 held after the call, 425.9 s encode while another agent held the CPU;
`Inkling2VisionLimitsTests.BigImageScratch_IsReleasedAfterTheEncode` pins the release on a 256-token image with a 32 MB keep). The stored result is **F16** (`ImageEmbeddings`, `tokens x 2304 x 2` bytes: 69 KB for a 15-token image, 2 MB at 640x640, 19 MB at the 4096-token limit).
Attention runs per (head, 16-query block) with a native score tile, so memory is O(N), not O(N^2).

**Prompt side.** The template renders ONE `<|content_image|><|unused_200054|>` per image part; the engine widens the placeholder to the image's token count (HF `Inkling2Processor`) and
`Inkling2TransformerModel.ForwardWithImages` replaces those positions' token embeddings with the image rows, **bypassing `token_embd_norm`** (llama.cpp: "do not normalize raw embeddings input").
llama.cpp's mtmd feeds `<|content_image|>` as text, then the image rows (no end marker); the token streams are identical (verified: oracle chunk 1 ends `... 200000 200005`, no doubling).
Attention over image tokens is causal (`mtmd_decode_use_non_causal` is false for inkling2v). A request with images reuses and retains no prefix cache (token ids alone cannot tell two images apart).

**Image limits (user ruling 2026-11-27: the reviewer's defaults accepted, all configurable).** `ImageLimits` (`src/DotLLM.Core/Vision/ImageLimits.cs`): 16 images and 2^26 decoded
pixels per request, 2^25 pixels and 65,536 px per side per image, a PNG stream too short for its declared size is refused before any allocation (the deflate-ratio bound), images with a
LoRA adapter are refused (server 400, CLI error, engine `NotSupportedException`). Server: `--max-images`, `--max-request-image-pixels`, `--max-image-pixels`, `--max-image-side`,
`--image-over-limit refuse|downscale`, or `DOTLLM_IMAGE_MAX_IMAGES`, `DOTLLM_IMAGE_MAX_REQUEST_PIXELS`, `DOTLLM_IMAGE_MAX_PIXELS`, `DOTLLM_IMAGE_MAX_SIDE`, `DOTLLM_IMAGE_OVER_LIMIT` (flag wins; a malformed
value fails at startup); `dotllm run` reads the variables and has `--image-over-limit`. **Over-limit (more than 4,096 image tokens): refuse (default) or downscale.** An extreme aspect ratio
that no downscale can fit is refused under both.

**Plumbing.** `ChatMessage.ContentParts` (text / image_url) is rendered to the template as the OpenAI part list; `InferenceOptions.Images` + `ImagePlaceholderTokenId` carry the encoded images into
`TextGenerator`; CLI `dotllm run --image <png|ppm> [--mmproj <gguf>]` (mmproj defaults to the `mmproj*.gguf` next to the model); server `/v1/chat/completions` accepts `image_url` parts
(`data:` URIs only, projector next to the model or `DOTLLM_MMPROJ`, loaded on first use; the continuous-batch scheduler is bypassed for such requests). PNG (all colour types, non-interlaced) and binary
PPM are decoded without dependencies; JPEG is refused by name. A backend plugs in by implementing `IImageEncoder` (host F16 output) and `IImageInputModel`;
`ForwardWithImages` takes the engine's `IKvCache` (review of #2037), so a model with an external KV handle (the Vulkan Inkling 2 model's `Inkling2KvHandle`) can implement it; there is no LoRA or MTP
parameter, the engine refuses an adapter with images and does not run MTP for an image request. Chunk-cache annotations (`chunk` on a text part, `at.part`) are supported on array content (docs/CHUNK_CACHE.md).

**Embedding precision (PR #2037, F32 vs F16 vs BF16 for `patch_embd` / `position_embd`; the user ruled BF16 the default on 2026-11-27).** relRMS of our F16 output embeddings:

| arm | control: F32-weights mmproj vs llama.cpp (80 / 190x110) | control: vs our F32 arm | shipped Q8_0 vs llama.cpp (80 / 190x110 / 640) | shipped vs our F32 arm |
|---|---|---|---|---|
| F32 | 2.12e-4 / 2.08e-4 | 0 | 1.78e-2 / 2.11e-2 / 3.25e-2 | 0 |
| F16 | 2.12e-4 / 2.08e-4 | 2.1e-5 / 1.7e-5 | 2.03e-2 / 2.18e-2 / 3.28e-2 | 1.8e-2 / 2.2e-2 / 3.2e-2 |
| BF16 | 2.12e-4 / 2.08e-4 | 2.2e-5 / 1.8e-5 | 1.79e-2 / 2.42e-2 / 3.49e-2 | 1.7e-2 / 2.6e-2 / 3.3e-2 |

End-to-end teacher-forced KL of the text logits (13 rows per image, shipped Q8_0 mmproj, **after the conv-history fix**; mean KL vs llama.cpp / mean KL vs our own F32 arm / top-1 / greedy mismatches):

| image | F32 | F16 | BF16 |
|---|---|---|---|
| 80 | 2.45e-3 / - / 13/13 / 0 | 6.48e-3 / 3.89e-3 / 13/13 / 0 | 3.40e-3 / 2.35e-3 / 13/13 / 0 |
| 190x110 | 4.43e-3 / - / 13/13 / 0 | 3.60e-3 / 2.56e-3 / 13/13 / 0 | 2.59e-3 / 4.90e-3 / 13/13 / 0 |
| 640 | 5.18e-3 / - / 13/13 / 0 | 4.76e-3 / 3.79e-3 / 12/13 / 1 | 2.58e-3 / 3.61e-3 / 13/13 / 0 |

(Before the fix, with the chunk-boundary error in every number: F32 1.82e-3 / 1.07e-2 / 9.60e-3, F16 3.44e-3 / 4.49e-3 / 6.70e-3, BF16 5.01e-3 / 3.22e-3 / 3.40e-3, one BF16 mismatch on 640.)

Reading: on the F32-weights control (the sharp one: the Q8_0 activation noise of the reference is gone) both arms move the embeddings by about 2e-5 relRMS, a tenth of the 2.1e-4 that
the F16 storage of our own output already costs, and the gap to llama.cpp does not change at the printed precision. On the shipped mmproj the arms are inside the 2% embedding noise floor and the end-to-end KL (now free of chunk-boundary error)
still does not order them: F16 and BF16 are below F32 on two or three of the images, and the arm-vs-F32 KL (2e-3 to 5e-3) is the size of the F32 arm's own gap to llama.cpp. The single visible flip moved from BF16 (before the fix) to F16 (after), on 640:
it is noise, not an arm effect. Memory saved: 4.45 MB. Tests: `Inkling2VisionEmbedPrecisionTests.Report_EmbedPrecisionArms_Embeddings`,
`Inkling2VisionEndToEndTests.Report_EmbedPrecisionArms_EndToEndKl`; the control files live next to the oracles in `~/.dotllm/test-cache/token-oracles/inkling2/vision-f32-control/`.

**Gates under the BF16 default.** The checkpoint and mutant gates (4-decimal early stages, marginal mutants) were calibrated on F32 and stay pinned to the F32 arm (`LoadEncoder()` / `Models()`); with BF16 the embedding cosine on 640
drops from 0.995 to 0.979 (relRMS 3.5e-2), tanh GELU stops being separable at the checkpoint gate (relRMS 2.57e-2 vs 2.42e-2) and the transposed-posemb mutant lands at KL 1.48e-2 against the 1.5e-2 end-to-end gate. The shipping default has its own gates:
`Embeddings_MatchLlamaCpp_UnderTheDefaultBf16` (relRMS < 6%, cosine > 0.97), and the end-to-end, single-pass and chunk-invariance tests run on the default encoder. `Default_IsBf16_OnAnEncoderLoadedWithoutAnArgument` pins the default.

**Chunked prefill of an image prompt (found in review of #2037, fixed by #2039 commit 4f01904d).** Before the fix the final-position logits of an image prompt depended on the prefill chunk size (KL vs the 32-token run
1.9e-3 at 5 tokens on the 80 image, 5.5e-2 at 7 and 1.3e-1 at 4 on 190x110, same top-1) although the `embed` trace was bit-identical across chunk sizes and the text-only control moved only 3e-5 to 2e-4. Root cause (found by the
DSpark agent, #2039): the in-batch conv history was not rounded to the BF16 state type while the stored state is, so N-row forwards differed from N single-row forwards; image rows skip `token_embd_norm`, so their large norms made the
boundary rounding visible. With the fix cherry-picked: image KL vs the 32-token run is exactly 0 at chunks 4, 5 and 7 (gate `ChunkedPrefill_ImageLogits_AreChunkSizeInvariant`, < 1e-5), also with `DOTLLM_INKLING2_CONV_STATE=f32`, and the
whole prompt in one pass (the engine default) equals the 32-token run (`ImagePrompt_SinglePassPrefill_MatchesLlamaCpp`). End-to-end mean KL vs llama.cpp after the fix, F32 arm: 2.45e-3 / 4.43e-3 / 5.18e-3 (80 / 190x110 / 640; before 1.82e-3 / 1.07e-2 / 9.60e-3);
with an F32 conv state 5.75e-3 / 1.87e-3 / 6.71e-3 (no better: the remaining gap is the reference's own Q8_0 activation noise, not state precision).

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

**F32-weights control (PR #2037 review).** With the mmproj's Q8_0 matrices dequantised to an F32 GGUF, the Q8_0 activation quantisation disappears on both sides (ggml and
`GemmF32`) and the gap collapses: relRMS 2.1e-4 (80) and 2.1e-4 (190x110), the F16 storage of our output. llama.cpp itself moves 1.9% when one input byte changes by 1 LSB (ours: 2.0%). So the 2%
Q8_0 gap is the reference's own sensitivity, and no systematic error above ~2e-4 hides in it.

**Mutants (review fixes, PR #2037, 2026-11-27).** Each was applied to the source after committing, the named tests run, and the file restored. Chunk cache on OpenAI array content: part `chunk` dropped
(10 of 14 new/changed tests failed), later-part engine offset not accumulated (5), `at.part` read as 0 (4), image part counted in the offset (5), locator sentinels not written into `ContentParts` (1).
Limits: refuse that downscales (1), decoder reading the constants not the limits (2), count cap constant (1), `DOTLLM_IMAGE_*` ignored (1), bicubic `a = -0.75` (6, all Pillow digests and the downscale test), encoder ignoring the policy (killed
by `Encoder_FollowsThePolicyItWasLoadedWith`, which the first draft lacked: that mutant initially survived). Engine: early return in `ResolveKvCache` removed (killed by the new prefix-trie test), adapter allowed with images (1), KV cache not passed (4), `max_tokens`
clamp on the raw value (1), thinking budget dropped (1), chunk-relative `TryImageRow` index (3, now on two more images), scratch not released (1), F16 arm not converting (1).

**Mutants** (all in the committed tests; `Inkling2VisionMutant`, `Inkling2Mutant.NormalizeImageEmbeddings`). Checkpoint gate at n = 80: align_corners posemb, transposed posemb, row/col RoPE swapped, NEOX RoPE,
merge order, no merge, no LayerScale, no CLS all killed (tanh GELU only marginally, pixel_shuffle rel 0.108 vs 0.08; it is separated by the op-level GELU test: erf error < 1e-6 vs 4.7e-4 for tanh).
Embedding gate: BGR (relRMS 0.41) and pad = 0 after normalisation killed. End-to-end mean KL (gate 0.015): normalise image embeddings 0.53, NEOX RoPE 0.51, merge skipped 0.22, no LayerScale 1.72, no CLS 1.12,
embeddings not injected 1.38, BGR 0.37, RoPE axes swapped 0.95, merge order 0.062, align_corners 0.032, transposed posemb 0.0195.

**Known gate limits (survivors).** The reference's pad quirk (-1.0 raw vs 0.0 raw, a 0.0078 difference on pad pixels) and the F16 im2col rounding are below the 2% noise floor at the embedding level and are
not separately gated (the early checkpoints pin the F16 rounding: patch_embd matches to 5e-5). Over-limit images (> 4096 tokens, downscale policy) use a port of the reference resize gated against Pillow, not against an llama.cpp encode of such an image. The position-embedding mutants are only
2-3x the end-to-end noise floor (sharp at the checkpoint gate).

## 12a. Vision on Vulkan, part A: the Vulkan text model takes image embeddings (issue #2062)

`VulkanInkling2TransformerModel` implements `IImageInputModel` (the CPU `Inkling2VisionEncoder` still produces the F16 `ImageEmbeddings`; the Vulkan ViT is part B).
Per prefill chunk the model gathers the F16 rows of the chunk's image positions into a compact host staging buffer, uploads it with a per-token selector (`0xFFFFFFFF` = token row), and after
`inkling2_embed` runs one more dispatch, `inkling2_embed_rows` (`native/vulkan/shaders/inkling2_embed_rows.glsl`): selected tokens' rows are overwritten with the F16 row rounded once to the BF16 activation
type, so image rows bypass `token_embd_norm` exactly like the CPU. A chunk with no image position runs the unchanged text path (no extra dispatch); decode never carries images. Spans are in absolute positions, so
chunked prefill passes the same list for every chunk. LoRA with images stays refused by the engine, CLI and server; the engine does not run MTP/DSpark for an image request.

**Byte costs** (allocated on the first image prefill only): `_imgRows` `chunk x 2304 x 2` = 589,824 B and `_imgSel` 512 B of host-visible device memory (chunk 128), plus the same rows as reused host staging
(one `ushort[chunk x 2304]`, 590 KB, allocated once). No F32 is stored: the F16 payload is copied, never widened on the host. The image embeddings themselves stay the encoder's F16 (`tokens x 2304 x 2`).

**Gates** (`Inkling2VulkanVisionTests`, CPU encoder with the BF16 default + Vulkan text model, vs the same llama.cpp b12780 oracles, `Inkling2VulkanOptions` defaults: BF16 activations, F16 KV). Teacher-forced mean KL of the text logits, 13 rows per image:

| image | Vulkan (32-token calls = one pass) | max KL | top-1 | greedy mismatches | CPU reference (section 12) |
|---|---|---|---|---|---|
| 80 | 4.85e-3 | 1.46e-2 | 13/13 | 0 | 3.40e-3 |
| 190x110 | 6.62e-3 | 2.22e-2 | 13/13 | 0 | 2.59e-3 |
| 640 | 5.41e-3 | 2.56e-2 | 13/13 | 0 | 2.58e-3 |

All under the 0.015 gate. The Vulkan numbers sit above the CPU's by 1.4e-3 to 4e-3. Decomposition (review of #2063, same oracles, mean KL x1e-3; "CPU, BF16 rows" feeds the CPU model
the encoder rows pre-rounded to BF16, "Vulkan F32 act" is the `_f32act` storage arm):

| image | CPU (F16 rows) | CPU, BF16 rows | Vulkan F32 act | Vulkan BF16 (shipped) |
|---|---|---|---|---|
| 80 | 3.40 | 8.24 | 2.72 | 4.85 |
| 190x110 | 2.59 | 3.39 | 3.84 | 6.62 |
| 640 | 2.58 | 2.30 | 3.49 | 5.41 |

The F32-activation arm, whose `embed` trace equals the CPU's bit for bit at the image rows, is within -0.7e-3..+1.3e-3 of the CPU, so the injection and addressing add no error.
The gap is the BF16 activation storage: it costs image prompts +1.9e-3 to +2.8e-3 over the F32 arm, against +0.5e-3 on the text passage gate (3.12e-3 vs 3.61e-3). Rounding only the
image rows to BF16 moves the CPU's KL by -0.3e-3 to +4.8e-3, so a single input rounding already swings this 13-row measurement by a few e-3. The F16 encoder output is shared with the CPU, so the
F16 -> BF16 double rounding is not a Vulkan-only cost. A wider residual for image prompts would need the precision-rule sign-off (CLAUDE.md rule 1).
Single-pass prefill equals the 32-token run to the printed digits (the 80 image fits one 128-token chunk; 640 spans three).
Chunk-size invariance: image-prompt logits KL vs the 32-token run is exactly 0 at chunks 4, 5 and 7 (gate < 1e-5), and the `embed` trace is bit-identical across chunkings with image positions equal to BF16(F16 row)
(`ChunkedPrefill_ImageRows_AreAddressedByAbsolutePosition`). Prefill of the image prompt (Vulkan, excluding the CPU encode): 32 tokens 0.36 s, 43 tokens 0.52 s, 284 tokens 2.89 s (warm-up included, single run, not a benchmark).

**Mutants** (committed first; mutated copies of the shader loaded through `DOTLLM_INKLING2_SPV_OVERRIDE`, tracked `.spv` untouched), each run against the op-level test `EmbedRows_OverwritesSelectedTokens` (both arms)
and the model-level addressing test on the 80 image at chunk 5: row index = token index (not the selector), F16 halves swapped, sentinel ignored (every token overwritten), no-op shader, row stride off by 2.
5 of 5 killed at both levels.

## 12b. Vision on Vulkan, part B: the Vulkan ViT encoder (issue #2062)

`VulkanInkling2VisionEncoder` (`src/DotLLM.Vulkan/`) runs the whole tower and projector on the GPU: `IImageEncoder`, output the same host F16 `ImageEmbeddings`. The Vulkan text model builds it for its own device
(`IImageEncoderProvider.CreateImageEncoder`); CLI `run --image` and the server pick it through `VisionSupport.LoadEncoderFor` (`DOTLLM_VISION_DEVICE=cpu` forces the CPU encoder). Verified end to end with
`dotllm run <gguf> --device vulkan:0 --image rainbow640.ppm`: a coherent description of the image, prefill 282 tokens in 2.9 s.

**Kernels** (`native/vulkan/shaders/vit_*.comp`, `.spv` committed; the DSpark agent's `inkling2_conv` shader is untouched):
`vit_patch_embd` (F16-rounded pixels gathered per patch x `patch_embd`, plus the bilinear position embedding, fused, one BF16 store into the residual stream at row `posOff`),
`vit_cls`, `vit_layernorm` (BF16 in/out, F32 statistics), `vit_gemm_q8` (Q8_0 weights, one 34-byte block per weight row per K step decoded into shared memory inside the tile load, BF16 X and Y, F32 accumulators in registers;
epilogues: bias / bias+erf GELU / `Y += ls * (bias + acc)` in place on the residual stream / bias + 2D RoPE of adjacent pairs from an F16 cos/sin table / GELU / F16 store; the 2x2 merge is the activation addressing of the `mm.1` GEMM),
`vit_attn` (flash-style: 16 queries x one head per workgroup, keys streamed in tiles of 32 through shared memory with an online softmax; **no score matrix of any precision is stored**; scores, running max/sum and the accumulator
live in registers and shared memory only). A long sequence is several submits: the attention is dispatched in 2,048-query chunks and the submit is flushed per chunk and per layer above 2,048 positions (a single
16k-query dispatch would outlive the GPU watchdog; the 4,096-token image ran with no watchdog event).

**Precision / expansion.** BF16 activations (user ruling); Q8_0 matrices imported in place through `VulkanWeightImportPolicy` (the same zero-copy / DirectStorage route as the text model, nothing expanded: no F16/F32 copy of any matrix);
`patch_embd` / `position_embd` BF16 by default (converted once at load through a transient native buffer, 4.55 MB resident; `DOTLLM_VISION_EMBD_PRECISION=f16` 4.55 MB, `f32` 9.11 MB read in place; the three arms agree to
the printed digits on the embedding gates); biases, LayerNorm gains and LayerScale vectors stay F32 as in the file. The RoPE table is F16 (`npos x 128` bytes: 2.1 MB at 16,385 positions), the image pixels are held as F16
(`W x H x 3 x 2` bytes: 39 MB at the 4,096-token limit; the reference rounds them to F16 anyway). The tower's activation scratch is BF16: peak 0.4 / 1.5 / 24.8 / 395.9 MB for the 4 / 15 / 256 / 4,096-token images,
released after the call when above 128 MB (`ScratchKeepBytes`). Resident weights 341.3 MB (BF16 embeddings), 345.9 MB (F32 arm). Transients that are not weights: the host F32 normalised image (78 MB at the limit) and its F16 copy are freed
after the upload, and the pixel upload stages through a transient host-visible buffer of the same size (input data, not weights).

**Embedding gates** (`Inkling2VulkanVisionEncoderTests.Embeddings_MatchReferences`; CPU references cached by `CpuCache_Populate` so the CPU and the iGPU never run together). relRMS of our F16 output:

| image | Vulkan vs CPU encoder on the F32 control (exact tower) | vs llama.cpp Q8_0 | vs llama.cpp F32 control | CPU shipping vs the same F32 control | CPU shipping vs llama.cpp Q8_0 (section 12) | min row cosine vs control |
|---|---|---|---|---|---|---|
| 80 | 1.73e-2 | 2.36e-2 | 1.73e-2 | 1.51e-2 | 1.8e-2 | 0.9997 |
| 190x110 | 1.81e-2 | 3.47e-2 | 1.81e-2 | 2.54e-2 | 2.4e-2 | 0.9995 |
| 640 | 4.11e-2 | 7.02e-2 | n/a | 6.45e-2 | 3.5e-2 | 0.976 |

The CPU reached 2.1e-4 against llama.cpp on the F32 control; the Vulkan encoder reaches **1.7e-2** (80, 190x110). That is not a defect of one kernel: it is the BF16 activation ruling. Layer by layer against the CPU
control (`Report_LayerwiseDivergenceFromF32Control`): embeddings 1.7e-3 (one BF16 rounding), then a gradual growth, 80 image 4.2e-3 after layer 0, 8.3e-3 at the output; 640 image 5.4e-3 after layer 0, 1.4e-2 (layer 5),
2.0e-2 (11), 3.3e-2 (17), 4.8e-2 (22). No step anywhere. Longer sequences grow it faster (1,025 keys in every softmax). The F16 RoPE table was not isolated as a contributor. The gates are
relRMS < 6e-2 vs the control, min row cosine > 0.97, relRMS < 9e-2 vs llama.cpp Q8_0: they catch gross errors; the sharp gates are the op-level tests and the end-to-end KL.

**End to end** (Vulkan encoder + Vulkan text model, vs the same llama.cpp oracles, teacher-forced over 13 rows; gate 0.015):

| image | Vulkan encoder: mean KL / max KL / top-1 / greedy mismatches | CPU encoder + Vulkan text (12a) | exact-tower embeddings (CPU F32 control) + Vulkan text |
|---|---|---|---|
| 80 | 3.67e-3 / 1.13e-2 / 12/13 / 1 | 4.85e-3 | 2.48e-3 (12/13, 1 mismatch) |
| 190x110 | 3.22e-3 / 1.09e-2 / 13/13 / 0 | 6.62e-3 | 5.01e-3 |
| 640 | **1.04e-2** / 3.88e-2 / 13/13 / 0 | 5.41e-3 | 1.32e-2 |

The 640 figure is 70% of the gate. It is the reference, not the encoder: feeding the **exact** tower's embeddings (no activation noise at all) gives 1.32e-2, worse than ours. The shipping CPU encoder scores lower because
it quantises activations to Q8_0 exactly like the llama.cpp oracle does, so it shares the oracle's noise; the Vulkan encoder, which never quantises activations, cannot and should not. Single-pass prefill equals the 32-token run to the printed
digits; image-prompt logits KL vs the 32-token run is exactly 0 at chunks 4, 5 and 7 (`VulkanEncoder_ChunkedPrefill_ImageLogits_AreChunkSizeInvariant`).

**Performance** (Strix 8060S, warm, single process, the Vulkan figures include preprocessing, pixel upload and the F16 download; CPU = the shipping BF16 encoder, 16 threads, cached wall times, run alone under the cpu-bench lock):

| image | image tokens | Vulkan encode | CPU encode | speedup | prefill + first token (Vulkan text) | TTFT Vulkan encoder | TTFT CPU encoder |
|---|---|---|---|---|---|---|---|
| 80 | 4 | 18.7 ms | 57 ms | 3.1x | 0.36 s | 0.37 s | 0.41 s |
| 190x110 | 15 | 21.2 ms | 109 ms | 5.1x | 0.46 s | 0.48 s | 0.57 s |
| 640 | 256 | 249 ms (211 ms in the TTFT run) | 1,075 ms | 4.3x | 3.17 s | 3.38 s | 4.24 s |
| 2560x2560 noise | 4,096 | 38.2 s (30.6 s in the TTFT run) | 650.5 s | 17x | 46.4 s | 77.0 s | 696.9 s |

TTFT = encode + whole-prompt prefill + first logits; the CPU column adds the cached CPU encode time to the same measured Vulkan prefill (the two never ran together). Run-to-run spread on the large image is ~20% (another agent
shares the iGPU between lock holders; this is not a tuned figure). Where the time goes at 4,096 tokens: 25 TFLOP of attention and 9.5 TFLOP of GEMMs in ~30-38 s, about 1 TFLOP/s overall; both the attention kernel (scalar FMAs, two
shared-memory reads per FMA) and the GEMM (64x64 tile, scalar) are untuned. Levers: coopmat tiles for the GEMM (section 11 / the Q5_0 / Q4_K coopmat templates), a 32-query register-tiled attention.

**Op-level parity and mutants** (`Inkling2VulkanVitKernelTests`, 17 tests, double-precision references written independently of the shaders; mutated shader copies compiled to a scratch directory and loaded through
`DOTLLM_INKLING2_SPV_OVERRIDE`, tracked `.spv` untouched; committed before mutating). Each mutant ran against the kernel tests and the embedding gate (7 cases: 3 images x arms):

| mutant (shader) | kernel tests failed | embedding-gate cases failed |
|---|---|---|
| posemb `align_corners` (vit_patch_embd) | 4 | 6 (the 640 image sits on the 32x32 table, where align_corners is the identity: it cannot fail there) |
| posemb transposed (vit_patch_embd) | 5 | 4 |
| posemb not added (vit_patch_embd) | 5 | 7 |
| RoPE axes swapped (vit_gemm_q8) | 1 | 7 |
| RoPE not applied (vit_gemm_q8) | 1 | 7 |
| merge order (vit_gemm_q8) | 1 | 7 |
| merge skipped (vit_gemm_q8) | 1 | 7 |
| no LayerScale (vit_gemm_q8) | 1 | 7 |
| no bias (vit_gemm_q8) | 4 | 7 |
| tanh GELU (vit_gemm_q8) | 1 (`Gelu_IsErfNotTanh_InTheNegativeTail`) | 0 |
| attention scale 1 instead of 1/8 | 2 | 7 |
| attention key tail unmasked | 1 | 4 |
| CLS row zero | 1 | 7 |
| LayerNorm without beta | 1 | 7 |

The tanh-GELU mutant **survived** the first draft of the kernel tests and the embedding gate (4.7e-4 absolute is invisible through BF16 and 6e-2 relRMS, as in the CPU section); it is killed by a dedicated test that evaluates the
epilogue at chosen points in the negative tail (zero weights, biases from -3.5 to 3), where tanh and erf differ by up to 25% relative (erf: 3.5e-3 error from BF16 only).

**Several threads on one device (issue #2068).** The server encodes an image on a `Task.Run` worker outside the request gate, so a Vulkan encode can submit while another request's text forward runs on the same
`VulkanDevice`. Vulkan requires external synchronisation of `vkQueueSubmit` on the shared queue, of a `VkCommandPool` and every command buffer recorded from it, and the device-wide hazard tracker
(`ActiveHazards`, set by a forward and nulled by every `Begin` / submit) assumed single-threaded recording. Fixed at the device, so it holds for every path (scheduler, MTP, DSpark drafter, warm-up, load-time uploads):
every `vkQueueSubmit` goes through one `QueueSubmit` under a queue lock (held for the submit call, never across the fence wait); each `SubmitContext`, staging command buffer and one-shot copy owns a private command
pool (the shared device pool is no longer recorded into); `ActiveHazards` is armed, seen and disarmed only by the arming thread (another thread's dispatches fall back to explicit barriers).
Cost: **1.00 `vkQueueSubmit` per decoded token** (measured over 32 decode steps, 32.5 ms/token) and an uncontended lock of 10.5 ns: 10 ns per token, 3e-7 of a decode step. Tests (`VulkanDeviceConcurrencyTests` and
`ConcurrentImageEncodeAndTextDecode_OnOneDevice_MatchTheirSerialRuns`: 150 image encodes beside 3 x 25 text steps; tokens and embeddings bit-identical to the serial runs, never more than one thread inside `vkQueueSubmit`).
Mutants: the queue lock removed: 4 threads inside `vkQueueSubmit` at once, the driver raised errors in the submitting threads (2 unit tests fail) and the model-level test **deadlocked the device** (it now fails by timeout after 180 s);
the shared plain hazard slot restored: `HazardTracker_ArmedOnOneThread_SurvivesAnotherThreadsSubmit_AndIsInvisibleToIt` fails (the model-level test cannot see it: the Vulkan ViT records through `OlalaKernel`, which has no tracker).
The private-pool change has no discriminating test (a shared pool race is undefined behaviour that did not show on this driver); it is covered by reasoning, not by a failing mutant. The RoPE table is now built once per
grid into native memory (`RopeTable_IsBuiltOncePerGrid`), not as a managed array per call. `DOTLLM_VISION_DEVICE` accepts `cpu`, `auto`, `gpu`, `device`; anything else is an error.

**Staged conversion (rule 3).** The one-time BF16 (or F16) conversion of `patch_embd` / `position_embd` (4.55 MB) reads the mapped F32 tensor on the host, converts into a transient native buffer and uploads it through the
staging buffer: a host bounce (read, convert, copy) of 4.55 MB at load. It is not the direct route (no DirectStorage / zero-copy import of the F32 pages can produce a converted tensor); the F32 arm takes the normal weight-import route (zero-copy or DirectStorage where the policy allows)
(9.11 MB resident). **Signed off by the user, 2026-11-28** (CLAUDE.md rule 3: the 4.55 MB host-staged BF16/F16 conversion of `patch_embd` / `position_embd` is accepted as a bounded one-time load cost; ruling given with the F16-activation decision, #2075).

## 12c. Activation storage precision study for image prompts (epic #2025; F16 became the default in #2075, see the decision below)

Question (user, 2026-11-27): BF16 activation storage costs image prompts about 2-3e-3 KL (0.5e-3 on text); would F16, or a targeted F32, do better, and at what cost? The study itself changed no default (the arms were opt-in);
the decision it led to is recorded after the recommendation. Hardware: Strix Halo iGPU, NVFP4 target, F16 KV, 128-token chunks, same oracles as 12a. Raw log: `.docs/ink-prec-results.txt`.

**Arms** (all accumulate in F32; only the stored scratch format differs): `DOTLLM_INKLING2_ACT=bf16|f16|f32` (text model; `_f16act` / `_f32act` shaders, `ACT_F16` in `olala_common.glsl`: RNE, overflow to inf),
`DOTLLM_INKLING2_ACT_F32_IMAGE_CHUNKS=1` (a chunk that carries image rows runs on a second, F32 scratch set; text chunks and decode stay BF16; nothing activation-typed is carried between chunks),
`DOTLLM_VISION_ACT=bf16|f16|f32` and `DOTLLM_VISION_ROPE_F32=1` (ViT; study shaders built by `native/vulkan/build_study_arms.sh`). Per-chunk switch proven: rainbow80 single-pass last-row logits are bit-identical to the F32 arm.
The BF16 `.spv` are byte-identical to before (checked with `cmp`).

**Metric.** The oracle KL (13 rows) swings by a few 1e-3 on a single input rounding (12a), as much as the arms differ, and the llama.cpp oracle carries its own Q8_0 mmproj noise. The ranking metric is
NF13: mean KL of the arm against this engine's F32 arm over the same 13 rows (no oracle noise). Oracle KL is kept as the absolute gate.

| text model arm | oracle KL x1e-3 80 / 190x110 / 640 | NF13 x1e-3 80 / 190x110 / 640 (mean) | text passage: oracle KL, vs-F32 (127 rows), top-1, greedy mism (confident) | scratch (chunk 128, H 2304) |
|---|---|---|---|---|
| BF16 (shipped) | 4.85 / 6.62 / 5.41 | 2.04 / 6.13 / 4.53 (4.23) | 3.61, 2.11, 251/254, 3/384 (0/345) | 24.8 MB |
| F16 | 2.53 / 2.31 / 6.11 | 0.04 / 0.91 / 2.80 (1.25) | 3.47, 1.64, 250/254, 6/384 (0/345) | 24.8 MB |
| F32 | 2.72 / 3.84 / 3.49 | 0 (reference) | 3.12, 0, 251/254, 5/384 (0/345) | 49.6 MB |
| BF16 + F32 image chunks | 2.52 / 2.94 / 3.30 | 0.67 / 1.64 / 0.15 (0.82) | = BF16 (no image rows) | 24.8 + 49.6 = 74.4 MB (second set allocated on the first image prefill) |

Image-chunk F32 is not exactly F32 (NF13 0.82, not 0) because the decode rows and text-only chunks after the image still run BF16. All image gates pass 13/13 top-1, 0 greedy mismatches in every arm.
Stored state: none besides scratch (KV stays F16, conv state BF16). Model total 17,808.7 MB (BF16 / F16), 17,833.5 MB (F32), 17,858.3 MB (BF16 + F32 image chunks).

**Overflow.** Max |activation| per store point (CPU F32 trace of the same graph, 3 images + the 256-token passage; `Inkling2ActSim` stats): residual stream (`layer_out`) 453 (image 640; passage 370), `ffnInp` 386, block copies 36.6,
embedding 37.7, MoE output 108, post-norm 170, `V` after conv 269, SITU 62, everything else <= 51. Margin to F16's 65,504: 145x on the residual stream. F16 overflow count 0 in every point; non-zero values below the F16 normal range:
<= 0.53% (attention output), 0.15% (SITU), <= 0.02% elsewhere; flushed to zero <= 0.004%. The F16 GPU arm produced no non-finite value on any prompt. Olala-style 1e9..1e14 ranges are absent here. Evidence is 4 prompts: a long-context / other-content sweep is advisable before a default flip.

**Vision tower** (Vulkan ViT, Q8_0 weights, arms on the same encoder; reference = CPU encoder on the F32-weights control, cached by the ViT tests; the 4,096-token noise image has no control, so it is vs the CPU shipping encoder):

| ViT arm | relRMS 80 | 190x110 | 640 | 4096-token | encode s 80 / 190x110 / 640 / 4096 | peak scratch MB 80 / 190x110 / 640 / 4096 |
|---|---|---|---|---|---|---|
| BF16 (shipped) | 1.73e-2 | 1.81e-2 | 4.11e-2 | 1.13e-2 | 0.024 / 0.027 / 0.240 / 33.3-33.6 | 0.4 / 1.5 / 24.8 / 395.9 |
| F16 | 1.66e-3 | 2.31e-3 | 5.22e-3 | 7.35e-3 | 0.028 / 0.030 / 0.251 / 33.6 | 0.4 / 1.5 / 24.8 / 395.9 |
| F32 | 5.2e-4 | 5.6e-4 | 8.2e-4 | 7.27e-3 | 0.019 / 0.021 / 0.245 / 34.0-34.8 | 0.8 / 2.7 / 45.8 / 731.4 |

(Encode times are medians of 5 interleaved rounds after a warm-up round, 2 for 4,096 tokens; the small images are within timer noise of each other. vs llama.cpp's F32 control, 80 and 190x110: BF16 1.73e-2 / 1.81e-2, F16 1.65e-3 / 2.30e-3, F32 4.5e-4 / 5.1e-4.)
Max |activation| per ViT stage (F32 trace): residual `layer_out` 49.2 (4,096-token noise 44.8), embeddings 32.5, `hid` (GELU) 42.4, pre_ln 14.8 (34.4 on noise), Q 12.5, K 18.3, V 10.0, attention 10.0: 1,300x margin to F16.
End-to-end oracle KL through the shipped text model, 80 / 190x110 / 640 (x1e-3): BF16 3.67 / 3.22 / 10.40, F16 2.63 / 3.23 / 11.38, F32 2.61 / 3.14 / 13.15. These do NOT improve with ViT precision: the llama.cpp oracle embeddings are
themselves Q8_0-mmproj-noisy (2% vs the F32 control), so oracle KL cannot rank ViT arms; rank them by relRMS against the F32 control.
**RoPE table.** An F32 instead of F16 table changes relRMS by under 3% for BF16 and F16 activations (BF16 1.73 -> 1.74, 1.81 -> 1.69, 4.11 -> 4.08 e-2; F16 1.66 -> 1.69, 2.31 -> 2.32, 5.22 -> 5.17 e-3) and only shows at the F32 floor
(5.2 / 5.6 / 8.2e-4 -> 5.0 / 5.1 / 5.1e-4): the F16 table is not part of the BF16 gap; the BF16 activation storage is. (The BF16 study GEMM is byte-identical to the shipped `vit_gemm_q8.spv`, so this isolates the table alone.)

**Performance** (same-session interleaved, pairwise bf16 vs arm because every model instance loads its own 17.5 GB via DirectStorage; 5 rounds after a warm-up round, direction alternated; medians, ranges in the log):
prefill of 300 text tokens 68-71 tok/s in every arm (range 39-87), decode 19.7-21.5 tok/s in every arm (range 12-29), the 640-image prompt (284 tokens) prefill 3.8-4.4 s in every arm (range 3.3-6.5). The box was noisy (2x
swings between rounds); no arm is separable from BF16 on speed. TTFT for an image prompt = encode (0.24-0.25 s for 640 on Vulkan) + prefill (about 3.8 s); the arms move neither.

**CPU simulation of mixed placements (options a and c): NOT validated, no finding.** `Inkling2ActSim` (a hook in the CPU reference, not in the shipped tree, see the decision below) rounds the CPU reference at the Vulkan store points (residual stream only in F32; F32 for the first N layer visits; image rows
in F32). Its BF16 / F16 numbers disagree with the real GPU arms (NF13 BF16 5.7 / 3.9 / 3.6 vs GPU 2.0 / 6.1 / 4.5; text vs-F32 3.9 vs 2.1; F16 text 3.4 vs GPU 1.6), because the CPU NVFP4 / Q8_0 GEMMs quantise their activation input to Q8_0
(`GemmNvfp4(..., preQuantizedInput)`), a noise the GPU kernels do not have, and any sub-ULP change re-rolls those 8-bit digits. The simulation's noise floor (3-4e-3) swamps the 0.5-3e-3 effects, so its residual-only and first-N
rows (in the log) must not be used. A real GPU arm for those would need mixed-format shaders (the depth-mix kernel reads blocks and the current state together): not built.

**Recommendation.**
1. F16 beats BF16 for image prompts at equal bytes and equal speed: NF13 4.23 -> 1.25e-3 (3.4x) on the text model, text passage vs-F32 2.11 -> 1.64e-3, ViT embedding error 10x lower (relRMS 1.7e-2 -> 1.7e-3). Oracle KL on the text model:
   2.53 / 2.31 / 6.11 vs 4.85 / 6.62 / 5.41 (the 640 oracle number is noise-level worse; NF13 is better on all three). It is overflow-safe on everything measured (145x margin on the LLM residual, 1,300x on the ViT). F16 is within precision rule 1 (FP16/BF16 widest default). Switching is the user's decision; before a default flip, sweep long-context prompts and the DSpark drafter taps for overflow.
2. F32 is not "far higher" than F16 here (F32 oracle 2.7 / 3.8 / 3.5 vs F16 2.5 / 2.3 / 6.1, within the 13-row noise), so with F16 no F32 exception is needed. If BF16 stays: the targeted option is F32 for image-carrying prefill chunks: oracle KL 2.52 / 2.94 / 3.30 (the F32 level), NF13 0.82e-3, cost one extra 49.6 MB scratch set (0.28% of the 17.8 GB resident, allocated on the first image prefill only, no stored state, no measurable speed cost); only the image prefill pays it, decode and text stay BF16. The ViT keeps its own choice: F16 ViT activations cost nothing extra (396 MB at 4,096 tokens) while F32 doubles it to 731 MB for no end-to-end gain.

**Decision (user, 2026-11-28, #2075): F16 is the default activation storage format of the Inkling-2 Vulkan text model and the Vulkan ViT**, after the safety gate below. BF16 stays selectable:
`DOTLLM_INKLING2_ACT=bf16` (text model), `DOTLLM_VISION_ACT=bf16` (ViT); `f32` remains the measurement arm (CLAUDE.md precision rule 1). The rule-3 sign-off for the 4.55 MB host-staged
`patch_embd` / `position_embd` conversion is recorded in 12b. The study-only paths stay under their env flags (`DOTLLM_INKLING2_ACT_F32_IMAGE_CHUNKS`, `DOTLLM_VISION_ROPE_F32`, the `_f32act` shaders); the CPU
storage-format simulation (`Inkling2ActSim`, a hook in the CPU reference) is not shipped, it lives in the history of branch `study/inkling2-act-precision` (commit `ba695dc6` and its parents).

**Gate 1, the rounding routine (`Inkling2F16RoundingTests`).** `fToAct` / `actToF` of `olala_common.glsl` (`ACT_F16`) against .NET `Half` over 2,754,638 inputs: 63,587 exact RNE ties (every midpoint of the half grid, both signs,
plus the 65,520 overflow tie, and the F32 ULPs either side), 1,350,581 in the subnormal range, 656,620 at or past the overflow boundary, 5,829 NaNs, infinities, zeros, F32 denormals, every exponent, and 2.5 M random F32 bit
patterns. 0 store mismatches, 0 decode mismatches. Five mutants (shader copies loaded through `DOTLLM_INKLING2_SPV_OVERRIDE`) all die: truncation instead of RNE (441,987 mismatches, first at 0x33000001 = 2.98e-8: 0x0000 vs 0x0001),
ties rounded up (30,785), subnormal ties rounded up (1,030), the overflow branch moved from 65,520 to 0x47A00000 (1,452), NaN mapped to the largest finite half (5,829). (Moving the branch only to 0x47800000 is an equivalent
mutant: the rounding carry already yields infinity for 65,520..65,535.) The BF16 twin passes its own definition.

**Gate 3, the guard.** *Text model:* the logits of every chunk that returns them are checked after the download (`TensorPrimitives.SumOfMagnitudes`, vectorised, NaN / +-inf propagate): non-finite throws
`InvalidOperationException` naming `DOTLLM_INKLING2_ACT=bf16`. It fails loudly instead of falling back because a chunk has already advanced the K/V ring and the conv state, so it cannot be re-run in BF16.
An overflowing F16 store makes the hidden state NaN within the chunk and K/V, conv state and attention carry it to every later chunk, so the last chunk of a prefill and every decode step expose it. Cost: 58.5 to 85.9 us per
200,058-float logits row (measured, two runs), about 0.2% of a ~50 ms decode step, once per prefill. *ViT:* the encoder is stateless, so a non-finite F16 embedding re-encodes that image in BF16 (`Bf16Fallbacks` counts them, 0 in every
measured run); the result then equals a plain BF16 encode (tested). Cost: one pass over `[n/4][2304]` halves per encode (2.4 M at 4,096 tokens, a few ms against 33 s; not separately measured). BF16 has no guard.
Tests: `Guard_NonFiniteLogits_ThrowLoudlyOnF16_AndCostIsNegligible`, `Guard_Vit_NonFiniteF16_RecomputesInBf16`.

**Gate 2, overflow sweep** (F16 GPU arm: every logit of every row of every 128-token chunk finite; `Inkling2F16GateTests`, opt-in `DOTLLM_INK_F16_SWEEP=1`):

| prompt | rows x logits checked | non-finite | max abs logit (cap 30) |
|---|---|---|---|
| wikitext text, 16,640 tokens (global K/V 1,262 MB, 271 s) | 16,640 x 200,058 | 0 | 29.95 |
| 5 images (4 + 15 + 256 + 15 + 4 image tokens), 434 tokens | 434 x 200,058 | 0 | 29.48 |
| 3 images + passage + greedy gates (12c above, 4 prompts) | | 0 | |

Max |activation| per store point (CPU F32 trace of the same graph; margin = 65,504 / max):

| store point | max \|x\| (3 images + passage) | margin to 65,504 | non-zero values below the F16 normal range | F16 overflows |
|---|---|---|---|---|
| `layer_out` | 452.9 | 145x | 0.011% | 0 |
| `ffnInp` | 385.9 | 170x | 0.021% | 0 |
| `vc` | 269.4 | 243x | 0.008% | 0 |
| `norm` | 169.8 | 386x | 0.010% | 0 |
| `moe` | 107.8 | 608x | 0.005% | 0 |
| `situ` | 62.4 | 1,050x | 0.150% | 0 |
| `kc` | 50.78 | 1,290x | 0.003% | 0 |
| `embed` | 37.71 | 1,737x | 0.019% | 0 |
| `ctrl_in` | 36.78 | 1,781x | 0.011% | 0 |
| `msg` | 36.78 | 1,781x | 0.012% | 0 |
| `blk` | 36.64 | 1,788x | 0.007% | 0 |
| `mix` | 34.89 | 1,877x | 0.011% | 0 |
| `vraw` | 33.22 | 1,972x | 0.007% | 0 |
| `vdyn` | 26.17 | 2,503x | 0.004% | 0 |
| `rel` | 24.51 | 2,673x | 0.004% | 0 |
| `q_clip` | 22.04 | 2,973x | 0.008% | 0 |
| `ctrl_gain` | 20.15 | 3,251x | 0.003% | 0 |
| `ctrl_forget` | 18.4 | 3,559x | 0.004% | 0 |
| `gateB` | 18.14 | 3,611x | 0.002% | 0 |
| `sink` | 18.1 | 3,619x | 0.000% | 0 |
| `kraw` | 14 | 4,679x | 0.005% | 0 |
| `kdyn` | 13.22 | 4,953x | 0.006% | 0 |
| `q` | 12.44 | 5,264x | 0.007% | 0 |
| `ctrl_g` | 10.84 | 6,044x | 0.008% | 0 |
| `attn` | 10.3 | 6,358x | 0.531% | 0 |
| `rel_clip` | 7.985 | 8,204x | 0.016% | 0 |
| `selw` | 7.862 | 8,331x | 0.000% | 0 |
| `attn_o` | 3.223 | 20,322x | 0.045% | 0 |

Long-context and multi-image census (CPU F32 trace of the same graph, 434-token 5-image prompt and a 16,640-token wikitext prompt; stats only, 2,481 s on the CPU reference), the same store points:

| store point | 5 images (434 tok) | 16,640-token text | max of the two | margin to 65,504 | below-normal share | overflows |
|---|---|---|---|---|---|---|
| `layer_out` | 456.4 | 473.2 | 473.2 | 138x | 0.011% | 0 |
| `ffnInp` | 388.4 | 298.7 | 388.4 | 169x | 0.020% | 0 |
| `vc` | 214.1 | 102.5 | 214.1 | 306x | 0.008% | 0 |
| `norm` | 171.4 | 168.8 | 171.4 | 382x | 0.010% | 0 |
| `moe` | 107.8 | 133.2 | 133.2 | 492x | 0.006% | 0 |
| `kc` | 50.32 | 62.96 | 62.96 | 1,040x | 0.003% | 0 |
| `situ` | 62.4 | 62.28 | 62.4 | 1,050x | 0.152% | 0 |
| `embed` | 37.71 | 38.01 | 38.01 | 1,723x | 0.001% | 0 |
| `ctrl_in` | 36.74 | 37.3 | 37.3 | 1,756x | 0.011% | 0 |
| `msg` | 36.74 | 37.3 | 37.3 | 1,756x | 0.012% | 0 |
| `blk` | 36.45 | 35.66 | 36.45 | 1,797x | 0.007% | 0 |
| `mix` | 34.65 | 36.04 | 36.04 | 1,818x | 0.011% | 0 |
| `vraw` | 28.11 | 20.17 | 28.11 | 2,330x | 0.007% | 0 |
| `rel` | 24.51 | 15.7 | 24.51 | 2,673x | 0.004% | 0 |
| `q_clip` | 22.05 | 22.97 | 22.97 | 2,852x | 0.008% | 0 |
| `vdyn` | 21.88 | 20.11 | 21.88 | 2,994x | 0.004% | 0 |
| `ctrl_gain` | 20.15 | 20.04 | 20.15 | 3,251x | 0.002% | 0 |
| `ctrl_forget` | 18.4 | 17.65 | 18.4 | 3,559x | 0.004% | 0 |
| `gateB` | 17.87 | 18.14 | 18.14 | 3,611x | 0.003% | 0 |
| `sink` | 16.77 | 17.79 | 17.79 | 3,683x | 0.001% | 0 |
| `kraw` | 13.17 | 15.68 | 15.68 | 4,177x | 0.005% | 0 |
| `kdyn` | 13.3 | 14.02 | 14.02 | 4,674x | 0.005% | 0 |
| `q` | 13.06 | 13 | 13.06 | 5,016x | 0.006% | 0 |
| `ctrl_g` | 10.16 | 11.38 | 11.38 | 5,755x | 0.008% | 0 |
| `attn` | 10.3 | 10.17 | 10.3 | 6,358x | 0.149% | 0 |
| `rel_clip` | 7.984 | 8.54 | 8.54 | 7,670x | 0.016% | 0 |
| `selw` | 7.831 | 7.984 | 7.984 | 8,204x | 0.000% | 0 |
| `attn_o` | 3.046 | 4.295 | 4.295 | 15,252x | 0.041% | 0 |

The residual-stream maxima per layer visit are in the raw log. The largest value anywhere is 473 (`layer_out`, 16K text): margin 138x. **DSpark tap rows** (the `IInkling2Tap` rows the drafter stores as 16-bit activations, max |value| per visit): visit 5: 345 (images) / 275 (16K text); visit 23: 199 / 174; visit 41: 91 / 92; visit 48 (final normed output): 171 / 169: the largest is 345, 190x below F16's ceiling. The drafter's own stored activations (a 3-layer Qwen3-like block fed by RMS-normalised mixes of those taps) were not instrumented; an F16 overflow there could only produce bad drafts (lower acceptance), never a wrong token, because the target verifies every draft, so the drafter is deliberately not guarded; the measured draft agreement of 96.0% (below) shows no degradation.


**Regression runs with F16 as the default, on the final head** (dev merged in, #2065's conv history rounding included; Strix, functional runs):
- Op-level kernel tests 37/37 (`Conv_CarriedState_MatchesReference` passes now); ViT encoder suite 15 passed, 3 skipped; **vision suite 28/28, including every `ChunkedPrefill_*` chunk-invariance case (KL vs the 32-token run exactly 0)**, without any override shader;
  Vulkan row invariance + DSpark threshold tests 18/18; Vulkan DSpark model tests 12/12; Vulkan KL gates 9/9 (below).
- Vulkan KL gates (mean KL x1e-3, 32-token calls, vs llama.cpp `-fa off`): passage F16 KV 5.338, q8_0 KV 3.450 (the earlier pre-merge read was 3.469 / 4.944), wikitext c1280 F16 KV 1.789 (position < 1024 1.519, >= 1024 2.195), q8_0 1.924, F32-act arm 1.777; greedy teacher-forced 6/384 with 0/345 at oracle margin >= 0.3 for both KV formats. The passage movement is NOT noise: see "Same-head A/B" below.
- Rounding test 2/2 and 5/5 mutants dead, guard tests 3/3 (the two sweeps are opt-in), unchanged.
- Known unrelated failure: `Probe533VulkanKvBisectTests` (a Llama-3.2-1B probe).

**The DSpark drafter on F16 activations (#2067, ported in #2075).** The drafter's own shaders read and write activation words (taps, ctx-mix input, the block rows, Q/attention output, the Markov pre-activations); they were hard-wired to BF16 while the shared helpers
(`Gemv`, `Rms`, the embedding kernel) follow the target's format, so an F16 target would have mixed formats silently. Port: the seven activation-typed bodies (`inkling2_dspark_{attn,ctxmix,eltwise,headnorm,markov_pre,markov_step,markov_logits}`) read and write
`ACT_T` through `actToF` / `fToAct` (weights keep their BF16 decode), each with an `_f16act` twin (`gen_dspark_wrappers.py`; `olala_gemv_q8_0_f16out_f16act` for the F16-logits arm); `mask_embd` rows are stored in the activation format; the taps copy is
format-agnostic. The BF16 `.spv` are byte-identical except `markov_pre` / `markov_step` (the same math, different SPIR-V ids after the weight and activation decoders stopped sharing one function). **Format guard:** `DSparkDrafter.Format` records the target's base format at attach; the
tap copy, the context injection and the draft block throw `InvalidOperationException` if the target's current scratch is in another format (the opt-in F32 image-chunk set), so a mismatch can never decode garbage. The F32 measurement arm stays refused at attach.
- *Tests:* the op-level DSpark kernel tests run in both formats (`Inkling2VulkanDSparkKernelTests`, 32 BF16 tests, and its `...F16ActTests` twin, 32 F16 tests; 64/64 pass).
- *Mutants* (a BF16 shader under F16 words, loaded through `DOTLLM_INKLING2_SPV_OVERRIDE`; failed tests of the F16 twin, baseline 0): ctx-mix 2, eltwise 2, head-norm 8, attention 8, Markov pre 5, Markov step 5, Markov logits 5 all die. At model level, with all six reading BF16:
  `Drafts_MatchLlamaCppDsparkOracle` falls from 96.0% to **0.0%** top-1 agreement. (Greedy identity alone would not have caught it: the target verifies every draft, so a garbage drafter only collapses the acceptance.)
- *F16-activation drafter results* (`Inkling2VulkanDSparkTests`, F16 default): draft top-1 agreement with llama.cpp **570/594 = 96.0%** (594 drafts, whole rounds identical 185/198), F16-logits drafts equal F32-logits drafts on 594/594 tokens; greedy identity 8 prompts x 40 tokens **8/8 identical**, K = 3: 168 rounds,
  **1.89 tokens/round, acceptance 30.6%** (per prompt 1.18 to 3.08; the CPU reference measured 1.90 / 30.9%).
**Same-head A/B on the final head (BF16 / F16 / F32 activations; `Llm_Arms_Kl`, dev with #2065 merged, softcap fix included).** Oracle KL x1e-3 (llama.cpp); NF13 / "vs-F32" = KL against this engine's F32 arm (no oracle noise):

| arm | image oracle KL 80 / 190x110 / 640 | image NF13 mean (80 / 190x110 / 640) | passage (254 rows): oracle, vs-F32 | wikitext c1280 (639 rows): oracle, vs-F32 |
|---|---|---|---|---|
| F32 | 2.01 / 4.70 / 3.21 | 0 | 3.29, 0 | 1.78, 0 |
| BF16 | 4.85 / 6.62 / 5.35 | 5.03 (2.09 / 7.46 / 5.53) | 4.12, 2.26 | 1.80, 1.23 |
| **F16 (default)** | 2.04 / 4.31 / 4.32 | **0.90** (0.11 / 1.21 / 1.39) | 5.34, 4.59 | 1.79, 0.83 |

F16 is clearly better than BF16 on image prompts (NF13 5.6x lower) and on the 639-row wikitext window (vs-F32 0.83 against 1.23). **On the 254-row passage it is worse than BF16** (vs-F32 4.59 against 2.26; oracle 5.34 against 4.12), a regression from the pre-merge head
(F16 3.47, BF16 3.61). Isolation: the same F16 run with the conv in-batch history left unrounded (the pre-#2065 shader, override) gives passage 3.47 / vs-F32 1.90 and wikitext 1.73 / 0.81, so the whole passage rise is the BF16 rounding of the conv history that #2065 added (the conv state is BF16 by ruling; under BF16 activations the
rounding is a no-op, under F16 it is a real F16 -> BF16 step, and under F32 it moves that arm's own KL by +0.2e-3). It is a few tail rows (max KL 0.107 vs 0.102 for BF16; mean over 127 scored rows), absent on the 639-row window (+0.06e-3) and on the images. It does not fail any gate (< 8e-3, top-1 251/254, greedy 0 confident mismatches).
Row/chunk invariance requires the rounding, so it cannot simply be dropped; an F16 conv state is ruled out. Open decision for the owner, with the data above.

**Softcap and the guard (review finding, fixed).** `inkling2_softcap` ran before the finite check and mapped +-inf (and NaN on AMD) to +-30, so the text guard could not see a real overflow, and my guard test poisoned the logits after the softcap. The softcap now passes non-finite values through (`Softcap_PassesNonFiniteThrough_ForTheF16Guard`; the old shader fails it).
A real-overflow arm drives the model itself (`Guard_RealF16Overflow_ThrowsOnTheFirstForward_AndBf16DoesNot`: the token-embedding gain x1e6 through a test seam, F16 throws the BF16-switch error on the first forward, BF16 does not); the override mutant that stores `x*1e6` in the F16 embed kernel throws the same error. **Gate 2 restated:** the earlier "0 non-finite" sweep column was blind (the softcap hid overflow); the evidence
that holds is max |activation| (473, 138x margin) and the sweeps re-run on the fixed head: 16,640 tokens: 0 non-finite, max |logit| 29.96; 5 images (434 tokens): 0 non-finite, max |logit| 29.69; the guard now fires on a real overflow, so these zeros mean something. Guard cost 59-89 us per logit row: about 1.2 s over a 16K all-logits prefill (16,640 rows), negligible per decode step; no opt-out.
The ViT recomputes in BF16 and throws if the BF16 result is non-finite too; the drafter format guard runs before the command buffer opens.
**Drafter A/B, F16 vs BF16 activations** (same session, interleaved F16, BF16, F16, BF16; `Report_DrafterMsAndConstrainedDownload_F32VsF16`, both locks held): median ms of a K = 3 draft, F32-logits / F16-logits arm: F16 activations 11.61 / 11.88 and 10.86 / 11.46, BF16 activations 12.14 / 11.84 and 11.75 / 11.63 (no difference beyond about 0.6 ms); the constrained-round 3-row logits download is 19.1 to 19.6 ms (F32) against 6.9 to 8.1 ms (F16 logits) in every run. Quality, one run each: F16 activations 570/594 = 96.0% draft agreement with llama.cpp, 168 rounds, 1.89 tokens/round, 30.6% acceptance; BF16 activations 568/594 = 95.6%, 164 rounds, 1.93 tokens/round, 31.8% (a difference of two drafts and four rounds); greedy identity 8/8 identical at 40 tokens in both.


## 13. DSpark drafter (issue #2039)

The drafter is `unsloth/Inkling-2-Mini-GGUF` `dspark-Inkling-2-Mini-Q8_0.gguf` (504 MB, `general.architecture = dflash`, 58 tensors). `mtp.safetensors` is **not** needed: the GGUF already carries the converter's folds (`markov_w2 = unembed @ hidden_out`, softmaxed `ctx_mix`, the replicated `mask_embd`).

**Sources.** llama.cpp (master; `llama-server` b12780 supports `--spec-type draft-dspark`): `src/models/dspark.cpp` (`build_dspark_markov_head`, `graph_dspark`), `src/models/dflash.cpp` (`build_inject_ctx_mix`, `graph_base::build_layers`, the shared embedding and lm-head), `common/speculative.cpp` (`common_speculative_impl_draft_dflash` / `_draft_dspark`: what is injected, what the block holds, the `f_rnn` ring), `src/models/inkling2-loop.cpp` and `inkling2.cpp` (`res->t_layer_inp[visit]`, the target features), `conversion/inkling2.py` (`Inkling2DSparkModel`), `docs/speculative.md`. Bench card: `hf://datasets/pratikgx/inkling-2-mini-dspark-m5-pro-mac-mini-bench` (README only).

**Config** (`mtp_config`, GGUF `dflash.*`): 3 layers, hidden 2304, 24 heads / 8 KV heads, head_dim 128, intermediate 9216, RoPE NeoX theta 1e6, RMS eps 1e-6, `sliding_window = 1025`, pattern `[local, GLOBAL, local]` (layers 0 and 2 windowed, layer 1 global), `block_size = 8`, `sample_from_anchor = false`, no confidence head, `has_markov_context = true` (`rnn_hidden`), rank 256, `final_logit_softcapping = 30`.

**Target features (what is captured, and when).** `target_layers = [5, 23, 41, 48]` count **layer visits** in execution order (48 per token: prelude 0-5, then 3 passes over core 6-17, then 18-23). The feature of id `v` is the **input** of visit `v` (`t_layer_inp[visit] = cur`, taken before the visit's layer is built): visit 5 is the input of prelude layer 5; visit 23 is the input of pass-1 layer 11 (cache slot 29, not 23); visit 41 is the input of pass-2 layer 17 (cache slot 41 only by numbering coincidence for this id; tapping by slot is a mutant); id 48 is not a layer input but the final normed output (after `mix(output_res_query)` and `output_norm`, before the LM head). Each target position yields one row per tap, held as F16.

**Context injection** (`build_inject_ctx_mix`; per injected position `p`, per drafter layer `l`): `m = sum_k ctx_mix_l[k] * tap_k` (the converter already applied the softmax), `n = rms(m) * ctx_norm_l`, `K = attn_k_ctx_l . n`, `V = attn_v_ctx_l . n`. K gets a per-head RMSNorm (`attn_k_ctx_norm`) then RoPE at `p`; V gets neither. K/V are stored F16 in the layer's context cache. The Markov head's `f_rnn(p) = rms(sum_k markov_ctx_mix[k] * tap_k) * markov_ctx_norm` is kept per position.

**Block.** A round with `n` drafts decodes `n + 1` rows (not always 8; attention is non-causal, so the drafts depend on `n`): row 0 is the **anchor** (the last committed token, at position `pos0`, not yet seen by the target), rows 1..n are `mask_embd` rows (positions `pos0 + i`). Embedding: the anchor is the target's raw `token_embd` row through the drafter's own `token_embd_norm` (the target's embedding path already applies its norm, so reusing it would norm twice); mask rows are `mask_embd` as stored (no norm, no embedding scale). Per layer (Qwen3-like): `rms(x) * attn_norm`, `attn_q/k/v`, per-head RMSNorm `attn_q_norm` / `attn_k_norm`, RoPE (NeoX, position `pos0 + i`), attention (scale `1/sqrt(128)`, no sinks), `attn_output`, `+x`, `rms * ffn_norm`, SwiGLU, down, `+x`. The block's K/V pass through the F16 cache (llama.cpp writes them there before attending). **Block-uniform mask** (`block_uniform = true`): every block row uses the mask row of the anchor. Context keys are positions `< pos0`; a windowed layer (n_swa 1025) sees `pos0 - 1024 .. pos0 - 1`, the global layer sees all; every block row sees every block key (non-causal). A key at distance over 1024 from the anchor is invisible to every row, including row `n`, which a per-query window would show.

**Head.** `hidden_i = rms(x_i) * output_norm` for rows `i = 1..n`; `base_i = lm_head(hidden_i)` with the **target's** Q8_0 head over the 200,058 unpadded rows, **uncapped**. Then, with `state = 0` and `prev = anchor`, for `i = 1..n`:

```
joint = joint_proj . [ state(256) | markov_w1[prev](256) | hidden_i(2304) | f_rnn(pos0 - 1)(2304) ] + joint_proj.bias     # 5120 -> 768
gate = sigmoid(joint[0:256]);  cand = tanh(joint[256:512]);  markov = tanh(joint[512:768])
state = gate * state + (1 - gate) * cand
bias = markov_w2 . markov + markov_w2.bias                                                                              # [200058]
col_i = 30 * tanh((base_i + bias) / 30)          # softcap AFTER the bias (softcap_after_bias = true)
prev = argmax(col_i)                             # greedy; the draft token of position pos0 + i
```

`f_rnn` is the feature of the position **before** the anchor (`pos0 - 1`, the last token the target processed); without it llama.cpp does not draft. `markov_w1`, `markov_w2` and `joint_proj` are BF16 and read in place.

**Verification.** None of its own: llama.cpp (`common/speculative.cpp`) hands the `n` tokens to the ordinary speculative verifier (the target decodes `[last, d1..dn]`, greedy acceptance against the target argmax). dotLLM reuses the unmodified `MtpSpeculativeDecoder` (the drafter sits behind `IDFlashDraftAttachable` / `IMtpState`), its adaptive gate and its rollback. After a rejection the target's recurrent state must rewind: the short-conv windows of the 48 slots (`ForwardWithRecurrentSnapshots` / `RestoreRecurrentStateToRow`, section 14); the KV rows past the accepted one are simply overwritten.

**Expected acceptance and speed (card).** Unsloth `llama-server` b12649, M5 Pro (Metal), IQ4_NL, `--spec-draft-n-max 5`: a repetitive continuation prompt 92.9% acceptance, 70.0 vs 23.3 tok/s (3.0x); a short code prompt 55.5%, 48.9 vs 34.3 tok/s; the "fair check" of three distinct greedy prompts 26.8% to 55.5% acceptance and **32.5 vs 33.4 tok/s (about 1.0x)**; sampled prompts (temp 0.7) 18-34% acceptance and a loss. Greedy text matched plain decoding on only 1 of 3 fair-check prompts (llama.cpp's own batch-vs-single numerics, section 14). llama.cpp's docs use `--spec-draft-n-max 3`.

## 14. Speculative verify needs a row-count-invariant target (issue #2039)

Greedy identity "by construction" only holds if an N-row forward equals N single-row forwards. Measured on the NVFP4 model (20-token prefix, then 8 rows against 8 single-row forwards; `Inkling2RowInvarianceTests`):

- **Before:** row 0 identical; rows 1-7 differed in 200,056-200,058 of 200,058 logits, max |diff| 0.91. Cause: `Inkling2ConvState.Run` read the in-batch history (the inputs of rows `t-1..t-3`) as exact F32, while sequential decoding reads them back out of the BF16 state.
- **Fix:** the in-batch history is rounded to the state type (BF16) exactly as `Store` does; only the current input stays exact. After it the two paths are **bit-identical on all 8 rows** (the NVFP4/Q8_0 GEMMs, the MoE and the F16-KV attention were already row-count invariant). This changes chunked-prefill numerics slightly, in the direction of decoding; the slice-1 KL gate is re-reported in section 15.
- This also explains why llama.cpp's greedy DSpark output differs from its plain output on the bench card; dotLLM does not have that problem.

**Rewinding the recurrent state.** `ForwardWithRecurrentSnapshots` records, for each of the 48 x 4 conv states, the window `[old state ; the inputs]` of the verify call in the state's own storage type (BF16: lossless, the window is BF16-exact). `RestoreRecurrentStateToRow(r)` stores rows `r+1 .. r+3` of it back and sets the sequence length to `pos0 + r + 1`. Cost: `(3 + rows) x 6656 channels x 48 slots x 2 B` = 7 MB at 8 rows, transient.

## 15. DSpark results (CPU, Strix Halo, NVFP4 target + Q8_0 drafter, 2026-11-27)

**Slice-1 KL gate with the conv-history fix** (`Inkling2RealModelTests`, 254 scored positions, defaults BF16 controller + BF16 conv state, vs llama.cpp `-fa off`): mean KL **4.053e-3** (was 4.10e-3), max 1.20e-1 (was 9.0e-2), top-1 97.6% (was 98.8%, three positions of 254), teacher-forced greedy mismatches 5/384 (**0/345 at oracle margin >= 0.3**, the gate's bar; the five are at margins 0.016-0.108). The mean did not move (the in-batch rounding makes chunked prefill agree with decoding instead of the unrounded batch path); max and top-1 move by single positions, inside llama.cpp's own flash-attention noise (4.05e-3 mean, 1.9e-1 max). The stored conv state is still BF16.

**Greedy identity** (`GreedySpeculative_EqualsPlainGreedy_OnEightPrompts`, the 8 oracle prompts x 40 tokens, K = 3, the unmodified `MtpSpeculativeDecoder`, row snapshots): **8/8 identical, 40/40 tokens each**. 167 rounds, 1.90 tokens/round, draft acceptance **30.9%** (per prompt 7%-75%). llama.cpp b12780 on the same prompts (K = 3, 48 tokens): 174 of 574 drafts accepted = **30.3%**.

**Draft parity vs llama.cpp** (`Drafts_MatchLlamaCppDsparkOracle`; oracle `dspark-oracle-k3.json` built from `llama-server -md dspark-... --spec-type draft-dspark --spec-draft-n-max 3 -dev none -devd none -v`; dotLLM is teacher-forced on llama.cpp's committed tokens and drafts at each round's reconstructed anchor): top-1 agreement **570/594 = 96.0%** (position 1: 98.5%), 183 of 198 whole rounds identical, mean |p diff| at position 1 0.0147 (p = softmax over the sampler's top 10). The residue is the target-feature noise floor (KL 4e-3) compounding along the Markov chain.

**Mutants** (each must change the drafts on real data; `Inkling2DSparkTests`, `Inkling2RowInvarianceTests`): killed: taps by cache slot instead of visit, taps one visit early (prelude only), taps one visit late (every loop), visit 48 before the output norm, `ctx_mix` softmaxed twice, anchor not normed, anchor normed twice, Markov state reset each step, softcap before the bias, `f_rnn` of the wrong position, layers 0/2 not windowed (`AllLayersGlobal`), per-query window instead of the anchor's (`PerQueryWindow`), skipped `RestoreRecurrentStateToRow` (restore test), unrounded in-batch conv history (row-invariance test, pre-fix). **Correction (#2046):** this section first recorded `AllLayersLocal` (the global layer 1 windowed too) as a survivor, 'bit-identical because layer 1's softmax is peaked'. That was an artefact: the drafter read `dflash.sliding_window_pattern`, the key is `dflash.attention.sliding_window_pattern`, a missing key defaulted to 'every layer windowed', so layer 1 WAS windowed on both backends and the mutant changed nothing. The key is fixed (and a missing pattern now throws): layer 1 is global, `AllLayersLocal` is **killed** (max |draft logit diff| 1.45e5 at ~1300 tokens of random context, with `AllLayersGlobal` 1.45e5 and `PerQueryWindow` 9.0e4), and the CPU numbers above did not move (identity 8/8, 30.9% acceptance, oracle 570/594 = 96.0%): the prompts are shorter than the window, so layer 1 never saw a key past 1024 positions.

**Allocation:** a warm draft (K = 7) allocates 0 managed bytes (`DraftHotPath_AllocatesNoManagedTensorMemory`); the drafter's activations and logits are native scratch.

**CPU speed (net slower; the gate keeps it off).** `dotllm run --device cpu -t 0 -n 64`, one process per run (Release build, `cpu-bench` lock, a fresh lock per run), decode tok/s:

| context | off (`--no-mtp`) | auto (default) | forced (`DOTLLM_MTP_ADAPTIVE=0`) | forced / off | tokens/round | drafter ms | verify ms (K+1 = 4 rows) |
|---|---|---|---|---|---|---|---|
| short (28 prompt tokens) | 9.79 | 9.61 (plain) | 6.32 / 6.55 | 0.65-0.67x | 2.81 | 27 | 362 |
| long (8.3 KB of markdown, about 2-3K tokens) | 5.36 | 5.35 (plain) | 3.17 | 0.59x | 1.88 | 28 | 615 |

The drafter is cheap (27 ms); the verify is not: a 4-row verify of a 128-expert top-8 MoE costs 3.5x a single decode on this bandwidth-bound box (362 ms vs about 102 ms), so even 2.8 tokens per round loses (139 vs 102 ms/token). This is the pratikgx card's "about 1.0x" on distinct prompts, a little worse here. **Adaptive gate:** the model is not "large" by the gate's resident-size prior (`ComputeMemoryBytes` is the state, not the mmapped weights), so a single `run` decides plain, which is why auto equals off. Over six requests through `dotllm serve` with an auto-detected `dspark-*.gguf` (hard-linked beside the target) the gate tried MTP once, measured 237.7 ms/token against 95-118 ms plain, fell back after 5 rounds and stayed on plain for the following requests. Draft width: `DOTLLM_DSPARK_DRAFT` (default 3).

**Precision.** Weights Q8_0 / BF16 read in place; the context K/V is F16 (windowed since #2060, section 16: 4 KiB per position plus a fixed ring; it was 12 KiB per position for 3 layers; no F32 anywhere stored); `f_rnn` rows F16; tap rows F16; activations are native scratch, F32 only within a call (as the CPU reference). The Markov recurrent state (256 floats, 1 KiB) is an F32 in-call accumulator on the stack, never stored across calls; a BF16/8-bit evaluation of it has **not** been done (open item for sign-off). The recorded conv-snapshot window is BF16.

## 16. DSpark drafter on Vulkan (issue #2046, windowed context K/V #2060, conv fix #2065)

`VulkanInkling2TransformerModel` (partial, `VulkanInkling2TransformerModel.DSpark.cs`) implements `IInkling2DSparkHost`: the resolver, `run`, `chat` and `serve` attach the drafter to it exactly as to the CPU model (`--dspark`, or an auto-detected `dspark-*.gguf` beside the target;
an explicit path that cannot attach refuses to start, an auto-detected one is logged and ignored). `SupportsMtp` is true only while a drafter is attached; the unmodified `MtpSpeculativeDecoder`, the adaptive gate and the `MtpMaxDraftTokens` cap are reused. The CPU drafter class is **not** instantiated
on this path (its constructor allocates host F32 scratch); the GGUF validation both backends share lives in `Inkling2DSparkSpec`. `VulkanDFlashBackend` (F32 scratch) is not reused and Qwen DFlash is untouched.

**Target hooks.** Taps at layer visits 5 / 23 / 41 (the INPUT of the visit, recorded before the layer) and 48 (the final normed output) are copied into a chunk-sized BF16 buffer `[128][4][H]` inside the same command buffer as the chunk, then
injected on the device (`inkling2_dspark_ctxmix`, K/V projections through `olala_gemv_q8_0`, `inkling2_dspark_headnorm`: per-head RMSNorm + NeoX RoPE + F16 store). Row snapshots: `inkling2_conv` (spec constant `SNAP`) writes the verify window `[old state (3 rows); the T inputs]` as BF16 for all
4 convs of all 48 slots (`ForwardWithRecurrentSnapshots`); `RestoreRecurrentStateToRow(r)` (`inkling2_conv_restore`) copies rows `r+1..r+3` back, covers slots 24..47 (the pass-2/3 core slots) and sets the length to `pos0 + r + 1`. The engine rolls the KV handle back BEFORE restoring
the recurrent state, so `Inkling2KvHandle.Rollback` now accepts a position inside the last recorded verify (it marks the model "restore pending" and the next forward refuses until `RestoreRecurrentStateToRow` ran) and still throws for any other rollback.

**Row invariance.** An N-row forward equals N single-row forwards **bit for bit** on the Vulkan target (2, 4, 8 rows over a 20-token prefix, all 200,058 logits; `Inkling2VulkanRowInvarianceTests`): the default BF16 conv is invariant by construction (its inputs are BF16-exact),
and the Q8_0 / NVFP4 GEMVs already reduced in the same order for 1 and N rows. The **F32-activation reference arm was not** (#2065: the conv shader did not round in-batch history to BF16 as `Inkling2ConvState.Run` does since #2039, which also made `Inkling2VulkanKernelTests.Conv_CarriedState_MatchesReference`
fail by 2e-3 to 3e-3 on dev). Fixed in `inkling2_conv.glsl` (`s2 = bf16(x)`: a no-op for BF16 activations). Tests: `Conv_EightRows_EqualEightSingleRows_BitForBit` (dyn / no dyn) and the mutant that removes the rounding (2099 and 2100 of 2400 outputs differ).

**Drafter kernels** (all `inkling2_dspark_*`, BF16 activations, F16 context K/V, weights read in place, F32 only in registers / shared memory): `copy`, `ctxmix` (tap mix + RMS, also the F16 `f_rnn` ring), `headnorm` (per-head norm, RoPE with a hi/lo split of `theta^(-2i/128)/(2 pi)` and an fma-recovered product so the angle is exact at any position, F16 ring or linear store),
`attn` (non-causal block attention, one workgroup per KV head, online softmax over 64-key tiles, ring or linear F16 K/V + the block's own F16 K/V), `eltwise` (residual add, SwiGLU), `markov_pre` / `markov_step` / `markov_logits` / `argmax` (the Markov head). Anchor row: `inkling2_embed` with the DRAFTER's `token_embd_norm` gain; mask rows: BF16 copy of `mask_embd` (converted once at load); LM head: the target's `olala_gemv_q8_0` over rows 1..n.

**Markov state (no stored F32).** The 256-float recurrent state is never stored across dispatches. Each walk step runs one workgroup that **replays steps 1..i in shared memory** from the stored BF16 state-independent pre-activations (`markov_pre`: `joint_bias + J[:, 512:] . [hidden_i ; f_rnn]`) and the stored token ids (`tok`), K <= 7 so at most 7 x 3 x 512 MACs per thread; it emits `tanh(c2)` as BF16 for the `markov_w2` pass, which adds the base logit and the bias, applies the softcap AFTER the bias, writes the row back and reduces to a per-workgroup argmax; a second dispatch reduces the 6,252 partials to the token id the next step reads. The whole draft (embed, 3 layers, head, walk) is ONE command buffer; only the n token ids come back.
Because nothing F32 crosses a dispatch, the "evaluate 8-bit / F16 / BF16 before F32" question for the Markov state is moot: there is no stored state to narrow. The stored values are the BF16 pre-activations (`[rows][768]`, 12 KiB at 8 rows), the BF16 `tanh(c2)` (512 B) and the token ids.

**Stored-F32 inventory (for sign-off).** (1) The drafter's logits `[(block - 1) x vocab]`, **F16 by default since #2046's review (2.8 MiB; `DOTLLM_DSPARK_LOGITS=f32` is the 5.6 MiB A/B arm, see the F16 table below)**, formerly F32 = 5.6 MiB allocated (2.4 MiB used at K = 3): the same pattern as the target's own logits (both are `VulkanDevice.Allocate`, host-visible F32, written by the LM-head GEMV; the drafter's Markov pass then updates its rows in place; no device-local F32 store on a hot path) (the head output must be F32 for the softcap + argmax; fusing the head into the Markov pass would read the 490 MB Q8_0 head once PER STEP instead of once (about 2.5 ms each at 196 GB/s), so it was not built).
(2) 8 B per workgroup of argmax partials (50 KB). (3) The GGUF's F32 weight vectors (norm gains, `ctx_mix`, `markov_w2.bias` 0.8 MB, `joint_proj.bias`), uploaded verbatim, and the 512 B RoPE table. No F32 activation, KV, tap or recurrent-state storage.

**Context K/V (#2060).** Layers 0 and 2 keep a ring of `window + MaxChunk` = 1025 + 128 = 1153 positions (4.7 MB per layer for K and V, flat in the sequence length); the global layer 1 keeps the whole context in a buffer that starts at 1,024 positions and doubles, **admitted** through `VulkanKvCache.AdmitGrowth` (usable capacity minus resident bytes minus other processes). When the growth is refused the state logs once, turns unusable and drafts the anchor repeated (the verify rejects it, the gate switches speculation off); it never throws and never spills.
Steady state: **4 KiB per position** (the global layer's K+V) instead of 12 KiB, plus the 9.2 MB ring: 17 MB at 2K tokens, 399 MiB at 100K tokens (the unbounded layout: 1,171 MiB). The minimum ring is `1024 + K` (a rejected verify may have written K rows ahead of the committed end); the shipped 1153 leaves 125 rows of slack, so a ring one row below the minimum is the mutant.
The CPU drafter got the same design (`Inkling2DSparkState`: ring `window + InjectChunk` = 1089, admitted global growth against `GC.GetGCMemoryInfo().TotalAvailableMemoryBytes - WorkingSet - 512 MiB`; K/V per position 12 -> 4 KiB, plus a fixed 8.5 MB ring. The global buffer doubles from 256 positions, so the saving shows above ~1.4K tokens: 16.5 MiB resident at 1,330 tokens (15.6 MiB in the old layout, the doubled capacity is 2,048 positions), 24.5 MiB at 4,030 (47.2 MiB old). CPU gates after the key fix: greedy identity 8/8 x 40 (167 rounds, 1.90 tokens/round, 30.9%), oracle 570/594 = 96.0% (position 1 98.5%, 183/198 rounds), ring 1089 vs 8192 bit-identical at 1,330 and 4,030 tokens, a ring of 1023 changes the drafts, refused admission degrades without throwing).

**Correctness (Strix, NVFP4 target + Q8_0 drafter, 2026-11-27).**

| gate | result |
|---|---|
| greedy identity, 8 oracle prompts x 40 tokens, K = 3, unmodified `MtpSpeculativeDecoder` + row snapshots + the Vulkan KV handle | **8/8 identical, 40/40 each**; 164 rounds, 1.93 tokens/round, draft acceptance **31.8%** (per prompt 7% to 75%; CPU 30.9%, llama.cpp 30.3%) |
| draft parity vs llama.cpp `dspark-oracle-k3.json` | top-1 **568/594 = 95.6%** (position 1: 99.0%), 181 of 198 rounds identical (CPU 96.0%, 98.5%, 183) |
| snapshot restore vs sequential decoding | next-token logits **bit-identical** (rows kept 0..2 of 6) across all 48 slots; mutants skipped restore, restore to row + 1, pass-0-only restore: all killed |
| ring 1153 vs an unbounded ring (8192), drafts at ~1300 and ~4000 tokens of random context | **bit-identical** (max logit diff 0); a ring of 1023 changes them (8.8e4 / 1.2e5 in the id-weighted fingerprint) |
| global layer 1 (global since the key fix): ring/global growth 1024 -> 4096 vs a preset 4096 buffer, ~4000 tokens | bit-identical (max logit diff 0); refused admission -> unusable, anchor drafts, no exception |
| two identical prefills | bit-identical K/V, `f_rnn` ring and drafts (this found a race: rows r and r + 16 of one chunk wrote the same `f_rnn` slot concurrently; fixed by letting only the last row of each slot store) |
| Vulkan target KL gates (`Inkling2VulkanGateTests`, `Inkling2VulkanTests`) after the conv / snapshot edits | all pass; final-code run of `Inkling2VulkanDSparkTests` + `Inkling2VulkanRowInvarianceTests` + `Inkling2VulkanGateTests`: 23/23 (one failure, the unusable-state throw above, fixed and rerun) |

**Mutants** (each must change the drafts on real data or fail an op-level test; `Inkling2VulkanDSparkTests`, `Inkling2VulkanDSparkKernelTests`): killed: taps by cache slot, taps one visit early (prelude), taps one visit late, visit 48 before the output norm, Markov state reset each step (logits and the op test), softcap before the bias (logits and the op test), `f_rnn` of the wrong position, per-query window instead of the anchor's, ring one row too short
(op-level attention test on constructed data where the oldest visible key decides the output: dropping it moves the output by 10.0, a ring one slot below the minimum by > 0.5; end to end 8.8e4 / 1.2e5), skipped / row+1 / pass-0-only restore, conv history unrounded. `AllLayersLocal` (the global layer windowed too) is **killed** on Vulkan after the pattern-key fix (max |draft logit diff| 1.14e5 at ~1300 tokens; the pre-fix 'survivor' on both backends was the misspelled-key artefact, section 15). Refused global-layer admission turns the state unusable and every later `NextDraft` answers the anchor (a first version threw at the position the stopped state no longer tracked; found by the test and fixed).
Not ported: the CPU's anchor-not-normed / anchor-normed-twice / `ctx_mix`-softmaxed-twice mutants (they act inside fused kernels or at weight load; the CPU tests still gate those semantics, and the Vulkan drafts agree with llama.cpp to 95.6%).

**Byte costs** (this PR): drafter weights + scratch 478.5 MiB device (504 MB file, read in place; imported zero-copy where the policy allows), conv snapshots 7.0 MB (8 rows x 48 slots x 6,656 channels x 2 B, grown on demand, only after the first `ForwardWithRecurrentSnapshots`), tap buffer 2.4 MB, drafter logits 5.6 MiB, state K/V as above.

**Speed (Strix 8060S, NVFP4 target, `run -t 0 -n 64`, `--device vulkan`, final code incl. the pattern-key fix; GPU lock held; one warm-up run discarded, then every configuration twice, once in forward and once in reverse order, so drift shows as forward/reverse disagreement; tok/s of the decode phase).**

| context | off (`--no-mtp`) | auto | forced (`DOTLLM_MTP_ADAPTIVE=0`) | forced / off | tokens / round | drafter ms | 4-row verify ms | 1-row decode ms |
|---|---|---|---|---|---|---|---|---|
| 23 prompt tokens | 30.58 / 30.13 | 27.29 / 27.07 (gate fell back after 5 rounds: 38.0 vs 33.8-34.5 ms/token) | 19.93 / 19.64 | **0.65x** | 1.88-1.94 | 9.6-10.4 | 86.0-87.5 | 32.7 |
| 2,628 prompt tokens (7 KB of this doc) | 25.80 / 25.67 | 24.84 / 24.22 (stayed on MTP 19 rounds, then fell back: 42.5 vs 37.9 ms/token) | 24.54 / 25.45 | **0.97x** | **3.06** | 12.5-16.2 | 97.9-98.1 | 38.9 |

**What moved with the pattern-key fix (layer 1 now global).** Correctness did not: identity 8/8 x 40 with the same 164 rounds / 1.93 tokens per round / 31.8%, oracle parity 568/594 = 95.6% (position 1 99.0%), the same, because those prompts are shorter than the window. At 2.6K tokens of context the
drafts got better: 3.06 tokens per round against 2.69 with layer 1 windowed, and forced / off went from 0.86x (22.60 vs 26.23 tok/s) to 0.97x. Short context is unchanged (0.65x against 0.66x). The earlier table (layer 1 windowed, two sessions: forced 0.57x then 0.66x short, 0.89x then 0.86x long) is superseded.

**The drafter is net slower or at best break-even on Vulkan, and the gate keeps it off** (CPU: 0.6x). The verify is the cost: a 4-row forward of a 128-expert top-8 MoE reads about 3x the expert weights of a single row, 86-98 ms against 33-39 ms for one decode: **2.6x**. Break-even needs
`(drafter + verify) / plain ms` tokens per round: (10 + 87) / 32.7 = **3.0** at short context and (14 + 98) / 38.9 = **2.9** at long context, against 1.9 measured on distinct prompts and 3.06 on a prompt that echoes its own text (that is the 0.97x: a round of 112-116 ms for 3.06 tokens is 36.6-37.9 ms per token against 38.9 plain, within the 3% noise of the wall-clock figure). A repetitive continuation (92.9% acceptance on the pratikgx card) is where it wins. The levers, in order: the small-row MoE verify (two rows should cost about 1.1x), then the drafter (10-16 ms against about 6 ms of weight traffic: about 1.1 GB per round at 190 GB/s; the walk's single-workgroup step kernel and the serial argmax levels are the visible waste).
The prefill cost of the drafter's injection is about 1 s on 2,628 tokens (3%).
**Context threshold (James, 2026-11-28).** The drafter never won below ~8K tokens, so the Vulkan model no longer attaches an auto-detected drafter at load: `DeferAutoAttach` only remembers the path, and each request asks `IContextGatedMtp.PrepareMtpFor(prompt tokens + max tokens)`; at or above `DOTLLM_DSPARK_MIN_CONTEXT` (default **12288**, `0` = always, `off` = never) the drafter is attached (about 0.5 GB, plus the adaptive gate's calibration rounds), below it nothing is allocated and the request decodes plainly. An explicit `--dspark` attaches at load and forces it on at every length; the adaptive gate stays active once attached.
Measured on the Strix iGPU (`run -t 0 -n 64`, one warm-up discarded, each context off/forced/forced/off = forward then reverse, GPU lock held; decode tok/s of the 64 generated tokens after a prefill of the stated size):

| prompt tokens | off | forced (`DOTLLM_MTP_ADAPTIVE=0`) | forced / off | tokens/round | drafter ms | 4-row verify ms |
|---|---|---|---|---|---|---|
| 4,039 | 25.76 / 25.45 | 16.69 / 17.66 | **0.67x** | 1.9-2.0 | 12.9-25.3 | 97-102 |
| 8,021 | 21.81 / 17.61 | 19.17 / 16.35 | **0.90x** (0.88 fwd, 0.93 rev) | 2.62 | 16.5-19.6 | 105-119 |
| 11,903 | 23.16 (+ one run disturbed by another job, see below) | 23.98 / 23.21 | **~1.0x** (1.04 / 1.00 against the one clean off run) | 2.94 | 17.4-19.4 | 109-113 |
| 15,771 | 13.49 / 12.94 | 19.08 / 18.96 | **1.44x** | 3.25 | 25.0-26.1 | 136-141 |

The 8K pair drifted between its forward and reverse halves (every reverse number is 10-20% lower, prefill included), so read it as 0.9x. 12K was measured later in a window where another agent's CPU job (`cpu-bench`, #2072) started halfway: the three runs before it are clean (off 23.16, forced 23.98 and 23.21 tok/s), the fourth (off, reverse) and the two default-threshold runs after it were disturbed (13.4 tok/s plain, 12.9 at 4K: prefill 57 instead of 85 tok/s) and are NOT used. So 12K is parity, not a loss, and the default stays **12288** (the gate stays active, so a parity-to-small-win region costs only its few calibration rounds). Plain decode falls from 25.6 to 13.2 tok/s between 4K and 16K (global attention grows with the context) while the verify grows less, which is where the drafter starts to pay: 0.67x at 4K, 0.90x at 8K, ~1.0x at 12K, 1.44x at 16K. The default-threshold CLI path itself was verified functionally in the disturbed window (timings discarded): at 4,039 tokens the log shows the drafter found and no attach and a plain run; at 15,835 tokens it attaches ("threshold 12288"), calibrates and the gate takes over. Not measured: contexts above 16K.

**F16 drafter logits (James 2026-11-28, instead of F32).** The LM-head GEMV writes F16 (`olala_gemv_q8_0_f16out`, an `OUT_F16` define in `olala_gemv.glsl`; the existing .spv are byte-identical) and `inkling2_dspark_markov_logits_f16` reads the F16 base logit, adds the bias and the softcap in F32 registers, writes the capped column back as F16 and takes the argmax on the unrounded F32 value. The target model's logits are unchanged.

| measure | F32 | F16 |
|---|---|---|
| logits buffer (block 8, vocab 200,058) | 5,601,624 B | 2,800,812 B |
| draft tokens on the 198 llama.cpp oracle rounds (594), F16 vs F32 | | **594/594 identical** (198/198 rounds) |
| top-1 vs llama.cpp on those rounds | 568/594 = 95.6% | 568/594 = 95.6% |
| argmax per drafted row on 177 probe rows (the short "history" prompt, every position) | | 177/177 identical; max column error 2.6e-2 (one F16 step at 16-32), mean 6.1e-3 |
| greedy identity, 8 prompts x 40, K = 3 | 8/8, 164 rounds, 1.933 tokens/round, 31.8% | 8/8, 164 rounds, 1.933 tokens/round, 31.8% |
| drafter wall time per K = 3 draft (median of 90, ~1.3K context, interleaved) | 11.55 ms | 11.55 ms |
| constrained-round download of 3 logits rows (host float[] incl. the F16 -> F32 conversion) | 19.0 ms | **6.7 ms** |
| op level, `MarkovHead_F16Logits_MatchesReferenceWalk` (n = 3, 7) | | worst normalised error 8.6e-4 (F16 storage), chain equals the double reference |

The Markov head updates the logits in place, but each row is read once (the walk depends on the argmax chain only, never on a stored column), so the F16 rounding of the base logit can only change a token through a near-tie; there were none in 594 + 177 rows. F16 is therefore the default. **Can the parity test fail?** Two coarser stores were substituted into the base-logit read of `inkling2_dspark_markov_logits_f16` (shader copies loaded through `DOTLLM_INKLING2_SPV_OVERRIDE`, tracked `.spv` untouched): (a) a 4-bit mantissa: oracle drafts 568/594 equal to F32 (92.6% vs llama.cpp), probes 172/177, max column error 10.9: both tests fail, also with the original thresholds; (b) a BF16 truncation (7-bit mantissa, steps of 0.06 at |x| 16-32): oracle drafts 589/594 equal, probes 176/177, max column error 7.2, mean 2.4e-2, greedy identity still 8/8. **The first version of the tests passed (b)** (bounds of 'agreement > 0.99'), which is a finding: argmax agreement on 594 + 177 rows is too coarse to tell F16 from BF16-truncated. The tests now require 594/594 equal draft tokens and a column error bound (max < 0.05, mean < 0.01; F16 measures 2.6e-2 and 6.1e-3), and both fail (b). Not evaluated as arms of their own: BF16 logits and 8-bit logits.
