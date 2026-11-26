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
| loop controllers (`loop_msg_query`, `ctrl_down/gain/forget`) | **F32, 113.7 MB** | `Inkling2Options.LoopController`: F32 in place, or a load-time Q8_0 / F16 / BF16 copy (30.2 / 56.8 / 56.8 MB) — see results, **user ruling pending** |
| KV history | — | F16 (llama.cpp default KV type); K is L2-clipped to 22.6 and V RMS-normed, so the range is safe |
| conv state | — | `Inkling2Options.ConvState` F32 / F16 / BF16 (3.8 MB at F32 for all 48 slots) |
| activations | — | F32 transient scratch inside a forward call (the CPU kernels take F32 activations); nothing wider than the GGUF tensors is stored |

**Oracle.** llama.cpp b12780 CPU (`-dev none -ngl 0`; llama.cpp's Vulkan backend is not an oracle), model `Inkling-2-Mini-NVFP4.gguf`. Gate data in `~/.dotllm/test-cache/token-oracles/inkling2/`:
`passage.txt` (866 tokens), `nvfp4-passage-c256*.kld` (`llama-perplexity --kl-divergence-base`, `-c 256`, 2 chunks, 254 scored positions, flash attention on/off),
`greedy-nvfp4*.json` (8 prompts x 48 greedy tokens with the oracle's own token ids and top-1/top-2 margins). **dotLLM is fed the oracle's token ids**, so tokenizer differences cannot contaminate the model KL.
The KL file keeps log-probs as uint16 over `[max-16, max]`; floor ~1e-6.

**Noise floor.** llama.cpp's own flash-attention on/off runs differ from each other by KL 4.0e-3 (the FA path rounds `bias + mask` to F16) and diverge in greedy decoding on 2 of 8 prompts, so
neither a KL below ~4e-3 nor free-running greedy identity is a meaningful bar against a single llama.cpp configuration. The gate therefore uses (a) mean KL against the F32-mask (`-fa off`) oracle,
(b) top-1 agreement, and (c) **teacher-forced** greedy agreement, counting only steps where the oracle's top-1/top-2 margin is >= 0.3 nats (free-running prefix identity is reported, not gated).

## 9. Not in this slice

- **Vision** (`mmproj-Inkling-2-Mini-Q8_0.gguf`, llama.cpp `tools/mtmd/models/inkling2v.cpp`): ViT of 23 layers, hidden 1024, 16 heads, patch 20, temporal patch 2, axial RoPE theta 1e4, exact GELU, merge 2, projector width 4096;
  images enter the text stream as `<|content_image|><|unused_200054|>` placeholders whose embeddings are injected *without* `token_embd_norm` (llama.cpp: "do not normalize raw embeddings input"). The loader must accept an embedding-input forward.
- **DSpark drafter** (`dspark-Inkling-2-Mini-Q8_0.gguf`, `arch = dflash`): 3 layers (intermediate 9216, local layers 0 and 2), block size 8, a Markov head (`markov_rank 256`, `rnn_hidden`, a folded `markov_w2 = unembed @ hidden_out`),
  `target_layer_ids = [5, 23, 41, 48]` counted in **layer visits** (the target exposes the input of visit `n` and the final norm output as id 48: `res->t_layer_inp[visit]` in `inkling2-loop.cpp`). The CPU model keeps the trace hook points
  (`pass{n}.l{ip}`, `result_norm`) where those features would be tapped.
- Vulkan / CUDA text decoders (NVFP4 expert kernels must read the blocks in place), optimisation, llama.cpp Vulkan comparison.

## 10. Results (CPU reference vs llama.cpp b12780 CPU, `Inkling-2-Mini-NVFP4.gguf`, Strix Halo, 2026-11-26)

Gate (`Inkling2RealModelTests`, 254 scored positions; dotLLM sub-chunks the prefill by 32 tokens, so carried KV/conv state is exercised; chunk size does not change the KL with F32 state):

| arm | mean KL | max KL | top-1 | teacher-forced greedy mismatches (of 384) |
|---|---|---|---|---|
| dotLLM vs llama.cpp `-fa off` (the F32-mask oracle; the gate) | **3.96e-3** | 6.6e-2 | 98.8% | 6 (0 of 345 at oracle margin >= 0.3; the six are at margins 0.015-0.141) |
| dotLLM vs llama.cpp default (flash attention on) | 5.39e-3 | 3.1e-1 | 98.0% | — |
| llama.cpp `-fa on` vs `-fa off` (its own noise floor) | 4.05e-3 | 1.9e-1 | 99.2% | greedy diverges on 2 of 8 prompts |

Free-running greedy prefix identity vs the `-fa off` oracle (of 48 tokens, 8 prompts): 10 / 3 / 48 / 48 / 48 / 27 / 12 / 48; every divergence is at an oracle top-1/top-2 margin below 0.15.

**Mutants** (same gate, unmutated mean KL 3.96e-3; all killed except one that cannot be observed on this passage):

| mutant | mean KL | top-1 |
|---|---|---|
| 2 loop passes instead of 3 | 0.632 | 69.7% |
| no final logit softcap | 2.310 | 98.8% (KL is the signal here: top-1 alone would miss it) |
| attention sink = 0 | 0.378 | 73.6% |
| no relative-position bias | 4.864 | 14.2% |
| NVFP4 per-expert `.scale` ignored | 12.95 | 2.0% |
| loop controllers off (no messages) | 1.019 | 64.6% |
| **no log scaling** | 3.96e-3 (identical) | 98.8% — **survives**: the factor is exactly 1 below position 8191 and the passage has 256 tokens. Pinned at op level (`Inkling2OpsTests.LogScale_*`); a >8K-token oracle run would be needed to observe it end to end |

**Loop-controller storage (user ruling pending).** Same gate; bytes are the stored size of `loop_msg_query` + 12 layers x 4 x (`down` 4608x64, `gain` 64x2304, `forget` 64x2304):

| storage | bytes | mean KL | max KL | top-1 | teacher-forced mismatches | free-running prefixes |
|---|---|---|---|---|---|---|
| F32 (GGUF, in place) | 113.7 MB | 3.96e-3 | 6.6e-2 | 98.8% | 6 | 10/3/48/48/48/27/12/48 |
| Q8_0 | 30.2 MB | 3.52e-3 | 6.9e-2 | 98.8% | 6 | 7/3/48/48/48/27/48/48 |
| F16 | 56.8 MB | 3.55e-3 | 7.6e-2 | 98.4% | 5 | 10/18/48/48/48/27/12/48 |
| BF16 | 56.8 MB | 3.37e-3 | 6.9e-2 | 99.2% | 4 | 10/18/48/48/48/27/48/48 |

The four are indistinguishable at this resolution (differences of 5e-4 in KL are below the oracle noise floor; no mismatch at a confident margin in any arm). Q8_0 saves 83.5 MB.

**Conv-state storage** (3.8 MB at F32 for all 48 slots; the sweep is only meaningful because the harness carries state across calls): F32 3.96e-3 / max 6.6e-2; **BF16 3.91e-3 / max 0.11** (4 mismatches, 0 confident); **F16 5.65e-3 / max 0.24, top-1 97.6%** — F16 is visibly worse than BF16 (a value range the 5-bit exponent cannot hold), BF16 is on par with F32 on mean KL.

**Throughput** (32 threads, `cpu-bench` lock held, NVFP4 file, F32 conv state; one measurement, llama.cpp b12780 CPU measures ~10.5 tok/s decode on the same file): prefill 128 tokens 21.4 tok/s, decode 9.1 tok/s (110 ms/token). Not tuned: attention, depth mixing and the convs are scalar loops, MoE is bucketed per expert.
