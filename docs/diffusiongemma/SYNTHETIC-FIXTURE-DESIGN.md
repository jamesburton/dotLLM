# Synthetic Gemma-4 / DiffusionGemma GGUF fixture — design

Goal: a deterministic, architecturally-complete, TINY gemma4 + diffusion-gemma GGUF that exercises
every feature of the real 26B at a fraction of the size, for (a) fast deterministic regression,
(b) cross-backend kernel optimization (CPU now; Vulkan/ROCm here, CUDA on a 12 GB T5500), and
(c) per-stage benchmark timing. A real `.gguf` file is the portable artifact every backend loads
identically via the normal `ModelLoader.LoadFromGguf` path. The whole 26B never needs to be resident.

## Quant coverage: F32 + Q8_0 + Q4_K + Q5 (per-tensor-class, mirroring the real model)
- F32: norms, scales, router gate (ffn_gate_inp), token_embd allowed F32 (or Q8_0).
- Q8_0: attn q/k/v/o, dense ffn gate/up/down.
- Q4_K: fused gate_up experts (ffn_gate_up_exps).
- Q5_0 or Q5_1: down experts (ffn_down_exps).
Quant types are CONFIGURABLE per tensor class; default = the real-model mix above. An all-F32 preset
gives the pure-correctness golden.

## NEW: quantizers (the hard part) — `Quantize` mirroring `Dequantize`
Implement F32 -> {Q8_0, Q5_0, Q5_1, Q4_K} producing blocks BIT-COMPATIBLE with the existing
dequantizers (src/DotLLM.Cpu/Kernels/Dequantize.cs for Q8_0/Q5_0/Q5_1, DequantizeKQuants.cs for Q4_K,
QK_K=256). VERIFY each by round-trip: quantize(F32) -> Dequantize -> compare within the format's
expected error (Q8_0 ~max/127; Q4_K ~K-quant error). Round-trip tests are the correctness gate.

## Tiny default config (all configurable; exercises BOTH layer types + V-less globals)
block_count=6 (layers 0-4 sliding, layer 5 global/V-less — sliding_window_pattern [1,1,1,1,1,0],
written as Int32 array; extractor handles int[]). hidden=64, head_count=4,
head_count_kv=[2,2,2,2,2,1], key_length=32 (global head_dim), key_length_swa=16 (sliding head_dim),
value_length=32, value_length_swa=16, rope.dimension_count=32, dimension_count_swa=16,
rope.freq_base=1e6, freq_base_swa=1e4, sliding_window=8, expert_count=8, expert_used_count=2,
expert_feed_forward_length=16, feed_forward_length=32, final_logit_softcapping=30,
embedding_length_per_layer_input=0, shared_kv_layers=0, context_length=512, eps=1e-6, vocab=256.
embedding_scale = sqrt(hidden). PartialRotaryFactor 0.25 is set by the extractor (not a GGUF tensor).
A "bench" preset scales dims up to be compute-bound while fitting ~12 GB.

## Tensors (GGUF shape = [ne0=in, ne1=out, ...]; per the real-model dump)
Per layer blk.{i}: attn_norm[H], attn_q[H, nHead*hd], attn_k[H, nKv*hd], attn_v[H, nKv*hd] (OMIT on
global layers 5,11,..), attn_q_norm[hd], attn_k_norm[hd], attn_output[nHead*hd, H],
post_attention_norm[H], ffn_norm[H], ffn_gate[H, ff], ffn_up[H, ff], ffn_down[ff, H],
ffn_gate_inp[H, nExp] (+ ffn_gate_inp scale[H] F32), ffn_gate_up_exps[H, 2*Ie, nExp],
ffn_down_exps[Ie, H, nExp] (+ ffn_down_exps scale[nExp] F32), pre_ffw_norm_2[H], post_ffw_norm_1[H],
post_ffw_norm_2[H], post_ffw_norm[H], layer_output_scale[1].
DIFFUSION adds per layer: enc_layer_output_scale[1]. Model-level (diffusion): self_cond_pre_norm[H],
self_cond_gate[H, ff], self_cond_up[H, ff], self_cond_down[ff, H].
Model-level: token_embd[H, vocab], output_norm[H]. (Gemma ties output to token_embd.)
hd = per-layer head_dim (16 sliding / 32 global). Global attn_q out = nHead*32, attn_k out = nKv_global*32.

## Metadata: general.architecture = "gemma4" (AR) or "diffusion-gemma"; all {arch}.* keys above;
tokenizer.ggml.* minimal (model="llama" or "gpt2", tokens[vocab] simple strings, scores, token_type,
bos/eos/unknown + diffusion mask_token_id=4). Forward/regression drive RAW token ids (no real
tokenizer round-trip needed); tokenizer metadata only needs to satisfy load + vocab size.
DIFFUSION: diffusion.canvas_length=8, attention.causal=false.

## Deliverables (CPU foundation)
1. `Quantize` (DotLLM.Cpu.Kernels) — Q8_0/Q5_0/Q5_1/Q4_K, round-trip verified.
2. GGUF writer + `SyntheticGemma4Gguf` builder (DotLLM.Models.Gguf) — configurable dims + per-class
   quant, gemma4 + diffusion-gemma, WriteToFile + ToBytes. Reuse GgufTestData's write conventions.
3. Regression test: load tiny gemma4 AR + diffusion-gemma fixtures, run a cacheless forward + a small
   diffusion denoise loop, assert DETERMINISTIC golden (logit checksum / argmax ids stable across runs,
   single-thread) AND assert all features present (dual head dim, V-less global loads, MoE config,
   DiffusionConfig + self_cond loaded, partial rope). The all-features verification gate.
4. Timestamp/benchmark harness: Stopwatch around load + warmup + N forwards (prefill) + diffusion
   steps; structured CSV/console per-phase output (load, forward total, and finer per-stage if cheap
   via existing hooks). Size-preset param. Plugs into benchmarks/DotLLM.Benchmarks for rigorous runs.
5. A way to emit the tiny .gguf to disk (helper/CLI) so the T5500 can consume it for CUDA kernel dev.
