# KV Quantization — Memory & Throughput Estimates

## Memory per Token per Layer

All figures for a single KV row = `numKvHeads * headDim` elements.

### Per-element byte cost

| Format | Bytes/element | Overhead/block | Effective bytes/element |
|--------|---------------|----------------|------------------------|
| FP32   | 4.000         | 0              | 4.000                  |
| FP16   | 2.000         | 0              | 2.000                  |
| Q8_0   | 1 + 2/32      | 2B scale/32 el | 1.0625                 |
| Q4_0   | 0.5 + 2/32    | 2B scale/32 el | 0.5625                 |

### KV bytes per token (K+V, one layer)

Formula: `2 * numKvHeads * headDim * bytesPerElement`

| Model shape          | KV heads | head_dim | kvStride | FP32    | FP16   | Q8_0   | Q4_0   | K8V4   |
|----------------------|----------|----------|----------|---------|--------|--------|--------|--------|
| Qwen3.5 27B Dense    | 4        | 128      | 512      | 4096 B  | 2048 B | 1088 B | 576 B  | 832 B  |
| Llama 3 8B           | 8        | 128      | 1024     | 8192 B  | 4096 B | 2176 B | 1152 B | 1664 B |
| Llama 3 70B          | 8        | 128      | 1024     | 8192 B  | 4096 B | 2176 B | 1152 B | 1664 B |
| Mistral 7B           | 8        | 128      | 1024     | 8192 B  | 4096 B | 2176 B | 1152 B | 1664 B |
| Phi-3 mini (3.8B)    | 8        | 96       | 768      | 6144 B  | 3072 B | 1632 B | 864 B  | 1248 B |

### Total KV cache for common configurations (all layers, per sequence)

| Model             | Layers | Ctx len | FP32       | FP16       | Q8_0       | Q4_0       | K8V4       |
|--------------------|--------|---------|------------|------------|------------|------------|------------|
| Qwen3.5 27B Dense  | 64     | 4096    | 1024 MB    | 512 MB     | 272 MB     | 144 MB     | 208 MB     |
| Qwen3.5 27B Dense  | 64     | 16384   | 4096 MB    | 2048 MB    | 1088 MB    | 576 MB     | 832 MB     |
| Qwen3.5 27B Dense  | 64     | 65536   | 16384 MB   | 8192 MB    | 4352 MB    | 2304 MB    | 3328 MB    |
| Llama 3 8B         | 32     | 4096    | 1024 MB    | 512 MB     | 272 MB     | 144 MB     | 208 MB     |
| Llama 3 8B         | 32     | 65536   | 16384 MB   | 8192 MB    | 4352 MB    | 2304 MB    | 3328 MB    |
| Llama 3 70B        | 80     | 4096    | 2560 MB    | 1280 MB    | 680 MB     | 360 MB     | 520 MB     |

### Hybrid model: Bonsai-2 27B on 12GB GPU (Qwen3.5 27B Dense, 20 GPU layers)

- GPU layers: 20 (FP16 KV) = 20 * 2048 B/token = 40 KB/token
- CPU layers: 44

At 64K context:

| CPU format | CPU KV bytes | GPU KV bytes | Total KV | Savings vs CPU FP32 |
|------------|-------------|--------------|----------|---------------------|
| FP32       | 11.5 GB     | 2.5 GB       | 14.0 GB  | baseline            |
| Q8_0       | 3.1 GB      | 2.5 GB       | 5.6 GB   | 60%                 |
| Q4_0       | 1.6 GB      | 2.5 GB       | 4.1 GB   | 71%                 |
| K8V4       | 2.3 GB      | 2.5 GB       | 4.8 GB   | 66%                 |

Fits in 36 GB RAM (T5500 / Framework) with all formats.
Q4_0 enables 64K context on 16 GB RAM systems.

## Target hardware estimates

### T5500 (36 GB RAM, i7-13700H)

Qwen3.5 27B Q4_K_M weights (~15 GB mmap) + KV at 64K:
- FP32 KV: 15 + 14 = 29 GB — fits but tight, leaves ~7 GB for OS
- Q8_0 KV: 15 + 5.6 = 20.6 GB — comfortable
- K8V4 KV: 15 + 4.8 = 19.8 GB — comfortable
- Q4_0 KV: 15 + 4.1 = 19.1 GB — comfortable

### Framework (64 GB RAM, i7-1165G7, AVX-512)

All configurations fit with ample headroom.
Primary concern is compute throughput, not memory.

### Strix (24 GB VRAM, 64 GB RAM)

Hybrid with 40+ GPU layers on 27B:
- GPU KV (FP16, 40 layers, 64K): 5 GB — leaves ~10 GB VRAM for weights
- CPU KV (24 layers): Q8_0 = 1.7 GB, Q4_0 = 0.9 GB — trivial

## Attention throughput: quantized vs F32

Microbench results (Framework i7-1165G7, AVX-512, single thread):

Measured via QuantizedKvAttentionCorrectnessTests — not a dedicated benchmark,
but indicative of kernel overhead. The tiled quantized path includes:
1. Dequantize KV tile (Q8_0 or Q4_0) to F32 scratch
2. Compute QK^T dot products
3. Softmax + V accumulation

Expected overhead vs F32: ~20-40% for Q8_0 (dequant is fast with AVX),
~30-50% for Q4_0 (nibble unpacking + shift). The tradeoff is worth it
at long context where memory bandwidth is the bottleneck — quantized KV
loads fewer cache lines per attention head.

For a formal benchmark, see `benchmarks/DotLLM.Benchmarks/` —
add `QuantizedAttentionBenchmark.cs` when throughput parity is a gate.
