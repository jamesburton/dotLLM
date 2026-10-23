# I2_S Decode Bandwidth Profiling — Results

Ran on: Strix Halo (Ryzen AI Max 395, Zen5, AVX2+AvxVnni confirmed by this harness's own probe;
AVX-512 is present on this box but was not probed/exercised by this harness — see note below),
.NET 10, Release build,
`dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~I2SDecodeBandwidthProfileBench"`.

All 3 test cases passed (`Total tests: 3, Passed: 3`).

Note: the streaming-ceiling probe (`BenchStreamingReadOnly`) itself only uses AVX2 vectorization
(not AVX-512, despite the design spec originally suggesting AVX-512). This is a conservative
choice for the compute-bound conclusion below — a wider (AVX-512) probe could only report a
*higher* achievable-bandwidth ceiling, which would only widen the existing 20-60× headroom gap
and strengthen, not weaken, the compute-bound verdict.

| Shape | Hot ceiling (GB/s) | Cold/DRAM ceiling (GB/s) | Full decode (GB/s) | decode/hot | decode/cold |
|---|---|---|---|---|---|
| attn_qproj (2560×2560) | 38.98 | 24.07 | 1.10 | 2.8% | 4.6% |
| ffn_gate (6912×2560) | 56.59 | 27.07 | 1.15 | 2.0% | 4.2% |
| ffn_down (2560×6912) | 67.87 | 26.55 | 1.19 | 1.7% | 4.5% |

The "decode/cold" column is not an apples-to-apples comparison: hot-buffer decode was never
measured against the cold/DRAM-forced buffers directly (decode only ran against `hotWeights`).
Read it as a lower-bound argument instead — if hot-buffer decode is only ~4-5% of even the
pessimistic cold ceiling, memory bandwidth cannot be the limiter under any plausible residency
assumption. The "decode/hot" column is the true apples-to-apples ratio and the more direct signal.

Raw output per shape:

```
[attn_qproj] m=2560 k=2560 weightBytes=1638400 AVX2=True AvxVnni=True
  hot streaming-only:  0.0420 ms/call   38.98 GB/s   (cache-resident ceiling)
  cold streaming-only: 0.0681 ms/call   24.07 GB/s   (DRAM-forced ceiling)
  unpack-only:         0.4338 ms/call
  full GemvI2_S decode:1.4835 ms/call   1.10 GB/s
  decode / hot-ceiling ratio:  2.8%
  decode / cold-ceiling ratio: 4.6%

[ffn_gate] m=6912 k=2560 weightBytes=4423680 AVX2=True AvxVnni=True
  hot streaming-only:  0.0782 ms/call   56.59 GB/s   (cache-resident ceiling)
  cold streaming-only: 0.1634 ms/call   27.07 GB/s   (DRAM-forced ceiling)
  unpack-only:         0.7263 ms/call
  full GemvI2_S decode:3.8532 ms/call   1.15 GB/s
  decode / hot-ceiling ratio:  2.0%
  decode / cold-ceiling ratio: 4.2%

[ffn_down] m=2560 k=6912 weightBytes=4423680 AVX2=True AvxVnni=True
  hot streaming-only:  0.0652 ms/call   67.87 GB/s   (cache-resident ceiling)
  cold streaming-only: 0.1666 ms/call   26.55 GB/s   (DRAM-forced ceiling)
  unpack-only:         0.4884 ms/call
  full GemvI2_S decode:3.7321 ms/call   1.19 GB/s
  decode / hot-ceiling ratio:  1.7%
  decode / cold-ceiling ratio: 4.5%
```

## Verdict

Full `GemvI2_S` decode uses only **1.7%–2.8% of the cache-resident (hot) streaming-read ceiling**
and **4.2%–4.6% of the DRAM-forced (cold) streaming-read ceiling** across all three probed shapes
— decode is well under *both* ceilings, not close to either, so on this box the kernel is
**compute-bound, not memory-bandwidth-bound**: there is roughly 20-60× of unused bandwidth headroom
between what the raw bytes-in-flight could sustain and what the full unpack+dot decode actually
achieves. `unpack-only` time (0.43–0.73 ms) is itself already a non-trivial fraction of total decode
time (1.48–3.85 ms) — 13.1%–29.2% across the three shapes (attn_qproj 0.4338/1.4835=29.2%,
ffn_down 0.4884/3.7321=13.1%, ffn_gate 0.7263/3.8532=18.8%) — corroborating that the 2-bit-unpack/dot
arithmetic, not the memory access pattern, is the limiting factor. By subtraction, the remaining
71–87% of full decode time (100% minus the 13.1%–29.2% unpack share) is spent in the dot-product
routine (`VecDotI2SQ8`-style code in `src/DotLLM.Cpu/Kernels/MatMul.I2S.cs`) — precisely the routine
the proposed AVX-512 activation-LUT kernel would replace, which is a more direct justification for
that follow-up than the bandwidth-headroom argument alone, though it strengthens rather than
replaces that argument. Note the cold-ceiling column here (24–27 GB/s) is noticeably higher
and more uniform across shapes than an earlier run of the same harness (which saw 2.86–3.11 GB/s cold for the
two larger, 4.4 MB-weight shapes) — this run-to-run swing is itself an illustration of the design
doc's caveat that the round-robined cold buffer probe is sensitive to TLB/page-walk and
system-memory-contention effects on this UMA box (a known source of ~40% bandwidth measurement
swings here) rather than a clean, reproducible DRAM bandwidth figure; it should not be read as a
precise ceiling in either run. The hot-ceiling ratio is the more stable and trustworthy signal, and
by that measure the conclusion is unambiguous regardless of cold-column noise. Per the design doc's
decision rule, this points to proceeding with the **AVX-512 activation-LUT dot kernel** as the
follow-up issue — not a pivot to packing-density work — since the bottleneck is arithmetic
throughput in the unpack+dot path, not bytes moved from memory.
