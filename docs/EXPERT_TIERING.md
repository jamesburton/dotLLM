# Expert tiering (prototype, #939 / parent #825)

Routed MoE experts live in the GGUF on SSD; a per-layer pool of device slots caches them. Vulkan qwen4exp only, decode-shaped (single-row) forwards.

* Enable: `DOTLLM_VK_EXPERT_SLOTS=N` (N >= top-k, clamped to the expert count). Off by default; the resident path is untouched.
* Miss policy: stall. After the router top-k the forward submits, the host reads the ids, loads missing experts (unbuffered overlapped reads into a mapped host-visible ring, no page-cache growth), copies them into slots, rewrites the ids as slot numbers and resumes. Kernels are unchanged, so greedy output equals the resident run.
* Requirements: every routed bank stays packed on the device (no F32 widening); `DOTLLM_GGUF_MAP_COW` unset. MTP is refused when tiered. Prefill runs token by token (slow; streaming prefill is future work).
* Knobs: `DOTLLM_VK_EXPERT_QD` (read queue depth, default 8), `DOTLLM_TIER_DEBUG=1`.
* Counters: `VulkanQwen4ExpTransformerModel.ExpertTierSummary` (hits, misses, evictions, SSD bytes, stall split).
* Probe: `benchmarks/Qwen4ExpRowProbe tier <gguf> <out> <n> [prompts] [warm]`.

Measured (UD-Q4_K_XL, Strix Halo, 6 prompts x 128 greedy tokens, token-for-token identical to the resident run; resident 17.9 tok/s, 77.9 GiB device):

| slots/layer | decode hit rate | tok/s | device |
|---|---|---|---|
| 64 | 70.6% | 1.69 | 15.1 GiB |
| 128 | 85.7% | 3.38 | 24.1 GiB |
| 256 | 91.4% | 5.06 | 42.0 GiB |

SSD-bound (~1 GB/s effective on the slow region). Next: warm RAM tier, expert-major sidecar (one 3 MiB read per expert), streamed prefill, other models' routing hooks.
