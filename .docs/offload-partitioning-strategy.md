# Partitioning LLM Inference Across an iGPU and a Bandwidth-Limited eGPU

**Question.** Given an Intel Arc iGPU (UMA, ~96 GB shared LPDDR5x, modest compute/bandwidth) and an
RTX 3060 eGPU (12 GB fast VRAM, fast compute, but reached over a *bandwidth-limited external link* —
Thunderbolt 3/4, USB4, or OCuLink), what is the best way to split a transformer across the two devices
so that communication at the split point(s) is minimized?

**Scope.** This document compares four split schemes — layer-split (pipeline), tensor-split,
KV-cache placement, and selective-expert/MoE split — quantifies bytes-crossed-per-token at each split,
and gives concrete recommendations for dotLLM's existing contiguous-layer-split design
(`HybridVulkanCudaTransformerModel`, `HybridVulkanCudaKvCache`).

> **Provenance tiering.** Claims are tagged `[Cited]` (verified against a primary/authoritative source,
> URL given), `[Reported]` (published figure surfaced via aggregation; treat as order-of-magnitude),
> `[Spec]` (theoretical/derived from a spec), or `[Synthesis]` (this document's own reasoning, not from a
> source). The latency figures for external links are the weakest-sourced item and are explicitly
> marked `[Reported]` — do not let them harden into stated fact.

---

## Executive summary (the ranked answer)

**Ranking, fewest bytes crossing the eGPU link → most:**

1. **Layer-split / pipeline (BEST).** Communication = **one hidden-state activation per cut per token**,
   independent of layer count and independent of *where* you cut. With a single cut: ~`hidden_size × dtype`
   bytes per decode token. This is what dotLLM already does, what llama.cpp `--n-gpu-layers` / `--split-mode
   layer` does, and what vLLM uses for pipeline parallelism across slow links. `[Cited]`
2. **MoE expert-parallel (route the activation), where applicable.** If the model is MoE, *pin* experts to
   devices and route the tiny token activation to the device holding the selected experts:
   ~`top_k × hidden_size × dtype` per MoE layer per token (kilobytes). Cheap, but data-dependent and adds
   round-trips. Only relevant for MoE models. `[Cited]` / `[Synthesis]`
3. **KV-cache placement — a *constraint*, not a scheme.** KV **must** be co-located with the attention that
   reads it. Done right it crosses **zero** extra bytes (dotLLM already co-locates per layer). Done wrong
   (KV on the other device) it forces moving the whole cache per token, growing with context length —
   catastrophic over an eGPU link. `[Synthesis]`
4. **Tensor-split / intra-layer (WORST for this hardware).** Requires **two blocking all-reduce collectives
   per layer per token** — ~`2 × hidden_size × L` crossings, ~`2L` serialized round-trips. Designed for
   NVLink, not an external cable. `[Cited]`
5. **MoE streaming of expert *weights* (AVOID).** Moving expert weight tensors to a fixed compute device is
   tens-to-hundreds of MB per expert load — orders of magnitude more than routing the activation. Viable
   only for the iGPU's UMA-resident experts (no link crossing), not across the eGPU link. `[Cited]`

### The single most important rule for this eGPU hardware

> **Keep the cut count at exactly 1, co-locate each layer's KV with that layer, and choose the split point V
> and the quantization to minimize the *slow iGPU's* share of weight bytes.** For a single decode token the
> handoff payload is tiny (a few KB) and costs **tens of microseconds**, while per-token *compute* on each
> device's layer block is **milliseconds** (memory-bandwidth-bound on weights). Communication is therefore
> well under 1% of compute — so the handoff is **not** the thing to optimize. The dominant per-token cost is
> reading weights, and the iGPU's LPDDR5x (~100–120 GB/s) is ~3× slower per byte than the RTX 3060's
> 360 GB/s. The high-leverage levers are therefore **(1) sizing the eGPU's layer block to fill its 12 GB**
> (so the slow iGPU does as little of the work as possible) and **(2) quantization** (fewer weight bytes on
> both devices). Downcasting the boundary activation FP32→FP16 and async-overlapping the handoff are real
> but *second-order* (they touch an already-<1% term); overlap only helps in **batched/multi-stream**
> serving, not single-stream batch=1 decode (see below).

---

## The hardware, quantified

### Effective host↔eGPU bandwidth (one-direction, *measured/effective*, not theoretical peak)

| Link | Tunnels | Theoretical (1-way) | **Effective (1-way)** | Tier |
|------|---------|--------------------:|----------------------:|------|
| Thunderbolt 3 / 4 eGPU | PCIe 3.0 x4 | 3.94 GB/s | **~2.0–2.6 GB/s** | `[Reported]` |
| USB4 v1 (40 Gb/s) | PCIe 3.0 x4 | 3.94 GB/s | **~3.8 GB/s** | `[Reported]` |
| USB4 v2 (80 Gb/s) | PCIe 4.0 x4 | 7.88 GB/s | **~5.4 GB/s** | `[Reported]` |
| **OCuLink** | PCIe 4.0 x4 (raw) | 7.88 GB/s | **~6.6–6.7 GB/s** | `[Measured]` |
| — *contrast:* PCIe 4.0 x16 | — | 31.5 GB/s | ~28–30 GB/s | `[Spec]` |
| — *contrast:* PCIe 5.0 x16 | — | 63.0 GB/s | — | `[Spec]` |
| — *contrast:* NVLink 3.0 (A100) | — | — | 600 GB/s (bidir aggregate) | `[Reported]` |
| — *contrast:* NVLink 4.0 (H100) | — | — | 900 GB/s (bidir aggregate) | `[Reported]` |

Notes. TB3/TB4 badge "40 Gb/s" includes DisplayPort/USB; PCIe tunneling is capped to ~22 Gb/s/direction,
and real eGPU benchmarks land at ~2.0–2.6 GB/s (≈55–65% of the PCIe 3.0 x4 ceiling) because of the
tunneling/buffering layer. **OCuLink is the standout** — it carries *raw* PCIe over the cable with no
tunneling, so it reaches ~85% of its PCIe 4.0 x4 ceiling. NVLink/PCIe-x16 figures are bidirectional
aggregates; normalize before quoting the "NVLink ≈ 7× PCIe" framing. `[Reported]`/`[Spec]`

**Implication: the eGPU link is 5×–300× slower than the in-box interconnects these split schemes were
designed for.** Tensor parallelism assumes NVLink-class bandwidth; an eGPU has none of it.

### Effective host↔eGPU latency (the weakly-sourced but decisive number)

- **Direct-slot PCIe baseline:** small-DMA latency is **sub-µs to ~1 µs** (Cambridge *pcie-bench*,
  Neugebauer et al., SIGCOMM 2018; ~1 µs small DMA, IOTLB miss adds ~330 ns). `[Cited — academic]`
- **Added latency over Thunderbolt/USB4:** rule-of-thumb **~1–2 µs per hop**; an eGPU path is ~2 hops plus
  PCIe-tunneling buffering → order-of-magnitude **a few µs, up to ~10 µs one-way** added vs a slotted card.
  `[Reported — NOT a clean controlled measurement; order-of-magnitude only.]`
- The USB4 spec itself confirms the tradeoff: *"The amount of buffering at the PCIe Adapter is
  implementation specific as it balances the tradeoff between PCIe tunneling performance and PCIe link
  latency"* — you tune for throughput **or** latency, not both. `[Cited]`

So a per-token handoff over an eGPU link has a **fixed cost on the order of tens of microseconds**
(one-way added latency + driver/submit overhead + the bandwidth time for a tiny payload), regardless of
how small the payload is.

---

## 1. Layer split (pipeline / vertical) — the recommended scheme

**Mechanism.** Contiguous layers `0..V-1` run on the iGPU; layers `V..L-1` run on the eGPU. The only thing
that crosses the link is the **hidden-state activation at the cut**: a `[tokens × hidden_size]` tensor,
once per cut, per forward pass. This is exactly dotLLM's `HybridVulkanCudaTransformerModel` (Vulkan layers
0..V, CUDA layers V..L, host-staged FP32 handoff at a single cut).

**Communication formula (bytes crossing the link):**

```
bytes_per_pass = tokens × hidden_size × dtype_bytes × num_cuts
  decode:  tokens = 1   → bytes = hidden_size × dtype_bytes × num_cuts
  prefill: tokens = S   → bytes = S × hidden_size × dtype_bytes × num_cuts
```

**Key property — cut-location invariance.** Each cut crosses exactly *one* activation no matter which layer
boundary it sits on. Therefore comm depends only on **`num_cuts`, not on layer indices**. `[Synthesis]`
→ **Keep `num_cuts = 1`.** Where the cut sits is a VRAM/compute-balance decision (Section 5), not a comm
decision.

> There is also a trivial **return trip each decode step**: the eGPU produces the logits/sampled token, and
> the next token must be re-embedded on the iGPU (the embedding table lives with layer 0). That is a single
> token-id (4 bytes) or one logits vector — negligible, but it means a decode step is one *round-trip*, not
> a one-way hop. The "1 cut" claim refers to the hidden-state activation; acknowledge the tiny return leg.
> `[Synthesis]`

### Bytes-per-cut table (single cut), by model size and dtype

`hidden_size` ranges chosen to bracket common sizes: 2048 (≈1–3B), 4096 (≈7–8B), 5120 (≈13B), 8192 (≈70B).

| hidden_size | dtype | **Decode (1 token)** | Prefill S=512 | Prefill S=2048 |
|------------:|:-----:|---------------------:|--------------:|---------------:|
| 2048 | FP16 | **4 KB** | 2 MB | 8 MB |
| 2048 | FP32 | **8 KB** | 4 MB | 16 MB |
| 4096 | FP16 | **8 KB** | 4 MB | 16 MB |
| 4096 | FP32 | **16 KB** | 8 MB | 32 MB |
| 5120 | FP16 | **10 KB** | 5 MB | 20 MB |
| 5120 | FP32 | **20 KB** | 10 MB | 40 MB |
| 8192 | FP16 | **16 KB** | 8 MB | 32 MB |
| 8192 | FP32 | **32 KB** | 16 MB | 64 MB |

(KB = 1000 B, MB = 1e6 B for readability; exact powers-of-two differ by ~2–5%.)

### Why this is negligible per decode token — the comm:compute ratio `[Synthesis]`

Take a 7–8B model, `hidden_size = 4096`, FP16 handoff = **8 KB/token**, on a TB4 link at **2.5 GB/s**:

```
handoff bandwidth time = 8e3 / 2.5e9 ≈ 3.2 µs
+ added link latency (one-way)        ≈ a few–10 µs        [Reported]
+ host-stage + submit overhead        ≈ a few µs
→ total handoff ≈ tens of µs per decode token
```

Now the eGPU's *compute* for its layer block in that same decode step. Decode is memory-bandwidth-bound on
weights: the eGPU must read every weight in its resident layer block once per token. If the eGPU holds,
say, ~6 GB of resident layer weights (a slice of a quantized 7–8B model — see Section 5), at the RTX 3060's
**360 GB/s** `[Cited]`:

```
eGPU block compute time ≈ 6e9 / 360e9 ≈ 16.7 ms  (lower bound; real >, kernels imperfect)
```

**Comm / compute ≈ tens-of-µs / ~17 ms ≈ < 0.2%.** `[Synthesis]`

Conclusions:

- **The handoff is not the bottleneck — weight reads are.** At <0.2% of compute, neither shrinking the
  handoff payload nor overlapping it can move single-stream decode throughput meaningfully. The per-token
  cost is dominated by reading every resident weight on each device once, and the iGPU's LPDDR5x
  (~100–120 GB/s) is ~3× slower per byte than the eGPU's 360 GB/s. So the levers that matter are the
  **split point V** (minimize the slow iGPU's weight share by filling the eGPU's 12 GB — Section 5) and
  **quantization** (fewer weight bytes everywhere). `[Synthesis]`
- **Overlap helps only in the batched/multi-stream regime.** In **single-stream batch=1** decode there is a
  hard autoregressive dependency: token *t+1*'s input is `embed(sample(logits_t))`, and `logits_t` is
  produced by the *eGPU* (final layers + LM head). The iGPU therefore cannot start the next token's layers
  0..V-1 until the eGPU has finished and returned — the two devices **alternate**, each idle ~half the time
  (a *pipeline bubble*), and the handoff sits unavoidably on the critical path every token. Cross-token
  async copy / double-buffering only hides the handoff when there are **multiple concurrent sequences**
  (microbatch pipelining), which is also the move that fills the bubble. For a single-user rig, accept the
  exposed handoff (it is <1% of token time) and do not build single-stream double-buffering — it cannot
  work. `[Synthesis]`
- **Payload shrinking is second-order.** FP32→FP16 on the boundary activation halves an already-<0.2% term
  for decode. Worth doing because it is free and *does* halve prefill transfer (which is bandwidth-relevant),
  but do not expect a decode-throughput change from it alone. `[Synthesis]`

> **Nuance on "latency dominates bandwidth."** At FP32/4096 the bandwidth time (~6.5 µs at 2.5 GB/s) is
> *comparable* to the added link latency, not dwarfed by it — so the crisp claim is **not** "latency >>
> bandwidth." The bulletproof claim is the ratio above: *the entire handoff (latency + bandwidth + overhead)
> is a fixed per-token cost of tens of µs, which is <1% of the ms-scale per-token compute.* That is why
> overlapping it matters more than minimizing either of its parts. `[Synthesis]`

**Prefill is the one bandwidth-relevant case.** At S=2048, hidden=4096, FP16 = **16 MB** across the cut.
At 2.5 GB/s that is ~6.7 ms — but it happens **once per cut for the whole prompt**, not per layer and not
per token. It overlaps poorly only if the cut is a hard fence; with chunked prefill + async copy it hides
behind compute too. Downcasting to FP16 halves it. `[Synthesis]`

**Sources:** vLLM Parallelism & Scaling docs (PP across nodes, TP within a node); llama.cpp
`docs/multi-gpu.md` and CLI README (`-sm layer` "minimizes data transfers between GPUs").

---

## 2. Tensor split (horizontal / intra-layer) — wrong for an eGPU

**Mechanism.** Each layer's weight matrices are partitioned column/row-wise across devices; both devices
compute partial results that must be combined. In the Megatron-LM formulation a transformer layer uses two
conjugate operators **f** and **g**: in the forward pass **f** is identity and **g** is an **all-reduce** —
one all-reduce after the attention block and one after the MLP block. `[Cited — Megatron-LM §3, Shoeybi et
al. 2019]`

**Communication per token:**

```
crossings_per_token ≈ 2 × L   (two blocking all-reduces per layer)
bytes_per_token     ≈ 2 × hidden_size × L × dtype_bytes × c
```

where `c` is an algorithm/topology constant (for a ring all-reduce across N devices, ≈ `2(N-1)/N`; for
N=2, ≈1). The decisive cost is not the byte volume but that there are **~2L *blocking, serialized*
round-trips per token**. For L=32 that is **~64 synchronizations per token**, each paying the eGPU link's
fixed tens-of-µs latency. `[Cited]` for the 2-per-layer structure; `[Synthesis]` for the round-trip count.

**Concrete contrast (L=32):** layer-split crosses the link **once** per decode token (~8 KB); tensor-split
crosses it **~64 times** with a blocking sync each time. Even ignoring bytes, ~64 × (tens of µs) ≈ low
**milliseconds of pure link stall per token**, on top of compute. `[Synthesis]`

**Why TP exists at all:** it is designed for NVLink-class fabric where an all-reduce is sub-µs and bandwidth
is hundreds of GB/s. Authoritative guidance is explicit:

- vLLM: *"if the GPUs on the node do not have NVLINK interconnect … leverage pipeline parallelism instead of
  tensor parallelism for higher throughput and lower communication overhead … Efficient tensor parallelism
  requires fast internode communication, preferably through high-speed network adapters such as
  InfiniBand."* `[Cited]`
- DeepSpeed: pipeline parallelism *"communicates over an order of magnitude less volume than the data and
  model parallel configurations and is 7× faster at small batch sizes."* `[Cited]`
- HuggingFace (BLOOM/Megatron-DeepSpeed): *"Due to the two all reduces per layer … TP requires a very fast
  interconnect between devices … not advisable to do TP across more than one node unless you have a very
  fast network."* `[Cited]`

**Verdict: do not use tensor-split across the eGPU link.** `[Synthesis]`

---

## 3. KV-cache placement — a co-location rule, not a scheme

**The rule.** KV cache for a layer **must live on the same device as the attention that reads it.** dotLLM
already does this (`HybridVulkanCudaKvCache` co-locates KV per layer with its compute device). Done
correctly, KV contributes **zero** bytes to the link.

**What goes wrong if violated.** If a layer's attention runs on device A but its KV lives on device B, then
*every decode token* must move that layer's entire KV cache across the link (or move all newly-computed Q
and stream K/V back — either way the traffic scales with context). The cost grows **with context length**:

```
KV bytes moved per token (remote KV, per affected layer) =
    context_len × n_kv_heads × head_dim × 2 (K and V) × dtype_bytes
```

Example: context_len=4096, n_kv_heads=8, head_dim=128, FP16 →
`4096 × 8 × 128 × 2 × 2 = 16.8 MB` **per layer per token**, and it *grows every token*. Across many layers
this is gigabytes/token over a 2.5 GB/s link → seconds per token. Co-located, the same quantity is **0**.
`[Synthesis]`

**Conclusion (confirmed, not refuted): KV must stay with its attention across a slow link.** The corollary
for layer-split: because the cut is contiguous, each device owns a contiguous span of layers *and their KV*
— there is never a reason for KV to cross the link. The eGPU's 12 GB budget must include its layers' KV at
max context (Section 5). `[Synthesis]`

---

## 4. Selective-expert / MoE split

Only relevant if the model is MoE (Mixtral, DeepSeek-V3, Qwen3-MoE). Two sub-cases, with very different
link traffic.

### (a) Route the activation to the device holding the experts (Expert Parallelism) — cheap

Experts are *pinned* to devices; per MoE layer the router selects `top_k` experts, and the tiny token
hidden state is dispatched (all-to-all) to whichever device holds them, then combined back.

```
comm ≈ top_k × hidden_size × dtype_bytes   per MoE layer per token   (data-dependent)
```

Example: Mixtral top-2, hidden=4096, FP16 → `2 × 4096 × 2 = 16 KB` per MoE layer per token. Kilobytes —
same order as a layer-split handoff. **But** it is *data-dependent* (which experts, hence which device, is
known only at runtime) and incurs **two all-to-all round-trips per MoE layer**, so over an eGPU link the
*round-trip count* (latency) is the concern, exactly as with tensor-split — only cheaper because it is
`top_k` activations, not full all-reduces, and only on MoE layers. This is how DeepSeek-V3 deploys
(Expert Parallelism, 256 routed + 1 shared expert, top-8) and how Qwen3-MoE is structured (128 experts,
top-8). `[Cited]`

**Routing counts (verified):** Mixtral 8×7B = top-2 of 8; DeepSeek-V3 = top-8 of 256 routed (+1 shared);
Qwen3-MoE = top-8 of 128 (no shared expert). `[Cited]`

### (b) Stream expert *weights* to a fixed compute device — expensive

Keep one compute device fixed; pull expert weight tensors from elsewhere (CPU RAM / disk) on demand.

```
comm ≈ expert_weight_bytes per expert load   (tens–hundreds of MB)
```

A single Mixtral expert is ~tens of MB even at 2–4-bit; FP16 experts are ~hundreds of MB. This is the
**memory-driven offload pattern** (Mixtral-offloading: keep attention + routers resident on GPU, offload
cold experts to RAM, LRU-cache hot experts, speculatively prefetch next-layer experts; FlexGen: LP-planned
weight/KV/activation streaming across GPU/CPU/NVMe). `[Cited]`

**Crucial caveat for *this* hardware:** offload patterns assume the streaming target is **CPU RAM over the
in-box PCIe x16 / UMA** — not an eGPU cable. Streaming hundreds of MB of expert weights *across the eGPU
link per token* is a non-starter (hundreds of MB ÷ 2.5 GB/s = ~100+ ms/token). FlexGen is also explicitly
**throughput-oriented (large batches), not latency** — *"latency-insensitive tasks with batched
processing."* `[Cited]`

### MoE guidance for the iGPU + eGPU topology `[Synthesis]`

- **Pin experts; never stream expert weights across the eGPU link.** Routing the activation (case a) moves
  KB/token; streaming weights (case b) moves MB–hundreds-of-MB/token.
- **Exploit the iGPU's UMA for the offload pattern.** The iGPU has ~96 GB shared LPDDR5x. Keep attention +
  router + **hot** experts on the eGPU's fast 12 GB VRAM; place **cold** experts in the iGPU's UMA where
  "loading" an expert is a local memory read with **no link crossing**. This is the Mixtral-offloading
  idea, but the slow link is *avoided* by mapping "GPU vs CPU-RAM" onto "eGPU-VRAM vs iGPU-UMA." `[Synthesis]`
- **If experts are split across both devices,** accept the per-MoE-layer all-to-all round-trips of case (a),
  and minimize how often a token's selected experts straddle the link (place the statistically hottest
  experts together on one device). The round-trip-count caveat from Section 2 applies. `[Synthesis]`

---

## 5. Concrete recommendations for dotLLM's contiguous-layer-split

dotLLM's `HybridVulkanCudaTransformerModel` already implements the *best* scheme (contiguous layer-split,
single cut, KV co-located per layer, host-staged handoff). The recommendations refine it:

1. **Keep `num_cuts = 1`.** This is already the design. Comm is invariant to cut *location* (Section 1), so
   never introduce a second cut — that doubles link crossings for no compute benefit. `[Synthesis]`

2. **Choose `V` (the cut layer) by VRAM + compute balance, not by comm.** The eGPU's 12 GB must hold:
   `eGPU layer weights (layers V..L-1) + those layers' KV cache at max context + activation scratch + final
   RMSNorm + LM head`. Size the eGPU's layer block to **fill ~12 GB** (leaving ~1 GB headroom) so the faster
   RTX 3060 does as much of the compute as fits; push the remainder onto the iGPU's large UMA. Concretely:
   compute per-layer weight bytes at the model's quantization, subtract per-layer KV at max context, and
   solve for the largest contiguous tail span that fits. Put the **later** layers on the eGPU (the cut is a
   tail span) so the LM head — already on the eGPU in the current design — stays co-located with the final
   layers. `[Synthesis]`

   *KV budgeting reminder:* per-layer KV at max context =
   `context_len × n_kv_heads × head_dim × 2 × dtype_bytes`. For context_len=8192, 8 KV heads, head_dim=128,
   FP16 → ~33.5 MB/layer; for a 16-layer eGPU span, ~537 MB of the 12 GB is KV. Don't forget it when sizing
   `V`. `[Synthesis]`

3. **Downcast the boundary activation to FP16 (or below).** The current handoff is Vulkan→host **FP32**→CUDA
   FP16. Casting to FP16 *before* the D2H copy halves the link bytes (and halves the bandwidth-relevant
   prefill transfer). This is free and harmless numerically at the layer boundary. It is a **second-order**
   decode win (the handoff is already <1% of compute) but a **real prefill win** (16 MB → 8 MB at S=2048,
   hidden 4096). `[Synthesis]` *(Optional, measure-first: BF16 or even FP8 boundary activation if parity
   tests stay green — the activation is a transient, not a weight.)*

4. **Async overlap helps batched serving, not single-stream decode — scope it accordingly.** In batch=1
   single-stream decode the devices alternate (a pipeline bubble: each idle ~half the time waiting for the
   other), and the handoff is unavoidably on the critical path because token *t+1*'s input depends on the
   eGPU's `logits_t`. Cross-token double-buffering **cannot** hide the handoff here — do not build it for the
   single-user path. Where async overlap *does* pay off is **multi-stream / microbatched serving**: with ≥2
   concurrent sequences the iGPU can run sequence B's early layers while the eGPU runs sequence A's late
   layers, which both fills the ~50% bubble and hides the handoff for free. The current design
   fence-serializes the device→host→device staging; the async-overlap work already underway is the right
   investment **for the batched server path**, not for single-stream latency. `[Synthesis]`

5. **Decode is latency/fixed-cost-bound at the link, but the link is not the bottleneck.** Decode payloads
   are KB, so the link's *fixed cost* (added latency + submit overhead + tiny-payload bandwidth time = tens
   of µs) dominates its *variable* (bandwidth) cost — but that whole handoff is <1% of per-token compute, so
   it is not where decode time goes (weight reads are — recs #2, #3). Still, keep the hot path clean: one
   cut, pinned staging buffers, and **no per-token sync the architecture does not strictly require** (e.g.,
   do not add diagnostic D2H reads on the decode hot path — they each pay the link's fixed latency). `[Synthesis]`

6. **Prefer OCuLink if the user can choose the link.** At ~6.6 GB/s effective (raw PCIe 4.0 x4, no
   tunneling) OCuLink is ~2.5× a TB3/TB4 eGPU and meaningfully lowers prefill transfer time and, plausibly,
   per-hop latency (no tunneling buffer). Bandwidth still does not matter much for decode, but prefill and
   any future weight-streaming both benefit. `[Reported]`

7. **If/when dotLLM gains MoE support on this topology:** pin experts per device and route activations
   (Section 4a); never stream expert weights across the eGPU link (4b). Map the Mixtral-offloading
   "GPU-resident vs CPU-RAM" split onto "eGPU-VRAM (attention + router + hot experts) vs iGPU-UMA (cold
   experts)" so cold-expert access is a local UMA read, not a link crossing. `[Synthesis]`

---

## Per-scheme summary — bytes crossing the eGPU link per *decode* token

| Scheme | Crossings/token | Bytes/token (hidden=4096, L=32, FP16) | Grows with | Verdict |
|--------|----------------:|---------------------------------------:|-----------|---------|
| **Layer-split, 1 cut** | 1 (+tiny return) | **~8 KB** | nothing (fixed) | **Best — use this** |
| MoE route-activation (top-2) | 2 per MoE layer | ~16 KB / MoE layer | top_k | Good, MoE only |
| KV co-located (correct) | 0 | **0** | — | Mandatory rule |
| KV remote (wrong) | per layer | ~17 MB/layer/token & rising | context length | Never |
| Tensor-split | ~2L = ~64 | ~`2·H·L·dtype` + 64 syncs | layer count | Avoid (needs NVLink) |
| MoE stream expert weights | per selected expert | tens–hundreds MB | expert size | Avoid across link |

*(All "bytes" are the link-crossing volume; co-located KV and resident weights never cross.)*

---

## Sources

**Pipeline vs tensor parallelism / layer-split**
- vLLM — Parallelism and Scaling (TP within a node, PP across nodes; "if the GPUs … do not have NVLINK …
  leverage pipeline parallelism"): https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
- llama.cpp — `docs/multi-gpu.md` ("pipeline-parallel … minimizes data transfers between GPUs"; tensor-split
  "much more bottlenecked by the GPU interconnect speed"):
  https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- llama.cpp — CLI README (`-sm layer` default "split layers and KV across GPUs (pipelined)"):
  https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md

**Tensor parallelism comm structure**
- Shoeybi et al. 2019, "Megatron-LM" (two all-reduces per layer in forward; f/g conjugate operators, §3):
  https://arxiv.org/abs/1909.08053
- Microsoft DeepSpeed — "Extreme-scale model training for everyone" (PP "an order of magnitude less"
  comm volume, "7× faster at small batch sizes"):
  https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/
- HuggingFace — "The Technology Behind BLOOM Training" (TP "requires a very fast interconnect"):
  https://huggingface.co/blog/bloom-megatron-deepspeed
- HuggingFace Transformers — multi-GPU training guide:
  https://huggingface.co/docs/transformers/en/perf_train_gpu_many

**MoE offloading / expert parallelism**
- Eliseev & Mazur 2023, "Fast Inference of Mixture-of-Experts Language Models with Offloading" (LRU expert
  cache, speculative prefetch, mixed quant, ~2–4 tok/s on 12–16 GB GPUs): https://arxiv.org/abs/2312.17238
  · impl: https://github.com/dvmazur/mixtral-offloading
- Sheng et al. 2023, "FlexGen" (LP-planned GPU/CPU/NVMe streaming; explicitly throughput-not-latency):
  https://arxiv.org/abs/2303.06865
- DeepSeek-V3 Technical Report (Expert Parallelism; 256 routed +1 shared, top-8; all-to-all over IB/NVLink):
  https://arxiv.org/abs/2412.19437
- Qwen3 Technical Report (128 experts, top-8, no shared expert): https://arxiv.org/abs/2505.09388

**eGPU link bandwidth & latency**
- eGPU.io — TB3 eGPU CUDA bandwidth (~2.0–2.6 GB/s effective):
  https://egpu.io/forums/pc-setup/slow-tb3-performance/ ·
  https://egpu.io/forums/thunderbolt-enclosures/technical-questions-on-tb3-pcie-tunnelling-bandwidth/
- rkblog — OCuLink/USB4 eGPU measured (PCIe 4.0 x4 ≈ 6.71 GB/s): https://rkblog.dev/posts/pc-hardware/gpd-win-max2/nvidia-egpu-gpd/
- Tom's Hardware — OCuLink vs TB5 eGPU transfer tests:
  https://www.tomshardware.com/pc-components/gpus/oculink-outpaces-thunderbolt-5-in-nvidia-rtx-5070-ti-tests-latter-up-to-14-percent-slower-on-average-in-gaming-benchmarks
- Notebookcheck — USB4 v2 eGPU bandwidth: https://www.notebookcheck.net/USB4-v2-shows-clear-gaming-performance-gains-over-40-Gbps-USB4-in-OneXGPU-Lite-tests.1136891.0.html
- PCIe bandwidth reference table: https://www.diskmfr.com/pcie-interface-bandwidth-speed-calculation/
- Neugebauer et al. 2018, "Understanding PCIe performance for end host networking" (pcie-bench; ~1 µs small
  DMA baseline): https://www.cl.cam.ac.uk/research/srg/netos/projects/pcie-bench/neugebauer2018understanding.pdf
- NVLink bandwidth reference: https://www.spheron.network/blog/what-is-nvlink-gpu-interconnect-bandwidth-explained/

**Hardware spec**
- RTX 3060 12GB — 360 GB/s GDDR6 memory bandwidth (192-bit, 15 Gbps):
  https://www.msi.com/Graphics-Card/GeForce-RTX-3060-Gaming-X-12G/Specification
