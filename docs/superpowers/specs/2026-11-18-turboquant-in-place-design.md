# TurboQuant KV, read in place, on every backend — design

Date: 2026-11-18. Replaces the ask in #1881. Owner decisions are marked **(James)**; designer calls are marked **(call)**.

## Intent

`--cache-type-k tq4` (and every `tq[1..8][q]` form) must really run TurboQuant KV wherever it is accepted. That covers CPU,
CUDA and Vulkan, decode and prefill, non-paged and paged caches, the scheduler, and prefix reuse. It must also comply with
CLAUDE.md rules 1 and 2: no FP32 storage and no expanded copy of the history. **(James)**

Why: today the option is silently dropped on every default serve path. That is the kind of parity gap the 11-18 rule forbids
(an explicit choice is never substituted).

- `KvCacheConfig.IsQuantized` excludes TurboQuant, so every branch falls through:
  - Vulkan gets the model's default `VulkanKvCache` (f16 or F32).
  - CUDA `CreateKvCache` gets an f16 `CudaKvCache`.
  - The CPU paged branch (paged is the default) gets F16 blocks.
- Only CPU with `--no-paged` builds a real `TurboQuantKvCache`.
- `VulkanTurboQuantKvCache` and `CudaTurboQuantKvCache` are built only by tests and `VulkanDecodeBandwidthHarness`.
- All three real TurboQuant caches decode `[0, CurrentLength)` into a shared FP32 scratch sized for `maxSeqLen`. That is
  exactly the "full-history KV dequant scratch" rule 2 calls a defect.

## Goals

1. **No silent substitution, starting with the first PR.** Any TurboQuant request that the chosen backend, host model and
   setting cannot yet serve for real fails at startup. The message names what is unsupported and suggests `q8_0`/`q4_0`.
   - This applies to serve, run and chat, and to K and V independently. `IsTurboQuant` reads only K today.
   - A V-only `tq*`, or K=tq with V=q8_0/q4_0/f16, is refused until step 6 supports it. **(James: mixed pairs will be
     supported.)**
   - Exception: CPU `tq*q` keeps today's scratch path until step 5, so a feature that works today is not removed. It is the
     only scratch consumer left in production, and step 5 removes it. **(call)**
   - Every other KV store inside a TurboQuant process either runs TurboQuant for real or refuses at startup. That covers MTP
     head KV, the speculative draft model's cache, and any secondary cache. **(call, from the anti-goal "a setting that quietly
     takes a different cache".)**
   - CUDA G3 prefill cannot read codes. It is switched off under TurboQuant, as for q8_0/q4_0: `G3ServesKvCache` must return
     false for tq, and that fix ships in step 1.
2. **In-place MSE attention.** The attention kernels read codes and norms directly. Nothing wider than the codes is ever
   stored for the history.
3. **Settings coverage, in this order (James):**
   1. Refuse.
   2. In-place MSE, dense non-paged, decode and prefill.
   3. Paged pools and the scheduler.
   4. Prefix reuse, growth and seeding (the original #1881).
   5. QJL (`tq*q`) in place on CPU, CUDA and Vulkan. This removes the CPU scratch path.
   6. Mixed K/V pairs: tq with q8_0, q4_0 or f16 on the other side, and V-only tq. **(James)**
   
   Each step lands on CPU, CUDA and Vulkan together. **(James, cross-device parity rule)**
4. **Merge gates (James):**
   - **Correctness gate:**
     - in-place output matches the scratch-path oracle (same codec, so any gap is our bug);
     - encode is byte-exact across CPU, CUDA and Vulkan.
   - **Quality** (KL, PPL and greedy identity against f16) is reported per bit width, not gated.
   - **Speed gate:** in-place must be no slower than today's scratch path at every measured context, since it replaces
     that path.
     - Numbers against f16 and q4_0 are reported.
     - Fast nibble loads are needed only if they are required to clear the gate.
5. Numbers in each PR (rule 1):
   - bytes per token against f16, q8_0 and q4_0;
   - KL and greedy identity against f16 KV;
   - decode and prefill speed at about 7.6K and 32K context, measured on this box (Vulkan and CPU) and on framework/T5500
     (CUDA, functional only on framework).

## Non-goals

- Head dims that are not a power of two, or are above 256: refused loudly (`MaxHeadDim` = 256), with a ticket. Gemma-4's
  512-dim layers fall here.
- Per-layer geometry: refused loudly, with a ticket.
- An FP32 recency window.
- Fusing the rotations into every attention variant (approach B). It may come later, per variant, only when it measures
  faster.

## Anti-goals (failure even if it "works")

- Any full-history or tile-persistent decoded copy of K or V, in any format.
- Any new FP32 storage. That includes a separate FP32 rotated-Q buffer.
- A backend or setting that quietly takes a different cache.
- "Passing" parity tests whose fixture cannot tell K signs from V signs (see Tests).

## The math (what makes in-place possible)

The codec decodes one head vector as `x = n · S · H · c / √d`. Here `c[i] = centroid[code_i]` (already scaled by 1/√d),
`H` is the unnormalized Walsh-Hadamard matrix (`H·H = d·I`), `S = diag(signs)`, and `n` is the stored norm. `S·H/√d` is
orthogonal. With `Rk = S_k·H/√d` for keys and `Rv` likewise for values:

- **Score:** `q·k_j = n_j · (Rkᵀ q)·c_j`. Rotate each query head once: `q̃ = H·(S_k ⊙ q)/√d`. Then
  `score_j = n_j · Σ_i q̃_i · centroid[code_ij]`.
- **Output:** `Σ_j p_j v_j = Rv · (Σ_j p_j n_j c_j)`. Accumulate `ã = Σ_j p_j n_j c_j` in code space with the usual online
  softmax. Then rotate once: `o = S_v ⊙ (H·ã)/√d`.

GQA works unchanged: one K sign set is shared by every head, so each query head rotates independently. RoPE comes before
encoding (K is cached post-RoPE), so it is unaffected. K and V must use their own sign sets.

## Approaches

- **A. Rotate Q before and O after; code-reading attention twins. CHOSEN (James).**
  - Two small in-place FWHT kernels per attention layer:
    - pre: Q ← H(S_k ⊙ Q)/√d, per query head;
    - post: O ← S_v ⊙ H(O)/√d, per query head.
  - The attention variants get a TurboQuant twin whose K and V loads are "centroid lookup × norm".
  - This fits the existing per-format twin machinery:
    - CUDA's `KLOAD`/`VLOAD` templated kvq kernels (`attention_kvq.cu`, `attention_flash_mma_*_kvq.cu`,
      `attention_flash_mma_decode_gqa_split_kvq.cu`);
    - Vulkan's `BlockKvTwins` sets (`attention_f32_*_kvq8.comp` family);
    - CPU `Attention.Execute(…, IQuantizedKvCache, …)`.
  - Cost: two cheap dispatches per layer.
- **B. Fully fused rotations.** Saves the two dispatches, but adds FWHT code and LDS to every variant. The known hd64 (341 B
  headroom) and hd256 LDS cliffs make this risky. Deferred, per variant, on measurement.
- **C. Tile dequant in kernel.** Redoes a d·log d transform per cached vector per query tile, so decode cost grows with
  context. Rejected.

## Decided parts of the how

- **Rotations run in place:**
  - Q is overwritten in its existing buffer and format, after RoPE and before attention.
  - O is overwritten in the attention output buffer before the output projection.
  - No new buffer. **(call: this is the only way that satisfies the "no new F32 store" rule, since the Vulkan Q and O buffers
    are F32 today.)**
  - Each kernel is one workgroup (CUDA: block) per (row, query head), with shared-memory butterflies. The existing
    `turboquant_dequant_f32.comp` step 2 is the template.
- **Encode, then attend.**
  - Prefill and decode encode the fresh K/V rows into codes first.
  - Attention then reads the whole history, current rows included, from codes. That matches the order used today.
  - A model does not need f16 K/V for the current chunk.
- **The encode kernels already exist** on all backends (CPU codec, `turboquant_encode_f32.comp`, `turboquant.cu` F32/F16).
  Keep them. Byte-exact agreement across backends is a test.
- **Storage layout:** per layer, codes packed LSB-first per head vector (`codeUintsPerVec` stride, plus 1 guard uint), and one
  norm per head vector. The CPU layout should match the GPU layout, so that one oracle and future disk persistence share a
  format.
- **Norm format (James: measured, smallest that passes; FP32 is not a default fallback).**
  - The norm is one positive scalar per head vector. FP32 costs 4 B, about 6% of a 4-bit hd128 vector.
  - Step 2 evaluates these candidates, smallest first, and picks the smallest that passes:
    - **8-bit, append-stable, no block:** FP8 E4M3 / E5M2, or an 8-bit log2 code with a fixed per-model range. 1 B.
      Decode writes one norm at a time, so a format that needs no neighbours fits the write path.
    - **q8_0 along positions:** blocks of 32 consecutive norms per head, with an f16 scale. About 1.06 B.
      - Appends rescale a partly filled block, so it needs a write-once or fixed-scale rule (for example, the scale is set by
        the block's first norm, with saturation counted).
      - Assess the drift that comes with that rule.
    - **F16 / BF16.** 2 B.
  - **Pass rule:** the KL cost of the norm format, against FP32 norms, is under 10% of the tq-vs-f16 KL at the same bit
    width, on the fixture models (James).
  - **If nothing below FP32 passes,** FP32 needs James's approval. The PR states the reasoning, the measured numbers per
    candidate, and the options still to explore (for example a per-head scale, a learned range, or a mixed 8-bit / f16
    split). It never falls back on its own.
  - The QJL residual norm gamma gets the same evaluation in step 5.
- **Separate K and V format slots from step 2 (James, mixed pairs in step 6):**
  - The cache types, twins and resolver carry a K format and a V format independently, like the `kvq8q4`/`kvq4q8` pairs.
  - Only the pre-rotation (K tq) and the post-rotation (V tq) depend on which side is TurboQuant.
- **Accumulation:** F32 in registers for scores, softmax sums and `ã` (rule 1 default).
- **Bit widths:** generic `mseBits` 1..8 with straddle extraction (as the existing shaders do).
  - Nibble-aligned 4-bit and 2-bit fast loads are allowed specializations.
  - Every accepted width must run; none may be refused for speed reasons.
- **The scratch caches become oracles.**
  - After step 2, `VulkanTurboQuantKvCache`, `CudaTurboQuantKvCache` and the CPU `TurboQuantKvCache` scratch path are used
    only as test references (the rule's reference exception).
  - Whether to delete them afterwards is James's call: ask, never delete unasked.
- **Hosts:**
  - Step 2 covers the hosts that already take in-place block KV:
    - Vulkan dense chain and Flint (#1884);
    - CUDA dense and paged (#1873/#1844);
    - CPU `TransformerModel`.
  - Every other host (Vulkan hosts in #1885, hybrids, MLA, Qwen4Exp, Qwen3HybridDense) refuses tq at startup. Each gets a
    ticket under the epic, the same day.
- **Refusal point:** one shared resolver, used by serve, run and chat, decides "TurboQuant real here?" from backend, host,
  head dim, setting and QJL. The CLI and `ServerStartup` call it, rather than each branch testing `IsTurboQuant`.
  - Fix `IsTurboQuant` to consider V too, or add `IsTurboQuantAny`.
  - Guard: a test enumerates every `ServerStartup` KV branch with a tq config and asserts that it refuses or builds an
    in-place TurboQuant cache, never anything else.

## Step outlines (each step is one issue with CPU, CUDA and Vulkan PRs that merge together)

1. **Refuse.** Shared resolver, startup errors, `--help`/`docs/CLI.md` text, the branch-enumeration guard test. At this
   point tq works only on CPU `--no-paged`, still through the scratch path (pre-existing, and unchanged by this step).
2. **In-place MSE, dense non-paged.**
   - Pre/post rotation kernels.
   - TurboQuant twins of every attention variant each host dispatches:
     - Vulkan: plain, qpos, sg, splitkv_gqa(_v2), flash GQA prefill, and the sink twins for Flint;
     - CUDA: attention_kvq, flash prefill (+hd256), split decode;
     - CPU: Execute.
   - New dense TurboQuant cache types per backend that own codes and norms only.
   - Wiring through the resolver. CPU `--no-paged` moves off the scratch path.
3. **Paged and scheduler, including #1882 (James).**
   - A TurboQuant block type in the CPU `PagedKvCacheFactory` and the CUDA paged pool. Block size must stay a multiple of
     whole head vectors.
   - Step 3 builds the Vulkan paged pool (#1882) with f16, q8_0, q4_0 and TurboQuant block types from the start, plus the
     Vulkan scheduler over it.
   - The prefix trie over TurboQuant blocks on all three backends.
4. **Prefix reuse (#1881).** On each dense TurboQuant cache:
   - `IPrefixReusableKvCache` (`CanTruncateTo`/`Rollback`);
   - `TryGrowTo`: copies codes and norms as stored, admitted like `VulkanKvCache.AdmitGrowth` / CUDA `AdmitReleasingG3`;
   - `IKvPrefixSeedable`;
   - `IDeviceResidentKvCache` (`DeviceBytes` counts codes, norms and the constant buffers).
   
   Compatibility for seeding and adoption includes `mseBits`, QJL, and a fingerprint of centroids and signs (the seed).
   Codes from another seed decode to plausible garbage.
5. **QJL on GPU.**
   - Score term: `qjlScale · γ_j · (S_qjl q)·sgn_j`. This needs a d×d sketch matvec on each query head; the result is
     written in place beside q̃ if a slot exists, otherwise kept in shared memory per workgroup. **(builder: pick and justify
     the bytes.)**
   - V residual: accumulate `Σ p_j γ_j sgn_j` in sketch space, then apply `S_qjlᵀ` once in the post kernel.

6. **Mixed pairs.**
   - The twins for tq on one side with f16, q8_0 or q4_0 on the other, and V-only tq.
   - Then lift the step-1 refusals for these pairs.

## Tests (CLAUDE.md "Prove the test can fail")

- **Kernel parity.** In-place output against the scratch-cache oracle on real random data:
  - hd 64/128/256;
  - GQA with NKHead > 1 and group > 1;
  - mseBits 3 (straddles uints) and 4;
  - distinct K/V seeds;
  - decode and prefill shapes.
  
  Tolerance scales with √(seq·d).
- **Mutants that must die:**
  - K signs used for V;
  - output rotation skipped;
  - norm dropped;
  - bit offset off by one;
  - `codeUintsPerVec` stride replaced by tight packing;
  - pre-rotation using the V signs.
- **Encode byte-exactness** across CPU, CUDA and Vulkan for the same input and seed.
- **End to end:**
  - greedy identity and KL against f16 KV on a real model (Qwen3-4B or TwIL-8B);
  - PPL on the wikitext fixture;
  - same-session speed A/B.
- **Step 4 mutants:**
  - growth that drops the norms;
  - a seed from another seed's cache that must refuse;
  - a refused growth that evicts the neighbour and retries (the #1843 path).
- **Step 1:** branch-enumeration guard, plus a mutant that routes tq to the default cache and must go red.

## Left to the builder

- Workgroup shapes and specializations per variant.
- Whether the CPU rotation uses AVX2 FWHT or `TensorPrimitives`.
- How the code twins share source with the q8_0/q4_0 twins (macro or `#include`, as today).
- Ticket breakdown inside each step.
- Benchmark model choice within the fixtures that already exist.
