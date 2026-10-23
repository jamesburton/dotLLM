---
date: 2026-05-18
audit_scope: feature/qwen3.6 @ 1a97fd6, post-Strix-Halo session
auditor: parallel worktree (agent-acad20d10a6c27258)
parallel_agents_excluded:
  - Phase 5b CPU intra-block matmul fusion
  - Phase 5f Vulkan intra-block matmul fusion
  - HybridPrefillDecodeBenchmarks BDN discovery skip
  - Gemma 3 prerequisite revival
---

# State audit — 2026-05-18

## Summary

The branch is in solid functional shape — Strix Halo session landed a critical
correctness fix (sliding-window mask off-by-one across 5 GPU kernels), four
incremental wirings (IQ3 dispatch in all 4 Vulkan transformer hosts), a real
throughput win (ForwardBatch CPU lm_head fusion, bit-exact parity), and a
definitive negative result that closes one branch of the LoRA Q8_0 search.
Code-side hygiene is good: tests for the most recent fix include
discriminating fixtures, and the cross-backend mandate from
`fdb39b4`/CLAUDE.md was honoured for the GPU work.

The brittle parts are now documentation and housekeeping: ROADMAP and README
roadmap-table counts are materially out of date (Phase 7 / 8 / 9 each
understate completed steps); PERFORMANCE.md §6.4 collides with §6.6 under a
duplicate header; LORA.md has not been updated with the Path B3 disconfirmation;
SUPPORTED_MODELS.md still says IQ3 per-host dispatch is a follow-up
(it's done); 11 stale `.continue-here-*.md` files litter the repo root; the
CUDA AttentionF32 and (entirely absent) CUDA AttentionF16 test surfaces have
no sliding-window discriminator despite both kernels having been touched by
`8e8c228`; CUDA PTX is stale relative to the patched `.cu` (acknowledged in the
commit message but should be filed); and one new ForwardBatch test mis-uses
`[Fact]` where the rest of the file uses `[SkippableFact]`, causing a hard
failure rather than a skip when the SmolLM-135M cache is absent.

Nothing in this audit is a correctness regression. The highest-leverage next
work is doc/housekeeping triage + closing the CUDA test coverage gap before
the next CUDA host is available.

## Critical gaps (must fix)

### C1. CUDA attention kernels have no sliding-window discriminator test

**Where:** `tests/DotLLM.Tests.Unit/Cuda/Kernels/AttentionF32ParityTests.cs`,
`tests/DotLLM.Tests.Unit/Cuda/CudaKernelComparisonTests.cs:480-527`.

**Problem:** Commit `8e8c228` fixed the sliding-window off-by-one in *both*
`native/kernels/attention_f32.cu` and `native/kernels/attention.cu` (F16
static + dynamic-seq paths). Three new Vulkan tests cover the equivalent shader
paths. Zero new CUDA tests cover the CUDA paths — both files exercise the
attention kernel with `slidingWindow: 0` only. The original GDN
head-broadcast bug shipped through CI for months precisely because every
test used parameters where the buggy and correct kernels coincided; the
sliding-window fix has the exact same shape (`>` vs `>=`) and the same
trap-the-bug discipline applies cross-backend.

Worse: there is **no CUDA F16 attention parity test at all** anywhere in
`tests/`. Grep for `attention_f16` / `AttentionF16` / `LaunchAttentionF16`
returns no hits in `tests/`. The F16 kernel has been live in the CUDA backend
(via `_attentionFunc` / `_attentionDynFunc` in `CudaKernels.cs:420-421`) and
gets exercised through end-to-end model tests, but the kernel-level
sliding-window invariant is completely untested.

**Next step:** When a CUDA host is next available, add at minimum:
- `AttentionF32ParityTests.AttentionF32_SlidingWindow_DiscriminatesOffByOne`
  (mirror of `VulkanAttentionF32KernelTests.Launch_SlidingWindow_DiscriminatesOffByOne`:
  `seqQ=1, seqKv=8, slidingWindow=4, posOffset=7` — boundary `tkv=3`).
- A new `AttentionF16ParityTests` (or extend the existing file) covering
  both `attention_f16` and `attention_f16_dyn` against the CPU F32 oracle
  (with a relaxed F16-quant tolerance ~5e-3).

**Why "critical":** This is the same class-of-bug as the GDN head-broadcast
trap. CI is green; the only thing standing between a regression and a
production-mis-mask is the .cu source file that the next agent might edit.

### C2. `TransformerModelForwardBatchTests.ForwardBatch_EmptyRequests_ReturnsEmpty` uses `[Fact]` with `Skip.IfNot`

**Where:** `tests/DotLLM.Tests.Unit/Models/Architectures/TransformerModelForwardBatchTests.cs:197-208`.

**Problem:** The other three tests in the file use `[SkippableFact]` (lines
36, 74, 130) so the `Skip.IfNot` call at the top of each test correctly skips
when the SmolLM-135M GGUF is not cached at
`~/.dotllm/test-cache/QuantFactory/SmolLM-135M-GGUF/`. The fourth test
(line 197) uses bare `[Fact]` but still calls `Skip.IfNot` (line 200) — on a
fresh CI host without that cached GGUF, the test throws
`Xunit.SkipException` and xUnit reports it as a failure (because the test
class isn't marked as `Skippable`-supporting in that test method). On the
Strix Halo dev host the cache exists, so the bug is invisible locally.

**Next step:** Change line 197 from `[Fact]` to `[SkippableFact]`. The empty-
requests case doesn't logically need the GGUF, so an alternative fix is to
remove the `Skip.IfNot` call and the `GgufFile.Open`/`TransformerModel.Load`
boilerplate — `ForwardBatch_EmptyRequests_ReturnsEmpty` should be testable
without any model loaded. Recommended: the latter (proper test scope).

## High-value improvements (worth a future session)

### H1. ROADMAP step-status drift — three phases understate completion

Concrete drift relative to the parent worktree's tip at `1a97fd6`:

- **Phase 9 Step 35 `Continuous batching`** is *not* marked `:white_check_mark:`
  in `docs/ROADMAP.md:167`, but `src/DotLLM.Engine/Scheduler/ContinuousBatchScheduler.cs`,
  the server wire-up in `ServerStartup.cs`, the docs in `docs/SCHEDULING.md`
  and `docs/SERVER.md`, and `IModel.ForwardBatch` are all in tree. README's
  Phase 9 status table says "In Progress (3/5)" but should be "(4/5)".
- **Phase 7 Step 47 `LoRA adapters`** is *not* marked `:white_check_mark:`
  in `docs/ROADMAP.md:132`, despite LoRA Phase 4a-4d.6 shipping (CPU forward,
  PEFT loader, multi-adapter batcher, server API, Vulkan delta dispatch,
  outer-product Stage 2 fast path). README's Phase 7 status table says
  "In Progress (1/5)" but should be "(2/5)".
- **Phase 8 Step 49 `ALiBi` and Step 57 `Gemma 4`** are correctly unchecked
  (ALiBi has scattered support but no roadmap-conformant landing; Gemma 4
  is the deferred Gemma 3 work pinned via tags). However, README's Phase 8
  status says "In Progress (3/6)" — the ROADMAP table actually shows
  **14 of 16 steps done** for Phase 8 (48 / 48a / 56 / 58 / 58a / 58b /
  58c + multi-line "step 8" entries). The README's "(3/6)" is a coarse
  bucket that doesn't reflect the multi-step Phase 8 expansion. Either
  recompute or change the column.

Also: **the Phase 10 Qwen3MoeHybrid row** in README:749 still says
"Real-GGUF GPU parity tests pending" — this is technically still true (the
30 GB Q6_K_XL CUDA path needs hardware not on hand), but the same row's
"In Progress (5/5 impl, parity pending)" understates the
session-2 IQ3-dispatch landing, the sliding-window fix, and the FA shader
that's now in tree.

**File paths to edit:**
- `docs/ROADMAP.md:132` — add `:white_check_mark:` to Step 47.
- `docs/ROADMAP.md:167` — add `:white_check_mark:` to Step 35.
- `README.md:738-749` — recompute Phase 7/8/9 numerators in the Roadmap table.

**Estimate:** 30 min including a final consistency pass.

### H2. PERFORMANCE.md has two `### 6.4` and two `### 6.5` headers

**Where:** `docs/PERFORMANCE.md:450,475,491,576`.

**Problem:** Commit `380e4b9` inserted §6.4 ("Vulkan Flash Attention
microbench") and §6.5 ("Outstanding measurements") above the existing §6.4
("Top-3 perf-headroom items") and §6.5 ("Reproduction commands"). Section
numbering now has two 6.4s and two 6.5s. `grep -n "^### " docs/PERFORMANCE.md`
shows the duplicate cleanly. Markdown renderers will TOC them both at the
same level; cross-references in commits and notes that say "§6.4" are now
ambiguous.

**Next step:** Renumber the pre-existing pair to §6.6 and §6.7 (or push the
new pair onto the end). The Vulkan FA microbench arguably *should* sit
between §6.3 (cross-machine reading) and the headroom analysis, so keep
the new 6.4/6.5 where they are and renumber the older sections.

**Estimate:** 10 min.

### H3. IQ3 per-host wiring shipped without per-host real-GGUF parity coverage

**Where:** `tests/DotLLM.Tests.Unit/Vulkan/` — search for `IQ3` returns 5
files, all kernel-level (`VulkanMatMulIq3*F32KernelTests.cs`,
`VulkanIq3*DequantF32KernelTests.cs`, `Iq3Fixture.cs`). No per-host
`*TransformerModel*IQ3*ForwardTests.cs` exists.

**Problem:** Commits `07f391f` (dense), `48d65fe` (Qwen3MoeHybrid),
`146d747` (NemotronH), `ad6b853` (Mamba3) wire IQ3_S + IQ3_XXS into each
Vulkan transformer host's `RecordMatmul` dispatch. The 16 kernel-level
parity tests cover the kernels themselves but not the dispatch — a
miswired branch (wrong codebook handle, wrong predicate, IQ2-vs-IQ3
typo at the case label) would compile, pass kernel tests, and silently
mis-decode an IQ3 GGUF at inference time. The 5293d37-style
"degenerate-shape trap" applies: if no host test exercises the IQ3 path,
the kernel-test ✓ doesn't prove the model-level ✓.

The reference pattern is the existing `VulkanMamba3TransformerModelQ4KForwardTests.cs`
+ siblings (Q5K, Q6K, Q8_0, F16, BF16) — six per host. The same shape for
IQ3 is the cheapest discriminator.

The blocker: **no host has a small synthetic IQ3-quant GGUF on hand**. The
CPU oracle path that the dispatch tests would compare against does
support IQ3 (`DequantizeIQ3.cs` + 4 oracle tests from `1be00fb`), so the
upstream side of the parity is solid.

`.continue-here-iq3-dispatch.md` (now obsolete) actually called this
out as a follow-up: "...add IQ3-specific integration test that loads a
small synthetic IQ3-quant model."

**Next step:** Either (a) commit a tiny synthetic IQ3_S + IQ3_XXS GGUF
fixture under `tests/DotLLM.Tests.Unit/Fixtures/` (the existing fixture
pattern), or (b) produce them inline in a per-host test via the GGUF
writer + IQ3 quantiser. Then mirror the existing per-quant test pattern
in each of the 4 hosts (~4 new test files, ~150 LoC each).

**Estimate:** 1 session (~3-4h). Possible to defer the host-tests until
a real Qwen3.6-A3B IQ3 GGUF is available — but the cross-backend mandate
argues for the synthetic fixture now.

### H4. ForwardBatch parity tests miss LoRA / MLA / MoE / quantized-KV scenarios

**Where:** `tests/DotLLM.Tests.Unit/Models/Architectures/TransformerModelForwardBatchTests.cs`.

**Problem:** Commit `479c23f` claims "LoRA / MLA / MoE / quantized KV caches
flow through correctly because the per-seq RunLayersAndFinalNormCore call
uses the unchanged layer-loop code path. Adapter scoping is per-seq (set/
clear inside the loop) so heterogeneous adapters work." The four parity
tests in tree are dense-Llama + SmolLM-135M Q8_0 only. There is no test
that:

1. Drives ForwardBatch with two sequences using *different* LoRA adapters
   (to verify per-seq Set/Clear).
2. Drives ForwardBatch on an MLA model (DeepSeek-V2-Lite path).
3. Drives ForwardBatch on a MoE model (Mixtral / Qwen-MoE / Granite-MoE).
4. Drives ForwardBatch with a paged + quantized KV-cache.

The risk is asymmetric: per-seq Set/Clear of `Adapter` could leak across
sequences in subtle ways (e.g. cached scratch buffers indexed by adapter
handle), and an MLA latent-cache that's *snapshot* into the stacked
hidden buffer may have stride assumptions that the per-seq path masks.

**Next step:** Add 2-4 targeted parity tests using either real cached
fixtures (Phi-3.5-mini for MoE-adjacent, TinyLlama for LoRA, V2-Lite for
MLA — all already used elsewhere as cached fixtures and skip-gated) or
synthetic fixtures. Keep each test under 50 LoC by extracting the
"two-seq Forward vs ForwardBatch" helper from the existing tests.

**Estimate:** half-day (~3h) for adapter + MoE; full day if MLA needs new
fixture work. Note: Phase 5b (parallel agent) is the *bigger* batched-fusion
work; this audit item is just the parity-test breadth for the lm_head-only
fusion that already shipped.

### H5. `docs/SCHEDULING.md` Kernel-Batched Forward section pre-dates Phase 5a

**Where:** `docs/SCHEDULING.md:48-67`.

**Problem:** The section says "The default interface implementation loops
over `Forward` per request — backends pay the per-sequence kernel-dispatch
overhead until they override with a fused implementation." It then bullets
override candidates ("CPU: bundle N sequences into a single GEMM (currently
N GEMVs)") and notes that the existing `FourConcurrentSchedulerTests`
verifies the API contract "even when the underlying backend is still using
the per-sequence-loop fallback".

This is now stale: Phase 5a (`479c23f`) shipped a CPU override
specifically for the lm_head GEMM (one batched dispatch at Σ N_i instead
of N small GEMMs). The doc should mention this as the first concrete
override and link to `TransformerModel.cs`'s `ForwardBatch` override.

**Next step:** Update §`Kernel-Batched Forward` to note "CPU override:
Phase 5a (lm_head batched GEMM) shipped; intra-block matmul fusion (Phase
5b) tracked separately." Document the Phase 5e Vulkan finding (lm_head
saves ~150-350 µs per step on the dense host — not worth a dispatch
override; Vulkan's win waits for Phase 5f intra-block fusion).

**Estimate:** 20 min.

### H6. CUDA PTX stale relative to patched .cu sources

**Where:** `native/ptx/attention.ptx`, `native/ptx/attention_f32.ptx` vs
`native/kernels/attention.cu` (modified 2026-05-17 by `8e8c228`).

**Status check:** `ls -la native/ptx/attention*.ptx native/kernels/attention*.cu`
shows `.cu` files dated 2026-05-17 (commit `8e8c228`) and `.ptx` dated
2026-05-01 / 2026-05-13. Acknowledged in the commit message ("CUDA PTX
regen deferred — same pattern as the Q3_K cross-backend fix; needs an
nvcc host. Source is on-tree so any future CUDA build picks it up
automatically") and tracked in `.planning/.continue-here.md` remaining-work
item 5.

**Why high-value:** Until PTX is regenerated, *any* CUDA model run with
`slidingWindow > 0` will silently use the buggy mask. The CUDA backend
ships the PTX, not the .cu; `CudaModule.LoadFromFile("attention.ptx")`
doesn't recompile. Mistral / Phi-3 / Gemma / Qwen sliding-window paths
on CUDA are all in this state.

**Next step:** First CUDA host wins — run `native/build.ps1` (or `.sh`)
with nvcc + bake the regenerated PTX into the next commit. The existing
PTX-regen pattern from `fdb39b4` (`gated_delta_net_scan.cu` →
`gdn_scan.ptx`) is the template.

**Estimate:** 15 min when a CUDA host is available.

## Stale state (housekeeping)

### S1. Eleven `.continue-here-*.md` files at repo root — most obsolete

Listing as of 2026-05-18 (relative to parent repo's working tree):

| File | Date | Status |
|---|---|---|
| `.continue-here.md` | May 1 | **OBSOLETE.** Belongs to a worktree, last_updated 2026-04-28. Superseded by `.planning/.continue-here.md`. Different content from parent's `.planning/.continue-here.md`. |
| `.continue-here-vulkan-fa.md` | May 14 | **UNTRACKED** (in git status). Mostly superseded by `docs/PERFORMANCE.md` §6.4 (Strix Halo numbers landed at `380e4b9`). The "Tuning headroom" section has not been captured elsewhere — folding that subset into PERFORMANCE.md §6.4 or `docs/ATTENTION.md` is the cleanest action. The "TODO: Strix Halo measurement" section can be deleted outright. |
| `.continue-here-iq3-dispatch.md` | May 15 | **OBSOLETE.** All four hosts wired (`07f391f`/`48d65fe`/`146d747`/`ad6b853`). The integration-test follow-up it specifies has not landed (see H3 above) but the file itself is no longer the right place to track that. Delete or migrate the integration-test TODO into a `.planning/notes/` entry. |
| `.continue-here-lora-final-mile.md` | May 15 | **OBSOLETE — actively misleading.** Premise was that Path B3 (R4 reuse) would close the −16% LoRA Q8_0 gap. `dd2892f` disconfirmed B3 empirically (~840 µs at canonical shape vs ≤470 µs acceptance bar). Should be deleted; the actual conclusion now lives at `.planning/notes/lora-q8-stage1-probe-results.md`. |
| `.continue-here-lora-macro-bench.md` | May 14 | Pre-Phase-4d.6 macro-bench plan. Superseded by `docs/LORA.md` Phase 4d.3/4d.6 sections + commits `ed2b6a0`/`b0959fa`/`b19642d`. Delete. |
| `.continue-here-lora-quantised-delta.md` | May 14 | Pre-Phase-4d quantised-delta plan. Phase 4d shipped at `9864bc6`. Delete. |
| `.continue-here-scheduler-batched-forward.md` | May 15 | Phase 5a shipped at `479c23f`; the ForwardBatch API + scheduler dispatch are documented in `docs/SCHEDULING.md`. The remaining-work pointer was migrated to `.planning/notes/forward-batch-impl-plan.md`. Delete. |
| `.continue-here-step35.md` | May 14 | Continuous batching MVP shipped. Delete. |
| `.continue-here-step38.md` | May 14 | Rate-limiting (Step 38) shipped, marked done in ROADMAP. Delete. |
| `.continue-here-step45.md` | May 14 | Metrics & tracing (Step 45) shipped, marked done. Delete. |
| `.continue-here-strix-perf.md` | May 14 | Vulkan-on-Strix-Halo baseline characterisation. The deliverable (PERFORMANCE.md §6) landed; this is the historical handoff. Delete. |
| `.continue-here-vulkan-host-mem.md` | May 15 | `VK_EXT_external_memory_host` zero-copy path (`bc474e7`/`841582c`/`6eeb884`/`7320897`/`fd2d7a3`/`2b9f390`/`9193582`) is in tree; doc landed at `docs/GPU.md`. Delete. |
| `.continue-here-vulkan-iq1.md` | May 14 | IQ1_S Vulkan kernels in tree. Delete. |
| `.continue-here-vulkan-lora-fused.md` | May 14 | Vulkan LoRA fused path (`cf9dfc2`/`731c53c`/`c578046`) shipped. Delete. |

**Action:** Single janitorial commit to delete 11 obsolete files + fold the
`.continue-here-vulkan-fa.md` tuning-headroom paragraph into `docs/ATTENTION.md`
or `docs/PERFORMANCE.md` §6.4, then delete that one too. The parent's
`.planning/.continue-here.md` is the current source of truth.

### S2. `HANDOFF.json` is 4 days stale and mis-frames the session

**Where:** `.planning/HANDOFF.json` (last updated 2026-05-14T07:19:58Z, head_commit
`433df26`).

**Problem:** The file describes the GDN head-broadcast bug-hunt session (which
shipped at `fdb39b4` / `6354676` / `0524276` / `21b3c30`). Since then:
- 13 commits on `feature/qwen3.6` (sliding-window fix, IQ3 dispatch x4,
  ForwardBatch CPU Phase 5a, LoRA Path B3 disconfirmed, etc.).
- The branch is at `1a97fd6`, not `433df26`.
- `remaining_tasks[13]` says "Quiescent re-measurement" still not started —
  unclear whether that's still desired given `docs/perf/baseline-qwen36-a3b-cpu.json`
  is the existing record.

The pattern in `.planning/.continue-here.md` (the active handoff) is much
more up to date. Either:
- (a) regenerate `HANDOFF.json` from the current state at `1a97fd6`, or
- (b) delete `HANDOFF.json` and let `.planning/.continue-here.md` be the
  single source of truth.

Looking at the structure, `HANDOFF.json` appears to be a machine-readable
sibling of `.continue-here.md` produced by some now-obsolete agent
workflow — there's no indication anything reads it programmatically.
Option (b) is cleanest.

**Action:** Confirm `HANDOFF.json` is not consumed by any tooling (grep
`HANDOFF.json` across `tools/`, `scripts/`, `.claude/` — preliminary check
shows no consumer). Delete it. If preserved, regenerate.

### S3. `.planning/notes/lora-q8-stage1-investigation.md` status field is stale

**Where:** `.planning/notes/lora-q8-stage1-investigation.md` frontmatter:
`status: research`, `date: 2026-05-17`.

The companion file `lora-q8-stage1-probe-results.md` (status `negative_result_B3`,
date 2026-05-18) carries the conclusion. The investigation note should
add a closing reference or update its status to `disconfirmed`. Cheap
change; keeps the audit trail navigable.

### S4. `.planning/notes/forward-batch-impl-plan.md` status `drafting` post-Phase-5a

**Where:** `.planning/notes/forward-batch-impl-plan.md:3` `status: drafting`.

Phase 5a shipped at `479c23f`; Phase 5e analysed and deferred at `9895ca3`.
The doc itself still says "drafting". Either:
- update to `status: phase-5a-shipped-phase-5b-active` (matches parallel agent), or
- archive after Phase 5b/5f ship in parallel worktrees.

### S5. `docs/LORA.md` does not document the Path B3 disconfirmation

**Where:** `docs/LORA.md` lines 218-240 still cite the Phase 4d.4 narrative
that frames the −16% LoRA Q8_0 prefill gap as gated on upstream `Avx512Vnni.V512`.
The Path B3 (`OuterProductGemmQ8_0` reuse) hypothesis is not in the doc;
its disconfirmation at `dd2892f` is also not in the doc.

`.planning/notes/lora-q8-stage1-probe-results.md` recommendation 2 explicitly
calls this out: "Update `docs/LORA.md` Phase 4d.5 with the empirical Stage
1 verdict — currently the doc cites the historical kernel-probe number
(0.97 ms) and frames the −16% gap as gated on the V512 PR. The new
number (770 us) and the B3 negative result narrow the gap slightly but
don't close it."

**Action:** Add a "Phase 4d.7 — Path B3 probe and disconfirmation" subsection
to `docs/LORA.md` between 4d.6 and the Vulkan-backend section. Reference
the probe results note and the commit (`dd2892f`). ~30 min.

### S6. `docs/SUPPORTED_MODELS.md` says IQ3 per-host dispatch is "follow-up"

**Where:** `docs/SUPPORTED_MODELS.md:240-241`:

> Upload-path predicates land in `VulkanWeights`; per-host matmul dispatch
> is the follow-up step (see `.continue-here-iq3-dispatch.md`).

Per-host dispatch shipped across 4 hosts at `07f391f` / `48d65fe` /
`146d747` / `ad6b853`. The doc is stale and points at a now-obsolete
`.continue-here-*.md` file. Two-line fix.

### S7. Phase 10 README News entry understates session-2 work

**Where:** `README.md:675` (the "2026-05" News bullet).

The bullet describes Phase 10 Qwen3MoeHybrid landing (Gated DeltaNet,
MoE, 1981 unit tests). It doesn't mention:
- The sliding-window mask fix that landed afterwards.
- IQ3 family Vulkan support (kernels + per-host dispatch).
- FA Vulkan shader.
- ForwardBatch CPU Phase 5a.
- Strix Halo Vulkan baseline + FA microbench.

The Roadmap-table column for Phase 10 still says "Real-GGUF GPU parity
tests pending" which is true (CUDA 30 GB constraint), but the entry
otherwise undersells the post-Phase-10 throughput / quant-coverage work.

**Action:** Add a follow-up 2026-05 News bullet covering the Strix-Halo
session (or merge into the existing one). One paragraph, ~10 min.

## Open follow-ups not tracked

Items mentioned in commit messages or notes that are not in
`.planning/.continue-here.md` `<remaining_work>` and not in any open
GitHub issue I can verify from here:

1. **CUDA `attention_f16` parity tests.** Never existed; the F16
   sliding-window fix has no test surface at all. See C1.
2. **CUDA `AttentionF32` sliding-window discriminator.** Not noted in
   commit `8e8c228`'s description as a follow-up (the commit only mentions
   PTX regen). Should be added to the same regen window. See C1.
3. **IQ3 per-host integration tests** (4 transformer hosts × 2 quant
   types). Mentioned in `.continue-here-iq3-dispatch.md` recommendation
   but not propagated to `.planning/.continue-here.md` after that file
   went obsolete. See H3.
4. **ForwardBatch parity test breadth.** Adapter / MLA / MoE / quantized-KV
   scenarios. Implicit in commit `479c23f` ("LoRA / MLA / MoE ... flow
   through correctly") but no test exercises the claim. See H4.
5. **HIP backend status.** `src/DotLLM.Hip/HipTransformerModel.cs` is a
   documented stub (throws `NotImplementedException` pointing at
   `docs/HIP.md`); `native/hip/kernels/` contains only `rmsnorm.hip`.
   None of the recent kernel work (sliding-window, IQ3, FA, GDN, MLA)
   has a HIP counterpart and there's no roadmap step. *Audit verdict:
   leave alone* — HIP is explicitly out of scope until Strix-Halo Vulkan
   matures, and the cross-backend mandate from CLAUDE.md only applies to
   ported kernels. Just confirming HIP isn't a hidden regression.
6. **`TODO(step-35)` markers in `src/DotLLM.Telemetry/EngineTelemetry.cs:90,97`**
   — wire `IScheduler.QueueDepth` / paged-KV-allocator usage now that
   Step 35 has landed. Two-line update; gated on Step 35 being marked
   done first (H1).
7. **The `2b9f390` cold-load zero-copy bench result** — `bench:
   cold-load zero-copy vs staging` ran but I see no PERFORMANCE.md
   section capturing the result. Either the numbers landed and I missed
   them or this is an unreported measurement worth ~15 min of doc work.

## Recommended next session priorities

Ranked by impact/effort ratio:

1. **Doc + housekeeping pass** (~1.5 h, no hardware required). C1 deferred.
   - H1: ROADMAP step-status fixes + README roadmap-table recount.
   - H2: PERFORMANCE.md §6.4/6.5 renumber.
   - H5: SCHEDULING.md ForwardBatch override note.
   - S1: Delete 11 stale `.continue-here-*.md`; fold the FA tuning-headroom
     paragraph into ATTENTION.md.
   - S2: Delete (or regenerate) HANDOFF.json.
   - S3/S4: Frontmatter touch-ups on two `.planning/notes/` entries.
   - S5: LORA.md Phase 4d.7 disconfirmation subsection.
   - S6: SUPPORTED_MODELS.md IQ3 per-host wiring update.
   - S7: README News bullet for the Strix-Halo session.
   - Single commit, low-risk, unblocks the next agent's context.

2. **C2: Fix `ForwardBatch_EmptyRequests_ReturnsEmpty` `[Fact]` mis-use**
   (~10 min, no hardware required). Trivial; could be folded into priority 1.
   But it's a real failure-vs-skip bug on a clean CI host.

3. **H3: IQ3 per-host parity tests** (~3-4 h, Vulkan host required).
   Cheapest insurance against the per-host dispatch regression. Cover
   all 4 hosts (dense / Qwen3MoeHybrid / NemotronH / Mamba3) with one
   synthetic IQ3_S + IQ3_XXS fixture per host. Closes the kernel-test ✓
   without model-test ✓ gap that the GDN head-broadcast bug taught us
   about.

4. **C1: CUDA sliding-window discriminator + missing AttentionF16 tests**
   (~1 h tests + 15 min PTX regen, CUDA host required). Best to batch
   with H6 PTX regen and any other CUDA-host work that piles up. The
   tests can be authored without a CUDA host (just won't run in CI
   until merged).

5. **H4: ForwardBatch parity test breadth** (~half-day, no hardware
   required for the LoRA case; cached real-weight fixtures for the
   MoE/MLA cases). Best done after Phase 5b lands (parallel agent) so
   the test surface covers the full batched path, not just lm_head.

Items 1+2 are pure documentation/housekeeping and could ship in a single
~2 h session with no hardware. Items 3+4 share the "next time we have a
GPU host" gate; if a CUDA host appears, do 4 first (covers the patched
.cu source which is the riskier surface). If only a Vulkan host
appears, do 3.

## Notes on items NOT covered (parallel agents)

For traceability — these were explicitly de-scoped per the audit brief
and not investigated:

- **Phase 5b CPU intra-block matmul fusion** — separate worktree.
- **Phase 5f Vulkan intra-block matmul fusion** — separate worktree.
- **`HybridPrefillDecodeBenchmarks` BDN discovery skip** — separate
  worktree. (Brief mention of the symptom is in PERFORMANCE.md §6.5
  "Outstanding measurements" which I read but did not assess.)
- **Gemma 3 prerequisite revival from reflog** — separate worktree.
  (`gemma3-step57-config-pending` / `gemma3-step57-geglu-pending` tags
  noted in `.planning/.continue-here.md`; manual merge plan at
  `.planning/notes/gemma3-merge-audit.md`.)

If any of those parallel agents shipped doc updates while this audit
was running, the doc-pass in priority 1 above will need to be
double-checked for conflicts.
