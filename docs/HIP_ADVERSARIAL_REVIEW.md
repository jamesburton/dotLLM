# Adversarial integration review prompt

Copy the prompt below into a new agent session and replace the path placeholders.
It requests an audit and a local report, not publication or remediation. The
reproduction guide is [HIP_TESTING.md](HIP_TESTING.md). Prior measurements are
claims to verify, not a passing verdict for the reviewed revision.

---

You are independently reviewing the integration of BlueCurve's ROCm/HIP backend
and selected SafeTensors improvements into `jamesburton/dotLLM`. Find concrete
correctness, resource-lifetime, compatibility, numerical and test-evidence defects.
Challenge assumptions and demonstrate failures; do not invent findings to satisfy
an adversarial tone. This is a bounded review of the integration and its affected
contracts, not a review of every pre-existing backend feature.

## Repository identity and starting context

| Role | Local path | Remote and reference |
| --- | --- | --- |
| Review checkout | `<TARGET_ROOT>` | Integration PR branch from `https://github.com/BlueCurveCorp/dotLLM`, or merged `https://github.com/jamesburton/dotLLM` target `dev` |
| Submission base | same checkout/history | `02dca05f71aa61fa5da02f912442fd6a7fa4d2dd`; resolve the actual PR base and prerequisite branches before choosing a diff |
| Original integration provenance | optional historical checkout/history | Original branch `issue/976-hip-bluecurve-integration`; baseline `35aaf18dc75e8172afd16284acdfcdfc99066d3b`, qualified through `26fc9c4` |
| Donor | `<DONOR_ROOT>` | `https://github.com/BlueCurveCorp/dotLLM`; inspected `main` at `debb6eb61bc2f21ee072932f7c0b4d157f6cfb66` |
| Upstream context | `<UPSTREAM_ROOT>` if present | `https://github.com/kkokosa/dotLLM`, [issue #125](https://github.com/kkokosa/dotLLM/issues/125) |
| Report/artifacts | `<REVIEW_OUTPUT>` | Fresh ignored or external scratch directory; never model storage inside the repository |

The seven submission slices cover mapped BF16, HF templates/tokenization, packed
CPU formats, Windows HIP, application/release wiring, HIP SafeTensors formats,
and Linux/qualification documentation. Dependent PRs may include prerequisite
commits until those land; audit both the complete integration and the selected
slice without counting prerequisites as new changes twice.

The original integration first reached `e2de144`; format support, Linux loading and results
documentation followed in `c0e7bcc`, `a6ae467`, `317c5b8`. Later changes add the Q8
acknowledgement gate and portable qualification harness. **Resolve the current
target HEAD and working diff yourself**; do not review only `317c5b8` or presume
that these commits represent the latest state. Resolve donor HEAD too, preserving
the inspected donor pin as a separate comparison point.

Read each repository's `AGENTS.md`/`CLAUDE.md`, the upstream issue discussion and
target [HIP issue #976](https://github.com/jamesburton/dotLLM/issues/976). If local
instructions require tracking, continue/create the review issue and local plan
under that workflow. Do not publish a review, send messages, push, merge or create
a PR unless separately instructed. Do not assume private `.docs/` history exists.

Establish graph project/generation and coverage if MCP graph tools are available;
use exact source for missing/stale/partial ranges. If unavailable, say so and use
bounded source searches. Do not claim graph verification you did not perform.
Capture remotes, revisions, status and the initial user diff before running tests.
Preserve unrelated edits, including any existing `.gitattributes` changes.

## Intended integration contract

This was a selective adaptation, not a wholesale merge. Preserve target interfaces,
CLI/server behavior, ownership, tensor formats and platform boundaries. The donor
did not supply a complete usable implementation for every imported idea; in
particular, compare actual kernel bodies and packed-format paths, not just class
names or feature lists. Identify both omitted useful donor behavior and incorrect
adaptations. Distinguish intentional documented limitations from regressions.

Current advertised scope is explicit `hip[:N]` for `run`, `chat`, `serve`; `auto`
does not select HIP. Dense Llama GGUF matrices accept F16/Q4_0/Q8_0. HF supports
F16/BF16 and the validated AWQ GEMM4 / legacy GPTQ4 subset. Quantized matrices and
KV remain packed; BF16 weights retain their bits. Activations/scratch are F16,
reductions F32 registers, and the existing final CPU-sampling result is F32.
F16 KV is default. Selecting either Q8 KV plane requires the exact process flag
`DOTLLM_HIP_ALLOW_Q8_KV=1`; it acknowledges known loss and does not select Q8 by
itself. Q8 model weights and non-HIP backends must remain unaffected.

Windows loading uses DirectStorage into a shared D3D12 heap imported by the matching
HIP device. Linux loading directly resolves `hipAmdFileRead` and uses AIS on a
compatible volume; it must fail closed instead of bouncing weights through host
RAM. There is no claim that `bench`, `perplexity`, `embed`, arbitrary HF formats,
other model architectures or all enumerated AMD devices are supported.

## Evidence to challenge

Read tracked `docs/HIP.md`, `docs/QUANTIZATION.md`, `docs/PRECISION.md` section 9,
`native/hip/README.md`, `docs/HIP_TESTING.md`, and the actual tests/scripts. Treat
all counts, hashes and results below as historical claims needing attribution:

- Windows RX 6900 XT/gfx1030, HIP 6.4.2: bounded integration selection reported
  734 passed / 27 skipped / 0 failed, with 122 Windows-applicable HIP checks.
  This was not the entire repository suite. Skips included optional fixtures and
  Linux-only methods; inspect current TRX, guards and actual executed test names.
- Original BF16/AWQ/GPTQ short-model tests reported 17/17 focused passes,
  independent pinned references and real-data wrong-layout/file-range controls.
  Reference generation uses pinned producer decoding plus HF eager inference;
  it is independent of dotLLM but still deserves oracle scrutiny.
- A real 8,192-input SmolLM2 F16 GGUF history scored 128 next-token rows at each
  1K/2K/4K/8K boundary (512 rows total). F16 passed max relative L2 < .02 and
  KL < .002; Q8 failed both gates at every boundary. CPU Q8 also failed. That
  does not prove a HIP bug, prove absence of one, or establish general Q8 quality.
- The original 8K run used frozen `e2de144` artifacts. A later kernel replay
  covered 1K and matched aggregate metrics; raw baseline logits were not retained.
  Do not upgrade this into bitwise-logit identity or final-revision 8K evidence.
- At 8K, reported owned F16/Q8 KV bytes were 188,989,440 / 100,515,840; reset and
  disposal counters reached zero. Driver-wide allocation behavior is separate.
- Linux ROCm 7.2.4 compiled kernels and published a CLI. The 54-check Linux loader
  probe used a CPU-backed simulated HIP runtime and actual POSIX direct reads.
  It did not test physical AIS or Linux GPU numerics. Existing Windows-only
  model-test guards cannot be counted as Linux passes.
- Original BF16/AWQ/GPTQ Windows run/HTTP smoke and package checks passed. A
  generated Paris answer is smoke, not a numerical or long-context oracle.

## Review procedure

1. Derive the current integration diff against the target baseline, including
   staged, unstaged and relevant new files. Compare relevant donor source/history
   against the same concerns. Build a small requirement-to-target-to-test map.
   Explain any missing baseline or inaccessible source instead of guessing.
2. Trace the affected loading, compute, KV, CLI/server and release paths. Read
   implementations and callers, not just tests/documentation. Keep source line
   references tied to the reviewed revision/working state.
3. Prioritize these checks:

   - **SafeTensors and producer formats:** real BF16 reads, F32 norm narrowing,
     exact dtype/shape/offset/byte-count validation, checked overflow and malformed
     headers/configs; AWQ output nibble permutation, group sizes/scales/zeros;
     GPTQ K packing, stored-zero-plus-one, `sym`, `desc_act`, nonidentity `g_idx`,
     duplicate/missing groups; zero-only biases, tied heads and shards. Verify
     physical bytes versus logical parameter counts, retained mappings and
     source lifetime through replacement/unload. Reject unsupported layouts
     explicitly, without relabeling them GGUF Q4_0 or silently widening weights.
   - **Direct loading:** Windows LUID-to-DXGI identity, DirectStorage completion
     and error records, external-memory ownership and ordering; Linux extension
     ABI and function-pointer signatures, statx/statfs layouts, O_DIRECT/P2PDMA
     prerequisites, alignment, exact tensor displacement, EOF tails, >32 MiB
     chunks, overflow, partial reads and both HIP/storage status errors. Exercise
     cleanup after allocation/import/read failures, independent buffer/session
     lifetime, repeated disposal, freeing the base allocation rather than a
     shifted view, and no hidden host-copy fallback.
   - **Compute/precision:** original packed components on device, F16 storage
     and register accumulation, no matrix/history expansion, no score scratch
     hidden behind a helper; head grouping, RoPE row order/scales, strides,
     partial pages and nondegenerate geometry. Separate the final sampler F32
     row from resident GPU storage. Examine numerical errors per row, not only
     average PPL or top-1 counts.
   - **KV and concurrency:** actual history advancement, request isolation,
     page-table bounds and lazy growth, rollback/reset/disposal, cancellation,
     repeated create/free, serialized scratch access, selected-device ownership,
     memory admission and failure cleanup. Test mixed K/V combinations. Verify
     Q8 requires opt-in through every high-level route and direct library cache
     creation, rejected values do not enable it, F16 still works without it,
     warning text matches known loss, and Q8 weights/other backends are unchanged.
   - **Applications/capabilities:** source ownership, HF tokenizer/chat-template
     semantics including tool templates, run/chat/serve routing, explicit HIP
     errors without CPU fallback, `auto` order, unsupported options, readiness
     versus mere enumeration, server device ordinal and lifecycle. Check affected
     CPU behavior and relevant shared contracts for regressions.
   - **Build/package/platform:** code objects contain required entry points and
     the intended architectures; match source/build/publish/package hashes.
     Confirm resolver names and missing-image diagnostics, HIP assembly/native
     payloads, and absence of redistributed DirectStorage DLLs. Separate Linux
     build success, stub tests, primitive hardware execution, actual AIS and
     full-model parity. Enumerate Windows-only guards and optional fixture skips.
4. Run bounded meaningful tests from the handbook, building current sources.
   Record commands, environment, exit status, counts/skips, artifact provenance
   and actual backend. Use explicit external fixture paths and pinned hashes.
   Validate independent reference construction and run the portable context
   harness if resources permit. The known Q8 numerical exit 10 is not a runtime
   crash, but must not hide F16 failures or become a passing quality result.
5. Prove important correctness assertions can fail. Use plausible mutants or
   wrong-component substitutions on **real discriminating data**, e.g. wrong
   producer order/zero/group index, RoPE layout, tensor offset, ignored read
   status, shortened context, rotated vocabulary or bypassed Q8 gate. A surviving
   mutant is evidence of a test gap; explain why it matters. Restore/rebuild your
   isolated test copy and rerun the valid implementation before reporting.
6. If a critical defect duplicates logic in other backends, trace all affected
   backend copies as required by `CLAUDE.md` and report the joint fix scope.
   This audit does not authorize production remediation or a partial fix.

## Execution boundaries

Use a disposable review copy/sandbox or precisely owned scratch artifacts for
mutations. Preserve the user's working changes and donor/upstream trees. Never
use broad `git reset --hard`, `git clean`, or recursive deletion on shared paths.
Do not copy models into the repo or rewrite shared fixture files. Coordinate
GPU access; do not run several inference harnesses concurrently and then attribute
contention/OOM to a defect. Keep original and mutated artifacts distinct.

Do not install/reconfigure drivers, change host/VM GPU assignment, remount or format
storage, bypass permissions, or enable fallback I/O to make tests run. Document
missing hardware/access and continue independent review. Read-only administrator
diagnostics may be supplied by the system owner. Linux tests require the actual
GPU/model-volume path; a VM without it cannot establish that evidence.

Do not change production code to fix findings during the review. Leave only the
requested report/evidence and explicitly permitted test artifacts. Restore only
your own mutations; verify final status against the initial snapshot.

## Required report

Lead with a concise verdict and its scope. List verified defects by severity,
then evidence gaps and intentional limitations separately. For each actionable
finding include:

- Severity and concrete affected behavior/impact.
- Target revision and `path:line` (donor/baseline references where relevant).
- Minimal reproduction or decisive source/call-chain evidence, expected versus
  actual behavior, and the relevant command/test artifact.
- Confidence, assumptions, and whether it was reproduced on hardware, through a
  stub, or established by source only.
- Smallest plausible correction and a discriminating verification requirement.

Include an evidence matrix covering source comparison, managed/native tests,
real checkpoint references, context quality, direct loading, applications and
packages for each tested platform. Give actual counts/skips and hashes. State
which claims were confirmed, contradicted or not tested; list remaining bounded
questions without presenting them as bugs. If no defect is verified, say so and
describe the tested scope and residual limits. Do not equate documented Q8 loss,
unavailable Linux hardware or a pre-existing unrelated failure with a new
integration defect without evidence.
