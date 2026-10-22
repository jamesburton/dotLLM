# Fresh re-profile of PQ2_0 GEMV family, post-#161 (F32-native), 2026-07-30/31

Follow-up to the 2026-07-22 advisor review (see `[[prismml-bonsai-model]]` memory) that found
the FFN gate+up/down GEMV kernels compute-bound (ALU pipe 78.2%/68.5%) and ranked "algebraic ALU
reduction in the ternary unpack loop" as an untried candidate — but that data predates #161's
F32-native rework, so it needed re-verification before committing to an implementation, per this
session's own repeated "verify before building" lesson (twice today, on the #199 attention thread).

## Methodology

`--kernel-name regex:"pq2_0_gemv.?_f32io$"` / `regex:"pq2_0_gemv.?_f32io_small$"`, `--depth 512`,
real Bonsai-27B. Verified by captured grid shape before trusting any number — one capture
(`pq2_0_gemv_f32io_small`, grid=15520) turned out to be a depth-seeding bulk-batch artifact, not a
real decode-shaped launch (discarded; real decode-shaped instances of that kernel are grid=384/640,
matching the 2026-07-27 baseline table almost exactly).

## Result: still the well-substantiated stopping point, occupancy-wise — but genuine compute headroom confirmed

| Kernel | Grid | Compute % | Memory % | Achieved Occ. | Duration |
|---|---|---:|---:|---:|---:|
| `pq2_0_gemv_f32io` (FFN gate/up, k=17408) | 320 | 75.10 | 75.10 | 71.41% | 77.4 us |
| `pq2_0_gemv_f32io_small` (attn/GDN, k=5120) | 384 | ~85 | ~85 | (matches 2026-07-27) | — |
| `pq2_0_gemv_f32io_small` | 640 | ~90 | ~90 | (matches 2026-07-27) | — |
| `pq2_0_gemv2_f32io_small` (fused) | 2176 | ~80 | ~97 | (matches 2026-07-27) | — |
| `pq2_0_gemv2_f32io_small` (GDN alpha/beta, tiny) | 6 | 4.12 | 5.87 | 16.65% | 21.5 us |

All real decode-shaped launches match the 2026-07-27 baseline table within a few points —
**#161's F32-native rework did not change this kernel family's occupancy profile**, consistent
with that PR's own documentation (occupancy-binding constraint unchanged: same register/shared-mem
tier). Confirms this is still the "well-substantiated stopping point" from the original PQ2_0
investigation for occupancy/coalescing levers specifically.

**But**: the dominant FFN kernel (grid=320, by far the largest share of decode time) sits at
75.10%/75.10% compute AND memory throughput simultaneously — comfortably above ncu's own "<60% =
latency issues" flag (i.e., NOT occupancy-bound), but not saturated either. This is exactly the
regime the 2026-07-22 advisor review's ALU-reduction candidate targets: real, measured ALU cost in
the ternary unpack loop, not a phantom lever. **Validated as a legitimate next target** — dispatched
as a fresh implementation attempt (see git history / issue tracker for the outcome).

Raw `.ncu-rep` files kept local only (~30MB), not committed.

## Outcome (2026-07-31, issue #244)

**Correction**: this README's framing was wrong — the ALU-reduction candidate it "validated as a
legitimate next target" had already been implemented and shipped nine days earlier, 2026-07-22,
commit `7c7101c` ("algebraic ALU reduction in PQ2_0 GEMV decode loop (#161)"). The 75.10% compute
figure measured above is headroom REMAINING AFTER that identity, not evidence it was still unapplied
— this session profiled the already-reduced kernels without checking `pq2_0_gemv.cu`'s own source
first. See that file's `#244: re-investigated, found ALREADY IMPLEMENTED — no code change` section
for the fresh re-verification (correctness + benchmark) done in place of a redundant reimplementation.
Issue #244 should be closed as a duplicate of #161's already-merged work, not left open awaiting a
fresh implementation that would just recreate what's already there.
