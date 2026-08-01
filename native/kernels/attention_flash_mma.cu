// Hand-fused FP16 flash-attention prefill kernel using Ampere tensor cores
// (mma.sync.m16n8k16). Prototype for the long-context go/no-go: the s x s scores
// are computed, softmaxed, and consumed entirely in shared memory / registers and
// are NEVER materialised to global memory (unlike the G3 cuBLAS+softmax path which
// round-trips numHeads * s^2 FP16 through HBM). One warp per (query head, 16-query
// tile); the KV axis is streamed in tiles of 16 keys with online softmax.
//
// WHERE THE MEASURED WIN COMES FROM (important — it is NOT just the round-trip):
//   This kernel sweeps KV tiles only up to a query tile's own diagonal
//   (kv_end = min(seq, q0+Q_TILE)), so over all query tiles it does ~HALF the
//   matmul FLOPs — only the causal triangle. The G3 cuBLAS path and the
//   "GEMM-only floor" both run a DENSE full s x s QK GEMM and zero the upper
//   triangle afterwards in softmax. So the GEMM-only floor is NOT a true lower
//   bound for this kernel (flash computes fewer FLOPs than the floor). On the
//   3060 at s=4096: floor(full square)=14.5ms => a causal-triangle GEMM at cuBLAS
//   efficiency would be ~7ms, yet flash is ~22.7ms — i.e. this untuned 1-warp/
//   block kernel's per-FLOP MMA throughput is ~3x worse than cuBLAS. Flash still
//   beats shipped-G3 (1.3-1.69x) by COMBINING causal work-reduction with score
//   fusion, DESPITE poor MMA utilisation. A tuned kernel (more warps/block,
//   double-buffered K/V loads) has substantial further headroom. NB a causal-aware
//   G3 (block-triangular GEMMs) would capture part of this win without hand-MMA.
//
// SCOPE (prototype, Llama-3.2-1B head shape):
//   headDim == 64 (4 k-steps of k16), causal, position_offset == 0, FP16 in/out.
//   GQA handled by the caller's grid (query head hq -> kv head hq / group). Output
//   is written ROW-MAJOR [seq, numHeads, headDim] to match attention_f16 exactly so
//   parity is a direct linear compare.
//
// Layout / fragment notes:
//   QK:  S[16q x 16k] = Q . K^T.  mma.m16n8k16 computes D[m,n] = sum_k A[m,k]*B[n,k]
//        with A row-major [m=16 x k=16], B row-major [n=8 x k=16]. Feeding A=Q rows,
//        B=K rows gives exactly sum_d Q[q,d]*K[k,d]. A 16x16 S tile = 2 n-subtiles
//        (keys 0..7, 8..15) x 4 k-steps (d 0..15,16..31,32..47,48..63) = 8 mma ops.
//        Q/K are read in their natural row-major [token][d] layout via ldmatrix.
//   PV:  O[16q x 64d] = sum_k P[q,k]*V[k,d].  mma needs B[n,k] = V[k,n] i.e. V
//        TRANSPOSED to [d][key] in shared. O = 8 n-subtiles (d 0..7,...,56..63)
//        x 1 k-step (the 16 keys of this tile), accumulated across KV tiles.
//   C/D accumulator map (m16n8k16, per PTX ISA): lane L holds 4 f32:
//        c0,c1 -> row (L>>2),    cols (L%4)*2 + {0,1}
//        c2,c3 -> row (L>>2)+8,  cols (L%4)*2 + {0,1}
//   We spill S out of the QK accumulator into shared via this map, so per-row
//   softmax max/sum is a plain shared reduction (no cross-lane shuffle), and the
//   normalised P sits row-major in shared ready as the PV A-operand. The same row
//   map rescales the O accumulator (O shares the query-row axis with S).

#include <cuda_fp16.h>
#include <float.h>

#define HEAD_DIM 64
#define Q_TILE 16
#define KV_TILE 16

// Shared-memory row-stride padding (bank-conflict fix, 2026-08-01). HEAD_DIM (64 halves =
// 128 bytes = exactly 32 banks) and KV_TILE (16 halves = 32 bytes = 8 banks) both divide the
// 32-bank cycle evenly, so every row of sK/sQ/sVt/sP lands in the same bank phase as the row
// before it -- the classic power-of-two-stride conflict. `ncu --set full` on this kernel
// (s=1024, .perf-runs/flash_mma_s1024_details.txt) measured this directly: 9.7-way shared-load
// bank conflicts (84% of load wavefronts, Est. Speedup 77.4%) and 4.0-way shared-store
// conflicts (72% of store wavefronts, Est. Speedup 66.4%) -- while DRAM throughput was only
// 2%, ruling out actual bandwidth as the bottleneck.
//
// Padding every row stride breaks the alignment -- the exact technique already proven in this
// codebase's I2_S GEMM kernel (MatMul.I2S.cs: 128-word stride -> 129) -- BUT sK/sQ/sVt/sP are
// read via `ldmatrix`, which requires each per-thread row address to be 16-byte aligned (CUDA
// error 716 "misaligned address" if violated -- confirmed the hard way: a naive +1-halfword
// pad breaks this, since `(lane & 15) * stride` must stay a multiple of 8 halfwords for every
// lane value, which only holds if the stride itself is a multiple of 8). So these four pad by
// +8 elements (stays a multiple of 8 halfwords => 16-byte-aligned rows; new stride not a
// multiple of the 128-byte/32-bank cycle either). sScore is float and read only via plain
// per-lane scalar loads (no ldmatrix), so it only needs its usual 4-byte alignment -- pads by
// +1 like the I2_S precedent. Combined cost across all five arrays at MAX_GROUP_WARPS=4 is a
// few hundred bytes -- negligible next to the 5-blocks/SM shared-mem occupancy ceiling.
#define HEAD_DIM_PAD (HEAD_DIM + 8)   // sK / sQ row stride (72 halves, ldmatrix-safe)
#define KV_TILE_PAD (KV_TILE + 8)     // sVt / sP row stride (24 halves, ldmatrix-safe)
#define KV_TILE_PAD_SCORE (KV_TILE + 1) // sScore row stride (17 floats, plain scalar access)

// ldmatrix.x4: load four 8x8 FP16 matrices from shared into a warp's fragment.
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem_ptr)
{
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

// ldmatrix.x2: load two 8x8 FP16 matrices.
__device__ __forceinline__ void ldmatrix_x2(unsigned (&r)[2], const void* smem_ptr)
{
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr));
}

// D[16x8] += A[16x16] * B[8x16]^T  (row.col), f32 accumulate.
//   a[0..3]: A fragment (4 x b16 regs, 8 fp16 per lane)
//   b[0..1]: B fragment (2 x b16 regs, 4 fp16 per lane)
//   d[0..3]: f32 accumulator (in/out)
__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// Max query heads per kv head we statically size the block's shared arrays for. The
// kernel launches `group` warps per block; if a model's GQA group exceeds this the caller
// falls back to G3 (CanUse caps group). Llama-3.2-1B: group=4. Kept tight (==4) so the
// per-warp shared arrays (sQ dominates at 2KB/warp) don't bloat past what occupancy can
// afford on sm_86 — oversizing to 8 left half the shared block dead and capped blocks/SM.
#define MAX_GROUP_WARPS 4

// TUNED: one BLOCK per (kv head, 16-query tile); `group` WARPS per block, one warp per
// query head sharing that kv head. All warps in the block process the SAME q-tile and the
// SAME kv head, so they stream the SAME K/V tiles with the SAME causal extent (kv_end
// depends only on q_tile, not head) — K/V is loaded ONCE per block into shared and reused
// across all `group` warps (group× fewer K/V global loads), and the extra warps give the
// Ampere MMA pipeline the latency hiding the 1-warp/block prototype lacked.
//   gridDim.x = numKvHeads, gridDim.y = ceil(seq / 16), blockDim.x = group * 32.
//   K/V load → compute barrier is __syncthreads() (cross-warp); QK / softmax / PV stay
//   per-warp under __syncwarp(). Per-warp state (sQ/sScore/sP/sM/sL/sCorr/O_frag) is
//   warp-indexed; sK/sVt are block-shared.
//
// 2026-08-01 — SHARED-MEMORY BANK-CONFLICT FIX, real measured win, supersedes the "double-
// buffered K/V loads" headroom guess above. `ncu --set full` at s=1024 (real Llama-3.2-1B
// shape, .perf-runs/flash_mma_s1024_details.txt) found the actual bottleneck was NOT latency/
// occupancy: DRAM throughput was only 2% (ruling out bandwidth) while shared LOADS showed a
// 9.7-way bank conflict (84% of load wavefronts, Est. Speedup 77.4%) and shared STORES a
// 4.0-way conflict (72% of store wavefronts, Est. Speedup 66.4%). Root cause: HEAD_DIM (64
// halves = 128 bytes = exactly 32 banks) and KV_TILE (16 halves/floats = 8/16 banks) both
// divide the 32-bank cycle evenly, so sK/sQ/sVt/sP/sScore's row strides put every row in the
// same bank phase. Fixed by padding every row stride (HEAD_DIM_PAD/KV_TILE_PAD/
// KV_TILE_PAD_SCORE below) — same technique as this codebase's I2_S GEMM kernel (128-word
// stride -> 129). One real subtlety: `ldmatrix` requires 16-byte-aligned per-thread row
// addresses, so sK/sQ/sVt/sP (all ldmatrix-accessed) pad by +8 elements (stays a multiple of
// 8 halfwords), not the naive +1 that immediately hit CUDA error 716 "misaligned address" on
// real hardware. sScore is float and only ever read via plain per-lane scalar loads (no
// ldmatrix), so it safely pads by +1 like the I2_S precedent.
//
// Measured (real interleaved A/B via CudaTensorCoreAttentionParityTests.
// CompletePathTimingVsAttentionF16, DOTLLM_CUDA_ATTN_BENCH=1, RTX 3060, Llama-3.2-1B shape):
// G3/flash speedup roughly DOUBLED to TRIPLED — 1.3-1.69x pre-fix -> 3.65x @ s=1024, 3.61x @
// s=2048, 2.96x @ s=4096 post-fix. Also closed most of the gap to the GEMM-only floor (the
// "~3x worse than cuBLAS per-FLOP" claim above): floor/flash is now ~1.4-1.7x (was implied
// ~3x), i.e. this kernel moved from ~33% to ~58-70% of cuBLAS's raw per-FLOP efficiency at
// this shape, purely from fixing shared-memory access pattern, no algorithmic change.
// Correctness: FlashMmaPath_MatchesAttentionF16 passes bit/tolerance-clean at s=512/1024/2048
// (5e-3 abs-or-rel), full CUDA suite (401+ tests) green, zero regressions.
//
// Build-system gotcha found along the way, fixed separately in DotLLM.Cuda.csproj: this file
// (and its decode sibling attention_flash_mma_decode_gqa_split.cu) needs compute_86 for its
// mma.sync instructions, but the MSBuild CompileCudaPtx target had no per-file arch override
// (unlike native/build_ptx.bat's ARCH_86 list) — a plain `dotnet build` after any edit to this
// file silently regenerated wrong-arch (compute_75) PTX. That doesn't fail the build; it fails
// SILENTLY at runtime (CudaKernels.cs's best-effort module load catches the JIT error and just
// leaves HasAttentionFlashMma false, falling back to G3 with no visible symptom). Added the
// same per-file %(CudaKernel.Arch) override to the csproj so plain `dotnet build` now matches
// build_ptx.bat's output.
//
// Post-fix `ncu` confirmation (same day): a second elevated `ncu --set full` capture on this
// exact kernel/shape directly confirms the mechanism, not just the wall-clock effect. Shared
// LOAD bank conflicts (9.7-way, 84% of wavefronts pre-fix) are now fully gone -- `ncu` no
// longer flags them at all. Shared STORE conflicts improved but did not fully clear: 4.0-way
// (72% of wavefronts) -> 2.5-way (60%), Est. Speedup dropped 66.4% -> 29.3%. Duration this
// launch: 1.55ms -> 0.959ms (-38%). Checked whether a different pad size closes the remaining
// store gap: it can't -- `ldmatrix`'s 16-byte alignment forces the stride to stay a multiple
// of 8 halfwords, which forces the bank-stride to a multiple of 4 banks, capping the
// achievable distribution period at 32/4=8 rows. +8 already hits that ceiling (verified +16
// and +32 are WORSE -- period 4 and 2 respectively -- and +24/+40 only tie it). The residual
// store conflict most likely comes from sScore/sP's per-LANE sequential write in the online-
// softmax loop (`wP[lane * KV_TILE_PAD + j]` inside a `for j` loop) -- a different access
// shape than the row-parallel ldmatrix reads this padding targeted. A real further lever, but
// needs a softmax-write restructure, not a stride tweak -- not attempted this pass. See
// docs/perf/CUDA_OPTIMIZATION_SWEEP_2026-08.md for the full writeup.
extern "C" __global__ void __launch_bounds__(MAX_GROUP_WARPS * 32) attention_flash_mma_f16(
    const half* __restrict__ q,   // [seq, numHeads,   headDim] row-major
    const half* __restrict__ k,   // [seq, numKvHeads, headDim] row-major
    const half* __restrict__ v,   // [seq, numKvHeads, headDim] row-major
    half* __restrict__ output,    // [seq, numHeads,   headDim] row-major
    const int seq,
    const int num_heads,
    const int num_kv_heads,
    const float scale)
{
    const int lane = threadIdx.x & 31;       // 0..31 within warp
    const int warp = threadIdx.x >> 5;       // 0..group-1 (which query head in the group)
    const int group = num_heads / num_kv_heads;

    const int hk = blockIdx.x;               // kv head (shared by this block's warps)
    const int hq = hk * group + warp;        // this warp's query head
    const int q_tile = blockIdx.y;           // which 16-query tile
    const int q0 = q_tile * Q_TILE;          // first query row of this tile
    if (q0 >= seq) return;

    const int q_stride = num_heads * HEAD_DIM;
    const int kv_stride = num_kv_heads * HEAD_DIM;

    // Block-shared K/V tiles (loaded once, reused by all `group` warps).
    __shared__ half sK[KV_TILE * HEAD_DIM_PAD];        // [key][d], row stride HEAD_DIM_PAD
    __shared__ half sVt[HEAD_DIM * KV_TILE_PAD];       // V transposed: [d][key], row stride KV_TILE_PAD
    // Per-warp state (warp-indexed slices).
    __shared__ half sQ[MAX_GROUP_WARPS][Q_TILE * HEAD_DIM_PAD];     // [warp][q][d], row stride HEAD_DIM_PAD
    __shared__ half sP[MAX_GROUP_WARPS][Q_TILE * KV_TILE_PAD];      // [warp] normalised P: [q][key], row stride KV_TILE_PAD
    __shared__ float sScore[MAX_GROUP_WARPS][Q_TILE * KV_TILE_PAD_SCORE]; // [warp] raw S spilled, row stride KV_TILE_PAD_SCORE
    __shared__ float sM[MAX_GROUP_WARPS][Q_TILE];              // [warp] running row max
    __shared__ float sL[MAX_GROUP_WARPS][Q_TILE];              // [warp] running row denom
    __shared__ float sCorr[MAX_GROUP_WARPS][Q_TILE];          // [warp] per-row O rescale

    half* wQ = sQ[warp];
    half* wP = sP[warp];
    float* wScore = sScore[warp];
    float* wM = sM[warp];
    float* wL = sL[warp];
    float* wCorr = sCorr[warp];

    // Load this warp's Q tile (row-major [q][d]). 16*64 = 1024 halves / 32 lanes = 32 each.
    for (int i = lane; i < Q_TILE * HEAD_DIM; i += 32)
    {
        int qr = i / HEAD_DIM, d = i % HEAD_DIM;
        int gq = q0 + qr;
        wQ[qr * HEAD_DIM_PAD + d] = (gq < seq) ? q[(size_t)gq * q_stride + hq * HEAD_DIM + d] : __float2half(0.0f);
    }

    // O accumulator: 16 queries x 64 headDim = 8 n-subtiles, each a {f32 x4} frag.
    // O_frag[nsub][0..3] follows the C/D map: lane L -> rows (L>>2),(L>>2)+8 and
    // cols within the subtile (L%4)*2+{0,1}.
    float O_frag[HEAD_DIM / 8][4];
    #pragma unroll
    for (int n = 0; n < HEAD_DIM / 8; n++)
        O_frag[n][0] = O_frag[n][1] = O_frag[n][2] = O_frag[n][3] = 0.0f;

    if (lane < Q_TILE) { wM[lane] = -FLT_MAX; wL[lane] = 0.0f; }

    // This lane owns C/D rows r0 and r0+8, and within an 8-wide n-subtile the two
    // columns col_lo, col_hi.
    const int r0 = lane >> 2;
    const int col_lo = (lane & 3) * 2;
    const int col_hi = col_lo + 1;

    // Stream the key axis. Causal: keys only up to the last query in this tile. kv_end is
    // identical for every warp in the block (depends on q_tile, not head), so the shared
    // K/V load and __syncthreads() barriers below never diverge across warps.
    const int kv_end = min(seq, q0 + Q_TILE);
    for (int k0 = 0; k0 < kv_end; k0 += KV_TILE)
    {
        // Cooperative block-wide K/V load: all `group*32` threads fill the shared tile once.
        // [key][d] for K, transposed [d][key] for V.
        for (int i = threadIdx.x; i < KV_TILE * HEAD_DIM; i += blockDim.x)
        {
            int kr = i / HEAD_DIM, d = i % HEAD_DIM;
            int gk = k0 + kr;
            half kv = (gk < seq) ? k[(size_t)gk * kv_stride + hk * HEAD_DIM + d] : __float2half(0.0f);
            sK[kr * HEAD_DIM_PAD + d] = kv;
            half vv = (gk < seq) ? v[(size_t)gk * kv_stride + hk * HEAD_DIM + d] : __float2half(0.0f);
            sVt[(size_t)d * KV_TILE_PAD + kr] = vv;   // transpose into [d][key]
        }
        __syncthreads();   // block-wide: K/V visible to every warp before any warp reads

        // ---- QK: S[16x16] = Q . K^T, two n-subtiles (keys 0..7, 8..15) ----
        float S_frag[2][4];
        #pragma unroll
        for (int ns = 0; ns < 2; ns++)
            S_frag[ns][0] = S_frag[ns][1] = S_frag[ns][2] = S_frag[ns][3] = 0.0f;

        #pragma unroll
        for (int ks = 0; ks < HEAD_DIM / 16; ks++)   // 4 k-steps of 16
        {
            // A = Q[16 x 16] for d in [ks*16, ks*16+16): ldmatrix.x4 over [q][d].
            unsigned a[4];
            // Each lane points at row (lane%16), the half-tile selected by lane/16
            // gives the 8-col offset; ldmatrix gathers the full 16x16.
            const half* aptr = &wQ[(lane & 15) * HEAD_DIM_PAD + ks * 16 + (lane >> 4) * 8];
            ldmatrix_x4(a, aptr);

            #pragma unroll
            for (int ns = 0; ns < 2; ns++)
            {
                // B = K[8 x 16] for keys [ns*8, ns*8+8): ldmatrix.x2 over [key][d].
                unsigned b[2];
                const half* bptr = &sK[(ns * 8 + (lane & 7)) * HEAD_DIM_PAD + ks * 16 + ((lane >> 3) & 1) * 8];
                ldmatrix_x2(b, bptr);
                mma_m16n8k16(S_frag[ns], a, b);
            }
        }

        // Spill S to this warp's shared [q][key] via the C/D map, applying the QK scale.
        #pragma unroll
        for (int ns = 0; ns < 2; ns++)
        {
            int kcol = ns * 8 + col_lo;
            wScore[r0 * KV_TILE_PAD_SCORE + kcol]       = S_frag[ns][0] * scale;
            wScore[r0 * KV_TILE_PAD_SCORE + ns * 8 + col_hi] = S_frag[ns][1] * scale;
            wScore[(r0 + 8) * KV_TILE_PAD_SCORE + kcol] = S_frag[ns][2] * scale;
            wScore[(r0 + 8) * KV_TILE_PAD_SCORE + ns * 8 + col_hi] = S_frag[ns][3] * scale;
        }
        __syncwarp();

        // ---- Online softmax over this KV tile, one lane per query row (16 rows) ----
        if (lane < Q_TILE)
        {
            int gq = q0 + lane;
            int causal_last = gq;            // query gq attends keys 0..gq
            float m_prev = wM[lane];
            float l_prev = wL[lane];

            // tile-local max over valid (causal, in-range) keys
            float m_cur = m_prev;
            for (int j = 0; j < KV_TILE; j++)
            {
                int gk = k0 + j;
                if (gk < seq && gk <= causal_last)
                {
                    float s = wScore[lane * KV_TILE_PAD_SCORE + j];
                    if (s > m_cur) m_cur = s;
                }
            }

            // O accumulator so far is scaled by exp(running_total - m_prev); rescale
            // to the new max. When m_prev is -inf (first tile) O is zero, correction
            // is irrelevant — use 1.0 so the multiply is a no-op.
            float correction = (m_prev == -FLT_MAX) ? 1.0f : __expf(m_prev - m_cur);
            float l_cur = l_prev * correction;

            for (int j = 0; j < KV_TILE; j++)
            {
                int gk = k0 + j;
                float p = 0.0f;
                if (gk < seq && gk <= causal_last)
                {
                    p = __expf(wScore[lane * KV_TILE_PAD_SCORE + j] - m_cur);
                    l_cur += p;
                }
                wP[lane * KV_TILE_PAD + j] = __float2half(p);
            }

            wM[lane] = m_cur;
            wL[lane] = l_cur;
            wCorr[lane] = correction;
        }
        __syncwarp();

        // Rescale O accumulator by this tile's correction, then add P.V.
        // correction for the two rows this lane owns:
        float corr0 = wCorr[r0];
        float corr1 = wCorr[r0 + 8];
        #pragma unroll
        for (int n = 0; n < HEAD_DIM / 8; n++)
        {
            O_frag[n][0] *= corr0; O_frag[n][1] *= corr0;
            O_frag[n][2] *= corr1; O_frag[n][3] *= corr1;
        }

        // ---- PV: O[16 x 64] += P[16 x 16] . V[16 x 64], 8 n-subtiles, 1 k-step ----
        // A = P[16x16] from this warp's sP via ldmatrix.x4.
        unsigned pa[4];
        const half* pptr = &wP[(lane & 15) * KV_TILE_PAD + (lane >> 4) * 8];
        ldmatrix_x4(pa, pptr);

        #pragma unroll
        for (int n = 0; n < HEAD_DIM / 8; n++)   // each subtile = 8 headDim cols
        {
            // B = V^T[8 x 16] = sVt[d in n*8..n*8+8][key 0..15]. mma B is [n=8 x k=16].
            unsigned vb[2];
            const half* vptr = &sVt[(n * 8 + (lane & 7)) * KV_TILE_PAD + ((lane >> 3) & 1) * 8];
            ldmatrix_x2(vb, vptr);
            mma_m16n8k16(O_frag[n], pa, vb);
        }
        // Block-wide barrier before the next tile overwrites the shared K/V — every warp
        // must have finished consuming sK/sVt (PV reads sVt) before the reload.
        __syncthreads();
    }

    // Final normalisation by the row denom and write row-major [seq, heads, headDim].
    #pragma unroll
    for (int n = 0; n < HEAD_DIM / 8; n++)
    {
        int gr0 = q0 + r0;
        int gr1 = q0 + r0 + 8;
        float inv0 = (wL[r0] > 0.0f) ? 1.0f / wL[r0] : 0.0f;
        float inv1 = (wL[r0 + 8] > 0.0f) ? 1.0f / wL[r0 + 8] : 0.0f;
        int d_lo = n * 8 + col_lo;
        int d_hi = n * 8 + col_hi;
        if (gr0 < seq)
        {
            output[(size_t)gr0 * q_stride + hq * HEAD_DIM + d_lo] = __float2half(O_frag[n][0] * inv0);
            output[(size_t)gr0 * q_stride + hq * HEAD_DIM + d_hi] = __float2half(O_frag[n][1] * inv0);
        }
        if (gr1 < seq)
        {
            output[(size_t)gr1 * q_stride + hq * HEAD_DIM + d_lo] = __float2half(O_frag[n][2] * inv1);
            output[(size_t)gr1 * q_stride + hq * HEAD_DIM + d_hi] = __float2half(O_frag[n][3] * inv1);
        }
    }
}
