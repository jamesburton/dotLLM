// Tensor-core (mma.sync.m16n8k16) FP16 DECODE attention, composed with the GQA-group +
// split-KV grid design -- issue #199 v2.
//
// ─── Why a v2 exists ────────────────────────────────────────────────────────────────────
// v1 (native/kernels/attention_flash_mma_decode.cu, branch issue/199-tensor-core-decode-
// attention, NOT merged) built a decode-only tensor-core kernel with genuine HMMA/LDSM SASS
// (confirmed via cuobjdump --dump-sass: 69 registers/thread, 0 spill) but scoped to ONE WARP
// PER BLOCK, one block per query head (grid=numHeads=24 for Bonsai-27B) -- deliberately NOT
// composed with issues #197/#198's GQA-group grid, so the new FP16/tensor-core precision axis
// stayed separable from the grid-regrid axis while bringing the kernel up. Real wall-clock
// A/B (v1's .perf-runs/issue199-mma-decode/README.md) found it 4-5x SLOWER than the shipping
// attention_f32 baseline at every realistic Bonsai depth, root-caused to occupancy: 42688B
// static shared memory caps co-residency to ~2 blocks/SM at blockDim=32, i.e. ~64 threads/SM
// out of 1536 -- ~4% theoretical occupancy, WORSE than attention_f32's own already-diagnosed
// ~16.5% achieved occupancy this whole #197/#198/#199 investigation exists to fix. v1's own
// writeup named the fix and declined to attempt it in scope: compose with #197/#198's grid.
//
// ─── This kernel's structural change ───────────────────────────────────────────────────
// Grid = (numKvHeads, kvSplit) -- IDENTICAL shape to attention_f32_gqa_split_kv (see
// attention_f32.cu's combined-kernel header). Each block owns one KV head and one KV
// sub-range, exactly like that kernel; the difference here is HOW the `group` query heads
// sharing that KV head are computed.
//
// KEY INSIGHT (the thing that makes this more than "v1 wrapped in a bigger grid"): the
// mma.sync.m16n8k16 instruction's M dimension is 16 rows regardless of how many are real.
// v1 wasted 15/16 of every QK/PV instruction on zero-padding (seqQ==1, Q_TILE=16 needed only
// for the instruction's fixed shape). Since this project caps GQA group at MAX_GQA_GROUP=8
// (CudaKernels.MaxGqaGroup) and 8 <= Q_TILE=16, PACKING all `group` heads' query rows into
// the SAME 16-row tile (rows 0..group-1 real, group..15 zero) lets ONE set of QK/PV mma
// instructions compute ALL group heads' attention against this KV tile AT ONCE -- the same
// instruction count as v1's single-head version, up to MAX_GQA_GROUP=8x more useful
// throughput per instruction, for free. This is why sQ/sO/sScore/sP below are NOT sized per
// warp or per group (contrast attention_f32_gqa_split_kv's `group`-sized DYNAMIC shared
// arrays, needed there because that kernel's register-blocking duplicates per-head state) --
// they stay Q_TILE=16-sized regardless of group, so this kernel's STATIC shared-memory
// footprint is the SAME ~42688B as v1's single-head kernel (see the layout comment above the
// kernel below) and fits the 48KB static cap with no dynamic-shared-memory opt-in needed,
// unlike what the v1 writeup predicted a naive per-warp-per-head duplication would require
// ("group=6 case's shared-memory footprint... exceeds the 48KB static cap" -- true for a
// naive design, not for this packed one).
//
// Occupancy therefore comes from TWO levers stacked, not one:
//   1. More resident THREADS/block: blockDim = NUM_WARPS*32 = 256 (vs v1's 32), all
//      cooperating on ONE shared K/V tile load + this block's group's QK/PV -- real
//      intra-block warp-level parallelism, not v1's single active warp.
//   2. More BLOCKS via kvSplit: grid = numKvHeads * kvSplit (vs v1's numHeads=24, WORSE than
//      grid=numKvHeads=4 alone would be -- #197/#198's own finding), reusing this project's
//      already-validated CudaKernels.ComputeAttentionKvSplit occupancy-target heuristic and
//      MaxSafeAttentionGqaSplit-style co-residency query (see CudaKernels.cs).
//
// Within a block, work is split across the NUM_WARPS=8 warps by PHASE, not by query head
// (query heads are already handled "for free" via the M-dim packing above, so warp-splitting
// by head would just redundantly recompute the same packed mma):
//   - QK (S[16x16] = Q_packed . K^T): warps 0 and 1 split the 2 N-subtiles (8 keys each) --
//     halves v1's serial 32-mma QK critical path to 16.
//   - Online-softmax reduction (per real row, group <= 8 lanes' worth of scalar work): warp 0
//     only, lanes < group -- cheap, not worth parallelizing further.
//   - PV (O[16x256] += P[16x16] . V[16x256]): ALL 8 warps split HEAD_DIM/8=32 d-chunks, 4
//     each, writing DISJOINT sO columns (no inter-warp sync needed within this phase) -- v1's
//     own writeup named this serialized 32-d-chunk loop "likely the dominant per-launch cost"
//     for the single-warp case; here it is genuinely 8-way parallel.
// This QK/PV split is intentionally simple (no cross-warp reduction) to keep the new-bug
// surface small: each warp's per-phase math is byte-for-byte the same computation v1already
// validated (ldmatrix/mma addressing, fast_exp_neg/expf split), just distributed across more
// warps/threads rather than reinvented. A further QK-side win (splitting the K-dimension
// across more warps with a small reduction) is a plausible follow-up, not attempted here --
// QK and PV have equal total mma-instruction counts per tile (32 each), so this leaves QK
// somewhat less parallelized (2-way) than PV (8-way); documented, not hidden.
//
// ─── Combine phase: ported verbatim from attention_f32_gqa_split_kv ───────────────────────
// The cross-split combine (grid.sync() + fast_exp_neg-reweighted merge of partial_max/
// partial_sum/partial_out, kv_split==1 fast path) is copied unchanged from that kernel's
// already-proven-correct implementation (attention_f32.cu, issues #197/#198), just applied
// per (hq = hkv*group+g) the same way. This is deliberate de-risking: the combine algebra is
// not new, only which values feed it (this kernel's tensor-core partials instead of that
// kernel's register-blocked-FP32 partials) is new.
//
// ─── Precision ──────────────────────────────────────────────────────────────────────────
// Reuses v1's hard-won precision groundwork verbatim: FP16 Q/K/V, FP32 mma accumulator
// (hardware property of mma.sync.aligned.m16n8k16...f32.f16.f16.f32, not a choice),
// fast_exp_neg for per-key softmax weights, PRECISE expf (not fast_exp_neg) for the
// cross-KV-tile online-softmax correction factor -- v1 found and fixed a real bug here
// (fast_exp_neg's ~1% approximation error compounds geometrically across KV_TILE=16's many
// more rescale events than attention_f32's TILE_KV=256 ever exercises). This kernel's KV_TILE
// is unchanged (still 16, still mma.sync.m16n8k16-dictated) so the SAME fix applies --
// re-verified (not just assumed) against the CPU oracle and the F32 GPU baseline in
// CudaAttentionMmaDecodeGqaSplitTests.cs, since the new multi-warp PV split and cross-split
// combine are each a new source of reassociated float summation that could in principle
// reopen a similar class of question (same reasoning attention_f32_gqa_split_kv's own header
// already gives for why ITS grouped-warp design needed its own bit-exactness check rather
// than inheriting attention_f32's for free).
//
// Ships opt-in (DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT=1, default OFF), same #180/#183 precedent as
// v1 and the plain GQA-split kernel: a new precision + reassociation axis on this
// architecture, not yet backed by a real generation-level validation pass.

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <cooperative_groups.h>

#define HEAD_DIM 256
#define Q_TILE 16
#define KV_TILE 16
#define MAX_GQA_GROUP 8          // MUST match CudaKernels.MaxGqaGroup / attention_f32.cu's MAX_GQA_GROUP.
#define NUM_WARPS 8               // blockDim = NUM_WARPS*32 = 256, matches this project's BlockSize convention.
#define PV_CHUNKS_PER_WARP (HEAD_DIM / 8 / NUM_WARPS)   // 256/8/8 = 4

// Schraudolph fast-exp constants (mirror FastMath.cs, attention_f32.cu, and v1's decode
// kernel). See the file header above for the precise-expf-for-cross-tile-correction /
// fast_exp_neg-for-per-key-P split this project found necessary at KV_TILE=16.
#define FASTEXP_C0 12102203.0f
#define FASTEXP_C1 1064866805.0f
#define FASTEXP_MIN_CLAMP -87.3f

__device__ __forceinline__ float fast_exp_neg(float x)
{
    x = fmaxf(x, FASTEXP_MIN_CLAMP);
    int bits = __float2int_rz(fmaf(x, FASTEXP_C0, FASTEXP_C1));
    return __int_as_float(bits);
}

// ── PTX helpers -- duplicated verbatim from attention_flash_mma.cu / v1's decode kernel ───
// (see attention_flash_mma.cu's header for the fragment-layout notes; unchanged here).

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem_ptr)
{
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned (&r)[2], const void* smem_ptr)
{
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr));
}

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

// Load / store a {f32 x4} C/D accumulator fragment from/to a row-major [Q_TILE x HEAD_DIM]
// f32 shared buffer, at the PTX C/D lane map -- identical to v1's decode kernel.
__device__ __forceinline__ void load_o_frag(
    float (&d)[4], const float* sO, int r0, int col_lo, int col_hi)
{
    d[0] = sO[r0 * HEAD_DIM + col_lo];
    d[1] = sO[r0 * HEAD_DIM + col_hi];
    d[2] = sO[(r0 + 8) * HEAD_DIM + col_lo];
    d[3] = sO[(r0 + 8) * HEAD_DIM + col_hi];
}

__device__ __forceinline__ void store_o_frag(
    float* sO, int r0, int col_lo, int col_hi, const float (&d)[4])
{
    sO[r0 * HEAD_DIM + col_lo] = d[0];
    sO[r0 * HEAD_DIM + col_hi] = d[1];
    sO[(r0 + 8) * HEAD_DIM + col_lo] = d[2];
    sO[(r0 + 8) * HEAD_DIM + col_hi] = d[3];
}

// Grid = (numKvHeads, kvSplit, 1). Block = (NUM_WARPS*32, 1, 1) = (256, 1, 1).
// Static shared-memory budget per block (same as v1's single-head kernel -- see file header
// for why packing `group` heads into ONE Q_TILE=16 tile keeps this group-INDEPENDENT):
//   sQ     16*256*2  =  8192 B
//   sK     16*256*2  =  8192 B
//   sVt    256*16*2  =  8192 B
//   sO     16*256*4  = 16384 B
//   sScore 16*16*4   =  1024 B
//   sP     16*16*2   =   512 B
//   sM/sL/sCorr 3*16*4 = 192 B
//   ---------------------------
//   total            ~ 42688 B  (< 48 KB static cap, no dynamic shared memory needed)
extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32) attention_flash_mma_decode_gqa_split_f16(
    const half* __restrict__ q,        // [numHeads, HEAD_DIM] row-major (seqQ==1)
    const half* __restrict__ k,        // [seqKv, numKvHeads, HEAD_DIM] row-major (FP16 KV cache)
    const half* __restrict__ v,        // [seqKv, numKvHeads, HEAD_DIM] row-major
    float* __restrict__ output,        // [numHeads, HEAD_DIM] row-major, F32
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const float scale,
    const int kv_split,
    float* __restrict__ partial_max,   // [num_heads, kv_split]
    float* __restrict__ partial_sum,   // [num_heads, kv_split]
    float* __restrict__ partial_out)   // [num_heads, kv_split, HEAD_DIM]
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int hkv = blockIdx.x;
    const int s = blockIdx.y;
    const int group = num_heads / num_kv_heads;   // <= MAX_GQA_GROUP, caller-enforced (CanUse).
    const int kv_stride = num_kv_heads * HEAD_DIM;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    __shared__ half sQ[Q_TILE * HEAD_DIM];
    __shared__ half sK[KV_TILE * HEAD_DIM];
    __shared__ half sVt[HEAD_DIM * KV_TILE];       // V transposed: [d][key]
    __shared__ float sO[Q_TILE * HEAD_DIM];        // O accumulator, shared across the WHOLE group
                                                     // (rows 0..group-1 real -- see file header).
    __shared__ float sScore[Q_TILE * KV_TILE];
    __shared__ half sP[Q_TILE * KV_TILE];
    __shared__ float sM[Q_TILE];
    __shared__ float sL[Q_TILE];
    __shared__ float sCorr[Q_TILE];

    // Pack `group` distinct query heads (rows 0..group-1) sharing this KV head into ONE
    // Q_TILE=16 mma tile; rows group..15 zero-padded (never read again -- same "wasted lanes
    // are not a correctness issue" convention v1 used for its 15 padding rows, just now only
    // Q_TILE-group of them are wasted instead of Q_TILE-1).
    for (int i = threadIdx.x; i < Q_TILE * HEAD_DIM; i += blockDim.x)
    {
        int row = i / HEAD_DIM, d = i % HEAD_DIM;
        sQ[i] = (row < group) ? q[(size_t)(hkv * group + row) * HEAD_DIM + d] : __float2half(0.0f);
    }
    for (int i = threadIdx.x; i < Q_TILE * HEAD_DIM; i += blockDim.x)
        sO[i] = 0.0f;
    // sP rows >= group are never written by the softmax phase below (it only ever touches
    // lane < group) -- zero them ONCE here rather than leaving them as uninitialized shared
    // memory. Not a correctness requirement for the REAL rows' mma output (mma.sync computes
    // each output row independently -- a garbage row >= group in the P operand cannot leak
    // into rows < group's accumulation), but avoids reading uninitialized memory at all
    // (matters for tools like compute-sanitizer --tool initcheck) at negligible one-time cost.
    for (int i = threadIdx.x; i < Q_TILE * KV_TILE; i += blockDim.x)
        sP[i] = __float2half(0.0f);
    if ((int)threadIdx.x < group) { sM[threadIdx.x] = -FLT_MAX; sL[threadIdx.x] = 0.0f; }
    __syncthreads();

    const int r0 = lane >> 2;
    const int col_lo = (lane & 3) * 2;
    const int col_hi = col_lo + 1;

    // This split's contiguous KV sub-range [kv_lo, kv_hi) -- identical chunking to
    // attention_f32_gqa_split_kv (attention_f32.cu), so the two kernels' kvSplit heuristics
    // and MaxSafe* co-residency queries stay directly comparable.
    int chunk = (seq_kv + kv_split - 1) / kv_split;
    int kv_lo = s * chunk;
    int kv_hi = kv_lo + chunk;
    if (kv_hi > seq_kv) kv_hi = seq_kv;
    if (kv_lo > seq_kv) kv_lo = seq_kv;

    for (int k0 = kv_lo; k0 < kv_hi; k0 += KV_TILE)
    {
        int gk_limit = min(KV_TILE, kv_hi - k0);

        // Cooperative K/V tile load -- ALL threadIdx.x (256) help, not just one warp's 32
        // lanes as in v1 -- a real, independent speedup for this phase on top of the mma win.
        for (int i = threadIdx.x; i < KV_TILE * HEAD_DIM; i += blockDim.x)
        {
            int kr = i / HEAD_DIM, d = i % HEAD_DIM;
            int gk = k0 + kr;
            half kv_ = (kr < gk_limit) ? k[(size_t)gk * kv_stride + hkv * HEAD_DIM + d] : __float2half(0.0f);
            sK[i] = kv_;
            half vv = (kr < gk_limit) ? v[(size_t)gk * kv_stride + hkv * HEAD_DIM + d] : __float2half(0.0f);
            sVt[(size_t)d * KV_TILE + kr] = vv;
        }
        __syncthreads();

        // ---- QK: S[16 x 16] = Q_packed . K^T -- warps 0/1 split the 2 N-subtiles (keys
        // 0..7 / 8..15), each doing all HEAD_DIM/16=16 k-steps for its own subtile. Halves
        // v1's serial 32-mma QK chain to 16 (see file header). ----
        if (warp_id < 2)
        {
            int ns = warp_id;
            float d_frag[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            #pragma unroll
            for (int ks = 0; ks < HEAD_DIM / 16; ks++)
            {
                unsigned a[4];
                const half* aptr = &sQ[(lane & 15) * HEAD_DIM + ks * 16 + (lane >> 4) * 8];
                ldmatrix_x4(a, aptr);

                unsigned b[2];
                const half* bptr = &sK[(ns * 8 + (lane & 7)) * HEAD_DIM + ks * 16 + ((lane >> 3) & 1) * 8];
                ldmatrix_x2(b, bptr);
                mma_m16n8k16(d_frag, a, b);
            }
            int kcol = ns * 8 + col_lo;
            sScore[r0 * KV_TILE + kcol] = d_frag[0] * scale;
            sScore[r0 * KV_TILE + ns * 8 + col_hi] = d_frag[1] * scale;
            sScore[(r0 + 8) * KV_TILE + kcol] = d_frag[2] * scale;
            sScore[(r0 + 8) * KV_TILE + ns * 8 + col_hi] = d_frag[3] * scale;
        }
        __syncthreads();

        // ---- Online softmax over this KV tile, warp 0 only, one lane per REAL query row
        // (lane < group, not lane < Q_TILE -- v1 iterated the full padded Q_TILE=16 since it
        // only ever had 1 real row; here up to MAX_GQA_GROUP=8 lanes do real work, cheaper
        // and correct). ----
        if (warp_id == 0 && lane < group)
        {
            float m_prev = sM[lane];
            float l_prev = sL[lane];

            float m_cur = m_prev;
            for (int j = 0; j < gk_limit; j++)
            {
                float sc = sScore[lane * KV_TILE + j];
                if (sc > m_cur) m_cur = sc;
            }

            // Precise expf here, NOT fast_exp_neg -- see file header (ported verbatim from
            // v1's bring-up finding, KV_TILE=16 unchanged so the same compounding-error
            // mechanism applies).
            float correction = (m_prev == -FLT_MAX) ? 1.0f : expf(m_prev - m_cur);
            float l_cur = l_prev * correction;

            for (int j = 0; j < KV_TILE; j++)
            {
                float p = 0.0f;
                if (j < gk_limit)
                {
                    p = fast_exp_neg(sScore[lane * KV_TILE + j] - m_cur);
                    l_cur += p;
                }
                sP[lane * KV_TILE + j] = __float2half(p);
            }

            sM[lane] = m_cur;
            sL[lane] = l_cur;
            sCorr[lane] = correction;
        }
        __syncthreads();

        // Rescale the shared O accumulator by this tile's per-row correction (real rows only
        // -- rows >= group have no valid sCorr entry and are never read downstream).
        for (int i = threadIdx.x; i < Q_TILE * HEAD_DIM; i += blockDim.x)
        {
            int row = i / HEAD_DIM;
            if (row < group) sO[i] *= sCorr[row];
        }
        __syncthreads();

        // ---- PV: O[16 x 256] += P[16 x 16] . V[16 x 256] -- HEAD_DIM/8=32 d-chunks split
        // across all NUM_WARPS=8 warps (PV_CHUNKS_PER_WARP=4 each), writing DISJOINT sO
        // columns per warp (no inter-warp sync needed within this loop). Real 8-way
        // within-block parallelism, unlike v1's single serialized warp and unlike
        // attention_f32_gqa_split_kv's sequential per-head reduction loop (see file header). ----
        unsigned pa[4];
        const half* pptr = &sP[(lane & 15) * KV_TILE + (lane >> 4) * 8];
        ldmatrix_x4(pa, pptr);

        #pragma unroll
        for (int c = 0; c < PV_CHUNKS_PER_WARP; c++)
        {
            int n = warp_id * PV_CHUNKS_PER_WARP + c;
            unsigned vb[2];
            const half* vptr = &sVt[(n * 8 + (lane & 7)) * KV_TILE + ((lane >> 3) & 1) * 8];
            ldmatrix_x2(vb, vptr);

            float o_frag[4];
            load_o_frag(o_frag, sO, r0, n * 8 + col_lo, n * 8 + col_hi);
            mma_m16n8k16(o_frag, pa, vb);
            store_o_frag(sO, r0, n * 8 + col_lo, n * 8 + col_hi, o_frag);
        }
        // Every warp must finish reading sK/sVt/sP before the next tile's cooperative load
        // overwrites them, and all PV writes to sO must land before the next tile's rescale.
        __syncthreads();
    }

    // Publish this split's partial (UNNORMALIZED) for EACH of this block's group heads, using
    // the EXACT SAME [numHeads, kv_split] / [numHeads, kv_split, headDim] layout
    // attention_f32_gqa_split_kv already defines (see file header) -- the combine phase below
    // and the C# scratch allocator depend on it being unchanged, just indexed by
    // hq = hkv*group + g and fed from THIS kernel's packed sM/sL/sO rows instead of that
    // kernel's per-head register-blocked accumulators.
    for (int g = 0; g < group; g++)
    {
        int hq = hkv * group + g;
        if (threadIdx.x == 0)
        {
            partial_max[(size_t)hq * kv_split + s] = sM[g];
            partial_sum[(size_t)hq * kv_split + s] = sL[g];
        }
        float* partial_out_vec = partial_out + ((size_t)hq * kv_split + s) * HEAD_DIM;
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
            partial_out_vec[d] = sO[g * HEAD_DIM + d];
    }

    grid.sync();   // ALL blocks (every kv head, every split) must have published before combine.

    if (s != 0) return; // only the split=0 block per KV head performs the final combine.

    __shared__ float s_combined_max;
    __shared__ float s_combined_sum;

    // ── Combine: ported verbatim from attention_f32_gqa_split_kv (attention_f32.cu) -- see
    // this file's header for why the algebra is not re-derived here. ──
    for (int g = 0; g < group; g++)
    {
        int hq = hkv * group + g;

        if (kv_split == 1)
        {
            // Trivial one-way combine: nothing to reassociate, skip fast_exp_neg(0) entirely
            // so this path stays as close as possible to the un-split accumulation.
            if (threadIdx.x == 0)
            {
                s_combined_max = partial_max[(size_t)hq * kv_split + 0];
                s_combined_sum = partial_sum[(size_t)hq * kv_split + 0];
            }
            __syncthreads();
            float sum_inv = (s_combined_sum > 1e-10f) ? (1.0f / s_combined_sum) : 0.0f;
            const float* partial_out_vec = partial_out + ((size_t)hq * kv_split + 0) * HEAD_DIM;
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
                output[(size_t)hq * HEAD_DIM + d] = partial_out_vec[d] * sum_inv;
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0)
        {
            float m = -FLT_MAX;
            for (int i = 0; i < kv_split; i++)
                m = fmaxf(m, partial_max[(size_t)hq * kv_split + i]);

            float l = 0.0f;
            for (int i = 0; i < kv_split; i++)
            {
                float mi = partial_max[(size_t)hq * kv_split + i];
                float li = partial_sum[(size_t)hq * kv_split + i];
                float w = (mi > -FLT_MAX + 1.0f) ? fast_exp_neg(mi - m) : 0.0f;
                l += li * w;
            }
            s_combined_max = m;
            s_combined_sum = l;
        }
        __syncthreads();

        float m = s_combined_max;
        float sum_inv = (s_combined_sum > 1e-10f) ? (1.0f / s_combined_sum) : 0.0f;

        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
        {
            float o = 0.0f;
            for (int i = 0; i < kv_split; i++)
            {
                float mi = partial_max[(size_t)hq * kv_split + i];
                float w = (mi > -FLT_MAX + 1.0f) ? fast_exp_neg(mi - m) : 0.0f;
                float oi = partial_out[((size_t)hq * kv_split + i) * HEAD_DIM + d];
                o += oi * w;
            }
            output[(size_t)hq * HEAD_DIM + d] = o * sum_inv;
        }
        __syncthreads();
    }
}
