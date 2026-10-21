// Tensor-core (mma.sync.m16n8k16) FP16 flash-style PREFILL attention body for head_dim 256 --
// issue #962 (F16 KV), shared with the quantized-KV entries of issue #1094.
//
// Shared by attention_flash_mma_prefill_hd256.cu (F16 KV, cp.async tile loads) and
// attention_flash_mma_prefill_hd256_kvq.cu (Q8_0 / Q4_0 KV dequantized in the tile load). The
// body is templated on a K and a V "tile loader" policy that fills one 16-byte (8 x half) chunk of
// the padded shared-memory K/V tile from (base, element index). Everything after the tile load --
// the QK / softmax / PV arithmetic and the shared-memory layout -- is one piece of code, so an
// entry whose loader writes the same halves the F16 entry would read produces bit-identical output.
//
// ─── Why ────────────────────────────────────────────────────────────────────────────────
// Bonsai-2 27B (qwen35 dense hybrid: 24 Q heads / 4 KV heads / head_dim 256, causal, 16 of 64
// layers full attention) prefilled through attention_f32 (one block per (query row, head),
// scalar F32 dot products, F32 K/V staged from the F16 cache). That path is O(S^2) scalar work
// and dominates long-context fill (512:144 tok/s ... 8192:49 tok/s on an RTX 3060). This kernel
// replaces it with tensor-core FlashAttention-2 style tiling and needs no F32 staging at all.
//
// ─── Design ─────────────────────────────────────────────────────────────────────────────
//   * Grid  = (ceil(seq_q/16) q-tiles, num_kv_heads). Block = `group` warps (group = Hq/Hkv,
//     <= 8), one warp per query head of the GQA group, ALL warps sharing the same 16 query rows.
//     => identical causal range for every warp of the block (no divergence), and each K/V tile
//     is fetched from global once and reused `group` times out of shared memory.
//   * q-tile blocks are scheduled heaviest-first (blockIdx.x reversed): causal work grows
//     linearly with the tile index, so launching the long ones first shrinks the tail.
//   * Each warp owns a 16 x 256 F32 O accumulator IN REGISTERS (128 regs/thread, mma C layout),
//     plus per-row running max / sum (two rows per thread, quad-shuffled reductions).
//   * Q (F16) lives in shared memory per warp (16 x 256, padded rows) and is re-read with
//     ldmatrix for every KV tile; K and V tiles are double-buffered with cp.async (the quantized
//     loaders cp.async the raw block bytes into the same buffers and decode them in place, see
//     the _kvq.cu).
//   * QK^T: S[16 x KV_TILE] via mma.m16n8k16 (16 k-steps over d; B fragments from K rows with
//     non-transposed ldmatrix.x4). Two accumulator sets (even/odd k-step) break the dependency
//     chain.
//   * Online softmax in the exp2 domain (scale*log2e folded into the scores). Masked entries are
//     -inf and P is explicitly 0 there; a row that has seen no unmasked key keeps m = -inf and is
//     handled with m_use = 0, so nothing ever produces inf-inf. O is only rescaled when some row
//     of the warp's max changed (warp vote), which is rare after the first tiles.
//   * PV: the C fragment of S is re-packed into the A fragment of P (FA2 register trick, no
//     shared-memory round trip); B fragments from V rows with ldmatrix.x4.trans.
//   * Causal / ragged handling: query row i has absolute position position_offset + i and attends
//     keys 0..position_offset+i (same semantics as attention_f32.cu; seq_kv >= position_offset +
//     seq_q). KV tiles entirely above the diagonal of the block's last row are never loaded.
//     Keys past seq_kv in the tail tile are clamped to a valid row for the load and masked.
//
// ─── Precision ──────────────────────────────────────────────────────────────────────────
// F16 Q/K/V/P operands, F32 mma accumulation, F32 softmax statistics and F32 output. P weights
// are rounded to F16 for the PV mma while the denominator sums the F32 weights (same convention
// as the decode kernel attention_flash_mma_decode_gqa_split.cu).
//
// Output: F32 [seq_q, num_heads*256] row-major (identical to attention_f32's output).
//
// Dynamic shared memory: group*16*264*2 (Q) + NBUF*2*KV_TILE*264*2 (K,V). At group=6, KV_TILE=16,
// NBUF=2 that is 50.7 + 33.8 = 84.5 KB, so the launcher opts the function into >48 KB. The
// quantized entries use exactly the same layout and size: K/V are stored as F16 in shared memory.

#pragma once

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

#define HEAD_DIM 256
#define Q_TILE 16
#ifndef KV_TILE
#define KV_TILE 16               // keys per pipeline stage; multiple of 16
#endif
#ifndef NBUF
#define NBUF 2                   // 1 = single buffered, 2 = cp.async double buffered
#endif
#define SMEM_STRIDE (HEAD_DIM + 8)   // +16 B pad => ldmatrix rows land in distinct banks
#define NT (KV_TILE / 8)             // n-subtiles of S per KV tile
#define MAX_GROUP 8
#define LOG2E 1.4426950408889634f
#ifndef PREFILL_STAGE_NP0
#define PREFILL_STAGE_NP0 3          // first PV np step after which a staged row pair is decoded
#endif
#ifndef PREFILL_STAGE_EVERY
#define PREFILL_STAGE_EVERY 4        // PV np steps between staged row pairs
#endif
#define NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* p)
{
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x4_trans(unsigned (&r)[4], const void* p)
{
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const unsigned (&a)[4],
                                             unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src)
{
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ unsigned pack_half2(float lo, float hi)
{
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

// ── Tile loader policies: chunk(dst, base, elem) fills 8 consecutive halves of the shared-memory
//    tile at `dst` (16-byte aligned) from element `elem` (multiple of 8) of the logical
//    [seq_kv, num_kv_heads*HEAD_DIM] KV grid at `base`. ──

//    A loader with kStaged == true (the quantized-KV one) copies RAW bytes in chunk() and turns
//    them into F16 in place with read_row(tile, r) / write_row(tile, r, h) (this lane's chunk of
//    tile row r). The body calls them through stage_decode_row / stage_decode_tile, with the same
//    thread -> (row, chunk) mapping as load_tile (i = threadIdx.x + s*blockDim.x, row i>>5, chunk
//    i&31; blockDim is a multiple of 32, so each row belongs to exactly one warp): tile 0 before
//    the first barrier, tile t+1 interleaved with tile t's PV MMAs. For F16 (kStaged == false)
//    those calls are compiled out.

// F16 KV: asynchronous 16-byte copy, completed by the body's cp.async commit / wait.
struct PrefillTileLoadF16
{
    __device__ __forceinline__ static void chunk(half* dst, const void* base, size_t elem)
    {
        cp_async16(dst, reinterpret_cast<const half*>(base) + elem);
    }

    static constexpr bool kStaged = false;
    static constexpr int kSlots = 1;
    __device__ __forceinline__ static uint4 read_row(const half*, int) { return make_uint4(0, 0, 0, 0); }
    __device__ __forceinline__ static void write_row(half*, int, uint4) {}
};

// In-place decode of row r of one K tile and one V tile (this warp's row; all 32 lanes): both
// rows' staged raw bytes are read into registers, __syncwarp, then both F16 rows are written --
// so no lane overwrites raw bytes another lane has yet to read.
template <class KL, class VL>
__device__ __forceinline__ void stage_decode_row(half* tk, half* tv, int r)
{
    const uint4 hk = KL::read_row(tk, r);
    const uint4 hv = VL::read_row(tv, r);
    __syncwarp();
    KL::write_row(tk, r, hk);
    VL::write_row(tv, r, hv);
}

// In-place decode of a whole K tile and V tile (tile 0, before the first barrier): this warp's rows.
template <class KL, class VL>
__device__ __forceinline__ void stage_decode_tile(half* tk, half* tv, int warp, int group)
{
    __syncwarp();   // this warp's completed cp.async (each lane waited on its own) visible warp-wide
    for (int r = warp; r < KV_TILE; r += group)
        stage_decode_row<KL, VL>(tk, tv, r);
}

// q       : [seq_q, num_heads*HEAD_DIM] F16 row-major
// k, v    : [seq_kv, num_kv_heads*HEAD_DIM] KV cache, in the KL / VL loader's format
// output  : [seq_q, num_heads*HEAD_DIM] F32 row-major
// scale   : softmax scale applied to Q.K (typically rsqrt(HEAD_DIM))
template <class KL, class VL>
__device__ __forceinline__ void attn_flash_prefill_hd256_body(
    const half* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int position_offset,
    const float scale)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = num_heads / num_kv_heads;
    const int nq_tiles = (seq_q + Q_TILE - 1) / Q_TILE;
    const int qt = nq_tiles - 1 - (int)blockIdx.x;      // heaviest causal tiles first
    const int hkv = blockIdx.y;
    const int hq = hkv * group + warp;
    const int qbase = qt * Q_TILE;
    const int q_stride = num_heads * HEAD_DIM;
    const int kv_stride = num_kv_heads * HEAD_DIM;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* sQ = reinterpret_cast<half*>(smem_raw);                 // [group][16][SMEM_STRIDE]
    half* sK = sQ + group * Q_TILE * SMEM_STRIDE;                 // [NBUF][KV_TILE][SMEM_STRIDE]
    half* sV = sK + NBUF * KV_TILE * SMEM_STRIDE;                 // [NBUF][KV_TILE][SMEM_STRIDE]

    // ---- Q tile -> smem (rows past seq_q zero-filled; they are never written back) ----
    for (int i = threadIdx.x; i < group * Q_TILE * (HEAD_DIM / 8); i += blockDim.x)
    {
        int c = i & 31;                       // 16-byte chunk within the row (HEAD_DIM/8 == 32)
        int r = (i >> 5) & (Q_TILE - 1);
        int w = i >> 9;                       // 16 rows * 32 chunks = 512 per warp
        int qrow = qbase + r;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (qrow < seq_q)
            val = *reinterpret_cast<const uint4*>(
                q + (size_t)qrow * q_stride + (size_t)(hkv * group + w) * HEAD_DIM + c * 8);
        *reinterpret_cast<uint4*>(sQ + (size_t)(w * Q_TILE + r) * SMEM_STRIDE + c * 8) = val;
    }

    // KV range of this block: keys 0 .. position_offset+qbase+15 (last row of the tile), <= seq_kv.
    int kv_end = position_offset + qbase + Q_TILE;
    if (kv_end > seq_kv) kv_end = seq_kv;
    const int num_tiles = (kv_end + KV_TILE - 1) / KV_TILE;

    // The ONLY place K and V enter shared memory: the loader policies decide how.
    auto load_tile = [&](int t, int buf)
    {
        const int k0 = t * KV_TILE;
        half* dK = sK + (size_t)buf * KV_TILE * SMEM_STRIDE;
        half* dV = sV + (size_t)buf * KV_TILE * SMEM_STRIDE;
        for (int i = threadIdx.x; i < KV_TILE * (HEAD_DIM / 8); i += blockDim.x)
        {
            int c = i & 31, r = i >> 5;
            int key = k0 + r;
            if (key > seq_kv - 1) key = seq_kv - 1;          // clamp: masked later, stays finite
            size_t g = (size_t)key * kv_stride + (size_t)hkv * HEAD_DIM + c * 8;
            KL::chunk(dK + (size_t)r * SMEM_STRIDE + c * 8, k, g);
            VL::chunk(dV + (size_t)r * SMEM_STRIDE + c * 8, v, g);
        }
    };

    float o[HEAD_DIM / 8][4];
    #pragma unroll
    for (int n = 0; n < HEAD_DIM / 8; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f; }
    float m_lo = NEG_INF, m_hi = NEG_INF;     // running max (log2 domain), rows lane>>2 and +8
    float l_lo = 0.0f, l_hi = 0.0f;               // per-lane partial denominators (quad-reduced at end)

    const float scale_log2 = scale * LOG2E;
    const int r_lo = lane >> 2;
    const int col_lo = (lane & 3) * 2;
    const int pos_lo = position_offset + qbase + r_lo;
    const int pos_hi = pos_lo + 8;
    const half* qw = sQ + (size_t)warp * Q_TILE * SMEM_STRIDE;

    if (NBUF == 2 && num_tiles > 0) { load_tile(0, 0); cp_async_commit(); }

    for (int t = 0; t < num_tiles; t++)
    {
        const int buf = t & (NBUF - 1);
        if (NBUF == 2)
        {
            if (t + 1 < num_tiles) load_tile(t + 1, (t + 1) & 1);
            cp_async_commit();
            cp_async_wait<1>();
        }
        else
        {
            load_tile(t, 0);
            cp_async_commit();
            cp_async_wait<0>();
        }
        // Loader hooks, after this thread's copies of tile t completed: a staging loader (the
        // quantized-KV one) decodes tile t in place here -- every read of K and V before any write;
        // no-ops for F16.
        // Staging loaders decode tile 0 here; every later tile is decoded during the previous
        // tile's compute (below). Compile-time false for F16.
        if (KL::kStaged && t == 0)
            stage_decode_tile<KL, VL>(sK + (size_t)buf * KV_TILE * SMEM_STRIDE,
                                      sV + (size_t)buf * KV_TILE * SMEM_STRIDE, warp, group);
        __syncthreads();

        if (warp < group)
        {
            const half* kb = sK + (size_t)buf * KV_TILE * SMEM_STRIDE;
            const half* vb = sV + (size_t)buf * KV_TILE * SMEM_STRIDE;
            const int k0 = t * KV_TILE;

            // ---- S = Q . K^T ----
            float sa[NT][4], sb[NT][4];
            #pragma unroll
            for (int n = 0; n < NT; n++)
            { sa[n][0] = sa[n][1] = sa[n][2] = sa[n][3] = 0.f; sb[n][0] = sb[n][1] = sb[n][2] = sb[n][3] = 0.f; }

            #pragma unroll
            for (int ks = 0; ks < HEAD_DIM / 16; ks++)
            {
                unsigned a[4];
                ldmatrix_x4(a, qw + (size_t)(lane & 15) * SMEM_STRIDE + ks * 16 + (lane >> 4) * 8);
                #pragma unroll
                for (int np = 0; np < NT / 2; np++)
                {
                    unsigned b[4];
                    ldmatrix_x4(b, kb + (size_t)(np * 16 + (lane & 7) + ((lane >> 4) & 1) * 8) * SMEM_STRIDE
                                       + ks * 16 + ((lane >> 3) & 1) * 8);
                    if ((ks & 1) == 0)
                    {
                        mma_m16n8k16(sa[2 * np], a, b[0], b[1]);
                        mma_m16n8k16(sa[2 * np + 1], a, b[2], b[3]);
                    }
                    else
                    {
                        mma_m16n8k16(sb[2 * np], a, b[0], b[1]);
                        mma_m16n8k16(sb[2 * np + 1], a, b[2], b[3]);
                    }
                }
            }

            // ---- scale, causal/tail mask, tile row max ----
            float tmax_lo = NEG_INF, tmax_hi = NEG_INF;
            #pragma unroll
            for (int n = 0; n < NT; n++)
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    int key = k0 + n * 8 + col_lo + (j & 1);
                    int pos = (j >> 1) ? pos_hi : pos_lo;
                    float sv = (sa[n][j] + sb[n][j]) * scale_log2;
                    sv = (key <= pos && key < seq_kv) ? sv : NEG_INF;
                    sa[n][j] = sv;
                    if (j >> 1) tmax_hi = fmaxf(tmax_hi, sv); else tmax_lo = fmaxf(tmax_lo, sv);
                }
            }
            tmax_lo = fmaxf(tmax_lo, __shfl_xor_sync(0xffffffffu, tmax_lo, 1));
            tmax_lo = fmaxf(tmax_lo, __shfl_xor_sync(0xffffffffu, tmax_lo, 2));
            tmax_hi = fmaxf(tmax_hi, __shfl_xor_sync(0xffffffffu, tmax_hi, 1));
            tmax_hi = fmaxf(tmax_hi, __shfl_xor_sync(0xffffffffu, tmax_hi, 2));

            const float mn_lo = fmaxf(m_lo, tmax_lo);
            const float mn_hi = fmaxf(m_hi, tmax_hi);
            const float mu_lo = (mn_lo == NEG_INF) ? 0.0f : mn_lo;   // avoid (-inf) - (-inf)
            const float mu_hi = (mn_hi == NEG_INF) ? 0.0f : mn_hi;

            // Rescale O / l only when some row of this warp saw a new max.
            const bool changed = (mn_lo != m_lo) || (mn_hi != m_hi);
            if (__any_sync(0xffffffffu, changed))
            {
                const float c_lo = exp2f(m_lo - mu_lo);   // exp2(-inf) = 0 on the first tile
                const float c_hi = exp2f(m_hi - mu_hi);
                l_lo *= c_lo; l_hi *= c_hi;
                #pragma unroll
                for (int n = 0; n < HEAD_DIM / 8; n++)
                { o[n][0] *= c_lo; o[n][1] *= c_lo; o[n][2] *= c_hi; o[n][3] *= c_hi; }
            }
            m_lo = mn_lo; m_hi = mn_hi;

            // ---- P = exp2(S - m), accumulate l, pack as A fragments ----
            unsigned pa[NT / 2][4];
            #pragma unroll
            for (int n = 0; n < NT; n++)
            {
                float p0 = exp2f(sa[n][0] - mu_lo);
                float p1 = exp2f(sa[n][1] - mu_lo);
                float p2 = exp2f(sa[n][2] - mu_hi);
                float p3 = exp2f(sa[n][3] - mu_hi);
                l_lo += p0 + p1; l_hi += p2 + p3;
                // C frag (rows r, r+8; keys n*8+col_lo..+1) -> A frag of the k16 step n/2.
                pa[n >> 1][(n & 1) * 2 + 0] = pack_half2(p0, p1);
                pa[n >> 1][(n & 1) * 2 + 1] = pack_half2(p2, p3);
            }

            // Staging loaders decode tile t+1 in place DURING this warp's PV MMAs: one K+V row pair
            // of this warp after PV steps PREFILL_STAGE_NP0 + PREFILL_STAGE_EVERY*s (3, 7, 11), so
            // the decode instructions issue while the tensor pipe works on the preceding HMMAs. Tile t+1's
            // copies were issued at the top of this iteration and had the QK phase to land; its
            // buffer is not read in this iteration and the closing barrier publishes it.
            // Compile-time false for F16.
            const bool stage_next = KL::kStaged && t + 1 < num_tiles;
            half* const tk_next = sK + (size_t)((t + 1) & (NBUF - 1)) * KV_TILE * SMEM_STRIDE;
            half* const tv_next = sV + (size_t)((t + 1) & (NBUF - 1)) * KV_TILE * SMEM_STRIDE;
            if (stage_next) { cp_async_wait<0>(); __syncwarp(); }

            // ---- O += P . V ----
            #pragma unroll
            for (int kk = 0; kk < KV_TILE / 16; kk++)
            {
                #pragma unroll
                for (int np = 0; np < HEAD_DIM / 16; np++)
                {
                    unsigned b[4];
                    ldmatrix_x4_trans(b, vb + (size_t)(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * SMEM_STRIDE
                                              + (np * 2 + (lane >> 4)) * 8);
                    mma_m16n8k16(o[2 * np], pa[kk], b[0], b[1]);
                    mma_m16n8k16(o[2 * np + 1], pa[kk], b[2], b[3]);
                    if (stage_next && kk == 0 && np >= PREFILL_STAGE_NP0
                        && (np - PREFILL_STAGE_NP0) % PREFILL_STAGE_EVERY == 0
                        && np < PREFILL_STAGE_NP0 + PREFILL_STAGE_EVERY * KL::kSlots)
                    {
                        const int r = warp + (np - PREFILL_STAGE_NP0) / PREFILL_STAGE_EVERY * group;
                        if (r < KV_TILE) stage_decode_row<KL, VL>(tk_next, tv_next, r);
                    }
                }
            }
            // Rows past the interleaved slots (group < 6 only).
            if (stage_next)
            {
                for (int r = warp + KL::kSlots * group; r < KV_TILE; r += group)
                    stage_decode_row<KL, VL>(tk_next, tv_next, r);
            }
        }
        __syncthreads();   // all warps done with this buffer before it is refilled
    }

    // ---- epilogue: O / l -> F32 output ----
    l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 1);
    l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 2);
    l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 1);
    l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 2);
    const float inv_lo = (l_lo > 0.0f) ? 1.0f / l_lo : 0.0f;
    const float inv_hi = (l_hi > 0.0f) ? 1.0f / l_hi : 0.0f;

    if (warp < group)
    {
        const int row_lo = qbase + r_lo;
        const int row_hi = row_lo + 8;
        float* out_lo = output + (size_t)row_lo * q_stride + (size_t)hq * HEAD_DIM + col_lo;
        float* out_hi = output + (size_t)row_hi * q_stride + (size_t)hq * HEAD_DIM + col_lo;
        #pragma unroll
        for (int n = 0; n < HEAD_DIM / 8; n++)
        {
            if (row_lo < seq_q)
                *reinterpret_cast<float2*>(out_lo + n * 8) = make_float2(o[n][0] * inv_lo, o[n][1] * inv_lo);
            if (row_hi < seq_q)
                *reinterpret_cast<float2*>(out_hi + n * 8) = make_float2(o[n][2] * inv_hi, o[n][3] * inv_hi);
        }
    }
}
