// Hand-fused FP16 flash-attention prefill kernel using Ampere tensor cores
// (mma.sync.m16n8k16). Prototype for the long-context go/no-go: the s x s scores
// are computed, softmaxed, and consumed entirely in shared memory / registers and
// are NEVER materialised to global memory (unlike the G3 cuBLAS+softmax path which
// round-trips numHeads * s^2 FP16 through HBM). One warp per (query head, 16-query
// tile); the KV axis is streamed in tiles of 16 keys with online softmax.
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

// One WARP per (query head, 16-query tile).
//   gridDim.x = numHeads, gridDim.y = ceil(seq / 16), blockDim.x = 32.
extern "C" __global__ void __launch_bounds__(32) attention_flash_mma_f16(
    const half* __restrict__ q,   // [seq, numHeads,   headDim] row-major
    const half* __restrict__ k,   // [seq, numKvHeads, headDim] row-major
    const half* __restrict__ v,   // [seq, numKvHeads, headDim] row-major
    half* __restrict__ output,    // [seq, numHeads,   headDim] row-major
    const int seq,
    const int num_heads,
    const int num_kv_heads,
    const float scale)
{
    const int lane = threadIdx.x;            // 0..31
    const int hq = blockIdx.x;               // query head
    const int q_tile = blockIdx.y;           // which 16-query tile
    const int q0 = q_tile * Q_TILE;          // first query row of this tile
    if (q0 >= seq) return;

    const int group = num_heads / num_kv_heads;
    const int hk = hq / group;               // kv head feeding this query head

    const int q_stride = num_heads * HEAD_DIM;
    const int kv_stride = num_kv_heads * HEAD_DIM;

    // Shared tiles. Q persists for the whole key loop; K/V refilled per KV tile.
    __shared__ half sQ[Q_TILE * HEAD_DIM];        // [q][d]
    __shared__ half sK[KV_TILE * HEAD_DIM];       // [key][d]
    __shared__ half sVt[HEAD_DIM * KV_TILE];      // V transposed: [d][key]
    __shared__ half sP[Q_TILE * KV_TILE];         // normalised P: [q][key]
    __shared__ float sScore[Q_TILE * KV_TILE];    // raw S spilled from accumulator
    __shared__ float sM[Q_TILE];                  // running row max
    __shared__ float sL[Q_TILE];                  // running row denom (sum exp)
    __shared__ float sCorr[Q_TILE];               // per-row O rescale for this KV tile

    // Load Q tile into shared (row-major [q][d]). 16*64 = 1024 halves / 32 lanes = 32 each.
    for (int i = lane; i < Q_TILE * HEAD_DIM; i += 32)
    {
        int qr = i / HEAD_DIM, d = i % HEAD_DIM;
        int gq = q0 + qr;
        sQ[i] = (gq < seq) ? q[(size_t)gq * q_stride + hq * HEAD_DIM + d] : __float2half(0.0f);
    }

    // O accumulator: 16 queries x 64 headDim = 8 n-subtiles, each a {f32 x4} frag.
    // O_frag[nsub][0..3] follows the C/D map: lane L -> rows (L>>2),(L>>2)+8 and
    // cols within the subtile (L%4)*2+{0,1}.
    float O_frag[HEAD_DIM / 8][4];
    #pragma unroll
    for (int n = 0; n < HEAD_DIM / 8; n++)
        O_frag[n][0] = O_frag[n][1] = O_frag[n][2] = O_frag[n][3] = 0.0f;

    if (lane < Q_TILE) { sM[lane] = -FLT_MAX; sL[lane] = 0.0f; }
    __syncwarp();

    // This lane owns C/D rows r0 and r0+8, and within an 8-wide n-subtile the two
    // columns col_lo, col_hi.
    const int r0 = lane >> 2;
    const int col_lo = (lane & 3) * 2;
    const int col_hi = col_lo + 1;

    // Stream the key axis. Causal: keys only up to the last query in this tile.
    const int kv_end = min(seq, q0 + Q_TILE);
    for (int k0 = 0; k0 < kv_end; k0 += KV_TILE)
    {
        // Load K tile [key][d] and V transposed [d][key].
        for (int i = lane; i < KV_TILE * HEAD_DIM; i += 32)
        {
            int kr = i / HEAD_DIM, d = i % HEAD_DIM;
            int gk = k0 + kr;
            half kv = (gk < seq) ? k[(size_t)gk * kv_stride + hk * HEAD_DIM + d] : __float2half(0.0f);
            sK[i] = kv;
            half vv = (gk < seq) ? v[(size_t)gk * kv_stride + hk * HEAD_DIM + d] : __float2half(0.0f);
            sVt[(size_t)d * KV_TILE + kr] = vv;   // transpose into [d][key]
        }
        __syncwarp();

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
            const half* aptr = &sQ[(lane & 15) * HEAD_DIM + ks * 16 + (lane >> 4) * 8];
            ldmatrix_x4(a, aptr);

            #pragma unroll
            for (int ns = 0; ns < 2; ns++)
            {
                // B = K[8 x 16] for keys [ns*8, ns*8+8): ldmatrix.x2 over [key][d].
                unsigned b[2];
                const half* bptr = &sK[(ns * 8 + (lane & 7)) * HEAD_DIM + ks * 16 + ((lane >> 3) & 1) * 8];
                ldmatrix_x2(b, bptr);
                mma_m16n8k16(S_frag[ns], a, b);
            }
        }

        // Spill S to shared [q][key] via the C/D map, applying the QK scale.
        #pragma unroll
        for (int ns = 0; ns < 2; ns++)
        {
            int kcol = ns * 8 + col_lo;
            sScore[r0 * KV_TILE + kcol]       = S_frag[ns][0] * scale;
            sScore[r0 * KV_TILE + ns * 8 + col_hi] = S_frag[ns][1] * scale;
            sScore[(r0 + 8) * KV_TILE + kcol] = S_frag[ns][2] * scale;
            sScore[(r0 + 8) * KV_TILE + ns * 8 + col_hi] = S_frag[ns][3] * scale;
        }
        __syncwarp();

        // ---- Online softmax over this KV tile, one lane per query row (16 rows) ----
        if (lane < Q_TILE)
        {
            int gq = q0 + lane;
            int causal_last = gq;            // query gq attends keys 0..gq
            float m_prev = sM[lane];
            float l_prev = sL[lane];

            // tile-local max over valid (causal, in-range) keys
            float m_cur = m_prev;
            for (int j = 0; j < KV_TILE; j++)
            {
                int gk = k0 + j;
                if (gk < seq && gk <= causal_last)
                {
                    float s = sScore[lane * KV_TILE + j];
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
                    p = __expf(sScore[lane * KV_TILE + j] - m_cur);
                    l_cur += p;
                }
                sP[lane * KV_TILE + j] = __float2half(p);
            }

            sM[lane] = m_cur;
            sL[lane] = l_cur;
            sCorr[lane] = correction;
        }
        __syncwarp();

        // Rescale O accumulator by this tile's correction, then add P.V.
        // correction for the two rows this lane owns:
        float corr0 = sCorr[r0];
        float corr1 = sCorr[r0 + 8];
        #pragma unroll
        for (int n = 0; n < HEAD_DIM / 8; n++)
        {
            O_frag[n][0] *= corr0; O_frag[n][1] *= corr0;
            O_frag[n][2] *= corr1; O_frag[n][3] *= corr1;
        }

        // ---- PV: O[16 x 64] += P[16 x 16] . V[16 x 64], 8 n-subtiles, 1 k-step ----
        // A = P[16x16] from sP via ldmatrix.x4.
        unsigned pa[4];
        const half* pptr = &sP[(lane & 15) * KV_TILE + (lane >> 4) * 8];
        ldmatrix_x4(pa, pptr);

        #pragma unroll
        for (int n = 0; n < HEAD_DIM / 8; n++)   // each subtile = 8 headDim cols
        {
            // B = V^T[8 x 16] = sVt[d in n*8..n*8+8][key 0..15]. mma B is [n=8 x k=16].
            unsigned vb[2];
            const half* vptr = &sVt[(n * 8 + (lane & 7)) * KV_TILE + ((lane >> 3) & 1) * 8];
            ldmatrix_x2(vb, vptr);
            mma_m16n8k16(O_frag[n], pa, vb);
        }
        __syncwarp();
    }

    // Final normalisation by the row denom and write row-major [seq, heads, headDim].
    #pragma unroll
    for (int n = 0; n < HEAD_DIM / 8; n++)
    {
        int gr0 = q0 + r0;
        int gr1 = q0 + r0 + 8;
        float inv0 = (sL[r0] > 0.0f) ? 1.0f / sL[r0] : 0.0f;
        float inv1 = (sL[r0 + 8] > 0.0f) ? 1.0f / sL[r0 + 8] : 0.0f;
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
