// attention_flash_mma_prefill_kvq.cu -- issue #1850.
//
// Tensor-core (mma.sync.m16n8k16) flash-style causal PREFILL attention for the dense quantized KV cache
// (CudaQuantizedKvCache), head_dim 64 and 128, reading KV-format Q8_0 / Q4_0 blocks IN PLACE: each K/V
// tile is dequantized in registers as it is loaded into shared memory, so prefill needs no full-history
// FP16 scratch (CLAUDE.md rule 2). Covers the cold prefill (position_offset 0) and the #1847 prefix-reuse
// suffix prefill (position_offset > 0) with one kernel.
//
// ─── Relation to attention_flash_mma_prefill_hd256.cuh (#962 / #1094) ─────────────────────────────────
// Same FlashAttention-2 design, copied rather than shared so the validated hd256 PTX cannot move:
//   * Grid = (ceil(seq_q/16) q-tiles, num_kv_heads), heaviest causal tiles first. Block = `group` warps
//     (one per query head of the GQA group, group <= 8), all sharing the same 16 query rows, so each K/V
//     tile is fetched once and reused `group` times out of shared memory.
//   * Per warp: 16 x HD F32 O accumulator in registers (mma C layout), F32 running max / sum, online
//     softmax in the exp2 domain, P re-packed from the S C-fragment into the PV A-fragment (no smem trip).
//   * Query row i has absolute position position_offset + i and attends keys 0..position_offset+i;
//     seq_kv >= position_offset + seq_q (AttentionSpanInvariant). Keys past seq_kv in the tail tile are
//     clamped to a valid row for the load and masked.
// Differences: HD is a template parameter (64 / 128), the output is F16 (the dense path's attention
// output buffer; no F32 storage), and the quantized rows are staged with 8-byte (hd128) / 4-byte (hd64)
// cp.async copies (#1923, PkLoadStaged below): a Q8_0 head-row is 136 B (hd128) / 68 B (hd64) and a Q4_0
// one 72 B / 36 B, none a multiple of 16, so #1094's 16-byte copies do not apply, but all are multiples
// of 8 / 4. The staged rows are decoded in place with the #988 block decoders (V2LoadQ8 / V2LoadQ4 in
// attention_mma_decode_v2.cuh, element = __float2half(d * (float)q), the eager dequant kernels'
// expression). Before #1923 each thread __ldg'd and decoded its chunk synchronously (PkLoadQuant).
// The F16 entry (cp.async; the dense F16 cache's prefill since #1923) runs the same body, so the
// quantized entry is bit-identical to dequant-into-F16 + the F16 entry.
//
// Double buffering is kept: tile t+1 is loaded at the top of iteration t into the buffer last read in
// iteration t-1 (released by that iteration's closing barrier), and published by iteration t's closing
// barrier -- for the synchronous loaders the stores simply complete before it. A staged tile t is
// decoded in place at the top of iteration t, after cp.async.wait and before the opening barrier (its
// raw copies were issued at the top of iteration t-1, so they overlapped tile t-1's MMAs).
//
// ─── Precision ─────────────────────────────────────────────────────────────────────────────────────────
// F16 Q/K/V/P operands, F32 mma accumulation and softmax statistics, in registers. Output F16.
//
// Dynamic shared memory: group*16*(HD+8)*2 (Q) + NBUF*2*KV_TILE*(HD+8)*2 (K, V); hd128 group 8 = 52 KB,
// so the launcher opts the functions into > 48 KB.
//
// Entries: attention_flash_mma_prefill_kvq_hd{64,128}_{q8q8,q4q4,q8q4,q4q8,f16f16}, and the TurboQuant in-place
// attention_flash_mma_prefill_kvq_hd{64,128}_tqtq (#1892, at the end of the file).

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include "attention_mma_decode_v2.cuh"   // V2LoadQ8 / V2LoadQ4 / V2LoadF16 (issue #988)

#define PK_Q_TILE 16
#define PK_KV_TILE 16
#define PK_NBUF 2
#define PK_NT (PK_KV_TILE / 8)
#define PK_LOG2E 1.4426950408889634f
#define PK_NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ void pk_ldmatrix_x4(unsigned (&r)[4], const void* p)
{
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void pk_ldmatrix_x4_trans(unsigned (&r)[4], const void* p)
{
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void pk_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void pk_cp_async16(void* smem_dst, const void* gmem_src)
{
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void pk_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void pk_cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ unsigned pk_pack_half2(float lo, float hi)
{
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

// Tile loaders: chunk(dst, base, elem) fills 8 consecutive halves of the shared tile at dst (16-byte
// aligned) from element elem (multiple of 8) of the logical [seq_kv, num_kv_heads*HD] KV grid at base.
struct PkLoadF16
{
    __device__ __forceinline__ static void chunk(half* dst, const void* base, size_t elem)
    {
        pk_cp_async16(dst, reinterpret_cast<const half*>(base) + elem);
    }
};

template <class L>
struct PkLoadQuant
{
    __device__ __forceinline__ static void chunk(half* dst, const void* base, size_t elem)
    {
        *reinterpret_cast<uint4*>(dst) = L::load8(base, elem);
    }
};

// ─── Raw cp.async staging for the Q8_0 / Q4_0 loaders (#1923, after #1094's hd256 PrefillTileLoadStaged) ───
// A raw head row (one key, one KV head) is HD/32 blocks: 136 / 72 B at hd128, 68 / 36 B at hd64. Not 16-byte multiples, but
// multiples of 8 (hd128) / 4 (hd64), and so is every head row's offset ((key*num_kv_heads + hkv) * row bytes from a page-aligned
// base; the launcher checks the base). So the row is copied with cp.async.ca of 8 / 4 bytes into the TAIL of its own padded F16
// slot ([slot - row, slot) of the (HD+8)*2-byte slot), and decoded in place once it has landed: same shared memory as before.
// Row r of a tile is owned by ONE warp for both the copy and the decode, so warp-level ordering suffices: each lane waits for
// its own copies, __syncwarp publishes the warp's rows, every lane decodes its chunk into registers, __syncwarp (all raw bytes
// read), then the F16 chunks are written (they overlap the raw tail: chunk c covers bytes [16c, 16c + 16)). The decode is the
// same V2LoadQ8 / V2LoadQ4::decode8 expression as the global loader, so the tile halves (and the output) are bit-identical.
template <class L, int HD>
struct PkLoadStaged
{
    static constexpr int kRowBytes = HD / 32 * L::kBlockBytes;      // 136 / 72 (hd128), 68 / 36 (hd64)
    static constexpr int kSlotBytes = (HD + 8) * 2;                  // 272 / 144
    static constexpr int kRawOff = kSlotBytes - kRowBytes;
    static constexpr int kUnit = (kRowBytes % 8 == 0) ? 8 : 4;      // cp.async.ca copy size
    static constexpr int kUnits = kRowBytes / kUnit;                 // 17 / 9
    static_assert(kRowBytes % kUnit == 0 && kRawOff % kUnit == 0, "raw row must be unit-copyable");
    static_assert(kRawOff >= 0, "raw row must fit its F16 slot");

    __device__ __forceinline__ static const uint8_t* row_src(const void* base, size_t head_row)
    {
        return reinterpret_cast<const uint8_t*>(base) + head_row * kRowBytes;
    }

    // Copy unit u of a raw head row into its slot's tail.
    __device__ __forceinline__ static void copy_unit(half* slot, const uint8_t* src, int u)
    {
        unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(slot) + kRawOff + u * kUnit));
        if constexpr (kUnit == 8)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(a), "l"(src + u * kUnit) : "memory");
        else
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(a), "l"(src + u * kUnit) : "memory");
    }

    // Decode 16-byte chunk c (elements 8c..8c+7) of the staged raw row in this slot.
    __device__ __forceinline__ static uint4 decode_chunk(const half* slot, int c)
    {
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(slot) + kRawOff;
        return L::template decode8<V2PlainLd>(raw + (c >> 2) * L::kBlockBytes, (c & 3) * 8);
    }
};

// Packed-half2 block decoders for the staged rows (#1923): same blocks and same results as V2LoadQ8 / V2LoadQ4::decode8, in about
// half the ALU work. An int8 (or nibble - 8) q is exact in F16, and d * q is exact in F32 (an 11-bit by 8-bit product), so the
// correctly rounded F16 product __float2half(d * (float)q) the eager dequant computes equals one mul.rn.f16x2 of the two halves.
// q is formed exactly as the F16 0x6400 | byte (= 1024 + byte) minus a constant (sub.rn.f16x2 of integers below 2048 is exact).
// Explicit .rn PTX keeps the sub and mul from being contracted into an fma.
__device__ __forceinline__ unsigned pk_hsub2_rn(unsigned a, unsigned b)
{
    unsigned r;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ unsigned pk_hmul2_rn(unsigned a, unsigned b)
{
    unsigned r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

struct PkQ8Fast
{
    static constexpr int kBlockBytes = 34;
    __device__ __forceinline__ static uint4 load8(const void* base, size_t elem)
    {
        return decode8<V2GlobalLd>(reinterpret_cast<const uint8_t*>(base) + (elem >> 5) * kBlockBytes, (int)(elem & 31));
    }
    template <class Ld>
    __device__ __forceinline__ static uint4 decode8(const uint8_t* b, const int j)
    {
        const unsigned d = Ld::u16(reinterpret_cast<const uint16_t*>(b));
        const unsigned d2 = d | (d << 16);
        const uint16_t* p = reinterpret_cast<const uint16_t*>(b + 2 + j);
        const unsigned w0 = (Ld::u16(p) | (Ld::u16(p + 1) << 16)) ^ 0x80808080u;   // int8 + 128, as unsigned bytes
        const unsigned w1 = (Ld::u16(p + 2) | (Ld::u16(p + 3) << 16)) ^ 0x80808080u;
        const unsigned bias = 0x64806480u;                                            // half2(1152, 1152) = 1024 + 128
        unsigned out[4];
        out[0] = pk_hmul2_rn(pk_hsub2_rn(__byte_perm(w0, 0x64646464u, 0x5140), bias), d2);
        out[1] = pk_hmul2_rn(pk_hsub2_rn(__byte_perm(w0, 0x64646464u, 0x7362), bias), d2);
        out[2] = pk_hmul2_rn(pk_hsub2_rn(__byte_perm(w1, 0x64646464u, 0x5140), bias), d2);
        out[3] = pk_hmul2_rn(pk_hsub2_rn(__byte_perm(w1, 0x64646464u, 0x7362), bias), d2);
        return make_uint4(out[0], out[1], out[2], out[3]);
    }
};

struct PkQ4Fast
{
    static constexpr int kBlockBytes = 18;
    __device__ __forceinline__ static uint4 load8(const void* base, size_t elem)
    {
        return decode8<V2GlobalLd>(reinterpret_cast<const uint8_t*>(base) + (elem >> 5) * kBlockBytes, (int)(elem & 31));
    }
    template <class Ld>
    __device__ __forceinline__ static uint4 decode8(const uint8_t* b, const int j)
    {
        const unsigned d = Ld::u16(reinterpret_cast<const uint16_t*>(b));
        const unsigned d2 = d | (d << 16);
        const uint16_t* p = reinterpret_cast<const uint16_t*>(b + 2 + (j >> 1));
        const unsigned w = Ld::u16(p) | (Ld::u16(p + 1) << 16);                     // 4 bytes = 8 nibbles
        const unsigned lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
        const unsigned u01 = __byte_perm(lo, hi, 0x5140);                             // lo0 hi0 lo1 hi1
        const unsigned u23 = __byte_perm(lo, hi, 0x7362);                             // lo2 hi2 lo3 hi3
        const unsigned bias = 0x64086408u;                                            // half2(1032, 1032) = 1024 + 8
        unsigned out[4];
        out[0] = pk_hmul2_rn(pk_hsub2_rn(__byte_perm(u01, 0x64646464u, 0x5140), bias), d2);
        out[1] = pk_hmul2_rn(pk_hsub2_rn(__byte_perm(u01, 0x64646464u, 0x7362), bias), d2);
        out[2] = pk_hmul2_rn(pk_hsub2_rn(__byte_perm(u23, 0x64646464u, 0x5140), bias), d2);
        out[3] = pk_hmul2_rn(pk_hsub2_rn(__byte_perm(u23, 0x64646464u, 0x7362), bias), d2);
        return make_uint4(out[0], out[1], out[2], out[3]);
    }
};

template <class L> struct PkStagedTraits { static constexpr bool staged = false; };
template <class L, int HD> struct PkStagedTraits<PkLoadStaged<L, HD>> { static constexpr bool staged = true; };

template <int HD, class KL, class VL>
__device__ __forceinline__ void pk_prefill_body(
    const half* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    half* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int position_offset,
    const float scale)
{
    constexpr int SS = HD + 8;           // padded row stride (halves): ldmatrix rows in distinct banks
    constexpr int CPR = HD / 8;          // 16-byte chunks per row
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = num_heads / num_kv_heads;
    const int nq_tiles = (seq_q + PK_Q_TILE - 1) / PK_Q_TILE;
    const int qt = nq_tiles - 1 - (int)blockIdx.x;      // heaviest causal tiles first
    const int hkv = blockIdx.y;
    const int hq = hkv * group + warp;
    const int qbase = qt * PK_Q_TILE;
    const int q_stride = num_heads * HD;
    const int kv_stride = num_kv_heads * HD;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* sQ = reinterpret_cast<half*>(smem_raw);                  // [group][16][SS]
    half* sK = sQ + group * PK_Q_TILE * SS;                        // [NBUF][KV_TILE][SS]
    half* sV = sK + PK_NBUF * PK_KV_TILE * SS;                     // [NBUF][KV_TILE][SS]

    for (int i = threadIdx.x; i < group * PK_Q_TILE * CPR; i += blockDim.x)
    {
        int c = i % CPR;
        int r = (i / CPR) % PK_Q_TILE;
        int w = i / (PK_Q_TILE * CPR);
        int qrow = qbase + r;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (qrow < seq_q)
            val = *reinterpret_cast<const uint4*>(
                q + (size_t)qrow * q_stride + (size_t)(hkv * group + w) * HD + c * 8);
        *reinterpret_cast<uint4*>(sQ + (size_t)(w * PK_Q_TILE + r) * SS + c * 8) = val;
    }

    int kv_end = position_offset + qbase + PK_Q_TILE;
    if (kv_end > seq_kv) kv_end = seq_kv;
    const int num_tiles = (kv_end + PK_KV_TILE - 1) / PK_KV_TILE;

    // #1923 staged loaders: row group g (RP rows) of a tile is owned by warp g % nwarps for the raw copy and the decode.
    constexpr bool kStaged = PkStagedTraits<KL>::staged;
    static_assert(PkStagedTraits<KL>::staged == PkStagedTraits<VL>::staged, "K and V are both staged or neither");
    constexpr int RJ = 2 * CPR;               // decode jobs per row: CPR K chunks + CPR V chunks
    constexpr int RP = 32 / RJ;               // rows per warp pass: 1 (hd128), 2 (hd64)
    constexpr int NG = PK_KV_TILE / RP;
    const int nwarps = blockDim.x >> 5;

    auto load_tile = [&](int t, int buf)
    {
        const int k0 = t * PK_KV_TILE;
        half* dK = sK + (size_t)buf * PK_KV_TILE * SS;
        half* dV = sV + (size_t)buf * PK_KV_TILE * SS;
        if constexpr (kStaged)
        {
            for (int g = warp; g < NG; g += nwarps)
            {
                for (int x = lane; x < RP * KL::kUnits; x += 32)
                {
                    const int r = g * RP + x / KL::kUnits;
                    int key = k0 + r;
                    if (key > seq_kv - 1) key = seq_kv - 1;  // clamp: masked later, stays finite
                    KL::copy_unit(dK + (size_t)r * SS, KL::row_src(k, (size_t)key * num_kv_heads + hkv), x % KL::kUnits);
                }
                for (int x = lane; x < RP * VL::kUnits; x += 32)
                {
                    const int r = g * RP + x / VL::kUnits;
                    int key = k0 + r;
                    if (key > seq_kv - 1) key = seq_kv - 1;
                    VL::copy_unit(dV + (size_t)r * SS, VL::row_src(v, (size_t)key * num_kv_heads + hkv), x % VL::kUnits);
                }
            }
        }
        else
        {
            for (int i = threadIdx.x; i < PK_KV_TILE * CPR; i += blockDim.x)
            {
                int c = i % CPR, r = i / CPR;
                int key = k0 + r;
                if (key > seq_kv - 1) key = seq_kv - 1;          // clamp: masked later, stays finite
                size_t g = (size_t)key * kv_stride + (size_t)hkv * HD + c * 8;
                KL::chunk(dK + (size_t)r * SS + c * 8, k, g);
                VL::chunk(dV + (size_t)r * SS + c * 8, v, g);
            }
        }
    };

    // Staged only: decode row group g of buffer buf in place. The caller has waited for this thread's copies of it and
    // __syncwarp'd (the warp's raw rows are complete and visible to all its lanes).
    auto decode_group = [&](int buf, int g)
    {
        if constexpr (kStaged)
        {
            const int j = lane % RJ;
            const bool isV = j >= CPR;
            const int c = isV ? j - CPR : j;
            half* slot = (isV ? sV : sK) + (size_t)buf * PK_KV_TILE * SS + (size_t)(g * RP + lane / RJ) * SS;
            const uint4 h = isV ? VL::decode_chunk(slot, c) : KL::decode_chunk(slot, c);
            __syncwarp();                     // every raw byte of these rows read before any F16 chunk overwrites it
            *reinterpret_cast<uint4*>(slot + c * 8) = h;
        }
    };
    auto decode_tile = [&](int buf)
    {
        if constexpr (kStaged)
        {
            __syncwarp();
            for (int g = warp; g < NG; g += nwarps) decode_group(buf, g);
        }
    };
    // Staged tile t+1 is decoded inside tile t's P.V loop (#1923, as #1094's hd256 kernel does): its copies were issued at the top
    // of iteration t and land during QK; this warp waits for them after P.V step 0 and decodes its row groups spread over the
    // remaining steps, so the shared-memory loads / ALU / stores issue while the tensor pipe runs the preceding HMMAs.
    // PK_STAGED_PV_DECODE=0 decodes at the top of iteration t instead (before the opening barrier; A/B only).
#ifndef PK_STAGED_PV_DECODE
#define PK_STAGED_PV_DECODE 1
#endif
    constexpr int NPV = HD / 16;             // P.V steps per tile (8 at hd128, 4 at hd64)
    const int my_groups = (NG - warp + nwarps - 1) / nwarps;

    float o[HD / 8][4];
    #pragma unroll
    for (int n = 0; n < HD / 8; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f; }
    float m_lo = PK_NEG_INF, m_hi = PK_NEG_INF;
    float l_lo = 0.0f, l_hi = 0.0f;

    const float scale_log2 = scale * PK_LOG2E;
    const int r_lo = lane >> 2;
    const int col_lo = (lane & 3) * 2;
    const int pos_lo = position_offset + qbase + r_lo;
    const int pos_hi = pos_lo + 8;
    const half* qw = sQ + (size_t)warp * PK_Q_TILE * SS;

    if (num_tiles > 0)
    {
        load_tile(0, 0);
        pk_cp_async_commit();
        if constexpr (kStaged && PK_STAGED_PV_DECODE) { pk_cp_async_wait<0>(); decode_tile(0); }
    }

    for (int t = 0; t < num_tiles; t++)
    {
        const int buf = t & 1;
        if (t + 1 < num_tiles) load_tile(t + 1, (t + 1) & 1);
        pk_cp_async_commit();
        pk_cp_async_wait<1>();
        if constexpr (kStaged && !PK_STAGED_PV_DECODE) decode_tile(buf);
        __syncthreads();

        if (warp < group)
        {
            const half* kb = sK + (size_t)buf * PK_KV_TILE * SS;
            const half* vb = sV + (size_t)buf * PK_KV_TILE * SS;
            const int k0 = t * PK_KV_TILE;

            // ---- S = Q . K^T ----
            float sa[PK_NT][4], sb[PK_NT][4];
            #pragma unroll
            for (int n = 0; n < PK_NT; n++)
            { sa[n][0] = sa[n][1] = sa[n][2] = sa[n][3] = 0.f; sb[n][0] = sb[n][1] = sb[n][2] = sb[n][3] = 0.f; }

            #pragma unroll
            for (int ks = 0; ks < HD / 16; ks++)
            {
                unsigned a[4];
                pk_ldmatrix_x4(a, qw + (size_t)(lane & 15) * SS + ks * 16 + (lane >> 4) * 8);
                #pragma unroll
                for (int np = 0; np < PK_NT / 2; np++)
                {
                    unsigned b[4];
                    pk_ldmatrix_x4(b, kb + (size_t)(np * 16 + (lane & 7) + ((lane >> 4) & 1) * 8) * SS
                                         + ks * 16 + ((lane >> 3) & 1) * 8);
                    if ((ks & 1) == 0)
                    {
                        pk_mma(sa[2 * np], a, b[0], b[1]);
                        pk_mma(sa[2 * np + 1], a, b[2], b[3]);
                    }
                    else
                    {
                        pk_mma(sb[2 * np], a, b[0], b[1]);
                        pk_mma(sb[2 * np + 1], a, b[2], b[3]);
                    }
                }
            }

            // ---- scale, causal/tail mask, tile row max ----
            float tmax_lo = PK_NEG_INF, tmax_hi = PK_NEG_INF;
            #pragma unroll
            for (int n = 0; n < PK_NT; n++)
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    int key = k0 + n * 8 + col_lo + (j & 1);
                    int pos = (j >> 1) ? pos_hi : pos_lo;
                    float sv = (sa[n][j] + sb[n][j]) * scale_log2;
                    sv = (key <= pos && key < seq_kv) ? sv : PK_NEG_INF;
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
            const float mu_lo = (mn_lo == PK_NEG_INF) ? 0.0f : mn_lo;
            const float mu_hi = (mn_hi == PK_NEG_INF) ? 0.0f : mn_hi;

            const bool changed = (mn_lo != m_lo) || (mn_hi != m_hi);
            if (__any_sync(0xffffffffu, changed))
            {
                const float c_lo = exp2f(m_lo - mu_lo);
                const float c_hi = exp2f(m_hi - mu_hi);
                l_lo *= c_lo; l_hi *= c_hi;
                #pragma unroll
                for (int n = 0; n < HD / 8; n++)
                { o[n][0] *= c_lo; o[n][1] *= c_lo; o[n][2] *= c_hi; o[n][3] *= c_hi; }
            }
            m_lo = mn_lo; m_hi = mn_hi;

            // ---- P = exp2(S - m), accumulate l, pack as A fragments ----
            unsigned pa[PK_NT / 2][4];
            #pragma unroll
            for (int n = 0; n < PK_NT; n++)
            {
                float p0 = exp2f(sa[n][0] - mu_lo);
                float p1 = exp2f(sa[n][1] - mu_lo);
                float p2 = exp2f(sa[n][2] - mu_hi);
                float p3 = exp2f(sa[n][3] - mu_hi);
                l_lo += p0 + p1; l_hi += p2 + p3;
                pa[n >> 1][(n & 1) * 2 + 0] = pk_pack_half2(p0, p1);
                pa[n >> 1][(n & 1) * 2 + 1] = pk_pack_half2(p2, p3);
            }

            // ---- O += P . V ----
            #pragma unroll
            for (int kk = 0; kk < PK_KV_TILE / 16; kk++)
            {
                #pragma unroll
                for (int np = 0; np < HD / 16; np++)
                {
                    unsigned b[4];
                    pk_ldmatrix_x4_trans(b, vb + (size_t)(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * SS
                                               + (np * 2 + (lane >> 4)) * 8);
                    pk_mma(o[2 * np], pa[kk], b[0], b[1]);
                    pk_mma(o[2 * np + 1], pa[kk], b[2], b[3]);
                    if constexpr (kStaged && PK_STAGED_PV_DECODE && PK_KV_TILE == 16)
                    {
                        if (np >= 1 && t + 1 < num_tiles)
                        {
                            if (np == 1) { pk_cp_async_wait<0>(); __syncwarp(); }
                            for (int u = (np - 1) * my_groups / (NPV - 1); u < np * my_groups / (NPV - 1); u++)
                                decode_group((t + 1) & 1, warp + u * nwarps);
                        }
                    }
                }
            }
        }
        __syncthreads();   // all warps done with this buffer before it is refilled
    }

    // ---- epilogue: O / l -> F16 output ----
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
        half* out_lo = output + (size_t)row_lo * q_stride + (size_t)hq * HD + col_lo;
        half* out_hi = output + (size_t)row_hi * q_stride + (size_t)hq * HD + col_lo;
        #pragma unroll
        for (int n = 0; n < HD / 8; n++)
        {
            if (row_lo < seq_q)
                *reinterpret_cast<__half2*>(out_lo + n * 8) = __floats2half2_rn(o[n][0] * inv_lo, o[n][1] * inv_lo);
            if (row_hi < seq_q)
                *reinterpret_cast<__half2*>(out_hi + n * 8) = __floats2half2_rn(o[n][2] * inv_hi, o[n][3] * inv_hi);
        }
    }
}

// Quantized loader variant (#1923 A/B): 0 = synchronous __ldg + V2 decode (pre-#1923), 1 = synchronous __ldg + packed-half2 decode,
// 2 = raw cp.async staging + packed-half2 decode (PK_STAGED_PV_DECODE picks where the decode runs). All are bit-identical.
#ifndef PK_KVQ_LOADER
#define PK_KVQ_LOADER 1
#endif
#if PK_KVQ_LOADER == 0
using PkStQ8_128 = PkLoadQuant<V2LoadQ8>;
using PkStQ4_128 = PkLoadQuant<V2LoadQ4>;
using PkStQ8_64 = PkLoadQuant<V2LoadQ8>;
using PkStQ4_64 = PkLoadQuant<V2LoadQ4>;
#elif PK_KVQ_LOADER == 1
using PkStQ8_128 = PkLoadQuant<PkQ8Fast>;
using PkStQ4_128 = PkLoadQuant<PkQ4Fast>;
using PkStQ8_64 = PkLoadQuant<PkQ8Fast>;
using PkStQ4_64 = PkLoadQuant<PkQ4Fast>;
#else
using PkStQ8_128 = PkLoadStaged<PkQ8Fast, 128>;
using PkStQ4_128 = PkLoadStaged<PkQ4Fast, 128>;
using PkStQ8_64 = PkLoadStaged<PkQ8Fast, 64>;
using PkStQ4_64 = PkLoadStaged<PkQ4Fast, 64>;
#endif

#define PK_ENTRY(NAME, HD, KLOAD, VLOAD)                                                          \
extern "C" __global__ void __launch_bounds__(256) NAME(                                           \
    const half* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v,           \
    half* __restrict__ output, const int seq_q, const int seq_kv, const int num_heads,            \
    const int num_kv_heads, const int position_offset, const float scale)                         \
{                                                                                                 \
    pk_prefill_body<HD, KLOAD, VLOAD>(q, k, v, output, seq_q, seq_kv, num_heads, num_kv_heads,    \
                                      position_offset, scale);                                    \
}

PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_q8q8, 128, PkStQ8_128, PkStQ8_128)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_q4q4, 128, PkStQ4_128, PkStQ4_128)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_q8q4, 128, PkStQ8_128, PkStQ4_128)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_q4q8, 128, PkStQ4_128, PkStQ8_128)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd128_f16f16, 128, PkLoadF16, PkLoadF16)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_q8q8, 64, PkStQ8_64, PkStQ8_64)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_q4q4, 64, PkStQ4_64, PkStQ4_64)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_q8q4, 64, PkStQ8_64, PkStQ4_64)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_q4q8, 64, PkStQ4_64, PkStQ8_64)
PK_ENTRY(attention_flash_mma_prefill_kvq_hd64_f16f16, 64, PkLoadF16, PkLoadF16)

// ─── TurboQuant in place (#1892 Task 6) ────────────────────────────────────────────────────────────────
// The dense in-place TurboQuant cache (CudaTurboQuantInPlaceKvCache) stores each K / V head vector as mse_bits-bit Lloyd-Max
// codes (LSB-first, coordinate i at bit i*mse_bits, code_uints_per_vec words per (position, kv head)) plus one norm per head
// vector. Attention runs in code space: the caller rotates Q into K code space before this kernel (tq_rotate_q_f16) and the
// F16 output back out of V code space after it (tq_rotate_o_f16), so a tile element is simply __float2half(norm * centroid[code])
// -- the same single F16 rounding of an mma operand the F16 / q8 / q4 loaders apply, in code space.
//
// The loader needs per-side context (codes, norms, centroids, bits, stride, norm format) that the stateless chunk(dst, base,
// elem) contract cannot carry, so the TQ entries pass a pointer to a per-side context struct as `base`. Everything is
// force-inlined into the entry, so the struct lives in registers (the PTX has no local depot). pk_prefill_body and the ten
// entries above are untouched, so their PTX cannot move. The centroid table is read with __ldg from global memory, never
// staged in shared memory; the loader is register-only, so the shared-memory size is the same as the q8 / q4 entries'.
// `elem` = key * num_kv_heads * HD + hkv * HD + c * 8, so the head vector is elem / HD and the first coordinate elem % HD.
struct PkTqSide
{
    const uint32_t* codes;
    const void* norms;
    const float* centroids;
    int bits;
    int code_uints_per_vec;
    int norm_format;   // 0 = F32 reference, 1 = F16, 2 = BF16 (turboquant.cu's tq_load_norm)
};

__device__ __forceinline__ float pk_tq_norm(const void* norms, const int fmt, const size_t i)
{
    if (fmt == 0) return __ldg(reinterpret_cast<const float*>(norms) + i);
    const unsigned short h = __ldg(reinterpret_cast<const unsigned short*>(norms) + i);
    if (fmt == 1) return __half2float(__ushort_as_half(h));
    return __uint_as_float(((unsigned)h) << 16);
}

template <int HD>
struct PkLoadTq
{
    __device__ __forceinline__ static void chunk(half* dst, const void* base, size_t elem)
    {
        const PkTqSide& s = *reinterpret_cast<const PkTqSide*>(base);
        const size_t hv = elem / HD;
        const unsigned bit0 = (unsigned)(elem % HD) * (unsigned)s.bits;
        const uint32_t* w = s.codes + hv * (size_t)s.code_uints_per_vec + (bit0 >> 5);
        const unsigned shift = bit0 & 31u;
        const unsigned need = shift + 8u * (unsigned)s.bits;   // bits to read starting at w[0]; reads stay inside the vector
        unsigned long long win = __ldg(w);
        if (need > 32u) win |= (unsigned long long)__ldg(w + 1) << 32;
        win >>= shift;
        if (need > 64u) win |= (unsigned long long)__ldg(w + 2) << (64u - shift);
        const unsigned mask = (1u << s.bits) - 1u;
        const float n = pk_tq_norm(s.norms, s.norm_format, hv);
        unsigned packed[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const float lo = __ldg(&s.centroids[(unsigned)(win >> ((2 * j) * s.bits)) & mask]);
            const float hi = __ldg(&s.centroids[(unsigned)(win >> ((2 * j + 1) * s.bits)) & mask]);
            packed[j] = pk_pack_half2(n * lo, n * hi);
        }
        *reinterpret_cast<uint4*>(dst) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
};

#define PK_TQ_ENTRY(NAME, HD)                                                                     \
extern "C" __global__ void __launch_bounds__(256) NAME(                                           \
    const half* __restrict__ q, const uint32_t* __restrict__ k_codes,                             \
    const uint32_t* __restrict__ v_codes, const void* __restrict__ k_norms,                       \
    const void* __restrict__ v_norms, const float* __restrict__ centroids,                        \
    half* __restrict__ output, const int seq_q, const int seq_kv, const int num_heads,            \
    const int num_kv_heads, const int position_offset, const float scale,                         \
    const int mse_bits, const int code_uints_per_vec, const int norm_format)                      \
{                                                                                                 \
    const PkTqSide ks = { k_codes, k_norms, centroids, mse_bits, code_uints_per_vec, norm_format };\
    const PkTqSide vs = { v_codes, v_norms, centroids, mse_bits, code_uints_per_vec, norm_format };\
    pk_prefill_body<HD, PkLoadTq<HD>, PkLoadTq<HD>>(q, &ks, &vs, output, seq_q, seq_kv, num_heads, \
                                                    num_kv_heads, position_offset, scale);        \
}

PK_TQ_ENTRY(attention_flash_mma_prefill_kvq_hd128_tqtq, 128)
PK_TQ_ENTRY(attention_flash_mma_prefill_kvq_hd64_tqtq, 64)
