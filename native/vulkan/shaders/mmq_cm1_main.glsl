// INT8 COOPMAT MMQ GEMM (issue #1226): C[N, M] = B[N, K] @ W[M, K]^T with W read PACKED in place and B pre-quantized to Q8_1.
//
// Port of llama.cpp ggml-vulkan's mul_mmq_cm1.comp + mul_mmq_cm1_funcs.glsl (MIT, (c) the ggml authors), the matmul llama.cpp
// runs on RDNA3/3.5 for quantized prefill: Sint8 x Sint8 -> Sint32 cooperative matrices (iu8 WMMA), one int32 tile per 32-element
// K block, folded into F32 accumulators with the weight-block scale x activation-block scale (+ the asymmetric min term).
// Compared to the F16 blocked template (gemm_coopmat_blocked_*.glsl) this:
//   * stages weights as raw int8 (no per-element float multiply, half the LDS bytes),
//   * stages activations as Q8_1 int8 (a quarter of the F32 bytes the F16 template streams),
//   * runs int8 WMMA (twice the F16 rate on RDNA3),
//   * prefetches the next K step into registers while the current one computes,
//   * walks the M tiles in L2-sized groups.
//
// CONTRACT - the wrapper #defines exactly one of MMQ_Q8_0 / MMQ_Q4_K / MMQ_Q5_0 / MMQ_Q4_1, optionally MMQ_GROUPED, and includes this file.
//
// MMQ_GROUPED (MoE expert-grouped prefill): the weight buffer is a bank [E, M, K]; the activation rows are packed in expert order
// (offsets[e] .. offsets[e+1], moe_expand_gather_group_q8_1) and the output rows are packed the same way. Workgroup x is the M tile,
// y a tile index into the work list moe_build_tile_list wrote after the offsets (tileRows = 64), or (tileList == 0) y = row tile,
// z = expert. No L2 swizzle there: one expert's rows rarely span more than one token tile.
//
// Geometry: BM = BN = 64, two wave64 subgroups (128 threads) each owning a 32x64 warp tile = 2x4 16x16 fragments, BK = 32 (one quant
// block), BK_STEP = 4 blocks staged per barrier. llama.cpp pins this kernel to wave32 on RDNA3.5; on the 8060S under dotLLM's pipeline
// setup the wave32 pin did not take (measured: the int32 accumulator came back in the wave64 layout and gl_SubgroupID paired the
// 32-lane halves), so the kernel is written for the native wave64: lane l owns column l % 16, rows l / 16 + 4e (e = 0..3). The host
// pins 64 and validates the layout with a small GEMM when it creates the pipeline.
//
// Bounds: rows >= M, token rows >= N and K blocks past K / 32 are zero-filled at staging; rows are skipped at store. K must be a
// multiple of 32 and, for Q4_K, of 256 (the host checks).
//
// Buffers:
//   0  W    packed weight rows (Q8_0 / Q5_0: 34 / 22-byte blocks, read as uint16; Q4_K / Q4_1: 144 / 20-byte blocks, read as uint)
//   1  Bq   uvec4[]  Q8_1 int8 activations, row r at uint offset r * K / 4 (QuantizeQ8_1RowsKernel layout)
//   2  Bds  vec2[]   per-32-block (d, d * sum(q)), row r at r * K / 32
//   3  C    float[]  [N, M] row-major

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require

#define WARP 64u
#define BM 64u
#define BN 64u
#define WM 32u
#define WN 64u
#define TM 16u
#define TN 16u
#define TK 16u
#define BK 32u
#define BK_STEP 4u
// LDS rows: BK_STEP blocks of 8 uints (32 int8) + 4 uints of padding against bank conflicts.
#define QPITCH (BK_STEP * (BK / 4u) + 4u)
#define CM_ELEMS ((TM * TN) / WARP)
// Weight bytes an L2-friendly group of M tiles may span before the next N tile starts over (llama.cpp GROUP_A_BUDGET).
#define GROUP_A_BUDGET (16u * 1024u * 1024u)

#define BLOCK_SIZE ((BM / WM) * (BN / WN) * WARP)
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;   // BLOCK_SIZE: two wave64 subgroups

#if defined(MMQ_Q8_0) || defined(MMQ_Q5_0)
layout(set = 0, binding = 0, std430) readonly buffer BufW { uint16_t w16[]; };
#define HAS_MIN 0
#if defined(MMQ_Q5_0)
#define LOAD_VEC_A 8u   // 4 lanes per block, 8 elements each (4 low + 4 high nibbles)
#else
#define LOAD_VEC_A 4u
#endif
#elif defined(MMQ_Q4_K) || defined(MMQ_Q4_1)
layout(set = 0, binding = 0, std430) readonly buffer BufW { uint w32[]; };
#define HAS_MIN 1
#define LOAD_VEC_A 8u
#else
#error "define MMQ_Q8_0, MMQ_Q5_0, MMQ_Q4_K or MMQ_Q4_1"
#endif
layout(set = 0, binding = 1, std430) readonly  buffer BufBq  { uvec4 bq[]; };
layout(set = 0, binding = 2, std430) readonly  buffer BufBds { vec2  bds[]; };
layout(set = 0, binding = 3, std430) writeonly buffer BufC   { float c[]; };
#ifdef MMQ_GROUPED
layout(set = 0, binding = 4, std430) readonly  buffer BufOff { uint offsets[]; };   // [E+1] group offsets (+ tile list)
#endif

layout(push_constant) uniform PushConstants {
    uint M;          // weight rows (per expert when grouped)
    uint K;          // contraction length
    uint N;          // token rows (total packed rows when grouped)
    uint rowStride;  // weight row stride: uint16 units (Q8_0 / Q5_0) or uint units (Q4_K)
    uint numExperts; // grouped only
    uint tileList;   // grouped only: index in `offsets` of the tile list, 0 = legacy 3-D grid
} pc;

// Tile-relative bounds, set once in main(): global weight row = g_aRowBase + m for m < pc.M; activation / output row =
// g_bRowBase + r for local token rows r < g_bRowLimit.
uint g_aRowBase;
uint g_bRowBase;
uint g_bRowLimit;
uint g_kBlocks;   // K / 32: blocks past it (a partial last staging step) are zero-filled

#define LOAD_VEC_B 16u

#if defined(MMQ_Q4_1)
#include "q8_1_ds_f16.glsl"
#endif

shared uint buf_a_qs[BM * QPITCH];
#if HAS_MIN
shared vec2 buf_a_dm[BM * BK_STEP];   // (d * scale, -dmin * min)
shared float buf_b_s[BN * BK_STEP];
#else
shared float buf_a_d[BM * BK_STEP];
#endif
shared uint buf_b_qs[BN * QPITCH];
shared float buf_b_d[BN * BK_STEP];

// ---- A side (weights) ------------------------------------------------------------------------------------------------

#if defined(MMQ_Q8_0)
struct block_a_prefetch { uint qs; float d; };

// Block kb (32 elements) of weight row `row`; lane loadr (0..7) takes int8 lanes 4*loadr .. 4*loadr+3.
block_a_prefetch block_a_load(uint row, uint kb, uint loadr) {
    block_a_prefetch blk;
    if (row >= pc.M || kb >= g_kBlocks) { blk.qs = 0u; blk.d = 0.0; return blk; }
    uint base = (g_aRowBase + row) * pc.rowStride + kb * 17u;   // 34-byte block = 17 uint16
    blk.d = float(uint16BitsToHalf(w16[base]));
    blk.qs = uint(w16[base + 1u + 2u * loadr]) | (uint(w16[base + 2u + 2u * loadr]) << 16);
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4u) + loadr] = blk.qs;
    if (loadr == 0u) buf_a_d[ks * BM + buf_ib] = blk.d;
}
#elif defined(MMQ_Q5_0)
struct block_a_prefetch { uint qs; uint qh; float d; };

// Q5_0 block (22 bytes = 11 uint16: fp16 d, qh, qs[16]); lane loadr (0..3) takes qs bytes 4*loadr..4*loadr+3, i.e. elements
// 4*loadr.. (low nibbles) and 16+4*loadr.. (high nibbles), each with its 5th bit from qh, recentred to q - 16 (signed int8).
block_a_prefetch block_a_load(uint row, uint kb, uint loadr) {
    block_a_prefetch blk;
    if (row >= pc.M || kb >= g_kBlocks) { blk.qs = 0u; blk.qh = 0u; blk.d = 0.0; return blk; }
    uint base = (g_aRowBase + row) * pc.rowStride + kb * 11u;
    blk.d = float(uint16BitsToHalf(w16[base]));
    blk.qh = uint(w16[base + 1u]) | (uint(w16[base + 2u]) << 16);
    blk.qs = uint(w16[base + 3u + 2u * loadr]) | (uint(w16[base + 4u + 2u * loadr]) << 16);
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint lo4 = blk.qs & 0x0F0F0F0Fu;
    uint hi4 = (blk.qs >> 4u) & 0x0F0F0F0Fu;
    lo4 |= ((blk.qh >> (4u * loadr)) & 0xFu) * 0x02040810u & 0x10101010u;
    hi4 |= ((blk.qh >> (4u * loadr + 16u)) & 0xFu) * 0x02040810u & 0x10101010u;
    lo4 = ((lo4 | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
    hi4 = ((hi4 | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4u) + loadr]      = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4u) + loadr + 4u] = hi4;
    if (loadr == 0u) buf_a_d[ks * BM + buf_ib] = blk.d;
}
#elif defined(MMQ_Q4_1)
struct block_a_prefetch { uint qs; vec2 dm; };

// Q4_1 block (20 bytes = 5 uints: fp16 d | fp16 m << 16, qs[16]); lane loadr (0..3) takes qs bytes 4*loadr..4*loadr+3, i.e.
// elements 4*loadr.. (low nibbles) and 16+4*loadr.. (high nibbles) as raw unsigned codes; the min term m * s_x restores the offset
// (llama.cpp vec_dot_q4_1_q8_1: d * d_x * sumi + m * s_x).
block_a_prefetch block_a_load(uint row, uint kb, uint loadr) {
    block_a_prefetch blk;
    if (row >= pc.M || kb >= g_kBlocks) { blk.qs = 0u; blk.dm = vec2(0.0); return blk; }
    uint base = (g_aRowBase + row) * pc.rowStride + kb * 5u;
    blk.dm = unpackHalf2x16(w32[base]);
    blk.qs = w32[base + 1u + loadr];
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4u) + loadr]      = blk.qs & 0x0F0F0F0Fu;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4u) + loadr + 4u] = (blk.qs >> 4u) & 0x0F0F0F0Fu;
    if (loadr == 0u) buf_a_dm[ks * BM + buf_ib] = blk.dm;
}
#else
struct block_a_prefetch { uint qs0; uint qs1; vec2 dm; };

// Sub-block kb (32 elements) of weight row `row`: super-block kb / 8, sub-block kb % 8. Lane loadr (0..3) takes codes
// 8*loadr .. 8*loadr+7 as raw unsigned nibbles (the min term absorbs the asymmetry); lane 0 also decodes the 6-bit
// (scale, min) pair, mirroring llama.cpp get_scale_min_k4.
block_a_prefetch block_a_load(uint row, uint kb, uint loadr) {
    block_a_prefetch blk;
    blk.dm = vec2(0.0);
    if (row >= pc.M || kb >= g_kBlocks) { blk.qs0 = 0u; blk.qs1 = 0u; return blk; }
    uint sb = (g_aRowBase + row) * pc.rowStride + (kb >> 3u) * 36u;   // 144-byte super-block = 36 uints
    uint sub = kb & 7u;
    uint q = sb + 4u + (sub >> 1u) * 8u + loadr * 2u;
    uint raw0 = w32[q], raw1 = w32[q + 1u];
    uint sh = (sub & 1u) * 4u;
    blk.qs0 = (raw0 >> sh) & 0x0F0F0F0Fu;
    blk.qs1 = (raw1 >> sh) & 0x0F0F0F0Fu;
    if (loadr == 0u) {
        vec2 dd = unpackHalf2x16(w32[sb]);
        uint s0 = w32[sb + 1u], s1 = w32[sb + 2u], s2 = w32[sb + 3u];
        uint sc, mn;
        if (sub < 4u) {
            uint b = sub * 8u;
            sc = (s0 >> b) & 63u;
            mn = (s1 >> b) & 63u;
        } else {
            uint b = (sub - 4u) * 8u;
            uint lo = s2 >> b;
            sc = (lo & 0xFu) | (((s0 >> b) >> 6u & 0x3u) << 4u);
            mn = ((lo >> 4u) & 0xFu) | (((s1 >> b) >> 6u & 0x3u) << 4u);
        }
        blk.dm = vec2(dd.x * float(sc), -(dd.y * float(mn)));
    }
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4u) + loadr * 2u]      = blk.qs0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4u) + loadr * 2u + 1u] = blk.qs1;
    if (loadr == 0u) buf_a_dm[ks * BM + buf_ib] = blk.dm;
}
#endif

// ---- B side (Q8_1 activations) ---------------------------------------------------------------------------------------

struct block_b_prefetch { uvec4 qs; vec2 ds; };

// Block kb of local token row `row`; lane loadr (0..1) takes int8 lanes 16*loadr .. 16*loadr+15.
block_b_prefetch block_b_load(uint row, uint kb, uint loadr) {
    block_b_prefetch blk;
    if (row >= g_bRowLimit || kb >= g_kBlocks) { blk.qs = uvec4(0u); blk.ds = vec2(0.0); return blk; }
    uint g = g_bRowBase + row;
    blk.qs = bq[g * (pc.K >> 4u) + kb * 2u + loadr];
    blk.ds = bds[g * (pc.K >> 5u) + kb];
#if defined(MMQ_Q4_1)
    blk.ds = roundToF16(blk.ds);   // fp16 (d, s) as block_q8_1 stores them: the CPU (#1159) and llama.cpp products
#endif
    return blk;
}

void block_b_to_shmem(block_b_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint base = buf_ib * QPITCH + ks * (BK / 4u) + loadr * 4u;
    buf_b_qs[base]      = blk.qs.x;
    buf_b_qs[base + 1u] = blk.qs.y;
    buf_b_qs[base + 2u] = blk.qs.z;
    buf_b_qs[base + 3u] = blk.qs.w;
    if (loadr == 0u) {
        buf_b_d[ks * BN + buf_ib] = blk.ds.x;
#if HAS_MIN
        buf_b_s[ks * BN + buf_ib] = blk.ds.y;
#endif
    }
}

// RDNA3 / 3.5 wave64 WMMA accumulator layout: element e of lane l is row (l / 16) + 4e, column l % 16.
uint cm_lane() { return gl_SubgroupInvocationID; }
uint cm_elem_row(uint e) { return (cm_lane() / TN) + 4u * e; }

const uint A_LOADSTRIDE = BLOCK_SIZE * LOAD_VEC_A / BK;   // weight rows staged per pass
const uint B_LOADSTRIDE = BLOCK_SIZE * LOAD_VEC_B / BK;   // token rows staged per pass
const uint A_LOADS = (BM + A_LOADSTRIDE - 1u) / A_LOADSTRIDE;
const uint B_LOADS = (BN + B_LOADSTRIDE - 1u) / B_LOADSTRIDE;   // with 256 threads one pass stages 128 > BN rows: the excess lanes idle

void main() {
#ifdef MMQ_GROUPED
    uint ir = gl_WorkGroupID.x;
    uint expert, localRowBase;
    if (pc.tileList != 0u) {
        expert = offsets[pc.tileList + 2u * gl_WorkGroupID.y];
        localRowBase = offsets[pc.tileList + 2u * gl_WorkGroupID.y + 1u];
    } else {
        expert = gl_WorkGroupID.z;
        localRowBase = gl_WorkGroupID.y * BN;
        if (expert >= pc.numExperts) return;
    }
    uint expertStart = offsets[expert];
    uint expertRows = offsets[expert + 1u] - expertStart;
    if (localRowBase >= expertRows) return;   // workgroup-uniform
    g_aRowBase = expert * pc.M;
    g_bRowBase = expertStart + localRowBase;
    g_bRowLimit = expertRows - localRowBase;
#else
    // L2-friendly order: consecutive workgroups sweep the N tiles of a group of M tiles whose weights fit GROUP_A_BUDGET,
    // so a weight panel is reused from L2 by every token tile before the next group streams in.
    uint blocks_m = (pc.M + BM - 1u) / BM;
    uint blocks_n = (pc.N + BN - 1u) / BN;
#if defined(MMQ_Q8_0)
    uint a_panel_bytes = BM * pc.K + (BM * pc.K) / 16u;
#elif defined(MMQ_Q4_1)
    uint a_panel_bytes = (BM * pc.K) / 2u + (BM * pc.K) / 8u;
#else
    uint a_panel_bytes = (BM * pc.K) / 2u + (BM * pc.K) / 16u;
#endif
    uint group_m = clamp(GROUP_A_BUDGET / max(a_panel_bytes, 1u), 1u, min(blocks_m, 32u));
    uint tiles_per_group = group_m * blocks_n;
    uint lin = gl_WorkGroupID.y * blocks_m + gl_WorkGroupID.x;
    uint group_id = lin / tiles_per_group;
    uint first_m = group_id * group_m;
    uint gsize = min(blocks_m - first_m, group_m);
    uint in_group = lin - group_id * tiles_per_group;
    uint ir = first_m + in_group % gsize;
    uint ic = in_group / gsize;
    g_aRowBase = 0u;
    g_bRowBase = ic * BN;
    g_bRowLimit = pc.N - g_bRowBase;
#endif

    uint tid = gl_LocalInvocationID.x;
    uint warp_i = gl_SubgroupID;
    uint warp_r = warp_i % (BM / WM);
    uint warp_c = warp_i / (BM / WM);
    uint elem_col0 = cm_lane() % TN;

    uint loadr_a = tid % (BK / LOAD_VEC_A);
    uint loadc_a = tid / (BK / LOAD_VEC_A);
    uint loadr_b = tid % (BK / LOAD_VEC_B);
    uint loadc_b = tid / (BK / LOAD_VEC_B);

    const uint CMS_ROW = WM / TM;
    const uint CMS_COL = WN / TN;
    const uint K_SUB = BK / TK;

    float sums[CMS_ROW * CMS_COL * CM_ELEMS];
    [[unroll]] for (uint i = 0u; i < CMS_ROW * CMS_COL * CM_ELEMS; i++) sums[i] = 0.0;

    block_a_prefetch pre_a[A_LOADS * BK_STEP];
    block_b_prefetch pre_b[B_LOADS * BK_STEP];

    uint kBlocks = pc.K / BK;
    g_kBlocks = kBlocks;
    uint mBase = ir * BM;

    // Prefetch the first K step.
    [[unroll]] for (uint li = 0u; li < A_LOADS; li++)
        [[unroll]] for (uint ks = 0u; ks < BK_STEP; ks++)
            pre_a[li * BK_STEP + ks] = block_a_load(mBase + loadc_a + li * A_LOADSTRIDE, ks, loadr_a);
    [[unroll]] for (uint li = 0u; li < B_LOADS; li++)
        [[unroll]] for (uint ks = 0u; ks < BK_STEP; ks++)
            pre_b[li * BK_STEP + ks] = block_b_load(loadc_b + li * B_LOADSTRIDE < BN ? loadc_b + li * B_LOADSTRIDE : 0xFFFFFFFFu, ks, loadr_b);

    uint a_row0 = warp_r * WM;
    uint b_col0 = warp_c * WN;
    bool active_col_tile = b_col0 < g_bRowLimit;

    for (uint kb = 0u; kb < kBlocks; kb += BK_STEP) {
        [[unroll]] for (uint li = 0u; li < A_LOADS; li++)
            [[unroll]] for (uint ks = 0u; ks < BK_STEP; ks++)
                block_a_to_shmem(pre_a[li * BK_STEP + ks], loadc_a + li * A_LOADSTRIDE, ks, loadr_a);
        [[unroll]] for (uint li = 0u; li < B_LOADS; li++)
            [[unroll]] for (uint ks = 0u; ks < BK_STEP; ks++)
                if (loadc_b + li * B_LOADSTRIDE < BN) block_b_to_shmem(pre_b[li * BK_STEP + ks], loadc_b + li * B_LOADSTRIDE, ks, loadr_b);
        barrier();

        // Double buffering: issue the next step's global loads before this step's math.
        uint next = kb + BK_STEP;
        if (next < kBlocks) {
            [[unroll]] for (uint li = 0u; li < A_LOADS; li++)
                [[unroll]] for (uint ks = 0u; ks < BK_STEP; ks++)
                    pre_a[li * BK_STEP + ks] = block_a_load(mBase + loadc_a + li * A_LOADSTRIDE, next + ks, loadr_a);
            [[unroll]] for (uint li = 0u; li < B_LOADS; li++)
                [[unroll]] for (uint ks = 0u; ks < BK_STEP; ks++)
                    pre_b[li * BK_STEP + ks] = block_b_load(loadc_b + li * B_LOADSTRIDE < BN ? loadc_b + li * B_LOADSTRIDE : 0xFFFFFFFFu, next + ks, loadr_b);
        }

        if (active_col_tile) {
            [[unroll]] for (uint ks = 0u; ks < BK_STEP; ks++) {
                coopmat<int8_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a[CMS_ROW * K_SUB];
                coopmat<int8_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b[CMS_COL * K_SUB];
                [[unroll]] for (uint r = 0u; r < CMS_ROW; r++)
                    [[unroll]] for (uint h = 0u; h < K_SUB; h++)
                        coopMatLoad(cache_a[r * K_SUB + h], buf_a_qs, (a_row0 + r * TM) * QPITCH + ks * (BK / 4u) + h * (TK / 4u),
                                    QPITCH, gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint cc = 0u; cc < CMS_COL; cc++)
                    [[unroll]] for (uint h = 0u; h < K_SUB; h++)
                        coopMatLoad(cache_b[cc * K_SUB + h], buf_b_qs, (b_col0 + cc * TN) * QPITCH + ks * (BK / 4u) + h * (TK / 4u),
                                    QPITCH, gl_CooperativeMatrixLayoutColumnMajor);

                float scale_b[CMS_COL];
#if HAS_MIN
                float sum_b[CMS_COL];
#endif
                [[unroll]] for (uint cc = 0u; cc < CMS_COL; cc++) {
                    scale_b[cc] = buf_b_d[ks * BN + b_col0 + cc * TN + elem_col0];
#if HAS_MIN
                    sum_b[cc] = buf_b_s[ks * BN + b_col0 + cc * TN + elem_col0];
#endif
                }

                [[unroll]] for (uint r = 0u; r < CMS_ROW; r++) {
                    float scale_a[CM_ELEMS];
#if HAS_MIN
                    float min_a[CM_ELEMS];
#endif
                    [[unroll]] for (uint e = 0u; e < CM_ELEMS; e++) {
#if HAS_MIN
                        vec2 dm = buf_a_dm[ks * BM + a_row0 + r * TM + cm_elem_row(e)];
                        scale_a[e] = dm.x;
                        min_a[e] = dm.y;
#else
                        scale_a[e] = buf_a_d[ks * BM + a_row0 + r * TM + cm_elem_row(e)];
#endif
                    }
                    [[unroll]] for (uint cc = 0u; cc < CMS_COL; cc++) {
                        coopmat<int32_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> acc =
                            coopmat<int32_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0);
                        [[unroll]] for (uint h = 0u; h < K_SUB; h++)
                            acc = coopMatMulAdd(cache_a[r * K_SUB + h], cache_b[cc * K_SUB + h], acc);
                        uint t = (r * CMS_COL + cc) * CM_ELEMS;
                        [[unroll]] for (uint e = 0u; e < CM_ELEMS; e++) {
#if HAS_MIN
                            sums[t + e] += fma(float(acc[e]) * scale_a[e], scale_b[cc], min_a[e] * sum_b[cc]);
#else
                            sums[t + e] += float(acc[e]) * scale_a[e] * scale_b[cc];
#endif
                        }
                    }
                }
            }
        }
        barrier();
    }

    uint dr = mBase + a_row0;
    [[unroll]] for (uint r = 0u; r < CMS_ROW; r++) {
        [[unroll]] for (uint cc = 0u; cc < CMS_COL; cc++) {
            uint t = (r * CMS_COL + cc) * CM_ELEMS;
            [[unroll]] for (uint e = 0u; e < CM_ELEMS; e++) {
                uint row_g = dr + r * TM + cm_elem_row(e);
                uint col_l = b_col0 + cc * TN + elem_col0;   // local token row
                if (row_g < pc.M && col_l < g_bRowLimit) c[(g_bRowBase + col_l) * pc.M + row_g] = sums[t + e];
            }
        }
    }
}
