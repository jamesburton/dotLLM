// GQA-grouped cooperative-matrix flash-attention prefill for head_dim <= 256 (issue #441).
//
// NOT compiled on its own (.glsl): attention_flash_gqa_cm{,_kvf16}.comp include it and pick the KV storage.
//
// Why a new kernel and not a retune of attention_flash_f32_coopmat_hd256: that shader reached ~1 TFLOP/s on Strix at depth
// because (a) a workgroup is ONE query head x 16 rows, so each KV row is fetched 24 times per 16 query positions for Bonsai 2
// (24 query / 4 KV heads), (b) every 32-column KV tile costs ~19 workgroup barriers (K and V are staged in four 64-wide d-chunks
// each), (c) the online softmax is 16 threads scanning rows serially while 240 wait, and (d) only 2 of its 4 subgroups run QK^T.
//
// Layout here:
//   - The query side of one KV head is the matrix of its G = numHeads / numKvHeads query heads' rows: row m <-> (token t = m / G,
//     head hq = hkv * G + m % G). A workgroup owns 64 consecutive rows (4 subgroups x a 16-row strip), so one K/V tile load feeds
//     64 query rows instead of 16, and all G heads of the group read it once.
//   - Flash-Attention-2 ownership: each subgroup owns its strip end to end. Q (hi and lo halves, see below) and the O accumulator
//     live in registers as cooperative matrices; the online softmax runs on the strip's own lanes (shuffle reductions), so the only
//     workgroup barriers are the four around the shared K and V tile stages.
//   - QK^T keeps the hd256 kernel's HI/LO Q split (S = Qh.K + Ql.K, + Qh.Klo for F32 K), because a plain f16 QK^T measured
//     NLL 2.272 -> 3.509 on the hd256 Qwen3.5 hybrids. With F16 K the K_lo term is exactly zero and is not computed.
//   - The O rescale by the per-row correction is a component-wise multiply by a 16x16 matrix whose row r holds corr[r] (all
//     accumulator fragments share one element layout), skipped when every row's correction is exactly 1.
//
// Numerics (precision rules, CLAUDE.md): QK^T and P.V accumulate in F32 (coopmat accumulators); running max, running sum and the
// correction are F32. P is rounded to F16 for the P.V multiply and the denominator sums the same F16-rounded P.
//
// #533 / #543 contract (AMD coopmat P.V is not invariant to 0 * v columns): a KV tile that is not full or that holds a masked
// cell for any row of the workgroup takes a scalar P.V (`precise`, F32 accumulation over F16 P and V) instead of the matrix one,
// gated by the REQUIRE_INVARIANT_PV specialization constant (0 on NVIDIA, which measured clean).
//
// KV storage (the .comp wrapper): KV_F16 reads F16 rows in place (BC = 32); the default reads F32 rows and splits K and V into
// hi/lo halves (one more MMA each for QK^T and P.V; BC = 16 so both tiles fit), so the F32 baseline is not rounded to F16. Nothing widens the history: each element is converted in the tile load. A future
// q8_0 cache is one more KV_LOAD_TILE body.
//
// LDS (KV_F16): K/V tile 32 x 264 f16 (16.5 KiB) + scores 4 x 16 x 36 f32 (9 KiB) + P 4 x 16 x 40 f16 (5 KiB) = 30.5 KiB, under
// the 32 KiB Strix limit.
//
// Precision-study variants (issue #1109; experiment only, selected by DOTLLM_VK_EXP_FA_ACC; the shipped pipelines define none of
// these and compile to the same SPIR-V as before):
//   QK_ACC_F16      S = Q K^T accumulates in an F16 cooperative matrix; Q is pre-scaled by the softmax scale (exact for a
//                   power-of-two scale such as 1/16 at head_dim 256) so the F16 accumulator holds scaled scores.
//   PV_ACC_F16_TILE P.V of one KV tile accumulates in F16, then is added to the F32 O accumulator (F32 across tiles).
//   PV_ACC_F16      O itself is an F16 accumulator over the whole KV range.
//   SCORE_STATS     binding 4 receives score statistics (see BufStats); the arithmetic is the default's.
//
// Dispatch: local_size_x = 4 * subgroupSize (specialization constant 1, the subgroup size pinned at pipeline creation),
//           workgroups = (ceil(seqQ * G / 64), numKvHeads).

#extension GL_KHR_cooperative_matrix          : require
#extension GL_KHR_memory_scope_semantics      : require
#extension GL_KHR_shader_subgroup_basic       : require
#extension GL_KHR_shader_subgroup_shuffle     : require
#extension GL_KHR_shader_subgroup_vote        : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes     : require

#define STRIP           16                       // query rows per subgroup
#define NSTRIP          4                        // subgroups (strips) per workgroup
#define WG_ROWS         (STRIP * NSTRIP)         // 64 query rows per workgroup
#define MAX_HEAD_DIM    256
#define HD_CHUNKS       (MAX_HEAD_DIM / 16)      // 16 coopmat d-chunks
#if defined(KV_F16)
#define BC              32
#else
#define BC              16
#endif
#define CF              (BC / 16)                // KV-column fragments per tile
#define T_STRIDE        (MAX_HEAD_DIM + 8)       // K/V tile row stride in halves (+16 B against bank conflicts, 16 B aligned)
#define T_STRIDE4       (T_STRIDE / 4)
#define S_STRIDE        (BC + 4)                 // score row stride in floats (16 B aligned)
#define S_STRIP         (STRIP * S_STRIDE)       // >= 256 floats: also holds the 16x16 corr / output / scalar-PV stage
#define P_STRIDE        (BC + 8)                 // P row stride in halves (16 B aligned)
#define P_STRIP         (STRIP * P_STRIDE)       // >= 256 halves: also holds the 16x16 Q staging
#define NEG_INF         (-3.4e38)

layout(constant_id = 0) const uint REQUIRE_INVARIANT_PV = 1u;
// 1 (default): QK^T adds the Q lo-half term (see the header). 0 drops it: 1.6x faster at depth on Strix (6.1-6.3 vs 3.8 TFLOP/s),
// but Q is then rounded to F16. Off only for a measured A/B (DOTLLM_VULKAN_FA_GQA_QLO=0) until perplexity parity is shown.
layout(constant_id = 2) const uint QK_SPLIT_Q = 1u;
layout(local_size_x_id = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ   { float q[]; };      // [seq_q, num_heads * head_dim] F32
#if defined(KV_F16)
// F16 rows read as packed words (4 halves per uvec2), so no 16-bit storage-buffer feature is needed; unpackHalf2x16 is exact.
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { uvec2 k4[]; };     // [seq_kv, num_kv_heads * head_dim] F16
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { uvec2 v4[]; };
#else
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { vec4 k4[]; };      // [seq_kv, num_kv_heads * head_dim] F32
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { vec4 v4[]; };
#endif
layout(set = 0, binding = 3, std430) writeonly buffer BufOut { float outp[]; };   // [seq_q, num_heads * head_dim] F32
#if defined(SCORE_STATS)
// [0] max |s| over every unmasked scaled score (float bits; non-negative floats order as uints), [1] max |Q_lo . K| * scale,
// [2 + b] rows whose max |s| falls in [b, b + 1) (b < 256, clamped), [258 + b] rows whose max |Q_lo . K| * scale falls in
// [2^(b-40), 2^(b-39)) (b < 64, clamped), [322] rows counted.
layout(set = 0, binding = 4, std430) buffer BufStats { uint stats[]; };
#endif
#if defined(PV_ACC_F16)
#define OACC float16_t
#else
#define OACC float
#endif
#if defined(QK_ACC_F16)
#define SACC float16_t
#else
#define SACC float
#endif

layout(push_constant) uniform PushConstants {
    uint  seqQ;
    uint  seqKv;
    uint  numHeads;
    uint  numKvHeads;
    uint  headDim;          // multiple of 16, <= 256 (host-checked)
    uint  positionOffset;
    uint  slidingWindow;    // 0 = disabled
    uint  useAlibi;         // must be 0 (host routes ALiBi elsewhere)
    float softCap;          // 0 = disabled
    float scaleOverride;    // 0 = 1/sqrt(head_dim)
    uint  maskMode;         // 0 = Causal, 1 = Bidirectional, 2 = Hybrid
    uint  prefixLen;
} pc;

shared f16vec4   kvT[BC * T_STRIDE4];            // K tile (hi), then V tile
#if !defined(KV_F16)
shared f16vec4   kLoT[BC * T_STRIDE4];           // K lo, then V lo halves (F32 KV only)
#endif
shared float     sS[NSTRIP * S_STRIP];
shared float16_t sP[NSTRIP * P_STRIP];

bool isMasked(uint tkv, uint posQ) {
    if (pc.maskMode == 1u) return false;
    if (pc.maskMode == 2u && posQ >= pc.prefixLen) return false;
    return tkv > posQ;
}

// K / V tiles move global -> registers -> LDS in two steps so the global fetch of the NEXT tile is in flight while this tile is
// being multiplied (fetch V during QK^T, fetch the next K during P.V). Rows past tileLen and dims past headDim are zero, so the
// 16-wide inner products never read garbage.
#if defined(KV_F16)
#define KVREG uvec2
#define KVREG_ZERO uvec2(0u)
#else
#define KVREG vec4
#define KVREG_ZERO vec4(0.0)
#endif
#define TILE_VEC4   (BC * (MAX_HEAD_DIM / 4))
#define PF_MAX      (TILE_VEC4 / 128)          // per-invocation share at the smallest workgroup (4 x 32 lanes)
KVREG pf[PF_MAX];

void fetchTile(uint which, uint tStart, uint tileLen, uint hkv, uint hd4, uint kvStride4) {
    [[unroll]] for (uint i = 0u; i < uint(PF_MAX); i++) {
        uint idx = gl_LocalInvocationIndex + i * gl_WorkGroupSize.x;
        KVREG x = KVREG_ZERO;
        if (idx < uint(TILE_VEC4)) {
            uint c  = idx / uint(MAX_HEAD_DIM / 4);
            uint d4 = idx - c * uint(MAX_HEAD_DIM / 4);
            if (c < tileLen && d4 < hd4) {
                uint src = (tStart + c) * kvStride4 + hkv * hd4 + d4;
                x = (which == 0u) ? k4[src] : v4[src];
            }
        }
        pf[i] = x;
    }
}

void storeTile() {
    [[unroll]] for (uint i = 0u; i < uint(PF_MAX); i++) {
        uint idx = gl_LocalInvocationIndex + i * gl_WorkGroupSize.x;
        if (idx < uint(TILE_VEC4)) {
            uint c  = idx / uint(MAX_HEAD_DIM / 4);
            uint dst = c * uint(T_STRIDE4) + (idx - c * uint(MAX_HEAD_DIM / 4));
#if defined(KV_F16)
            kvT[dst] = f16vec4(vec4(unpackHalf2x16(pf[i].x), unpackHalf2x16(pf[i].y)));
#else
            f16vec4 hi = f16vec4(pf[i]);
            kvT[dst] = hi;
            kLoT[dst] = f16vec4(pf[i] - vec4(hi));   // K lo for QK^T, then V lo for P.V
#endif
        }
    }
}

void main() {
    uint G         = pc.numHeads / pc.numKvHeads;
    uint totalRows = pc.seqQ * G;
    uint numWg     = (totalRows + uint(WG_ROWS) - 1u) / uint(WG_ROWS);
    // Reverse order: the last query rows see the longest causal KV range, so they start first and the tail balances.
    uint wgIdx     = numWg - 1u - gl_WorkGroupID.x;
    uint hkv       = gl_WorkGroupID.y;
    uint wgRow0    = wgIdx * uint(WG_ROWS);
    uint wgRowEnd  = min(wgRow0 + uint(WG_ROWS), totalRows);

    uint strip    = gl_SubgroupID;                    // 0..3: the host pins the subgroup size so there are exactly 4
    uint lane     = gl_SubgroupInvocationID;
    uint sgSize   = gl_SubgroupSize;
    uint lpr      = sgSize / uint(STRIP);             // lanes per strip row (4 at wave64, 2 at wave32)
    uint myRow    = lane / lpr;                       // the strip row this lane serves in the softmax
    uint myJ      = lane - myRow * lpr;
    uint row0     = wgRow0 + strip * uint(STRIP);
    bool stripLive = row0 < totalRows;
    uint stripRowEnd = min(row0 + uint(STRIP), totalRows);

    uint qStride   = pc.numHeads * pc.headDim;
    uint hd4       = pc.headDim >> 2u;
    uint kvStride4 = (pc.numKvHeads * pc.headDim) >> 2u;
    uint hdChunks  = pc.headDim >> 4u;
    float scale = (pc.scaleOverride > 0.0) ? pc.scaleOverride : inversesqrt(float(pc.headDim));
#if defined(QK_ACC_F16)
    // F16 S: fold the scale into Q (exact for a power-of-two scale) so the F16 accumulator holds scaled scores.
#define Q_SCALED(x) ((x) * scale)
#define S_SCALED(x) (x)
#else
#define Q_SCALED(x) (x)
#define S_SCALED(x) ((x) * scale)
#endif

    uint sBase = strip * uint(S_STRIP);
    uint pBase = strip * uint(P_STRIP);

    // This lane's softmax row.
    uint mRow    = row0 + myRow;
    bool rowLive = mRow < totalRows;
    uint posQ    = pc.positionOffset + (rowLive ? mRow / G : 0u);

    // 1. Q fragments (hi / lo f16 halves of the F32 row), staged one 16x16 block at a time through this strip's P scratch.
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> qh[HD_CHUNKS];
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ql[HD_CHUNKS];
    [[unroll]] for (uint dc = 0u; dc < uint(HD_CHUNKS); dc++) {
        qh[dc] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>(0.0);
        ql[dc] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>(0.0);
        if (stripLive && dc < hdChunks) {
            for (uint half_ = 0u; half_ < (QK_SPLIT_Q != 0u ? 2u : 1u); half_++) {
                for (uint e = lane; e < 256u; e += sgSize) {
                    uint r = e >> 4u;
                    uint d = dc * 16u + (e & 15u);
                    uint m = row0 + r;
                    float x = 0.0;
                    if (m < totalRows) {
                        uint t = m / G;
                        uint hq = hkv * G + (m - t * G);
                        x = Q_SCALED(q[t * qStride + hq * pc.headDim + d]);
                    }
                    float16_t hi = float16_t(x);
                    sP[pBase + e] = (half_ == 0u) ? hi : float16_t(x - float(hi));
                }
                subgroupMemoryBarrierShared();
                subgroupBarrier();
                if (half_ == 0u) coopMatLoad(qh[dc], sP, pBase, 16u, gl_CooperativeMatrixLayoutRowMajor);
                else             coopMatLoad(ql[dc], sP, pBase, 16u, gl_CooperativeMatrixLayoutRowMajor);
                subgroupBarrier();
            }
        }
    }

    coopmat<OACC, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> o[HD_CHUNKS];
    [[unroll]] for (uint j = 0u; j < uint(HD_CHUNKS); j++) {
        o[j] = coopmat<OACC, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    }
    float runMax = NEG_INF;
    float runSum = 0.0;
#if defined(SCORE_STATS)
    float statAbsMax = 0.0;   // this lane's max |s| over its row
    float statLoMax = 0.0;    // this lane's max |Q_lo . K| * scale
#endif

    // 2. KV walk. Causal early exit at the workgroup's last position; per strip, tiles past its own last position are skipped.
    uint kvEnd = pc.seqKv;
    if (pc.maskMode == 0u) {
        uint lastPos = pc.positionOffset + (wgRowEnd - 1u) / G;
        kvEnd = min(pc.seqKv, lastPos + 1u);
    }
    uint wgFirstPos   = pc.positionOffset + wgRow0 / G;
    uint stripLastPos = pc.positionOffset + (stripLive ? (stripRowEnd - 1u) / G : 0u);

    fetchTile(0u, 0u, min(uint(BC), kvEnd), hkv, hd4, kvStride4);
    for (uint tStart = 0u; tStart < kvEnd; tStart += uint(BC)) {
        uint tileLen = min(uint(BC), kvEnd - tStart);
        bool tileSafe = (REQUIRE_INVARIANT_PV == 0u)
            || (tileLen == uint(BC) && (pc.maskMode == 1u || (tStart + uint(BC)) <= wgFirstPos + 1u));
        bool stripOn = stripLive && (pc.maskMode != 0u || tStart <= stripLastPos);

        // 2a. K tile (fetched during the previous tile's P.V); start fetching this tile's V.
        storeTile();
        barrier();
        fetchTile(1u, tStart, tileLen, hkv, hd4, kvStride4);

        float corr = 1.0;
        if (stripOn) {
            // 2b. S = Q K^T (F32 accumulate), stored to this strip's score rows.
            [[unroll]] for (uint cf = 0u; cf < uint(CF); cf++) {
                coopmat<SACC, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
                    coopmat<SACC, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
                [[unroll]] for (uint dc = 0u; dc < uint(HD_CHUNKS); dc++) {
                    if (dc < hdChunks) {
                        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> kt;
                        coopMatLoad(kt, kvT, cf * 16u * uint(T_STRIDE4) + dc * 4u, uint(T_STRIDE4), gl_CooperativeMatrixLayoutColumnMajor);
                        acc = coopMatMulAdd(qh[dc], kt, acc);
                        if (QK_SPLIT_Q != 0u) acc = coopMatMulAdd(ql[dc], kt, acc);
#if !defined(KV_F16)
                        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> ktLo;
                        coopMatLoad(ktLo, kLoT, cf * 16u * uint(T_STRIDE4) + dc * 4u, uint(T_STRIDE4), gl_CooperativeMatrixLayoutColumnMajor);
                        acc = coopMatMulAdd(qh[dc], ktLo, acc);
#endif
                    }
                }
#if defined(QK_ACC_F16)
                coopMatStore(coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc), sS, sBase + cf * 16u,
                             uint(S_STRIDE), gl_CooperativeMatrixLayoutRowMajor);
#else
                coopMatStore(acc, sS, sBase + cf * 16u, uint(S_STRIDE), gl_CooperativeMatrixLayoutRowMajor);
#endif
            }
            subgroupMemoryBarrierShared();
            subgroupBarrier();

            // 2c. Online softmax on the strip's lanes: lane (myRow, myJ) holds columns myJ, myJ + lpr, ... of its row.
            float sv[BC / 2];
            float tileMax = NEG_INF;
            uint cols = uint(BC) / lpr;
            for (uint i = 0u; i < cols; i++) {
                uint c = myJ + i * lpr;
                uint tkv = tStart + c;
                float s = NEG_INF;
                if (rowLive && c < tileLen) {
                    bool masked = isMasked(tkv, posQ)
                        || (pc.slidingWindow > 0u && posQ >= tkv && (posQ - tkv) >= pc.slidingWindow);
                    if (!masked) {
                        s = S_SCALED(sS[sBase + myRow * uint(S_STRIDE) + c]);
#if defined(SCORE_STATS)
                        statAbsMax = max(statAbsMax, abs(s));
#endif
                        if (pc.softCap > 0.0) s = pc.softCap * tanh(s / pc.softCap);
                    }
                }
                sv[i] = s;
                tileMax = max(tileMax, s);
            }
#if defined(SCORE_STATS)
            // The Q lo-half's share of each score, recomputed here (the score rows are already in registers).
            subgroupBarrier();
            [[unroll]] for (uint cf = 0u; cf < uint(CF); cf++) {
                coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> accLo =
                    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
                [[unroll]] for (uint dc = 0u; dc < uint(HD_CHUNKS); dc++) {
                    if (dc < hdChunks) {
                        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> kt;
                        coopMatLoad(kt, kvT, cf * 16u * uint(T_STRIDE4) + dc * 4u, uint(T_STRIDE4), gl_CooperativeMatrixLayoutColumnMajor);
                        accLo = coopMatMulAdd(ql[dc], kt, accLo);
                    }
                }
                coopMatStore(accLo, sS, sBase + cf * 16u, uint(S_STRIDE), gl_CooperativeMatrixLayoutRowMajor);
            }
            subgroupMemoryBarrierShared();
            subgroupBarrier();
            for (uint i = 0u; i < cols; i++) {
                uint c = myJ + i * lpr;
                if (sv[i] > NEG_INF * 0.5) statLoMax = max(statLoMax, abs(S_SCALED(sS[sBase + myRow * uint(S_STRIDE) + c])));
            }
#endif
            for (uint off = 1u; off < lpr; off <<= 1u) {
                tileMax = max(tileMax, subgroupShuffleXor(tileMax, off));
            }
            float newMax = max(runMax, tileMax);
            corr = (runMax > NEG_INF * 0.5) ? exp(runMax - newMax) : 0.0;
            if (!rowLive) corr = 1.0;   // dead rows: never written, keep them NaN-free
            float tileSum = 0.0;
            for (uint i = 0u; i < cols; i++) {
                uint c = myJ + i * lpr;
                float w = (sv[i] > NEG_INF * 0.5) ? exp(sv[i] - newMax) : 0.0;
                float16_t wh = float16_t(w);
                sP[pBase + myRow * uint(P_STRIDE) + c] = wh;
                tileSum += float(wh);
            }
            for (uint off = 1u; off < lpr; off <<= 1u) {
                tileSum += subgroupShuffleXor(tileSum, off);
            }
            runSum = runSum * corr + tileSum;
            runMax = newMax;

            // 2d. O *= corr (row-wise), through a 16x16 matrix whose row r is corr[r]; the scores are consumed, so it reuses sS.
            if (!subgroupAll(corr == 1.0)) {
                subgroupBarrier();
                for (uint c = myJ; c < 16u; c += lpr) {
                    sS[sBase + myRow * 16u + c] = corr;
                }
                subgroupMemoryBarrierShared();
                subgroupBarrier();
                coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> cm;
                coopMatLoad(cm, sS, sBase, 16u, gl_CooperativeMatrixLayoutRowMajor);
#if defined(PV_ACC_F16)
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> cmh =
                    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(cm);
                [[unroll]] for (uint j = 0u; j < uint(HD_CHUNKS); j++) {
                    o[j] = o[j] * cmh;
                }
#else
                [[unroll]] for (uint j = 0u; j < uint(HD_CHUNKS); j++) {
                    o[j] = o[j] * cm;
                }
#endif
            }
            subgroupMemoryBarrierShared();
            subgroupBarrier();
        }
        barrier();   // every strip is done with the K tile

        // 2e. V tile; start fetching the next K tile.
        storeTile();
        barrier();
        if (tStart + uint(BC) < kvEnd) {
            fetchTile(0u, tStart + uint(BC), min(uint(BC), kvEnd - tStart - uint(BC)), hkv, hd4, kvStride4);
        }

        if (stripOn) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> pm[CF];
            [[unroll]] for (uint cf = 0u; cf < uint(CF); cf++) {
                coopMatLoad(pm[cf], sP, pBase + cf * 16u, uint(P_STRIDE), gl_CooperativeMatrixLayoutRowMajor);
            }
            if (tileSafe) {
                // 2f. O += P V on the matrix cores.
                [[unroll]] for (uint j = 0u; j < uint(HD_CHUNKS); j++) {
                    if (j < hdChunks) {
#if defined(PV_ACC_F16_TILE)
                        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> t16 =
                            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif
                        [[unroll]] for (uint cf = 0u; cf < uint(CF); cf++) {
                            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> vm;
                            coopMatLoad(vm, kvT, cf * 16u * uint(T_STRIDE4) + j * 4u, uint(T_STRIDE4), gl_CooperativeMatrixLayoutRowMajor);
#if defined(PV_ACC_F16_TILE)
                            t16 = coopMatMulAdd(pm[cf], vm, t16);
#else
                            o[j] = coopMatMulAdd(pm[cf], vm, o[j]);
#endif
#if !defined(KV_F16)
                            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> vl;
                            coopMatLoad(vl, kLoT, cf * 16u * uint(T_STRIDE4) + j * 4u, uint(T_STRIDE4), gl_CooperativeMatrixLayoutRowMajor);
                            o[j] = coopMatMulAdd(pm[cf], vl, o[j]);
#endif
                        }
#if defined(PV_ACC_F16_TILE)
                        o[j] = o[j] + coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(t16);
#endif
                    }
                }
            } else {
                // 2f'. #533: scalar P V for a tile whose contents depend on the dispatch's KV length (partial or masked).
                [[unroll]] for (uint j = 0u; j < uint(HD_CHUNKS); j++) {
                    if (j < hdChunks) {
                        for (uint e = lane; e < 256u; e += sgSize) {
                            uint r = e >> 4u;
                            uint d = j * 16u + (e & 15u);
                            precise float acc = 0.0;
                            for (uint c = 0u; c < tileLen; c++) {
                                f16vec4 v = kvT[c * uint(T_STRIDE4) + (d >> 2u)];
                                float vv = float(v[d & 3u]);
#if !defined(KV_F16)
                                vv += float(kLoT[c * uint(T_STRIDE4) + (d >> 2u)][d & 3u]);
#endif
                                acc += float(sP[pBase + r * uint(P_STRIDE) + c]) * vv;
                            }
                            sS[sBase + e] = acc;
                        }
                        subgroupMemoryBarrierShared();
                        subgroupBarrier();
                        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> t;
                        coopMatLoad(t, sS, sBase, 16u, gl_CooperativeMatrixLayoutRowMajor);
#if defined(PV_ACC_F16)
                        o[j] = o[j] + coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(t);
#else
                        o[j] = o[j] + t;
#endif
                        subgroupBarrier();
                    }
                }
            }
        }
        barrier();   // every strip is done with the V tile before the next K tile overwrites it
    }

#if defined(SCORE_STATS)
    for (uint off = 1u; off < lpr; off <<= 1u) {
        statAbsMax = max(statAbsMax, subgroupShuffleXor(statAbsMax, off));
        statLoMax = max(statLoMax, subgroupShuffleXor(statLoMax, off));
    }
    if (rowLive && stripLive && myJ == 0u) {
        atomicMax(stats[0], floatBitsToUint(statAbsMax));
        atomicMax(stats[1], floatBitsToUint(statLoMax));
        atomicAdd(stats[2u + min(uint(statAbsMax), 255u)], 1u);
        int lb = (statLoMax > 0.0) ? clamp(int(floor(log2(statLoMax))) + 40, 0, 63) : 0;
        atomicAdd(stats[258u + uint(lb)], 1u);
        atomicAdd(stats[322], 1u);
    }
#endif

    // 3. Normalise (row-wise 1/sum through the same matrix trick) and write the strip's rows.
    if (stripLive) {
        float inv = (runSum > 1e-10) ? (1.0 / runSum) : 0.0;
        for (uint c = myJ; c < 16u; c += lpr) {
            sS[sBase + myRow * 16u + c] = inv;
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> im;
        coopMatLoad(im, sS, sBase, 16u, gl_CooperativeMatrixLayoutRowMajor);
        subgroupBarrier();
        [[unroll]] for (uint j = 0u; j < uint(HD_CHUNKS); j++) {
            if (j < hdChunks) {
#if defined(PV_ACC_F16)
                coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> r =
                    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(o[j]) * im;
#else
                coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> r = o[j] * im;
#endif
                coopMatStore(r, sS, sBase, 16u, gl_CooperativeMatrixLayoutRowMajor);
                subgroupMemoryBarrierShared();
                subgroupBarrier();
                for (uint e = lane; e < 256u; e += sgSize) {
                    uint m = row0 + (e >> 4u);
                    if (m < totalRows) {
                        uint t = m / G;
                        uint hq = hkv * G + (m - t * G);
                        outp[t * qStride + hq * pc.headDim + j * 16u + (e & 15u)] = sS[sBase + e];
                    }
                }
                subgroupBarrier();
            }
        }
    }
}
