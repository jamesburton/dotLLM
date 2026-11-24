#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered  : require
#extension GL_KHR_shader_subgroup_shuffle    : require
#extension GL_EXT_control_flow_attributes    : require

// GQA-grouped split-KV (flash-decoding) PARTIAL pass, v2 (issue #1161). Same contract as attention_f32_splitkv_gqa.glsl: one
// workgroup per (KV head, split), each K / V element read once for all G = numHeads / numKvHeads query heads; identical push
// constants, bindings and partial layout, so attention_f32_splitkv_merge.comp finishes the row.
//
// What changed (v1 streamed a 32K Flint full layer at ~45 GB/s on Strix):
//   - 16 lanes per KV position, each lane owning 8 adjacent columns (one 16-byte load of F16 KV), so a wave covers 4 positions
//     with coalesced rows and a 16-lane partial dot product is finished by subgroupClusteredAdd(16);
//   - each phase issues all of its tile's K (or V) loads before using any of them, so a thread has TILE_KV / 16 row loads in
//     flight instead of one, and the V loads are issued before the softmax barrier;
//   - P.V uses the same lane -> column mapping with 16 position slots per pass, and every thread keeps G x 8 accumulators; the
//     slots are reduced once per split (shuffles within the wave, LDS across waves).
//
// Groups wider than MAX_GROUP (issue #1612: glm4moe has 12 query heads per KV head) are served in ceil(G / MAX_GROUP) balanced
// sub-groups (12 -> 6 + 6, 9 -> 5 + 4), one workgroup per (KV head, sub-group, split): the KV rows are read once per sub-group,
// ceil(G / MAX_GROUP) times per decoded token instead of G. The dispatch is numKvHeads * subGroups * numSplits workgroups.
//
// Bounds: headDim <= MAX_HEAD_DIM and a multiple of 8, wave64 (the host gates both).
// LDS: 4 KiB Q (reused by the final reduce) + TILE_KV * 32 B scores.

#define TILE_KV      128
#define MAX_HEAD_DIM 128
#define MAX_GROUP    8
// #1926 E1b: the largest sub-group THIS pipeline serves (2..MAX_GROUP). It sizes the per-thread accumulators, the per-pass dot
// products and clustered adds, and the final reduce, so a 4-head group no longer carries 8 heads' registers and reductions. The
// shared-memory layout (and so the LDS footprint) stays sized on MAX_GROUP. The host builds the pipeline with the shape's sub-group
// size; the default (a pipeline built without specialization data) is the safe MAX_GROUP. A pipeline must never see G > GROUP.
layout(constant_id = 0) const uint GROUP = 8u;
const uint GROUP_PAD = (GROUP + 1u) & ~1u;   // the final reduce moves heads in pairs
#define WG_SIZE      256
#define LANES        16                     // lanes per KV position, 8 columns each
#define SLOTS        (WG_SIZE / LANES)      // KV positions per pass
#define PASSES       (TILE_KV / SLOTS)
#define NEG_INF      (-3.4e38)

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ       { float q[]; };       // [1, numHeads * head_dim]
#if defined(KV_TQ)
// TurboQuant (#1892, kv_load.glsl KV_TQ): a lane's 8 columns are 8 consecutive codes of one head vector times its norm, in code space.
//
// #1926 E3 (steps 1-2, bit-identical to the kvTqK8 / kvTqV8 loaders):
//   - the 2^bits centroid table is copied once per workgroup into shared memory (tqCentSh) instead of 8 buffer gathers per lane-row;
//   - a staged row is the lane's 8 codes as a 64-bit window shifted to bit 0 plus the decoded norm (3 registers), with only the code
//     words the format can touch loaded: 1 word for 1, 2 and 4 bits (8 codes never cross a word), 2 for 3, 5, 6 and 8, 3 for 7 bits.
//     The extra word of a window ends at most one word past the vector, the guard word every code buffer carries.
//   - codes are looked up and multiplied by the norm at use (lo4 / hi4), the same centroid * norm product the old loader formed.
#include "kv_load.glsl"
shared float tqCentSh[1u << TQ_MSE_BITS];
const uint TQ_WORDS = TQ_MSE_BITS == 7u ? 3u : (TQ_MSE_BITS == 8u ? 2u : ((32u % TQ_MSE_BITS) == 0u ? 1u : 2u));
#define RAW uvec3
// The 64 bits starting at bit `off` of (w0, w1, w2); off < 32. The two-shift form keeps off == 0 from shifting by 32.
RAW tqWindow(uint w0, uint w1, uint w2, uint off, float norm) {
    return uvec3((w0 >> off) | ((w1 << 1u) << (31u - off)), (w1 >> off) | ((w2 << 1u) << (31u - off)), floatBitsToUint(norm));
}
// Code j (0..7, a constant after unrolling) of a window.
uint tqCodeAt(RAW r, uint j) {
    uint p = j * TQ_MSE_BITS;
    uint c = p >= 32u ? (r.y >> (p - 32u)) : ((r.x >> p) | ((r.y << 1u) << (31u - p)));
    return c & TQ_CODE_MASK;
}
RAW loadK8(uint i) {
    uint hv = i >> TQ_HD_LOG2, bit = (i & ((1u << TQ_HD_LOG2) - 1u)) * TQ_MSE_BITS;
    uint w = hv * TQ_CODE_UINTS + (bit >> 5u);
    uint w1 = TQ_WORDS >= 2u ? kw[w + 1u] : 0u, w2 = TQ_WORDS >= 3u ? kw[w + 2u] : 0u;
    return tqWindow(kw[w], w1, w2, bit & 31u, tqNorm(kn[tqNormWord(hv)], hv));
}
RAW loadV8(uint i) {
    uint hv = i >> TQ_HD_LOG2, bit = (i & ((1u << TQ_HD_LOG2) - 1u)) * TQ_MSE_BITS;
    uint w = hv * TQ_CODE_UINTS + (bit >> 5u);
    uint w1 = TQ_WORDS >= 2u ? vw[w + 1u] : 0u, w2 = TQ_WORDS >= 3u ? vw[w + 2u] : 0u;
    return tqWindow(vw[w], w1, w2, bit & 31u, tqNorm(vn[tqNormWord(hv)], hv));
}
vec4 lo4(RAW r) { return vec4(tqCentSh[tqCodeAt(r, 0u)], tqCentSh[tqCodeAt(r, 1u)], tqCentSh[tqCodeAt(r, 2u)], tqCentSh[tqCodeAt(r, 3u)]) * uintBitsToFloat(r.z); }
vec4 hi4(RAW r) { return vec4(tqCentSh[tqCodeAt(r, 4u)], tqCentSh[tqCodeAt(r, 5u)], tqCentSh[tqCodeAt(r, 6u)], tqCentSh[tqCodeAt(r, 7u)]) * uintBitsToFloat(r.z); }
#elif defined(KV_Q8) || defined(KV_Q4) || defined(KV_MIXED)
// Block KV: Q8_0 (36-byte blocks, [float scale][32 x int8]) and Q4_0 (#1884, 20-byte blocks, [float scale][16 bytes], adjacent-pair
// nibbles), each side in its own format under KV_MIXED (KV_K_Q8 / KV_K_Q4, KV_V_Q8 / KV_V_Q4). A lane's 8 columns are 8 consecutive
// elements of one block.
//
// #1926 E1a: a staged row is the PACKED words (Q8_0: 2 data words + the scale word; Q4_0: 1 data word + the scale word), converted to
// F32 only in lo/hi at the point of use. The values and the operations on them are the ones the load-time dequant did, so the output
// is bit-identical; what changes is that a tile's 8 staged K and 8 staged V rows hold 2-3 registers each instead of 8.
#if defined(KV_Q8)
#define KV_K_Q8 1
#define KV_V_Q8 1
#elif defined(KV_Q4)
#define KV_K_Q4 1
#define KV_V_Q4 1
#endif
layout(set = 0, binding = 1, std430) readonly  buffer BufK       { uint kw[]; };
layout(set = 0, binding = 2, std430) readonly  buffer BufV       { uint vw[]; };
vec4 q8x4(uint w, float sc) { return vec4(bitfieldExtract(int(w), 0, 8), bitfieldExtract(int(w), 8, 8), bitfieldExtract(int(w), 16, 8), bitfieldExtract(int(w), 24, 8)) * sc; }
vec4 q4x4(uint h, float sc) { return (vec4(float(h & 15u), float((h >> 4u) & 15u), float((h >> 8u) & 15u), float((h >> 12u) & 15u)) - 8.0) * sc; }
// Q8_0: (data word, next data word, scale word). Q4_0: (data word, scale word).
#define Q8_LOAD(buf, i) uvec3(buf[((i) >> 5u) * 9u + 1u + (((i) & 31u) >> 2u)], buf[((i) >> 5u) * 9u + 2u + (((i) & 31u) >> 2u)], buf[((i) >> 5u) * 9u])
#define Q4_LOAD(buf, i) uvec2(buf[((i) >> 5u) * 5u + 1u + (((i) & 31u) >> 3u)], buf[((i) >> 5u) * 5u])
#if defined(KV_K_Q8)
#define RAWK uvec3
RAWK loadK8(uint i) { return Q8_LOAD(kw, i); }
vec4 loK4(RAWK r) { return q8x4(r.x, uintBitsToFloat(r.z)); }
vec4 hiK4(RAWK r) { return q8x4(r.y, uintBitsToFloat(r.z)); }
#else
#define RAWK uvec2
RAWK loadK8(uint i) { return Q4_LOAD(kw, i); }
vec4 loK4(RAWK r) { return q4x4(r.x, uintBitsToFloat(r.y)); }
vec4 hiK4(RAWK r) { return q4x4(r.x >> 16u, uintBitsToFloat(r.y)); }
#endif
#if defined(KV_V_Q8)
#define RAWV uvec3
RAWV loadV8(uint i) { return Q8_LOAD(vw, i); }
vec4 loV4(RAWV r) { return q8x4(r.x, uintBitsToFloat(r.z)); }
vec4 hiV4(RAWV r) { return q8x4(r.y, uintBitsToFloat(r.z)); }
#else
#define RAWV uvec2
RAWV loadV8(uint i) { return Q4_LOAD(vw, i); }
vec4 loV4(RAWV r) { return q4x4(r.x, uintBitsToFloat(r.y)); }
vec4 hiV4(RAWV r) { return q4x4(r.x >> 16u, uintBitsToFloat(r.y)); }
#endif
#elif defined(KV_F16)
layout(set = 0, binding = 1, std430) readonly  buffer BufK       { uvec4 k8[]; };      // [seq_kv, num_kv_heads * head_dim] F16, 8 per element
layout(set = 0, binding = 2, std430) readonly  buffer BufV       { uvec4 v8[]; };
#define RAW uvec4
RAW loadK8(uint i) { return k8[i >> 3u]; }
RAW loadV8(uint i) { return v8[i >> 3u]; }
vec4 lo4(RAW r) { return vec4(unpackHalf2x16(r.x), unpackHalf2x16(r.y)); }
vec4 hi4(RAW r) { return vec4(unpackHalf2x16(r.z), unpackHalf2x16(r.w)); }
#else
layout(set = 0, binding = 1, std430) readonly  buffer BufK       { vec4 k4[]; };       // [seq_kv, num_kv_heads * head_dim] F32
layout(set = 0, binding = 2, std430) readonly  buffer BufV       { vec4 v4[]; };
#define RAW mat2x4
RAW loadK8(uint i) { return mat2x4(k4[i >> 2u], k4[(i >> 2u) + 1u]); }
RAW loadV8(uint i) { return mat2x4(v4[i >> 2u], v4[(i >> 2u) + 1u]); }
vec4 lo4(RAW r) { return r[0]; }
vec4 hi4(RAW r) { return r[1]; }
#endif
#if !defined(RAWK)
#define RAWK RAW
#define RAWV RAW
#define loK4 lo4
#define hiK4 hi4
#define loV4 lo4
#define hiV4 hi4
#endif
// #1926 E4: PART_F16 = 0 writes the F32 un-normalized accumulator (the control, bit-identical to the pre-change body: a float stored
// through its bits); PART_F16 = 1 writes the accumulator NORMALIZED by the split's running sum as packed halves, two per word, so
// |value| <= max|v| and the F16 range cannot be exceeded by the softmax mass of a long split. attention_f32_splitkv_merge.comp is
// specialised the same way and multiplies the split's running sum back in. partMS stays F32 (an F16 running max near 20 would move
// exp() weights by ~1%, docs/PRECISION.md).
layout(constant_id = 1) const uint PART_F16 = 0u;
layout(set = 0, binding = 3, std430) writeonly buffer BufPartOut { uint partOutW[]; }; // F32: [numHeads*numSplits, head_dim]; F16: [.., head_dim / 2]
layout(set = 0, binding = 4, std430) writeonly buffer BufPartMS  { float partMS[]; };  // [numHeads*numSplits, 2] (running max, running sum)

layout(push_constant) uniform PushConstants {
    uint seqQ;            // must be 1 (decode); the single query is at position positionOffset
    uint seqKv;
    uint numHeads;
    uint numKvHeads;
    uint headDim;
    uint positionOffset;
    uint slidingWindow;   // 0 = disabled
    uint useAlibi;        // 0 = disabled, 1 = standard fixed ALiBi slopes
    float softCap;        // 0.0 = disabled
    float scaleOverride;  // 0.0 = use 1/sqrt(headDim)
    uint maskMode;        // 0 = Causal, 1 = Bidirectional, 2 = Hybrid
    uint prefixLen;       // Hybrid only
    uint numSplits;       // KV splits per KV head
    uint kvStart;         // first KV position the splits cover
} pc;

bool isMasked(uint tkv, uint posQ, uint maskMode, uint prefixLen) {
    if (maskMode == 1u) return false;
    if (maskMode == 2u && posQ >= prefixLen) return false;
    return tkv > posQ;
}

uint previousPowerOfTwo(uint value) {
    uint power = 1u;
    while ((power << 1u) <= value) power <<= 1u;
    return power;
}

float alibiSlopePowerOfTwo(uint head, uint numHeads) {
    float start = exp2(-exp2(-(log2(float(numHeads)) - 3.0)));
    return pow(start, float(head + 1u));
}

float alibiSlope(uint head, uint numHeads) {
    uint power = previousPowerOfTwo(numHeads);
    if (power == numHeads || head < power) return alibiSlopePowerOfTwo(head, power);
    uint expanded = power << 1u;
    uint extraHead = (head - power) << 1u;
    return alibiSlopePowerOfTwo(extraHead, expanded);
}

shared vec4 qShared[MAX_GROUP * MAX_HEAD_DIM / 4];
shared float scoreTile[MAX_GROUP * TILE_KV];
shared float runMax[MAX_GROUP];
shared float runSum[MAX_GROUP];
shared float corr[MAX_GROUP];

void main() {
    uint blockId = gl_WorkGroupID.x;

    // A KV head's G query heads are served in nSubG balanced sub-groups of at most MAX_GROUP heads (see the header); this
    // workgroup serves heads hq0 .. hq0 + G - 1 of KV head hkv, where G is now the sub-group's size.
    uint Gfull    = pc.numHeads / pc.numKvHeads;
    uint nSubG    = (Gfull + uint(MAX_GROUP) - 1u) / uint(MAX_GROUP);
    uint Gsub     = (Gfull + nSubG - 1u) / nSubG;
    uint units    = pc.numKvHeads * nSubG;
    if (blockId >= units * pc.numSplits) return;

    // (KV head, sub-group) units interleave across consecutive workgroups: a cache row holds every KV head's columns, so
    // co-resident workgroups read neighbouring slices of the same rows. Split-major order (one head's slice of many rows)
    // measured 90 vs 200 GB/s on Strix.
    uint splitId  = blockId / units;
    uint unit     = blockId - splitId * units;
    uint hkv      = unit / nSubG;
    uint sub      = unit - hkv * nSubG;
    uint hq0      = hkv * Gfull + sub * Gsub;
    uint G        = min(Gsub, Gfull - sub * Gsub);
    uint tid      = gl_LocalInvocationID.x;
    uint hd       = pc.headDim;
    uint hd4      = hd >> 2u;
    uint kvStride = pc.numKvHeads * hd;
    uint posQ     = pc.positionOffset;
    float scale   = (pc.scaleOverride > 0.0) ? pc.scaleOverride : inversesqrt(float(hd));

    uint spanLen    = pc.seqKv - pc.kvStart;
    uint splitLen   = (spanLen + pc.numSplits - 1u) / pc.numSplits;
    uint chunkStart = pc.kvStart + splitId * splitLen;
    uint chunkEnd   = min(chunkStart + splitLen, pc.seqKv);

    for (uint i = tid; i < G * hd4; i += WG_SIZE) {
        uint g = i / hd4;
        uint c = i - g * hd4;
        uint b = (hq0 + g) * hd + c * 4u;
        qShared[g * (MAX_HEAD_DIM / 4) + c] = vec4(q[b], q[b + 1u], q[b + 2u], q[b + 3u]);
    }
    if (tid < G) {
        runMax[tid] = NEG_INF;
        runSum[tid] = 0.0;
    }

    uint lane = tid & (LANES - 1u);   // columns 8 * lane .. 8 * lane + 7
    uint slot = tid / LANES;          // KV position slot within a pass
    bool laneOn = lane * 8u < hd;
    uint q4 = lane * 2u;              // this lane's first vec4 of a Q row in qShared

#if defined(KV_TQ)
    if (tid < (1u << TQ_MSE_BITS)) tqCentSh[tid] = tqCent[tid];   // published by the barrier below, before the first tile
#endif
    vec4 acc[GROUP_PAD * 2u];
    [[unroll]] for (uint i = 0u; i < GROUP_PAD * 2u; i++) acc[i] = vec4(0.0);
    barrier();

    uint nSub    = gl_NumSubgroups;
    uint sgId    = gl_SubgroupID;
    uint sgLane  = gl_SubgroupInvocationID;
    uint sgSize  = gl_SubgroupSize;
    uint colBase = hkv * hd + lane * 8u;

    for (uint tStart = chunkStart; tStart < chunkEnd; tStart += uint(TILE_KV)) {
        uint tileLen = min(tStart + uint(TILE_KV), chunkEnd) - tStart;

        // 1. Scores. All K loads first (an out-of-tile slot re-reads the tile's last row, so no load depends on the slot), then
        //    the dot products; `t` is uniform across each 16-lane cluster and every lane reaches the clustered adds.
        RAWK kr[PASSES];
        [[unroll]] for (uint p = 0u; p < PASSES; p++) {
            uint t = min(p * SLOTS + slot, tileLen - 1u);
            kr[p] = laneOn ? loadK8((tStart + t) * kvStride + colBase) : RAWK(0);
        }
        [[unroll]] for (uint p = 0u; p < PASSES; p++) {
            uint t = p * SLOTS + slot;
            uint tkv = tStart + t;
            vec4 ka = loK4(kr[p]);
            vec4 kb = hiK4(kr[p]);
            float part[GROUP];
            [[unroll]] for (uint g = 0u; g < GROUP; g++) {
                part[g] = 0.0;
                if (g < G && laneOn)
                    part[g] = dot(qShared[g * (MAX_HEAD_DIM / 4) + q4], ka) + dot(qShared[g * (MAX_HEAD_DIM / 4) + q4 + 1u], kb);
            }
            [[unroll]] for (uint g = 0u; g < GROUP; g++) part[g] = subgroupClusteredAdd(part[g], LANES);
            if (t < tileLen && lane < G) {
                bool masked =
                    isMasked(tkv, posQ, pc.maskMode, pc.prefixLen) ||
                    (pc.slidingWindow > 0u && posQ >= tkv && (posQ - tkv) >= pc.slidingWindow);
                float sc = NEG_INF;
                if (!masked) {
                    float mine = part[0];
                    [[unroll]] for (uint g = 1u; g < GROUP; g++) {
                        if (lane == g) mine = part[g];
                    }
                    float alibi = pc.useAlibi != 0u ? alibiSlope(hq0 + lane, pc.numHeads) * float(posQ - tkv) : 0.0;
                    sc = mine * scale - alibi;
                    if (pc.softCap > 0.0) sc = pc.softCap * tanh(sc / pc.softCap);
                }
                scoreTile[lane * TILE_KV + t] = sc;
            }
        }

        // The V rows do not depend on the softmax: issue them before the barrier so their latency overlaps it.
        RAWV vr[PASSES];
        [[unroll]] for (uint p = 0u; p < PASSES; p++) {
            uint t = min(p * SLOTS + slot, tileLen - 1u);
            vr[p] = laneOn ? loadV8((tStart + t) * kvStride + colBase) : RAWV(0);
        }
        barrier();

        // 2. Online softmax per query head: subgroup s owns heads s, s + nSub, ...
        for (uint g = sgId; g < G; g += nSub) {
            float m = NEG_INF;
            for (uint t = sgLane; t < tileLen; t += sgSize) m = max(m, scoreTile[g * TILE_KV + t]);
            float tileMax = subgroupMax(m);
            float oldMax = runMax[g];
            float newMax = max(oldMax, tileMax);
            float c = (oldMax > NEG_INF * 0.5) ? exp(oldMax - newMax) : 0.0;
            float s = 0.0;
            for (uint t = sgLane; t < tileLen; t += sgSize) {
                float sc = scoreTile[g * TILE_KV + t];
                float w = (sc > NEG_INF * 0.5) ? exp(sc - newMax) : 0.0;
                scoreTile[g * TILE_KV + t] = w;
                s += w;
            }
            float tileSum = subgroupAdd(s);
            if (subgroupElect()) {
                runSum[g] = runSum[g] * c + tileSum;
                runMax[g] = newMax;
                corr[g] = c;
            }
        }
        barrier();

        // 3. P.V on the rows loaded above; an out-of-tile slot adds nothing.
        [[unroll]] for (uint g = 0u; g < GROUP; g++) {
            float cg = g < G ? corr[g] : 0.0;
            acc[2u * g] *= cg;
            acc[2u * g + 1u] *= cg;
        }
        [[unroll]] for (uint p = 0u; p < PASSES; p++) {
            uint t = p * SLOTS + slot;
            if (t < tileLen) {
                vec4 va = loV4(vr[p]);
                vec4 vb = hiV4(vr[p]);
                [[unroll]] for (uint g = 0u; g < GROUP; g++) {
                    if (g < G) {
                        float w = scoreTile[g * TILE_KV + t];
                        acc[2u * g] += w * va;
                        acc[2u * g + 1u] += w * vb;
                    }
                }
            }
        }
        barrier();
    }

    // 4. Reduce the 16 slots: the 4 slots of a wave64 by shuffles, then the waves through qShared (dead after the tile loop), two
    //    query heads per round (4 waves x 2 heads x 32 vec4 fits exactly). Partials go out in the layout of
    //    attention_f32_splitkv.glsl: row (hq * numSplits + split).
    [[unroll]] for (uint i = 0u; i < GROUP * 2u; i++) {
        acc[i] += subgroupShuffleXor(acc[i], 16u);
        acc[i] += subgroupShuffleXor(acc[i], 32u);
    }
    [[unroll]] for (uint r = 0u; r < GROUP_PAD / 2u; r++) {
        if (2u * r >= G) break;
        if (sgLane < LANES) {
            [[unroll]] for (uint gg = 0u; gg < 2u; gg++) {
                qShared[(sgId * 2u + gg) * (MAX_HEAD_DIM / 4) + q4] = acc[2u * (2u * r + gg)];
                qShared[(sgId * 2u + gg) * (MAX_HEAD_DIM / 4) + q4 + 1u] = acc[2u * (2u * r + gg) + 1u];
            }
        }
        barrier();
        for (uint i = tid; i < 2u * hd4; i += WG_SIZE) {
            uint gg = i / hd4;
            uint c = i - gg * hd4;
            uint g = 2u * r + gg;
            if (g < G) {
                vec4 sum = vec4(0.0);
                for (uint w = 0u; w < nSub; w++) sum += qShared[(w * 2u + gg) * (MAX_HEAD_DIM / 4) + c];
                uint row = (hq0 + g) * pc.numSplits + splitId;
                if (PART_F16 != 0u) {
                    float rs = runSum[g];
                    vec4 o4 = clamp(sum * (rs > 0.0 ? 1.0 / rs : 0.0), vec4(-65504.0), vec4(65504.0));
                    uint ow = row * (hd >> 1u) + c * 2u;
                    partOutW[ow] = packHalf2x16(o4.xy);
                    partOutW[ow + 1u] = packHalf2x16(o4.zw);
                } else {
                    uint o = row * hd + c * 4u;
                    partOutW[o] = floatBitsToUint(sum.x);
                    partOutW[o + 1u] = floatBitsToUint(sum.y);
                    partOutW[o + 2u] = floatBitsToUint(sum.z);
                    partOutW[o + 3u] = floatBitsToUint(sum.w);
                }
            }
        }
        barrier();
    }
    if (tid < G) {
        uint row = (hq0 + tid) * pc.numSplits + splitId;
        partMS[row * 2u + 0u] = runMax[tid];
        partMS[row * 2u + 1u] = runSum[tid];
    }
}
