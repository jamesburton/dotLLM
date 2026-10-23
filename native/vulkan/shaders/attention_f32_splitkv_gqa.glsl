#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#if defined(STATS16) || defined(PV16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

// GQA-grouped split-KV (flash-decoding) PARTIAL pass, decode only (seqQ == 1). Issue #1102.
//
// attention_f32_splitkv.glsl runs one workgroup per (QUERY head, split), so the G = numHeads / numKvHeads query heads of a
// KV head each stream that KV head's rows from memory: G reads of every cached byte per decoded token (Bonsai 2: G = 6).
// At long context those reads are the decode cost. Here one workgroup per (KV head, split) reads each K and V element ONCE
// and applies it to all G query heads:
//
//   - scores: one subgroup per KV position, lanes across head_dim; one K load feeds G dot products (G subgroupAdds);
//   - online softmax per query head: subgroup s owns heads s, s + nSub, ... (subgroupMax / subgroupAdd over the tile);
//   - P.V: thread d owns column d (head_dim <= WG_SIZE) and keeps G accumulators in registers; one V load feeds G FMAs.
//
// It writes the same un-normalized partials, per (query head, split), as attention_f32_splitkv.glsl, so the same merge pass
// (attention_f32_splitkv_merge.comp) finishes the row. Masking (causal / bidirectional / hybrid), sliding window, ALiBi,
// soft-cap, scale override and kvStart follow attention_f32_splitkv.glsl. Push constants are identical; the dispatch is
// numKvHeads * numSplits workgroups.
//
// Precision-study variants (issue #1109; experiment only, selected by DOTLLM_VK_EXP_DECODE_ACC; the shipped pipelines define
// none of these and compile to the same SPIR-V as before):
//   STATS16  the running max, the correction and the running sum are rounded to F16 at every update, and each lane's share of
//            a tile's sum accumulates in F16 (the cross-lane reduction and the exp stay F32).
//   PV16     the P.V accumulators are F16 (P rounded to F16, V exact), rescaled by an F16 correction.
//
// Bounds: G <= MAX_GROUP, head_dim <= MAX_HEAD_DIM (= WG_SIZE). LDS: 8 KiB Q + 4 KiB scores.

#define TILE_KV      128
#define MAX_HEAD_DIM 256
#define MAX_GROUP    8
#define WG_SIZE      256
#define NEG_INF      (-3.4e38)

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ       { float q[]; };       // [1, numHeads * head_dim]
#include "kv_load.glsl"                                                               // BufK / BufV (binding 1 / 2), KV_K(i) / KV_V(i)
layout(set = 0, binding = 3, std430) writeonly buffer BufPartOut { float partOut[]; }; // [numHeads*numSplits, head_dim] (un-normalized acc)
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

shared float qShared[MAX_GROUP * MAX_HEAD_DIM];
shared float scoreTile[MAX_GROUP * TILE_KV];
shared float runMax[MAX_GROUP];
shared float runSum[MAX_GROUP];
shared float corr[MAX_GROUP];

void main() {
    uint blockId = gl_WorkGroupID.x;
    if (blockId >= pc.numKvHeads * pc.numSplits) return;

    uint hkv     = blockId / pc.numSplits;
    uint splitId = blockId - hkv * pc.numSplits;
    uint G       = pc.numHeads / pc.numKvHeads;
    uint hq0     = hkv * G;
    uint tid     = gl_LocalInvocationID.x;
    uint kvStride = pc.numKvHeads * pc.headDim;
    uint posQ    = pc.positionOffset;
    float scale  = (pc.scaleOverride > 0.0) ? pc.scaleOverride : inversesqrt(float(pc.headDim));

    uint spanLen    = pc.seqKv - pc.kvStart;
    uint splitLen   = (spanLen + pc.numSplits - 1u) / pc.numSplits;
    uint chunkStart = pc.kvStart + splitId * splitLen;
    uint chunkEnd   = min(chunkStart + splitLen, pc.seqKv);

    for (uint i = tid; i < G * pc.headDim; i += WG_SIZE) {
        uint g = i / pc.headDim;
        uint d = i - g * pc.headDim;
        qShared[g * MAX_HEAD_DIM + d] = q[(hq0 + g) * pc.headDim + d];
    }
    if (tid < G) {
        runMax[tid] = NEG_INF;
        runSum[tid] = 0.0;
    }

    float slopes[MAX_GROUP];
#if defined(PV16)
    float16_t acc[MAX_GROUP];
    for (uint g = 0u; g < MAX_GROUP; g++) {
        slopes[g] = (pc.useAlibi != 0u && g < G) ? alibiSlope(hq0 + g, pc.numHeads) : 0.0;
        acc[g] = float16_t(0.0);
    }
#else
    float acc[MAX_GROUP];
    for (uint g = 0u; g < MAX_GROUP; g++) {
        slopes[g] = (pc.useAlibi != 0u && g < G) ? alibiSlope(hq0 + g, pc.numHeads) : 0.0;
        acc[g] = 0.0;
    }
#endif
    barrier();

    uint nSub   = gl_NumSubgroups;
    uint sgId   = gl_SubgroupID;
    uint lane   = gl_SubgroupInvocationID;
    uint sgSize = gl_SubgroupSize;
    uint kvHeadBase = hkv * pc.headDim;

    for (uint tStart = chunkStart; tStart < chunkEnd; tStart += uint(TILE_KV)) {
        uint tileLen = min(tStart + uint(TILE_KV), chunkEnd) - tStart;

        // 1. Scores: subgroup -> KV position, lanes -> head_dim. One K load feeds all G query heads. `masked` is
        //    subgroup-uniform, so every subgroupAdd runs under uniform control flow.
        for (uint t = sgId; t < tileLen; t += nSub) {
            uint tkv = tStart + t;
            bool masked =
                isMasked(tkv, posQ, pc.maskMode, pc.prefixLen) ||
                (pc.slidingWindow > 0u && posQ >= tkv && (posQ - tkv) >= pc.slidingWindow);
            if (masked) {
                if (subgroupElect()) {
                    for (uint g = 0u; g < G; g++) scoreTile[g * TILE_KV + t] = NEG_INF;
                }
                continue;
            }

            float part[MAX_GROUP];
            for (uint g = 0u; g < MAX_GROUP; g++) part[g] = 0.0;
            uint kBase = tkv * kvStride + kvHeadBase;
            for (uint d = lane; d < pc.headDim; d += sgSize) {
                float kd = KV_K(kBase + d);
                for (uint g = 0u; g < MAX_GROUP; g++) {
                    if (g < G) part[g] += qShared[g * MAX_HEAD_DIM + d] * kd;
                }
            }
            for (uint g = 0u; g < MAX_GROUP; g++) {
                if (g >= G) break;
                float score = subgroupAdd(part[g]) * scale - slopes[g] * float(posQ - tkv);
                if (pc.softCap > 0.0) score = pc.softCap * tanh(score / pc.softCap);
                if (subgroupElect()) scoreTile[g * TILE_KV + t] = score;
            }
        }
        barrier();

        // 2. Online softmax per query head: subgroup s owns heads s, s + nSub, ...
        for (uint g = sgId; g < G; g += nSub) {
            float m = NEG_INF;
            for (uint t = lane; t < tileLen; t += sgSize) m = max(m, scoreTile[g * TILE_KV + t]);
            float tileMax = subgroupMax(m);
            float oldMax = runMax[g];
            float newMax = max(oldMax, tileMax);
#if defined(STATS16)
            if (newMax > NEG_INF * 0.5) newMax = float(float16_t(newMax));
            float c = (oldMax > NEG_INF * 0.5) ? float(float16_t(exp(oldMax - newMax))) : 0.0;
            float16_t s16 = float16_t(0.0);
            for (uint t = lane; t < tileLen; t += sgSize) {
                float sc = scoreTile[g * TILE_KV + t];
                float w = (sc > NEG_INF * 0.5) ? exp(sc - newMax) : 0.0;
                scoreTile[g * TILE_KV + t] = w;
                s16 += float16_t(w);
            }
            float tileSum = subgroupAdd(float(s16));
            if (subgroupElect()) {
                runSum[g] = float(float16_t(runSum[g] * c + tileSum));
                runMax[g] = newMax;
                corr[g] = c;
            }
#else
            float c = (oldMax > NEG_INF * 0.5) ? exp(oldMax - newMax) : 0.0;
            float s = 0.0;
            for (uint t = lane; t < tileLen; t += sgSize) {
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
#endif
        }
        barrier();

        // 3. P.V: thread d owns column d; one V load feeds all G accumulators. A position every head masked is not read.
        if (tid < pc.headDim) {
            for (uint g = 0u; g < MAX_GROUP; g++) {
#if defined(PV16)
                if (g < G) acc[g] *= float16_t(corr[g]);
#else
                if (g < G) acc[g] *= corr[g];
#endif
            }
            for (uint t = 0u; t < tileLen; t++) {
                float wAny = 0.0;
                for (uint g = 0u; g < MAX_GROUP; g++) {
                    if (g < G) wAny += scoreTile[g * TILE_KV + t];
                }
                if (wAny > 0.0) {
                    float vd = KV_V((tStart + t) * kvStride + kvHeadBase + tid);
                    for (uint g = 0u; g < MAX_GROUP; g++) {
#if defined(PV16)
                        if (g < G) acc[g] += float16_t(scoreTile[g * TILE_KV + t]) * float16_t(vd);
#else
                        if (g < G) acc[g] += scoreTile[g * TILE_KV + t] * vd;
#endif
                    }
                }
            }
        }
        barrier();
    }

    // 4. Partials in the layout of attention_f32_splitkv.glsl: row (hq * numSplits + split).
    if (tid < pc.headDim) {
        for (uint g = 0u; g < MAX_GROUP; g++) {
#if defined(PV16)
            if (g < G) partOut[((hq0 + g) * pc.numSplits + splitId) * pc.headDim + tid] = float(acc[g]);
#else
            if (g < G) partOut[((hq0 + g) * pc.numSplits + splitId) * pc.headDim + tid] = acc[g];
#endif
        }
    }
    if (tid < G) {
        uint row = (hq0 + tid) * pc.numSplits + splitId;
        partMS[row * 2u + 0u] = runMax[tid];
        partMS[row * 2u + 1u] = runSum[tid];
    }
}
