// Body of attention_f32_splitkv_rows.comp and its F16-KV twin attention_f32_splitkv_rows_kvf16.comp (issue #1171; the #1102 kv_load.glsl split).
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

// Split-KV (flash-decoding) PARTIAL pass for a SMALL BATCH OF ROWS (speculative verify, seqQ = 2..8), with per-row query offsets:
// row r is at position positionOffset + r and attends to the first positionOffset + r + 1 keys. Everything below the dispatch geometry is
// attention_f32_splitkv.comp verbatim, so each row reproduces the 1-row decode kernel's result bit for bit (see the rows-merge pass too).
//
// Original header (attention_f32_splitkv.comp):
// Split-KV (flash-decoding) PARTIAL pass. Decode-only (seqQ == 1).
//
// One workgroup per (query head, KV split). Each workgroup runs the same tiled
// flash-attention online softmax as attention_f32_sg.comp, but over ONLY its
// split's KV sub-range [chunkStart, chunkEnd), and it writes the UN-normalized
// partial state — running max m, running sum l, and the un-divided weighted-V
// accumulator acc[head_dim] — for attention_f32_splitkv_merge.comp to combine
// across splits. This is the classic Flash-Decoding decomposition.
//
// Two differences from attention_f32_sg.comp (everything else is identical):
//   1. The Q.K score loop is COALESCED: one subgroup computes one KV position,
//      lanes map across head_dim so consecutive lanes read consecutive K
//      addresses (k[kBase + d]), reduced with a single subgroupAdd. The naive
//      per-token kernel maps thread->KV-position, which strides K by kvStride
//      (uncoalesced). Because each subgroup reduces its own position, this is
//      portable across subgroup widths with no cross-subgroup combine.
//   2. The KV range is the split's chunk, not [0, seqKv), and the output is the
//      partial (m, l, acc) rather than the normalized row.
//
// Raising the workgroup count from numHeads to numHeads*numSplits is the whole
// point: it lifts decode occupancy above the ~numHeads floor where the coalesced
// read alone regressed (it serialized KV positions at low occupancy).

#define TILE_KV      256
#define MAX_HEAD_DIM 512
#define WG_SIZE      256
#define NEG_INF      (-3.4e38)

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ       { float q[]; };       // [1,       numHeads    * head_dim]
#include "kv_load.glsl"                                       // BufK / BufV (binding 1 / 2), KV_K(i) / KV_V(i)
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
    uint maxSplits;       // split slots per (row, head) in the partial scratch (= the splits of the longest row, >= 2)
    uint targetWg;        // split heuristic of the 1-row kernel (VulkanSplitKvAttentionKernel.ComputeSplits), so each row picks ITS 1-row split count
    uint minKv;
    uint hasSinks;        // 1 = attention-sink layer (#996): the KV span of row r is clipped to its sliding window exactly as the 1-row sink decode path does (kvStart = posQ - W + 1)
} pc;

// Splits the 1-row decode kernel would use for a context of kvLen keys (VulkanSplitKvAttentionKernel.ComputeSplits).
uint splitsFor(uint kvLen) {
    uint byKv = (kvLen + pc.minKv - 1u) / pc.minKv;
    uint byOcc = max(1u, pc.targetWg / pc.numHeads);
    return max(1u, min(byOcc, byKv));
}

// Identical to attention_f32_sg.comp::isMasked.
bool isMasked(uint tkv, uint posQ, uint maskMode, uint prefixLen) {
    if (maskMode == 1u) return false;
    if (maskMode == 2u && posQ >= prefixLen) return false;
    return tkv > posQ;
}

shared float qShared[MAX_HEAD_DIM];
shared float scoreTile[TILE_KV];
shared float outAccum[MAX_HEAD_DIM];

shared float partialMax[64];
shared float partialSum[64];
shared float broadcastSlot;

// Cross-subgroup workgroup max/sum — copied verbatim from attention_f32_sg.comp.
float workgroupMaxSg(float val) {
    float sgMax = subgroupMax(val);
    if (subgroupElect()) partialMax[gl_SubgroupID] = sgMax;
    barrier();
    if (gl_SubgroupID == 0u) {
        float x = (gl_SubgroupInvocationID < gl_NumSubgroups) ? partialMax[gl_SubgroupInvocationID] : NEG_INF;
        float reduced = subgroupMax(x);
        if (subgroupElect()) broadcastSlot = reduced;
    }
    barrier();
    float m = broadcastSlot;
    barrier();
    return m;
}

float workgroupSumSg(float val) {
    float sgSum = subgroupAdd(val);
    if (subgroupElect()) partialSum[gl_SubgroupID] = sgSum;
    barrier();
    if (gl_SubgroupID == 0u) {
        float x = (gl_SubgroupInvocationID < gl_NumSubgroups) ? partialSum[gl_SubgroupInvocationID] : 0.0;
        float reduced = subgroupAdd(x);
        if (subgroupElect()) broadcastSlot = reduced;
    }
    barrier();
    float s = broadcastSlot;
    barrier();
    return s;
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

void main() {
    // One workgroup per (row, query head, split slot). Row r is the query at position positionOffset + r; it attends to the
    // positionOffset + r + 1 keys up to and including itself, split exactly as the 1-row kernel would split a decode step at that
    // length (same split count, same chunk boundaries, same per-split code), so the row is bit-identical to that decode step.
    uint blockId = gl_WorkGroupID.x;
    uint perRow  = pc.numHeads * pc.maxSplits;
    if (blockId >= pc.seqQ * perRow) return;

    uint row     = blockId / perRow;
    uint rem     = blockId - row * perRow;
    uint hq      = rem / pc.maxSplits;
    uint splitId = rem - hq * pc.maxSplits;

    uint rowKv     = pc.positionOffset + row + 1u;
    // Sink layers clip the span to the window (the 1-row sink decode passes kvStart = posQ - W + 1 and splits the clipped span); others scan [0, rowKv).
    uint kvStart   = (pc.hasSinks != 0u && pc.slidingWindow > 0u && rowKv > pc.slidingWindow) ? rowKv - pc.slidingWindow : 0u;
    uint spanLen   = rowKv - kvStart;
    uint numSplits = splitsFor(spanLen);
    if (splitId >= numSplits) return;   // uniform per workgroup; this row needs fewer splits than the longest

    uint groupSize = pc.numHeads / pc.numKvHeads;
    uint hkv = hq / groupSize;

    uint tid = gl_LocalInvocationID.x;

    uint qStride  = pc.numHeads   * pc.headDim;
    uint kvStride = pc.numKvHeads * pc.headDim;

    uint posQ = pc.positionOffset + row;
    float scale = (pc.scaleOverride > 0.0) ? pc.scaleOverride : inversesqrt(float(pc.headDim));
    float slope = (pc.useAlibi != 0u) ? alibiSlope(hq, pc.numHeads) : 0.0;

    uint qBase = (row * pc.numHeads + hq) * pc.headDim;

    // This split's KV chunk.
    uint splitLen   = (spanLen + numSplits - 1u) / numSplits;
    uint chunkStart = kvStart + splitId * splitLen;
    uint chunkEnd   = min(chunkStart + splitLen, rowKv);

    // 1. Load Q and zero the accumulator.
    for (uint d = tid; d < pc.headDim; d += WG_SIZE) {
        qShared[d]  = q[qBase + d];
        outAccum[d] = 0.0;
    }
    barrier();

    float runningMax = NEG_INF;
    float runningSum = 0.0;

    // Subgroup geometry for the coalesced score loop.
    uint nSub   = gl_NumSubgroups;
    uint sgId   = gl_SubgroupID;
    uint lane   = gl_SubgroupInvocationID;
    uint sgSize = gl_SubgroupSize;

    for (uint tStart = chunkStart; tStart < chunkEnd; tStart += uint(TILE_KV)) {
        uint tEnd = min(tStart + uint(TILE_KV), chunkEnd);
        uint tileLen = tEnd - tStart;

        // 2. Coalesced scores: subgroup -> KV position, lane -> head_dim.
        //    `masked` is subgroup-uniform (depends only on tkv/posQ), so the
        //    subgroupAdd below executes under uniform control flow.
        for (uint t = sgId; t < tileLen; t += nSub) {
            uint tkv = tStart + t;
            bool masked =
                isMasked(tkv, posQ, pc.maskMode, pc.prefixLen) ||
                (pc.slidingWindow > 0u && posQ >= tkv && (posQ - tkv) >= pc.slidingWindow);
            float score;
            if (masked) {
                score = NEG_INF;
            } else {
                uint kBase = tkv * kvStride + hkv * pc.headDim;
                float partial = 0.0;
                for (uint d = lane; d < pc.headDim; d += sgSize) {
                    partial += qShared[d] * KV_K(kBase + d);
                }
                float dotqk = subgroupAdd(partial);
                score = dotqk * scale - slope * float(posQ - tkv);
                if (pc.softCap > 0.0) score = pc.softCap * tanh(score / pc.softCap);
            }
            if (subgroupElect()) scoreTile[t] = score;
        }
        barrier();

        // 3. Tile max reduction.
        float localMax = NEG_INF;
        for (uint t = tid; t < tileLen; t += WG_SIZE) localMax = max(localMax, scoreTile[t]);
        float tileMax = workgroupMaxSg(localMax);

        // 4. Online-softmax rescale of running state.
        float newMax = max(runningMax, tileMax);
        float correction = (runningMax > NEG_INF * 0.5) ? exp(runningMax - newMax) : 0.0;
        runningSum *= correction;
        for (uint d = tid; d < pc.headDim; d += WG_SIZE) outAccum[d] *= correction;
        runningMax = newMax;
        barrier();

        // 5. exp(score - newMax); accumulate tile sum.
        float localSum = 0.0;
        for (uint t = tid; t < tileLen; t += WG_SIZE) {
            float s = scoreTile[t];
            float w = (s > NEG_INF * 0.5) ? exp(s - runningMax) : 0.0;
            scoreTile[t] = w;
            localSum += w;
        }
        float tileSum = workgroupSumSg(localSum);
        runningSum += tileSum;
        barrier();

        // 6. Weighted V (coalesced V read: thread -> head_dim).
        for (uint d = tid; d < pc.headDim; d += WG_SIZE) {
            float vAcc = 0.0;
            for (uint t = 0u; t < tileLen; t++) {
                float w = scoreTile[t];
                if (w > 0.0) {
                    uint vBase = (tStart + t) * kvStride + hkv * pc.headDim;
                    vAcc += w * KV_V(vBase + d);
                }
            }
            outAccum[d] += vAcc;
        }
        barrier();
    }

    // 7. Write the un-normalized partial (acc, max, sum). An empty chunk leaves
    //    runningMax = NEG_INF / runningSum = 0, contributing nothing in the merge.
    uint partBase = blockId * pc.headDim;
    for (uint d = tid; d < pc.headDim; d += WG_SIZE) {
        partOut[partBase + d] = outAccum[d];
    }
    if (tid == 0u) {
        partMS[blockId * 2u + 0u] = runningMax;
        partMS[blockId * 2u + 1u] = runningSum;
    }
}
