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
// Bounds: G <= MAX_GROUP, headDim <= MAX_HEAD_DIM and a multiple of 8, wave64 (the host gates all three).
// LDS: 4 KiB Q (reused by the final reduce) + TILE_KV * 32 B scores.

#define TILE_KV      128
#define MAX_HEAD_DIM 128
#define MAX_GROUP    8
#define WG_SIZE      256
#define LANES        16                     // lanes per KV position, 8 columns each
#define SLOTS        (WG_SIZE / LANES)      // KV positions per pass
#define PASSES       (TILE_KV / SLOTS)
#define NEG_INF      (-3.4e38)

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ       { float q[]; };       // [1, numHeads * head_dim]
#if defined(KV_F16)
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

shared vec4 qShared[MAX_GROUP * MAX_HEAD_DIM / 4];
shared float scoreTile[MAX_GROUP * TILE_KV];
shared float runMax[MAX_GROUP];
shared float runSum[MAX_GROUP];
shared float corr[MAX_GROUP];

void main() {
    uint blockId = gl_WorkGroupID.x;
    if (blockId >= pc.numKvHeads * pc.numSplits) return;

    // KV heads interleave across consecutive workgroups: a cache row holds every KV head's columns, so co-resident workgroups
    // read neighbouring slices of the same rows. Split-major order (one head's slice of many rows) measured 90 vs 200 GB/s on Strix.
    uint splitId  = blockId / pc.numKvHeads;
    uint hkv      = blockId - splitId * pc.numKvHeads;
    uint G        = pc.numHeads / pc.numKvHeads;
    uint hq0      = hkv * G;
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

    vec4 acc[MAX_GROUP * 2];
    [[unroll]] for (uint i = 0u; i < MAX_GROUP * 2u; i++) acc[i] = vec4(0.0);
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
        RAW kr[PASSES];
        [[unroll]] for (uint p = 0u; p < PASSES; p++) {
            uint t = min(p * SLOTS + slot, tileLen - 1u);
            kr[p] = laneOn ? loadK8((tStart + t) * kvStride + colBase) : RAW(0);
        }
        [[unroll]] for (uint p = 0u; p < PASSES; p++) {
            uint t = p * SLOTS + slot;
            uint tkv = tStart + t;
            vec4 ka = lo4(kr[p]);
            vec4 kb = hi4(kr[p]);
            float part[MAX_GROUP];
            [[unroll]] for (uint g = 0u; g < MAX_GROUP; g++) {
                part[g] = 0.0;
                if (g < G && laneOn)
                    part[g] = dot(qShared[g * (MAX_HEAD_DIM / 4) + q4], ka) + dot(qShared[g * (MAX_HEAD_DIM / 4) + q4 + 1u], kb);
            }
            [[unroll]] for (uint g = 0u; g < MAX_GROUP; g++) part[g] = subgroupClusteredAdd(part[g], LANES);
            if (t < tileLen && lane < G) {
                bool masked =
                    isMasked(tkv, posQ, pc.maskMode, pc.prefixLen) ||
                    (pc.slidingWindow > 0u && posQ >= tkv && (posQ - tkv) >= pc.slidingWindow);
                float sc = NEG_INF;
                if (!masked) {
                    float mine = part[0];
                    [[unroll]] for (uint g = 1u; g < MAX_GROUP; g++) {
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
        RAW vr[PASSES];
        [[unroll]] for (uint p = 0u; p < PASSES; p++) {
            uint t = min(p * SLOTS + slot, tileLen - 1u);
            vr[p] = laneOn ? loadV8((tStart + t) * kvStride + colBase) : RAW(0);
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
        [[unroll]] for (uint g = 0u; g < MAX_GROUP; g++) {
            float cg = g < G ? corr[g] : 0.0;
            acc[2u * g] *= cg;
            acc[2u * g + 1u] *= cg;
        }
        [[unroll]] for (uint p = 0u; p < PASSES; p++) {
            uint t = p * SLOTS + slot;
            if (t < tileLen) {
                vec4 va = lo4(vr[p]);
                vec4 vb = hi4(vr[p]);
                [[unroll]] for (uint g = 0u; g < MAX_GROUP; g++) {
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
    [[unroll]] for (uint i = 0u; i < MAX_GROUP * 2u; i++) {
        acc[i] += subgroupShuffleXor(acc[i], 16u);
        acc[i] += subgroupShuffleXor(acc[i], 32u);
    }
    [[unroll]] for (uint r = 0u; r < MAX_GROUP / 2u; r++) {
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
                uint o = ((hq0 + g) * pc.numSplits + splitId) * hd + c * 4u;
                partOut[o] = sum.x;
                partOut[o + 1u] = sum.y;
                partOut[o + 2u] = sum.z;
                partOut[o + 3u] = sum.w;
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
