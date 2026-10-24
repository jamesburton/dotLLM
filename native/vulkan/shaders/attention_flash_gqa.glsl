// GQA-grouped scalar flash-attention prefill for head_dim <= 256 (issue #441), for devices without cooperative matrices
// (the Meteor Lake Arc).
//
// NOT compiled on its own (.glsl): attention_flash_gqa{,_kvf16}.comp include it and pick the KV storage.
//
// Same row space as attention_flash_gqa_cm.glsl: the query side of one KV head is the matrix of its G query heads' rows,
// row m <-> (token m / G, head hkv * G + m % G), so one K/V tile load feeds every head of the group.
//
// Ownership: a workgroup owns ROWS = 16 rows; 8 consecutive lanes own one row, lane j holding dims 4j + 32i (i < headDim / 32)
// of that row's Q and O in registers. QK^T: each lane dots its 32 dims against the K row from the shared tile, and a 3-step
// shuffle-xor butterfly sums the 8 partials (every lane of the row ends with the bit-identical score, because each butterfly
// step adds the same two values in either order). The online softmax then runs redundantly in the row's 8 lanes, with no
// shared-memory round trip, and P.V accumulates straight into the lane's O registers. The only barriers are the two around the
// K/V tile stage.
//
// Numerics: all arithmetic is F32 (QK^T, softmax, running max / sum, P.V); K / V are converted from their stored format in the
// load that consumes them.
//
// KV storage (the .comp wrapper): KV_F16 reads F16 rows as packed words, the default F32 rows; either is converted to F32 in the
// LDS tile load (2 x 16 KiB tiles, BC = 16), a per-tile transient, never a widened history. No 16-bit device feature is used. Requires subgroups of at least 8 lanes (host-checked).
//
// Dispatch: local_size_x = 128, workgroups = (ceil(seqQ * G / 16), numKvHeads).

#extension GL_KHR_shader_subgroup_basic   : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require

#define ROWS            16
#define TPR             8                        // lanes per row
#define WG_SIZE         (ROWS * TPR)             // 128
#define MAX_HEAD_DIM    256
#define DSTEPS          (MAX_HEAD_DIM / (4 * TPR))   // 8 vec4 granules per lane
// K / V tiles are staged in LDS as F32 whatever the cache stores: a K element is read by all 16 rows of the workgroup, so it is
// converted once in the load instead of once per reading lane (measured on the Arc: the per-lane unpack of an F16 tile doubled
// the inner loop's instruction count). BC = 16 keeps the two tiles at 2 x 16 KiB.
#define BC              16
#define ROW_WORDS       (MAX_HEAD_DIM / 4)       // vec4 per tile row
#define NEG_INF         (-3.4e38)

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ   { vec4 q4[]; };      // [seq_q, num_heads * head_dim] F32
#if defined(KV_F16)
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { uvec2 k4[]; };     // [seq_kv, num_kv_heads * head_dim] F16
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { uvec2 v4[]; };
vec4 loadK(uint i) { uvec2 w = k4[i]; return vec4(unpackHalf2x16(w.x), unpackHalf2x16(w.y)); }
vec4 loadV(uint i) { uvec2 w = v4[i]; return vec4(unpackHalf2x16(w.x), unpackHalf2x16(w.y)); }
#else
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { vec4 k4[]; };      // [seq_kv, num_kv_heads * head_dim] F32
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { vec4 v4[]; };
vec4 loadK(uint i) { return k4[i]; }
vec4 loadV(uint i) { return v4[i]; }
#endif
shared vec4 kT[BC * ROW_WORDS];
shared vec4 vT[BC * ROW_WORDS];
layout(set = 0, binding = 3, std430) writeonly buffer BufOut { vec4 out4[]; };    // [seq_q, num_heads * head_dim] F32

layout(push_constant) uniform PushConstants {
    uint  seqQ;
    uint  seqKv;
    uint  numHeads;
    uint  numKvHeads;
    uint  headDim;          // multiple of 32, <= 256 (host-checked)
    uint  positionOffset;
    uint  slidingWindow;    // 0 = disabled
    uint  useAlibi;         // must be 0 (host routes ALiBi elsewhere)
    float softCap;          // 0 = disabled
    float scaleOverride;    // 0 = 1/sqrt(head_dim)
    uint  maskMode;         // 0 = Causal, 1 = Bidirectional, 2 = Hybrid
    uint  prefixLen;
} pc;

bool isMasked(uint tkv, uint posQ) {
    if (pc.maskMode == 1u) return false;
    if (pc.maskMode == 2u && posQ >= pc.prefixLen) return false;
    return tkv > posQ;
}

void main() {
    uint G         = pc.numHeads / pc.numKvHeads;
    uint totalRows = pc.seqQ * G;
    uint numWg     = (totalRows + uint(ROWS) - 1u) / uint(ROWS);
    uint wgIdx     = numWg - 1u - gl_WorkGroupID.x;   // longest causal ranges first
    uint hkv       = gl_WorkGroupID.y;
    uint wgRow0    = wgIdx * uint(ROWS);
    uint wgRowEnd  = min(wgRow0 + uint(ROWS), totalRows);

    // A 1-D workgroup's subgroups are runs of consecutive invocations, so the 8 lanes of a row (8-aligned) share a subgroup of
    // 8 or more lanes and shuffle-xor 1/2/4 stays inside the row. (gl_SubgroupID * gl_SubgroupSize was tried first and broke on the
    // Meteor Lake Arc: only row 0 of each workgroup came out.)
    uint lin  = gl_LocalInvocationIndex;
    uint r    = lin / uint(TPR);
    uint j    = lin - r * uint(TPR);
    uint m    = wgRow0 + r;
    bool rowLive = m < totalRows;
    uint t    = rowLive ? m / G : 0u;
    uint hq   = hkv * G + (m - t * G);
    uint posQ = pc.positionOffset + t;

    uint hd4       = pc.headDim >> 2u;
    uint qStride4  = (pc.numHeads * pc.headDim) >> 2u;
    uint kvStride4 = (pc.numKvHeads * pc.headDim) >> 2u;
    uint dSteps    = pc.headDim / uint(4 * TPR);
    float scale = (pc.scaleOverride > 0.0) ? pc.scaleOverride : inversesqrt(float(pc.headDim));

    vec4 qr[DSTEPS];
    vec4 o[DSTEPS];
    [[unroll]] for (uint i = 0u; i < uint(DSTEPS); i++) {
        qr[i] = vec4(0.0);
        o[i] = vec4(0.0);
        if (rowLive && i < dSteps) qr[i] = q4[t * qStride4 + hq * hd4 + j + uint(TPR) * i];
    }
    float runMax = NEG_INF;
    float runSum = 0.0;

    uint kvEnd = pc.seqKv;
    if (pc.maskMode == 0u) {
        uint lastPos = pc.positionOffset + (wgRowEnd - 1u) / G;
        kvEnd = min(pc.seqKv, lastPos + 1u);
    }

    for (uint tStart = 0u; tStart < kvEnd; tStart += uint(BC)) {
        uint tileLen = min(uint(BC), kvEnd - tStart);
        // Stage K and V rows [tStart, tStart + tileLen); rows past tileLen and dims past headDim are never read.
        for (uint idx = gl_LocalInvocationIndex; idx < uint(BC * ROW_WORDS); idx += uint(WG_SIZE)) {
            uint c  = idx / uint(ROW_WORDS);
            uint d4 = idx - c * uint(ROW_WORDS);
            if (c < tileLen && d4 < hd4) {
                uint src = (tStart + c) * kvStride4 + hkv * hd4 + d4;
                kT[idx] = loadK(src);
                vT[idx] = loadV(src);
            }
        }
        barrier();

        // QK^T: the row's 8 lanes each dot 32 dims, then butterfly-sum (uniform control flow for the shuffles).
        float s[BC];
        float tileMax = NEG_INF;
        [[unroll]] for (uint c = 0u; c < uint(BC); c++) {
            float part = 0.0;
            if (c < tileLen) {
                [[unroll]] for (uint i = 0u; i < uint(DSTEPS); i++) {
                    if (i < dSteps) part += dot(qr[i], kT[c * uint(ROW_WORDS) + j + uint(TPR) * i]);
                }
            }
            part += subgroupShuffleXor(part, 1u);
            part += subgroupShuffleXor(part, 2u);
            part += subgroupShuffleXor(part, 4u);
            uint tkv = tStart + c;
            float sc = NEG_INF;
            if (rowLive && c < tileLen) {
                bool masked = isMasked(tkv, posQ)
                    || (pc.slidingWindow > 0u && posQ >= tkv && (posQ - tkv) >= pc.slidingWindow);
                if (!masked) {
                    sc = part * scale;
                    if (pc.softCap > 0.0) sc = pc.softCap * tanh(sc / pc.softCap);
                }
            }
            s[c] = sc;
            tileMax = max(tileMax, sc);
        }

        float newMax = max(runMax, tileMax);
        float corr = (runMax > NEG_INF * 0.5) ? exp(runMax - newMax) : 0.0;
        [[unroll]] for (uint i = 0u; i < uint(DSTEPS); i++) o[i] *= corr;
        float tileSum = 0.0;
        [[unroll]] for (uint c = 0u; c < uint(BC); c++) {
            float w = (s[c] > NEG_INF * 0.5) ? exp(s[c] - newMax) : 0.0;
            tileSum += w;
            if (w != 0.0) {
                [[unroll]] for (uint i = 0u; i < uint(DSTEPS); i++) {
                    if (i < dSteps) o[i] += w * vT[c * uint(ROW_WORDS) + j + uint(TPR) * i];
                }
            }
        }
        runSum = runSum * corr + tileSum;
        runMax = newMax;
        barrier();   // the next tile overwrites kT / vT
    }

    if (rowLive) {
        float inv = (runSum > 1e-10) ? (1.0 / runSum) : 0.0;
        [[unroll]] for (uint i = 0u; i < uint(DSTEPS); i++) {
            if (i < dSteps) out4[t * qStride4 + hq * hd4 + j + uint(TPR) * i] = o[i] * inv;
        }
    }
}
