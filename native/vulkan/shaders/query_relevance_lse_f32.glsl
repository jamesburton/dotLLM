// Query-relevance probe, pass 1 (issue #1531 step 4): log-sum-exp of the causal attention scores of one (query row, query head).
//   lse[row * numHeads + head] = log( sum_{j <= qpos[row]} exp( q . k_j * scale ) )
// One workgroup per (row, head). K is read in place from the KV cache (F32, or F16 with KV_F16). Body shared by the F32 / F16 wrappers.
#define WG_SIZE 256
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ    { float q[]; };       // [rows, numHeads * headDim] post-RoPE
#include "kv_load.glsl"                                                              // binding 1 K, binding 2 V (V unused: declared by the include)
layout(set = 0, binding = 3, std430) readonly  buffer BufPos  { int qpos[]; };      // [rows]
layout(set = 0, binding = 4, std430) writeonly buffer BufLse  { float lse[]; };     // [rows * numHeads]

layout(push_constant) uniform PC {
    uint rows;
    uint numHeads;
    uint numKvHeads;
    uint headDim;
    uint nKeys;
    float scale;
} pc;

shared float red[WG_SIZE];

float wgMax(float v) {
    uint t = gl_LocalInvocationID.x;
    red[t] = v; barrier();
    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) { if (t < s) red[t] = max(red[t], red[t + s]); barrier(); }
    float r = red[0]; barrier();
    return r;
}
float wgSum(float v) {
    uint t = gl_LocalInvocationID.x;
    red[t] = v; barrier();
    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) { if (t < s) red[t] += red[t + s]; barrier(); }
    float r = red[0]; barrier();
    return r;
}

void main() {
    uint block = gl_WorkGroupID.x;
    if (block >= pc.rows * pc.numHeads) return;
    uint row = block / pc.numHeads;
    uint head = block - row * pc.numHeads;
    uint hkv = head / (pc.numHeads / pc.numKvHeads);
    uint tid = gl_LocalInvocationID.x;
    uint visible = uint(qpos[row]) + 1u;   // ALL visible keys, including the query rows' own (the softmax normalises over them; nKeys only bounds the scored output)
    uint qBase = row * pc.numHeads * pc.headDim + head * pc.headDim;
    uint kvStride = pc.numKvHeads * pc.headDim;

    float m = -3.4e38;
    for (uint j = tid; j < visible; j += WG_SIZE) {
        float s = 0.0;
        uint kb = j * kvStride + hkv * pc.headDim;
        for (uint d = 0u; d < pc.headDim; d++) s += q[qBase + d] * KV_K(kb + d);
        m = max(m, s * pc.scale);
    }
    float mx = wgMax(m);
    float sum = 0.0;
    for (uint j = tid; j < visible; j += WG_SIZE) {
        float s = 0.0;
        uint kb = j * kvStride + hkv * pc.headDim;
        for (uint d = 0u; d < pc.headDim; d++) s += q[qBase + d] * KV_K(kb + d);
        sum += exp(s * pc.scale - mx);
    }
    float tot = wgSum(sum);
    if (tid == 0u) lse[block] = mx + log(tot);
}
