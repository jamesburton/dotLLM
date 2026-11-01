// Query-relevance probe, pass 2 (issue #1531 step 4): per context key j, add the head-mean attention probability summed over the query rows.
//   acc[j] += (1 / numHeads) * sum_rows sum_heads [ j <= qpos[row] ] * exp( q . k_j * scale - lse[row, head] )
// One thread per key; deterministic (no atomics): every acc[j] has exactly one writer per dispatch. Dispatches of different layers are
// ordered by a barrier, so the layers accumulate in order.
#define WG_SIZE 64
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufQ    { float q[]; };       // [rows, numHeads * headDim] post-RoPE
#include "kv_load.glsl"
layout(set = 0, binding = 3, std430) readonly  buffer BufPos  { int qpos[]; };      // [rows]
layout(set = 0, binding = 4, std430) readonly  buffer BufLse  { float lse[]; };     // [rows * numHeads]
layout(set = 0, binding = 5, std430)           buffer BufAcc  { float acc[]; };     // [nKeys]

layout(push_constant) uniform PC {
    uint rows;
    uint numHeads;
    uint numKvHeads;
    uint headDim;
    uint nKeys;
    float scale;
} pc;

void main() {
    uint j = gl_GlobalInvocationID.x;
    if (j >= pc.nKeys) return;
    uint group = pc.numHeads / pc.numKvHeads;
    uint kvStride = pc.numKvHeads * pc.headDim;
    float total = 0.0;
    for (uint r = 0u; r < pc.rows; r++) {
        if (j > uint(qpos[r])) continue;
        for (uint h = 0u; h < pc.numHeads; h++) {
            uint kb = j * kvStride + (h / group) * pc.headDim;
            uint qb = r * pc.numHeads * pc.headDim + h * pc.headDim;
            float s = 0.0;
            for (uint d = 0u; d < pc.headDim; d++) s += q[qb + d] * KV_K(kb + d);
            total += exp(s * pc.scale - lse[r * pc.numHeads + h]);
        }
    }
    acc[j] += total / float(pc.numHeads);
}
