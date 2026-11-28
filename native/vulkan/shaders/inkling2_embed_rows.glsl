// Inkling 2 image-embedding injection (issue #2062): for every token t of the chunk whose sel[t] is not 0xFFFFFFFF, y[t, :] = rows[sel[t], :] (F16 rows from
// the image encoder, rounded once to the activation type). Runs AFTER inkling2_embed, so it overwrites the (placeholder) token embedding and the image row
// bypasses token_embd_norm (llama.cpp: "do not normalize raw embeddings input"). One 256-thread workgroup per token; no reduction, no F32 storage.
#include "inkling2_common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufRows { uint rows[]; };   // packed F16 pairs, [selected row][H]
layout(set = 0, binding = 1, std430) readonly  buffer BufSel  { uint sel[]; };
layout(set = 0, binding = 2, std430)           buffer BufY    { ACT_T yout[]; };

layout(push_constant) uniform PushConstants {
    uint H;
} pc;

void main() {
    uint t = gl_WorkGroupID.x;
    uint r = sel[t];
    if (r == 0xFFFFFFFFu) return;
    for (uint k = gl_LocalInvocationID.x; k < pc.H; k += 256u) {
        uint idx = r * pc.H + k;
        vec2 h = unpackHalf2x16(rows[idx >> 1u]);
        yout[t * pc.H + k] = fToAct((idx & 1u) == 0u ? h.x : h.y);
    }
}
