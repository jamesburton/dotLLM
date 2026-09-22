// Issue #470 — multi-column PQ2_0 GEMV: y[s, m] = W[m, :] . x[s, :] for s < ncols <= NCOLS.
//
// Included by thin per-variant .comp files that #define NCOLS (2, 4, 8). A speculative /
// MTP verify forward runs S = 2..8 tokens; the #446 small-n loop re-dispatched the
// single-column GEMV once per token, so every column re-read the whole weight matrix and a
// verify of S tokens cost ~S x one decode step. This kernel decodes each weight byte ONCE
// and applies it to all NCOLS activation rows, so weight traffic is flat in S.
//
// Numerics contract. The thread partitioning, per-lane accumulation order, and tree reduce
// are IDENTICAL to matmul_pq2_0_f32_gemv.comp, so each column performs the same operations
// as a standalone dispatch of that kernel. The results are not bit-identical: on gfx1151 the
// driver fuses multiply-adds differently in the two pipelines, which moves some outputs by
// 1 ULP. VulkanMatMulPQ2_0GemvF32KernelTests holds the looped kernel as a 1e-5 oracle.
//
// Runtime tail. pc.ncols may be smaller than NCOLS (S = 3 runs the NCOLS = 4 variant). Dead
// columns clamp their x row to the last live one — the loads stay in bounds and the
// arithmetic is simply discarded — and never write y.
//
// Weight layout: see matmul_pq2_0_f32_gemv.comp (34-byte groups: fp16 scale + 32 bytes of
// 2-bit codes, four CONSECUTIVE elements per byte at ascending bit offsets).

#ifndef NCOLS
#error "NCOLS must be defined by the including .comp"
#endif

#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufW { uint  weight[]; };
// x is read as vec4: a code byte covers four consecutive elements, so one 16-byte load feeds
// it. The caller guarantees xOff % 4 == 0 (K is a multiple of 128, so every column stays aligned).
layout(set = 0, binding = 1, std430) readonly  buffer BufX { vec4 x4[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufY { float y[]; };

layout(push_constant) uniform PushConstants {
    uint M;
    uint K;
    uint blocksPerRow;   // = K / 128
    uint rowUints;       // unused; push-constant layout parity with the single-column kernel
    uint xOff;           // first element of column 0's activation row, a multiple of 4; column s is at xOff + s*K
    uint yOff;           // first element of column 0's output row;     column s is at yOff + s*M
    uint ncols;          // live columns, 1..NCOLS
} pc;

const uint PQ2_GROUP_SIZE   = 128u;
const uint PQ2_GROUP_BYTES  = 34u;
const uint PQ2_CODE_BYTES   = 32u;

uint readByte(uint absByteOff) {
    uint u = weight[absByteOff >> 2u];
    uint shift = (absByteOff & 3u) * 8u;
    return (u >> shift) & 0xFFu;
}

float readGroupScale(uint off) {
    uint lo = readByte(off);
    uint hi = readByte(off + 1u);
    uint half16 = lo | (hi << 8u);
    return unpackHalf2x16(half16).x;
}

shared float partials[NCOLS][128];

void main() {
    uint m = gl_WorkGroupID.x;
    if (m >= pc.M) return;

    uint tid = gl_LocalInvocationID.x;
    uint threads = gl_WorkGroupSize.x;

    uint rowBytes = pc.blocksPerRow * PQ2_GROUP_BYTES;
    uint rowByteBase = m * rowBytes;

    // Column s's activation row, in vec4 units.
    uint colX4[NCOLS];
    [[unroll]] for (uint s = 0u; s < NCOLS; s++)
        colX4[s] = (pc.xOff + min(s, pc.ncols - 1u) * pc.K) >> 2u;

    float acc[NCOLS];
    [[unroll]] for (uint s = 0u; s < NCOLS; s++) acc[s] = 0.0;

    uint totalCodeBytes = pc.blocksPerRow * PQ2_CODE_BYTES;
    for (uint cb = tid; cb < totalCodeBytes; cb += threads) {
        uint g  = cb / PQ2_CODE_BYTES;
        uint gp = cb - g * PQ2_CODE_BYTES;

        uint groupByteBase = rowByteBase + g * PQ2_GROUP_BYTES;
        float scale = readGroupScale(groupByteBase);
        uint pk = readByte(groupByteBase + 2u + gp);
        float c0 = float(int( pk        & 3u) - 1);
        float c1 = float(int((pk >> 2u) & 3u) - 1);
        float c2 = float(int((pk >> 4u) & 3u) - 1);
        float c3 = float(int((pk >> 6u) & 3u) - 1);

        // Element g*128 + 4*gp is vec4 index g*32 + gp, which is cb itself.
        [[unroll]] for (uint s = 0u; s < NCOLS; s++) {
            vec4 xv = x4[colX4[s] + cb];
            acc[s] += scale * (c0 * xv.x
                              + c1 * xv.y
                              + c2 * xv.z
                              + c3 * xv.w);
        }
    }

    [[unroll]] for (uint s = 0u; s < NCOLS; s++) partials[s][tid] = acc[s];
    barrier();
    for (uint stride = threads >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            [[unroll]] for (uint s = 0u; s < NCOLS; s++)
                partials[s][tid] = partials[s][tid] + partials[s][tid + stride];
        }
        barrier();
    }

    if (tid < pc.ncols) y[pc.yOff + tid * pc.M + m] = partials[tid][0];
}
