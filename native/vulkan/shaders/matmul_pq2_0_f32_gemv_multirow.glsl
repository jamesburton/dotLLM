// Issue #474 — multi-row (and multi-column) PQ2_0 GEMV: y[s, m] = W[m, :] . x[s, :] for
// ROWS output rows per workgroup and s < ncols <= NCOLS.
//
// Included by thin per-variant .comp files that #define:
//   ROWS        output rows per workgroup (1, 2, 4, 8)
//   NCOLS       compiled column capacity (1..8); pc.ncols live columns, dead ones clamped
//   LANE_BYTES  code bytes per lane per step: 1 (the #470 mapping) or 4 (one uint = 16 elements)
//   WG          workgroup size (64 or 128)
//
// Why. matmul_pq2_0_f32_gemv.comp runs one workgroup per row and every workgroup re-reads the
// whole activation vector, so activation load instructions outnumber weight loads ~15:1 per
// column. Here each activation vec4 is loaded once per step and applied to ROWS weight rows;
// LANE_BYTES = 4 additionally makes each weight load instruction fetch 4 code bytes, not 1.
//
// Alignment. Every byte offset in a PQ2_0 buffer is even: rowBytes = blocksPerRow * 34 is even and
// a group is 34 bytes, so a group base is 0 or 2 mod 4. Hence the fp16 scale never straddles a
// uint (one load, shift by 0 or 16), and a 4-byte code chunk at groupBase + 2 + 4q is either
// uint-aligned (one load) or half-aligned (two loads, funnel-shifted). The second load is issued
// only when needed, so the last chunk of the buffer never reads past it.
//
// Ragged M. Rows past M clamp to row M-1 (in-bounds loads, discarded arithmetic) and never write.
//
// Numerics. Each code byte contributes scale * (c0*x0 + c1*x1 + c2*x2 + c3*x3), the same
// expression as the single-row kernel, but the per-lane accumulation order and the reduction tree
// differ for LANE_BYTES = 4, and pipelines contract FMAs differently. Tests hold this to 1e-5 of
// the single-row kernel and to the scalar reference.

#ifndef ROWS
#error "ROWS must be defined by the including .comp"
#endif
#ifndef NCOLS
#error "NCOLS must be defined by the including .comp"
#endif
#ifndef LANE_BYTES
#define LANE_BYTES 1
#endif
#ifndef WG
#define WG 128
#endif

#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = WG, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufW { uint  weight[]; };
// The caller guarantees xOff % 4 == 0 (K is a multiple of 128, so every column stays aligned).
layout(set = 0, binding = 1, std430) readonly  buffer BufX { vec4 x4[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufY { float y[]; };

layout(push_constant) uniform PushConstants {
    uint M;
    uint K;
    uint blocksPerRow;   // = K / 128
    uint rowUints;       // unused; push-constant layout parity
    uint xOff;           // column 0's activation row (elements, multiple of 4); column s at xOff + s*K
    uint yOff;           // column 0's output row (elements);                     column s at yOff + s*M
    uint ncols;          // live columns, 1..NCOLS
} pc;

const uint PQ2_GROUP_BYTES = 34u;
const uint PQ2_CODE_BYTES  = 32u;

// fp16 scale at even byte offset `off`: never straddles a uint.
float readScale(uint off) {
    uint u = weight[off >> 2u];
    return unpackHalf2x16(u >> ((off & 2u) * 8u)).x;
}

// Four code bytes at even byte offset `off`.
uint readCodes4(uint off) {
    uint i = off >> 2u;
    uint lo = weight[i];
    if ((off & 2u) == 0u) return lo;
    return (lo >> 16u) | (weight[i + 1u] << 16u);
}

uint readCode1(uint off) {
    return (weight[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}

float byteDot(uint pk, vec4 xv) {
    float c0 = float(int( pk        & 3u) - 1);
    float c1 = float(int((pk >> 2u) & 3u) - 1);
    float c2 = float(int((pk >> 4u) & 3u) - 1);
    float c3 = float(int((pk >> 6u) & 3u) - 1);
    return c0 * xv.x + c1 * xv.y + c2 * xv.z + c3 * xv.w;
}

shared float partials[ROWS * NCOLS][WG];

void main() {
    uint mBase = gl_WorkGroupID.x * ROWS;
    uint tid = gl_LocalInvocationID.x;

    uint rowBytes = pc.blocksPerRow * PQ2_GROUP_BYTES;

    uint rowBase[ROWS];
    [[unroll]] for (uint r = 0u; r < ROWS; r++)
        rowBase[r] = min(mBase + r, pc.M - 1u) * rowBytes;

    uint colX4[NCOLS];
    [[unroll]] for (uint s = 0u; s < NCOLS; s++)
        colX4[s] = (pc.xOff + min(s, pc.ncols - 1u) * pc.K) >> 2u;

    float acc[ROWS][NCOLS];
    [[unroll]] for (uint r = 0u; r < ROWS; r++)
        [[unroll]] for (uint s = 0u; s < NCOLS; s++) acc[r][s] = 0.0;

#if LANE_BYTES == 1
    // One code byte (4 elements) per lane per step; byte cb is vec4 index cb of the row's x.
    uint total = pc.blocksPerRow * PQ2_CODE_BYTES;
    for (uint cb = tid; cb < total; cb += WG) {
        uint g  = cb / PQ2_CODE_BYTES;
        uint gp = cb - g * PQ2_CODE_BYTES;
        uint gOff = g * PQ2_GROUP_BYTES;

        vec4 xv[NCOLS];
        [[unroll]] for (uint s = 0u; s < NCOLS; s++) xv[s] = x4[colX4[s] + cb];

        [[unroll]] for (uint r = 0u; r < ROWS; r++) {
            uint gb = rowBase[r] + gOff;
            float scale = readScale(gb);
            uint pk = readCode1(gb + 2u + gp);
            [[unroll]] for (uint s = 0u; s < NCOLS; s++)
                acc[r][s] += scale * byteDot(pk, xv[s]);
        }
    }
#elif LANE_BYTES == 4
    // One uint of codes (16 elements) per lane per step: 8 chunks per group; chunk ch covers
    // vec4 indices 4*ch .. 4*ch + 3 of the row's x.
    uint total = pc.blocksPerRow * 8u;
    for (uint ch = tid; ch < total; ch += WG) {
        uint g = ch >> 3u;
        uint q = ch & 7u;
        uint gOff = g * PQ2_GROUP_BYTES;

        float scale[ROWS];
        uint  codes[ROWS];
        [[unroll]] for (uint r = 0u; r < ROWS; r++) {
            uint gb = rowBase[r] + gOff;
            scale[r] = readScale(gb);
            codes[r] = readCodes4(gb + 2u + 4u * q);
        }

        [[unroll]] for (uint b = 0u; b < 4u; b++) {
            vec4 xv[NCOLS];
            [[unroll]] for (uint s = 0u; s < NCOLS; s++) xv[s] = x4[colX4[s] + 4u * ch + b];
            [[unroll]] for (uint r = 0u; r < ROWS; r++) {
                uint pk = (codes[r] >> (8u * b)) & 0xFFu;
                [[unroll]] for (uint s = 0u; s < NCOLS; s++)
                    acc[r][s] += scale[r] * byteDot(pk, xv[s]);
            }
        }
    }
#else
#error "LANE_BYTES must be 1 or 4"
#endif

    [[unroll]] for (uint r = 0u; r < ROWS; r++)
        [[unroll]] for (uint s = 0u; s < NCOLS; s++) partials[r * NCOLS + s][tid] = acc[r][s];
    barrier();
    for (uint stride = WG >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            [[unroll]] for (uint i = 0u; i < ROWS * NCOLS; i++)
                partials[i][tid] = partials[i][tid] + partials[i][tid + stride];
        }
        barrier();
    }

    // Lane i writes output (row i / NCOLS, column i % NCOLS).
    if (tid < ROWS * NCOLS) {
        uint r = tid / NCOLS;
        uint s = tid - r * NCOLS;
        uint m = mBase + r;
        if (m < pc.M && s < pc.ncols) y[pc.yOff + s * pc.M + m] = partials[tid][0];
    }
}
