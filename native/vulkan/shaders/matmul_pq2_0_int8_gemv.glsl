// Issue #496 — int8-activation PQ2_0 GEMV: y[s, m] = W[m, :] . x[s, :] for ROWS output rows per
// workgroup and s < ncols <= NCOLS. The Vulkan twin of CUDA #485 (native/kernels/pq2_0_gemv_dp4a.cu),
// which measured +10% plain decode / +25% MTP on Bonsai 2 27B with byte-identical greedy output.
//
// Included by thin per-variant .comp files that #define:
//   ROWS   output rows per workgroup (this family ships 4, mirroring #474)
//   NCOLS  compiled column capacity (1..8); pc.ncols live columns, dead ones clamped
//   WG     workgroup size (64)
//
// ───────────────────────── Why integers ─────────────────────────
// The #474 float kernel (matmul_pq2_0_f32_gemv_multirow.glsl) spends, per code byte per column,
// four `int -> float` converts plus four FMAs. Here the activations are quantized ONCE per
// projection input to int8 (quantize_pq2_0_int8.comp) and each 4 weights x 4 activations become
// one dotPacked4x8AccSatEXT — SPIR-V OpSDotAccSat with PackedVectorFormat4x8Bit, which gfx1151
// advertises as integerDotProduct4x8BitPackedSignedAccelerated. Per 32-element block per column
// the whole cost is 8 dot instructions + one I2F + one FMA, and the weight decode (8 shift/and
// pairs) is shared by every column.
//
// ───────────────────────── The decode trick ─────────────────────────
// A PQ2_0 group is 34 bytes: fp16 scale then 32 code bytes; code byte b packs the codes for the
// four CONSECUTIVE elements 4b..4b+3 at ascending bit offsets {0,2,4,6}; value = code - 1, with
// code in {0,1,2,3} -> {-1,0,+1,+2} (code 3 is +2 — PQ2_0 is not strictly ternary).
// So in a 32-bit code word covering 16 elements, element 4b+i sits at bit 8b+2i and
//     (word >> 2i) & 0x03030303  =  bytes { code(i), code(4+i), code(8+i), code(12+i) }
// — a valid packed signed 4x8 operand, matched against the quantizer's PERMUTED activation word i
// (see quantize_pq2_0_int8.comp for the 4x4 byte transpose). Codes 0..3 are valid signed int8, so
//     sum_e (code_e - 1) * q_e  =  dot(code, q) - sum_e q_e
// is exact in int32, and the "- sum q" is free: the dot chain starts at -xsum (carried in xmeta).
// Decoding to code-1 per byte instead would borrow across bytes on code 0.
// Saturation is unreachable: |sum code*q| <= 3*127*32 = 12192.
//
// ───────────────────────── Lane mapping: block per lane ─────────────────────────
// A LANE owns one whole 32-element activation block, i.e. 8 consecutive code bytes of a group
// (bytes 2 + 8*qb .. +7 for qb = 0..3). This is the CUDA mapping, and unlike the float family it
// is also the right one here: #474 ships LANE_BYTES = 1 for NCOLS >= 2 because the uint mapping
// left adjacent lanes' *float* activation loads 64 bytes apart. With int8 activations a lane's 32
// elements are 32 BYTES, so consecutive lanes read consecutive 32-byte spans — a 64-lane wave
// covers 2 KB contiguous per column, fully coalesced, and 512 contiguous code bytes per row.
// Do not "fix" this back to a byte-per-lane mapping.
//
// ───────────────────────── Numerics ─────────────────────────
// The integer part is exact; the result differs from the float kernel only by the activation
// quantization (the CPU W2A8 tier — see quantize_pq2_0_int8.comp) and FP32 summation order.
// It is therefore NEVER bit-identical to the float kernel, which is the cheap way to prove this
// shader actually ran. Tests hold it to the CPU W2A8 reference and to an argmax-exact +
// RMS-bounded comparison against the float kernel.
//
// Ragged M: rows past M clamp to row M-1 (in-bounds loads, discarded arithmetic) and never write.
// Contract: K % 128 == 0; xq/xmeta are COMPACTED (column s starts at element s*K), so this kernel
// has no xOff — the quantizer consumed it. yOff is still honoured.

#ifndef ROWS
#error "ROWS must be defined by the including .comp"
#endif
#ifndef NCOLS
#error "NCOLS must be defined by the including .comp"
#endif
#ifndef WG
#define WG 64
#endif

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product     : require

layout(local_size_x = WG, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufW     { uint  weight[]; };  // PQ2_0 blob
layout(set = 0, binding = 1, std430) readonly  buffer BufXq    { uint  xq[]; };      // [ncols*K/4] permuted int8
layout(set = 0, binding = 2, std430) readonly  buffer BufXMeta { uvec2 xmeta[]; };   // [ncols*K/32] (bits(d), sum q)
layout(set = 0, binding = 3, std430) writeonly buffer BufY     { float y[]; };       // [ncols*M] from yOff

layout(push_constant) uniform PushConstants {
    uint M;
    uint K;
    uint blocksPerRow;   // = K / 128 (PQ2_0 weight groups per row)
    uint qBlocksPerCol;  // = K / 32  (activation quant blocks per column)
    uint yOff;           // column 0's output row; column s at yOff + s*M
    uint ncols;          // live columns, 1..NCOLS
} pc;

const uint PQ2_GROUP_BYTES = 34u;

// fp16 group scale at even byte offset `off`: 34 is even, so it never straddles a uint.
float readScale(uint off) {
    uint u = weight[off >> 2u];
    return unpackHalf2x16(u >> ((off & 2u) * 8u)).x;
}

// Eight code bytes (two code words, 32 elements) at even byte offset `off`. Either uint-aligned
// (two loads) or half-aligned (three loads, funnel-shifted) — same phase argument as #474's
// readCodes4, and the same one-uint overhang at the very end of the buffer.
uvec2 readCodes8(uint off) {
    uint i = off >> 2u;
    uint a = weight[i];
    uint b = weight[i + 1u];
    if ((off & 2u) == 0u) return uvec2(a, b);
    uint c = weight[i + 2u];
    return uvec2((a >> 16u) | (b << 16u), (b >> 16u) | (c << 16u));
}

shared float partials[ROWS * NCOLS][WG];

void main() {
    uint mBase = gl_WorkGroupID.x * ROWS;
    uint tid = gl_LocalInvocationID.x;

    uint rowBytes = pc.blocksPerRow * PQ2_GROUP_BYTES;

    uint rowBase[ROWS];
    [[unroll]] for (uint r = 0u; r < ROWS; r++)
        rowBase[r] = min(mBase + r, pc.M - 1u) * rowBytes;

    // Column s's compacted activation base, in uint units (K/4 uints per column), and its
    // metadata base in uvec2 units. Dead columns clamp to the last live one.
    uint colXq[NCOLS];
    uint colMeta[NCOLS];
    [[unroll]] for (uint s = 0u; s < NCOLS; s++) {
        uint live = min(s, pc.ncols - 1u);
        colXq[s]   = live * (pc.K >> 2u);
        colMeta[s] = live * pc.qBlocksPerCol;
    }

    float acc[ROWS][NCOLS];
    [[unroll]] for (uint r = 0u; r < ROWS; r++)
        [[unroll]] for (uint s = 0u; s < NCOLS; s++) acc[r][s] = 0.0;

    // One 32-element activation block per lane per step: 4 blocks per 128-element weight group,
    // so a 64-lane wave sweeps 16 consecutive groups.
    for (uint blk = tid; blk < pc.qBlocksPerCol; blk += WG) {
        uint g  = blk >> 2u;          // weight group
        uint qb = blk & 3u;           // 32-element block within it -> code bytes 8*qb .. +7
        uint gOff = g * PQ2_GROUP_BYTES;

        // Decode both/all rows' 32 codes into 8 code-plane words each; reused for every column.
        // wp[r][4h + i] = (codeWord_h >> 2i) & 0x03030303, h = 0 (elements 0..15), 1 (16..31).
        int   wp[ROWS][8];
        float ws[ROWS];
        [[unroll]] for (uint r = 0u; r < ROWS; r++) {
            uint gb = rowBase[r] + gOff;
            ws[r] = readScale(gb);
            uvec2 cw = readCodes8(gb + 2u + 8u * qb);
            [[unroll]] for (uint i = 0u; i < 4u; i++) {
                wp[r][i]      = int((cw.x >> (2u * i)) & 0x03030303u);
                wp[r][4u + i] = int((cw.y >> (2u * i)) & 0x03030303u);
            }
        }

        uint xBase = blk << 3u;   // 8 uints per 32-element block
        [[unroll]] for (uint s = 0u; s < NCOLS; s++) {
            uint xi = colXq[s] + xBase;
            int  xw[8];
            [[unroll]] for (uint j = 0u; j < 8u; j++) xw[j] = int(xq[xi + j]);

            uvec2 md = xmeta[colMeta[s] + blk];
            float d = uintBitsToFloat(md.x);
            int xsum = int(md.y);

            [[unroll]] for (uint r = 0u; r < ROWS; r++) {
                int isum = -xsum;   // sum (code-1)*q = dot(code, q) - sum q
                [[unroll]] for (uint j = 0u; j < 8u; j++)
                    isum = dotPacked4x8AccSatEXT(wp[r][j], xw[j], isum);
                acc[r][s] += float(isum) * (d * ws[r]);
            }
        }
    }

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
