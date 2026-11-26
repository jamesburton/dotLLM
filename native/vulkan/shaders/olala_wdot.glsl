// Row dot products against quantized weights read IN PLACE (never widened): Q8_0, Q6_K or BF16 rows times activation columns.
// Included by the GEMV and MoE kernels. The includer defines WT_Q8_0 | WT_Q6_K | WT_BF16, WG_SIZE (= 64 lanes, one workgroup per output
// row), a specialization constant NCOL (columns per dispatch of this routine), declares
//   readonly buffer { uint weight[]; }   (binding 0)    the raw weight blob
// and a function  float xAt(uint col, uint k)           activation element k of (the already clamped) column `col`.
// Every weight is decoded once per lane and FMA'd against NCOL columns; accumulation is F32 (private array gAcc).

float gAcc[NCOL];

uint readU16(uint idx16) {
    uint w = weight[idx16 >> 1u];
    return ((idx16 & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
}
uint read4(uint idx16) { return readU16(idx16) | (readU16(idx16 + 1u) << 16u); }
vec4 unpackS8(uint u) {
    ivec4 v = ivec4(int(u << 24), int(u << 16), int(u << 8), int(u)) >> 24;
    return vec4(v);
}
int sbyte(uint u16, uint hi) {
    uint b = hi != 0u ? (u16 >> 8u) & 0xFFu : u16 & 0xFFu;
    return int(b ^ 0x80u) - 128;
}

#if defined(WT_Q8_0)
// 32 weights = one 34-byte block = 17 uint16: [0] fp16 scale, [1..16] 32 int8. Four lanes per block, 8 int8 each.
void rowDot(uint row, uint K, uint lane, uint colBase, uint colLimit) {
    uint bpr = K >> 5u;
    uint rowBase16 = row * bpr * 17u;
    uint blockInWindow = lane >> 2u;
    uint sub = lane & 3u;
    for (uint w0 = 0u; w0 < bpr; w0 += WG_SIZE / 4u) {
        uint blk = w0 + blockInWindow;
        if (blk >= bpr) continue;
        uint b16 = rowBase16 + blk * 17u;
        float d = unpackHalf2x16(readU16(b16)).x;
        uint qb = b16 + 1u + 4u * sub;
        vec4 wa = unpackS8(readU16(qb) | (readU16(qb + 1u) << 16u));
        vec4 wb = unpackS8(readU16(qb + 2u) | (readU16(qb + 3u) << 16u));
        uint k0 = blk * 32u + sub * 8u;
        [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) {
            uint col = min(colBase + cc, colLimit);
            float dq = wa.x * xAt(col, k0);
            dq = fma(wa.y, xAt(col, k0 + 1u), dq); dq = fma(wa.z, xAt(col, k0 + 2u), dq); dq = fma(wa.w, xAt(col, k0 + 3u), dq);
            dq = fma(wb.x, xAt(col, k0 + 4u), dq); dq = fma(wb.y, xAt(col, k0 + 5u), dq);
            dq = fma(wb.z, xAt(col, k0 + 6u), dq); dq = fma(wb.w, xAt(col, k0 + 7u), dq);
            gAcc[cc] = fma(d, dq, gAcc[cc]);
        }
    }
}
#elif defined(WT_Q6_K)
// Super-block of 256 = 210 bytes = 105 uint16: ql[128] qh[64] scales[16] int8, fp16 d. Sixteen lanes per super-block, 16 outputs each
// (layout derivation: matmul_q6_k_gemv_f32_coalesced.comp).
void rowDot(uint row, uint K, uint lane, uint colBase, uint colLimit) {
    uint bpr = K >> 8u;
    uint rowBase16 = row * bpr * 105u;
    uint blockInWindow = lane >> 4u;
    uint t = lane & 15u;
    uint hf = t >> 3u;
    uint j = t & 7u;
    uint l0 = 4u * j;
    uint isc = j >> 2u;
    for (uint w0 = 0u; w0 < bpr; w0 += WG_SIZE / 16u) {
        uint blk = w0 + blockInWindow;
        if (blk >= bpr) continue;
        uint b16 = rowBase16 + blk * 105u;
        float d = unpackHalf2x16(readU16(b16 + 104u)).x;
        uint qlA = read4(b16 + (64u * hf + l0) / 2u);
        uint qlB = read4(b16 + (64u * hf + 32u + l0) / 2u);
        uint qh  = read4(b16 + 64u + (32u * hf + l0) / 2u);
        float s0 = float(sbyte(readU16(b16 + 96u + 4u * hf + (isc + 0u) / 2u), (isc + 0u) & 1u));
        float s1 = float(sbyte(readU16(b16 + 96u + 4u * hf + (isc + 2u) / 2u), (isc + 2u) & 1u));
        float s2 = float(sbyte(readU16(b16 + 96u + 4u * hf + (isc + 4u) / 2u), (isc + 4u) & 1u));
        float s3 = float(sbyte(readU16(b16 + 96u + 4u * hf + (isc + 6u) / 2u), (isc + 6u) & 1u));
        uvec4 lA = uvec4(qlA & 0xFFu, (qlA >> 8) & 0xFFu, (qlA >> 16) & 0xFFu, (qlA >> 24) & 0xFFu);
        uvec4 lB = uvec4(qlB & 0xFFu, (qlB >> 8) & 0xFFu, (qlB >> 16) & 0xFFu, (qlB >> 24) & 0xFFu);
        uvec4 hB = uvec4(qh & 0xFFu, (qh >> 8) & 0xFFu, (qh >> 16) & 0xFFu, (qh >> 24) & 0xFFu);
        vec4 q0 = vec4((lA & 0xFu) | (((hB >> 0u) & 3u) << 4u)) - vec4(32.0);
        vec4 q1 = vec4((lB & 0xFu) | (((hB >> 2u) & 3u) << 4u)) - vec4(32.0);
        vec4 q2 = vec4((lA >> 4u)  | (((hB >> 4u) & 3u) << 4u)) - vec4(32.0);
        vec4 q3 = vec4((lB >> 4u)  | (((hB >> 6u) & 3u) << 4u)) - vec4(32.0);
        uint k0 = blk * 256u + hf * 128u + l0;   // x[256 blk + 128 hf + l0 .. +3]; the other groups at +32, +64, +96
        [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) {
            uint col = min(colBase + cc, colLimit);
            vec4 x0 = vec4(xAt(col, k0), xAt(col, k0 + 1u), xAt(col, k0 + 2u), xAt(col, k0 + 3u));
            vec4 x1 = vec4(xAt(col, k0 + 32u), xAt(col, k0 + 33u), xAt(col, k0 + 34u), xAt(col, k0 + 35u));
            vec4 x2 = vec4(xAt(col, k0 + 64u), xAt(col, k0 + 65u), xAt(col, k0 + 66u), xAt(col, k0 + 67u));
            vec4 x3 = vec4(xAt(col, k0 + 96u), xAt(col, k0 + 97u), xAt(col, k0 + 98u), xAt(col, k0 + 99u));
            gAcc[cc] += d * (s0 * dot(q0, x0) + s1 * dot(q1, x1) + s2 * dot(q2, x2) + s3 * dot(q3, x3));
        }
    }
}
#elif defined(WT_BF16)
// Row-major BF16, two elements per uint word (K is even).
void rowDot(uint row, uint K, uint lane, uint colBase, uint colLimit) {
    uint pairs = K >> 1u;
    uint rowWord = row * pairs;
    for (uint p = lane; p < pairs; p += WG_SIZE) {
        uint w = weight[rowWord + p];
        float w0 = bf16Lo(w), w1 = bf16Hi(w);
        [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) {
            uint col = min(colBase + cc, colLimit);
            gAcc[cc] = fma(w0, xAt(col, 2u * p), gAcc[cc]);
            gAcc[cc] = fma(w1, xAt(col, 2u * p + 1u), gAcc[cc]);
        }
    }
}
#else
#error "define WT_Q8_0, WT_Q6_K or WT_BF16"
#endif

// Workgroup sum of one value per lane (subgroup add + shared pass); the result is valid in lane 0 only.
shared float wdotPartials[NCOL][WG_SIZE];
float gTot[NCOL];
void reduceAcc(uint lane) {
    float sums[NCOL];
    [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) sums[cc] = subgroupAdd(gAcc[cc]);
    if (subgroupElect())
        [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) wdotPartials[cc][gl_SubgroupID] = sums[cc];
    barrier();
    [[unroll]] for (uint cc = 0u; cc < NCOL; ++cc) {
        float tot = 0.0;
        if (lane == 0u)
            for (uint i = 0u; i < gl_NumSubgroups; i++) tot += wdotPartials[cc][i];
        gTot[cc] = tot;
    }
    barrier();
}
