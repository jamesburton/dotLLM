// Grouped Q4_1 MoE expert GEMM, v2 (issue #1161, Recursive Flint prefill on Strix). Body shared by two entry points:
//
//   GATEUP (moe_grouped_q4_1_gateup_v2.comp): act[p, m] = silu(X[src(p)] . Wg[e, m]) * (X[src(p)] . Wu[e, m])
//            X = the normalised hidden rows [tokens, K]; src(p) = perm[p] / topK gathers each packed row's token row, so the topK broadcast
//            and the grouped copy of the activations are gone. Gate and up of the same 16 output columns sit in the same wave, so SwiGLU is
//            applied to the accumulators and only the activation is written (packed order, [rows, M]).
//   DOWN   (moe_grouped_q4_1_down_v2.comp):   y[perm[p], m] = (X[p] . Wd[e, m]) * rowScale[p]
//            X = the packed activation rows; the output goes straight to the (token, slot) row the weighted scatter reads, so the ungroup
//            scatter and the separate row-scale pass are gone (the scale undoes the power-of-two downscale that keeps X inside F16).
//
// Work item = (expert, 64-row token tile) from moe_build_tile_list.comp (tileRows = 64) x one weight-row tile (x dimension). Each workgroup
// streams its weight rows ONCE for up to 64 token rows (four 16-row B fragments); v1 tiled tokens by 16, so an expert with 17-32 rows read
// its weights twice. Weight tile: 128 rows (GATEUP: 64 gate + 64 up rows; DOWN: 128 rows), NB Q4_1 blocks per row per staging round.
//
// Numerics match moe_grouped_matmul_q4_1_coopmat_m64.comp: the dequantised weight d * nibble + m and the activation are folded into F16
// operands, accumulation is F32 in the same K order (two TK = 16 steps per block), so each output element is bit-identical to v1's.
// Bank layout: [E, M, K/32 blocks of 20 bytes]: fp16 d, fp16 m, qs[16] (low nibble of byte j = element j, high nibble = element j + 16).

#extension GL_KHR_cooperative_matrix      : require
#extension GL_KHR_memory_scope_semantics  : require
#extension GL_KHR_shader_subgroup_basic   : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage    : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufW0      { uint w0[]; };       // gate (GATEUP) / down (DOWN) bank
layout(set = 0, binding = 1, std430) readonly  buffer BufW1      { uint w1[]; };       // up bank (GATEUP; DOWN binds anything)
layout(set = 0, binding = 2, std430) readonly  buffer BufX       { vec4 x4[]; };       // GATEUP: [tokens, K]; DOWN: [rows, K] packed
layout(set = 0, binding = 3, std430) readonly  buffer BufOffsets { uint offsets[]; };  // [E+1] expert offsets, then the tile list
layout(set = 0, binding = 4, std430) readonly  buffer BufPerm    { uint perm[]; };     // packed row -> (token * topK + slot)
layout(set = 0, binding = 5, std430) writeonly buffer BufY       { float y[]; };       // GATEUP: [rows, M] packed; DOWN: [rows, M] unpermuted
layout(set = 0, binding = 6, std430) readonly  buffer BufScale   { float rowScale[]; }; // DOWN with applyScale: per packed row

layout(push_constant) uniform PushConstants {
    uint M;             // output columns
    uint K;             // contraction length (multiple of 32)
    uint rows;          // packed rows (tokens * topK)
    uint numExperts;
    uint blocksPerRow;  // K / 32
    uint rowBytes;      // weight row stride in bytes
    uint tileList;      // index in `offsets` of the tile work list
    uint topK;          // GATEUP: slots per token
    uint applyScale;    // DOWN: multiply by rowScale[packed row]
} pc;

const uint TM = 16u;
const uint TN = 16u;
const uint TK = 16u;
#ifndef NB
#define NB 2u                           // Q4_1 blocks per weight row per staging round
#endif
const uint BK = 32u * NB;
const uint STRIDE4 = (BK + 8u) / 4u;    // LDS row stride in f16vec4 (BK halves + 8 pad)
const uint BM = 128u;                   // weight rows staged per round
const uint MAX_ROWS = 64u;              // token rows per work item
const uint NFRAG = MAX_ROWS / TN;
const uint WAVE = 64u;
const uint TPB = 256u / (BM * NB);      // threads per staged block (2: one nibble plane each; 1: whole block)
const uint EPT = 32u / TPB;             // elements each thread dequantises per round
const uint BV4 = MAX_ROWS * BK / 4u / 256u;   // activation vec4 per thread per round

#ifdef GATEUP
const uint M_TILE = 64u;                // output columns per workgroup (each staged for gate and up)
#else
const uint M_TILE = 128u;
#endif

shared f16vec4 sA[BM * STRIDE4];
shared f16vec4 sB[MAX_ROWS * STRIDE4];
shared float stage[4u * TM * TN];

struct Stage { uint aw[5]; vec4 b[BV4]; };   // aw[0]: d | m << 16, aw[1..4]: qs bytes 0..15

void main() {
    uint listIdx = pc.tileList + 2u * gl_WorkGroupID.y;
    uint expert = offsets[listIdx];
    uint localRowBase = offsets[listIdx + 1u];
    uint expertStart = offsets[expert];
    uint expertRows = offsets[expert + 1u] - expertStart;
    if (localRowBase >= expertRows) return;
    uint nRows = min(MAX_ROWS, expertRows - localRowBase);
    uint nFrag = (nRows + TN - 1u) / TN;            // workgroup-uniform
    uint packedBase = expertStart + localRowBase;

    uint tid = gl_LocalInvocationID.x;
    uint wave = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint mBase = gl_WorkGroupID.x * M_TILE;

    // A staging: block (aRow, aBlk) of the round; with TPB = 2 the two threads take the low / high nibble planes.
    uint aBlockIdx = tid / TPB;
    uint part = tid % TPB;
    uint aRow = aBlockIdx / NB;
    uint aBlk = aBlockIdx % NB;
#ifdef GATEUP
    bool upRow = aRow >= 64u;
    uint m = mBase + (aRow & 63u);
#else
    uint m = mBase + aRow;
#endif
    bool aValid = m < pc.M;
    uint aWord = (expert * pc.M + m) * (pc.rowBytes >> 2u) + aBlk * 5u;

    // B staging: BV4 vec4 of the 64 x BK activation tile per thread; rows past nRows stage zeros.
    uint k4 = pc.K >> 2u;
    uint bRow[BV4];
    uint xb[BV4];
    bool bOk[BV4];
    [[unroll]] for (uint i = 0u; i < BV4; i++) {
        uint idx = tid + i * 256u;
        bRow[i] = idx / (BK / 4u);
        uint c4 = idx % (BK / 4u);
        bOk[i] = bRow[i] < nRows;
#ifdef GATEUP
        uint src = bOk[i] ? perm[packedBase + bRow[i]] / pc.topK : 0u;
#else
        uint src = packedBase + bRow[i];
#endif
        xb[i] = src * k4 + c4;
    }

    coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> acc0[NFRAG];
    coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> acc1[NFRAG];
    [[unroll]] for (uint f = 0u; f < NFRAG; f++) {
        acc0[f] = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);
        acc1[f] = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);
    }

    uint rounds = pc.blocksPerRow / NB;
    Stage st;
    // Round r's raw words are loaded one round ahead (register prefetch) and converted after round r-1's MMAs.
    for (uint i = 0u; i < 5u; i++) st.aw[i] = 0u;
    if (aValid) {
#ifdef GATEUP
        if (upRow) { for (uint i = 0u; i < 5u; i++) st.aw[i] = w1[aWord + i]; }
        else       { for (uint i = 0u; i < 5u; i++) st.aw[i] = w0[aWord + i]; }
#else
        for (uint i = 0u; i < 5u; i++) st.aw[i] = w0[aWord + i];
#endif
    }
    [[unroll]] for (uint i = 0u; i < BV4; i++) st.b[i] = bOk[i] ? x4[xb[i]] : vec4(0.0);

#ifdef GATEUP
    uint aOff0 = (wave * TM) * STRIDE4;            // gate rows of this wave's 16 columns
    uint aOff1 = (64u + wave * TM) * STRIDE4;      // up rows of the same columns
#else
    uint aOff0 = (wave * 32u) * STRIDE4;
    uint aOff1 = (wave * 32u + TM) * STRIDE4;
#endif

    for (uint r = 0u; r < rounds; r++) {
        // Commit: dequantise this thread's nibbles into f16vec4 and the activations into one f16vec4 each.
        {
            vec2 dm = unpackHalf2x16(st.aw[0]);
            [[unroll]] for (uint plane = 0u; plane < 2u / TPB; plane++) {
                uint shift = (part + plane) * 4u;
                uint aBase = aRow * STRIDE4 + aBlk * 8u + (part + plane) * 4u;
                [[unroll]] for (uint j = 0u; j < 4u; j++) {
                    uint nw = st.aw[j + 1u];
                    vec4 v = vec4(float((nw >> shift) & 0xFu), float((nw >> (8u + shift)) & 0xFu),
                                  float((nw >> (16u + shift)) & 0xFu), float((nw >> (24u + shift)) & 0xFu));
                    sA[aBase + j] = f16vec4(dm.x * v + dm.y);
                }
            }
            [[unroll]] for (uint i = 0u; i < BV4; i++) {
                uint idx = tid + i * 256u;
                sB[bRow[i] * STRIDE4 + idx % (BK / 4u)] = f16vec4(st.b[i]);
            }
        }
        barrier();

        if (r + 1u < rounds) {
            uint wa = aWord + (r + 1u) * NB * 5u;
            if (aValid) {
#ifdef GATEUP
                if (upRow) { for (uint i = 0u; i < 5u; i++) st.aw[i] = w1[wa + i]; }
                else       { for (uint i = 0u; i < 5u; i++) st.aw[i] = w0[wa + i]; }
#else
                for (uint i = 0u; i < 5u; i++) st.aw[i] = w0[wa + i];
#endif
            }
            uint xo = (r + 1u) * (BK / 4u);
            [[unroll]] for (uint i = 0u; i < BV4; i++) st.b[i] = bOk[i] ? x4[xb[i] + xo] : vec4(0.0);
        }

        [[unroll]] for (uint ks = 0u; ks < BK / TK; ks++) {
            coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> a0;
            coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> a1;
            coopMatLoad(a0, sA, aOff0 + ks * (TK / 4u), STRIDE4, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(a1, sA, aOff1 + ks * (TK / 4u), STRIDE4, gl_CooperativeMatrixLayoutRowMajor);
            [[unroll]] for (uint f = 0u; f < NFRAG; f++) {
                if (f < nFrag) {
                    coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> b;
                    coopMatLoad(b, sB, (f * TN) * STRIDE4 + ks * (TK / 4u), STRIDE4, gl_CooperativeMatrixLayoutColumnMajor);
                    acc0[f] = coopMatMulAdd(a0, b, acc0[f]);
                    acc1[f] = coopMatMulAdd(a1, b, acc1[f]);
                }
            }
        }
        barrier();
    }

    // Epilogue: each wave stages one 16 x 16 fragment at a time in its own LDS slice and writes the valid rows.
    uint sOff = wave * TM * TN;
#ifdef GATEUP
    [[unroll]] for (uint f = 0u; f < NFRAG; f++) {
        if (f < nFrag) {
            coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> act;
            for (int i = 0; i < act.length(); i++) {
                float g = acc0[f][i];
                act[i] = g / (1.0 + exp(-g)) * acc1[f][i];
            }
            coopMatStore(act, stage, sOff, TM, gl_CooperativeMatrixLayoutColumnMajor);
            subgroupMemoryBarrierShared();
            subgroupBarrier();
            for (uint i = 0u; i < 4u; i++) {
                uint idx = lane + i * WAVE;
                uint t = idx / TM, mm = idx - t * TM;
                uint r = f * TN + t;
                uint col = mBase + wave * TM + mm;
                if (r < nRows && col < pc.M) y[(packedBase + r) * pc.M + col] = stage[sOff + idx];
            }
            subgroupMemoryBarrierShared();
            subgroupBarrier();
        }
    }
#else
    [[unroll]] for (uint f = 0u; f < NFRAG; f++) {
        if (f < nFrag) {
            [[unroll]] for (uint h = 0u; h < 2u; h++) {
                coopMatStore(h == 0u ? acc0[f] : acc1[f], stage, sOff, TM, gl_CooperativeMatrixLayoutColumnMajor);
                subgroupMemoryBarrierShared();
                subgroupBarrier();
                for (uint i = 0u; i < 4u; i++) {
                    uint idx = lane + i * WAVE;
                    uint t = idx / TM, mm = idx - t * TM;
                    uint r = f * TN + t;
                    uint col = mBase + wave * 32u + h * TM + mm;
                    if (r < nRows && col < pc.M) {
                        uint p = packedBase + r;
                        float v = stage[sOff + idx];
                        if (pc.applyScale != 0u) v *= rowScale[p];
                        y[perm[p] * pc.M + col] = v;
                    }
                }
                subgroupMemoryBarrierShared();
                subgroupBarrier();
            }
        }
    }
#endif
}
