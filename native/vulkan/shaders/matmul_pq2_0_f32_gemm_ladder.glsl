// SHARED TEMPLATE for the issue #439 PQ2_0 coopmat GEMM arithmetic-intensity ladder.
//
// NOT compiled on its own: the `.glsl` extension keeps it out of native/vulkan/build.{sh,ps1}'s
// `*.comp` glob. Three thin wrappers include it, each #defining the tile geometry:
//
//   matmul_pq2_0_f32_gemm_ladder_16x16x1.comp    BM=16  BN=16  NSG=1   4.0 MAC/byte  (control)
//   matmul_pq2_0_f32_gemm_ladder_64x64x4.comp    BM=64  BN=64  NSG=4  16.0 MAC/byte
//   matmul_pq2_0_f32_gemm_ladder_128x128x4.comp  BM=128 BN=128 NSG=4  32.0 MAC/byte
//
// The issue asked for a specialization constant for the workgroup size. There is no
// VkSpecializationInfo plumbing in this tree (VulkanModule.CreateComputePipeline takes no
// specialization data), so the geometry is a compile-time #define instead. For a three-point
// ladder the two are equivalent — one source, three instantiations — and adding specialization
// plumbing would have been unrelated risk inside a measurement change.
//
// ----------------------------------------------------------------------------------
// WHY BK=32 IS NOT ONLY POSSIBLE BUT TRIVIAL — a correction to the shipping shader's header.
//
// matmul_pq2_0_f32_gemm_coopmat32.comp's header claims "BK is one PQ2_0 group (128 elements):
// the byte layout forces the whole group to be staged together (a single packed byte feeds
// positions {gp,gp+32,gp+64,gp+96}, spanning all 128)". That premise is STALE — it describes the
// strided layout an early revision of this shader assumed, and the SAME FILE corrects it 60 lines
// lower ("Ascending bit offsets {0,2,4,6} -> consecutive elements {4*gp,+1,+2,+3} — verified
// against PrismML's reference dequantize_row_q2_0"). .docs/COOPMAT_GEMM_DIAGNOSIS.md and issue
// #439 both inherited the stale premise and proposed bit-slicing (K-slice s = all 32 code bytes
// with bit field 6-2s) to work around it.
//
// No bit-slicing is needed. Byte `gp` of a group holds the four CONSECUTIVE elements
// {4gp, 4gp+1, 4gp+2, 4gp+3}, so a 32-element K-slice is exactly 8 CONTIGUOUS code bytes:
// slice s (s = 0..3) of a group is code bytes [8s, 8s+8). Every byte belongs to exactly one
// slice, nothing is re-read, and the group's single fp16 scale is simply read once per slice
// instead of once per group. BK=32 therefore costs 4x the barriers and 4x the scale reads per
// group and nothing else.
//
// That matters because BK=128 is what caps the tile: 128x128 at BK=128 would need 64 KB of LDS
// against this device's 32 KB limit and the pipeline would not create. At BK=32 a 128x128 tile
// stages 16 KB (20 KB with padding) and reaches llama.cpp's 32.0 MAC/byte.
// ----------------------------------------------------------------------------------
//
// Everything else — the weight layout, the exactness argument for the F16 A operand, the
// ColumnMajor B / ColumnMajor store convention, the 1-ULP coopMatMulAdd drift on gfx1151 — is
// unchanged from matmul_pq2_0_f32_gemm_coopmat32.comp; read that file's header for the reasoning.
//
// Structure follows llama.cpp's ggml-vulkan mul_mm.comp COOPMAT branch (the non-coopmat2 `else`,
// which is the one that applies on gfx1151): NSG subgroups tiled (BM/WM) x (BN/WN) over the
// output tile, one live A fragment and one live B fragment, accumulators held across the whole
// K loop, per-subgroup 16x16 LDS staging for the boundary store.
//
// DELIBERATELY NOT OPTIMIZED beyond the tile change: no double buffering, no global prefetch, no
// vectorized loads. This is a discriminating experiment, and every extra variable would make a
// win unattributable.

#extension GL_KHR_cooperative_matrix      : require
#extension GL_KHR_memory_scope_semantics  : require
#extension GL_KHR_shader_subgroup_basic   : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage    : require
// [[unroll]] on the fragment loops. Required: the coopmat accumulator array is indexed by the
// loop variables and must be register-resident, exactly as mul_mm.comp relies on.
#extension GL_EXT_control_flow_attributes : require

#define BLOCK_SIZE (NSG * WAVE)

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer BufW { uint  weight[]; }; // PQ2_0 blob
layout(set = 0, binding = 1, std430) readonly  buffer BufB { float b[]; };      // [N*K]
layout(set = 0, binding = 2, std430) writeonly buffer BufC { float c[]; };      // [N*M]

layout(push_constant) uniform PushConstants {
    uint M;             // output dim (number of weight rows)
    uint K;             // contraction dim (must be a multiple of 128)
    uint N;             // batch size (number of input rows)
    uint blocksPerRow;  // = K / 128
    uint rowUints;      // unused here; kept for push-constant layout parity
} pc;

const uint TM = 16u;    // coopmat M per fragment
const uint TN = 16u;    // coopmat N per fragment
const uint TK = 16u;    // coopmat K per coopMatMulAdd
const uint WARP = WAVE;

const uint BK = 32u;    // K elements staged per outer iteration
// LDS row padding. llama.cpp pads its coopmat shared stride for the same reason (its
// SHMEM_STRIDE_PAD): an unpadded 32-element f16 row is 64 B, so rows 0 and 2 land on the same
// LDS banks. PAD is IDENTICAL across all three ladder points, so it cannot bias the ladder —
// it only means "(a) vs the shipping kernel" bundles BK=128->32 together with the padding.
const uint PAD    = 8u;
const uint STRIDE = BK + PAD;   // 40 f16 = 80 B per staged row

const uint PQ2_GROUP_BYTES = 34u;   // 2 (fp16 scale) + 32 (packed codes)
const uint PQ2_SLICE_BYTES = 8u;    // code bytes covering ONE BK=32 K-slice of a row
const float16_t FS_ZERO = float16_t(0.0);

const uint SG_ROWS     = BM / WM;   // subgroup grid: SG_ROWS x (NSG/SG_ROWS) == NSG
const uint CMS_PER_ROW = WM / TM;   // 16x16 fragments down one subgroup's tile
const uint CMS_PER_COL = WN / TN;   // 16x16 fragments across it

const uint A_BYTES   = BM * PQ2_SLICE_BYTES;   // code bytes staged per K-chunk
const uint A_PER_THR = A_BYTES / BLOCK_SIZE;   // consecutive code bytes per thread
const uint B_ELEMS   = BN * BK;                // activations staged per K-chunk
const uint B_PER_THR = B_ELEMS / BLOCK_SIZE;



// Shared weight tile: [BM rows][STRIDE] f16, holding ALREADY-SCALED ternary (+/-scale or 0),
// which is exact in F16 — see the shipping shader's header for why.
shared float16_t sharedA[BM * STRIDE];
// Shared input tile: [BN rows][STRIDE] f16.
shared float16_t sharedB[BN * STRIDE];
// Per-subgroup 16x16 staging for the boundary store path.
shared float storeStage[NSG * TM * TN];

uint readByte(uint absByteOff) {
    uint u = weight[absByteOff >> 2u];
    uint shift = (absByteOff & 3u) * 8u;
    return (u >> shift) & 0xFFu;
}

// Little-endian fp16 group scale at absolute byte offset `off`. 34-byte groups straddle uint-word
// boundaries after the first, so this reads byte-wise rather than via a uint load.
float readGroupScale(uint off) {
    uint lo = readByte(off);
    uint hi = readByte(off + 1u);
    return unpackHalf2x16(lo | (hi << 8u)).x;
}

void main() {
    uint mBase = gl_WorkGroupID.x * BM;   // weight rows (C columns)
    uint tBase = gl_WorkGroupID.y * BN;   // input rows (C rows)

    uint tid = gl_LocalInvocationID.x;
    uint rowBytes = pc.blocksPerRow * PQ2_GROUP_BYTES;

    // A_PER_THR consecutive code bytes per thread. A_PER_THR divides PQ2_SLICE_BYTES for every
    // instantiation (4|8 and 8|8), so all of a thread's bytes belong to ONE weight row and the
    // group scale is read once per thread per K-chunk.
    uint aByte0   = tid * A_PER_THR;
    uint rowLocal = aByte0 / PQ2_SLICE_BYTES;   // weight row within the tile
    uint sp0      = aByte0 % PQ2_SLICE_BYTES;   // first code byte within the K-slice
    uint mGlobal  = mBase + rowLocal;
    uint sBase    = rowLocal * STRIDE;
    bool rowValid = mGlobal < pc.M;
    uint rowByteBase = mGlobal * rowBytes;

    // Subgroup grid over the output tile, laid out exactly as mul_mm.comp does.
    //
    // WAVE WIDTH: this ladder runs at the DRIVER'S NATIVE wave64 and does NOT pin the pipeline
    // with VkPipelineShaderStageRequiredSubgroupSizeCreateInfo, unlike the shipping
    // matmul_pq2_0_f32_gemm_coopmat32.comp. That is a measured decision, not an oversight, but
    // the underlying driver behaviour is NOT fully isolated — read the next paragraph as three
    // observations, not as a mechanism.
    //
    // With requiredSubgroupSize=32 + RequireFullSubgroups:
    //   * at NSG=1 (32 threads), merely READING gl_SubgroupID broke the coopmat store — output
    //     rows congruent to {2,3} mod 4 were never written. Confirmed by A/B in both directions
    //     with nothing else changed. Removing the read fixed it completely.
    //   * at NSG=4 (128 threads), the DIRECT store path was correct (the 576x1024x64 parity
    //     shape passes, and it can only pass if warp_c took both values, i.e. if four subgroups
    //     really existed) while the LDS-STAGED boundary path was wrong, with the staging slots
    //     for subgroups 2 and 3 reading back uninitialised memory. Those two facts do not have
    //     one obvious common cause and I did not find it.
    //   * recomputing the index as gl_LocalInvocationID.x/32 instead of gl_SubgroupID made NSG=4
    //     worse, not better, in a way neither story explains.
    // Unpinned wave64 is correct on all 11 parity gates, so that is what the ladder uses.
    //
    // The cost is that the ladder is not directly comparable to the shipping wave32 kernel on
    // wave width — issue #236 measured wave32 at 1.29-1.79x wave64 for the SAME 16x16 tile. That
    // is exactly why point (a) exists: it is a wave64 16x16 single-subgroup control, so the
    // (a) -> (b) -> (c) intensity ladder is internally consistent, and the shipping kernel is
    // reported separately as the absolute bar rather than as the ladder's baseline.
    //
    // NOTE FOR WHOEVER PICKS THIS UP: because the pinned NSG=4 direct path IS correct, and both
    // timed shapes are exact multiples of 128 in M and N, a pinned-wave32 (c) can be TIMED today
    // even though it cannot yet pass the boundary gate. Given #236 that is the obvious next
    // measurement and it is minutes of GPU time, not a driver blocker.
#if NSG == 1
    uint warp_i = 0u;
#else
    uint warp_i = gl_SubgroupID;
#endif
    uint warp_r = warp_i % SG_ROWS;
    uint warp_c = warp_i / SG_ROWS;

    coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[CMS_PER_ROW * CMS_PER_COL];
    [[unroll]] for (uint i = 0u; i < CMS_PER_ROW * CMS_PER_COL; i++)
        sums[i] = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);

#ifdef FAST_UNPACK
    // Hoisted across the chunk loop: the group scale changes only every 4th chunk.
    float16_t fsPos = float16_t(0.0);
    float16_t fsNeg = float16_t(0.0);
#endif

    uint chunks = pc.blocksPerRow * 4u;   // K / BK

    for (uint ch = 0u; ch < chunks; ch++) {
        uint kBase      = ch * BK;
        uint grp        = ch >> 2u;       // PQ2_0 group index
        uint sliceInGrp = ch & 3u;        // 8-byte code slice within that group

        // ---- 1. Stage sharedB[BN, BK] as F16. ----
        // Strided by BLOCK_SIZE so consecutive lanes read consecutive activations (coalesced),
        // unlike the shipping kernel's per-thread-contiguous mapping.
        for (uint i = 0u; i < B_PER_THR; i++) {
            uint idx = i * BLOCK_SIZE + tid;
            uint row = idx / BK;
            uint col = idx % BK;
            uint tGlobal = tBase + row;
            float v = (tGlobal < pc.N) ? b[tGlobal * pc.K + kBase + col] : 0.0;
            sharedB[row * STRIDE + col] = float16_t(v);
        }

        // ---- 2. Unpack sharedA[BM, BK] as SCALED ternary in F16. ----
#ifdef FAST_UNPACK
        // CHEAP-UNPACK ARM (issue #440's counters, not #439's tile hypothesis). RGP on the
        // shipping kernel measured the matrix pipe 99.5% idle with ~95 VALU ops issued per WMMA,
        // and that ratio is set by THIS block, not by the tile. Two changes, both pure ALU:
        //
        //  1. The staged product is exactly {-scale, +/-0, +scale} (the whole reason the F16 A
        //     operand is lossless here), so it is a SELECT, not arithmetic. The default arm pays
        //     int-extract + int->float + f32 multiply + f32->f16 PER ELEMENT; this one hoists
        //     +scale/-scale/0 out of the loop and pays two compares and two selects.
        //  2. The group scale spans 128 elements = FOUR BK=32 chunks, so re-reading it every
        //     chunk (4 byte loads + shifts + unpackHalf2x16) is 4x more often than necessary.
        //     Refresh it only when the group index changes.
        //
        // Numerically identical to the default arm by construction — it selects among values the
        // default arm computes exactly — and held to that by the same one-hot 1-ULP gate.
        if (rowValid) {
            if ((ch & 3u) == 0u) {
                fsPos = float16_t(readGroupScale(rowByteBase + grp * PQ2_GROUP_BYTES));
                fsNeg = -fsPos;
            }
            uint codeBase = rowByteBase + grp * PQ2_GROUP_BYTES + 2u + sliceInGrp * PQ2_SLICE_BYTES;

            for (uint i = 0u; i < A_PER_THR; i++) {
                uint sp = sp0 + i;
                uint pk = readByte(codeBase + sp);
                uint outIdx = sBase + 4u * sp;
                uint c0 =  pk        & 3u;
                uint c1 = (pk >> 2u) & 3u;
                uint c2 = (pk >> 4u) & 3u;
                uint c3 = (pk >> 6u) & 3u;
                sharedA[outIdx]      = (c0 == 1u) ? FS_ZERO : ((c0 == 2u) ? fsPos : fsNeg);
                sharedA[outIdx + 1u] = (c1 == 1u) ? FS_ZERO : ((c1 == 2u) ? fsPos : fsNeg);
                sharedA[outIdx + 2u] = (c2 == 1u) ? FS_ZERO : ((c2 == 2u) ? fsPos : fsNeg);
                sharedA[outIdx + 3u] = (c3 == 1u) ? FS_ZERO : ((c3 == 2u) ? fsPos : fsNeg);
            }
        } else {
#else
        if (rowValid) {
            uint groupByteBase = rowByteBase + grp * PQ2_GROUP_BYTES;
            float scale = readGroupScale(groupByteBase);
            uint codeBase = groupByteBase + 2u + sliceInGrp * PQ2_SLICE_BYTES;

            for (uint i = 0u; i < A_PER_THR; i++) {
                uint sp = sp0 + i;                      // 0..7 within the K-slice
                uint pk = readByte(codeBase + sp);
                uint outIdx = sBase + 4u * sp;          // consecutive elements {4sp..4sp+3}
                sharedA[outIdx]      = float16_t(float(int( pk        & 3u) - 1) * scale);
                sharedA[outIdx + 1u] = float16_t(float(int((pk >> 2u) & 3u) - 1) * scale);
                sharedA[outIdx + 2u] = float16_t(float(int((pk >> 4u) & 3u) - 1) * scale);
                sharedA[outIdx + 3u] = float16_t(float(int((pk >> 6u) & 3u) - 1) * scale);
            }
        } else {
#endif
            for (uint i = 0u; i < A_PER_THR; i++) {
                uint outIdx = sBase + 4u * (sp0 + i);
                sharedA[outIdx]      = float16_t(0.0);
                sharedA[outIdx + 1u] = float16_t(0.0);
                sharedA[outIdx + 2u] = float16_t(0.0);
                sharedA[outIdx + 3u] = float16_t(0.0);
            }
        }

        barrier();
        memoryBarrierShared();

        // ---- 3. BK/TK coopMatMulAdd per fragment. ----
        // Fragment order mirrors mul_mm.comp: hoist the A fragment, sweep B. One live A and one
        // live B fragment at a time keeps operand registers off the accumulators' back.
        [[unroll]] for (uint i = 0u; i < BK; i += TK) {
            [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; cm_row++) {
                coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a;
                coopMatLoad(cache_a, sharedA,
                            (warp_r * WM + cm_row * TM) * STRIDE + i, STRIDE,
                            gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; cm_col++) {
                    coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b;
                    coopMatLoad(cache_b, sharedB,
                                (warp_c * WN + cm_col * TN) * STRIDE + i, STRIDE,
                                gl_CooperativeMatrixLayoutColumnMajor);

                    sums[cm_col * CMS_PER_ROW + cm_row] =
                        coopMatMulAdd(cache_a, cache_b, sums[cm_col * CMS_PER_ROW + cm_row]);
                }
            }
        }

        barrier();
    }

    // ---- 4. Store. C layout: [N, M] row-major; ColumnMajor + stride M places sums[m,t] at
    // base + t*M + m, which is exactly that. No trailing scale: PQ2_0's per-group scales were
    // folded into sharedA at staging time, exactly.
    uint dr = mBase + warp_r * WM;
    uint dc = tBase + warp_c * WN;

    // The fast/slow choice is made on the WHOLE WORKGROUP tile, not per fragment, so the
    // barriers in the staged path sit in workgroup-uniform control flow. Deciding per fragment
    // would let two subgroups disagree, and a workgroup barrier reached by only some subgroups
    // is undefined behaviour.
    //
    // mul_mm.comp uses a subgroup-scope controlBarrier and a per-fragment decision instead; that
    // is correct there and would probably be correct here, but this form needs no reasoning about
    // which predicates are subgroup-uniform, and the boundary path is never on a timed shape
    // (lm_head is 248320x5120 against a 128x128 tile — exact in both dimensions).
    bool tileAllIn = (mBase + BM) <= pc.M && (tBase + BN) <= pc.N;

    if (tileAllIn) {
        [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; cm_row++) {
            [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; cm_col++) {
                coopMatStore(sums[cm_col * CMS_PER_ROW + cm_row], c,
                             (dc + cm_col * TN) * pc.M + dr + cm_row * TM, pc.M,
                             gl_CooperativeMatrixLayoutColumnMajor);
            }
        }
    } else {
        [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; cm_row++) {
            [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; cm_col++) {
                coopMatStore(sums[cm_col * CMS_PER_ROW + cm_row], storeStage, warp_i * TM * TN, TM,
                             gl_CooperativeMatrixLayoutColumnMajor);
                barrier();
                memoryBarrierShared();

                // Scatter EVERY subgroup's staged fragment with the whole workgroup, deriving the
                // owning subgroup from the flat index rather than from a per-lane subgroup
                // builtin. Lane-mapped scatters (mul_mm.comp's store_r/store_c form) miscompiled
                // here: at NSG=4 only the warp_c==0 half of the tile was ever written, on shapes
                // that take this path. Index-derived addressing is uniform, costs nothing on a
                // path that never runs on an aligned shape, and is correct by construction.
                for (uint z = tid; z < NSG * TM * TN; z += BLOCK_SIZE) {
                    uint sg = z / (TM * TN);
                    uint zz = z % (TM * TN);
                    uint mG = mBase + (sg % SG_ROWS) * WM + cm_row * TM + (zz % TM);
                    uint tG = tBase + (sg / SG_ROWS) * WN + cm_col * TN + (zz / TM);
                    if (tG < pc.N && mG < pc.M)
                        c[tG * pc.M + mG] = storeStage[z];
                }
                barrier();
            }
        }
    }
}
