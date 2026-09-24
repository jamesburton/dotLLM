// SHARED BLOCKED COOPMAT GEMM TEMPLATE — main() half. Issue #443.
// Read gemm_coopmat_blocked_decl.glsl's header for the contract and the mechanism.
//
// Structure follows llama.cpp's ggml-vulkan mul_mm.comp COOPMAT branch (the non-coopmat2
// `else`, which is the one that applies on gfx1151): NSG subgroups tiled
// (BM/WM) x (BN/WN) over the output tile, one live A fragment and one live B fragment,
// accumulators held across the whole K loop, per-subgroup 16x16 LDS staging for the
// boundary store.
//
// ----------------------------------------------------------------------------------
// WAVE WIDTH IS A CORRECTNESS PRECONDITION, NOT A TUNING KNOB. READ THIS BEFORE SELECTING.
//
// The workgroup is a FIXED `NSG * WAVE` threads and the subgroup grid is derived from
// `gl_SubgroupID`, so `WAVE` must equal the device's NATIVE subgroup width. At NSG=4, WAVE=64
// that is 256 threads forming four subgroups in a 2x2 grid. On a 32-wide device the SAME 256
// threads form EIGHT subgroups: ids 4-7 compute warp_c = 2,3, which
//   * `coopMatLoad` from `sharedB` past `BN * STRIDE` — out-of-bounds LDS, and
//   * `coopMatStore` into `g_tBase + 128 ..` — the NEXT workgroup's output rows,
// while `tileAllIn` still reports the direct path safe because it is evaluated on the workgroup
// tile. The result is silent wrong answers, not a pipeline-creation failure, so nothing catches
// it at load time.
//
// Every C# variant that names one of these shaders therefore carries
// `RequiresNativeSubgroupSize = 64` (or, for I2SGemmVariant, an explicit `device.SubgroupSize ==
// 64` in SelectFor). The principled fix is a wave-count specialization constant, or pinning the
// pipeline to 64 — neither is what #443 measured, so the gate is the conservative form.
//
// NOTE FOR WHOEVER TOUCHES PQ2_0: matmul_pq2_0_f32_gemm_ladder.glsl has the SAME latent
// assumption and PQ2_0GemmVariant.Ladder128x128x4 does NOT gate on it. Bonsai is AMD-only in
// practice so nothing is broken today, but the gate belongs there too.
// ----------------------------------------------------------------------------------
//
// These instantiations run at the driver's NATIVE wave64 and do NOT pin the
// pipeline with VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. That is measured, not
// an oversight: #443 timed a wave32-pinned 128x128x4 PQ2_0 tile at 1.01x / 1.00x — the pin's
// 1.29-1.79x on the OLD 16x16 kernel (#236) was lane utilisation, and a wave64 subgroup that
// owns 16 fragments does not waste half its lanes the way one owning a single fragment did.
// The pin also has a known-broken LDS-staged boundary path at NSG=4 whose cause was never
// isolated. Unpinned wave64 passes every gate; do not spend time re-testing the pin.

void main() {
    g_mBase = gl_WorkGroupID.x * BM;   // weight rows (C columns)
    g_tBase = gl_WorkGroupID.y * BN;   // LOCAL token rows (C rows)
    g_tid   = gl_LocalInvocationID.x;

#ifdef GEMM_HAS_PROLOGUE
    // Workgroup-uniform early-out (MoE: expert bounds). Must be uniform across the whole
    // workgroup, because every barrier below sits in control flow this predicate dominates.
    if (!gemmPrologue()) return;
#endif

    uint warp_i = gl_SubgroupID;
    uint warp_r = warp_i % SG_ROWS;
    uint warp_c = warp_i / SG_ROWS;

    coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[CMS_PER_ROW * CMS_PER_COL];
    [[unroll]] for (uint i = 0u; i < CMS_PER_ROW * CMS_PER_COL; i++)
        sums[i] = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);

    uint chunks = GEMM_CHUNKS;

    for (uint ch = 0u; ch < chunks; ch++) {
        uint kBase = ch * BK;

        // ---- 1. Stage sharedB[BN, BK] as F16. ----
        // Strided by BLOCK_SIZE so consecutive lanes read consecutive activations (coalesced).
        for (uint i = 0u; i < B_PER_THR; i++) {
            uint idx = i * BLOCK_SIZE + g_tid;
            uint row = idx / BK;
            uint col = idx % BK;
            uint tLocal = g_tBase + row;
            float v = (tLocal < uint(GEMM_ROW_LIMIT))
                ? GEMM_B_BUF[(GEMM_ROW_BASE + tLocal) * GEMM_K + kBase + col]
                : 0.0;
            sharedB[row * STRIDE + col] = float16_t(v);
        }

        // ---- 2. Stage sharedA[BM, BK] — the one per-quant hook. ----
        gemmStageA(ch);

        barrier();
        memoryBarrierShared();

        // ---- 3. BK/TK coopMatMulAdd per fragment. ----
        // Fragment order mirrors mul_mm.comp: hoist the A fragment, sweep B. One live A and
        // one live B fragment at a time keeps operand registers off the accumulators' back.
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

#ifdef GEMM_EPILOGUE_SCALE
    // A single per-tensor scale applied to the F32 accumulator AFTER the K loop — which is
    // why such formats can stage the A operand losslessly (I2_S: raw ternary {-1,0,+1}).
    {
        float epiScale = GEMM_EPILOGUE_SCALE;
        [[unroll]] for (uint i = 0u; i < CMS_PER_ROW * CMS_PER_COL; i++)
            sums[i] = sums[i] * epiScale;
    }
#endif

    // ---- 4. Store. C layout: [rows, M] row-major; ColumnMajor + stride M places sums[m,t]
    // at base + t*M + m, which is exactly that.
    uint dr = g_mBase + warp_r * WM;             // first weight row this subgroup owns
    uint dc = g_tBase + warp_c * WN;             // first LOCAL token row this subgroup owns

    // The fast/slow choice is made on the WHOLE WORKGROUP tile, not per fragment, so the
    // barriers in the staged path sit in workgroup-uniform control flow. Deciding per
    // fragment would let two subgroups disagree, and a workgroup barrier reached by only
    // some subgroups is undefined behaviour.
    bool tileAllIn = (g_mBase + BM) <= uint(GEMM_M) && (g_tBase + BN) <= uint(GEMM_ROW_LIMIT);

    if (tileAllIn) {
        [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; cm_row++) {
            [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; cm_col++) {
                coopMatStore(sums[cm_col * CMS_PER_ROW + cm_row], GEMM_C_BUF,
                             (GEMM_ROW_BASE + dc + cm_col * TN) * GEMM_M + dr + cm_row * TM,
                             GEMM_M, gl_CooperativeMatrixLayoutColumnMajor);
            }
        }
    } else {
        [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; cm_row++) {
            [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; cm_col++) {
                coopMatStore(sums[cm_col * CMS_PER_ROW + cm_row], storeStage,
                             warp_i * TM * TN, TM, gl_CooperativeMatrixLayoutColumnMajor);
                barrier();
                memoryBarrierShared();

                // Scatter EVERY subgroup's staged fragment with the whole workgroup, deriving
                // the owning subgroup from the flat index rather than from a per-lane subgroup
                // builtin. Lane-mapped scatters (mul_mm.comp's store_r/store_c form)
                // miscompiled here: at NSG=4 only the warp_c==0 half of the tile was ever
                // written, on shapes that take this path. Index-derived addressing is uniform,
                // costs nothing on a path that never runs on an aligned shape, and is correct
                // by construction.
                for (uint z = g_tid; z < NSG * TM * TN; z += BLOCK_SIZE) {
                    uint sg = z / (TM * TN);
                    uint zz = z % (TM * TN);
                    uint mG = g_mBase + (sg % SG_ROWS) * WM + cm_row * TM + (zz % TM);
                    uint tL = g_tBase + (sg / SG_ROWS) * WN + cm_col * TN + (zz / TM);
                    if (tL < uint(GEMM_ROW_LIMIT) && mG < uint(GEMM_M))
                        GEMM_C_BUF[(GEMM_ROW_BASE + tL) * GEMM_M + mG] = storeStage[z];
                }
                barrier();
            }
        }
    }
}
