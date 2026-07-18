using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Core.PositionEncoding;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the cooperative-matrix Flash-Attention prefill
/// kernel (issue #149) against the scalar CPU reference
/// <see cref="Attention.ExecuteScalar"/> — the same oracle the scalar FA
/// kernel (<see cref="VulkanFlashAttentionF32KernelTests"/>) validates against.
/// </summary>
/// <remarks>
/// <para>
/// Tolerance: the kernel rounds Q/K/V to f16 for the two matrix multiplies
/// (F32 accumulators, F32 softmax state) — the same input-rounding class
/// llama.cpp's flash-attention ships (its KV cache is f16, and its FA_COOPMAT1
/// default even uses an f16 P·V accumulator, which we do NOT). Parity vs an
/// all-F32 oracle is therefore epsilon-level, not bit-exact: abs 1e-3 /
/// rel 1e-2 (≈2x the historical coopmat-vs-F32-GPU envelope of 5e-4 / 5e-3,
/// headroom for the CPU-reduction-order delta the F32 kernels also carry).
/// End-to-end greedy-token stability is gated separately via
/// DOTLLM_BENCH_DUMP_TOKENS A/B (see the issue #149 ledger).
/// </para>
/// <para>
/// Shapes follow the repo discriminating-shape rule: GQA groups where
/// <c>hq/groupSize != hq%groupSize</c>, head_dim that is NOT a multiple of the
/// 16-wide coopmat chunk (80), partial Q/KV tiles, non-zero positionOffset
/// (chunked prefill), and every mask mode the dispatcher can route here.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanFlashAttentionCoopmatKernelTests
{
    private const float AbsTol = 1e-3f;
    private const float RelTol = 1e-2f;

    [SkippableFact]
    public void Launch_Mha_ShortPrefill()
        // Partial Q-tile (4 < BR=16) + partial KV tile (4 < BC=64).
        => RunOne(seqQ: 4, seqKv: 4, numHeads: 1, numKvHeads: 1, headDim: 64, positionOffset: 0);

    [SkippableFact]
    public void Launch_Gqa3_SmolLm_Prefill64()
        // SmolLM shape: 9 heads / 3 kv heads (groupSize 3 — asymmetric GQA
        // broadcast, discriminates hq/group vs hq%group).
        => RunOne(seqQ: 64, seqKv: 64, numHeads: 9, numKvHeads: 3, headDim: 64, positionOffset: 0);

    [SkippableFact]
    public void Launch_Mha_HeadDim128()
        // Full MAX_HEAD_DIM: exercises both P·V rounds.
        => RunOne(seqQ: 32, seqKv: 32, numHeads: 8, numKvHeads: 8, headDim: 128, positionOffset: 0);

    [SkippableFact]
    public void Launch_HeadDim80_NonChunkMultiple()
        // head_dim 80 = 5 x 16 chunks, hdCeil == 80 < 128: exercises the
        // padded-chunk loads, the skipped slices in P·V round 1
        // (dBlock 80/96/112 >= hdCeil) and the d0 >= headDim write guard.
        => RunOne(seqQ: 48, seqKv: 48, numHeads: 4, numKvHeads: 2, headDim: 80, positionOffset: 0);

    [SkippableFact]
    public void Launch_HeadDim72_PaddedTailChunk()
        // head_dim 72: hdCeil = 80 > headDim — the final 16-wide chunk is
        // half-padded, discriminating the zero-fill inside a chunk from the
        // whole-chunk skip that headDim 80 exercises.
        => RunOne(seqQ: 33, seqKv: 40, numHeads: 4, numKvHeads: 2, headDim: 72, positionOffset: 0);

    [SkippableFact]
    public void Launch_Gqa8_Prefill_512()
        // Llama-3-ish: 32 heads / 4 kv heads, 512x512 — multi-KV-tile outer
        // loop + causal early-exit across many Q tiles.
        => RunOne(seqQ: 512, seqKv: 512, numHeads: 32, numKvHeads: 4, headDim: 64, positionOffset: 0);

    [SkippableFact]
    public void Launch_Gqa8_Prefill_2048()
        // Long-context prefill: 32 KV tiles per Q-tile, 128 Q-tiles per head.
        => RunOne(seqQ: 2048, seqKv: 2048, numHeads: 8, numKvHeads: 2, headDim: 64, positionOffset: 0);

    // Issue #378: headDim<=64 + seqKv>=SeqKvThreshold(640) routes to the
    // LDS-halved hd64 shader — none of the shapes above cross that seqKv
    // threshold (max is 512), so they all still exercise the base 128-dim
    // shader post-#378. These specifically exercise the hd64 dispatch path.
    [SkippableFact]
    public void Launch_Hd64_Gqa3_SmolLm_LongPrefill()
        // SmolLM shape (9 heads / 3 kv heads) at seqKv just past the
        // hd64 dispatch threshold.
        => RunOne(seqQ: 640, seqKv: 640, numHeads: 9, numKvHeads: 3, headDim: 64, positionOffset: 0);

    [SkippableFact]
    public void Launch_Hd64_PartialTiles_LongPrefill()
        // Non-tile-multiple seqQ/seqKv at hd64-eligible length — validates
        // the zero-padded partial-tile paths under the smaller MAX_HEAD_DIM.
        => RunOne(seqQ: 777, seqKv: 809, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 0);

    [SkippableFact]
    public void Launch_Hd64_ChunkedPrefill_PositionOffset()
        => RunOne(seqQ: 128, seqKv: 768, numHeads: 8, numKvHeads: 2, headDim: 64, positionOffset: 640);

    [SkippableFact]
    public void Launch_Hd64_SlidingWindow()
        => RunOne(seqQ: 96, seqKv: 700, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, slidingWindow: 100);

    [SkippableFact]
    public void Launch_Hd64_Alibi()
        => RunOne(seqQ: 64, seqKv: 704, numHeads: 6, numKvHeads: 2, headDim: 64,
            positionOffset: 0, useAlibi: true);

    [SkippableFact]
    public void Launch_Hd64_HeadDim32_SmallerThanTile()
        // headDim (32) strictly less than the hd64 shader's own MAX_HEAD_DIM
        // (64) — exercises the padded-chunk / skipped-slice logic inside the
        // smaller tile, mirroring what Launch_HeadDim80_NonChunkMultiple does
        // for the 128-dim shader.
        => RunOne(seqQ: 64, seqKv: 700, numHeads: 4, numKvHeads: 2, headDim: 32, positionOffset: 0);

    [SkippableFact]
    public void Launch_Hd64_JustBelowThreshold_UsesBaseShader()
        // seqKv = SeqKvThreshold - 1 must NOT dispatch hd64 — this shape is
        // a regression guard for the threshold boundary itself (asserts
        // correctness of whichever shader actually gets selected, not which
        // one that is).
        => RunOne(seqQ: 64, seqKv: VulkanFlashAttentionCoopmatKernel.SeqKvThreshold - 1,
            numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 0);

    [SkippableFact]
    public void Launch_ChunkedPrefill_PositionOffset()
        // Second chunk of a chunked prefill: 64 new queries against 192 total
        // KV rows with positionOffset 128 — the causal frontier sits mid-KV.
        => RunOne(seqQ: 64, seqKv: 192, numHeads: 8, numKvHeads: 2, headDim: 64, positionOffset: 128);

    [SkippableFact]
    public void Launch_SlidingWindow_4()
        => RunOne(seqQ: 16, seqKv: 32, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, slidingWindow: 4);

    [SkippableFact]
    public void Launch_SlidingWindow_CrossTile()
        // Window 100 with 256 KV rows: the window boundary crosses BC=64 tile
        // boundaries at different columns per Q row.
        => RunOne(seqQ: 128, seqKv: 256, numHeads: 4, numKvHeads: 2, headDim: 128,
            positionOffset: 128, slidingWindow: 100);

    [SkippableFact]
    public void Launch_SoftCap_50()
        => RunOne(seqQ: 32, seqKv: 32, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, softCap: 50.0f);

    [SkippableFact]
    public void Launch_ScaleOverride_Qpas()
        // Gemma-3 QPAS-style custom scale (1/sqrt(256) instead of 1/sqrt(64)).
        => RunOne(seqQ: 32, seqKv: 32, numHeads: 8, numKvHeads: 2, headDim: 64,
            positionOffset: 0, scaleOverride: 0.0625f);

    [SkippableFact]
    public void Launch_Alibi_Mha()
        // 6 heads: non-power-of-two ALiBi slope table.
        => RunOne(seqQ: 16, seqKv: 16, numHeads: 6, numKvHeads: 2, headDim: 64,
            positionOffset: 0, useAlibi: true);

    [SkippableFact]
    public void Launch_PartialKvTile()
        // seqKv = 33: final KV tile is partial (33 mod 64) — validates the
        // tileLen clamp + zero-padded K/V columns + P zero-fill past tileLen.
        => RunOne(seqQ: 16, seqKv: 33, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 0);

    [SkippableFact]
    public void Launch_PartialQTile()
        // seqQ = 33: final Q-tile has 1 valid row — validates zero-padded Q
        // rows and the P padding rows the softmax never writes.
        => RunOne(seqQ: 33, seqKv: 64, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 0);

    [SkippableFact]
    public void Launch_Bidirectional_Prefill()
        // Early rows attend to future keys — discriminates maskMode handling.
        => RunOne(seqQ: 16, seqKv: 16, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, maskMode: AttentionMaskMode.Bidirectional);

    [SkippableFact]
    public void Launch_Bidirectional_MultiTile()
        // Bidirectional must NOT take the causal kvEnd early-exit: 96 KV rows
        // for 40 queries, every query sees all rows.
        => RunOne(seqQ: 40, seqKv: 96, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, maskMode: AttentionMaskMode.Bidirectional);

    [SkippableFact]
    public void Launch_Hybrid_PrefixCausal_CanvasBidirectional()
        => RunOne(seqQ: 48, seqKv: 48, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, maskMode: AttentionMaskMode.Hybrid, prefixLen: 20);

    // Issue #382: 2-row-block doubled-query-tile variant (corrects #381's
    // invalid BR=32 coopmat approach — every coopmat op here stays at the
    // native M=16, looped over 2 row blocks). Opt-in via ForceRb2ForTests so
    // these don't need the process-wide DOTLLM_VULKAN_FA_COOPMAT_2RB env var.
    // Same shape coverage as #381's (reverted) tests: GQA groups, partial
    // tiles crossing the row-block boundary, chunked-prefill position
    // offset, sliding window, ALiBi, headDim smaller than MAX_HEAD_DIM, a
    // single-row-block case (seqQ < 16), and the canonical p=512 shape.
    [SkippableFact]
    public void Launch_Rb2_Gqa3_SmolLm_P512()
        // The canonical SmolLM-135M perf-matrix shape this issue targets.
        => RunOne(seqQ: 512, seqKv: 512, numHeads: 9, numKvHeads: 3, headDim: 64,
            positionOffset: 0, forceRb2: true);

    [SkippableFact]
    public void Launch_Rb2_PartialTiles()
        // Non-tile-multiple seqQ/seqKv: exercises rowsInTile values that
        // land in EITHER row block (777 % 32 = 9, so the final Q-tile has
        // only 9 valid rows, entirely inside row block 0 -- see
        // Launch_Rb2_PartialTile_SecondRowBlock for a row-block-1 case).
        => RunOne(seqQ: 777, seqKv: 809, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, forceRb2: true);

    [SkippableFact]
    public void Launch_Rb2_PartialTile_SecondRowBlock()
        // seqQ = 40: final Q-tile has 8 valid rows (40 - 32), which live
        // ENTIRELY in row block 1 (rows 16-31 of the tile, of which only
        // 16-23 are valid) -- discriminates the rb=1 zero-padding/masking
        // path specifically, which Launch_Rb2_PartialTiles' 777 case does
        // not reach (its remainder sits in row block 0).
        => RunOne(seqQ: 40, seqKv: 64, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, forceRb2: true);

    [SkippableFact]
    public void Launch_Rb2_ChunkedPrefill_PositionOffset()
        => RunOne(seqQ: 128, seqKv: 768, numHeads: 8, numKvHeads: 2, headDim: 64,
            positionOffset: 640, forceRb2: true);

    [SkippableFact]
    public void Launch_Rb2_SlidingWindow()
        => RunOne(seqQ: 96, seqKv: 700, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, slidingWindow: 100, forceRb2: true);

    [SkippableFact]
    public void Launch_Rb2_Alibi()
        => RunOne(seqQ: 64, seqKv: 704, numHeads: 6, numKvHeads: 2, headDim: 64,
            positionOffset: 0, useAlibi: true, forceRb2: true);

    [SkippableFact]
    public void Launch_Rb2_HeadDim32_SmallerThanTile()
        => RunOne(seqQ: 64, seqKv: 700, numHeads: 4, numKvHeads: 2, headDim: 32,
            positionOffset: 0, forceRb2: true);

    [SkippableFact]
    public void Launch_Rb2_SingleRowBlock()
        // seqQ < ROW_BLOCK (16): every row lives in row block 0; row block 1
        // is entirely padding. Discriminates row-block-1-fully-inactive.
        => RunOne(seqQ: 11, seqKv: 64, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, forceRb2: true);

    // ─────────────────────────────────────────────────────────────

    private static void RunOne(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow = 0, float softCap = 0.0f, bool useAlibi = false,
        float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0,
        bool forceRb2 = false)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(
            VulkanFlashAttentionCoopmatKernel.SupportsDevice(device),
            "Device does not expose a subgroup-scope 16x16x16 F16xF16->F32 cooperative-matrix tile.");

        var rng = new Random(0xC0091 + seqQ * 41 + seqKv * 17 + numHeads * 7 + headDim);
        float[] qh = RandomFloats(rng, seqQ * numHeads * headDim);
        float[] kh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] vh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] expected = new float[seqQ * numHeads * headDim];

        ComputeExpected(qh, kh, vh, expected,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            slidingWindow, softCap, useAlibi, scaleOverride, maskMode, prefixLen);

        using var kernel = VulkanFlashAttentionCoopmatKernel.Create(device, spvDir);
        if (forceRb2)
        {
            Skip.If(!File.Exists(Path.Combine(spvDir, "attention_flash_f32_coopmat_hd64_2rb.spv")),
                "attention_flash_f32_coopmat_hd64_2rb.spv not built.");
            kernel.ForceRb2ForTests = true;
        }

        using var bufQ   = device.Allocate((long)qh.Length * sizeof(float));
        using var bufK   = device.Allocate((long)kh.Length * sizeof(float));
        using var bufV   = device.Allocate((long)vh.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(qh.AsSpan(), bufQ);
        device.Upload(kh.AsSpan(), bufK);
        device.Upload(vh.AsSpan(), bufV);

        kernel.Launch(bufQ, bufK, bufV, bufOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim,
            positionOffset: positionOffset, slidingWindow: slidingWindow,
            useAlibi: useAlibi, softCap: softCap, scaleOverride: scaleOverride,
            maskMode: maskMode, prefixLen: prefixLen);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        AssertClose(expected, actual, seqQ, seqKv, numHeads, numKvHeads, headDim);
    }

    /// <summary>
    /// CPU reference — <see cref="Attention.ExecuteScalar"/> (which covers
    /// softCap and maskMode natively in the current signature) with the
    /// scaleOverride substituted for the default 1/sqrt(headDim) when set.
    /// </summary>
    private static void ComputeExpected(
        float[] q, float[] k, float[] v, float[] output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow, float softCap, bool useAlibi,
        float scaleOverride, AttentionMaskMode maskMode, int prefixLen)
    {
        int? swArg = slidingWindow > 0 ? slidingWindow : null;
        float scale = scaleOverride > 0.0f ? scaleOverride : 1.0f / MathF.Sqrt(headDim);
        ReadOnlySpan<float> slopes = useAlibi
            ? AlibiPositionEncoding.CreateSlopes(numHeads)
            : default;
        Attention.ExecuteScalar(q, k, v, output,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            scale, slopes, swArg, softCap, maskMode, prefixLen);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"Coopmat FlashAttention drift exceeded tolerance " +
            $"(seqQ={seqQ},seqKv={seqKv},nh={numHeads},nkv={numKvHeads},hd={headDim}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
