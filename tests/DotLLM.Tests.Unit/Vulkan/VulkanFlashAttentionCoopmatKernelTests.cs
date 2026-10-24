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
/// default even uses an f16 P×V accumulator, which we do NOT). Parity vs an
/// all-F32 oracle is therefore epsilon-level, not bit-exact: abs 1e-3 /
/// rel 1e-2 (~2x the historical coopmat-vs-F32-GPU envelope of 5e-4 / 5e-3,
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
/// <para>
/// Issue #240: the <c>Launch_Pinned64_*</c> tests below cover the SAME kernel
/// created via <see cref="VulkanFlashAttentionCoopmatKernel.Create(VulkanDevice, string, FlashAttentionCoopmatVariant)"/>
/// with <see cref="FlashAttentionCoopmatVariant.Pinned64"/> — an explicit
/// <c>requiredSubgroupSize=64</c> pin on the SAME SPIR-V the unpinned tests
/// above already validate (see that overload's remarks: no new shader, only
/// a different pipeline-creation-time subgroup constraint). This is a
/// representative discriminating subset of the shapes above (short/partial
/// tiles, asymmetric GQA, full head_dim 128, padded-tail head_dim 72, the
/// hd64 dispatch path, a large multi-tile GQA8 512 shape, sliding-window
/// cross-tile, and bidirectional multi-tile masking) chosen to cover both
/// the base and hd64 pipelines and every masking mode without doubling the
/// full suite — sufficient to catch a subgroup-pin-induced correctness
/// regression (e.g. a full-subgroups assumption the shader's redundant-slice
/// design silently violates) without exhaustively re-running every shape.
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
        // Full MAX_HEAD_DIM: exercises both P×V rounds.
        => RunOne(seqQ: 32, seqKv: 32, numHeads: 8, numKvHeads: 8, headDim: 128, positionOffset: 0);

    [SkippableFact]
    public void Launch_HeadDim80_NonChunkMultiple()
        // head_dim 80 = 5 x 16 chunks, hdCeil == 80 < 128: exercises the
        // padded-chunk loads, the skipped slices in P×V round 1
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

    // Issue #685: head_dim > 128 routes to the hd256 pipeline (K/V staged in 64-wide d-chunks, 16 register O cells per thread).
    [SkippableFact]
    public void Launch_Hd256_Mha_ShortPrefill()
        => RunOne(seqQ: 4, seqKv: 4, numHeads: 1, numKvHeads: 1, headDim: 256, positionOffset: 0);

    [SkippableFact]
    public void Launch_Hd256_Gqa8_Prefill_512()
        // Qwen3.6-35B-A3B attention shape: 16 heads / 2 kv heads (group 8, discriminates hq/group vs hq%group).
        => RunOne(seqQ: 512, seqKv: 512, numHeads: 16, numKvHeads: 2, headDim: 256, positionOffset: 0);

    [SkippableFact]
    public void Launch_Hd256_PartialTiles()
        // Ragged Q and KV tiles (rowsInTile < BR, tileLen < BC) across several KV tiles.
        => RunOne(seqQ: 301, seqKv: 333, numHeads: 4, numKvHeads: 2, headDim: 256, positionOffset: 0);

    [SkippableFact]
    public void Launch_Hd256_ChunkedPrefill_PositionOffset()
        => RunOne(seqQ: 100, seqKv: 612, numHeads: 8, numKvHeads: 2, headDim: 256, positionOffset: 512);

    [SkippableFact]
    public void Launch_Hd256_LongPrefill_2048()
        => RunOne(seqQ: 2048, seqKv: 2048, numHeads: 4, numKvHeads: 2, headDim: 256, positionOffset: 0);

    [SkippableFact]
    public void Launch_Hd256_SlidingWindow()
        => RunOne(seqQ: 96, seqKv: 700, numHeads: 4, numKvHeads: 2, headDim: 256, positionOffset: 0, slidingWindow: 100);

    [SkippableFact]
    public void Launch_Hd192_PaddedDChunks()
        // head_dim 192 = three full 64-wide d-chunks of a 256 tile, whole pvRounds 3 of 4 skipped.
        => RunOne(seqQ: 70, seqKv: 150, numHeads: 4, numKvHeads: 2, headDim: 192, positionOffset: 0);

    [SkippableFact]
    public void Launch_Hd144_HalfPaddedLastDChunk()
        // head_dim 144: the last d-chunk is only 16 wide, the P.V round 2 owns one live 16-block.
        => RunOne(seqQ: 70, seqKv: 150, numHeads: 4, numKvHeads: 2, headDim: 144, positionOffset: 0);

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

    // ─────────────────────────────────────────────────────────────────────
    // Issue #240: requiredSubgroupSize=64 pin, same SPIR-V, representative
    // discriminating subset (see class remarks).
    // ─────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Launch_Pinned64_Mha_ShortPrefill()
        => RunOne(seqQ: 4, seqKv: 4, numHeads: 1, numKvHeads: 1, headDim: 64, positionOffset: 0,
            variant: FlashAttentionCoopmatVariant.Pinned64);

    [SkippableFact]
    public void Launch_Pinned64_Gqa3_SmolLm_Prefill64()
        => RunOne(seqQ: 64, seqKv: 64, numHeads: 9, numKvHeads: 3, headDim: 64, positionOffset: 0,
            variant: FlashAttentionCoopmatVariant.Pinned64);

    [SkippableFact]
    public void Launch_Pinned64_Mha_HeadDim128()
        => RunOne(seqQ: 32, seqKv: 32, numHeads: 8, numKvHeads: 8, headDim: 128, positionOffset: 0,
            variant: FlashAttentionCoopmatVariant.Pinned64);

    [SkippableFact]
    public void Launch_Pinned64_HeadDim72_PaddedTailChunk()
        => RunOne(seqQ: 33, seqKv: 40, numHeads: 4, numKvHeads: 2, headDim: 72, positionOffset: 0,
            variant: FlashAttentionCoopmatVariant.Pinned64);

    [SkippableFact]
    public void Launch_Pinned64_Hd64_Gqa3_SmolLm_LongPrefill()
        => RunOne(seqQ: 640, seqKv: 640, numHeads: 9, numKvHeads: 3, headDim: 64, positionOffset: 0,
            variant: FlashAttentionCoopmatVariant.Pinned64);

    [SkippableFact]
    public void Launch_Pinned64_Gqa8_Prefill_512()
        => RunOne(seqQ: 512, seqKv: 512, numHeads: 32, numKvHeads: 4, headDim: 64, positionOffset: 0,
            variant: FlashAttentionCoopmatVariant.Pinned64);

    [SkippableFact]
    public void Launch_Pinned64_SlidingWindow_CrossTile()
        => RunOne(seqQ: 128, seqKv: 256, numHeads: 4, numKvHeads: 2, headDim: 128,
            positionOffset: 128, slidingWindow: 100, variant: FlashAttentionCoopmatVariant.Pinned64);

    [SkippableFact]
    public void Launch_Pinned64_Bidirectional_MultiTile()
        => RunOne(seqQ: 40, seqKv: 96, numHeads: 4, numKvHeads: 2, headDim: 64,
            positionOffset: 0, maskMode: AttentionMaskMode.Bidirectional,
            variant: FlashAttentionCoopmatVariant.Pinned64);

    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// #533/#543 invariance at head_dim 256: the rows a chunk writes must be BIT-identical to the same rows of a single-pass
    /// prefill, for odd and even KV lengths (the partial-tile / masked-tile path takes the f32 tail, the full tiles the coopmat).
    /// (A chunk boundary that is not a multiple of BR re-tiles the rows, which changes which KV tiles are coopmat vs tail for a given
    /// row; that is inherent to the 128-dim shader too and is not asserted here.) A 1-token chunk is the discriminating case: without the tail gate the coopmat P.V returns different 0*v contributions.
    /// </summary>
    [SkippableTheory]
    [InlineData(61, 60)]
    [InlineData(63, 62)]
    [InlineData(65, 64)]
    [InlineData(67, 66)]
    public void Hd256_ChunkedPrefill_IsBitInvariant(int total, int firstChunk)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(VulkanFlashAttentionCoopmatKernel.SupportsDevice(device), "no coopmat tile");
        const int nh = 4, nkv = 2, hd = 256;
        var rng = new Random(685 + total);
        float[] q = RandomFloats(rng, total * nh * hd);
        float[] k = RandomFloats(rng, total * nkv * hd);
        float[] v = RandomFloats(rng, total * nkv * hd);
        using var kernel = VulkanFlashAttentionCoopmatKernel.Create(device, spvDir);

        using var bq = device.Allocate((long)q.Length * sizeof(float));
        using var bk = device.Allocate((long)k.Length * sizeof(float));
        using var bv = device.Allocate((long)v.Length * sizeof(float));
        device.Upload(q.AsSpan(), bq);
        device.Upload(k.AsSpan(), bk);
        device.Upload(v.AsSpan(), bv);

        float[] full = new float[q.Length];
        using (var bo = device.Allocate((long)full.Length * sizeof(float)))
        {
            kernel.Launch(bq, bk, bv, bo, total, total, nh, nkv, hd);
            device.Download(bo, full);
        }

        // Chunk 1: rows [0, firstChunk) against KV [0, firstChunk).
        float[] c1 = new float[firstChunk * nh * hd];
        using (var bo = device.Allocate((long)c1.Length * sizeof(float)))
        {
            kernel.Launch(bq, bk, bv, bo, firstChunk, firstChunk, nh, nkv, hd);
            device.Download(bo, c1);
        }
        for (int i = 0; i < c1.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(c1[i]) == BitConverter.SingleToInt32Bits(full[i]),
                $"chunk 1 differs from single pass at element {i}: {c1[i]} vs {full[i]}");

        // Chunk 2: the remaining rows against the whole KV, positionOffset = firstChunk. Q rows are offset in the Q buffer, so
        // re-upload just that slice.
        int rest = total - firstChunk;
        float[] q2 = q.AsSpan(firstChunk * nh * hd).ToArray();
        using var bq2 = device.Allocate((long)q2.Length * sizeof(float));
        device.Upload(q2.AsSpan(), bq2);
        float[] c2 = new float[rest * nh * hd];
        using (var bo = device.Allocate((long)c2.Length * sizeof(float)))
        {
            kernel.Launch(bq2, bk, bv, bo, rest, total, nh, nkv, hd, positionOffset: firstChunk);
            device.Download(bo, c2);
        }
        for (int i = 0; i < c2.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(c2[i]) == BitConverter.SingleToInt32Bits(full[firstChunk * nh * hd + i]),
                $"chunk 2 differs from single pass at element {i}: {c2[i]} vs {full[firstChunk * nh * hd + i]}");
    }

    /// <summary>
    /// #685: the head_dim-256 hybrids have LARGE attention scores, and a plain f16 QK^T has an error proportional to |score|
    /// (Tev1-4B NLL 2.27 -> 3.51 in situ). The hi/lo split must keep the error near f32 as the score scale grows: at amp 16 a
    /// plain-f16 kernel is off by 4e-2 (measured), the split by 4e-4.
    /// </summary>
    [SkippableTheory]
    [InlineData(1f)]
    [InlineData(4f)]
    [InlineData(16f)]
    public void Hd256_LargeScores_StayNearF32(float amp)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(VulkanFlashAttentionCoopmatKernel.SupportsDevice(device), "no coopmat tile");
        const int seqQ = 256, seqKv = 256, nh = 16, nkv = 4, hd = 256;
        var rng = new Random(7);
        float[] q = RandomFloats(rng, seqQ * nh * hd);
        for (int i = 0; i < q.Length; i++) q[i] *= amp;
        float[] k = RandomFloats(rng, seqKv * nkv * hd);
        for (int i = 0; i < k.Length; i++) k[i] *= amp;
        float[] v = RandomFloats(rng, seqKv * nkv * hd);
        float[] exp = new float[q.Length];
        ComputeExpected(q, k, v, exp, seqQ, seqKv, nh, nkv, hd, 0, 0, 0f, false, 0f, AttentionMaskMode.Causal, 0);
        using var kernel = VulkanFlashAttentionCoopmatKernel.Create(device, spvDir);
        using var bq = device.Allocate((long)q.Length * sizeof(float));
        using var bk = device.Allocate((long)k.Length * sizeof(float));
        using var bv = device.Allocate((long)v.Length * sizeof(float));
        using var bo = device.Allocate((long)q.Length * sizeof(float));
        device.Upload(q.AsSpan(), bq);
        device.Upload(k.AsSpan(), bk);
        device.Upload(v.AsSpan(), bv);
        kernel.Launch(bq, bk, bv, bo, seqQ, seqKv, nh, nkv, hd);
        float[] act = new float[q.Length];
        device.Download(bo, act);
        double maxAbs = 0;
        for (int i = 0; i < act.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(act[i] - exp[i]));
        Assert.True(maxAbs < 2e-3, $"amp {amp}: max abs error {maxAbs} (plain-f16 QK^T gives ~4e-2 at amp 16)");
    }

    private static void RunOne(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow = 0, float softCap = 0.0f, bool useAlibi = false,
        float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0,
        FlashAttentionCoopmatVariant variant = default)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(
            VulkanFlashAttentionCoopmatKernel.SupportsDevice(device),
            "Device does not expose a subgroup-scope 16x16x16 F16xF16->F32 cooperative-matrix tile.");
        Skip.IfNot(variant.IsSupportedOn(device),
            $"Device cannot pin the compute stage to requiredSubgroupSize={variant.RequiredSubgroupSize}.");

        var rng = new Random(0xC0091 + seqQ * 41 + seqKv * 17 + numHeads * 7 + headDim);
        float[] qh = RandomFloats(rng, seqQ * numHeads * headDim);
        float[] kh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] vh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] expected = new float[seqQ * numHeads * headDim];

        ComputeExpected(qh, kh, vh, expected,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            slidingWindow, softCap, useAlibi, scaleOverride, maskMode, prefixLen);

        using var kernel = VulkanFlashAttentionCoopmatKernel.Create(device, spvDir, variant);

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
