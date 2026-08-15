using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda.Kernels;

/// <summary>
/// CPU-vs-CUDA parity for RoPE with dense YaRN scaling active (issue #366).
/// </summary>
/// <remarks>
/// <para>
/// Before #366 the CUDA RoPE kernels derived every frequency from <c>theta</c> alone
/// (<c>native/kernels/rope.cu</c>, <c>rope_f32.cu</c>, <c>fused_rope_kv_write.cu</c>) and
/// <c>src/DotLLM.Cuda/</c> contained no reference to <c>ScalingType</c>,
/// <c>ScalingFactor</c>, <c>AttnFactor</c>, <c>BetaFast</c> or <c>BetaSlow</c> whatsoever.
/// The CPU reference, by contrast, rebuilds its cos/sin tables through
/// <see cref="RoPE.PrecomputeFrequencyTableYarn"/> whenever
/// <see cref="RoPEConfig.IsDenseYarnActive"/> holds.
/// </para>
/// <para>
/// This is NOT only a long-context concern. YaRN's mscale multiplies <em>both</em> cos and
/// sin (<c>RoPE.PrecomputeFrequencyTableYarn</c>, the <c>* mscaleMultiplier</c> on the
/// cos/sin stores), so at position 0 — where the rotation would otherwise be the identity
/// <c>(cos, sin) = (1, 0)</c> — it becomes <c>(mscale, 0)</c> and scales Q and K outright.
/// For gpt-oss's shipped metadata (<c>rope.scaling.type=yarn</c>, <c>factor=32</c>,
/// <c>original_context_length=4096</c>) mscale is <c>1 + 0.1*ln(32) ≈ 1.3466</c> — a ~35%
/// error on the very first token. The <c>*_IgnoringYarn_Diverges*</c> tests below pin that
/// down at position 0 specifically, so they fail if the scaling is ever dropped again.
/// </para>
/// <para>
/// Tolerances follow the sibling <c>RoPEF32ParityTests</c> convention (documented
/// mixed abs/rel tolerance, never <c>SequenceEqual</c>): the GPU evaluates
/// <c>cosf</c>/<c>sinf</c> where the CPU evaluates <c>MathF.Cos</c>/<c>MathF.Sin</c>, so
/// exact equality is not expected even though both consume the identical inverse
/// frequencies from <see cref="RoPE.ComputeYarnInverseFrequencies"/>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("CudaKernels")]
public sealed class RoPEYarnParityTests : IDisposable
{
    private readonly CudaKernelTestHarness _harness = new();

    public void Dispose() => _harness.Dispose();

    // gpt-oss-20b geometry and RoPE metadata (mirrors GptOssConfigTests.BuildGptOssMetadata:
    // gpt-oss.rope.scaling.type=yarn, factor=32, original_context_length=4096, theta 150000,
    // head dim 64, NeoX pairing).
    private const int HeadDim = 64;
    private const int RopeDim = HeadDim;
    private const int HalfRope = RopeDim / 2;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const float Theta = 150000.0f;
    private const float ScalingFactor = 32.0f;
    private const int OrigCtx = 4096;
    private const float BetaFast = 32.0f;
    private const float BetaSlow = 1.0f;

    private static RoPEConfig GptOssRope() => new(
        Theta: Theta, DimensionCount: RopeDim, Type: RoPEType.NeoX,
        ScalingType: RoPEScalingType.YaRN, ScalingFactor: ScalingFactor,
        OrigMaxSeqLen: OrigCtx, AttnFactor: 1.0f,
        BetaFast: BetaFast, BetaSlow: BetaSlow);

    /// <summary>
    /// The gate the whole fix hangs on: gpt-oss's mscale is materially different from 1,
    /// so it changes numerics at EVERY position — position 0 included.
    /// </summary>
    [Fact]
    public void GptOssYarnMscale_IsNotUnity_SoShortContextNumericsAreAffected()
    {
        RoPEConfig rope = GptOssRope();
        Assert.True(rope.IsDenseYarnActive);

        float mscale = rope.ComputeYarnMscaleMultiplier(Architecture.GptOss);
        // 1 + 0.1 * ln(32) = 1.34657...
        Assert.Equal(1.0f + 0.1f * MathF.Log(32.0f), mscale, 5);
        Assert.True(mscale > 1.3f,
            $"gpt-oss YaRN mscale collapsed to {mscale}; the CUDA fix would become a no-op.");

        // Other dense-YaRN architectures keep the plain AttnFactor convention.
        Assert.Equal(1.0f, rope.ComputeYarnMscaleMultiplier(Architecture.SmolLM3));
    }

    /// <summary>
    /// The inverse frequencies the GPU uploads must be the very ones the CPU table was
    /// built from — otherwise the two would drift silently. Reconstructs the CPU cos/sin
    /// table from <see cref="RoPE.ComputeYarnInverseFrequencies"/> and checks it matches
    /// <see cref="RoPE.PrecomputeFrequencyTableYarn"/> exactly.
    /// </summary>
    [Fact]
    public void ComputeYarnInverseFrequencies_ReproducesPrecomputedYarnTable()
    {
        const int maxPos = 8;
        float mscale = GptOssRope().ComputeYarnMscaleMultiplier(Architecture.GptOss);

        float[] cos = new float[maxPos * HalfRope];
        float[] sin = new float[maxPos * HalfRope];
        RoPE.PrecomputeFrequencyTableYarn(maxPos, RopeDim, Theta, ScalingFactor, OrigCtx,
                                          BetaFast, BetaSlow, mscale, cos, sin);

        float[] invFreq = new float[HalfRope];
        RoPE.ComputeYarnInverseFrequencies(RopeDim, Theta, ScalingFactor, OrigCtx,
                                           BetaFast, BetaSlow, invFreq);

        for (int pos = 0; pos < maxPos; pos++)
        {
            for (int i = 0; i < HalfRope; i++)
            {
                float angle = pos * invFreq[i];
                Assert.Equal(MathF.Cos(angle) * mscale, cos[pos * HalfRope + i], 6);
                Assert.Equal(MathF.Sin(angle) * mscale, sin[pos * HalfRope + i], 6);
            }
        }
    }

    [SkippableFact]
    public void RoPEF32_Yarn_MatchesCpuYarnReference()
    {
        _harness.SkipIfUnavailable();

        const int seqLen = 4;
        var rng = new Random(366);
        float[] q = CudaKernelTestHarness.RandomF32(rng, seqLen * NumHeads * HeadDim);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqLen * NumKvHeads * HeadDim);
        // Positions start at 0 deliberately: mscale must apply there too.
        int[] positions = [0, 1, 2, 3];

        (float[] cpuQ, float[] cpuK) = CpuYarnReference(q, k, positions, seqLen);

        nint devInvFreq = _harness.Upload(YarnInvFreq());
        float mscale = GptOssRope().ComputeYarnMscaleMultiplier(Architecture.GptOss);
        (float[] gpuQ, float[] gpuK) = RunGpuRoPEF32(q, k, positions, seqLen, devInvFreq, mscale);

        CudaKernelTestHarness.AssertClose("RoPEF32-YaRN Q", cpuQ, gpuQ, 1e-5f, 1e-4f);
        CudaKernelTestHarness.AssertClose("RoPEF32-YaRN K", cpuK, gpuK, 1e-5f, 1e-4f);
    }

    /// <summary>
    /// Discriminating (trap-the-bug) counterpart: running the SAME kernel with the pre-#366
    /// arguments (no inverse-frequency table, mscale = 1) must diverge from the CPU YaRN
    /// reference — and specifically at POSITION 0, which is what makes this a short-context
    /// correctness bug rather than a long-context extrapolation gap. A degenerate check over
    /// the whole tensor could be satisfied by the far positions alone, so position 0 is
    /// asserted on its own.
    /// </summary>
    [SkippableFact]
    public void RoPEF32_IgnoringYarn_DivergesFromCpuAtPositionZero()
    {
        _harness.SkipIfUnavailable();

        const int seqLen = 4;
        var rng = new Random(367);
        float[] q = CudaKernelTestHarness.RandomF32(rng, seqLen * NumHeads * HeadDim);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqLen * NumKvHeads * HeadDim);
        int[] positions = [0, 1, 2, 3];

        (float[] cpuQ, _) = CpuYarnReference(q, k, positions, seqLen);

        // BUG INJECTION: the pre-#366 call — no table, no mscale.
        (float[] gpuQ, _) = RunGpuRoPEF32(q, k, positions, seqLen, ropeInvFreq: 0, mscale: 1.0f);

        // Position 0 occupies the first NumHeads * HeadDim elements of Q.
        int pos0Count = NumHeads * HeadDim;
        float maxAbsPos0 = 0;
        for (int i = 0; i < pos0Count; i++)
            maxAbsPos0 = MathF.Max(maxAbsPos0, MathF.Abs(cpuQ[i] - gpuQ[i]));

        Assert.True(maxAbsPos0 > 1e-2f,
            $"Dropping YaRN scaling left POSITION 0 indistinguishable from the CPU YaRN " +
            $"reference (maxAbs={maxAbsPos0:E4}). YaRN's mscale multiplies cos AND sin at " +
            $"every position, so position 0 must be scaled by ~1.3466 for gpt-oss. If this " +
            $"test ever passes, the mscale convention changed and #366's premise needs review.");
    }

    [SkippableFact]
    public void RoPEF16_Yarn_MatchesCpuYarnReference()
    {
        _harness.SkipIfUnavailable();

        const int seqLen = 4;
        var rng = new Random(368);
        float[] q = CudaKernelTestHarness.RandomF32(rng, seqLen * NumHeads * HeadDim);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqLen * NumKvHeads * HeadDim);
        int[] positions = [0, 1, 2, 3];

        // Round the inputs through Half first so the CPU reference sees exactly what the
        // GPU kernel loads — otherwise the comparison also charges input quantization to
        // the kernel.
        Half[] qh = ToHalf(q), kh = ToHalf(k);
        (float[] cpuQ, float[] cpuK) = CpuYarnReference(ToFloat(qh), ToFloat(kh), positions, seqLen);

        nint devQ = _harness.Upload(qh);
        nint devK = _harness.Upload(kh);
        nint devPos = _harness.Upload(positions);
        nint devInvFreq = _harness.Upload(YarnInvFreq());
        float mscale = GptOssRope().ComputeYarnMscaleMultiplier(Architecture.GptOss);

        _harness.Kernels.LaunchRoPE(devQ, devK, devPos, seqLen, NumHeads, NumKvHeads,
                                    HeadDim, RopeDim, Theta, CudaKernels.ToCudaRopeType(RoPEType.NeoX),
                                    _harness.StreamHandle, devInvFreq, mscale);
        _harness.Synchronize();

        float[] gpuQ = ToFloat(_harness.DownloadHalves(devQ, qh.Length));
        float[] gpuK = ToFloat(_harness.DownloadHalves(devK, kh.Length));

        // FP16 storage: ~1e-3 relative resolution, so the tolerance is looser than the F32
        // sibling by roughly the half-precision epsilon rather than for any kernel reason.
        CudaKernelTestHarness.AssertClose("RoPEF16-YaRN Q", cpuQ, gpuQ, 2e-3f, 5e-3f);
        CudaKernelTestHarness.AssertClose("RoPEF16-YaRN K", cpuK, gpuK, 2e-3f, 5e-3f);
    }

    /// <summary>
    /// The decode path. <c>CudaTransformerModel</c> takes the fused RoPE+KV-write kernel by
    /// default for seqLen==1 (both eager and CUDA-graph replay), so a YaRN fix that stopped
    /// at rope_f16/rope_f32 would leave every GENERATED token unscaled while prefill was
    /// correct — the hardest failure mode to spot. Verified at position 0 and at a non-zero
    /// position.
    /// </summary>
    [SkippableTheory]
    [InlineData(0)]
    [InlineData(7)]
    public void FusedRopeKvWrite_Yarn_MatchesCpuYarnReference(int position)
    {
        _harness.SkipIfUnavailable();
        Skip.IfNot(_harness.Kernels.HasFusedRopeKvWriteKernel,
                   "fused_rope_kv_write.ptx not loaded.");

        const int maxSeq = 16;
        int kvStride = NumKvHeads * HeadDim;
        var rng = new Random(369 + position);

        Half[] qh = ToHalf(CudaKernelTestHarness.RandomF32(rng, NumHeads * HeadDim));
        Half[] kh = ToHalf(CudaKernelTestHarness.RandomF32(rng, kvStride));
        Half[] vh = ToHalf(CudaKernelTestHarness.RandomF32(rng, kvStride));

        // CPU reference over a single token at `position`.
        (float[] cpuQ, float[] cpuK) = CpuYarnReference(
            ToFloat(qh), ToFloat(kh), [position], seqLen: 1);

        nint devQ = _harness.Upload(qh);
        nint devK = _harness.Upload(kh);
        nint devV = _harness.Upload(vh);
        nint devPos = _harness.Upload(new[] { position });
        nint devInvFreq = _harness.Upload(YarnInvFreq());
        nint kCache = _harness.Allocate((long)maxSeq * kvStride * sizeof(ushort));
        nint vCache = _harness.Allocate((long)maxSeq * kvStride * sizeof(ushort));
        float mscale = GptOssRope().ComputeYarnMscaleMultiplier(Architecture.GptOss);

        _harness.Kernels.LaunchFusedRopeKvWriteF16(
            devQ, devK, devV, kCache, vCache, devPos, position,
            NumHeads, NumKvHeads, HeadDim, RopeDim, kvStride, Theta,
            CudaKernels.ToCudaRopeType(RoPEType.NeoX),
            _harness.StreamHandle, devInvFreq, mscale);
        _harness.Synchronize();

        float[] gpuQ = ToFloat(_harness.DownloadHalves(devQ, qh.Length));
        // The rotated K row lands at cache row `position`.
        float[] gpuKRow = ToFloat(_harness.DownloadHalves(
            kCache + (nint)((long)position * kvStride * sizeof(ushort)), kvStride));
        float[] gpuVRow = ToFloat(_harness.DownloadHalves(
            vCache + (nint)((long)position * kvStride * sizeof(ushort)), kvStride));

        CudaKernelTestHarness.AssertClose($"FusedRopeKv-YaRN Q@{position}", cpuQ, gpuQ, 2e-3f, 5e-3f);
        CudaKernelTestHarness.AssertClose($"FusedRopeKv-YaRN K@{position}", cpuK, gpuKRow, 2e-3f, 5e-3f);
        // V is copied verbatim (never rotated) — YaRN must not touch it.
        CudaKernelTestHarness.AssertClose($"FusedRopeKv-YaRN V@{position}", ToFloat(vh), gpuVRow, 0f, 0f);
    }

    /// <summary>
    /// Discriminating counterpart for the decode path, at position 0.
    /// </summary>
    [SkippableFact]
    public void FusedRopeKvWrite_IgnoringYarn_DivergesFromCpuAtPositionZero()
    {
        _harness.SkipIfUnavailable();
        Skip.IfNot(_harness.Kernels.HasFusedRopeKvWriteKernel,
                   "fused_rope_kv_write.ptx not loaded.");

        const int maxSeq = 16;
        const int position = 0;
        int kvStride = NumKvHeads * HeadDim;
        var rng = new Random(370);

        Half[] qh = ToHalf(CudaKernelTestHarness.RandomF32(rng, NumHeads * HeadDim));
        Half[] kh = ToHalf(CudaKernelTestHarness.RandomF32(rng, kvStride));
        Half[] vh = ToHalf(CudaKernelTestHarness.RandomF32(rng, kvStride));

        (float[] cpuQ, _) = CpuYarnReference(ToFloat(qh), ToFloat(kh), [position], seqLen: 1);

        nint devQ = _harness.Upload(qh);
        nint devK = _harness.Upload(kh);
        nint devV = _harness.Upload(vh);
        nint devPos = _harness.Upload(new[] { position });
        nint kCache = _harness.Allocate((long)maxSeq * kvStride * sizeof(ushort));
        nint vCache = _harness.Allocate((long)maxSeq * kvStride * sizeof(ushort));

        // BUG INJECTION: pre-#366 arguments.
        _harness.Kernels.LaunchFusedRopeKvWriteF16(
            devQ, devK, devV, kCache, vCache, devPos, position,
            NumHeads, NumKvHeads, HeadDim, RopeDim, kvStride, Theta,
            CudaKernels.ToCudaRopeType(RoPEType.NeoX),
            _harness.StreamHandle, ropeInvFreq: 0, ropeMscale: 1.0f);
        _harness.Synchronize();

        float[] gpuQ = ToFloat(_harness.DownloadHalves(devQ, qh.Length));

        float maxAbs = 0;
        for (int i = 0; i < cpuQ.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(cpuQ[i] - gpuQ[i]));

        Assert.True(maxAbs > 1e-2f,
            $"Dropping YaRN scaling on the fused DECODE kernel left position 0 " +
            $"indistinguishable from the CPU YaRN reference (maxAbs={maxAbs:E4}).");
    }

    /// <summary>
    /// Non-YaRN callers must be untouched: passing the (0, 1.0f) sentinel has to reproduce
    /// the plain non-YaRN CPU reference exactly as before #366.
    /// </summary>
    [SkippableFact]
    public void RoPEF32_NoYarnSentinel_StillMatchesPlainCpuReference()
    {
        _harness.SkipIfUnavailable();

        const int seqLen = 4;
        var rng = new Random(371);
        float[] q = CudaKernelTestHarness.RandomF32(rng, seqLen * NumHeads * HeadDim);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqLen * NumKvHeads * HeadDim);
        int[] positions = [0, 1, 2, 3];

        float[] cos = new float[(seqLen + 1) * HalfRope];
        float[] sin = new float[(seqLen + 1) * HalfRope];
        RoPE.PrecomputeFrequencyTable(seqLen + 1, RopeDim, Theta, cos, sin);

        float[] cpuQ = (float[])q.Clone();
        float[] cpuK = (float[])k.Clone();
        RoPE.Execute(cpuQ, cpuK, positions, NumHeads, NumKvHeads, HeadDim, RopeDim,
                     cos, sin, RoPEType.NeoX);

        (float[] gpuQ, float[] gpuK) = RunGpuRoPEF32(q, k, positions, seqLen, ropeInvFreq: 0, mscale: 1.0f);

        CudaKernelTestHarness.AssertClose("RoPEF32-NoYaRN Q", cpuQ, gpuQ, 1e-5f, 1e-4f);
        CudaKernelTestHarness.AssertClose("RoPEF32-NoYaRN K", cpuK, gpuK, 1e-5f, 1e-4f);
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static float[] YarnInvFreq()
    {
        float[] invFreq = new float[HalfRope];
        RoPE.ComputeYarnInverseFrequencies(RopeDim, Theta, ScalingFactor, OrigCtx,
                                           BetaFast, BetaSlow, invFreq);
        return invFreq;
    }

    /// <summary>
    /// CPU oracle: exactly the path <c>TransformerModel.BuildFromPrebuiltWeightsInternal</c>
    /// takes for a dense-YaRN gpt-oss model — YaRN cos/sin tables, then NeoX rotation.
    /// </summary>
    private static (float[] Q, float[] K) CpuYarnReference(
        float[] q, float[] k, int[] positions, int seqLen)
    {
        int maxPos = 1;
        foreach (int p in positions) maxPos = Math.Max(maxPos, p + 1);

        float mscale = GptOssRope().ComputeYarnMscaleMultiplier(Architecture.GptOss);
        float[] cos = new float[maxPos * HalfRope];
        float[] sin = new float[maxPos * HalfRope];
        RoPE.PrecomputeFrequencyTableYarn(maxPos, RopeDim, Theta, ScalingFactor, OrigCtx,
                                          BetaFast, BetaSlow, mscale, cos, sin);

        float[] cpuQ = (float[])q.Clone();
        float[] cpuK = (float[])k.Clone();
        RoPE.Execute(cpuQ, cpuK, positions, NumHeads, NumKvHeads, HeadDim, RopeDim,
                     cos, sin, RoPEType.NeoX);
        _ = seqLen;
        return (cpuQ, cpuK);
    }

    private (float[] Q, float[] K) RunGpuRoPEF32(
        float[] q, float[] k, int[] positions, int seqLen, nint ropeInvFreq, float mscale)
    {
        nint devQ = _harness.Upload(q);
        nint devK = _harness.Upload(k);
        nint devPos = _harness.Upload(positions);

        _harness.Kernels.LaunchRoPEF32(devQ, devK, devPos, seqLen, NumHeads, NumKvHeads,
                                       HeadDim, RopeDim, Theta,
                                       CudaKernels.ToCudaRopeType(RoPEType.NeoX),
                                       _harness.StreamHandle,
                                       ropeInvFreq: ropeInvFreq, ropeMscale: mscale);
        _harness.Synchronize();

        return (_harness.DownloadFloats(devQ, q.Length), _harness.DownloadFloats(devK, k.Length));
    }

    private static Half[] ToHalf(float[] src)
    {
        var dst = new Half[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (Half)src[i];
        return dst;
    }

    private static float[] ToFloat(Half[] src)
    {
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
        return dst;
    }
}
