using System;
using DotLLM.Core.Attention;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Measures whether Vulkan attention output depends on the <b>padded</b> KV length — the
/// cross-backend half of #525, tracked as #532.
/// </summary>
/// <remarks>
/// <para>
/// <b>The question.</b> #525 established that on CPU, attention reduces over the padded
/// <c>seqKv</c> rather than the causally visible prefix. The padding contributes 0.0
/// mathematically, but changing the span length changes lane assignment and the remainder tail,
/// so the <i>real</i> elements accumulate in a different order. On a quantized model that ULP is
/// then digitized by activation quantization and can change the emitted token.
/// </para>
/// <para>
/// <b>The experiment.</b> Hold the real prefix fixed (identical Q, and identical K/V rows
/// <c>0..posQ</c>), then present the kernel with two different <c>seqKv</c> values. Everything
/// past <c>posQ</c> is causally masked, so both calls must compute the same thing. Any
/// difference is pure reduction-order drift caused by the cache length.
/// </para>
/// <para>
/// <b>Prediction from reading the shaders</b> (recorded before running, so the result can
/// falsify it):
/// </para>
/// <list type="bullet">
/// <item><description>
/// <c>attention_f32.comp</c> — <b>invariant</b>. It tiles from 0 in fixed <c>TILE_KV=256</c>
/// steps, so a real row keeps its <c>(tile, t)</c> slot; masked rows become <c>NEG_INF</c> and
/// are <i>skipped</i> by <c>if (w &gt; 0.0)</c> rather than added as zero; the max/sum reductions
/// are fixed-width trees over <c>WG_SIZE</c>, not over <c>tileLen</c>; and an extra all-masked
/// tile contributes <c>correction = exp(0) = 1.0</c>, which is bit-preserving.
/// </description></item>
/// <item><description>
/// <c>attention_f32_splitkv.comp</c> — <b>exposed</b>. <c>splitLen = ceil(seqKv/numSplits)</c>
/// and <c>numSplits = ComputeSplits(seqKv, …)</c>, so both the chunk width and the chunk count
/// move with the cache length and real rows are regrouped into different partials.
/// </description></item>
/// </list>
/// <para>
/// <b>This probe reports; it does not gate.</b> An assertion pinning today's behaviour would
/// freeze a defect. The split-KV case emits its measured delta so the issue carries a number,
/// and only the single-pass case — where invariance is a structural property worth protecting —
/// is asserted.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class Probe532AttentionKvLengthInvarianceTests
{
    private readonly ITestOutputHelper _out;

    public Probe532AttentionKvLengthInvarianceTests(ITestOutputHelper output) => _out = output;

    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 64;

    /// <summary>
    /// The single-pass kernel must be <b>bit-identical</b> across padded cache lengths. This one
    /// is asserted: it is structurally invariant today (see the remarks) and a regression would
    /// mean the CPU defect had been reintroduced on GPU.
    /// </summary>
    [SkippableTheory]
    [InlineData(300, 301)]    // no padding vs none — self-control, must trivially hold
    [InlineData(300, 320)]    // padding inside the same TILE_KV tile
    [InlineData(300, 600)]    // padding that opens a SECOND tile (the exp(0)=1.0 rescale path)
    [InlineData(255, 1024)]   // real prefix ends one row before a tile boundary
    public void SinglePass_IsBitIdentical_AcrossPaddedKvLengths(int posQ, int paddedKv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int tightKv = posQ + 1;
        Skip.If(paddedKv < tightKv, "padded length must be >= the visible prefix");

        using var device = VulkanDevice.Create();
        using var kernel = AttentionF32Kernel.Create(device, spvDir);

        var fx = Fixture.Build(Math.Max(paddedKv, tightKv));

        float[] tight = RunSinglePass(device, kernel, fx, posQ, tightKv);
        float[] padded = RunSinglePass(device, kernel, fx, posQ, paddedKv);

        int diff = CountDiffering(tight, padded, out float worst);
        // Record WHICH shader produced this. AttentionF32Kernel picks one of three variants at
        // construction (attention_f32 / _sg / _coopmat), so "the single-pass kernel is
        // invariant" is meaningless without naming the dispatched path — and a mutant applied
        // to the wrong variant silently changes nothing, which reads as a passing gate.
        _out.WriteLine($"single-pass [{kernel.Mode}] posQ={posQ} kv {tightKv} vs {paddedKv}: differing={diff}/{tight.Length} worst={worst:E3}");

        Assert.Equal(0, diff);
    }

    /// <summary>
    /// The split-KV kernel, measured and REPORTED rather than asserted. Shapes are chosen so the
    /// split count actually moves — otherwise the probe would report invariance for the trivial
    /// reason that nothing changed.
    /// </summary>
    [SkippableTheory]
    [InlineData(300, 4096)]
    [InlineData(300, 8192)]
    [InlineData(1000, 4096)]
    public void SplitKv_KvLengthDependence_IsMeasured(int posQ, int paddedKv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int tightKv = posQ + 1;
        int sTight = VulkanSplitKvAttentionKernel.ComputeSplits(tightKv, NumHeads);
        int sPadded = VulkanSplitKvAttentionKernel.ComputeSplits(paddedKv, NumHeads);

        // Discrimination guard: if the split geometry does not move, this shape cannot show the
        // effect and a clean result would be meaningless.
        Skip.If(sTight == sPadded && (tightKv + sTight - 1) / sTight == (paddedKv + sPadded - 1) / sPadded,
            $"split geometry identical (S={sTight}) — shape cannot discriminate.");

        using var device = VulkanDevice.Create();
        using var kernel = VulkanSplitKvAttentionKernel.Create(device, spvDir);

        var fx = Fixture.Build(Math.Max(paddedKv, tightKv));

        float[] tight = RunSplitKv(device, kernel, fx, posQ, tightKv);
        float[] padded = RunSplitKv(device, kernel, fx, posQ, paddedKv);

        int diff = CountDiffering(tight, padded, out float worst);
        _out.WriteLine(
            $"#532 split-KV posQ={posQ} kv {tightKv}(S={sTight}, len={(tightKv + sTight - 1) / sTight}) " +
            $"vs {paddedKv}(S={sPadded}, len={(paddedKv + sPadded - 1) / sPadded}): " +
            $"differing={diff}/{tight.Length} worst_abs={worst:E3}");

        // Deliberately NOT asserted bit-identity — that is #532's fix, not its measurement.
        // What IS asserted: the padded run must still be finite and numerically close, so a
        // catastrophic divergence (as opposed to ULP drift) fails loudly.
        foreach (float f in padded) Assert.True(float.IsFinite(f), "split-KV padded run produced a non-finite value.");
        Assert.True(worst < 1e-3f, $"divergence {worst:E3} is far beyond reduction-order drift — this is a real bug, not ULP.");
    }

    /// <summary>
    /// CONTROL, and the one that makes the split-KV numbers interpretable: at a <b>fixed</b>
    /// cache length the kernel must be bit-deterministic run to run.
    /// </summary>
    /// <remarks>
    /// Without this, "230 of 256 outputs differ" is ambiguous — it could mean the split-KV kernel
    /// is simply nondeterministic (atomics, non-fixed workgroup scheduling), in which case the
    /// comparison says nothing about <c>seqKv</c> at all. This arm is <i>sensitive</i>: if
    /// nondeterminism were the explanation, it would fail here, at the same shape, with the same
    /// harness. It passing is what licenses attributing the divergence to the cache length.
    /// </remarks>
    [SkippableTheory]
    [InlineData(300, 4096)]
    [InlineData(300, 8192)]
    public void SplitKv_SameKvLength_IsBitDeterministic(int posQ, int seqKv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        using var kernel = VulkanSplitKvAttentionKernel.Create(device, spvDir);
        var fx = Fixture.Build(seqKv);

        float[] a = RunSplitKv(device, kernel, fx, posQ, seqKv);
        float[] b = RunSplitKv(device, kernel, fx, posQ, seqKv);

        int diff = CountDiffering(a, b, out float worst);
        _out.WriteLine($"#532 control split-KV posQ={posQ} kv={seqKv} twice: differing={diff}/{a.Length} worst={worst:E3}");

        Assert.Equal(0, diff);
    }

    private sealed record Fixture(float[] Q, float[] K, float[] V)
    {
        public static Fixture Build(int maxKv)
        {
            var rng = new Random(0x532);
            return new Fixture(
                RandomFloats(rng, NumHeads * HeadDim),
                RandomFloats(rng, maxKv * NumKvHeads * HeadDim),
                RandomFloats(rng, maxKv * NumKvHeads * HeadDim));
        }
    }

    private static float[] RunSinglePass(
        VulkanDevice device, AttentionF32Kernel kernel, Fixture fx, int posQ, int seqKv)
    {
        var outh = new float[NumHeads * HeadDim];
        long kvFloats = (long)seqKv * NumKvHeads * HeadDim;

        using var bq = device.Allocate((long)fx.Q.Length * sizeof(float));
        using var bk = device.Allocate(kvFloats * sizeof(float));
        using var bv = device.Allocate(kvFloats * sizeof(float));
        using var bo = device.Allocate((long)outh.Length * sizeof(float));

        device.Upload(fx.Q.AsSpan(), bq);
        device.Upload(fx.K.AsSpan(0, (int)kvFloats), bk);
        device.Upload(fx.V.AsSpan(0, (int)kvFloats), bv);

        kernel.Launch(bq, bk, bv, bo,
            seqQ: 1, seqKv: seqKv, numHeads: NumHeads, numKvHeads: NumKvHeads, headDim: HeadDim,
            positionOffset: posQ, maskMode: AttentionMaskMode.Causal);

        device.Download(bo, outh);
        return outh;
    }

    private static float[] RunSplitKv(
        VulkanDevice device, VulkanSplitKvAttentionKernel kernel, Fixture fx, int posQ, int seqKv)
    {
        var outh = new float[NumHeads * HeadDim];
        long kvFloats = (long)seqKv * NumKvHeads * HeadDim;

        using var bq = device.Allocate((long)fx.Q.Length * sizeof(float));
        using var bk = device.Allocate(kvFloats * sizeof(float));
        using var bv = device.Allocate(kvFloats * sizeof(float));
        using var bo = device.Allocate((long)outh.Length * sizeof(float));

        device.Upload(fx.Q.AsSpan(), bq);
        device.Upload(fx.K.AsSpan(0, (int)kvFloats), bk);
        device.Upload(fx.V.AsSpan(0, (int)kvFloats), bv);

        kernel.Launch(bq, bk, bv, bo,
            seqQ: 1, seqKv: seqKv, numHeads: NumHeads, numKvHeads: NumKvHeads, headDim: HeadDim,
            positionOffset: posQ, maskMode: AttentionMaskMode.Causal);

        device.Download(bo, outh);
        return outh;
    }

    private static int CountDiffering(float[] a, float[] b, out float worstAbs)
    {
        int n = 0;
        worstAbs = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i].Equals(b[i])) continue;   // bitwise-ish: exact float equality
            n++;
            worstAbs = Math.Max(worstAbs, Math.Abs(a[i] - b[i]));
        }
        return n;
    }

    private static float[] RandomFloats(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
