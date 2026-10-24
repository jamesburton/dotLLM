using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.MiMoV2;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Flint;

/// <summary>
/// Vulkan Flint (#1171) against the validated CPU reference on the SAME synthetic GGUF bytes. Discriminating shapes throughout: GQA 8:2,
/// Q width 512 over hidden 256, sinks drawn on every layer (full AND sliding), NoPE on the full layers (positions up to 40), a sliding
/// window shorter than the sequence, two leading dense layers, sigmoid top-3 of 8 with no selection bias, one ungated shared expert, and
/// every projection Q4_1 (dense, routed banks, shared expert) so the new Q4_1 kernels carry the whole forward. The KV cache is F16.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class FlintVulkanParityTests
{
    private readonly ITestOutputHelper _output;
    public FlintVulkanParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>All-Q4_1 Flint at Vulkan-friendly widths (packed Q4_1 banks need K % 256 == 0).</summary>
    internal static SyntheticFlintGguf.Options Q41(int window, bool fullSinks = true, int headDim = 64) => new()
    {
        Hidden = 256, MoeFf = 256, DenseFf = 256, HeadDim = headDim, Window = window, RouterStd = 0.15f,
        DenseQuant = QuantizationType.Q4_1, RoutedQuant = QuantizationType.Q4_1, SharedQuant = QuantizationType.Q4_1,
        FullLayerSinks = fullSinks,
    };

    internal static int[] Ids(int n) => Enumerable.Range(0, n).Select(i => (i * 7 + 3) % SyntheticFlintGguf.Vocab).ToArray();

    internal sealed class Fixture : IDisposable
    {
        private readonly string _path;
        public GgufFile Gguf { get; }
        public ModelConfig Config { get; }

        public Fixture(SyntheticFlintGguf.Options options)
        {
            _path = Path.Combine(Path.GetTempPath(), $"dotllm-flint-vk-{Guid.NewGuid():N}.gguf");
            File.WriteAllBytes(_path, SyntheticFlintGguf.Build(options));
            Gguf = GgufFile.Open(_path);
            Config = GgufModelConfigExtractor.Extract(Gguf.Metadata);
        }

        /// <summary>
        /// CPU reference logits. The dense Q4_1 projections run on F32 activations here (the pre-#1159 CPU prefill), which is
        /// what the Vulkan dense prefill implements; since #1159 the CPU default is llama.cpp's Q8_1-activation dot.
        /// </summary>
        public float[] CpuPrefill(int[] ids)
        {
            using var cpu = TransformerModel.LoadFromGguf(Gguf, Config, ThreadingConfig.SingleThreaded);
            DotLLM.Cpu.Kernels.MatMul.Q4PackedAllColumnsOverride = false;
            try
            {
                using ITensor t = cpu.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), deviceId: -1);
                return Copy(t);
            }
            finally { DotLLM.Cpu.Kernels.MatMul.Q4PackedAllColumnsOverride = null; }
        }

        public void Dispose()
        {
            Gguf.Dispose();
            try { File.Delete(_path); } catch { /* best effort */ }
        }
    }

    internal static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    /// <summary>Vulkan logits of every position: a <paramref name="prefill"/>-token prefill into an F16 cache, then 1-token decode steps.</summary>
    private static float[] VulkanRows(VulkanTransformerModel vk, int[] ids, int prefill, out VulkanKvCache cacheOut)
    {
        int[] pos = Enumerable.Range(0, ids.Length).ToArray();
        var cache = vk.CreateKvCache(128);
        cacheOut = cache;
        var all = new List<float>();
        vk.TrySetAllRowLogitsLimit(prefill + 1);
        using (ITensor t = vk.Forward(ids.AsSpan(0, prefill), pos.AsSpan(0, prefill), deviceId: 0, cache))
            all.AddRange(Copy(t));
        for (int i = prefill; i < ids.Length; i++)
        {
            using ITensor t = vk.Forward(ids.AsSpan(i, 1), pos.AsSpan(i, 1), deviceId: 0, cache);
            all.AddRange(Copy(t));
        }
        return all.ToArray();
    }

    internal static double[] RowKl(float[] vk, float[] cpu, int vocab)
    {
        int rows = cpu.Length / vocab;
        var kl = new double[rows];
        for (int r = 0; r < rows; r++)
            kl[r] = MiMoV2TinyOracleParityTests.Compare(vk.AsSpan(r * vocab, vocab).ToArray(), cpu.AsSpan(r * vocab, vocab).ToArray(), vocab).MaxKl;
        return kl;
    }

    private (double Median, double Good) Summarise(string tag, double[] kl, double rowLimit)
    {
        var sorted = kl.OrderBy(x => x).ToArray();
        double median = sorted[sorted.Length / 2];
        double good = kl.Count(x => x < rowLimit) / (double)kl.Length;
        string line = $"[flint vulkan {tag}] rows={kl.Length} medianKL={median:E2} maxKL={sorted[^1]:E2} rows<{rowLimit}={good:P0} | {string.Join(' ', kl.Select(x => x.ToString("E0")))}";
        _output.WriteLine(line);
        Console.WriteLine(line);
        return (median, good);
    }

    /// <summary>
    /// Random-weight MoEs flip a top-k expert choice on near-ties between engines (Q8_1 activation rounding differs), which wrecks that ONE row only.
    /// A systematic defect moves every row: median per-row KL tiny and at most ~15 % of the rows off.
    /// </summary>
    private void AssertRowsAgree(string tag, float[] vk, float[] cpu, int vocab)
    {
        Assert.Equal(cpu.Length, vk.Length);
        var (median, good) = Summarise(tag, RowKl(vk, cpu, vocab), 0.05);
        Assert.True(median < 0.01, $"{tag}: median per-row KL {median}");
        Assert.True(good >= 0.84, $"{tag}: only {good:P0} of rows agree");
    }

    /// <summary>Every projection stays Q4_1 on the device (no widened copy), and the KV cache is F16 (CLAUDE.md rules 1 and 2).</summary>
    [SkippableFact]
    public void Q4_1Weights_StayPacked_AndKvIsF16()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fx = new Fixture(Q41(window: 4));
        using var vk = VulkanTransformerModel.LoadFromGguf(fx.Gguf, fx.Config, spvDir);
        var wide = vk.DeviceQuantTypesForTest().Where(x => x.DeviceQt != QuantizationType.Q4_1).ToArray();
        Assert.True(wide.Length == 0, "widened on upload: " + string.Join(", ", wide.Select(x => $"blk.{x.Layer}.{x.Tensor}={x.DeviceQt}")));
        Assert.Equal(VulkanKvStorage.F16, vk.DefaultKvStorage);
        using var cache = vk.CreateKvCache(100);
        Assert.Equal(VulkanKvStorage.F16, cache.Storage);
        long expected = 2L * SyntheticFlintGguf.Layers * 100 * SyntheticFlintGguf.Kv * 64 * sizeof(ushort);
        Assert.Equal(expected, cache.AllocatedBytes);
    }

    /// <summary>16-token prefill into the F16 cache, window 4 &lt; 16: the F16-KV GQA flash sink kernel on every layer (#1200).</summary>
    [SkippableTheory]
    [InlineData(64)]
    [InlineData(128)]   // real Flint head_dim
    public void Prefill_F16Kv_MatchesCpu(int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fx = new Fixture(Q41(window: 4, headDim: headDim));
        int[] ids = Ids(16);
        float[] cpu = fx.CpuPrefill(ids);
        using var vk = VulkanTransformerModel.LoadFromGguf(fx.Gguf, fx.Config, spvDir);
        float[] rows = VulkanRows(vk, ids, prefill: ids.Length, out var cache);
        using (cache) { }
        AssertRowsAgree("prefill f16kv", rows, cpu, fx.Config.VocabSize);
        Assert.True(vk.SinkFlashDispatchCount > 0, "prefill did not reach the GQA flash sink kernel (#1200)");
    }

    /// <summary>
    /// 8-token prefill then 32 decode steps (positions 8..39) with window 20: the sliding layers' decode spans clip to the window and split (&gt;= 17
    /// keys) on the F16 split-KV kernel with the sink merge, the full layers (NoPE) split over everything.
    /// </summary>
    [SkippableFact]
    public void Decode_F16Kv_MatchesCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fx = new Fixture(Q41(window: 20));
        int[] ids = Ids(40);
        float[] cpu = fx.CpuPrefill(ids);
        using var vk = VulkanTransformerModel.LoadFromGguf(fx.Gguf, fx.Config, spvDir);
        long split0 = vk.GptOssSinkSplitDispatchCount;
        float[] rows = VulkanRows(vk, ids, prefill: 8, out var cache);
        using (cache) { }
        Assert.True(vk.GptOssSinkSplitDispatchCount > split0, "the F16 split-KV sink decode never ran");
        AssertRowsAgree("decode f16kv", rows, cpu, fx.Config.VocabSize);
    }

    /// <summary>
    /// A 2..4-row forward from an F16 cache takes the F16 small-row split-KV kernel on every layer and matches the CPU rows. (Not compared with
    /// the 1-row decode: there the Q4_1 projections run the dp4a MMVQ on Q8_1 activations, while 2..4 rows take the F32-activation GEMM.)
    /// </summary>
    [SkippableTheory]
    [InlineData(2)]
    [InlineData(4)]
    public void SmallRowForward_F16Kv_MatchesCpu(int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fx = new Fixture(Q41(window: 20));
        const int ctx = 40;
        int[] ids = Ids(ctx + rows);
        int[] pos = Enumerable.Range(0, ctx + rows).ToArray();
        float[] cpu = fx.CpuPrefill(ids);
        int vocab = fx.Config.VocabSize;
        using var vk = VulkanTransformerModel.LoadFromGguf(fx.Gguf, fx.Config, spvDir);
        vk.TrySetAllRowLogitsLimit(8);
        using var cache = vk.CreateKvCache(128);
        using (vk.Forward(ids.AsSpan(0, ctx), pos.AsSpan(0, ctx), deviceId: 0, cache)) { }
        long a0 = vk.SmallRowAttnCount;
        float[] batched;
        using (ITensor t = vk.Forward(ids.AsSpan(ctx, rows), pos.AsSpan(ctx, rows), deviceId: 0, cache))
            batched = Copy(t);
        Assert.Equal(SyntheticFlintGguf.Layers, vk.SmallRowAttnCount - a0);
        var kl = RowKl(batched, cpu.AsSpan(ctx * vocab, rows * vocab).ToArray(), vocab);
        var (median, _) = Summarise($"smallrow rows={rows}", kl, 0.05);
        Assert.True(median < 0.01, $"small-row rows off the CPU: median KL {median}");
    }

    /// <summary>
    /// Routed SwiGLU outputs past the F16 maximum (real Flint has tokens above 65504): the grouped coopmat MoE kernels fold the down input into
    /// an F16 operand, which turned such a token into inf / NaN on Strix until the per-row power-of-two rescale (#1171). 16 rows take the
    /// grouped path on a wave64 coopmat device; elsewhere the MMVQ arm (F32 scales) runs and the test still checks parity.
    /// </summary>
    [SkippableFact]
    public void Prefill_DownInputBeyondF16Range_StaysFiniteAndMatchesCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fx = new Fixture(Q41(window: 4) with { ExpertGateUpScale = 200f });
        int[] ids = Ids(16);
        float[] cpu = fx.CpuPrefill(ids);
        Assert.All(cpu, v => Assert.True(float.IsFinite(v)));
        using var vk = VulkanTransformerModel.LoadFromGguf(fx.Gguf, fx.Config, spvDir);
        float[] rows = VulkanRows(vk, ids, prefill: ids.Length, out var cache);
        using (cache) { }
        Assert.True(rows.All(float.IsFinite), "non-finite Vulkan logits: the F16 down operand overflowed");
        AssertRowsAgree("prefill f16-range down input", rows, cpu, fx.Config.VocabSize);

        // Control: where the grouped kernels ran, the same forward without the rescale must overflow, or this test cannot see the bug.
        if (vk.GroupedMoeDispatchCount > 0)
        {
            using var vkOff = VulkanTransformerModel.LoadFromGguf(fx.Gguf, fx.Config, spvDir);
            vkOff.MoeDownRescaleEnabled = false;
            float[] off = VulkanRows(vkOff, ids, prefill: ids.Length, out var cacheOff);
            using (cacheOff) { }
            Assert.False(off.All(float.IsFinite), "without the rescale the grouped path stayed finite: the synthetic down input does not reach the F16 limit");
        }
    }

    /// <summary>
    /// Sensitivity controls: each feature the parity tests cover must be visible in the logits, or a Vulkan path that silently dropped it would
    /// still pass. "full-sinks": the CPU on a file whose FULL-layer sinks are -30 (sliding unchanged) must disagree with Vulkan on the real file;
    /// "nope": Vulkan rotating the full layers must disagree with the CPU.
    /// </summary>
    [SkippableTheory]
    [InlineData("full-sinks")]
    [InlineData("nope")]
    public void Mutant_IsDetected(string mutant)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fx = new Fixture(Q41(window: 4));
        int[] ids = Ids(16);
        float[] cpu;
        if (mutant == "full-sinks")
        {
            using var noFullSinks = new Fixture(Q41(window: 4, fullSinks: false));
            cpu = noFullSinks.CpuPrefill(ids);
        }
        else
        {
            cpu = fx.CpuPrefill(ids);
        }
        var cfg = mutant == "nope" ? fx.Config with { NoRopeLayers = null } : fx.Config;
        using var vk = VulkanTransformerModel.LoadFromGguf(fx.Gguf, cfg, spvDir);
        float[] rows = VulkanRows(vk, ids, prefill: ids.Length, out var cache);
        using (cache) { }
        var kl = RowKl(rows, cpu, fx.Config.VocabSize);
        Summarise($"mutant {mutant}", kl, 0.02);
        double moved = kl.Count(x => x > 0.02) / (double)kl.Length;
        Assert.True(moved > 0.4, $"mutant '{mutant}' is invisible: only {moved:P0} rows moved");
    }
}
