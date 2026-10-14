using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.MiMoV2;

/// <summary>
/// Vulkan MiMo-V2 (#955) against the validated CPU reference on the SAME synthetic GGUF bytes: zero-row-padded V (K 48 / V 32), the
/// attention-output compaction + value scale, per-layer KV heads (2 global / 4 sliding) and dual rope theta on the inverse-frequency kernel,
/// sliding-window sink attention, sigmoid+bias MoE with a dense layer 0, and the packed IQ routed banks on the MMVQ (decode) and grouped coopmat
/// (prefill) arms. GPU tests: take scripts/gpu-lock.sh first.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class MiMoV2VulkanParityTests
{
    private readonly ITestOutputHelper _output;
    public MiMoV2VulkanParityTests(ITestOutputHelper output) => _output = output;

    private static readonly SyntheticMiMoV2Gguf.Options Q80 = new() { GateUpQuant = QuantizationType.Q8_0, DownQuant = QuantizationType.Q8_0, RouterStd = 0.3f };

    private static readonly int[] Ids16 = MiMoV2SyntheticGoldenTests.Ids;

    private static (float[] Cpu, float[] Vk, int Vocab) Run(SyntheticMiMoV2Gguf.Options options, int[] ids, bool vkDecode = false,
        Func<ModelConfig, ModelConfig>? mutateVk = null)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = Path.Combine(Path.GetTempPath(), $"dotllm-mimo2-vk-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticMiMoV2Gguf.Build(options));
        try
        {
            using var gguf = GgufFile.Open(path);
            ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            int[] pos = Enumerable.Range(0, ids.Length).ToArray();
            float[] cpu;
            using (var cpuModel = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded))
            {
                using ITensor t = cpuModel.Forward(ids, pos, deviceId: -1);
                cpu = Copy(t);
            }
            var vkCfg = mutateVk is null ? cfg : mutateVk(cfg);
            using var vk = VulkanTransformerModel.LoadFromGguf(gguf, vkCfg, spvDir);
            float[] vkLogits;
            if (!vkDecode)
            {
                vk.TrySetAllRowLogitsLimit(ids.Length + 1);
                using ITensor t = vk.Forward(ids, pos, deviceId: 0);
                vkLogits = Copy(t);
            }
            else
            {
                using var cache = vk.CreateKvCache(64);
                var all = new List<float>();
                for (int i = 0; i < ids.Length; i++)
                {
                    using ITensor t = vk.Forward(ids.AsSpan(i, 1), pos.AsSpan(i, 1), deviceId: 0, cache);
                    all.AddRange(Copy(t));
                }
                vkLogits = all.ToArray();
            }
            return (cpu, vkLogits, cfg.VocabSize);
        }
        finally
        {
            try { File.Delete(path); } catch { /* best effort */ }
        }
    }

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    /// <summary>Per-row KL(cpu||vk) and top-1 flags.</summary>
    private static (double[] Kl, bool[] Top1) PerRow(float[] vk, float[] cpu, int vocab)
    {
        int rows = cpu.Length / vocab;
        var kl = new double[rows]; var top = new bool[rows];
        for (int r = 0; r < rows; r++)
        {
            var (_, k, t) = MiMoV2TinyOracleParityTests.Compare(vk.AsSpan(r * vocab, vocab).ToArray(), cpu.AsSpan(r * vocab, vocab).ToArray(), vocab);
            kl[r] = k; top[r] = t >= 1.0;
        }
        return (kl, top);
    }

    /// <summary>
    /// Random-weight MoEs flip a top-k expert choice on near-ties between engines (activation rounding differs), which wrecks that ONE row only.
    /// A systematic defect moves every row, so the check is: median per-row KL tiny and at most ~15 % of the rows flipped.
    /// </summary>
    private static void AssertRowsAgree(float[] vk, float[] cpu, int vocab, double medianLimit = 0.01, double rowLimit = 0.05, double minFraction = 0.84)
    {
        var (rk, _) = PerRow(vk, cpu, vocab);
        var sorted = rk.OrderBy(x => x).ToArray();
        double median = sorted[sorted.Length / 2];
        double good = rk.Count(x => x < rowLimit) / (double)rk.Length;
        Assert.True(median < medianLimit, $"median per-row KL {median}");
        Assert.True(good >= minFraction, $"only {good:P0} of rows agree (KL<{rowLimit})");
    }

    private string Report(string tag, float[] cpu, float[] vk, int vocab)
    {
        var (maxAbs, maxKl, top1) = MiMoV2TinyOracleParityTests.Compare(vk, cpu, vocab);
        var (rk, rt) = PerRow(vk, cpu, vocab);
        var sorted = rk.OrderBy(x => x).ToArray();
        string perRow = string.Join(' ', rk.Select(x => x.ToString("E0")));
        string line = $"[mimo2 vulkan {tag}] rows={cpu.Length / vocab} vs CPU: maxAbs={maxAbs:E2} maxKL={maxKl:E2} medianKL={sorted[sorted.Length / 2]:E2} top1={top1:P0} | perRowKL {perRow}";
        _output.WriteLine(line);
        Console.WriteLine(line);
        return line;
    }

    [SkippableFact]
    public void Q80Banks_Prefill_MatchesCpu()
    {
        var (cpu, vk, vocab) = Run(Q80, Ids16);
        Assert.Equal(cpu.Length, vk.Length);
        Report("Q8_0 banks prefill", cpu, vk, vocab);
        AssertRowsAgree(vk, cpu, vocab);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Q80Banks_Hidden256_MatchCpu(bool decode)
    {
        var o = new SyntheticMiMoV2Gguf.Options { HiddenSize = 256, ExpertFf = 256, DenseFfWidth = 256, GateUpQuant = QuantizationType.Q8_0, DownQuant = QuantizationType.Q8_0, RouterStd = 0.15f };
        var (cpu, vk, vocab) = Run(o, Ids16, vkDecode: decode);
        Report($"Q8_0 banks hidden256 {(decode ? "decode" : "prefill")}", cpu, vk, vocab);
        AssertRowsAgree(vk, cpu, vocab);
    }

    [SkippableFact]
    public void Q80Banks_KvCacheDecode_MatchesCpu()
    {
        var (cpu, vk, vocab) = Run(Q80, Ids16, vkDecode: true);
        Report("Q8_0 banks decode", cpu, vk, vocab);
        AssertRowsAgree(vk, cpu, vocab);
    }

    [SkippableTheory]
    [InlineData(QuantizationType.IQ1_M, QuantizationType.IQ2_XXS, false)]
    [InlineData(QuantizationType.IQ1_M, QuantizationType.IQ2_XS, true)]
    [InlineData(QuantizationType.IQ1_M, QuantizationType.IQ3_S, false)]
    public void IqBanks_MatchCpu(QuantizationType gateUp, QuantizationType down, bool decode)
    {
        var o = new SyntheticMiMoV2Gguf.Options { HiddenSize = 256, ExpertFf = 256, DenseFfWidth = 256, GateUpQuant = gateUp, DownQuant = down, RouterStd = 0.15f };
        var (cpu, vk, vocab) = Run(o, Ids16, vkDecode: decode);
        Report($"{gateUp}/{down} {(decode ? "decode" : "prefill")}", cpu, vk, vocab);
        // The IQ MMVQ / grouped kernels quantise the activation (Q8_1 / F16 inputs) where the CPU path uses Q8_K: noise, not a defect.
        AssertRowsAgree(vk, cpu, vocab);
    }

    [SkippableTheory]
    [InlineData("value-scale")]
    [InlineData("rope-theta-swapped")]
    [InlineData("window")]
    public void Mutant_Of_Vulkan_Config_Is_Detected(string mutant)
    {
        Func<ModelConfig, ModelConfig> m = mutant switch
        {
            "value-scale" => c => c with { AttentionValueScale = null },
            "rope-theta-swapped" => c => c with { RoPEConfig = c.GlobalRoPEConfig, GlobalRoPEConfig = c.RoPEConfig },
            _ => c => c with { PerLayerSlidingWindow = c.PerLayerSlidingWindow!.Select(w => w is null ? (int?)null : 4096).ToArray() },
        };
        var (cpu, vk, vocab) = Run(Q80, Ids16, mutateVk: m);
        Report($"mutant {mutant}", cpu, vk, vocab);
        var (rk, _) = PerRow(vk, cpu, vocab);
        double bad = rk.Count(x => x > 0.02) / (double)rk.Length;
        Assert.True(bad > 0.4, $"mutant '{mutant}' is invisible: only {bad:P0} rows moved");
    }

    // ---- #996: small-row (2..4 rows) forwards --------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// Feeds ids[ctx..ctx+rows) as ONE forward (all-row logits) and as <paramref name="rows"/> sequential 1-row decode steps from the same cache state, under
    /// the given small-row switches. Returns (batched rows, sequential rows) and the dispatch counters of the batched forward.
    /// </summary>
    private static (float[] Batched, float[] Sequential, long Attn, long F32, long Q8, long Q6) SmallRowRun(
        SyntheticMiMoV2Gguf.Options options, int ctx, int rows, int attnMax, int f32Max, int q8Max = 4, int q6Max = 4, int window = 20)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = Path.Combine(Path.GetTempPath(), $"dotllm-mimo2-sr-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticMiMoV2Gguf.Build(options));
        try
        {
            using var gguf = GgufFile.Open(path);
            ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            // The synthetic window (4) leaves every sliding layer's clipped span below the 17 keys a split needs, so widen it: the sink layers must exercise the
            // window-clipped split-KV merge, like the real model's 128.
            cfg = cfg with { PerLayerSlidingWindow = cfg.PerLayerSlidingWindow!.Select(w => w is null ? (int?)null : window).ToArray() };
            using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spvDir);
            vk.TrySetAllRowLogitsLimit(8);
            int[] ids = Enumerable.Range(0, ctx + rows).Select(i => (i * 7 + 3) % cfg.VocabSize).ToArray();
            int[] pos = Enumerable.Range(0, ctx + rows).ToArray();
            using var cache = (VulkanKvCache)vk.CreateKvCache(128);
            using (vk.Forward(ids.AsSpan(0, ctx), pos.AsSpan(0, ctx), deviceId: 0, cache)) { }
            var seq = new List<float>();
            for (int i = 0; i < rows; i++)
            {
                using ITensor t = vk.Forward(ids.AsSpan(ctx + i, 1), pos.AsSpan(ctx + i, 1), deviceId: 0, cache);
                seq.AddRange(Copy(t));
            }
            cache.Rollback(ctx);
            vk.SmallRowAttnMaxRows = attnMax; vk.SmallRowF32MaxRows = f32Max; vk.SmallRowQ8MaxRows = q8Max; vk.SmallRowQ6KMaxRows = q6Max;
            long a0 = vk.SmallRowAttnCount, f0 = vk.SmallRowF32Count, q80 = vk.SmallRowQ8Count, q60 = vk.SmallRowQ6KCount;
            float[] batched;
            using (ITensor t = vk.Forward(ids.AsSpan(ctx, rows), pos.AsSpan(ctx, rows), deviceId: 0, cache))
            {
                Assert.Equal(rows, (int)t.Shape[0]);
                batched = Copy(t);
            }
            return (batched, seq.ToArray(), vk.SmallRowAttnCount - a0, vk.SmallRowF32Count - f0, vk.SmallRowQ8Count - q80, vk.SmallRowQ6KCount - q60);
        }
        finally { try { File.Delete(path); } catch { /* best effort */ } }
    }

    private static double MaxAbs(float[] a, float[] b) { double m = 0; for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i])); return m; }

    /// <summary>
    /// A 2..4 row forward must take the small-row paths (counted), match the same tokens fed one at a time to near-ULP level, and be measurably CLOSER to
    /// the 1-row result than the old path (f16 coopmat flash attention + tiled F32 GEMM): the sensitive control that proves the comparison could disagree.
    /// </summary>
    [SkippableTheory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void SmallRowForward_TakesTheMultiColumnPaths_AndTracksTheOneRowDecode(int rows)
    {
        var o = new SyntheticMiMoV2Gguf.Options { HiddenSize = 256, ExpertFf = 256, DenseFfWidth = 256, GateUpQuant = QuantizationType.Q8_0, DownQuant = QuantizationType.Q8_0, RouterStd = 0.15f };
        const int ctx = 40;   // window 20 < ctx: sliding layers clip to 20 keys (>= 17: they split) and the rows straddle nothing special; global layers see 40+
        var on = SmallRowRun(o, ctx, rows, attnMax: 4, f32Max: 4);
        var off = SmallRowRun(o, ctx, rows, attnMax: 0, f32Max: 0, q8Max: 0, q6Max: 0);
        double dOn = MaxAbs(on.Batched, on.Sequential), dOff = MaxAbs(off.Batched, off.Sequential);
        var noAttn = SmallRowRun(o, ctx, rows, attnMax: 0, f32Max: 4);
        var noF32 = SmallRowRun(o, ctx, rows, attnMax: 4, f32Max: 0);
        _output.WriteLine($"[mimo2 smallrow rows={rows}] attribution: attn off {MaxAbs(noAttn.Batched, noAttn.Sequential):E2}, f32 off {MaxAbs(noF32.Batched, noF32.Sequential):E2}");
        Console.WriteLine($"[mimo2 smallrow rows={rows}] attribution: attn off {MaxAbs(noAttn.Batched, noAttn.Sequential):E2}, f32 off {MaxAbs(noF32.Batched, noF32.Sequential):E2}");
        _output.WriteLine($"[mimo2 smallrow rows={rows}] attn {on.Attn} f32 {on.F32} q8 {on.Q8} | max|batched - sequential|: new {dOn:E2} old {dOff:E2}");
        Console.WriteLine($"[mimo2 smallrow rows={rows}] attn {on.Attn} f32 {on.F32} q8 {on.Q8} | max|batched - sequential|: new {dOn:E2} old {dOff:E2}");
        Assert.Equal(SyntheticMiMoV2Gguf.Layers, on.Attn);   // one rows-kernel dispatch per attention layer (sink layers AND the global layers)
        Assert.True(on.F32 > 0, "F32 multi-column GEMV never ran (router / F32 projections)");
        Assert.Equal(0, off.Attn); Assert.Equal(0, off.F32);
        // 0 at 2 and 3 rows; ~1e-2 at 4 rows (F32 multi NC=4 contracts its FMAs differently from the 1-row GEMV, one near-tie MoE expert choice flips): ULP noise, vs 0.2 for the old path.
        Assert.True(dOn < 5e-2, $"new batched forward off the 1-row decode: {dOn:E2}");
        Assert.True(dOff > 3 * dOn, $"old path not measurably farther from the 1-row decode (new {dOn:E2}, old {dOff:E2}): the comparison cannot disagree");
    }
}
