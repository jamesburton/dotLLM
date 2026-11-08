using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1150 acceptance criterion 1: an F32 reference arm for the two dense Q8_0 prefill GEMMs, on REAL Q8_0 tensors of a cached
/// model with REAL activations (token embeddings of natural text through the layer-0 RMSNorm, and the SwiGLU product for
/// <c>ffn_down</c>). The reference is the exact dequantised weight (<c>d*q</c>, representable in F32) times the F32 activation,
/// accumulated in F64. Both Vulkan paths are measured against it:
/// <list type="bullet">
/// <item>dp4a MMQ (the pre-#1150 default): weights exact, activations rounded to Q8_1 per 32 (~2^-8 of the block max);</item>
/// <item>F16-operand / F32-accumulate blocked coopmat (the #1150 default on AMD wave64): <c>d*q</c> and the activation each rounded to F16 (~2^-11 relative).</item>
/// </list>
/// The gate is relative (coopmat at least as close to the reference as MMQ on every tensor) plus an absolute bound. A deliberately
/// wrong kernel, the F16-accumulate control (<see cref="Q8_0GemmCoopmatVariant.F16AccumulateControl"/>), is the RED arm: it has to
/// fail the gate the shipped kernel passes, which is what makes the gate a measurement and not a tautology.
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class VulkanQ8_0PrefillF32ReferenceTests(ITestOutputHelper output)
{
    /// <summary>Token rows per GEMM: ragged against the 128-row coopmat tile so the staged boundary store is on the measured path.</summary>
    private const int Rows = 100;

    /// <summary>Absolute bound on the relative RMS error of the shipped coopmat GEMM. F16 operands give ~4e-4 per product (two 2^-11
    /// roundings); real activations correlate with the weights, so the dot product is larger than a random-sign sum and the measured
    /// values sit at 3-6e-4 (PR body). 2e-3 keeps 3x headroom and still rejects the F16-accumulate control at every K.</summary>
    private const double CoopmatRelRmsBound = 2e-3;

    public static IEnumerable<object[]> Models()
    {
        yield return new object[] { "Llama-3.2-1B-Q8_0", "DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf" };
        yield return new object[] { "LFM3-1.7B-Q8_0", "DOTLLM_LFM3_GGUF", "LiquidAI", "LFM3-1.7B-GGUF", "LFM3-1.7B-Q8_0.gguf" };
    }

    private sealed record Arm(string Name, double RelRms, double MaxAbs);

    [SkippableTheory]
    [MemberData(nameof(Models))]
    public void RealTensors_CoopmatF32Acc_IsCloserToF32ReferenceThanMmq_AndF16AccControlIsNot(
        string label, string envVar, string org, string repo, string file)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var fixture = TestFixtureResolver.ResolveFile(envVar, org, repo, file);
        Skip.If(!fixture.Found, fixture.SkipMessage(label));

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "no VK_KHR_cooperative_matrix");
        Skip.IfNot(device.HasIntegerDotProduct, "no integer dot product: the MMQ arm cannot run");
        var shipped = Q8_0GemmCoopmatVariant.SelectFor(device, spvDir);
        Skip.IfNot(shipped == Q8_0GemmCoopmatVariant.Blocked128x128x4,
            $"the device selects {shipped.SpvFileName}, not the blocked wave64 tile #1150 gates on (SubgroupSize={device.SubgroupSize})");

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int hidden = config.HiddenSize, inter = config.IntermediateSize;
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] tokens = Q8_0PrefillTestText.Tokens(tokenizer, Rows);

        // Real activations: the layer-0 block input = RMSNorm(embedding row) with the layer's own norm weight.
        float[] x = EmbeddingRows(gguf, tokens, hidden);
        float[] normW = F32Vector(gguf, "blk.0.attn_norm.weight", hidden) ?? F32Vector(gguf, "blk.0.ffn_norm.weight", hidden)
            ?? throw new InvalidOperationException("no layer-0 norm weight");
        for (int t = 0; t < Rows; t++)
            RmsNorm.Execute(x.AsSpan(t * hidden, hidden), normW, config.NormEpsilon, x.AsSpan(t * hidden, hidden));

        // Every Q8_0 projection of layer 0 that reads the block input, plus ffn_down on the real SwiGLU product.
        var hiddenIn = gguf.Tensors.Where(t => t.Name.StartsWith("blk.0.", StringComparison.Ordinal) && t.QuantizationType == QuantizationType.Q8_0
                                               && t.Shape.Rank == 2 && t.Shape[0] == hidden && !t.Name.Contains("ffn_down")).ToList();
        Assert.NotEmpty(hiddenIn);
        var gate = gguf.TensorsByName.GetValueOrDefault("blk.0.ffn_gate.weight");
        var up = gguf.TensorsByName.GetValueOrDefault("blk.0.ffn_up.weight");
        var down = gguf.TensorsByName.GetValueOrDefault("blk.0.ffn_down.weight");

        using var mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir) ?? throw new InvalidOperationException("matmul_q8_0_mmq.spv missing");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new InvalidOperationException("quantize_q8_1_rows.spv missing");
        using var coopmat = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir, shipped);
        var controlVariant = Q8_0GemmCoopmatVariant.F16AccumulateControl;
        Skip.IfNot(controlVariant.IsSupportedOn(device, spvDir), $"{controlVariant.SpvFileName} not compiled: the RED arm cannot run, so the gate is unproven here");
        using var control = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir, controlVariant);

        var results = new List<(string Tensor, int M, int K, Arm Mmq, Arm Coopmat, Arm Control, Arm CpuW8A8)>();
        foreach (var w in hiddenIn)
        {
            var (arms, _) = Measure(device, gguf, w, x, hidden, mmq, quant, coopmat, control);
            results.Add((w.Name, w.Shape[1], hidden, arms[0], arms[1], arms[2], arms[3]));
        }
        if (gate is { } gg && up is { } uu && down is { } dd && gg.Shape[1] == inter && uu.Shape[1] == inter && dd.Shape[0] == inter)
        {
            // SwiGLU(gate x) * (up x), from the F64 reference outputs: the real ffn_down input of layer 0.
            double[] gOut = Reference(gguf, gg, x, hidden), uOut = Reference(gguf, uu, x, hidden);
            float[] h = new float[Rows * inter];
            for (int i = 0; i < h.Length; i++) h[i] = (float)(Silu(gOut[i]) * uOut[i]);
            var (arms, _) = Measure(device, gguf, dd, h, inter, mmq, quant, coopmat, control);
            results.Add((dd.Name, dd.Shape[1], inter, arms[0], arms[1], arms[2], arms[3]));
        }

        output.WriteLine($"[{label}] {Rows} real rows; relative RMS error vs F64(dequant W x F32 activation), max abs in brackets");
        output.WriteLine($"{"tensor",-28} {"M",6} {"K",6} | {"mmq",-22} {"coopmat F32-acc",-22} {"F16-acc control",-22} {"cpu W8A8",-22}");
        foreach (var r in results)
            output.WriteLine($"{r.Tensor,-28} {r.M,6} {r.K,6} | {Fmt(r.Mmq),-22} {Fmt(r.Coopmat),-22} {Fmt(r.Control),-22} {Fmt(r.CpuW8A8),-22}");

        foreach (var r in results)
        {
            Assert.True(r.Coopmat.RelRms <= r.Mmq.RelRms,
                $"{label} {r.Tensor}: coopmat relRMS {r.Coopmat.RelRms:E3} is worse than MMQ {r.Mmq.RelRms:E3}");
            Assert.True(r.Coopmat.RelRms <= CoopmatRelRmsBound,
                $"{label} {r.Tensor}: coopmat relRMS {r.Coopmat.RelRms:E3} > {CoopmatRelRmsBound:E1}");
        }
        // RED arm: the F16 accumulator must be rejected by the same two assertions on at least every K >= hidden tensor.
        int controlRejected = results.Count(r => r.Control.RelRms > CoopmatRelRmsBound || r.Control.RelRms > r.Mmq.RelRms);
        Assert.True(controlRejected == results.Count,
            $"{label}: the F16-accumulate control passed the gate on {results.Count - controlRejected}/{results.Count} tensors; the gate cannot distinguish accumulator precision here");
        Assert.All(results, r => Assert.True(r.Control.RelRms > 2 * r.Coopmat.RelRms,
            $"{label} {r.Tensor}: control {r.Control.RelRms:E3} is not clearly worse than the shipped kernel {r.Coopmat.RelRms:E3}"));
    }

    private static double Silu(double v) => v / (1.0 + Math.Exp(-v));
    private static string Fmt(Arm a) => $"{a.RelRms:E2} [{a.MaxAbs:E1}]";

    /// <summary>Runs the four arms on one real tensor and returns their errors plus the F64 reference output ([Rows, M]).</summary>
    private static (Arm[] Arms, double[] Reference) Measure(VulkanDevice device, GgufFile gguf, GgufTensorDescriptor w, float[] x, int k,
        MatMulQ8_0MmqKernel mmq, QuantizeQ8_1RowsKernel quant, MatMulQ8_0GemmCoopmatKernel coopmat, MatMulQ8_0GemmCoopmatKernel control)
    {
        int m = w.Shape[1];
        Assert.Equal(k, w.Shape[0]);
        long rowBytes = (long)(k / 32) * 34;
        long wBytes = m * rowBytes;
        byte* wPtr = (byte*)gguf.TensorDataPointer(w);
        double[] reference = Reference(gguf, w, x, k);

        using var bufW = device.Allocate((wBytes + 3) & ~3L);
        using var bufX = device.Allocate((long)Rows * k * sizeof(float));
        using var bufC = device.Allocate((long)Rows * m * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(Rows, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(Rows, k));
        device.Upload(new ReadOnlySpan<byte>(wPtr, (int)wBytes), bufW);
        device.Upload(x, bufX);
        var outBuf = new float[Rows * m];
        var zeros = new float[Rows * m];

        // MMQ: quantize rows + integer-dot GEMM, exactly the production pair. The output is zeroed before every arm so an
        // unwritten row (a boundary-store bug) reads as a gross error instead of inheriting the previous arm's result.
        device.Upload(zeros, bufC);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, n: Rows, k: k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m: m, k: k, n: Rows);
            ctx.SubmitAndWait();
        }
        device.Download(bufC, outBuf);
        var armMmq = Compare("mmq", outBuf, reference);

        device.Upload(zeros, bufC);
        coopmat.Launch(bufW, bufX, bufC, m, k, Rows);
        device.Download(bufC, outBuf);
        var armCm = Compare("coopmat", outBuf, reference);

        device.Upload(zeros, bufC);
        control.Launch(bufW, bufX, bufC, m, k, Rows);
        device.Download(bufC, outBuf);
        var armCtl = Compare("f16acc-control", outBuf, reference);

        // CPU W8A8 (the CPU oracle's own arithmetic), for context only.
        fixed (float* xp = x) fixed (float* cp = outBuf)
            MatMul.GemmQ8_0(wPtr, xp, cp, m, k, Rows);
        var armCpu = Compare("cpu-w8a8", outBuf, reference);
        return (new[] { armMmq, armCm, armCtl, armCpu }, reference);
    }

    private static Arm Compare(string name, float[] actual, double[] reference)
    {
        double num = 0, den = 0, maxAbs = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            num += d * d; den += reference[i] * reference[i];
            if (Math.Abs(d) > maxAbs) maxAbs = Math.Abs(d);
        }
        return new Arm(name, Math.Sqrt(num / Math.Max(den, 1e-300)), maxAbs);
    }

    /// <summary>F64 reference: exact dequantised Q8_0 weights times the F32 activations, [Rows, M].</summary>
    private static double[] Reference(GgufFile gguf, GgufTensorDescriptor w, float[] x, int k)
    {
        int m = w.Shape[1];
        long rowBytes = (long)(k / 32) * 34;
        byte* wPtr = (byte*)gguf.TensorDataPointer(w);
        var result = new double[Rows * m];
        Parallel.For(0, m, row =>
        {
            float* wf = stackalloc float[k];
            KvQuantize.Q8_0ToF32(wPtr + row * rowBytes, wf, k);
            for (int t = 0; t < Rows; t++)
            {
                double s = 0;
                int xo = t * k;
                for (int i = 0; i < k; i++) s += (double)wf[i] * x[xo + i];
                result[t * m + row] = s;
            }
        });
        return result;
    }

    private static float[] EmbeddingRows(GgufFile gguf, int[] tokens, int hidden)
    {
        var emb = gguf.TensorsByName["token_embd.weight"];
        Assert.Equal(hidden, emb.Shape[0]);
        byte* basePtr = (byte*)gguf.TensorDataPointer(emb);
        var x = new float[tokens.Length * hidden];
        for (int t = 0; t < tokens.Length; t++)
        {
            Span<float> dst = x.AsSpan(t * hidden, hidden);
            switch (emb.QuantizationType)
            {
                case QuantizationType.Q8_0:
                    fixed (float* dp = dst) KvQuantize.Q8_0ToF32(basePtr + (long)tokens[t] * (hidden / 32) * 34, dp, hidden);
                    break;
                case QuantizationType.F32:
                    new ReadOnlySpan<float>(basePtr + (long)tokens[t] * hidden * 4, hidden).CopyTo(dst);
                    break;
                case QuantizationType.F16:
                    var h = new ReadOnlySpan<Half>(basePtr + (long)tokens[t] * hidden * 2, hidden);
                    for (int i = 0; i < hidden; i++) dst[i] = (float)h[i];
                    break;
                default:
                    throw new InvalidOperationException($"token_embd is {emb.QuantizationType}; this test reads Q8_0/F16/F32 tables");
            }
        }
        return x;
    }

    private static float[]? F32Vector(GgufFile gguf, string name, int n)
    {
        if (!gguf.TensorsByName.TryGetValue(name, out var t) || t.QuantizationType != QuantizationType.F32 || t.Shape[0] != n) return null;
        return new ReadOnlySpan<float>((void*)gguf.TensorDataPointer(t), n).ToArray();
    }
}

/// <summary>Natural text for the #1150 tests: the wikitext-2 test corpus when the generated fixture is present, else a built-in paragraph tiled to length.</summary>
internal static class Q8_0PrefillTestText
{
    private const string Fallback =
        "The castle stood on a low hill above the river, and for three centuries its garrison had watched the valley road. " +
        "Merchants paid the toll at the lower gate, soldiers drilled in the outer yard, and the chapel bell rang the hours " +
        "for the town below. When the railway came in 1847 the toll lapsed, the garrison left, and the county turned the " +
        "keep into a records office whose clerks complained about the damp. A survey in 1902 counted 412 charters, nine " +
        "account rolls and a single illuminated psalter, which was sold to pay for a new roof. ";

    public static string Text()
    {
        string corpus = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw");
        if (File.Exists(corpus))
        {
            string s = File.ReadAllText(corpus);
            // Skip the leading blank lines/headings so the window starts in prose.
            int start = Math.Min(s.Length / 4, 60_000);
            return s.Substring(start, Math.Min(s.Length - start, 80_000));
        }
        return string.Concat(Enumerable.Repeat(Fallback, 40));
    }

    /// <summary>Exactly <paramref name="count"/> token ids of natural text.</summary>
    public static int[] Tokens(DotLLM.Tokenizers.ITokenizer tokenizer, int count)
    {
        int[] ids = tokenizer.Encode(Text());
        if (ids.Length < count) throw new InvalidOperationException($"only {ids.Length} tokens of text for {count}");
        return ids.Take(count).ToArray();
    }
}
