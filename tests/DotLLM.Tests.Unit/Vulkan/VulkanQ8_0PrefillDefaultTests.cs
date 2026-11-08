using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1150 at model level on the dense Vulkan host (Llama-3.2-1B Q8_0): the two prefill GEMMs (dp4a MMQ vs F32-accumulate
/// coopmat) on one real 512-token window, each against the CPU oracle (W8A8) and, when <c>DOTLLM_LLAMA32_1B_F32_GGUF</c> names an
/// F32-decoded copy of the same file (<c>scripts/make_f32_decoded_gguf.py</c>), against a true F32 forward. Reports per-row KL,
/// top-1 agreement and the greedy continuation of both arms (identity, not "noise": on a quantised model a reduction-order
/// difference becomes different text). Also the path proof: the device default must actually dispatch the kernel it claims to
/// (<see cref="VulkanTransformerModel.Q8_0PrefillCoopmatDispatchCount"/>), so a gate that silently falls back cannot pass.
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class VulkanQ8_0PrefillDefaultTests(ITestOutputHelper output)
{
    private const int Window = 512;
    private const int GreedySteps = 32;

    private static string SpvDir()
    {
        string dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "spv"));
        if (!Directory.Exists(dir) || Directory.GetFiles(dir, "*.spv").Length == 0)
            dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));
        return dir;
    }

    private sealed record Run(float[] Logits, int[] Greedy, long CoopmatDispatches, long MmqDispatches);

    [SkippableFact]
    public void Llama32_1B_Q8_0_CoopmatDefault_PathProof_KL_AndGreedyIdentity()
    {
        var fixture = TestFixtureResolver.ResolveFile("DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Llama-3.2-1B Q8_0 GGUF"));
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int vocab = config.VocabSize;
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] tokens = Q8_0PrefillTestText.Tokens(tokenizer, Window);
        int[] positions = Enumerable.Range(0, Window).ToArray();

        using var device = VulkanDevice.Create();
        var (vkModel, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, SpvDir());
        using var vk = (IDisposable)vkModel;
        var dense = Assert.IsType<VulkanTransformerModel>(vkModel);
        Assert.True(dense.TrySetAllRowLogitsLimit(Window), "the Vulkan host cannot return every row's logits for the window");
        // The expected default is derived from the DEVICE, not from the model's own gate property: a test that reads the gate it is
        // checking follows any mutation of it (the first version of this test passed with the gate hard-wired to MMQ).
        bool expectCoopmat = device.VendorId == 0x1002 && device.HasCooperativeMatrix && device.SubgroupSize == 64
            && DotLLM.Vulkan.Kernels.Q8_0GemmCoopmatVariant.Blocked128x128x4.IsSupportedOn(device, SpvDir())
            && Environment.GetEnvironmentVariable(VulkanTransformerModel.Q8_0PrefillEnvVar) is null;
        output.WriteLine($"device {device.DeviceName} vendor 0x{device.VendorId:X} subgroup {device.SubgroupSize}; expect coopmat default={expectCoopmat}; model says {dense.Q8_0CoopmatIsDeviceDefault}; min tokens {VulkanTransformerModel.Q8_0CoopmatPreferSeqLen}");
        Assert.Equal(expectCoopmat, dense.Q8_0CoopmatIsDeviceDefault);
        Assert.True(Window >= VulkanTransformerModel.Q8_0CoopmatPreferSeqLen, "the window must be above the coopmat threshold for Auto to engage");

        Run RunVulkan(VulkanTransformerModel.Q8_0PrefillKernel mode)
        {
            dense.Q8_0Prefill = mode;
            long cm0 = dense.Q8_0PrefillCoopmatDispatchCount, mmq0 = dense.Q8_0PrefillMmqDispatchCount;
            vkModel.ResetSequenceState();
            using var full = dense.Forward(tokens, positions, deviceId: -1);   // every row's logits (the cacheless forward)
            long cm = dense.Q8_0PrefillCoopmatDispatchCount - cm0, mmq = dense.Q8_0PrefillMmqDispatchCount - mmq0;
            var logits = AllRows(full, vocab, Window);
            vkModel.ResetSequenceState();
            using var kv = kvFactory(Window + GreedySteps + 8);
            using var last = Step(dense, tokens, positions, kv, lastOnly: true);
            var greedy = Greedy(dense, kv, LastRow(last, vocab), Window, vocab);
            return new Run(logits, greedy, cm, mmq);
        }

        // Path proof first: forced arms dispatch only their own kernel; Auto dispatches what the device gate says.
        var mmqRun = RunVulkan(VulkanTransformerModel.Q8_0PrefillKernel.Mmq);
        var cmRun = RunVulkan(VulkanTransformerModel.Q8_0PrefillKernel.Coopmat);
        var autoRun = RunVulkan(VulkanTransformerModel.Q8_0PrefillKernel.Auto);
        output.WriteLine($"dispatches (coopmat/mmq) over a {Window}-row prefill: mmq arm {mmqRun.CoopmatDispatches}/{mmqRun.MmqDispatches}, coopmat arm {cmRun.CoopmatDispatches}/{cmRun.MmqDispatches}, auto {autoRun.CoopmatDispatches}/{autoRun.MmqDispatches}");
        Assert.True(mmqRun.MmqDispatches > 0 && mmqRun.CoopmatDispatches == 0, "forced MMQ arm did not run only MMQ");
        Assert.True(cmRun.CoopmatDispatches > 0 && cmRun.MmqDispatches == 0, "forced coopmat arm did not run only the coopmat GEMM");
        Assert.True(cmRun.CoopmatDispatches >= config.NumLayers * 7, "fewer coopmat dispatches than q, k, v, o, gate, up, down per layer");   // +1 when the Q8_0 lm_head scores every row
        if (expectCoopmat)
        {
            Assert.True(autoRun.CoopmatDispatches == cmRun.CoopmatDispatches && autoRun.MmqDispatches == 0,
                $"Auto on a qualifying device must take the coopmat path for every Q8_0 projection: got {autoRun.CoopmatDispatches}/{autoRun.MmqDispatches}");
            AssertSame(cmRun.Logits, autoRun.Logits, "auto vs forced coopmat");
        }
        else
        {
            Assert.True(autoRun.MmqDispatches == mmqRun.MmqDispatches && autoRun.CoopmatDispatches == 0,
                $"Auto on a non-qualifying device must stay on MMQ: got {autoRun.CoopmatDispatches}/{autoRun.MmqDispatches}");
            AssertSame(mmqRun.Logits, autoRun.Logits, "auto vs forced mmq");
        }

        // CPU oracle (W8A8 projections) on the same window + greedy continuation.
        using var cpuModel = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        float[] cpuLogits; int[] cpuGreedy;
        using (var full = cpuModel.Forward(tokens, positions, deviceId: -1))
            cpuLogits = AllRows(full, vocab, Window);
        using (var ckv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, Window + GreedySteps + 8))
        {
            using var last = Step(cpuModel, tokens, positions, ckv, lastOnly: true);
            cpuGreedy = Greedy(cpuModel, ckv, LastRow(last, vocab), Window, vocab);
        }

        // Optional true-F32 reference: the same weights dequantised to F32 by gguf-py, F32 activations on the CPU.
        float[]? f32Logits = null;
        string? f32Path = Environment.GetEnvironmentVariable("DOTLLM_LLAMA32_1B_F32_GGUF");
        if (f32Path is not null && File.Exists(f32Path))
        {
            using var g32 = GgufFile.Open(f32Path);
            var c32 = GgufModelConfigExtractor.Extract(g32.Metadata);
            using var m32 = ModelLoader.CreateCpuModelFromGguf(g32, c32);
            using var r = m32.Forward(tokens, positions, deviceId: -1);
            f32Logits = AllRows(r, vocab, Window);
        }

        Report("KL(cpu||mmq)", cpuLogits, mmqRun.Logits, vocab);
        Report("KL(cpu||coopmat)", cpuLogits, cmRun.Logits, vocab);
        var ab = Report("KL(mmq||coopmat)", mmqRun.Logits, cmRun.Logits, vocab);
        if (f32Logits is not null)
        {
            var fm = Report("KL(f32||mmq)", f32Logits, mmqRun.Logits, vocab);
            var fc = Report("KL(f32||coopmat)", f32Logits, cmRun.Logits, vocab);
            Report("KL(f32||cpu W8A8)", f32Logits, cpuLogits, vocab);
            // #1150 acceptance: coopmat at least as accurate as MMQ against the F32 reference (median over the window).
            Assert.True(fc.Median <= fm.Median, $"coopmat median KL vs F32 {fc.Median:E3} is worse than MMQ {fm.Median:E3}");
            Assert.True(fc.Top1 >= fm.Top1 - 0.01, $"coopmat top-1 vs F32 {fc.Top1:P2} is worse than MMQ {fm.Top1:P2}");
        }
        else
        {
            output.WriteLine("F32 reference arm skipped (set DOTLLM_LLAMA32_1B_F32_GGUF to an F32-decoded copy; see scripts/make_f32_decoded_gguf.py)");
        }

        string Text(int[] ids) => tokenizer.Decode(ids).Replace("\n", "\\n");
        int same = CountPrefix(mmqRun.Greedy, cmRun.Greedy), sameCpuMmq = CountPrefix(cpuGreedy, mmqRun.Greedy), sameCpuCm = CountPrefix(cpuGreedy, cmRun.Greedy);
        output.WriteLine($"greedy {GreedySteps} steps after the {Window}-token prefill: mmq==coopmat for {same}/{GreedySteps} (identical prefix), cpu==mmq {sameCpuMmq}, cpu==coopmat {sameCpuCm}");
        output.WriteLine($"  cpu    : {Text(cpuGreedy)}");
        output.WriteLine($"  mmq    : {Text(mmqRun.Greedy)}");
        output.WriteLine($"  coopmat: {Text(cmRun.Greedy)}");

        // Both arms must sit within the summation-order / activation-precision floor of each other and of the CPU oracle; the per-tensor
        // F32 gate is VulkanQ8_0PrefillF32ReferenceTests. The bounds are 3x the measured values (PR body).
        Assert.True(ab.Median <= 2e-3, $"median KL(mmq||coopmat) {ab.Median:E3} is above the inter-path floor");
        Assert.True(ab.Top1 >= 0.97, $"top-1 agreement mmq vs coopmat {ab.Top1:P2} < 97%");
    }

    /// <summary>The startup override spelling (<see cref="VulkanTransformerModel.Q8_0PrefillEnvVar"/>); anything unrecognised is Auto. CPU-only.</summary>
    [Theory]
    [InlineData(null, VulkanTransformerModel.Q8_0PrefillKernel.Auto)]
    [InlineData("auto", VulkanTransformerModel.Q8_0PrefillKernel.Auto)]
    [InlineData("", VulkanTransformerModel.Q8_0PrefillKernel.Auto)]
    [InlineData("nonsense", VulkanTransformerModel.Q8_0PrefillKernel.Auto)]
    [InlineData("mmq", VulkanTransformerModel.Q8_0PrefillKernel.Mmq)]
    [InlineData(" MMQ ", VulkanTransformerModel.Q8_0PrefillKernel.Mmq)]
    [InlineData("0", VulkanTransformerModel.Q8_0PrefillKernel.Mmq)]
    [InlineData("coopmat", VulkanTransformerModel.Q8_0PrefillKernel.Coopmat)]
    [InlineData("cm", VulkanTransformerModel.Q8_0PrefillKernel.Coopmat)]
    [InlineData("1", VulkanTransformerModel.Q8_0PrefillKernel.Coopmat)]
    public void Q8_0PrefillEnv_Parses(string? value, VulkanTransformerModel.Q8_0PrefillKernel expected)
        => Assert.Equal(expected, VulkanTransformerModel.ParseQ8_0PrefillEnv(value));

    private sealed record KlStats(double Median, double P99, double Max, double Top1);

    private KlStats Report(string what, float[] p, float[] q, int vocab)
    {
        int rows = p.Length / vocab;
        var kl = new double[rows];
        int agree = 0;
        var lp = new double[vocab]; var lq = new double[vocab];
        for (int r = 0; r < rows; r++)
        {
            LogSoftmax(p.AsSpan(r * vocab, vocab), lp); LogSoftmax(q.AsSpan(r * vocab, vocab), lq);
            double s = 0; int ap = 0, aq = 0;
            for (int i = 0; i < vocab; i++)
            {
                s += Math.Exp(lp[i]) * (lp[i] - lq[i]);
                if (lp[i] > lp[ap]) ap = i;
                if (lq[i] > lq[aq]) aq = i;
            }
            kl[r] = s; if (ap == aq) agree++;
        }
        Array.Sort(kl);
        var st = new KlStats(kl[rows / 2], kl[(int)(rows * 0.99)], kl[^1], agree / (double)rows);
        output.WriteLine($"{what,-22}: median {st.Median:E3}  p99 {st.P99:E3}  max {st.Max:E3}  top-1 agree {st.Top1:P2} over {rows} rows");
        return st;
    }

    private static void LogSoftmax(ReadOnlySpan<float> x, double[] dst)
    {
        double m = double.NegativeInfinity; foreach (float v in x) if (v > m) m = v;
        double z = 0; foreach (float v in x) z += Math.Exp(v - m);
        double lz = Math.Log(z) + m;
        for (int i = 0; i < x.Length; i++) dst[i] = x[i] - lz;
    }

    private static void AssertSame(float[] a, float[] b, string what)
    {
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) Assert.Fail($"{what}: logits differ at {i} ({a[i]} vs {b[i]}): the same kernel must give the same bits");
    }

    private static int CountPrefix(int[] a, int[] b) { int n = 0; while (n < a.Length && n < b.Length && a[n] == b[n]) n++; return n; }

    private static ITensor Step(IModel model, ReadOnlyMemory<int> tokens, ReadOnlyMemory<int> positions, IKvCache kv, bool lastOnly = false)
        => model.ForwardBatch([new SequenceForwardRequest { TokenIds = tokens, Positions = positions, KvCache = kv, LastTokenLogitsOnly = lastOnly }], -1)[0];

    private static float[] AllRows(ITensor t, int vocab, int expectedRows)
    {
        Assert.Equal(expectedRows, t.Shape[0]);
        Assert.Equal(vocab, t.Shape[1]);
        return new ReadOnlySpan<float>((float*)t.DataPointer, expectedRows * vocab).ToArray();
    }

    private static float[] LastRow(ITensor t, int vocab)
    {
        Assert.Equal(vocab, t.Shape[1]);
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)(t.Shape[0] - 1) * vocab, vocab).ToArray();
    }

    /// <summary>Greedy continuation through the KV cache (1-row forwards: identical kernels on both arms, so only the prefill differs).</summary>
    private static int[] Greedy(IModel model, IKvCache kv, float[] lastPrefillRow, int start, int vocab)
    {
        var outIds = new int[GreedySteps];
        int next = Argmax(lastPrefillRow);
        for (int s = 0; s < GreedySteps; s++)
        {
            outIds[s] = next;
            using var r = Step(model, new[] { next }, new[] { start + s }, kv, lastOnly: true);
            next = Argmax(new ReadOnlySpan<float>((float*)r.DataPointer + (long)(r.Shape[0] - 1) * vocab, vocab));
        }
        return outIds;
    }

    private static int Argmax(ReadOnlySpan<float> row) { int a = 0; for (int i = 1; i < row.Length; i++) if (row[i] > row[a]) a = i; return a; }
}
