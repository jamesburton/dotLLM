using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1600: opt-in real-file checks for <c>glm4moe</c> on Vulkan (REAP-82B IQ4_XS, ~44 GiB resident). Skipped unless
/// <c>DOTLLM_GLM4MOE_GGUF</c> points at the file. Needs the Strix:GPU and Strix:MEMORY locks. One model load serves every assertion.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Glm4MoeVulkanRealFileTests
{
    private readonly ITestOutputHelper _out;
    public Glm4MoeVulkanRealFileTests(ITestOutputHelper output) => _out = output;

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    /// <summary>KL(softmax(p) || softmax(q)) in nats.</summary>
    private static double Kl(float[] p, float[] q)
    {
        static double[] LogSoftmax(float[] x)
        {
            double mx = x.Max(), s = 0;
            foreach (float f in x) s += Math.Exp(f - mx);
            double ls = Math.Log(s);
            return x.Select(f => f - mx - ls).ToArray();
        }
        var lp = LogSoftmax(p); var lq = LogSoftmax(q);
        double kl = 0;
        for (int i = 0; i < lp.Length; i++) kl += Math.Exp(lp[i]) * (lp[i] - lq[i]);
        return kl;
    }

    /// <summary>Real wikitext token ids (the S1 KL corpus), from DOTLLM_GLM4MOE_IDS when set; otherwise a fixed in-vocab span.</summary>
    private static int[] LoadIds(int count)
    {
        string? p = Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_IDS");
        if (!string.IsNullOrEmpty(p) && File.Exists(p))
            return File.ReadAllText(p).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Take(count).Select(int.Parse).ToArray();
        return [12516, 5193, 264, 882, 11, 304, 264, 2613, 14117, 553, 279, 9392, 11, 1052, 12157, 264, 7640, 4753, 6941, 3757, 13, 3757, 572, 641];
    }

    private static float[] Step(VulkanTransformerModel m, VulkanKvCache cache, ReadOnlySpan<int> ids, int pos0)
    {
        int[] pos = Enumerable.Range(pos0, ids.Length).ToArray();
        using ITensor t = m.Forward(ids, pos, deviceId: -1, cache);
        return Copy(t);
    }

    [SkippableFact]
    public void RealFile_Packed_Residency_And_Row_Independence()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_GLM4MOE_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        _out.WriteLine("pre-load: " + device.MemorySnapshot());
        VulkanTransformerModel model;
        try { model = VulkanTransformerModel.LoadFromGguf(device, gguf, cfg, spvDir); }
        catch (Exception e)
        {
            _out.WriteLine("LOAD FAILED " + e.GetType().Name + ": " + device.MemorySnapshot());
            var rep = (VulkanResidencyReport?)typeof(VulkanWeights).GetField("_residencyReport",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)?.GetValue(null);
            if (rep is not null) DumpResidency(rep);
            throw;
        }
        using var _ = model;
        _out.WriteLine("post-load: " + device.MemorySnapshot());

        // ---- memory proof: every tensor kept in (or read in place from) its file format; nothing widened ----
        var report = VulkanWeights.LastResidencyReport!;
        DumpResidency(report);
        _out.WriteLine($"upload paths: zero-copy {VulkanWeights.LastUploadZeroCopyMatrices} matrices / {VulkanWeights.LastUploadZeroCopyBytes >> 20} MiB, staged {VulkanWeights.LastUploadStagingMatrices} matrices");
        Assert.Equal(0, report.ExpandedTensorCount);

        // device bytes vs the file's trunk tensors (the NextN block and the host-gathered token table never reach the device)
        long fileTrunkDevice = 0, fileTotal = 0;
        foreach (var kv in gguf.TensorsByName)
        {
            long bytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize((int)kv.Value.Shape[0], kv.Value.QuantizationType)
                         * (kv.Value.Shape.ElementCount / kv.Value.Shape[0]);
            fileTotal += bytes;
            bool nextn = kv.Key.StartsWith($"blk.{cfg.NumLayers}.", StringComparison.Ordinal);
            if (!nextn && kv.Key != "token_embd.weight") fileTrunkDevice += bytes;
        }
        long device1 = 0;
        foreach (var line in device.MemorySnapshot().Split("; "))
            if (line.StartsWith("heap1 ours=", StringComparison.Ordinal)) device1 = long.Parse(line.Split(' ')[1].Split('=')[1]) << 20;
        _out.WriteLine($"device weight heap {device1 >> 20} MiB vs trunk-tensor file bytes {fileTrunkDevice >> 20} MiB (file total {fileTotal >> 20} MiB); ratio {(double)device1 / fileTrunkDevice:F4}");

        // ---- forward: F16 KV, row independence (#1016): a 4-row chunk and four 1-row steps from the same prefix must agree ----
        int[] ids = LoadIds(24);
        const int prefix = 16;
        using var cacheA = model.CreateKvCache(256);
        Assert.Equal(VulkanKvStorage.F16, cacheA.Storage);
        Step(model, cacheA, ids.AsSpan(0, prefix), 0);
        long q51Before = model.Q5_1DenseDispatchCount;
        float[] chunk4 = Step(model, cacheA, ids.AsSpan(prefix, 4), prefix);

        using var cacheB = model.CreateKvCache(256);
        Step(model, cacheB, ids.AsSpan(0, prefix), 0);
        float[] last1 = [];
        for (int i = 0; i < 4; i++) last1 = Step(model, cacheB, ids.AsSpan(prefix + i, 1), prefix + i);

        using var cacheC = model.CreateKvCache(256);
        float[] whole = Step(model, cacheC, ids.AsSpan(0, prefix + 4), 0);   // one 20-row prefill

        double klChunk = Kl(last1, chunk4), klWhole = Kl(last1, whole);
        int a1 = Argmax(last1), a4 = Argmax(chunk4), aw = Argmax(whole);
        _out.WriteLine($"row independence: KL(1-row steps||4-row chunk)={klChunk:E3} KL(1-row steps||20-row prefill)={klWhole:E3} argmax {a1}/{a4}/{aw}; Q5_1 dense dispatches {model.Q5_1DenseDispatchCount}");
        Assert.True(model.Q5_1DenseDispatchCount > q51Before, "the packed Q5_1 shared-down kernels never ran");
        Assert.True(klChunk < 0.02, $"4-row chunk diverges from 1-row steps: KL {klChunk}");
        Assert.True(klWhole < 0.02, $"20-row prefill diverges from 1-row steps: KL {klWhole}");
    }

    private static float[] LastRow(ITensor t, int vocab)
    {
        float[] all = Copy(t);
        return all.Length == vocab ? all : all.AsSpan(all.Length - vocab, vocab).ToArray();
    }

    private static int[] CorpusIds(GgufFile gguf, int need)
    {
        string corpus = Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_CORPUS")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw");
        Skip.If(!File.Exists(corpus), $"corpus not found: {corpus} (set DOTLLM_GLM4MOE_CORPUS)");
        string text = File.ReadAllText(corpus);
        var tk = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] ids = tk.Encode(text.Substring(0, Math.Min(text.Length, need * 6)));
        Assert.True(ids.Length >= need, $"corpus gave {ids.Length} tokens, need {need}");
        return ids.AsSpan(0, need).ToArray();
    }

    private sealed record KlStats(double Mean, double Median, double Max, int Top1Agree, int Steps)
    {
        public static KlStats Of(IReadOnlyList<float[]> p, IReadOnlyList<float[]> q)
        {
            var kls = new double[p.Count];
            int agree = 0;
            for (int i = 0; i < p.Count; i++) { kls[i] = Kl(p[i], q[i]); if (Argmax(p[i]) == Argmax(q[i])) agree++; }
            var sorted = kls.OrderBy(x => x).ToArray();
            return new(kls.Average(), sorted[sorted.Length / 2], sorted[^1], agree, p.Count);
        }
        public override string ToString() => $"mean {Mean:E3} median {Median:E3} max {Max:E3} top1 {Top1Agree}/{Steps}";
    }

    /// <summary>
    /// #1614: the fused single-token MoE decode chain on the real file. Teacher-forced decode steps after a 512- and a 2048-token wikitext prefix,
    /// plus 48 greedy tokens at 512, with the chain ON and OFF in one model load (same kernels, different schedule: the two must agree far inside
    /// the CPU-vs-Vulkan floor), then the dotLLM CPU engine on the same schedule in the same process after the Vulkan model is released.
    /// Every 1-row step of every MoE layer must have taken the fused chain (counter asserted).
    /// </summary>
    [SkippableFact]
    public void RealFile_FusedDecode_Kl_And_Greedy_Vs_Unfused_And_Cpu()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_GLM4MOE_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int ctxA = 512, ctxB = 2048, steps = 32, greedyLen = 48, chunk = 512;
        int cpuStepsB = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_CPU_STEPS_2K"), out int s2) ? s2 : steps;

        using var gguf = GgufFile.Open(path!);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int vocab = cfg.VocabSize;
        int[] ids = CorpusIds(gguf, ctxB + steps + 1);
        int moeLayers = cfg.NumLayers - (cfg.Moe?.MlpOnlyLayers?.Count ?? 0);

        // ---- Vulkan: fused on / off, one load ----
        var vk = new Dictionary<(bool fused, int ctx), List<float[]>>();
        var greedy = new Dictionary<bool, int[]>();
        var fusedLayerSteps = new Dictionary<bool, long>();
        bool saved = VulkanTransformerModel.MoeDecodeFusedEnabled;
        try
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanTransformerModel.LoadFromGguf(device, gguf, cfg, spvDir);
            _out.WriteLine("post-load: " + device.MemorySnapshot());
            foreach (bool fused in new[] { true, false })
            {
                VulkanTransformerModel.MoeDecodeFusedEnabled = fused;
                long c0 = model.GenericMoeDecodeFusedCount;
                foreach (int ctx in new[] { ctxA, ctxB })
                {
                    using var cache = model.CreateKvCache(ctxB + steps + greedyLen + 8);
                    for (int at = 0; at < ctx; at += chunk) Step(model, cache, ids.AsSpan(at, Math.Min(chunk, ctx - at)), at);
                    var rows = new List<float[]>();
                    for (int i = 0; i < steps; i++)
                    {
                        int[] pos = [ctx + i];
                        using ITensor t = model.Forward(ids.AsSpan(ctx + i, 1), pos, deviceId: -1, cache);
                        rows.Add(LastRow(t, vocab));
                    }
                    vk[(fused, ctx)] = rows;
                    if (ctx == ctxA)
                    {
                        var gen = new List<int>();
                        int next = Argmax(rows[^1]);
                        for (int i = 0; i < greedyLen; i++)
                        {
                            gen.Add(next);
                            int[] pos = [ctx + steps + i];
                            using ITensor t = model.Forward([next], pos, deviceId: -1, cache);
                            next = Argmax(LastRow(t, vocab));
                        }
                        greedy[fused] = gen.ToArray();
                    }
                }
                fusedLayerSteps[fused] = model.GenericMoeDecodeFusedCount - c0;
            }
        }
        finally { VulkanTransformerModel.MoeDecodeFusedEnabled = saved; }
        // Every one-row forward of every MoE layer took the fused chain in the ON arm ((2 * steps + greedyLen) forwards), none in the OFF arm.
        _out.WriteLine($"fused layer-steps: on {fusedLayerSteps[true]} (expected {(long)moeLayers * (2 * steps + greedyLen)}), off {fusedLayerSteps[false]}");
        Assert.Equal((long)moeLayers * (2 * steps + greedyLen), fusedLayerSteps[true]);
        Assert.Equal(0L, fusedLayerSteps[false]);

        var klVk512 = KlStats.Of(vk[(true, ctxA)], vk[(false, ctxA)]);
        var klVk2k = KlStats.Of(vk[(true, ctxB)], vk[(false, ctxB)]);
        int greedyAgree = greedy[true].Zip(greedy[false]).Count(p => p.First == p.Second);
        int firstDiv = -1;
        for (int i = 0; i < greedyLen; i++) if (greedy[true][i] != greedy[false][i]) { firstDiv = i; break; }
        _out.WriteLine($"fused vs unfused: KL@512 {klVk512}; KL@2048 {klVk2k}; greedy identity {greedyAgree}/{greedyLen} (first divergence {(firstDiv < 0 ? "none" : firstDiv.ToString())})");

        // Gates on the Vulkan-only part first (also what the mutant runs use: DOTLLM_GLM4MOE_SKIP_CPU=1 stops here).
        // Fused vs unfused is a schedule change over the same math (the quantiser / norm / combine kernels differ, so 1-ULP drift digitises on
        // this 4-bit file): it must sit at the reordering floor measured for V1-vs-V2 kernels (decodekl: mean 1.5e-3, median 2e-4, top-1 98.7%),
        // far below the CPU-vs-Vulkan floor (S1/S2: ~0.03 mean) and two orders below a wrong activation row / expert (A4b mutant: mean 0.54).
        Assert.True(klVk512.Mean < 5e-3 && klVk2k.Mean < 5e-3, $"fused vs unfused KL too large: {klVk512} / {klVk2k}");
        Assert.True(klVk512.Top1Agree >= steps - 3 && klVk2k.Top1Agree >= steps - 3, "fused vs unfused argmax disagreement beyond near-tie steps");
        Assert.True(greedyAgree >= greedyLen * 4 / 5, $"fused vs unfused greedy identity {greedyAgree}/{greedyLen}");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_SKIP_CPU") == "1", "Vulkan-only gates passed; CPU reference skipped on request");

        // ---- CPU reference (Vulkan model released above) ----
        var cpu = new Dictionary<int, List<float[]>>();
        int[] cpuGreedy = [];
        {
            using var model = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.Auto);
            foreach (int ctx in new[] { ctxA, ctxB })
            {
                int n = ctx == ctxA ? steps : cpuStepsB;
                using var cache = new DotLLM.Engine.KvCache.SimpleKvCache(KvGeometry.FromConfig(cfg), ctxB + steps + greedyLen + 8);
                for (int at = 0; at < ctx; at += chunk)
                {
                    int len = Math.Min(chunk, ctx - at);
                    using ITensor t = model.Forward(ids.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), deviceId: -1, cache, lastTokenLogitsOnly: true);
                }
                var rows = new List<float[]>();
                for (int i = 0; i < n; i++)
                {
                    using ITensor t = model.Forward(ids.AsSpan(ctx + i, 1), [ctx + i], deviceId: -1, cache, lastTokenLogitsOnly: true);
                    rows.Add(LastRow(t, vocab));
                }
                cpu[ctx] = rows;
                if (ctx == ctxA)
                {
                    var gen = new List<int>();
                    int next = Argmax(rows[^1]);
                    for (int i = 0; i < greedyLen; i++)
                    {
                        gen.Add(next);
                        using ITensor t = model.Forward([next], [ctx + steps + i], deviceId: -1, cache, lastTokenLogitsOnly: true);
                        next = Argmax(LastRow(t, vocab));
                    }
                    cpuGreedy = gen.ToArray();
                }
            }
        }
        var fusedCpu512 = KlStats.Of(cpu[ctxA], vk[(true, ctxA)]);
        var plainCpu512 = KlStats.Of(cpu[ctxA], vk[(false, ctxA)]);
        var fusedCpu2k = KlStats.Of(cpu[ctxB], vk[(true, ctxB)].Take(cpuStepsB).ToList());
        var plainCpu2k = KlStats.Of(cpu[ctxB], vk[(false, ctxB)].Take(cpuStepsB).ToList());
        int gFusedCpu = greedy[true].Zip(cpuGreedy!).Count(p => p.First == p.Second), gPlainCpu = greedy[false].Zip(cpuGreedy!).Count(p => p.First == p.Second);
        _out.WriteLine($"KL(cpu||fused)@512 {fusedCpu512}; KL(cpu||unfused)@512 {plainCpu512}");
        _out.WriteLine($"KL(cpu||fused)@2048 {fusedCpu2k}; KL(cpu||unfused)@2048 {plainCpu2k}");
        _out.WriteLine($"greedy identity vs CPU: fused {gFusedCpu}/{greedyLen}, unfused {gPlainCpu}/{greedyLen}");

        // Fused is no further from the CPU engine than the unfused chain is (both at the activation-precision floor).
        Assert.True(fusedCpu512.Mean < plainCpu512.Mean + 0.01 && fusedCpu2k.Mean < plainCpu2k.Mean + 0.01, "fused chain drifted from the CPU engine beyond the unfused chain");
        Assert.True(fusedCpu512.Mean < 0.1 && fusedCpu2k.Mean < 0.1, "fused chain vs CPU above the floor class");
    }

    private void DumpResidency(VulkanResidencyReport rep)
    {
        _out.WriteLine(rep.Describe());
        foreach (var g in rep.Entries.GroupBy(x => (x.Name.StartsWith("blk.", StringComparison.Ordinal) ? x.Name[(x.Name.IndexOf('.', 4) + 1)..] : x.Name) + " " + x.Source + "->" + x.Device))
            _out.WriteLine($"  RES {g.Key} x{g.Count()} packed={g.Sum(x => x.PackedBytes) >> 20} MiB uploaded={g.Sum(x => x.UploadedBytes) >> 20} MiB");
    }
}
