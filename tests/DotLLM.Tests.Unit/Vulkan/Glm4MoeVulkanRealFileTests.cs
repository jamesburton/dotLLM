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

    private void DumpResidency(VulkanResidencyReport rep)
    {
        _out.WriteLine(rep.Describe());
        foreach (var g in rep.Entries.GroupBy(x => (x.Name.StartsWith("blk.", StringComparison.Ordinal) ? x.Name[(x.Name.IndexOf('.', 4) + 1)..] : x.Name) + " " + x.Source + "->" + x.Device))
            _out.WriteLine($"  RES {g.Key} x{g.Count()} packed={g.Sum(x => x.PackedBytes) >> 20} MiB uploaded={g.Sum(x => x.UploadedBytes) >> 20} MiB");
    }
}
