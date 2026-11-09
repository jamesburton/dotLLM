using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models;

/// <summary>
/// Vulkan twin of <see cref="TernOverlayRealModelTests"/> (#1706 S3): <c>--tensor-overlay</c> with TERN_B64_F16 on REAL cached models, run on the Vulkan backend and
/// compared with the CPU path on the SAME base + overlay. OLMoE-1B-7B Q4_K_M mixes ternary and original-format experts inside every touched layer (indexed
/// per-expert formats, norm_topk_prob = false); SmolLM2-135M f16 overrides dense FFNs. Controls: the Vulkan output must visibly move because of the overlay
/// (a path that ignored it would sit at the GPU-vs-CPU floor), and the Vulkan/CPU gap with the overlay must stay near that floor and far below the effect.
/// </summary>
[Collection(GpuCollection.Name)]
public sealed class VulkanTernOverlayRealModelTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-ternvk-{Guid.NewGuid():N}");

    public VulkanTernOverlayRealModelTests(ITestOutputHelper output) { _out = output; Directory.CreateDirectory(_dir); }

    public void Dispose() { try { Directory.Delete(_dir, true); } catch { /* best effort */ } }

    private const string Text =
        "The history of the Roman Empire spans many centuries, during which the city grew from a small settlement on the banks of the Tiber into the "
        + "capital of a vast realm. Its legions marched across Europe, and its roads linked provinces from Britain to Syria. Trade flourished along the "
        + "Mediterranean, and aqueducts carried water to cities. In the third century the empire faced civil war, plague and invasion, yet it endured.";

    private sealed record Ov(string Name, int In, int Out, byte[] Data);

    private static Ov Tern(string name, float[] w, int inDim, int outDim)
    {
        var trits = new sbyte[w.Length]; var scales = new float[w.Length / 64];
        for (int b = 0; b < scales.Length; b++)
        {
            double mean = 0;
            for (int j = 0; j < 64; j++) mean += Math.Abs(w[b * 64 + j]);
            float s = (float)(Half)(mean / 64);
            scales[b] = s;
            for (int j = 0; j < 64; j++)
                trits[b * 64 + j] = (sbyte)(s == 0 ? 0 : (int)Math.Clamp(Math.Round(w[b * 64 + j] / s), -1, 1));
        }
        return new Ov(name, inDim, outDim, TernB64F16.Pack(trits, scales));
    }

    private string Write(string baseSha, IEnumerable<Ov> tensors, string tag)
    {
        var w = new GgufWriter();
        w.AddString("general.architecture", "dotllm-overlay");
        w.AddString(TensorOverlay.BaseSha256Key, baseSha);
        foreach (var t in tensors) w.AddTensor(t.Name, [t.In, t.Out], (uint)QuantizationType.TERN_B64_F16, t.Data);
        string path = Path.Combine(_dir, tag + ".gguf");
        File.WriteAllBytes(path, w.Build());
        return path;
    }

    private static float[] Dequant(GgufFile g, string tensor, long elemOffset, int count)
    {
        var d = g.TensorsByName[tensor];
        nint basePtr = g.TensorDataPointer(d);
        long byteOff = d.QuantizationType.ComputeByteCount(elemOffset);
        var f = new float[count];
        Dequantize.ToFloat32(basePtr + (nint)byteOff, count, d.QuantizationType, f);
        return f;
    }

    private static string? SpvDir()
    {
        foreach (string c in new[] { Path.Combine(AppContext.BaseDirectory, "spv"), Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv") })
            if (Directory.Exists(Path.GetFullPath(c)) && Directory.GetFiles(Path.GetFullPath(c), "*.spv").Length > 0) return Path.GetFullPath(c);
        return null;
    }

    private static unsafe float[] LastRow(IModel model, int[] ids, int deviceId)
    {
        using ITensor t = model.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), deviceId);
        int rows = t.Shape[0], vocab = t.Shape[1];
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * vocab, vocab).ToArray();
    }

    private static float[] Run(string basePath, string? overlay, bool vulkan, int[]? ids, out int[] usedIds)
    {
        using var gguf = GgufFile.Open(basePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        usedIds = ids ?? GgufBpeTokenizerFactory.Load(gguf.Metadata).Encode(Text);
        if (overlay is not null) TensorOverlay.Attach(gguf, basePath, overlay, verifyBaseHash: true);
        if (vulkan)
        {
            using var device = VulkanDevice.Create();
            var (model, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, SpvDir()!);
            using (model)
            {
                gguf.Overlay?.VerifyAllApplied();
                return LastRow(model, usedIds, 0);
            }
        }
        using var cpu = ModelLoader.CreateCpuModelFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        gguf.Overlay?.VerifyAllApplied();
        return LastRow(cpu, usedIds, -1);
    }

    private static double Kl(float[] p, float[] q)
    {
        static double[] LogSoftmax(float[] x) { double m = x.Max(); double s = 0; foreach (float v in x) s += Math.Exp(v - m); double lz = m + Math.Log(s); return x.Select(v => v - lz).ToArray(); }
        var lp = LogSoftmax(p); var lq = LogSoftmax(q); double kl = 0;
        for (int i = 0; i < p.Length; i++) kl += Math.Exp(lp[i]) * (lp[i] - lq[i]);
        return kl;
    }

    private void Check(string label, string basePath, string ov)
    {
        var cpuBase = Run(basePath, null, false, null, out int[] ids);
        var cpuTern = Run(basePath, ov, false, ids, out _);
        var vkBase = Run(basePath, null, true, ids, out _);
        var vkTern = Run(basePath, ov, true, ids, out _);
        double floor = Kl(cpuBase, vkBase), gap = Kl(cpuTern, vkTern), effect = Kl(vkBase, vkTern);
        _out.WriteLine($"{label}: KL(cpuBase||vkBase) {floor:E3} (GPU-vs-CPU floor); KL(cpuTern||vkTern) {gap:E3}; KL(vkBase||vkTern) {effect:E3} (overlay effect on Vulkan); KL(cpuBase||cpuTern) {Kl(cpuBase, cpuTern):E3}");
        Assert.True(effect > 20 * floor && effect > 1e-3, $"{label}: overlay had no visible effect on Vulkan (effect {effect:E3}, floor {floor:E3})");
        Assert.True(gap < effect / 10, $"{label}: Vulkan ternary differs from the CPU ternary path by KL {gap:E3} vs overlay effect {effect:E3}");
        Assert.True(gap < Math.Max(10 * floor, 2e-3), $"{label}: Vulkan/CPU gap {gap:E3} exceeds ten times the unmodified-model floor {floor:E3}");
        Assert.Equal(Array.IndexOf(cpuTern, cpuTern.Max()), Array.IndexOf(vkTern, vkTern.Max()));
    }

    [SkippableFact]
    public void Olmoe_PerExpertMixedFormats_VulkanMatchesCpu()
    {
        Skip.If(SpvDir() is null, "no spv dir");
        var fx = TestFixtureResolver.ResolveFile("DOTLLM_OLMOE_GGUF", "bartowski", "OLMoE-1B-7B-0924-Instruct-GGUF", "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("OLMoE-1B-7B Q4_K_M GGUF"));
        string basePath = fx.Path!;
        var tern = new List<Ov>();
        using (var g = GgufFile.Open(basePath))
        {
            var cfg = GgufModelConfigExtractor.Extract(g.Metadata);
            int hidden = cfg.HiddenSize, ff = cfg.Moe!.MoeIntermediateSize;
            foreach (int L in new[] { 2, 9 })
                for (int E = 0; E < 64; E += 7)   // scattered subset; gate/up/down are ternary on different expert sets
                {
                    long per = (long)ff * hidden;
                    tern.Add(Tern($"blk.{L}.ffn_gate.{E}.weight", Dequant(g, $"blk.{L}.ffn_gate_exps.weight", E * per, (int)per), hidden, ff));
                    if (E % 2 == 0) tern.Add(Tern($"blk.{L}.ffn_up.{E}.weight", Dequant(g, $"blk.{L}.ffn_up_exps.weight", E * per, (int)per), hidden, ff));
                    tern.Add(Tern($"blk.{L}.ffn_down.{E}.weight", Dequant(g, $"blk.{L}.ffn_down_exps.weight", E * per, (int)per), ff, hidden));
                }
        }
        Check("OLMoE Q4_K_M mixed per-expert", basePath, Write(TensorOverlay.BaseHash.Compute(basePath), tern, "vk-olmoe"));
    }

    [SkippableFact]
    public void SmolLM2_DenseTernary_VulkanMatchesCpu()
    {
        Skip.If(SpvDir() is null, "no spv dir");
        var fx = TestFixtureResolver.ResolveFile("DOTLLM_SMOLLM2_135M_INSTRUCT_F16_GGUF", "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-f16.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("SmolLM2-135M-Instruct f16 GGUF"));
        string basePath = fx.Path!;
        var tern = new List<Ov>();
        using (var g = GgufFile.Open(basePath))
        {
            var cfg = GgufModelConfigExtractor.Extract(g.Metadata);
            int hidden = cfg.HiddenSize, ff = cfg.IntermediateSize, n = hidden * ff;
            foreach (int L in new[] { 12, 14, 16 })
            {
                tern.Add(Tern($"blk.{L}.ffn_gate.weight", Dequant(g, $"blk.{L}.ffn_gate.weight", 0, n), hidden, ff));
                tern.Add(Tern($"blk.{L}.ffn_up.weight", Dequant(g, $"blk.{L}.ffn_up.weight", 0, n), hidden, ff));
                tern.Add(Tern($"blk.{L}.ffn_down.weight", Dequant(g, $"blk.{L}.ffn_down.weight", 0, n), ff, hidden));
            }
        }
        Check("SmolLM2-135M f16 dense", basePath, Write(TensorOverlay.BaseHash.Compute(basePath), tern, "vk-smol"));
    }
}
