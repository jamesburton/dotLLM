using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan.Ternary;

/// <summary>
/// End to end (#1618): a ternary adapter attached to the Vulkan transformer changes logits like the CPU path
/// (KL vs CPU), and detaching restores the pre-attach logits bit for bit. Covers a tiny synthetic Llama (D=64) and the real
/// SmolLM-135M Q8_0 GGUF from the HF cache (D=576, 30 layers) when present.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTernaryAdapterEndToEndTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-tern-{Guid.NewGuid():N}");
    public VulkanTernaryAdapterEndToEndTests(ITestOutputHelper output) { _out = output; Directory.CreateDirectory(_scratch); }
    public void Dispose() { try { Directory.Delete(_scratch, true); } catch { } }

    private static unsafe float[] Copy(ITensor t)
    {
        int n = t.Shape[0] * t.Shape[1];
        var a = new float[n];
        new ReadOnlySpan<float>((void*)t.DataPointer, n).CopyTo(a);
        return a;
    }

    private static double Kl(float[] p, float[] q)
    {
        double zp = LogSumExp(p), zq = LogSumExp(q), kl = 0;
        for (int i = 0; i < p.Length; i++) { double lp = p[i] - zp, lq = q[i] - zq; kl += Math.Exp(lp) * (lp - lq); }
        return kl;
    }

    private static double LogSumExp(float[] x)
    {
        double m = x.Max(), s = 0;
        foreach (float v in x) s += Math.Exp(v - m);
        return m + Math.Log(s);
    }

    private static void AssertBits(float[] a, float[] b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]), $"{what}: logit {i}: {a[i]:R} vs {b[i]:R}");
    }

    private sealed record Models(Func<int[], int[], float[]> Cpu, Func<int[], int[], float[]> Vk, Action<IVulkanResidualModifier?> SetVk, Action<TernaryResidualHook?> SetCpu, int Hidden, int Layers, int Vocab);

    private void RunScenario(Models m, int layer, float alpha, string label)
    {
        var pkg = VulkanTernaryAdapterKernelTests.Synth(8, m.Hidden, 321, alpha, 0f, layer);
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        using var dev = VulkanDevice.Create();
        foreach (var tokens in new[] { new[] { 1, 2, 3, 4, 5, 6 }, new[] { 7 } })
        {
            int[] pos = Enumerable.Range(0, tokens.Length).ToArray();
            string tag = $"{label} n={tokens.Length}";
            // Vulkan: base, attached, detached.
            var vkBase = m.Vk(tokens, pos);
            using var mod = VulkanTernaryResidualModifier.Create(dev, spv, [pkg]);
            mod.ValidateFor(m.Hidden, m.Layers);
            m.SetVk(mod);
            var vkWith = m.Vk(tokens, pos);
            m.SetVk(null);
            var vkDetached = m.Vk(tokens, pos);
            AssertBits(vkBase, vkDetached, tag + " detached == before");
            double shift = Kl(vkBase, vkWith);
            Assert.True(vkWith.All(float.IsFinite));
            Assert.True(shift > 1e-4, $"{tag}: adapter changed the Vulkan logits by KL only {shift:E3}; hook not firing?");

            // CPU: base, with.
            var cpuBase = m.Cpu(tokens, pos);
            using var hook = new TernaryResidualHook(m.Hidden);
            hook.Add(pkg);
            m.SetCpu(hook);
            var cpuWith = m.Cpu(tokens, pos);
            m.SetCpu(null);

            double noise = Kl(cpuBase, vkBase), klAttached = Kl(cpuWith, vkWith), klEffect = Kl(cpuBase, cpuWith);
            _out.WriteLine($"{tag}: KL(cpu_base||vk_base)={noise:E3}  KL(cpu_with||vk_with)={klAttached:E3}  adapter effect KL(cpu_base||cpu_with)={klEffect:E3}  vk effect={shift:E3}");
            // The control arm is sensitive: the adapter effect dwarfs backend noise, and attached parity stays at the same order as base parity.
            Assert.True(klEffect > 20 * Math.Max(noise, 1e-9), $"{tag}: adapter effect {klEffect:E3} is not >> backend noise {noise:E3}");
            Assert.True(klAttached <= Math.Max(10 * noise, 1e-4), $"{tag}: attached KL {klAttached:E3} vs base KL {noise:E3}");
        }
    }

    // ------------------------------------------------------------------ synthetic Llama, D = 64 ----

    [SkippableFact]
    public void Synthetic_D64_AdapterAttach_MatchesCpu_DetachIsBitIdentical()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        const int H = 64, NH = 4, HD = 16, I = 128, V = 32, L = 3;
        string path = Path.Combine(_scratch, "tiny.safetensors");
        var rng = new Random(42);
        float[] R(int n, float s) { var v = new float[n]; for (int i = 0; i < n; i++) v[i] = (float)((rng.NextDouble() * 2 - 1) * s); return v; }
        float[] Ones(int n) => Enumerable.Repeat(1f, n).ToArray();
        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [V, H], R(V * H, 0.05f));
        b.AddFloat32("model.norm.weight", [H], Ones(H));
        for (int i = 0; i < L; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [H], Ones(H));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [H], Ones(H));
            b.AddFloat32($"{p}.self_attn.q_proj.weight", [NH * HD, H], R(NH * HD * H, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight", [NH * HD, H], R(NH * HD * H, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight", [NH * HD, H], R(NH * HD * H, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight", [H, NH * HD], R(H * NH * HD, 0.05f));
            b.AddFloat32($"{p}.mlp.gate_proj.weight", [I, H], R(I * H, 0.05f));
            b.AddFloat32($"{p}.mlp.up_proj.weight", [I, H], R(I * H, 0.05f));
            b.AddFloat32($"{p}.mlp.down_proj.weight", [H, I], R(H * I, 0.05f));
        }
        b.AddFloat32("lm_head.weight", [V, H], R(V * H, 0.05f));
        b.WriteTo(path);
        var cfg = new ModelConfig
        {
            Architecture = Architecture.Llama, VocabSize = V, HiddenSize = H, IntermediateSize = I, NumLayers = L,
            NumAttentionHeads = NH, NumKvHeads = NH, HeadDim = HD, MaxSequenceLength = 128, NormEpsilon = 1e-5f,
            RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: HD, Type: RoPEType.Norm),
        };
        using var sf = SafetensorsFile.Open(path);
        using var cpu = TransformerModel.LoadFromSafetensors(sf, cfg);
        using var vk = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spv);
        float[] Last(ITensor t, int n) { var a = Copy(t); return a.AsSpan((n - 1) * V, V).ToArray(); }
        var m = new Models(
            (t, p) => { using var o = cpu.Forward(t, p, -1); return Last(o, t.Length); },
            (t, p) => { using var o = vk.Forward(t, p, -1); return Copy(o); },
            x => vk.ResidualModifier = x, x => cpu.ResidualModifier = x, H, L, V);
        RunScenario(m, layer: 1, alpha: 1f, "synthetic-D64 layer1");
        RunScenario(m, layer: 0, alpha: 1f, "synthetic-D64 layer0");
    }

    // ------------------------------------------------------------------ real SmolLM-135M Q8_0, D = 576 ----

    [SkippableFact]
    public void SmolLM135M_Q8_0_D576_AdapterAttach_MatchesCpu_DetachIsBitIdentical()
    {
        var loc = TestFixtureResolver.ResolveFile("DOTLLM_SMOLLM135M_GGUF", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.IfNot(loc.Found, loc.SkipMessage("SmolLM-135M Q8_0"));
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        using var gguf = GgufFile.Open(loc.Path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var cpu = TransformerModel.LoadFromGguf(gguf, cfg);
        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spv);
        int V = cfg.VocabSize;
        float[] Last(ITensor t, int n) { var a = Copy(t); return a.AsSpan((n - 1) * V, V).ToArray(); }
        var m = new Models(
            (t, p) => { using var o = cpu.Forward(t, p, -1); return Last(o, t.Length); },
            (t, p) => { using var o = vk.Forward(t, p, -1); return Copy(o); },
            x => vk.ResidualModifier = x, x => cpu.ResidualModifier = x, cfg.HiddenSize, cfg.NumLayers, V);
        RunScenario(m, layer: 12, alpha: 0.05f, "smollm135m-Q8_0 layer12");
    }
}
