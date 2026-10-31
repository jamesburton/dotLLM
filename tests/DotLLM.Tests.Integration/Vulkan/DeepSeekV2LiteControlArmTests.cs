using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1278 control arm. Three-way comparison of the step-0 logits of DeepSeek-V2-Lite Q4_K_M:
/// CPU on the original Q4_K_M file (Q8_K activations), Vulkan on the same file, and CPU on a weights-decoded
/// control (every quantized tensor replaced by its gguf-py decode, stored F16) - the closest thing to the
/// "true" model with unquantized activations. Whichever of CPU/Vulkan is farther from the control is the outlier.
/// Gated on <c>DOTLLM_DSV2_DECODED_CONTROL_GGUF</c> (build with scripts/make_f32_decoded_gguf.py); skipped otherwise.
/// </summary>
[Trait("Category", "GPU")]
[Collection(VulkanResidencyReportCollection.Name)]
public sealed class DeepSeekV2LiteControlArmTests
{
    private readonly ITestOutputHelper _output;
    public DeepSeekV2LiteControlArmTests(ITestOutputHelper output) => _output = output;

    private static unsafe float[] Last(IModel m, int[] ids, int vocab)
    {
        int[] pos = Enumerable.Range(0, ids.Length).ToArray();
        m.ResetSequenceState();
        using ITensor t = m.Forward(ids, pos, -1);
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)(t.Shape[0] - 1) * vocab, vocab).ToArray();
    }

    private static (double LInf, double Rms, int ArgIdx) Cmp(float[] a, float[] b)
    {
        double m = 0, s = 0; int mi = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = Math.Abs(a[i] - b[i]);
            if (d > m) { m = d; mi = i; }
            s += d * d;
        }
        return (m, Math.Sqrt(s / a.Length), mi);
    }

    private static void Save(string path, float[][] rows)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var w = new BinaryWriter(File.Create(path));
        foreach (var r in rows) foreach (float f in r) w.Write(f);
    }

    private static void Load(string path, float[][] rows, int vocab)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        for (int i = 0; i < rows.Length; i++) { rows[i] = new float[vocab]; for (int j = 0; j < vocab; j++) rows[i][j] = r.ReadSingle(); }
    }

    private static int Arg(float[] x) { int b = 0; for (int i = 1; i < x.Length; i++) if (x[i] > x[b]) b = i; return b; }

    [SkippableFact]
    public void Dsv2Lite_ControlArm()
    {
        var q4 = TestFixtureResolver.ResolveFile("DOTLLM_DEEPSEEK_V2_LITE_Q4_K_M_GGUF", "mradermacher", "DeepSeek-V2-Lite-GGUF", "DeepSeek-V2-Lite.Q4_K_M.gguf");
        Skip.If(!q4.Found, "no Q4_K_M");
        string? ctl = Environment.GetEnvironmentVariable("DOTLLM_DSV2_DECODED_CONTROL_GGUF");
        Skip.If(string.IsNullOrEmpty(ctl) || !File.Exists(ctl), "DOTLLM_DSV2_DECODED_CONTROL_GGUF not set");
        Skip.IfNot(VulkanDevice.IsAvailable(), "no vulkan");
        string spv = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "spv"));
        if (!Directory.Exists(spv) || Directory.GetFiles(spv, "*.spv").Length == 0)
            spv = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));

        string[] prompts = ["The capital of France is", "def fibonacci(n):", "In 1969, the first humans landed on the Moon, an event watched by hundreds of millions of people"];

        string? cache = Environment.GetEnvironmentVariable("DOTLLM_DSV2_CACHE_DIR");
        string cqPath = cache is null ? "" : Path.Combine(cache, "cpuq.bin"), cc = cache is null ? "" : Path.Combine(cache, "ctl.bin");
        float[][] cpuQ = new float[prompts.Length][], vk = new float[prompts.Length][], ctlL = new float[prompts.Length][];
        int[][] ids = new int[prompts.Length][];
        float[][] cpuExp = new float[prompts.Length][];
        int vocab;
        using (var g = GgufFile.Open(q4.Path!))
        {
            var cfg = GgufModelConfigExtractor.Extract(g.Metadata);
            vocab = cfg.VocabSize;
            var tok = GgufBpeTokenizerFactory.Load(g.Metadata);
            for (int i = 0; i < prompts.Length; i++) ids[i] = tok.Encode(prompts[i]).ToArray();
            if (cache is not null && File.Exists(cqPath)) Load(cqPath, cpuQ, vocab);
            else
            {
                using (IModel cpu = ModelLoader.CreateCpuModelFromGguf(g, cfg))
                    for (int i = 0; i < prompts.Length; i++) cpuQ[i] = Last(cpu, ids[i], vocab);
                if (cache is not null) Save(cqPath, cpuQ);
            }
            var vcfg = cfg.MlaConfig is null ? cfg : cfg with { MlaConfig = cfg.MlaConfig with { UseLatentCache = false, UseHybridMlaCache = false } };
            if (Environment.GetEnvironmentVariable("DOTLLM_DSV2_CPU_EXPANDED") == "1")
            {
                using IModel cpuE = ModelLoader.CreateCpuModelFromGguf(g, vcfg);
                for (int i = 0; i < prompts.Length; i++) cpuExp[i] = Last(cpuE, ids[i], vocab);
            }
            using var dev = VulkanDevice.Create();
            (IModel? vm, _) = VulkanModelLoader.CreateFromGguf(dev, g, vcfg, spv);
            using (vm)
                for (int i = 0; i < prompts.Length; i++) vk[i] = Last(vm!, ids[i], vocab);
        }
        if (cache is not null && File.Exists(cc)) Load(cc, ctlL, vocab);
        else
        {
            using var g = GgufFile.Open(ctl!);
            var cfg = GgufModelConfigExtractor.Extract(g.Metadata);
            using IModel cpu = ModelLoader.CreateCpuModelFromGguf(g, cfg);
            for (int i = 0; i < prompts.Length; i++) ctlL[i] = Last(cpu, ids[i], vocab);
            if (cache is not null) Save(cc, ctlL);
        }
        for (int i = 0; i < prompts.Length; i++)
        {
            var cq = Cmp(cpuQ[i], ctlL[i]); var vc = Cmp(vk[i], ctlL[i]); var cv = Cmp(cpuQ[i], vk[i]);
            _output.WriteLine($"prompt {i} ({ids[i].Length} tok): argmax cpuQ4={Arg(cpuQ[i])} vk={Arg(vk[i])} ctl={Arg(ctlL[i])}");
            _output.WriteLine($"   cpuQ4 vs ctl: Linf={cq.LInf:F3} rms={cq.Rms:F4}");
            _output.WriteLine($"   vulkan vs ctl: Linf={vc.LInf:F3} rms={vc.Rms:F4}");
            if (cpuExp[i] is not null) { var ce = Cmp(cpuExp[i], ctlL[i]); var ec = Cmp(cpuExp[i], vk[i]); _output.WriteLine($"   cpuExpandedMLA vs ctl: Linf={ce.LInf:F3} rms={ce.Rms:F4}; vs vulkan Linf={ec.LInf:F3} rms={ec.Rms:F4}"); }
            _output.WriteLine($"   cpuQ4 vs vulkan: Linf={cv.LInf:F3} rms={cv.Rms:F4}");
        }
    }
}
