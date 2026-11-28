using System.Diagnostics;
using System.Globalization;
using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Gguf;
using DotLLM.Models.Vision;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The Vulkan ViT encoder (#2062 part B) against the CPU encoder and llama.cpp. Two phases so the CPU and the iGPU are never busy at once:
/// <list type="number">
/// <item><c>CpuCache_Populate</c> (set <c>DOTLLM_VK_VISION_CPU_CACHE=1</c>; hold the <c>cpu-bench</c> lock): runs the CPU encoder on the test images and caches its embeddings and wall times
/// under <c>vision-vk-cache/</c> next to the oracles.</item>
/// <item>every other test reads that cache (hold the Strix GPU lock); a missing cache skips.</item>
/// </list>
/// The sharp reference is the CPU encoder on the F32-weights control mmproj (the Q8_0 matrices dequantised, so the same weight VALUES, F32 activations, no Q8_0 activation noise); llama.cpp's own
/// Q8_0 reference sits 2% from it (docs/INKLING2.md section 12).
/// </summary>
public sealed class Inkling2VulkanVisionEncoderTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private VulkanDevice? _dev;
    public Inkling2VulkanVisionEncoderTests(ITestOutputHelper output) => _output = output;
    public void Dispose() => _dev?.Dispose();

    internal static string CacheDir => Path.GetFullPath(Path.Combine(Inkling2VisionFixtures.OracleDir, "..", "vision-vk-cache"));
    private static string ControlDir => Path.GetFullPath(Path.Combine(Inkling2VisionFixtures.OracleDir, "..", "vision-f32-control"));

    internal static DecodedImage LoadImage(string name)
    {
        if (name == "noise2560") return Noise(2560);
        return ImageDecoder.Load(Path.Combine(Inkling2VisionFixtures.OracleDir, name, "image.ppm"));
    }

    /// <summary>Deterministic 2560x2560 noise: 64 x 64 = 4,096 image tokens, the limit.</summary>
    internal static DecodedImage Noise(int side)
    {
        var rgb = new byte[(long)side * side * 3];
        new Random(12345).NextBytes(rgb);
        return new DecodedImage(side, side, rgb);
    }

    private VulkanDevice Dev()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
        return _dev ??= VulkanDevice.Create();
    }

    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return f;
    }

    private static float[]? Cached(string file) => File.Exists(Path.Combine(CacheDir, file)) ? ReadF32(Path.Combine(CacheDir, file)) : null;

    internal static float[] Floats(ImageEmbeddings e)
    {
        var all = new float[e.Tokens * e.Dimension];
        var row = new float[e.Dimension];
        for (int r = 0; r < e.Tokens; r++) { e.ReadRow(r, row); row.CopyTo(all, r * e.Dimension); }
        return all;
    }

    internal static double RelRms(float[] got, float[] want)
    {
        double num = 0, den = 0;
        for (int i = 0; i < want.Length; i++) { double d = got[i] - want[i]; num += d * d; den += (double)want[i] * want[i]; }
        return Math.Sqrt(num / den);
    }

    /// <summary>Minimum over rows of the cosine between got and want rows.</summary>
    internal static double MinRowCosine(float[] got, float[] want, int dim)
    {
        double min = 1;
        for (int r = 0; r < want.Length / dim; r++)
        {
            double dot = 0, a = 0, b = 0;
            for (int i = 0; i < dim; i++) { double x = got[r * dim + i], y = want[r * dim + i]; dot += x * y; a += x * x; b += y * y; }
            min = Math.Min(min, dot / Math.Sqrt(a * b));
        }
        return min;
    }

    private static string Spv() => Inkling2VulkanTests.SpvDir();

    private VulkanInkling2VisionEncoder Load(VisionEmbedPrecision? arm = null, string? mmproj = null)
    {
        string? mm = mmproj ?? Inkling2VisionFixtures.FindMmproj();
        Skip.If(mm is null, "mmproj-Inkling-2-Mini-Q8_0.gguf not in the HF cache");
        return VulkanInkling2VisionEncoder.Load(Dev(), mm!, Spv(), ImageOverLimit.Refuse, arm);
    }

    // ───────────── phase 1: CPU only ─────────────

    [SkippableFact]
    public void CpuCache_Populate()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_VK_VISION_CPU_CACHE") == "1", "set DOTLLM_VK_VISION_CPU_CACHE=1 (and hold the cpu-bench lock) to populate the CPU cache");
        string? mm = Inkling2VisionFixtures.FindMmproj();
        Skip.If(mm is null, "mmproj not in the HF cache");
        string control = Path.Combine(ControlDir, "mmproj-f32.gguf");
        Directory.CreateDirectory(CacheDir);
        var times = new List<string>();
        void Save(string file, ImageEmbeddings e)
        {
            var f = Floats(e);
            var bytes = new byte[f.Length * 4];
            Buffer.BlockCopy(f, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(CacheDir, file), bytes);
        }
        string timesFile = Path.Combine(CacheDir, "cpu_times.txt");
        if (File.Exists(timesFile)) times.AddRange(File.ReadAllLines(timesFile));
        // Resumable: an entry already cached is skipped (the 4,096-token encode takes minutes and a crashed host must not redo the rest).
        using (var enc = Inkling2VisionEncoder.Load(mm!, ThreadingConfig.Auto))   // the shipping default (BF16 embeddings)
            foreach (string name in new[] { "rainbow80", "rainbow190x110", "rainbow640", "noise2560" })
            {
                if (File.Exists(Path.Combine(CacheDir, $"cpu_bf16_{name}.f32")) && times.Any(t => t.StartsWith($"cpu_bf16 {name} "))) continue;
                var img = LoadImage(name);
                if (name != "noise2560") { using var warm = enc.Encode(img); }
                var sw = Stopwatch.StartNew();
                using var e = enc.Encode(img);
                times.RemoveAll(t => t.StartsWith($"cpu_bf16 {name} "));
                times.Add($"cpu_bf16 {name} {e.Tokens} {sw.Elapsed.TotalSeconds.ToString("F3", CultureInfo.InvariantCulture)}");
                Save($"cpu_bf16_{name}.f32", e);
                File.WriteAllLines(timesFile, times);
            }
        if (File.Exists(control))
            using (var enc = Inkling2VisionEncoder.Load(control, ThreadingConfig.Auto, embedPrecision: VisionEmbedPrecision.F32))
                foreach (string name in new[] { "rainbow80", "rainbow190x110", "rainbow640" })
                {
                    if (File.Exists(Path.Combine(CacheDir, $"cpu_ctrl_f32_{name}.f32"))) continue;
                    using var e = enc.Encode(LoadImage(name));
                    Save($"cpu_ctrl_f32_{name}.f32", e);
                }
        foreach (var t in times) _output.WriteLine(t);
    }

    // ───────────── phase 2: GPU ─────────────

    /// <summary>
    /// Embedding gate, per arm and image: relRMS and minimum row cosine of the Vulkan encoder (Q8_0 mmproj, BF16 activations) against (a) the CPU encoder on the F32-weights control, the sharp
    /// reference; (b) the CPU shipping encoder; (c) llama.cpp's Q8_0 oracle (<c>embd.f32</c>, 2% noise floor); (d) llama.cpp on the F32 control (80 and 190x110 only).
    /// </summary>
    [SkippableTheory]
    [InlineData("rainbow80", "bf16")]
    [InlineData("rainbow190x110", "bf16")]
    [InlineData("rainbow640", "bf16")]
    [InlineData("rainbow80", "f32")]
    [InlineData("rainbow190x110", "f32")]
    [InlineData("rainbow80", "f16")]
    [InlineData("rainbow190x110", "f16")]
    public void Embeddings_MatchReferences(string name, string armName)
    {
        var arm = armName switch { "f32" => VisionEmbedPrecision.F32, "f16" => VisionEmbedPrecision.F16, _ => VisionEmbedPrecision.BF16 };
        var ctrl = Cached($"cpu_ctrl_f32_{name}.f32"); var cpu = Cached($"cpu_bf16_{name}.f32");
        Skip.If(ctrl is null || cpu is null, "CPU cache not populated (DOTLLM_VK_VISION_CPU_CACHE=1 CpuCache_Populate)");
        using var enc = Load(arm);
        Assert.Equal(arm, enc.EmbedPrecision);
        using var e = enc.Encode(LoadImage(name));
        var got = Floats(e);
        double vsCtrl = RelRms(got, ctrl!), vsCpu = RelRms(got, cpu!), cosCtrl = MinRowCosine(got, ctrl!, e.Dimension);
        var llamaQ8 = ReadF32(Path.Combine(Inkling2VisionFixtures.OracleDir, name, "embd.f32"));
        double vsQ8 = RelRms(got, llamaQ8);
        string f32c = Path.Combine(ControlDir, $"llamacpp_f32_{name}.f32");
        string vsLlamaCtrl = File.Exists(f32c) ? RelRms(got, ReadF32(f32c)).ToString("E2") : "n/a";
        double cpuVsCtrl = RelRms(cpu!, ctrl!);
        _output.WriteLine($"{name} arm={armName}: Vulkan vs CPU-F32-control {vsCtrl:E2} (min row cos {cosCtrl:F5}); vs CPU shipping {vsCpu:E2}; vs llama.cpp Q8_0 {vsQ8:E2}; vs llama.cpp F32 control {vsLlamaCtrl}; " +
                          $"[CPU shipping vs CPU control {cpuVsCtrl:E2}]; weights {enc.WeightBytes / 1e6:F1} MB (embeddings {enc.EmbedBytes / 1e6:F2} MB), peak scratch {enc.LastPeakScratchBytes / 1e6:F1} MB");
        Assert.True(vsCtrl < Gates.VsControl, $"relRMS vs the F32-control CPU encoder {vsCtrl:E2} (gate {Gates.VsControl:E1})");
        Assert.True(cosCtrl > Gates.MinCosine, $"min row cosine {cosCtrl:F5}");
        Assert.True(vsQ8 < Gates.VsLlamaQ8, $"relRMS vs llama.cpp Q8_0 {vsQ8:E2} (gate {Gates.VsLlamaQ8:E1})");
    }

    internal static class Gates
    {
        public const double VsControl = 6e-2, MinCosine = 0.97, VsLlamaQ8 = 9e-2;
    }

    /// <summary>Encode time per image, Vulkan vs the cached CPU wall times, plus the 4,096-token image's embeddings against the CPU result. Warm (one untimed encode first), median of 3.</summary>
    [SkippableFact]
    public void Timing_VulkanVsCpu()
    {
        string tf = Path.Combine(CacheDir, "cpu_times.txt");
        Skip.IfNot(File.Exists(tf), "CPU cache not populated");
        var cpuTimes = File.ReadAllLines(tf).Select(l => l.Split(' ')).Where(p => p[0] == "cpu_bf16").ToDictionary(p => p[1], p => double.Parse(p[3], CultureInfo.InvariantCulture));
        using var enc = Load();
        foreach (string name in new[] { "rainbow80", "rainbow190x110", "rainbow640", "noise2560" })
        {
            var img = LoadImage(name);
            using (var warm = enc.Encode(img)) { }
            int reps = name == "noise2560" ? 2 : 3;
            var secs = new List<double>(); var gpu = new List<double>();
            ImageEmbeddings? last = null;
            for (int i = 0; i < reps; i++)
            {
                var sw = Stopwatch.StartNew();
                last?.Dispose();
                last = enc.Encode(img);
                secs.Add(sw.Elapsed.TotalSeconds); gpu.Add(enc.LastGpuSeconds);
            }
            secs.Sort(); gpu.Sort();
            double med = secs[secs.Count / 2], cpu = cpuTimes.TryGetValue(name, out var c) ? c : double.NaN;
            string rel = "";
            if (Cached($"cpu_bf16_{name}.f32") is { } cpuE) rel = $"; relRMS vs CPU shipping {RelRms(Floats(last!), cpuE):E2}";
            _output.WriteLine($"TIMING {name}: {last!.Tokens} tokens; Vulkan encode {med * 1000:F1} ms (GPU part {gpu[gpu.Count / 2] * 1000:F1} ms; incl. preprocessing + upload + download); CPU {cpu * 1000:F1} ms; speedup {cpu / med:F1}x{rel}; peak scratch {enc.LastPeakScratchBytes / 1e6:F1} MB");
            last.Dispose();
        }
    }

    /// <summary>
    /// MEASUREMENT (hold BOTH locks: the CPU control encode and the GPU encode run one after the other in this process): the relRMS of the Vulkan tower's residual stream against the CPU encoder
    /// on the F32-weights control (same weight values, F32 activations) at every layer, to show where the Vulkan-vs-control gap comes from (gradual BF16 rounding growth, or a step).
    /// </summary>
    [SkippableTheory]
    [InlineData("rainbow80")]
    [InlineData("rainbow640")]
    public void Report_LayerwiseDivergenceFromF32Control(string name)
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_VK_VISION_LAYERWISE") == "1", "set DOTLLM_VK_VISION_LAYERWISE=1 (hold the GPU and cpu-bench locks)");
        string control = Path.Combine(ControlDir, "mmproj-f32.gguf");
        Skip.IfNot(File.Exists(control), "F32 control mmproj not available");
        var cpu = new Dictionary<string, float[]>();
        using (var enc = Inkling2VisionEncoder.Load(control, ThreadingConfig.Auto, embedPrecision: VisionEmbedPrecision.F32))
        {
            enc.Trace = (n, data, rows, cols) => { if (n is "embeddings" or "pre_ln" or "encoder_out" || n.StartsWith("layer_out-")) cpu[n] = data.ToArray(); };
            using var e = enc.Encode(LoadImage(name));
        }
        using var vk = Load();
        var lines = new List<string>();
        vk.Trace = (n, data, rows, cols) =>
        {
            if (cpu.TryGetValue(n, out var want)) lines.Add($"{n}: relRMS {RelRms(data, want):E2}");
        };
        using var r = vk.Encode(LoadImage(name));
        foreach (var l in lines) _output.WriteLine($"{name} {l}");
    }

    /// <summary>The RoPE table is built once per patch grid (native memory, not a managed array per call) and not re-uploaded while the scratch holds it.</summary>
    [SkippableFact]
    public void RopeTable_IsBuiltOncePerGrid()
    {
        using var enc = Load();
        using (var a = enc.Encode(LoadImage("rainbow80"))) { }
        using (var b = enc.Encode(LoadImage("rainbow80"))) { }
        Assert.Equal(1, enc.CachedRopeTables);
        using (var c = enc.Encode(LoadImage("rainbow190x110"))) { }
        Assert.Equal(2, enc.CachedRopeTables);
        using (var d = enc.Encode(LoadImage("rainbow80"))) { }
        Assert.Equal(2, enc.CachedRopeTables);
    }
}
