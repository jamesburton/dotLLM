using System.Diagnostics;
using System.Globalization;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Core.Vision;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.Vision;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// MEASUREMENT STUDY (epic #2025, docs/INKLING2.md section 12c): which activation storage format for Inkling-2-Mini image prompts on Vulkan (BF16 shipped, F16, F32, mixed / targeted).
/// Opt-in: every test skips unless <c>DOTLLM_INK_PREC_STUDY=1</c>. Results are appended to <c>DOTLLM_INK_PREC_OUT</c> (default <c>%TEMP%\ink-prec-results.txt</c>) as well as the test output.
/// GPU tests need the Strix GPU lock; the CPU-simulation tests and the CPU encoder run on the CPU only (hold the cpu-bench lock); never run them beside GPU work.
/// </summary>
public sealed class Inkling2ActivationPrecisionStudyTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private static readonly object FileLock = new();
    public Inkling2ActivationPrecisionStudyTests(ITestOutputHelper o) => _out = o;
    public void Dispose() { }

    private static void Gate() => Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_STUDY") == "1", "set DOTLLM_INK_PREC_STUDY=1");

    private void Log(string line)
    {
        _out.WriteLine(line);
        string path = Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_OUT") ?? Path.Combine(Path.GetTempPath(), "ink-prec-results.txt");
        lock (FileLock) File.AppendAllText(path, line + Environment.NewLine);
    }

    private static readonly string[] Images = ["rainbow80", "rainbow190x110", "rainbow640"];
    private const string Passage = "nvfp4-passage-c256-fa-off.kld";

    private static unsafe float[] Row(ITensor t, int row, int vocab)
    {
        var a = new float[vocab];
        new ReadOnlySpan<float>((float*)t.DataPointer + (long)row * t.Shape[1], vocab).CopyTo(a);
        return a;
    }

    private static unsafe float[] Rows(ITensor t, int rows, int vocab)
    {
        var a = new float[rows * vocab];
        new ReadOnlySpan<float>((void*)t.DataPointer, a.Length).CopyTo(a);
        return a;
    }

    internal readonly record struct KlStat(double Mean, double Max, int Top1, int Rows, int Mismatch, int Steps)
    {
        public override string ToString() => $"mean {Mean * 1e3:F2}e-3 max {Max * 1e3:F1}e-3 top1 {Top1}/{Rows} greedy-mism {Mismatch}/{Steps}";
    }

    /// <summary>Teacher-forced image-prompt KL: the oracle's own continuation, 32-token prefill calls (the reviewer's and the gate's setup).</summary>
    private static KlStat ImageKl(Inkling2VisionEndToEndTests.Oracle o, Func<int[], int[], IReadOnlyList<ImageSpan>, ITensor> fwd, Action reset, ImageEmbeddings emb, int sub = 32)
    {
        var prompt = o.Before.Concat(Enumerable.Repeat(200054, emb.Tokens)).Concat(o.After).ToArray();
        var spans = new[] { new ImageSpan(o.Before.Length, emb) };
        int V = o.NVocab, mism = 0; double sum = 0, max = 0; int top = 0;
        reset();
        float[] ours = [];
        for (int at = 0; at < prompt.Length; at += sub)
        {
            int len = Math.Min(sub, prompt.Length - at);
            using var t = fwd(prompt[at..(at + len)], Enumerable.Range(at, len).ToArray(), spans);
            if (at + len == prompt.Length) ours = Row(t, 0, V);
        }
        for (int r = 0; r < o.Gen.Length + 1; r++)
        {
            var (kl, same) = Inkling2VisionEndToEndTests.FullKl(o.Logits.AsSpan(r * V, V), ours);
            sum += kl; max = Math.Max(max, kl); if (same) top++;
            if (r < o.Gen.Length)
            {
                if (Inkling2RealModelTests.ArgMax(ours) != o.Gen[r]) mism++;
                using var step = fwd([o.Gen[r]], [prompt.Length + r], spans);
                ours = Row(step, 0, V);
            }
        }
        return new KlStat(sum / (o.Gen.Length + 1), max, top, o.Gen.Length + 1, mism, o.Gen.Length);
    }

    /// <summary>Text passage gate (254 positions per window, 32-token calls) plus teacher-forced greedy identity of the 8 oracle prompts (margin >= 0.3 counted separately).</summary>
    private static (KlStat Kl, int MismConf, int Conf) TextGate(int vocab, Func<int[], int[], ITensor> fwdAll, Func<int[], int[], ITensor> fwdLast, Action reset, bool greedy)
    {
        var r = Inkling2RealModelTests.LoadKlRef(Passage)!;
        int first = r.NCtx / 2, nScored = r.NCtx - 1 - first;
        double sum = 0, max = 0; int top = 0, n = 0;
        for (int chunk = 0; chunk < r.NChunk; chunk++)
        {
            var baseRows = Inkling2RealModelTests.ReadChunk(r, chunk);
            var ids = r.Tokens.AsSpan(chunk * r.NCtx, r.NCtx).ToArray();
            reset();
            var got = new float[r.NCtx * vocab];
            for (int at = 0; at < r.NCtx; at += 32)
            {
                int len = Math.Min(32, r.NCtx - at);
                using var part = fwdAll(ids[at..(at + len)], Enumerable.Range(at, len).ToArray());
                Rows(part, len, vocab).CopyTo(got, at * vocab);
            }
            for (int i = 0; i < nScored; i++)
            {
                var (kl, same) = Inkling2RealModelTests.Kl(baseRows[i], got.AsSpan((first + i) * vocab, vocab));
                sum += kl; max = Math.Max(max, kl); if (same) top++; n++;
            }
        }
        int steps = 0, mism = 0, mc = 0, conf = 0;
        string path = Path.Combine(Inkling2RealModelTests.OracleDir, "greedy-nvfp4-fa-off.json");
        if (greedy && File.Exists(path))
        {
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
            foreach (var p in doc.RootElement.EnumerateObject())
            {
                int[] prompt = p.Value.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                int[] want = p.Value.GetProperty("gen_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                double[] margin = p.Value.GetProperty("margins").EnumerateArray().Select(e => e.GetDouble()).ToArray();
                reset();
                using var pre = fwdLast(prompt, Enumerable.Range(0, prompt.Length).ToArray());
                int arg = Inkling2RealModelTests.ArgMax(Rows(pre, 1, vocab));
                for (int j = 0; j < want.Length; j++)
                {
                    steps++; bool c = margin[j] >= 0.3; if (c) conf++;
                    if (arg != want[j]) { mism++; if (c) mc++; }
                    if (j + 1 == want.Length) break;
                    using var st = fwdLast([want[j]], [prompt.Length + j]);
                    arg = Inkling2RealModelTests.ArgMax(Rows(st, 1, vocab));
                }
            }
        }
        return (new KlStat(sum / n, max, top, n, mism, steps), mc, conf);
    }

    private static Inkling2VulkanOptions Arm(string arm) => arm switch
    {
        "bf16" => new Inkling2VulkanOptions(),
        "f16" => new Inkling2VulkanOptions(ActivationF16: true),
        "f32" => new Inkling2VulkanOptions(ActivationF32: true),
        "bf16+f32img" => new Inkling2VulkanOptions(F32ImageChunks: true),
        "f16+f32img" => new Inkling2VulkanOptions(ActivationF16: true, F32ImageChunks: true),
        _ => throw new ArgumentException(arm),
    };

    // ───────────────────────────── GPU: LLM arms ─────────────────────────────

    /// <summary>KL of each Vulkan activation arm on the 3 images (CPU encoder, shipping BF16 embeddings, 32-token calls) and the text passage gate with greedy identity.</summary>
    [SkippableTheory]
    [InlineData("bf16")]
    [InlineData("f16")]
    [InlineData("f32")]
    [InlineData("bf16+f32img")]
    public void Llm_Arm_Kl(string arm)
    {
        Gate();
        Skip.IfNot(VulkanDevice.IsAvailable(), "no vulkan");
        string? gguf = Inkling2RealModelTests.FindGguf(); Skip.If(gguf is null);
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null);
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var enc = Inkling2VisionEncoder.Load(mm!, ThreadingConfig.Auto);
        using var dev = VulkanDevice.Create();
        using var vk = VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, Inkling2VulkanTests.SpvDir(), Arm(arm));
        foreach (string name in Images)
        {
            var o = Inkling2VisionEndToEndTests.LoadOracle(name); Skip.If(o is null);
            using var emb = enc.Encode(o!.Image);
            var s = ImageKl(o, (i, p, sp) => vk.ForwardWithImages(i, p, -1, sp, null, true), vk.ResetSequenceState, emb);
            Log($"LLMKL arm={arm} {name}: {s}; scratch bf16 {vk.ActivationScratchBytes(Inkling2ActFormat.Bf16) / 1e6:F1} MB f16 {vk.ActivationScratchBytes(Inkling2ActFormat.F16) / 1e6:F1} MB f32 {vk.ActivationScratchBytes(Inkling2ActFormat.F32) / 1e6:F1} MB");
        }
        if (arm != "bf16+f32img")   // no image rows in the text gate: identical to bf16 by construction
        {
            var (kl, mc, conf) = TextGate(cfg.VocabSize,
                (i, p) => vk.Forward(i, p, -1), (i, p) => vk.Forward(i, p, -1, null, true), vk.ResetSequenceState, greedy: true);
            Log($"LLMKL arm={arm} text-passage: {kl}; greedy confident-mismatches {mc}/{conf}");
        }
        Log($"LLMMEM arm={arm}: ComputeMemoryBytes {vk.ComputeMemoryBytes / 1e6:F1} MB (weights + scratch + KV + conv)");
    }

    // ───────────────────────────── GPU: LLM performance ─────────────────────────────

    /// <summary>
    /// Prefill / decode / image TTFT per arm, interleaved round-robin with a warm-up round (the clock-ramp confound); the arms' models are all resident (weights are imported zero-copy).
    /// DOTLLM_INK_PREC_ROUNDS (default 5).
    /// </summary>
    [SkippableFact]
    public void Llm_Perf_Interleaved()
    {
        Gate();
        Skip.IfNot(VulkanDevice.IsAvailable(), "no vulkan");
        string? gguf = Inkling2RealModelTests.FindGguf(); Skip.If(gguf is null);
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null);
        var r = Inkling2RealModelTests.LoadKlRef("nvfp4-wikitext-c1280-fa-off.kld"); Skip.If(r is null);
        var o = Inkling2VisionEndToEndTests.LoadOracle("rainbow640"); Skip.If(o is null);
        int rounds = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_ROUNDS"), out int rr) ? rr : 5;
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var enc = Inkling2VisionEncoder.Load(mm!, ThreadingConfig.Auto);
        using var emb = enc.Encode(o!.Image);
        var prompt = o.Before.Concat(Enumerable.Repeat(200054, emb.Tokens)).Concat(o.After).ToArray();
        var spans = new[] { new ImageSpan(o.Before.Length, emb) };
        int[] text = r!.Tokens.AsSpan(0, 300).ToArray();
        string[] arms = ["bf16", "f16", "f32", "bf16+f32img"];
        using var dev = VulkanDevice.Create();
        var models = arms.ToDictionary(a => a, a => VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, Inkling2VulkanTests.SpvDir(), Arm(a)));
        try
        {
            var res = arms.ToDictionary(a => a, _ => new List<(double Pre, double Dec, double Img)>());
            for (int round = 0; round <= rounds; round++)
            {
                var order = round % 2 == 0 ? arms : arms.Reverse().ToArray();   // alternate direction: no arm is always first (clock ramp)
                foreach (string a in order)
                {
                    var vk = models[a];
                    vk.ResetSequenceState();
                    var sw = Stopwatch.StartNew();
                    using (vk.Forward(text, Enumerable.Range(0, text.Length).ToArray(), -1, null, true)) { }
                    double pre = text.Length / sw.Elapsed.TotalSeconds;
                    int next = 1234, steps = 32;
                    sw.Restart();
                    for (int j = 0; j < steps; j++)
                    {
                        using var t = vk.Forward([next], [text.Length + j], -1, null, true);
                        next = Inkling2RealModelTests.ArgMax(Row(t, 0, cfg.VocabSize));
                    }
                    double dec = steps / sw.Elapsed.TotalSeconds;
                    vk.ResetSequenceState();
                    sw.Restart();
                    using (vk.ForwardWithImages(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, spans, null, true)) { }
                    double img = sw.Elapsed.TotalSeconds;
                    if (round > 0) res[a].Add((pre, dec, img));
                }
            }
            foreach (string a in arms)
            {
                var l = res[a];
                string Rng(IEnumerable<double> v, string f) => $"{v.Min().ToString(f, CultureInfo.InvariantCulture)}-{v.Max().ToString(f, CultureInfo.InvariantCulture)} (median {v.OrderBy(x => x).ElementAt(v.Count() / 2).ToString(f, CultureInfo.InvariantCulture)})";
                Log($"LLMPERF arm={a} rounds={l.Count}: prefill300 tok/s {Rng(l.Select(x => x.Pre), "F0")}; decode tok/s {Rng(l.Select(x => x.Dec), "F2")}; image640 prompt ({prompt.Length} tok) prefill s {Rng(l.Select(x => x.Img), "F3")}");
            }
        }
        finally { foreach (var m in models.Values) m.Dispose(); }
    }

    // ───────────────────────────── CPU: simulation of the storage formats ─────────────────────────────

    private static string[] SimArms => (Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_SIM") ?? "f32,bf16,f16,resid-f32,f16+resid-bf16,f16+resid-f32,first8-f32,first16-f32,first24-f32,imgrows-f32,nonresid-f32").Split(',');

    private static Inkling2ActSim MakeSim(string arm, int imgStart, int imgEnd)
    {
        var s = new Inkling2ActSim();
        switch (arm)
        {
            case "f32": s.Round = false; break;
            case "bf16": break;
            case "f16": s.Base = Inkling2SimFormat.F16; break;
            case "resid-f32": s.Residual = Inkling2SimFormat.F32; break;                       // everything BF16 except the residual stream
            case "f16+resid-bf16": s.Base = Inkling2SimFormat.F16; s.Residual = Inkling2SimFormat.Bf16; break;
            case "f16+resid-f32": s.Base = Inkling2SimFormat.F16; s.Residual = Inkling2SimFormat.F32; break;
            case "nonresid-f32": s.Base = Inkling2SimFormat.F32; s.Residual = Inkling2SimFormat.Bf16; break;   // the opposite split: BF16 residual, F32 inside the layers
            case "imgrows-f32": s.RowIsF32 = p => p >= imgStart && p < imgEnd; break;
            default:
                if (arm.StartsWith("first") && arm.EndsWith("-f32")) { s.F32UpToVisit = int.Parse(arm["first".Length..^"-f32".Length]); break; }
                throw new ArgumentException(arm);
        }
        return s;
    }

    private static Inkling2TransformerModel LoadCpu(GgufFile file, ModelConfig cfg) => Inkling2TransformerModel.LoadFromGguf(file, cfg, ThreadingConfig.Auto);

    /// <summary>
    /// CPU simulation of the storage formats at the Vulkan forward's store points (docs/INKLING2.md 12c): KL on the 3 images and the text passage, for every arm of
    /// DOTLLM_INK_PREC_SIM; also prints the max |activation| per point and per residual-stream layer for the F32 (stats only) arm. CPU only: hold the cpu-bench lock.
    /// </summary>
    [SkippableFact]
    public void Cpu_Sim_Arms()
    {
        Gate();
        string? gguf = Inkling2RealModelTests.FindGguf(); Skip.If(gguf is null);
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null);
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var enc = Inkling2VisionEncoder.Load(mm!, ThreadingConfig.Auto);
        using var cpu = LoadCpu(file, cfg);
        var embs = new Dictionary<string, ImageEmbeddings>();
        foreach (string n in Images) embs[n] = enc.Encode(Inkling2VisionEndToEndTests.LoadOracle(n)!.Image);
        try
        {
            foreach (string arm in SimArms)
            {
                var allStats = new List<(string Set, Inkling2ActSim Sim)>();
                foreach (string name in Images)
                {
                    var o = Inkling2VisionEndToEndTests.LoadOracle(name)!;
                    var sim = MakeSim(arm, o.Before.Length, o.Before.Length + embs[name].Tokens);
                    cpu.Sim = sim;
                    var s = ImageKl(o, (i, p, sp) => cpu.ForwardWithImages(i, p, -1, sp, null, true), cpu.ResetSequenceState, embs[name]);
                    cpu.Sim = null;
                    Log($"CPUSIM arm={arm} {name}: {s}");
                    allStats.Add((name, sim));
                }
                {
                    var sim = MakeSim(arm, int.MaxValue, int.MaxValue);
                    cpu.Sim = sim;
                    var (kl, mc, conf) = TextGate(cfg.VocabSize, (i, p) => cpu.Forward(i, p, -1), (i, p) => cpu.Forward(i, p, -1, null, true), cpu.ResetSequenceState, greedy: false);
                    cpu.Sim = null;
                    Log($"CPUSIM arm={arm} text-passage: {kl}");
                    allStats.Add(("text", sim));
                }
                if (arm == "f32") DumpStats(allStats);
            }
        }
        finally { foreach (var e in embs.Values) e.Dispose(); }
    }

    private void DumpStats(List<(string Set, Inkling2ActSim Sim)> sets)
    {
        var points = sets.SelectMany(s => s.Sim.Stats.Keys).Distinct().OrderBy(k => k).ToList();
        foreach (string p in points)
        {
            var parts = sets.Select(s => s.Sim.Stats.TryGetValue(p, out var st) ? st : null).Where(x => x is not null).ToList();
            float max = parts.Max(x => x!.MaxAbs);
            long cnt = parts.Sum(x => x!.Count), sub = parts.Sum(x => x!.F16Subnormal), fl = parts.Sum(x => x!.F16Flushed), ov = parts.Sum(x => x!.F16Overflow);
            string per = string.Join(" ", sets.Select(s => s.Sim.Stats.TryGetValue(p, out var st) ? $"{s.Set}={st.MaxAbs:G4}" : $"{s.Set}=-"));
            Log($"ACTMAX point={p,-12} max {max,10:G5} | {per} | F16: overflow {ov}, subnormal {sub} ({100.0 * sub / cnt:F3}%), flushed-to-0 {fl} ({100.0 * fl / cnt:F4}%) of {cnt}");
        }
        foreach (var s in sets)
            Log($"ACTRESID set={s.Set} residual-stream max per visit (embed, 48 layer visits incl. block copies, output): " + string.Join(" ", s.Sim.ResidualMaxPerVisit.Take(50).Select(v => v.ToString("G4"))));
    }

    // ───────────────────────────── GPU: the vision tower ─────────────────────────────

    /// <summary>
    /// ViT arms (BF16 shipped, F16, F32, each with the F16 / F32 RoPE table): embedding relRMS against the CPU F32-control encoder (cached by the ViT tests) and llama.cpp's F32 control, end-to-end KL
    /// through the (BF16, shipped) Vulkan text model, encode time interleaved with warm-up, scratch bytes; the 2560x2560 noise image gives the 4,096-token time / scratch (relRMS vs the CPU shipping encoder).
    /// </summary>
    [SkippableFact]
    public void Vit_Arms()
    {
        Gate();
        Skip.IfNot(VulkanDevice.IsAvailable(), "no vulkan");
        string? gguf = Inkling2RealModelTests.FindGguf(); Skip.If(gguf is null);
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null);
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var dev = VulkanDevice.Create();
        using var vk = VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, Inkling2VulkanTests.SpvDir(), new Inkling2VulkanOptions());
        using var venc = VulkanInkling2VisionEncoder.Load(dev, mm!, Inkling2VulkanTests.SpvDir());
        string cache = Inkling2VulkanVisionEncoderTests.CacheDir;
        string ctrlDir = Path.GetFullPath(Path.Combine(Inkling2VisionFixtures.OracleDir, "..", "vision-f32-control"));
        float[] Read(string path) { var b = File.ReadAllBytes(path); var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, b.Length); return f; }
        (string Name, Inkling2ActFormat Act, bool Rope)[] arms =
        [
            ("bf16", Inkling2ActFormat.Bf16, false), ("bf16+rope32", Inkling2ActFormat.Bf16, true),
            ("f16", Inkling2ActFormat.F16, false), ("f16+rope32", Inkling2ActFormat.F16, true),
            ("f32", Inkling2ActFormat.F32, false), ("f32+rope32", Inkling2ActFormat.F32, true),
        ];
        void Set((string Name, Inkling2ActFormat Act, bool Rope) a) { venc.ActFormat = a.Act; venc.RopeF32 = a.Rope; }

        // accuracy + end-to-end KL per arm and image
        foreach (var a in arms)
        {
            Set(a);
            foreach (string name in Images)
            {
                var o = Inkling2VisionEndToEndTests.LoadOracle(name)!;
                using var e = venc.Encode(o.Image);
                var got = Inkling2VulkanVisionEncoderTests.Floats(e);
                int bad = got.Count(v => !float.IsFinite(v));
                double vsCtrl = Inkling2VulkanVisionEncoderTests.RelRms(got, Read(Path.Combine(cache, $"cpu_ctrl_f32_{name}.f32")));
                string f32c = Path.Combine(ctrlDir, $"llamacpp_f32_{name}.f32");
                string vsLl = File.Exists(f32c) ? Inkling2VulkanVisionEncoderTests.RelRms(got, Read(f32c)).ToString("E2") : "n/a";
                var s = ImageKl(o, (i, p, sp) => vk.ForwardWithImages(i, p, -1, sp, null, true), vk.ResetSequenceState, e);
                Log($"VITARM arm={a.Name} {name}: relRMS vs CPU-F32-control {vsCtrl:E2}; vs llama.cpp-F32-control {vsLl}; non-finite {bad}; E2E {s}; peak scratch {venc.LastPeakScratchBytes / 1e6:F1} MB");
            }
        }
        // max |activation| per ViT stage (F32 arm, traced)
        Set(arms[4]);
        foreach (string name in new[] { "rainbow80", "rainbow190x110", "rainbow640", "noise2560" })
        {
            var img = name == "noise2560" ? Inkling2VulkanVisionEncoderTests.Noise(2560) : Inkling2VisionEndToEndTests.LoadOracle(name)!.Image;
            var mx = new Dictionary<string, float>(); var resid = new List<float>();
            venc.Trace = (n, d, r, c) =>
            {
                float m = 0; foreach (float v in d) { float av = Math.Abs(v); if (av > m) m = av; }
                string stage = n.Contains('-') ? n[..n.IndexOf('-')] : n;
                mx[stage] = mx.TryGetValue(stage, out float old) ? Math.Max(old, m) : m;
                if (n.StartsWith("layer_out-")) resid.Add(m);
            };
            using (venc.Encode(img)) { }
            venc.Trace = null;
            Log($"VITMAX {name}: " + string.Join(", ", mx.Select(kv => $"{kv.Key} {kv.Value:G4}")));
            Log($"VITRESID {name}: layer_out max per layer " + string.Join(" ", resid.Select(v => v.ToString("G4"))));
        }
        // timing, interleaved, per arm, on 4 images
        Set(arms[0]);
        int rounds = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_ROUNDS"), out int rr) ? rr : 5;
        var timed = arms.Where(a => !a.Rope || a.Act == Inkling2ActFormat.Bf16).ToArray();   // F16 / F32 table variants cost the same as the plain arms
        foreach (string name in new[] { "rainbow80", "rainbow190x110", "rainbow640", "noise2560" })
        {
            var img = name == "noise2560" ? Inkling2VulkanVisionEncoderTests.Noise(2560) : Inkling2VisionEndToEndTests.LoadOracle(name)!.Image;
            int r = name == "noise2560" ? Math.Min(rounds, 3) : rounds;
            var times = timed.ToDictionary(a => a.Name, _ => new List<double>());
            var scratch = timed.ToDictionary(a => a.Name, _ => 0L);
            for (int round = 0; round <= r; round++)
                foreach (var a in (round % 2 == 0 ? timed : timed.Reverse().ToArray()))
                {
                    Set(a);
                    var sw = Stopwatch.StartNew();
                    using (venc.Encode(img)) { }
                    double wall = sw.Elapsed.TotalSeconds;
                    if (round > 0) times[a.Name].Add(wall);
                    scratch[a.Name] = Math.Max(scratch[a.Name], venc.LastPeakScratchBytes);
                }
            foreach (var a in timed)
                Log($"VITTIME arm={a.Name} {name}: encode s {times[a.Name].Min():F3}-{times[a.Name].Max():F3} (median {times[a.Name].OrderBy(x => x).ElementAt(times[a.Name].Count / 2):F3}); peak scratch {scratch[a.Name] / 1e6:F1} MB");
        }
        // 4096-token relRMS vs the CPU shipping encoder (no F32-control cache for it)
        string cpuNoise = Path.Combine(cache, "cpu_bf16_noise2560.f32");
        if (File.Exists(cpuNoise))
        {
            var cpuRef = Read(cpuNoise);
            foreach (var a in arms.Where(a => !a.Rope)) { Set(a); using var e = venc.Encode(Inkling2VulkanVisionEncoderTests.Noise(2560)); Log($"VITNOISE arm={a.Name}: relRMS vs CPU shipping encoder {Inkling2VulkanVisionEncoderTests.RelRms(Inkling2VulkanVisionEncoderTests.Floats(e), cpuRef):E2}; non-finite {Inkling2VulkanVisionEncoderTests.Floats(e).Count(v => !float.IsFinite(v))}"); }
        }
    }
}
