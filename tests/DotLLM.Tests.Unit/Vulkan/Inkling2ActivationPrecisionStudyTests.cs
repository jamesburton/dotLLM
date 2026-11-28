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
/// GPU tests need the Strix GPU lock (and cpu-bench where the CPU encoder runs in the process). The CPU storage-format simulation used for the 12c mixed-placement attempt and the max-|activation| census (an `Inkling2ActSim` hook in the CPU reference) is not part of the shipped tree, see docs/INKLING2.md 12c.
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
    private static KlStat ImageKl(Inkling2VisionEndToEndTests.Oracle o, Func<int[], int[], IReadOnlyList<ImageSpan>, ITensor> fwd, Action reset, ImageEmbeddings emb, int sub = 32, List<float[]>? keepRows = null)
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
            keepRows?.Add(ours);
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

    // ───────────────────────────── noise-free metric ─────────────────────────────
    // The 13-row oracle KL swings by a few 1e-3 on a single input rounding (docs 12a), more than the arms differ. The ranking metric is therefore the KL of each arm against
    // the F32-activation arm of the SAME engine over every prompt row (before the image / image rows / after the image): no oracle noise, only the storage format.

    private static float[][] PromptRows(Inkling2VisionEndToEndTests.Oracle o, ImageEmbeddings emb, Func<int[], int[], IReadOnlyList<ImageSpan>, ITensor> fwdAll, Action reset, int V)
    {
        var prompt = o.Before.Concat(Enumerable.Repeat(200054, emb.Tokens)).Concat(o.After).ToArray();
        var spans = new[] { new ImageSpan(o.Before.Length, emb) };
        var rows = new float[prompt.Length][];
        reset();
        for (int at = 0; at < prompt.Length; at += 32)
        {
            int len = Math.Min(32, prompt.Length - at);
            using var t = fwdAll(prompt[at..(at + len)], Enumerable.Range(at, len).ToArray(), spans);
            for (int r = 0; r < len; r++) rows[at + r] = Row(t, r, V);
        }
        return rows;
    }

    /// <summary>KL(F32 arm || this arm) over the 13 teacher-forced rows that the oracle KL scores (last prompt row + the 12 generation steps): the same rows, no oracle noise.</summary>
    private static string Nf13(List<float[]> reference, List<float[]> got)
    {
        double sum = 0, max = 0;
        for (int i = 0; i < reference.Count; i++) { double kl = Inkling2VisionEndToEndTests.FullKl(reference[i], got[i]).Kl; sum += kl; max = Math.Max(max, kl); }
        return $"NF13 (vs-F32, same 13 rows) mean {sum / reference.Count * 1e3:F3}e-3 max {max * 1e3:F2}e-3";
    }

    private static string NoiseFree(float[][] reference, float[][] got, int b0, int b1)
    {
        double kb = 0, ki = 0, ka = 0;
        for (int r = 0; r < reference.Length; r++)
        {
            double kl = Inkling2VisionEndToEndTests.FullKl(reference[r], got[r]).Kl;
            if (r < b0) kb += kl; else if (r < b1) ki += kl; else ka += kl;
        }
        int n = reference.Length;
        return $"vs-F32 KL x1e-3: before {kb / Math.Max(1, b0) * 1e3:F3}, image {ki / (b1 - b0) * 1e3:F3}, after {ka / Math.Max(1, n - b1) * 1e3:F3}, all {(kb + ki + ka) / n * 1e3:F3}";
    }

    private static float[][] TextRows(int V, Func<int[], int[], ITensor> fwdAll, Action reset)
    {
        var r = Inkling2RealModelTests.LoadKlRef(Passage)!;
        int first = r.NCtx / 2;
        var ids = r.Tokens.AsSpan(0, r.NCtx).ToArray();
        reset();
        var rows = new float[r.NCtx][];
        for (int at = 0; at < r.NCtx; at += 32)
        {
            int len = Math.Min(32, r.NCtx - at);
            using var part = fwdAll(ids[at..(at + len)], Enumerable.Range(at, len).ToArray());
            for (int k = 0; k < len; k++) if (at + k >= first) rows[at + k] = Row(part, k, V);
        }
        return rows.Skip(first).Take(r.NCtx - 1 - first).ToArray();
    }

    private static string NoiseFreeText(float[][] reference, float[][] got)
    {
        double s = 0; for (int i = 0; i < reference.Length; i++) s += Inkling2VisionEndToEndTests.FullKl(reference[i], got[i]).Kl;
        return $"vs-F32 KL x1e-3 {s / reference.Length * 1e3:F3} over {reference.Length} rows";
    }

    // ───────────────────────────── GPU: LLM arms ─────────────────────────────

    /// <summary>
    /// KL of each Vulkan activation arm on the 3 images (CPU encoder, shipping BF16 embeddings, 32-token calls) and the text passage with greedy identity: against the llama.cpp oracle (absolute) and
    /// against this engine's F32 arm (noise-free). The F32 arm runs first and is the reference. Also proves the per-chunk F32 switch: rainbow80 prefill under bf16+f32img is bit-identical to f32.
    /// </summary>
    [SkippableFact]
    public void Llm_Arms_Kl()
    {
        Gate();
        Skip.IfNot(VulkanDevice.IsAvailable(), "no vulkan");
        string? gguf = Inkling2RealModelTests.FindGguf(); Skip.If(gguf is null);
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null);
        string[] arms = (Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_ARMS") ?? "f32,bf16,f16,bf16+f32img").Split(',');
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        int V = cfg.VocabSize;
        using var enc = Inkling2VisionEncoder.Load(mm!, ThreadingConfig.Auto);
        var embs = Images.ToDictionary(n => n, n => enc.Encode(Inkling2VisionEndToEndTests.LoadOracle(n)!.Image));
        var refImg = new Dictionary<string, float[][]>(); float[][]? refText = null; float[]? refLast80 = null; var ref13 = new Dictionary<string, List<float[]>>();
        using var dev = VulkanDevice.Create();
        try
        {
            foreach (string arm in arms)
            {
                using var vk = VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, Inkling2VulkanTests.SpvDir(), Arm(arm));
                if (arm == arms[0]) Log($"LLMLOAD {vk.WeightLoadReport}");
                foreach (string name in Images)
                {
                    var o = Inkling2VisionEndToEndTests.LoadOracle(name)!;
                    var emb = embs[name];
                    var keep = new List<float[]>();
                    var s = ImageKl(o, (i, p, sp) => vk.ForwardWithImages(i, p, -1, sp, null, true), vk.ResetSequenceState, emb, 32, keep);
                    var rows = PromptRows(o, emb, (i, p, sp) => vk.ForwardWithImages(i, p, -1, sp, null, false), vk.ResetSequenceState, V);
                    if (arm == "f32") { refImg[name] = rows; ref13[name] = keep; }
                    string nf = refImg.TryGetValue(name, out var rf) ? NoiseFree(rf, rows, o.Before.Length, o.Before.Length + emb.Tokens) : "no f32 reference yet";
                    Log($"LLMKL arm={arm} {name}: oracle {s}; {(ref13.TryGetValue(name, out var r13) ? Nf13(r13, keep) : "no f32 ref")}; all-prompt-rows {nf}");
                }
                {   // single-pass (128-token chunks) last-row logits of rainbow80: the per-chunk switch must reproduce the f32 arm exactly
                    var o = Inkling2VisionEndToEndTests.LoadOracle("rainbow80")!; var emb = embs["rainbow80"];
                    var prompt = o.Before.Concat(Enumerable.Repeat(200054, emb.Tokens)).Concat(o.After).ToArray();
                    vk.ResetSequenceState();
                    using var t = vk.ForwardWithImages(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, [new ImageSpan(o.Before.Length, emb)], null, true);
                    var last = Row(t, 0, V);
                    if (arm == "f32") refLast80 = last;
                    else if (refLast80 is not null) Log($"LLMIDENT arm={arm} rainbow80 single-pass last-row logits bit-identical to f32: {last.AsSpan().SequenceEqual(refLast80)}; max |diff| {last.Zip(refLast80, (a, b) => Math.Abs(a - b)).Max():E2}");
                }
                if (arm != "bf16+f32img")   // no image rows in the text gate: identical to bf16 by construction
                {
                    var (kl, mc, conf) = TextGate(V, (i, p) => vk.Forward(i, p, -1), (i, p) => vk.Forward(i, p, -1, null, true), vk.ResetSequenceState, greedy: true);
                    var tr = TextRows(V, (i, p) => vk.Forward(i, p, -1), vk.ResetSequenceState);
                    if (arm == "f32") refText = tr;
                    Log($"LLMKL arm={arm} text-passage: oracle {kl}; greedy confident-mismatches {mc}/{conf}; {(refText is null ? "no f32 reference yet" : NoiseFreeText(refText, tr))}");
                }
                Log($"LLMMEM arm={arm}: ComputeMemoryBytes {vk.ComputeMemoryBytes / 1e6:F1} MB; activation scratch bf16 {vk.ActivationScratchBytes(Inkling2ActFormat.Bf16) / 1e6:F1} MB, f16 {vk.ActivationScratchBytes(Inkling2ActFormat.F16) / 1e6:F1} MB, f32 {vk.ActivationScratchBytes(Inkling2ActFormat.F32) / 1e6:F1} MB (chunk 128, hidden 2304)");
            }
        }
        finally { foreach (var e in embs.Values) e.Dispose(); }
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
        string[] arms = (Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_PERF_ARMS") ?? "bf16,f16,f32,bf16+f32img").Split(',');
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

    // ───────────────────────────── GPU: the vision tower ─────────────────────────────

    private static readonly (string Name, Inkling2ActFormat Act, bool Rope)[] VitArms =
    [
        ("bf16", Inkling2ActFormat.Bf16, false), ("bf16+rope32", Inkling2ActFormat.Bf16, true),
        ("f16", Inkling2ActFormat.F16, false), ("f16+rope32", Inkling2ActFormat.F16, true),
        ("f32", Inkling2ActFormat.F32, false), ("f32+rope32", Inkling2ActFormat.F32, true),
    ];

    private static float[] ReadF32(string path) { var b = File.ReadAllBytes(path); var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, b.Length); return f; }

    private static DecodedImage VitImage(string name) => name == "noise2560" ? Inkling2VulkanVisionEncoderTests.Noise(2560) : Inkling2VisionEndToEndTests.LoadOracle(name)!.Image;

    /// <summary>
    /// ViT accuracy per arm: embedding relRMS against the CPU F32-control encoder (cached by the ViT tests) and llama.cpp's F32 control, end-to-end KL through the (BF16, shipped) Vulkan text model, non-finite count,
    /// peak scratch. The shipped BF16 vit_gemm_q8.spv is byte-identical to the study GEMM built without defines, so the "+rope32" arms isolate the RoPE table alone.
    /// </summary>
    [SkippableFact]
    public void Vit_Accuracy()
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
        foreach (var a in VitArms)
        {
            venc.ActFormat = a.Act; venc.RopeF32 = a.Rope;
            foreach (string name in Images)
            {
                var o = Inkling2VisionEndToEndTests.LoadOracle(name)!;
                using var e = venc.Encode(o.Image);
                var got = Inkling2VulkanVisionEncoderTests.Floats(e);
                int bad = got.Count(v => !float.IsFinite(v));
                double vsCtrl = Inkling2VulkanVisionEncoderTests.RelRms(got, ReadF32(Path.Combine(cache, $"cpu_ctrl_f32_{name}.f32")));
                string f32c = Path.Combine(ctrlDir, $"llamacpp_f32_{name}.f32");
                string vsLl = File.Exists(f32c) ? Inkling2VulkanVisionEncoderTests.RelRms(got, ReadF32(f32c)).ToString("E2") : "n/a";
                var s = ImageKl(o, (i, p, sp) => vk.ForwardWithImages(i, p, -1, sp, null, true), vk.ResetSequenceState, e);
                Log($"VITARM arm={a.Name} {name}: relRMS vs CPU-F32-control {vsCtrl:E2}; vs llama.cpp-F32-control {vsLl}; non-finite {bad}; E2E oracle {s}; peak scratch {venc.LastPeakScratchBytes / 1e6:F1} MB");
            }
        }
    }

    /// <summary>Max |activation| per ViT stage (F32 arm, traced) on the 3 images and the 4,096-token noise image; per-layer residual maxima.</summary>
    [SkippableFact]
    public void Vit_MaxActivations()
    {
        Gate();
        Skip.IfNot(VulkanDevice.IsAvailable(), "no vulkan");
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null);
        using var dev = VulkanDevice.Create();
        using var venc = VulkanInkling2VisionEncoder.Load(dev, mm!, Inkling2VulkanTests.SpvDir());
        venc.ActFormat = Inkling2ActFormat.F32;
        string[] names = (Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_VIT_IMAGES") ?? "rainbow80,rainbow190x110,rainbow640,noise2560").Split(',');
        foreach (string name in names)
        {
            var mx = new Dictionary<string, float>(); var resid = new List<float>();
            venc.Trace = (n, d, r, c) =>
            {
                float m = 0; foreach (float v in d) { float av = Math.Abs(v); if (av > m) m = av; }
                string stage = n.Contains('-') ? n[..n.IndexOf('-')] : n;
                mx[stage] = mx.TryGetValue(stage, out float old) ? Math.Max(old, m) : m;
                if (n.StartsWith("layer_out-")) resid.Add(m);
            };
            using (venc.Encode(VitImage(name))) { }
            venc.Trace = null;
            Log($"VITMAX {name}: " + string.Join(", ", mx.Select(kv => $"{kv.Key} {kv.Value:G4}")));
            Log($"VITRESID {name}: layer_out max per layer " + string.Join(" ", resid.Select(v => v.ToString("G4"))));
        }
    }

    /// <summary>Encode time per arm, interleaved with a warm-up (DOTLLM_INK_PREC_VIT_IMAGES, DOTLLM_INK_PREC_ROUNDS); peak scratch per arm. 4,096-token noise image: 30-38 s per encode, so give it its own run.</summary>
    [SkippableFact]
    public void Vit_Time()
    {
        Gate();
        Skip.IfNot(VulkanDevice.IsAvailable(), "no vulkan");
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null);
        using var dev = VulkanDevice.Create();
        using var venc = VulkanInkling2VisionEncoder.Load(dev, mm!, Inkling2VulkanTests.SpvDir());
        int rounds = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_ROUNDS"), out int rr) ? rr : 5;
        string[] names = (Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_VIT_IMAGES") ?? "rainbow80,rainbow190x110,rainbow640").Split(',');
        var timed = VitArms.Where(a => !a.Rope || a.Act == Inkling2ActFormat.Bf16).ToArray();   // the F16 / F32 table variants cost the same as the plain arms
        foreach (string name in names)
        {
            var img = VitImage(name);
            int r = name == "noise2560" ? Math.Min(rounds, 2) : rounds;
            var times = timed.ToDictionary(a => a.Name, _ => new List<double>());
            var gpu = timed.ToDictionary(a => a.Name, _ => new List<double>());
            var scratch = timed.ToDictionary(a => a.Name, _ => 0L);
            for (int round = 0; round <= r; round++)
                foreach (var a in (round % 2 == 0 ? timed : timed.Reverse().ToArray()))
                {
                    venc.ActFormat = a.Act; venc.RopeF32 = a.Rope;
                    var sw = Stopwatch.StartNew();
                    using (venc.Encode(img)) { }
                    if (round > 0) { times[a.Name].Add(sw.Elapsed.TotalSeconds); gpu[a.Name].Add(venc.LastGpuSeconds); }
                    scratch[a.Name] = Math.Max(scratch[a.Name], venc.LastPeakScratchBytes);
                }
            foreach (var a in timed)
                Log($"VITTIME arm={a.Name} {name}: encode wall s {times[a.Name].Min():F3}-{times[a.Name].Max():F3} (median {times[a.Name].OrderBy(x => x).ElementAt(times[a.Name].Count / 2):F3}); GPU part median {gpu[a.Name].OrderBy(x => x).ElementAt(gpu[a.Name].Count / 2):F3}; peak scratch {scratch[a.Name] / 1e6:F1} MB");
        }
    }

    /// <summary>4,096-token noise image: relRMS of each arm against the CPU shipping encoder (cached) and the non-finite count (F16 overflow shows here first).</summary>
    [SkippableFact]
    public void Vit_Noise4096_Accuracy()
    {
        Gate();
        Skip.IfNot(VulkanDevice.IsAvailable(), "no vulkan");
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null);
        string cpuNoise = Path.Combine(Inkling2VulkanVisionEncoderTests.CacheDir, "cpu_bf16_noise2560.f32");
        Skip.IfNot(File.Exists(cpuNoise), "cpu_bf16_noise2560.f32 not cached");
        var cpuRef = ReadF32(cpuNoise);
        using var dev = VulkanDevice.Create();
        using var venc = VulkanInkling2VisionEncoder.Load(dev, mm!, Inkling2VulkanTests.SpvDir());
        foreach (var a in VitArms.Where(a => !a.Rope))
        {
            venc.ActFormat = a.Act; venc.RopeF32 = a.Rope;
            using var e = venc.Encode(Inkling2VulkanVisionEncoderTests.Noise(2560));
            var f = Inkling2VulkanVisionEncoderTests.Floats(e);
            Log($"VITNOISE arm={a.Name}: relRMS vs CPU shipping encoder {Inkling2VulkanVisionEncoderTests.RelRms(f, cpuRef):E2}; non-finite {f.Count(v => !float.IsFinite(v))}; peak scratch {venc.LastPeakScratchBytes / 1e6:F1} MB");
        }
    }
}
