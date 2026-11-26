using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Inkling-2-Mini-NVFP4 (<c>ggml-org/Inkling-2-Mini-GGUF</c>, HF hub cache) against llama.cpp b12780 CPU (<c>-dev none</c>): teacher-forced KL on a fixed
/// passage (the <c>llama-perplexity --kl-divergence-base</c> file) and greedy text identity (issue #2026). Oracle files live in
/// <c>~/.dotllm/test-cache/token-oracles/inkling2/</c>; skipped when the model or the oracle is missing. Heavy: loads a 17.6 GB model on the CPU.
/// </summary>
/// <remarks>
/// The model is fed the oracle's own token ids, so tokenizer differences cannot contaminate the model KL. The KL file stores log-probabilities
/// quantised to uint16 over [max-16, max] (llama.cpp's format), so a floor of ~1e-6 on KL is the oracle's own precision.
/// </remarks>
public sealed class Inkling2RealModelTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2RealModelTests(ITestOutputHelper output) => _output = output;

    private static string OracleDir => Environment.GetEnvironmentVariable("DOTLLM_INKLING2_ORACLE_DIR")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "token-oracles", "inkling2");

    internal static string? FindGguf()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_NVFP4_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string hub = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
            ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME") is { Length: > 0 } h ? h : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");
        string snaps = Path.Combine(hub, "models--ggml-org--Inkling-2-Mini-GGUF", "snapshots");
        if (!Directory.Exists(snaps)) return null;
        foreach (var d in Directory.GetDirectories(snaps))
        {
            string p = Path.Combine(d, "Inkling-2-Mini-NVFP4.gguf");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    private sealed record KlRef(int NCtx, int NVocab, int NChunk, int[] Tokens, string Path, long RowsOffset);

    private static KlRef? LoadKlRef()
    {
        string path = Path.Combine(OracleDir, Environment.GetEnvironmentVariable("DOTLLM_INKLING2_KLD") ?? "nvfp4-passage-c256-fa-off.kld");
        if (!File.Exists(path)) return null;
        using var f = File.OpenRead(path);
        using var br = new BinaryReader(f);
        if (new string(br.ReadChars(8)) != "_logits_") throw new InvalidDataException("not a llama.cpp logits file");
        int nCtx = br.ReadInt32(), nVocab = br.ReadInt32(), nChunk = br.ReadInt32();
        var tokens = new int[nCtx * nChunk];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = br.ReadInt32();
        return new KlRef(nCtx, nVocab, nChunk, tokens, path, f.Position);
    }

    /// <summary>Per-position log-probabilities of the base (oracle) model for one chunk: <c>[nScored][nVocab]</c>, floor-clamped like llama.cpp.</summary>
    private static float[][] ReadChunk(KlRef r, int chunk)
    {
        int first = r.NCtx / 2, nScored = r.NCtx - 1 - first;
        int nv = 2 * ((r.NVocab + 1) / 2) + 4;                 // uint16 slots per row (llama.cpp layout)
        long rowBytes = (long)nv * 2;
        using var f = File.OpenRead(r.Path);
        f.Seek(r.RowsOffset + chunk * nScored * rowBytes, SeekOrigin.Begin);
        var rows = new float[nScored][];
        var buf = new byte[rowBytes];
        for (int i = 0; i < nScored; i++)
        {
            f.ReadExactly(buf);
            float scale = BitConverter.ToSingle(buf, 0), min = BitConverter.ToSingle(buf, 4);
            var lp = new float[r.NVocab];
            for (int v = 0; v < r.NVocab; v++) lp[v] = scale * BitConverter.ToUInt16(buf, 8 + 2 * v) + min;
            rows[i] = lp;
        }
        return rows;
    }

    private static unsafe float[] Rows(ITensor t, int rows, int vocab)
    {
        var a = new float[rows * vocab];
        new ReadOnlySpan<float>((void*)t.DataPointer, a.Length).CopyTo(a);
        return a;
    }

    /// <summary><c>sum_i p_base(i) * (log p_base(i) - log p_ours(i))</c> over the entries llama.cpp keeps (log p_base &gt; -16), and top-1 agreement.</summary>
    internal static (double Kl, bool SameTop) Kl(ReadOnlySpan<float> baseLogProb, ReadOnlySpan<float> logits)
    {
        float max = float.NegativeInfinity; int amax = 0;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > max) { max = logits[i]; amax = i; }
        double sum = 0;
        for (int i = 0; i < logits.Length; i++) sum += Math.Exp(logits[i] - max);
        double lse = max + Math.Log(sum);
        double kl = 0; int bmax = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (baseLogProb[i] > baseLogProb[bmax]) bmax = i;
            if (baseLogProb[i] > -16f)
            {
                double pb = Math.Exp(baseLogProb[i]);
                kl += pb * (baseLogProb[i] - (logits[i] - lse));
            }
        }
        return (kl, amax == bmax);
    }

    private (double MeanKl, double MaxKl, double Top1, int N) ScoreKl(Inkling2TransformerModel model, KlRef r)
    {
        int first = r.NCtx / 2, nScored = r.NCtx - 1 - first, V = model.Config.VocabSize;
        double sum = 0, max = 0; int top = 0, n = 0;
        for (int chunk = 0; chunk < r.NChunk; chunk++)
        {
            var baseRows = ReadChunk(r, chunk);
            var ids = r.Tokens.AsSpan(chunk * r.NCtx, r.NCtx).ToArray();
            model.ResetSequenceState();
            // Sub-chunked prefill (default 32 tokens per call) so the carried KV / conv state is exercised, not only the in-call window.
            int sub = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_INKLING2_KL_SUBCHUNK"), out int sc) && sc > 0 ? sc : 32;
            var got = new float[r.NCtx * V];
            for (int at = 0; at < r.NCtx; at += sub)
            {
                int len = Math.Min(sub, r.NCtx - at);
                using var part = model.Forward(ids.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1);
                Rows(part, len, V).CopyTo(got, at * V);
            }
            for (int i = 0; i < nScored; i++)
            {
                var (kl, same) = Kl(baseRows[i], got.AsSpan((first + i) * V, V));
                sum += kl; max = Math.Max(max, kl); if (same) top++; n++;
                if (Environment.GetEnvironmentVariable("DOTLLM_INKLING2_KL_TRACE") == "1")
                    _output.WriteLine($"  chunk {chunk} pos {first + i}: KL {kl:E2}");
            }
        }
        return (sum / n, max, (double)top / n, n);
    }

    private static Inkling2TransformerModel Load(GgufFile file, Inkling2Options options, Inkling2Mutant mutant = Inkling2Mutant.None)
    {
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        var m = Inkling2TransformerModel.LoadFromGguf(file, cfg, ThreadingConfig.Auto, options);
        m.Mutant = mutant;
        return m;
    }

    /// <summary>Greedy-identity scores: free-running prefix identity per prompt, and teacher-forced argmax disagreements split by the oracle's top-1/top-2 margin.</summary>
    private sealed record GreedyScore(int[] Prefix, int Steps, int Mismatch, int MismatchConfident, int Confident);

    private GreedyScore ScoreGreedy(Inkling2TransformerModel model, double confidentMargin = 0.3)
    {
        string path = Path.Combine(OracleDir, Environment.GetEnvironmentVariable("DOTLLM_INKLING2_GREEDY") ?? "greedy-nvfp4-fa-off.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var prefix = new List<int>();
        int steps = 0, mismatch = 0, mismatchConf = 0, confident = 0;
        foreach (var p in doc.RootElement.EnumerateObject())
        {
            int[] prompt = p.Value.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] want = p.Value.GetProperty("gen_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            double[] margin = p.Value.TryGetProperty("margins", out var mg) ? mg.EnumerateArray().Select(e => e.GetDouble()).ToArray() : new double[want.Length];

            // free-running greedy
            model.ResetSequenceState();
            var got = new List<int>();
            using (var pre = model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, lastTokenLogitsOnly: true))
            {
                int next = ArgMax(Rows(pre, 1, model.Config.VocabSize));
                for (int j = 0; j < want.Length; j++)
                {
                    got.Add(next);
                    if (j + 1 == want.Length) break;
                    using var step = model.Forward([next], [prompt.Length + j], -1, null, lastTokenLogitsOnly: true);
                    next = ArgMax(Rows(step, 1, model.Config.VocabSize));
                }
            }
            int same = 0;
            while (same < want.Length && got[same] == want[same]) same++;
            prefix.Add(same);

            // teacher-forced: feed the oracle's tokens, compare the argmax at every step
            model.ResetSequenceState();
            int pm = 0;
            using (var pre = model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, lastTokenLogitsOnly: true))
            {
                int arg = ArgMax(Rows(pre, 1, model.Config.VocabSize));
                for (int j = 0; j < want.Length; j++)
                {
                    steps++;
                    bool conf = margin[j] >= confidentMargin;
                    if (conf) confident++;
                    if (arg != want[j]) { mismatch++; pm++; if (conf) mismatchConf++; _output.WriteLine($"  teacher-forced {p.Name} step {j}: ours {arg} oracle {want[j]} (oracle margin {margin[j]:F3})"); }
                    if (j + 1 == want.Length) break;
                    using var step = model.Forward([want[j]], [prompt.Length + j], -1, null, lastTokenLogitsOnly: true);
                    arg = ArgMax(Rows(step, 1, model.Config.VocabSize));
                }
            }
            _output.WriteLine($"greedy/{p.Name}: free-running {same}/{want.Length} identical; teacher-forced mismatches {pm}/{want.Length}");
        }
        return new GreedyScore(prefix.ToArray(), steps, mismatch, mismatchConf, confident);
    }

    private static int ArgMax(float[] v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }

    [SkippableFact]
    public void Nvfp4_MatchesLlamaCppCpu_KlAndGreedy()
    {
        string? gguf = FindGguf(); var kl = LoadKlRef();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        Skip.If(kl is null, "oracle file nvfp4-passage-c256.kld missing");
        using var file = GgufFile.Open(gguf!);
        using var model = Load(file, Inkling2Options.FromEnvironment());
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (mean, max, top1, n) = ScoreKl(model, kl!);
        _output.WriteLine($"KL vs llama.cpp CPU ({model.Options}): mean {mean:E3} max {max:E3} top-1 {top1:P1} over {n} positions in {sw.Elapsed.TotalSeconds:F0}s");
        var g = ScoreGreedy(model);
        _output.WriteLine($"greedy summary: free-running prefixes {string.Join("/", g.Prefix)} of 48; teacher-forced mismatches {g.Mismatch}/{g.Steps} ({g.MismatchConfident}/{g.Confident} at oracle margin >= 0.3)");
        Assert.True(mean < 8e-3, $"mean KL {mean:E3} too high");
        Assert.True(top1 > 0.95, $"top-1 agreement {top1:P1} too low");
        Assert.True(g.MismatchConfident == 0, $"{g.MismatchConfident} teacher-forced argmax disagreements where the oracle was confident (margin >= 0.3)");
    }

    [SkippableTheory]
    [InlineData("TwoPasses")]
    [InlineData("NoFinalSoftcap")]
    [InlineData("NoSink")]
    [InlineData("NoRelativeBias")]
    [InlineData("NoExpertScale")]
    [InlineData("NoControl")]
    public void Mutants_AreDetectedByTheGate(string mutantName)
    {
        string? gguf = FindGguf(); var kl = LoadKlRef();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        Skip.If(kl is null, "oracle file nvfp4-passage-c256.kld missing");
        using var file = GgufFile.Open(gguf!);
        var mutant = Enum.Parse<Inkling2Mutant>(mutantName);
        using var model = Load(file, new Inkling2Options(), mutant);
        var (mean, max, top1, n) = ScoreKl(model, kl!);
        _output.WriteLine($"MUTANT {mutant}: KL mean {mean:E3} max {max:E3} top-1 {top1:P1} over {n} positions");
        Assert.True(mean >= 5e-3 || top1 <= 0.9, $"mutant {mutant} survived the gate: KL {mean:E3}, top-1 {top1:P1}");
    }

    /// <summary>
    /// <c>NoLogScale</c> is deliberately NOT in the mutant theory: <c>1 + alpha*ln(max(1, (pos+1)/8192))</c> is exactly 1 below position 8191, so a
    /// 256-token passage cannot observe it (measured: KL identical to the unmutated run, 3.959E-003). It is pinned at the op level instead
    /// (<c>Inkling2OpsTests.LogScale_*</c>).
    /// </summary>
    [SkippableTheory]
    [InlineData("f16")]
    [InlineData("bf16")]
    public void ConvStateStorage_Sweep(string storage)
    {
        string? gguf = FindGguf(); var kl = LoadKlRef();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        Skip.If(kl is null, "oracle KL file missing");
        var cs = storage == "f16" ? Inkling2ConvStateStorage.F16 : Inkling2ConvStateStorage.BF16;
        using var file = GgufFile.Open(gguf!);
        using var model = Load(file, new Inkling2Options(ConvState: cs));
        var (mean, max, top1, n) = ScoreKl(model, kl!);
        _output.WriteLine($"CONV-STATE {storage}: KL mean {mean:E3} max {max:E3} top-1 {top1:P1} over {n}");
        var g = ScoreGreedy(model);
        _output.WriteLine($"CONV-STATE {storage}: greedy free-running prefixes {string.Join("/", g.Prefix)} of 48; teacher-forced mismatches {g.Mismatch}/{g.Steps}, confident {g.MismatchConfident}/{g.Confident}");
    }

    /// <summary>Prefill and decode throughput of the CPU reference (printed, not asserted). Hold the <c>cpu-bench</c> lock when running it.</summary>
    [SkippableFact]
    public void Throughput_PrefillAndDecode()
    {
        string? gguf = FindGguf();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        using var file = GgufFile.Open(gguf!);
        using var model = Load(file, Inkling2Options.FromEnvironment());
        var kl = LoadKlRef();
        int[] prompt = kl is null ? Enumerable.Range(1000, 128).ToArray() : kl.Tokens.AsSpan(0, 128).ToArray();
        model.ResetSequenceState();
        using (model.Forward(prompt.AsSpan(0, 8), Enumerable.Range(0, 8).ToArray(), -1, null, true)) { }   // warm-up (JIT, page cache)
        model.ResetSequenceState();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using (model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, true)) { }
        double pre = sw.Elapsed.TotalSeconds;
        int next = 1234, steps = 32;
        sw.Restart();
        for (int j = 0; j < steps; j++)
        {
            using var t = model.Forward([next], [prompt.Length + j], -1, null, true);
            next = ArgMax(Rows(t, 1, model.Config.VocabSize));
        }
        double dec = sw.Elapsed.TotalSeconds;
        _output.WriteLine($"THROUGHPUT prefill {prompt.Length} tokens: {prompt.Length / pre:F1} tok/s ({pre:F2}s); decode {steps} steps: {steps / dec:F2} tok/s ({dec / steps * 1000:F0} ms/token); threads {Environment.ProcessorCount}");
    }

    [SkippableTheory]
    [InlineData("q8_0")]
    [InlineData("f16")]
    [InlineData("bf16")]
    public void LoopControllerStorage_Sweep(string storage)
    {
        string? gguf = FindGguf(); var kl = LoadKlRef();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        Skip.If(kl is null, "oracle file nvfp4-passage-c256.kld missing");
        var qt = storage switch { "q8_0" => QuantizationType.Q8_0, "f16" => QuantizationType.F16, _ => QuantizationType.BF16 };
        using var file = GgufFile.Open(gguf!);
        using var model = Load(file, new Inkling2Options(LoopController: qt));
        var (mean, max, top1, n) = ScoreKl(model, kl!);
        _output.WriteLine($"LOOP-CONTROLLER {storage}: {model.LoopControllerBytes / 1e6:F1} MB; KL mean {mean:E3} max {max:E3} top-1 {top1:P1} over {n}");
        var g = ScoreGreedy(model);
        _output.WriteLine($"LOOP-CONTROLLER {storage}: greedy free-running prefixes {string.Join("/", g.Prefix)} of 48; teacher-forced mismatches {g.Mismatch}/{g.Steps}, confident {g.MismatchConfident}/{g.Confident}");
    }
}
