using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// End-to-end parity + pp (prompt-processing) timing for the wired G-flash hand-fused
/// mma.sync prefill-attention path against the FP32 CUDA-core <c>attention_f16</c>
/// baseline, on a real <see cref="CudaTransformerModel"/> forward. Mirrors
/// <see cref="CudaG3PrefillForwardHarness"/>: both arms run in ONE warmed process with the
/// dispatch toggled between fresh KV caches (interleaved, min-of-N; never divide separate
/// fresh-process minima — the 3060 drifts ~2× across heavy runs).
/// </summary>
/// <remarks>
/// <para>
/// <b>Arms.</b> OFF = flash and G3 both disabled → <c>attention_f16</c> (the FP32 reference).
/// ON = flash enabled, G3 disabled → the fused mma.sync kernel for every layer whose call
/// clears <see cref="CudaFlashAttention.CanUse"/> (square-causal global prefill, headDim 64,
/// positionOffset 0). The crossover is pinned to 1 for the test so flash fires at every
/// swept length, not only s ≥ the production threshold.
/// </para>
/// <para>
/// <b>Proof of fire.</b> The isolated-kernel parity test
/// (<see cref="CudaTensorCoreAttentionParityTests.FlashMmaPath_MatchesAttentionF16"/>) is
/// necessary but not sufficient — an end-to-end "parity" pass is vacuous if
/// <see cref="CudaFlashAttention.CanUse"/> silently fell through to attention_f16 on BOTH
/// arms. This harness asserts <see cref="CudaFlashAttention.DispatchCount"/> advanced during
/// the ON forward (one launch per eligible layer) before trusting any parity / perplexity
/// number, and verifies the OFF arm did NOT launch flash.
/// </para>
/// <para>
/// Parity gate is layer-0 attention OUTPUT on real forward data (abs/rel 5e-3, coopmat FP16
/// precedent); last-token LOGITS amplify the small FP16 difference past 5e-3 over 16 layers
/// + the 128k LM head, so those are gated by greedy-top-1 match + a &lt;1% prefill-perplexity
/// move (the G3/G1 precedent), not by tight logit tolerance. Opt-in via
/// <c>DOTLLM_CUDA_FLASH_E2E=1</c>; model via <c>DOTLLM_CUDA_FLASH_GGUF</c> or the cached
/// Llama-3.2-1B-Instruct-Q8_0.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaFlashPrefillForwardHarness
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 5e-3f;

    // ProfileCategory.Attention index (see CudaTransformerModel.ProfileCategory).
    private const int AttentionCategory = 4;

    // Mirror diagnostic lines to a file when DOTLLM_CUDA_FLASH_REPORT is set — `dotnet test`
    // does not surface ITestOutputHelper to stdout on a PASS, so a passing run's numbers
    // (parity, perplexity, the flash-vs-G3 table) would otherwise be lost.
    private readonly string? _reportPath = Environment.GetEnvironmentVariable("DOTLLM_CUDA_FLASH_REPORT");

    private readonly ITestOutputHelper _output;

    public CudaFlashPrefillForwardHarness(ITestOutputHelper output)
    {
        if (_reportPath is not null) File.WriteAllText(_reportPath, string.Empty);
        _output = new TeeOutput(output, _reportPath);
    }

    /// <summary>Test-output facade that also appends to the report file when configured.</summary>
    private sealed class TeeOutput(ITestOutputHelper inner, string? path) : ITestOutputHelper
    {
        public void WriteLine(string message)
        {
            inner.WriteLine(message);
            if (path is not null) File.AppendAllText(path, message + Environment.NewLine);
        }

        public void WriteLine(string format, params object[] args) => WriteLine(string.Format(format, args));
    }

    [SkippableFact]
    public void FlashPrefill_ParityAndPpSpeedup()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_CUDA_FLASH_E2E") == "1", "DOTLLM_CUDA_FLASH_E2E=1 not set.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string? modelPath = ResolveModelPath();
        Skip.If(modelPath is null, "No model. Set DOTLLM_CUDA_FLASH_GGUF or cache Llama-3.2-1B-Q8_0.");
        string ptxDir = ResolvePtxDir();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "attention_flash_mma.ptx")), "attention_flash_mma.ptx not built.");

        _output.WriteLine($"model={modelPath}");
        using var gguf = GgufFile.Open(modelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        Skip.IfNot(config.HeadDim == 64,
            $"Flash prototype is headDim==64 only; model headDim={config.HeadDim} won't engage flash.");

        int maxLayers = ParseEnvInt("DOTLLM_CUDA_FLASH_MAXLAYERS", 0);
        if (maxLayers > 0) { model.DebugMaxLayers = maxLayers; _output.WriteLine($"DebugMaxLayers={maxLayers}"); }

        int[] seqs = ParseSeqs(Environment.GetEnvironmentVariable("DOTLLM_CUDA_FLASH_SEQS")) ?? [512, 1024, 2048];
        int reps = ParseEnvInt("DOTLLM_CUDA_FLASH_REPS", 25);
        int warmup = ParseEnvInt("DOTLLM_CUDA_FLASH_WARMUP", 5);

        // Force the crossover to 1 so flash engages at EVERY swept length in this test (the
        // production default keeps short sequences on G3). Read once at type init from the
        // env var, so this test process must have DOTLLM_CUDA_FLASH_ATTN_MINSEQ=1 set; assert
        // it so a misconfigured run skips rather than silently testing only s>=1024.
        Skip.IfNot(CudaFlashAttention.CrossoverSeqLen <= 512,
            $"Set DOTLLM_CUDA_FLASH_ATTN_MINSEQ=1 so flash engages at all swept lengths "
          + $"(current crossover={CudaFlashAttention.CrossoverSeqLen}).");

        int parityLen = seqs[^1];

        // ── Check 1: OFF/OFF determinism (the baseline the whole A/B rests on). ──
        float[] off1 = PrefillLastTokenLogits(model, config, parityLen, flashOn: false);
        float[] off2 = PrefillLastTokenLogits(model, config, parityLen, flashOn: false);
        int offDiff = 0; float offMaxAbs = 0f;
        for (int i = 0; i < off1.Length; i++) { float d = MathF.Abs(off1[i] - off2[i]); if (d > 0) offDiff++; if (d > offMaxAbs) offMaxAbs = d; }
        _output.WriteLine($"=== Check1 OFF/OFF determinism: differing={offDiff}/{off1.Length} maxAbs={offMaxAbs:E3} ===");

        // ── Check 2: PROOF OF FIRE — flash actually launched on the ON arm, not the OFF arm. ──
        // Without this an E2E parity pass is vacuous (CanUse could fall through to attention_f16
        // on both arms). One flash launch per eligible layer per forward.
        int numLayers = maxLayers > 0 ? Math.Min(maxLayers, config.NumLayers) : config.NumLayers;

        long beforeOff = CudaFlashAttention.DispatchCount;
        _ = PrefillLastTokenLogits(model, config, parityLen, flashOn: false);
        long offLaunches = CudaFlashAttention.DispatchCount - beforeOff;

        long beforeOn = CudaFlashAttention.DispatchCount;
        _ = PrefillLastTokenLogits(model, config, parityLen, flashOn: true);
        long onLaunches = CudaFlashAttention.DispatchCount - beforeOn;

        _output.WriteLine($"=== Check2 proof-of-fire: flash launches OFF={offLaunches} ON={onLaunches} "
            + $"(expected ON={numLayers}, OFF=0) ===");
        Assert.True(offLaunches == 0, $"Flash launched {offLaunches}× on the OFF arm — A/B is contaminated.");
        Assert.True(onLaunches == numLayers,
            $"Flash launched {onLaunches}× on the ON arm at s={parityLen}, expected {numLayers} (one per layer). "
          + "Either CanUse fell through (parity would be vacuous) or layer count differs.");

        // ── Check 3: layer-0 attention-OUTPUT parity on real data (before downstream amplification). ──
        ushort[] attnOff = CaptureLayer0AttnOutput(model, config, parityLen, flashOn: false);
        ushort[] attnOn = CaptureLayer0AttnOutput(model, config, parityLen, flashOn: true);
        int attnMis = 0; float attnMaxAbs = 0f, attnMaxRel = 0f;
        for (int i = 0; i < attnOff.Length; i++)
        {
            float a = (float)BitConverter.UInt16BitsToHalf(attnOff[i]);
            float b = (float)BitConverter.UInt16BitsToHalf(attnOn[i]);
            float abs = MathF.Abs(a - b), rel = abs / (MathF.Abs(a) + 1e-6f);
            if (!(abs <= AbsTol || rel <= RelTol)) attnMis++;
            if (abs > attnMaxAbs) { attnMaxAbs = abs; attnMaxRel = rel; }
        }
        _output.WriteLine($"=== Check3 layer-0 attn-output parity: mismatches={attnMis}/{attnOff.Length} "
            + $"worst abs={attnMaxAbs:E3} rel={attnMaxRel:E3} (tol abs {AbsTol} OR rel {RelTol}) ===");

        // PRIMARY PARITY GATE — attention output on real data. The flash path differs from
        // attention_f16 only in FP16 intermediates (P stored FP16 between QK and PV, MMA FP16
        // inputs); at the attention-output level this holds the 5e-3 coopmat-FP16 bar.
        Assert.True(attnMis == 0,
            $"Flash attention output diverges from attention_f16 on real data: {attnMis}/{attnOff.Length} "
          + $"outside tol; worst abs {attnMaxAbs} rel {attnMaxRel}. This WOULD indicate a wiring bug.");

        // ── 1. Last-token logit divergence (DIAGNOSTIC — downstream amplification). ──
        _output.WriteLine($"=== last-token logits OFF vs ON (diagnostic; gated by perplexity, not tol) ===");
        float[] logitsOff = off1;
        float[] logitsOn = PrefillLastTokenLogits(model, config, parityLen, flashOn: true);

        int mismatches = 0, vocab = logitsOff.Length;
        float maxAbs = 0f, maxRel = 0f;
        int worst = -1;
        for (int i = 0; i < vocab; i++)
        {
            float a = logitsOff[i], b = logitsOn[i];
            float abs = MathF.Abs(a - b);
            float rel = abs / (MathF.Abs(a) + 1e-6f);
            if (!(abs <= AbsTol || rel <= RelTol)) { mismatches++; if (abs > maxAbs) { maxAbs = abs; maxRel = rel; worst = i; } }
        }
        int argmaxOff = Argmax(logitsOff), argmaxOn = Argmax(logitsOn);
        _output.WriteLine($"vocab={vocab} logits>5e-3={mismatches} worst_abs={maxAbs:E3} worst_rel={maxRel:E3} @ {worst}");
        _output.WriteLine($"argmax OFF={argmaxOff} ON={argmaxOn} (top-1 {(argmaxOff == argmaxOn ? "MATCH" : "DIFFER")})");
        Assert.True(argmaxOff == argmaxOn, $"Flash changed greedy top-1 token ({argmaxOff} → {argmaxOn}).");

        // ── 1b. Quality gate: growing-prefix prefill perplexity OFF vs ON (< 1%, G1 precedent). ──
        int pplLen = ParseEnvInt("DOTLLM_CUDA_FLASH_PPL_TOKENS", 192);
        int[] pplTokens = RealCorpusTokens(gguf, pplLen);
        _output.WriteLine($"ppl corpus: {pplTokens.Length} real tokens");
        double pplOff = PrefillGrowingPrefixPerplexity(model, pplTokens, vocab, flashOn: false);
        double pplOn = PrefillGrowingPrefixPerplexity(model, pplTokens, vocab, flashOn: true);
        double pplRatio = pplOn / pplOff;
        _output.WriteLine($"=== prefill PPL: off={pplOff:F5} on={pplOn:F5} ratio={pplRatio:F5} "
            + $"({(pplRatio - 1) * 100:+0.000;-0.000}%) gate <1% ===");
        Assert.True(Math.Abs(pplRatio - 1.0) < 0.01,
            $"Flash prefill perplexity moved {(pplRatio - 1) * 100:+0.00;-0.00}% (off={pplOff:F4} on={pplOn:F4}) — exceeds 1% gate.");

        // ── 2. pp timing: interleaved OFF(attention_f16)/ON(flash) and the G3 arm, min-of-N. ──
        model.ProfilingEnabled = true;
        _output.WriteLine($"\n=== pp timing (interleaved, min of {reps}, GPU ms) ===");
        _output.WriteLine($"{"seq",6} | {"f16 (ms)",10} | {"G3 (ms)",10} | {"flash (ms)",11} | {"f16/fl",7} | {"G3/fl",7} | {"attn f16",9} | {"attn G3",9} | {"attn fl",9} | {"aG3/afl",8}");
        _output.WriteLine(new string('-', 120));

        foreach (int s in seqs)
        {
            int[] prompt = SyntheticPrompt(config, s);
            int[] positions = Positions(s);

            (double Whole, double Attn) MeasureOnce(bool flashOn, bool g3On)
            {
                CudaFlashAttention.Enabled = flashOn;
                CudaG3Attention.Enabled = g3On;
                using var cache = model.CreateKvCache(maxSeqLen: s + 4);
                using ITensor logits = model.Forward(prompt, positions, deviceId: 0, cache);
                return (model.LastGpuLaunchMs, model.LastCategoryMs[AttentionCategory]);
            }

            // f16: flash off, G3 off. G3: flash off, G3 on. flash: flash on (wins dispatch).
            for (int w = 0; w < warmup; w++) { MeasureOnce(false, false); MeasureOnce(false, true); MeasureOnce(true, false); }

            double f16Whole = double.MaxValue, g3Whole = double.MaxValue, flWhole = double.MaxValue;
            double f16Attn = double.MaxValue, g3Attn = double.MaxValue, flAttn = double.MaxValue;
            for (int r = 0; r < reps; r++)
            {
                var f16 = MeasureOnce(false, false);
                var g3 = MeasureOnce(false, true);
                var fl = MeasureOnce(true, false);
                f16Whole = Math.Min(f16Whole, f16.Whole); f16Attn = Math.Min(f16Attn, f16.Attn);
                g3Whole = Math.Min(g3Whole, g3.Whole); g3Attn = Math.Min(g3Attn, g3.Attn);
                flWhole = Math.Min(flWhole, fl.Whole); flAttn = Math.Min(flAttn, fl.Attn);
            }

            _output.WriteLine($"{s,6} | {f16Whole,8:F3}ms | {g3Whole,8:F3}ms | {flWhole,9:F3}ms | "
                + $"{f16Whole / flWhole,6:F2}x | {g3Whole / flWhole,6:F2}x | "
                + $"{f16Attn,7:F3}ms | {g3Attn,7:F3}ms | {flAttn,7:F3}ms | {g3Attn / flAttn,7:F2}x");
        }
        _output.WriteLine(new string('-', 120));
        _output.WriteLine("f16/fl, G3/fl = whole-prefill GPU speedup of flash over each baseline (>1 = flash wins).");
        _output.WriteLine("aG3/afl = attention-category G3/flash (the isolated kernel win; whole-prefill dilutes it with FFN/proj).");

        model.ProfilingEnabled = false;
    }

    private static ushort[] CaptureLayer0AttnOutput(CudaTransformerModel model, ModelConfig config, int len, bool flashOn)
    {
        SetDispatch(flashOn);
        model.DebugCaptureAttnLayer = 0;
        try
        {
            int[] prompt = SyntheticPrompt(config, len);
            int[] positions = Positions(len);
            using var cache = model.CreateKvCache(maxSeqLen: len + 4);
            using ITensor _ = model.Forward(prompt, positions, deviceId: 0, cache);
            return model.DebugAttnOutputCapture!;
        }
        finally { model.DebugCaptureAttnLayer = -1; }
    }

    // Teacher-forced growing-prefix prefill perplexity (G1 precedent methodology): for each
    // target t, prefill tokens[0..t] (seqLen=t+1>1, engaging the prefill path) and score the
    // NLL of tokens[t+1] from the last-row logits. The OFF/ON ratio on identical tokens is the
    // quality signal. Stride keeps the O(N^2) sweep brisk. NB: at small t (< crossover=1 here,
    // so always) flash engages once seqLen>1.
    private static unsafe double PrefillGrowingPrefixPerplexity(
        CudaTransformerModel model, int[] tokens, int vocab, bool flashOn)
    {
        SetDispatch(flashOn);
        double sumNll = 0; int scored = 0;
        int stride = Math.Max(1, (tokens.Length - 2) / 64);  // ~64 scored targets
        for (int t = 1; t < tokens.Length - 1; t += stride)
        {
            int len = t + 1;
            int[] prefix = tokens[..len];
            int[] positions = Positions(len);
            using var cache = model.CreateKvCache(maxSeqLen: len + 1);
            using ITensor logits = model.Forward(prefix, positions, deviceId: 0, cache);
            var row = new ReadOnlySpan<float>((void*)logits.DataPointer, vocab);
            sumNll += -StableLogProb(row, tokens[t + 1]);
            scored++;
        }
        return Math.Exp(sumNll / scored);
    }

    // Real English corpus (same text as the G1/G3 perplexity test) tokenized via the model's
    // own BPE tokenizer, capped to maxTokens.
    private static int[] RealCorpusTokens(GgufFile gguf, int maxTokens)
    {
        const string corpus =
            "The history of natural language processing began in the nineteen fifties, when researchers "
            + "first attempted to translate text between human languages using simple rule based systems. "
            + "Over the following decades, statistical methods gradually replaced handwritten rules, and the "
            + "introduction of neural networks transformed the field once more. Today, large language models "
            + "are trained on vast collections of text drawn from books, articles, and conversations, learning "
            + "to predict the next word from everything that came before it. This deceptively simple objective "
            + "turns out to capture a surprising amount of structure about grammar, facts, and reasoning. "
            + "Researchers continue to probe how these systems represent meaning, whether they truly reason, "
            + "and how their behaviour can be made reliable, transparent, and aligned with human intentions.";
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] ids = tokenizer.Encode(corpus).ToArray();
        return ids.Length > maxTokens ? ids[..maxTokens] : ids;
    }

    private static double StableLogProb(ReadOnlySpan<float> row, int target)
    {
        float max = row[0];
        for (int j = 1; j < row.Length; j++) if (row[j] > max) max = row[j];
        double sumExp = 0;
        for (int j = 0; j < row.Length; j++) sumExp += Math.Exp(row[j] - max);
        return (row[target] - max) - Math.Log(sumExp);
    }

    private static float[] PrefillLastTokenLogits(CudaTransformerModel model, ModelConfig config, int len, bool flashOn)
    {
        SetDispatch(flashOn);
        int[] prompt = SyntheticPrompt(config, len);
        int[] positions = Positions(len);
        using var cache = model.CreateKvCache(maxSeqLen: len + 4);
        using ITensor logits = model.Forward(prompt, positions, deviceId: 0, cache);
        return CopyLogits(logits);
    }

    // ON arm = flash wins the dispatch (G3 must be off, else order is flash→G3 anyway but
    // keep it explicit). OFF arm = both off → attention_f16 reference.
    private static void SetDispatch(bool flashOn)
    {
        CudaFlashAttention.Enabled = flashOn;
        CudaG3Attention.Enabled = false;
    }

    // Deterministic pseudo-random token ids in [1, vocab) — exercises a realistic prefill
    // without needing a tokenizer (parity is token-content-independent).
    private static int[] SyntheticPrompt(ModelConfig config, int len)
    {
        int vocab = config.VocabSize;
        var rng = new Random(1234);
        var ids = new int[len];
        for (int i = 0; i < len; i++)
            ids[i] = 1 + rng.Next(vocab - 1);
        return ids;
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        return span.ToArray();
    }

    private static int Argmax(float[] v)
    {
        int idx = 0; float best = v[0];
        for (int i = 1; i < v.Length; i++) if (v[i] > best) { best = v[i]; idx = i; }
        return idx;
    }

    private static int[] Positions(int count)
    {
        var p = new int[count];
        for (int i = 0; i < count; i++) p[i] = i;
        return p;
    }

    private static int[]? ParseSeqs(string? csv)
    {
        if (string.IsNullOrWhiteSpace(csv)) return null;
        var parts = csv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var list = new List<int>();
        foreach (var p in parts) if (int.TryParse(p, out int v) && v > 1) list.Add(v);
        return list.Count > 0 ? list.ToArray() : null;
    }

    private static int ParseEnvInt(string key, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(key), out int n) && n > 0 ? n : fallback;

    private static string? ResolveModelPath()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_CUDA_FLASH_GGUF");
        if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return Path.GetFullPath(env);

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
            Path.Combine(home, ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
        ];
        foreach (var c in candidates) if (File.Exists(c)) return Path.GetFullPath(c);
        return null;
    }

    private static string ResolvePtxDir()
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            string cand = Path.Combine(dir, "native", "ptx");
            if (Directory.Exists(cand)) return cand;
            dir = Path.GetDirectoryName(dir);
        }
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));
    }
}
