using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// The default-on gate for G1, the CUDA cuBLAS <c>COMPUTE_16F</c> prefill-GEMM
/// optimization (<see cref="CudaGemm.Use16FCompute"/>). G1 switches the linear
/// projection compute type from <c>CUBLAS_COMPUTE_32F</c> (FP16 inputs, FP32
/// accumulate) to <c>CUBLAS_COMPUTE_16F</c> (FP16 accumulate), which runs at the
/// full tensor-core rate on throttled GeForce Ampere but accumulates in FP16 and
/// so carries an accuracy risk. It is measured at ~1.06x median / 1.22x best
/// whole-prefill speedup and is OFF by default pending this quality check.
/// </summary>
/// <remarks>
/// <para>
/// <b>Which path G1 changes.</b> <see cref="CudaGemm.Use16FCompute"/> gates
/// <see cref="CudaGemm.LinearF16"/>, which serves <i>both</i> the prefill GEMM
/// (<c>m == seqLen &gt; 1</c>) and the decode GEMV (<c>m == 1</c>, via
/// <see cref="CudaGemm.GemvF16"/>). G1 is named a <i>prefill</i> optimization, so
/// this test exercises the prefill GEMM specifically: every scored forward runs a
/// multi-row <c>LinearF16</c> (<c>seqLen &gt; 1</c>).
/// </para>
/// <para>
/// <b>Why a growing-prefix prefill PPL, not <see cref="DotLLM.Engine.Evaluation.PerplexityEvaluator.Evaluate"/>.</b>
/// The CUDA model's <c>Forward</c> returns only the last-position logit row
/// (shape <c>[1, vocab]</c>), so the all-positions windowed evaluator cannot run
/// against a GPU model. Instead, for each target index <c>t</c> we prefill
/// <c>tokens[0..t]</c> (so the forward has <c>seqLen == t + 1 &gt; 1</c> and the
/// 16F prefill GEMM engages on every layer's Q/K/V/O/gate/up/down projection),
/// read the last row, and accumulate the NLL of the true next token
/// <c>tokens[t + 1]</c>. This is teacher-forced prefill perplexity that actually
/// exercises the GEMM G1 optimizes. It is O(N^2) in corpus length; a few hundred
/// tokens on a 1B model is comfortably fast.
/// </para>
/// <para>
/// <b>Toggle mechanics.</b> <see cref="CudaGemm.Use16FCompute"/> is a static
/// field whose <i>initializer</i> reads <c>DOTLLM_CUDA_GEMM_16F</c> exactly once
/// at type init; <c>Environment.SetEnvironmentVariable</c> mid-process does not
/// re-latch it. The field is read per GEMM call (mutable by design so a benchmark
/// can interleave settings in one warmed process), so this test loads the model
/// once and flips the field between the OFF and ON passes — same weights, same
/// device buffers, only the compute type differs. The 3060's clocks drift across
/// consecutive heavy runs, so a single warmed process with the field flipped is
/// the correct A/B for a quality (not timing) measurement; only the toggle differs.
/// </para>
/// <para>
/// <b>Verdict framing.</b> The quality delta measured here is architecture-
/// independent (FP16 accumulate is the same math on any CUDA device); the speedup
/// is GeForce-Ampere-specific (datacenter cards do not throttle FP32 tensor
/// accumulate). FP16-accumulation error also grows with K (hidden dim) and
/// compounds across layers, so Llama-3.2-1B (hidden 2048) is the honest worst
/// case for an "adopt default-on" decision and is the model measured here.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaGemm16FPerplexityTests
{
    private readonly ITestOutputHelper _output;

    public CudaGemm16FPerplexityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableFact]
    public unsafe void PrefillPerplexity_Gemm16FOnVsOff_RealModel_QuantifiesQualityDelta()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string? ggufPath = ResolveModelPath();
        Skip.If(ggufPath is null,
            "Llama-3.2-1B-Instruct-Q8_0 GGUF not found. Set DOTLLM_CUDA_GEMM16F_GGUF or place it "
            + "under ~/.dotllm/models/bartowski/Llama-3.2-1B-Instruct-GGUF/.");

        string ptxDir = ResolvePtxDir();
        _output.WriteLine($"GGUF: {ggufPath}");
        _output.WriteLine($"PTX dir: {ptxDir}");

        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        _output.WriteLine(
            $"Config: arch={config.Architecture} layers={config.NumLayers} hidden={config.HiddenSize} "
            + $"heads={config.NumAttentionHeads}/{config.NumKvHeads} vocab={config.VocabSize}");

        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        // A few hundred tokens of ordinary English. Absolute PPL is secondary —
        // the off-vs-on ratio on identical tokens is the signal.
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
        int[] tokenIds = tokenizer.Encode(corpus);
        Assert.True(tokenIds.Length >= 64, $"corpus too short ({tokenIds.Length} tokens).");
        // Cap to keep the O(N^2) growing-prefix sweep brisk while still scoring
        // a few hundred targets across genuinely long prefixes.
        const int maxTokens = 200;
        if (tokenIds.Length > maxTokens)
            tokenIds = tokenIds[..maxTokens];
        _output.WriteLine($"corpus: {tokenIds.Length} tokens, scoring {tokenIds.Length - 1} prefill targets");

        // Load the model ONCE; flip the field between passes (per-call read).
        bool prior = CudaGemm.Use16FCompute;
        try
        {
            using var model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

            // Engagement check on a genuine prefill forward (seqLen > 1): the
            // last-row logits must differ between 16F off and on, else the toggle
            // did not reach the prefill GEMM and the whole measurement is vacuous.
            int probeLen = Math.Min(16, tokenIds.Length);
            int[] probeTokens = tokenIds[..probeLen];
            int[] probePositions = Positions(probeLen, 0);

            CudaGemm.Use16FCompute = false;
            float[] probeOff = LastRowLogits(model, probeTokens, probePositions, config.VocabSize);
            CudaGemm.Use16FCompute = true;
            float[] probeOn = LastRowLogits(model, probeTokens, probePositions, config.VocabSize);

            float maxAbs = 0f;
            double sumSq = 0;
            bool anyDiffer = false;
            for (int i = 0; i < probeOff.Length; i++)
            {
                float d = MathF.Abs(probeOff[i] - probeOn[i]);
                if (d > 0f) anyDiffer = true;
                if (d > maxAbs) maxAbs = d;
                sumSq += (double)d * d;
            }
            double probeRms = Math.Sqrt(sumSq / probeOff.Length);
            _output.WriteLine(
                $"engagement (seqLen={probeLen} prefill): max|Δ|={maxAbs:E3} rms={probeRms:E3} differ={anyDiffer}");
            Assert.True(anyDiffer,
                "16F-on prefill logits were BIT-IDENTICAL to 16F-off — the COMPUTE_16F toggle did not "
                + "engage on the prefill GEMM. The measurement is VACUOUS; investigate before trusting any ratio.");

            // ── Decode engagement probe ───────────────────────────────────────
            // Use16FCompute also gates the decode GEMV (CudaGemm.GemvF16), but for
            // a QUANTIZED model (Q8_0 here) the seqLen==1 decode projections route
            // through the integer GEMV path (CudaTransformerModel.Project lines
            // 1746/1748) and never reach GemvF16 — so COMPUTE_16F is prefill-only
            // in practice for quant models. We confirm rather than assume.
            //
            // The probe MUST isolate the decode GEMV: prefill with 16F held OFF for
            // BOTH runs (a 16F prefill would write a different K/V cache, and the
            // decode step would then read different K/V via attention — a difference
            // inherited from prefill, NOT from the decode GEMV engaging 16F). So we
            // prefill 16F-off, then flip 16F only on the single seqLen==1 decode step.
            // If decode logits are bit-identical, the decode GEMV is 16F-independent.
            bool decodeDiffers;
            {
                int[] decPrefix = tokenIds[..probeLen];
                int[] decPrefixPos = Positions(probeLen, 0);
                int decodeToken = tokenIds[probeLen];
                int[] decodePos = { probeLen };

                float[] decOff = DecodeStepLogits(model, decPrefix, decPrefixPos, decodeToken, decodePos,
                    config.VocabSize, prefill16F: false, decode16F: false);
                float[] decOn = DecodeStepLogits(model, decPrefix, decPrefixPos, decodeToken, decodePos,
                    config.VocabSize, prefill16F: false, decode16F: true);

                float decMaxAbs = 0f;
                decodeDiffers = false;
                for (int i = 0; i < decOff.Length; i++)
                {
                    float d = MathF.Abs(decOff[i] - decOn[i]);
                    if (d > 0f) decodeDiffers = true;
                    if (d > decMaxAbs) decMaxAbs = d;
                }
                _output.WriteLine(
                    $"decode engagement (seqLen=1, prefill held 16F-off): max|Δ|={decMaxAbs:E3} differ={decodeDiffers} "
                    + $"→ {(decodeDiffers ? "16F engages decode GEMV" : "decode uses integer GEMV; COMPUTE_16F is PREFILL-ONLY for this quant model")}");
            }

            // Growing-prefix prefill perplexity under each setting.
            CudaGemm.Use16FCompute = false;
            double pplOff = PrefillGrowingPrefixPerplexity(model, tokenIds, config.VocabSize);
            CudaGemm.Use16FCompute = true;
            double pplOn = PrefillGrowingPrefixPerplexity(model, tokenIds, config.VocabSize);

            double ratio = pplOn / pplOff;
            _output.WriteLine(
                $"prefill PPL: off={pplOff:F5} on={pplOn:F5} ratio(on/off)={ratio:F5} "
                + $"({(ratio - 1) * 100:+0.000;-0.000}%)");

            // Decode-mode (teacher-forced, one token at a time) perplexity. Every
            // forward is seqLen==1, so NOTHING here touches the prefill GEMM — this
            // measures purely the decode path under the 16F toggle. For a quant
            // model the flat +0.000% IS the result: it confirms decode is integer
            // and COMPUTE_16F is prefill-only. We always run it (its flatness is
            // the measurement, not a reason to skip).
            CudaGemm.Use16FCompute = false;
            double decPplOff = DecodeModePerplexity(model, tokenIds, config.VocabSize);
            CudaGemm.Use16FCompute = true;
            double decPplOn = DecodeModePerplexity(model, tokenIds, config.VocabSize);
            double decRatio = decPplOn / decPplOff;
            _output.WriteLine(
                $"decode PPL:  off={decPplOff:F5} on={decPplOn:F5} ratio(on/off)={decRatio:F5} "
                + $"({(decRatio - 1) * 100:+0.000;-0.000}%)");
            Assert.True(Math.Abs(decRatio - 1.0) < 0.01,
                $"G1 (COMPUTE_16F) decode perplexity moved {(decRatio - 1) * 100:+0.00;-0.00}% — exceeds the 1% gate.");

            // Quality gate: a perplexity move under 1% on the worst-case (1B,
            // hidden 2048) is the threshold the sibling DP4a gate uses. A larger
            // gap argues for keeping G1 gated on quality grounds; a smaller gap
            // supports adopting default-on (where the Ampere speedup also holds).
            Assert.True(Math.Abs(ratio - 1.0) < 0.01,
                $"G1 (COMPUTE_16F) prefill perplexity moved {(ratio - 1) * 100:+0.00;-0.00}% "
                + $"(off={pplOff:F4} on={pplOn:F4}) — exceeds the 1% quality gate.");
        }
        finally
        {
            CudaGemm.Use16FCompute = prior;
        }
    }

    /// <summary>
    /// Teacher-forced prefill perplexity: for each target index <c>t</c>, prefill
    /// <c>tokens[0..t]</c> against a fresh KV cache (seqLen = t+1 &gt; 1, exercising
    /// the 16F prefill GEMM) and score the NLL of <c>tokens[t+1]</c> from the
    /// last-row logits.
    /// </summary>
    private static unsafe double PrefillGrowingPrefixPerplexity(
        CudaTransformerModel model, int[] tokenIds, int vocabSize)
    {
        double sumNll = 0;
        int scored = 0;
        for (int t = 1; t < tokenIds.Length - 1; t++)
        {
            int len = t + 1;                       // prefix length tokens[0..t]
            int[] prefix = tokenIds[..len];
            int[] positions = Positions(len, 0);

            using var cache = model.CreateKvCache(maxSeqLen: len + 1);
            float[] logits = LastRowLogits(model, prefix, positions, vocabSize);

            sumNll += -StableLogProb(logits, tokenIds[t + 1]);
            scored++;
        }
        return Math.Exp(sumNll / scored);
    }

    /// <summary>
    /// Prefills <paramref name="prefix"/> against a fresh KV cache, then runs one
    /// seqLen==1 decode step for <paramref name="decodeToken"/> and returns its
    /// last-row logits. <paramref name="prefill16F"/> and <paramref name="decode16F"/>
    /// pin <see cref="CudaGemm.Use16FCompute"/> independently for the two phases so
    /// the caller can isolate the decode GEMV from prefill-inherited K/V differences.
    /// </summary>
    private static unsafe float[] DecodeStepLogits(
        CudaTransformerModel model, int[] prefix, int[] prefixPos,
        int decodeToken, int[] decodePos, int vocabSize, bool prefill16F, bool decode16F)
    {
        using var cache = model.CreateKvCache(maxSeqLen: prefix.Length + 2);
        CudaGemm.Use16FCompute = prefill16F;
        using (ITensor _ = model.Forward(prefix, prefixPos, deviceId: 0, cache)) { }
        CudaGemm.Use16FCompute = decode16F;
        using ITensor logits = model.Forward([decodeToken], decodePos, deviceId: 0, cache);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
        return span.ToArray();
    }

    /// <summary>
    /// Teacher-forced decode-mode perplexity: prefill the first token, then feed
    /// each subsequent true token one at a time through the KV cache (every forward
    /// after the first is seqLen==1 → decode path), scoring the NLL of the next
    /// true token. Only meaningful when the decode path engages COMPUTE_16F.
    /// </summary>
    private static unsafe double DecodeModePerplexity(
        CudaTransformerModel model, int[] tokenIds, int vocabSize)
    {
        using var cache = model.CreateKvCache(maxSeqLen: tokenIds.Length + 1);
        // Prefill token[0] (seqLen==1 is acceptable here; the first forward seeds
        // the cache and its target is token[1]).
        double sumNll = 0;
        int scored = 0;
        for (int t = 0; t < tokenIds.Length - 1; t++)
        {
            int[] one = { tokenIds[t] };
            int[] pos = { t };
            using ITensor logits = model.Forward(one, pos, deviceId: 0, cache);
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
            float[] row = span.ToArray();
            sumNll += -StableLogProb(row, tokenIds[t + 1]);
            scored++;
        }
        return Math.Exp(sumNll / scored);
    }

    private static unsafe float[] LastRowLogits(
        CudaTransformerModel model, int[] tokens, int[] positions, int vocabSize)
    {
        using var cache = model.CreateKvCache(maxSeqLen: tokens.Length + 1);
        using ITensor logits = model.Forward(tokens, positions, deviceId: 0, cache);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
        return span.ToArray();
    }

    private static double StableLogProb(float[] logitsRow, int target)
    {
        float max = logitsRow[0];
        for (int j = 1; j < logitsRow.Length; j++)
            if (logitsRow[j] > max) max = logitsRow[j];
        double sumExp = 0;
        for (int j = 0; j < logitsRow.Length; j++)
            sumExp += Math.Exp(logitsRow[j] - max);
        return (logitsRow[target] - max) - Math.Log(sumExp);
    }

    private static int[] Positions(int count, int start)
    {
        int[] positions = new int[count];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = start + i;
        return positions;
    }

    private static string? ResolveModelPath()
    {
        string? envPath = Environment.GetEnvironmentVariable("DOTLLM_CUDA_GEMM16F_GGUF");
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] relativeCandidates =
        [
            Path.Combine("bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
        ];
        string[] roots =
        [
            Path.Combine(home, ".dotllm", "models"),
            Path.Combine(home, ".dotllm", "test-cache"),
        ];

        foreach (string root in roots)
            foreach (string relative in relativeCandidates)
            {
                string candidate = Path.Combine(root, relative);
                if (File.Exists(candidate))
                    return candidate;
            }

        return null;
    }

    private static string ResolvePtxDir()
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            string candidate = Path.Combine(dir, "native", "ptx");
            if (Directory.Exists(candidate))
                return candidate;
            dir = Path.GetDirectoryName(dir);
        }

        return Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));
    }
}
