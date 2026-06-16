using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Models;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Real-weight end-to-end validation for masked-diffusion decode driven from a
/// GGUF checkpoint (LLaDA-8B). LLaDA-8B is architecturally a <b>Llama</b> model
/// (RMSNorm, SiLU, RoPE θ=500000, 32 layers, hidden 4096, vocab 126464) but
/// generates by <b>masked diffusion</b> (absorbing-state, bidirectional, mask
/// token id 126336). GGUF carries no diffusion metadata, so the
/// <see cref="DiffusionConfig"/> (mask token + canvas / steps / temperatures) is
/// injected explicitly at load via
/// <see cref="ModelLoader.LoadGgufAsDiffusion(string, DiffusionConfig, DotLLM.Core.Configuration.ThreadingConfig?)"/>.
/// The resulting model is a normal Llama <c>TransformerModel</c> that
/// <see cref="DiffusionTextGenerator"/> drives through the hybrid (causal-prefix
/// + bidirectional-canvas) attention mask.
/// </summary>
/// <remarks>
/// Gated on <c>DOTLLM_LLADA_GGUF</c> (path to the .gguf). When unset or the file
/// is missing the test self-skips, so the build never requires the multi-gig
/// checkpoint. CPU-only — no GPU backend is touched. The orchestrator runs the
/// real validation by pointing the env var at
/// <c>C:\models\llada\LLaDA-8B-Instruct.Q4_K_M.gguf</c>.
/// </remarks>
public sealed class LladaDiffusionGgufTests
{
    private const string ModelPathEnvVar = "DOTLLM_LLADA_GGUF";

    /// <summary>LLaDA-8B absorbing-state mask token id (not present in GGUF metadata).</summary>
    private const int LladaMaskTokenId = 126336;

    private readonly ITestOutputHelper _output;

    public LladaDiffusionGgufTests(ITestOutputHelper output) => _output = output;

    private static string? TryResolveModelPath()
    {
        string? path = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return null;
        return path;
    }

    /// <summary>
    /// FAST diagnostic (~one forward per mask mode, seconds not minutes): build a short
    /// [prompt | MASK x N] sequence and run a SINGLE forward under each attention-mask mode,
    /// printing the argmax token decoded at each masked position. Tells us (a) the model
    /// loads/dequantises coherently, (b) which mask mode LLaDA wants, (c) whether EOS is the
    /// degenerate output. Gated on DOTLLM_LLADA_GGUF.
    /// </summary>
    [SkippableFact]
    public unsafe void Llada8B_Diagnostic_SingleForward_PerMaskMode()
    {
        string? path = TryResolveModelPath();
        Skip.If(path is null, $"Set {ModelPathEnvVar} to a LLaDA-8B GGUF to run this diagnostic.");

        var diffusion = new DiffusionConfig
        {
            CanvasLength = 8,
            MaxDenoisingSteps = 4,
            MaskTokenId = LladaMaskTokenId,
        };
        var (model, gguf, config, tokenizer) = ModelLoader.LoadGgufAsDiffusion(path!, diffusion);
        using var _ = gguf;
        using var __ = model;

        int[] promptIds = tokenizer.Encode("The capital of France is");
        _output.WriteLine($"vocab={config.VocabSize} eos={tokenizer.EosTokenId} mask={LladaMaskTokenId}");
        _output.WriteLine($"prompt ids=[{string.Join(",", promptIds)}] decoded='{tokenizer.Decode(promptIds)}'");

        const int nMask = 8;
        int promptLen = promptIds.Length;
        int seqLen = promptLen + nMask;
        int[] seq = new int[seqLen];
        Array.Copy(promptIds, seq, promptLen);
        for (int i = promptLen; i < seqLen; i++) seq[i] = LladaMaskTokenId;
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        foreach (var (name, spec) in new (string, AttentionMaskSpec)[]
        {
            ("Bidirectional", AttentionMaskSpec.Bidirectional),
            ("Hybrid(promptLen)", AttentionMaskSpec.Hybrid(promptLen)),
        })
        {
            var sw = Stopwatch.StartNew();
            using ITensor logits = model.Forward(seq, positions, deviceId: -1, kvCache: null, adapter: null, spec);
            sw.Stop();
            int vocab = logits.Shape[1];
            var preds = new int[nMask];
            float* p = (float*)logits.DataPointer;
            for (int i = 0; i < nMask; i++)
            {
                int row = promptLen + i;
                int best = 0; float bestV = float.NegativeInfinity;
                for (int v = 0; v < vocab; v++) { float x = p[(long)row * vocab + v]; if (x > bestV) { bestV = x; best = v; } }
                preds[i] = best;
            }
            _output.WriteLine($"[{name}] forward {sw.ElapsedMilliseconds}ms  argmax ids=[{string.Join(",", preds)}]");
            _output.WriteLine($"[{name}] decoded='{tokenizer.Decode(preds)}'");
            int eosCount = preds.Count(t => t == tokenizer.EosTokenId);
            int maskCount = preds.Count(t => t == LladaMaskTokenId);
            _output.WriteLine($"[{name}] eos@masked={eosCount}/{nMask} mask@masked={maskCount}/{nMask}");
        }
    }

    [SkippableFact]
    public void Llada8B_MaskedDiffusionDecode_ProducesNonDegenerateText()
    {
        string? path = TryResolveModelPath();
        Skip.If(path is null,
            $"Set {ModelPathEnvVar} to a LLaDA-8B GGUF (e.g. LLaDA-8B-Instruct.Q4_K_M.gguf) to run this test.");

        // Inject the diffusion schedule GGUF cannot carry: mask token + canvas /
        // steps / temperature bounds. Canvas kept modest (64) for a CPU smoke run.
        var diffusion = new DiffusionConfig
        {
            // LLaDA needs FULLY bidirectional attention over [prompt | canvas];
            // Hybrid (causal prompt) yields degenerate all-EOS output on LLaDA.
            CanvasAttentionMode = AttentionMaskMode.Bidirectional,
            CanvasLength = 16,            // short answer; keeps the 8B CPU run tractable
            MaxDenoisingSteps = 32,
            TemperatureMax = 0.8f,
            TemperatureMin = 0.4f,
            MaskTokenId = LladaMaskTokenId,
        };

        var loadSw = Stopwatch.StartNew();
        var (model, gguf, config, tokenizer) = ModelLoader.LoadGgufAsDiffusion(path!, diffusion);
        loadSw.Stop();
        using var _ = gguf;
        using var __ = model;

        // The GGUF Llama load must yield a usable model + tokenizer.
        Assert.NotNull(tokenizer);
        Assert.Equal(Architecture.Llama, config.Architecture);
        Assert.NotNull(config.DiffusionConfig);
        Assert.Equal(LladaMaskTokenId, config.DiffusionConfig!.MaskTokenId);
        // Mask token id must be addressable within the model's vocab (126464 > 126336).
        Assert.True(LladaMaskTokenId < config.VocabSize,
            $"Mask token {LladaMaskTokenId} must be < vocab size {config.VocabSize}.");

        int[] promptIds = tokenizer.Encode("The capital of France is");
        Assert.NotEmpty(promptIds);

        var generator = new DiffusionTextGenerator(model, tokenizer);

        var genSw = Stopwatch.StartNew();
        DiffusionResult result = generator.Generate(promptIds);
        genSw.Stop();

        // ── Non-degenerate output assertions ──────────────────────────────
        Assert.True(result.GeneratedTokenCount > 0, "Expected at least one generated token.");
        // No surviving mask token — a finished canvas must be fully materialised.
        Assert.DoesNotContain(LladaMaskTokenId, result.GeneratedTokenIds);
        // Every generated id is a valid, finite vocab index.
        foreach (int id in result.GeneratedTokenIds)
            Assert.InRange(id, 0, config.VocabSize - 1);
        Assert.False(string.IsNullOrWhiteSpace(result.Text), "Decoded text should be non-empty.");
        Assert.True(result.TotalDenoisingSteps >= 1);

        // ── Timing / throughput ───────────────────────────────────────────
        double genSec = genSw.Elapsed.TotalSeconds;
        double tokPerSec = genSec > 0 ? result.GeneratedTokenCount / genSec : 0;
        double stepsPerSec = genSec > 0 ? result.TotalDenoisingSteps / genSec : 0;

        _output.WriteLine($"[LLaDA-8B GGUF] {path}");
        _output.WriteLine($"  load wall   : {loadSw.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"  vocab size  : {config.VocabSize}");
        _output.WriteLine($"  prompt toks : {result.PromptTokenCount}");
        _output.WriteLine($"  gen toks    : {result.GeneratedTokenCount}");
        _output.WriteLine($"  canvases    : {result.CanvasCount}");
        _output.WriteLine($"  denoise stps: {result.TotalDenoisingSteps}");
        _output.WriteLine($"  finish      : {result.FinishReason}");
        _output.WriteLine($"  gen wall    : {genSec:F2} s");
        _output.WriteLine($"  tokens/sec  : {tokPerSec:F2}");
        _output.WriteLine($"  steps/sec   : {stepsPerSec:F2}");
        _output.WriteLine($"  text        : {result.Text}");
    }
}
