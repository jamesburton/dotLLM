using DotLLM.Core.Configuration;
using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.Evaluation;
using DotLLM.Models;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.CrossBackend;

/// <summary>
/// The cross-backend quantization gate (issue #256): loads a real, single-block-type GGUF
/// fixture per <see cref="QuantizationType"/> and proves the CPU, CUDA and Vulkan backends
/// agree on it end-to-end, instead of only agreeing on synthetic per-kernel parity tensors.
/// </summary>
/// <remarks>
/// <para><b>Why this exists.</b> #254 (CUDA Q4_0/Q4_1 returning ~1e11 perplexity while exiting
/// 0) and #255 (CPU Q4_0 throwing "Unsupported quantization type" on every load) both shipped
/// to <c>dev</c> green, because every existing test scored a per-kernel synthetic tensor
/// against a CPU reference — never a real quantized model loaded end-to-end and compared
/// against a second backend on real tokens. This class is that missing gate.</para>
///
/// <para><b>In-process, not subprocess.</b> The issue describes the gate in terms of the
/// <c>dotllm perplexity</c> / <c>dotllm run</c> CLI invocations, but this harness calls the
/// same library code those commands call (<see cref="BackendPerplexityModel"/>,
/// <see cref="PerplexityEvaluator"/>, <see cref="TextGenerator"/>) directly in-process, exactly
/// like <c>RealGgufVulkanParityTests</c> and <c>CudaGemm16FPerplexityTests</c> already do. This
/// is a deliberate difference from a literal CLI-subprocess reproduction: two of the issue's own
/// named traps — "a generation leg must not treat non-empty stdout as success" and "the CLI can
/// print an error message to stdout and still exit 0" — are traps of shelling out and parsing
/// text. Calling the API directly means a broken decode path throws a real .NET exception or
/// returns a real (possibly astronomical) number, either of which xunit sees plainly; there is no
/// stdout to misparse. The third trap — coverage read from observed tensor block types, not
/// filenames — still applies and is enforced explicitly by
/// <see cref="Cpu_ObservedBlockType_MatchesFixtureTarget"/> below, reading
/// <see cref="GgufFile.Tensors"/> directly (the same data <c>dotllm debug gguf-tensors</c>
/// prints) rather than assuming the fixture's file name or the llama-quantize argument used to
/// generate it.</para>
///
/// <para><b>Fixtures.</b> One <c>--pure</c> single-block-type GGUF per <see cref="QuantizationType"/>,
/// generated from one shared Llama-3.2-1B-Instruct F16 source (hidden_size 2048, so
/// <c>ne[0] % 256 == 0</c> for every 256-superblock K-quant/IQ type — a 135M-class model's
/// hidden_size of 576 fails that constraint). Generation recipe, the llama-quantize argument used
/// for each dotLLM <see cref="QuantizationType"/> (including the <c>IQ2_S</c>/<c>IQ2_M</c>
/// ftype-vs-block-type mismatch the issue calls out by name), and current disk-budget gaps are
/// documented in <c>.docs/corpora/QUANT_FIXTURES.md</c>. Every test below resolves its fixture via
/// <see cref="ResolveFixturePath"/> (env var override, else the conventional
/// <c>~/.dotllm/test-cache/quant-fixtures/Llama-3.2-1B-pure/</c> path) and self-skips — not
/// fails — when the fixture is absent, matching the existing <c>DOTLLM_*_GGUF</c> convention
/// used by <c>RealGgufVulkanParityTests</c> et al.</para>
///
/// <para><b>Two legs, because they cover different kernels.</b>
/// <see cref="Cpu_TeacherForcedPerplexity_IsFinite"/> / the perplexity half of
/// <see cref="Backend_AgreesWithCpu"/> exercise the prefill/GEMM path (multi-token forward).
/// <see cref="Cpu_GreedyGeneration_CompletesWithTokens"/> / the decode half of
/// <see cref="Backend_AgreesWithCpu"/> exercise the decode/GEMV path (single-token forward,
/// repeated). The issue found several types pass the first and fail the second (a fused-decode
/// kernel that throws "does not support &lt;TYPE&gt;. Use standard Gemm path" with no automatic
/// fallback — #257) — scoring only one leg would miss that class of defect entirely.</para>
///
/// <para><b>Assert on cross-backend spread, not absolute perplexity.</b>
/// <see cref="Backend_AgreesWithCpu"/> compares <see cref="PerplexityResult.MeanNegativeLogLikelihood"/>
/// (nats) between the CPU and GPU backend on the identical fixture/corpus/tokens, not the raw
/// perplexity magnitude. <c>NatsTolerance</c> (0.05) is the bound #276 (the first defect this gate
/// actually caught, post-#256) used to catch a CPU-vs-GPU IQ1_S divergence of 0.24 nats (27%
/// relative) on a fixture legitimately destroyed by 1-bit quantization, where CUDA and Vulkan
/// agreed with each other to 0.0033 nats. A magnitude-only threshold (perplexity not
/// "astronomical") would have missed it — both large numbers looked equally destroyed until
/// diffed against a third backend on identical tokens.</para>
/// </remarks>
public sealed class CrossBackendQuantGateTests
{
    private readonly ITestOutputHelper _output;

    public CrossBackendQuantGateTests(ITestOutputHelper output) => _output = output;

    // ════════════════════════════════════════════════════════════════════
    // Fixture ladder
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Every <see cref="QuantizationType"/> the gate covers. Deliberately excludes:
    /// <see cref="QuantizationType.F32"/> (pass-through, not a quantized kernel — no fixture
    /// needed), <see cref="QuantizationType.I2_S"/> and <see cref="QuantizationType.PQ2_0"/>
    /// (BitNet / PrismML block layouts with no mainline llama.cpp <c>--pure</c> equivalent —
    /// see BitNet/PrismML-specific fixture notes in QUANT_FIXTURES.md instead).
    /// </summary>
    public static readonly QuantizationType[] AllGateTypes =
    {
        QuantizationType.F16,
        QuantizationType.BF16,
        QuantizationType.Q4_0,
        QuantizationType.Q4_1,
        QuantizationType.Q5_0,
        QuantizationType.Q5_1,
        QuantizationType.Q8_0,
        QuantizationType.Q2_K,
        QuantizationType.Q3_K,
        QuantizationType.Q4_K,
        QuantizationType.Q5_K,
        QuantizationType.Q6_K,
        QuantizationType.IQ4_NL,
        QuantizationType.IQ4_XS,
        QuantizationType.IQ2_XXS,
        QuantizationType.IQ2_XS,
        QuantizationType.IQ2_S,
        QuantizationType.IQ1_S,
        QuantizationType.IQ3_XXS,
        QuantizationType.IQ3_S,
        QuantizationType.MXFP4,
    };

    public static IEnumerable<object[]> AllGateTypeCases()
    {
        foreach (var t in AllGateTypes)
            yield return new object[] { t };
    }

    /// <summary>Which GPU backend a <see cref="Backend_AgreesWithCpu"/> case targets.</summary>
    public enum GateBackend { Cuda, Vulkan }

    public static IEnumerable<object[]> AllBackendCases()
    {
        foreach (var t in AllGateTypes)
        {
            yield return new object[] { t, GateBackend.Cuda };
            yield return new object[] { t, GateBackend.Vulkan };
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Shared corpus / prompt
    // ════════════════════════════════════════════════════════════════════

    // Ordinary English, long enough for a stable teacher-forced NLL estimate on a --pure
    // (frequently near-destroyed) fixture without being so long that the O(n^2) growing-prefix
    // path (required whenever a backend returns only the last logits row — every GPU backend
    // here) gets slow. Absolute perplexity on these fixtures is not meaningful (see class
    // remarks); only the cross-backend NLL delta on identical tokens is.
    private const string Corpus =
        "The quick brown fox jumps over the lazy dog near the old stone bridge. "
        + "Scientists have long studied how small variations in early conditions can lead to "
        + "very different outcomes over time, a phenomenon popularly known as the butterfly "
        + "effect. In computing, this sensitivity shows up whenever a tiny rounding error "
        + "compounds across many sequential steps of a calculation. Engineers who build "
        + "numerical software therefore pay close attention to where a computation accumulates "
        + "error, and to whether two independent implementations of the same algorithm still "
        + "agree once that error has had time to build up. When two implementations diverge only "
        + "slightly, on the order of a fraction of a percent, that is ordinary floating point "
        + "noise from a different instruction order or a different accumulation width. When they "
        + "diverge by many orders of magnitude, or when one produces a plausible number and the "
        + "other silently returns nonsense, that divergence usually points to a genuine defect "
        + "rather than to noise, and is worth investigating carefully before trusting either result.";

    private const string Prompt = "The capital of France is";

    private const int PerplexityContext = 256;
    private const int DecodeSteps = 8;

    /// <summary>
    /// Bound (nats) on <c>|cpu.MeanNegativeLogLikelihood - backend.MeanNegativeLogLikelihood|</c>
    /// for the teacher-forced leg. Sourced from #276: CUDA and Vulkan agreed with each other to
    /// 0.0033 nats on a fixture where CPU diverged from both by 0.24 nats (27% relative
    /// perplexity) — a genuine CPU IQ1_S defect that a magnitude-only ("perplexity isn't
    /// astronomical") check would have missed, since a --pure 1-bit fixture is *supposed* to
    /// have a huge absolute perplexity.
    /// </summary>
    private const double NatsTolerance = 0.05;

    /// <summary>
    /// Bound on mean <c>(1 - cosine_similarity)</c> between per-step decode logits, CPU vs
    /// backend, over <see cref="DecodeSteps"/> greedy steps. #276 measured 1-cos of 2.8e-3
    /// (CUDA) / 3.0e-3 (Vulkan) against CPU on IQ1_S as "inside the healthy continuum" even
    /// while the *prefill* NLL for the same fixture was 13x outside its own bound — decode and
    /// prefill are different kernels and can fail independently (#257). This bound is set well
    /// above that healthy band (so ordinary FP-order noise never trips it) but far below what a
    /// genuinely broken decode kernel produces: #254's CUDA Q4_0/Q4_1 decode returned ~1e10-1e11
    /// perplexity token after token, which corresponds to near-orthogonal (1-cos ~1) logit
    /// vectors, not a few-percent wobble.
    /// </summary>
    private const double OneMinusCosineTolerance = 0.05;

    // ════════════════════════════════════════════════════════════════════
    // CPU-only tests — run in every `dotnet test`, no GPU required.
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Trap #1 from the issue: "coverage must be read from observed tensor block types, never
    /// filenames" — llama.cpp ftypes do not map 1:1 to ggml block types (ftype <c>IQ2_S</c>
    /// actually yields <c>IQ2_XS</c> blocks; ftype <c>IQ2_M</c> is what yields <c>IQ2_S</c>).
    /// This reads <see cref="GgufFile.Tensors"/> directly — the same data
    /// <c>dotllm debug gguf-tensors</c> prints — and asserts every transformer-block weight
    /// tensor (attn_q/k/v/o, ffn_gate/up/down) is the EXPECTED block type, regardless of what
    /// llama-quantize argument or file name produced the fixture. <c>token_embd</c>/<c>output</c>
    /// are pinned to Q8_0 at generation time (some low-bit ftypes need imatrix coverage the
    /// embedding tensor lacks) and are deliberately excluded from this check — see
    /// QUANT_FIXTURES.md.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllGateTypeCases))]
    public void Cpu_ObservedBlockType_MatchesFixtureTarget(QuantizationType quantType)
    {
        string? path = ResolveFixturePath(quantType);
        if (path is null)
        {
            _output.WriteLine($"[SKIP] {quantType}: fixture not found. {FixtureHint(quantType)}");
            return;
        }

        using var gguf = GgufFile.Open(path);
        var blockTensors = gguf.Tensors.Where(IsTransformerBlockWeight).ToList();
        Assert.NotEmpty(blockTensors);

        var wrongType = blockTensors.Where(t => t.QuantizationType != quantType).ToList();
        foreach (var t in wrongType)
            _output.WriteLine($"[{quantType}] unexpected block type on '{t.Name}': observed {t.QuantizationType}");

        Assert.True(wrongType.Count == 0,
            $"[{quantType}] {wrongType.Count}/{blockTensors.Count} transformer-block tensors were NOT "
            + $"observed as {quantType} (e.g. '{wrongType.FirstOrDefault().Name}' -> {wrongType.FirstOrDefault().QuantizationType}). "
            + "Fixture generation likely used the wrong llama-quantize argument for this dotLLM type "
            + "— see the ftype-vs-block-type mapping notes in QUANT_FIXTURES.md.");

        _output.WriteLine($"[{quantType}] {blockTensors.Count} transformer-block tensors, all observed as {quantType}.");
    }

    /// <summary>
    /// CPU prefill/GEMM smoke: teacher-forced perplexity must be a finite, positive number.
    /// Also the source of the CPU-only baseline numbers reported alongside this PR — absolute
    /// values are expected to be enormous for the most aggressively quantized fixtures (that is
    /// the model being destroyed by design, not a bug; see class remarks).
    /// </summary>
    [Theory]
    [MemberData(nameof(AllGateTypeCases))]
    public void Cpu_TeacherForcedPerplexity_IsFinite(QuantizationType quantType)
    {
        string? path = ResolveFixturePath(quantType);
        if (path is null)
        {
            _output.WriteLine($"[SKIP] {quantType}: fixture not found. {FixtureHint(quantType)}");
            return;
        }

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

        var result = ScoreTeacherForced(model, deviceId: -1, tokenizer);

        _output.WriteLine(
            $"[{quantType}] CPU teacher-forced: ppl={result.Perplexity:E4} meanNll={result.MeanNegativeLogLikelihood:F6} "
            + $"nats scored={result.ScoredTokens}");

        Assert.True(double.IsFinite(result.MeanNegativeLogLikelihood),
            $"[{quantType}] CPU mean NLL was not finite ({result.MeanNegativeLogLikelihood}).");
        Assert.True(result.ScoredTokens > 0, $"[{quantType}] no tokens were scored.");
    }

    /// <summary>
    /// CPU decode/GEMV smoke. Traps #2/#3 from the issue: an actual generated-token count is
    /// asserted (not "stdout was non-empty" — there is no stdout here, only the real
    /// <see cref="GenerationToken"/> stream or a real exception), and the resulting continuation
    /// is scored for perplexity rather than merely checked for a clean completion — "gen=ok"
    /// alone proved decode executed, not that the output was sensible (#254's CUDA Q4_0 scored
    /// "ok" at 1e11 perplexity).
    /// </summary>
    /// <remarks>
    /// Expected to be RED on today's <c>dev</c> for the 14 types #257 lists (fused decode throws
    /// "does not support &lt;TYPE&gt;. Use standard Gemm path" with no automatic GEMM fallback).
    /// That is the gate working as intended — #257 explicitly asks for a test that fails without
    /// its fix, "the reason this reached dev is that no test loads a Q4_0 model end-to-end."
    /// </remarks>
    [Theory]
    [MemberData(nameof(AllGateTypeCases))]
    public async Task Cpu_GreedyGeneration_CompletesWithTokens(QuantizationType quantType)
    {
        string? path = ResolveFixturePath(quantType);
        if (path is null)
        {
            _output.WriteLine($"[SKIP] {quantType}: fixture not found. {FixtureHint(quantType)}");
            return;
        }

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

        var (text, generatedTokenCount, finishReason) = await GenerateGreedyAsync(model, tokenizer, DecodeSteps);

        _output.WriteLine($"[{quantType}] CPU decode: {generatedTokenCount} tokens, finish={finishReason}, text='{text}'");

        Assert.True(generatedTokenCount > 0,
            $"[{quantType}] CPU decode produced zero tokens (finish={finishReason}).");

        // Score the continuation, not just the prompt: a decode kernel that emits real-looking
        // tokens but garbage logits (#254's "gen=ok at 1e11 perplexity") is caught here, not by
        // the token count above.
        int[] fullSequence = tokenizer.Encode(Prompt + text);
        Assert.True(fullSequence.Length >= 2, $"[{quantType}] full sequence too short to score ({fullSequence.Length}).");
        var perplexityModel = new BackendPerplexityModel(model, -1,
            BackendPerplexityModel.Probe(model, -1));
        var options = new PerplexityOptions(PerplexityMode.TeacherForced, PerplexityContext, PerplexityContext);
        var result = PerplexityEvaluator.Evaluate(perplexityModel, fullSequence, options);

        _output.WriteLine($"[{quantType}] CPU continuation NLL={result.MeanNegativeLogLikelihood:F6} nats");
        Assert.True(double.IsFinite(result.MeanNegativeLogLikelihood),
            $"[{quantType}] CPU continuation NLL was not finite — decode produced tokens the model itself "
            + "assigns zero/undefined probability, i.e. self-inconsistent output.");
    }

    // ════════════════════════════════════════════════════════════════════
    // Cross-backend gate — GPU. NOT run by default `dotnet test`; filter
    // Category=GPU in. Each case independently skips when its fixture is
    // absent or the backend is unavailable on this host.
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// The gate itself: CPU vs one GPU backend, on the identical fixture and corpus, both legs.
    /// See class remarks for the full methodology and the #276 provenance of the thresholds.
    /// </summary>
    [Trait("Category", "GPU")]
    [SkippableTheory]
    [MemberData(nameof(AllBackendCases))]
    public async Task Backend_AgreesWithCpu(QuantizationType quantType, GateBackend backend)
    {
        string? path = ResolveFixturePath(quantType);
        Skip.If(path is null, $"{quantType}: fixture not found. {FixtureHint(quantType)}");

        SkipUnlessBackendAvailable(backend, out string? auxDir);

        using var cpuGguf = GgufFile.Open(path!);
        var cpuConfig = GgufModelConfigExtractor.Extract(cpuGguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);
        using var cpuModel = ModelLoader.CreateCpuModelFromGguf(cpuGguf, cpuConfig, ThreadingConfig.SingleThreaded);

        using var gpuGguf = GgufFile.Open(path!);
        var gpuConfig = GgufModelConfigExtractor.Extract(gpuGguf.Metadata);
        IDisposable? ownedDevice = null;
        IModel gpuModel;
        int gpuDeviceId;
        try
        {
            (gpuModel, gpuDeviceId, ownedDevice) = LoadGpuModel(backend, gpuGguf, gpuConfig, auxDir!);
        }
        catch (NotSupportedException ex)
        {
            // Trap-adjacent but distinct from #254/#255: a LOUD "not supported" (e.g. #258's "GPU
            // dequantization not supported for <TYPE>") is a known capability gap, not the silent
            // correctness defect this gate hunts for. Recorded as a skip so the gap is visible in
            // test output without conflating it with a cross-backend disagreement.
            Skip.If(true, $"[{quantType}/{backend}] backend does not support this type: {ex.Message}");
            return;
        }

        try
        {
            // ── Leg 1: teacher-forced perplexity (prefill/GEMM) ──
            var cpuPpl = ScoreTeacherForced(cpuModel, deviceId: -1, tokenizer);
            var gpuPpl = ScoreTeacherForced(gpuModel, gpuDeviceId, tokenizer);

            double natsDelta = Math.Abs(cpuPpl.MeanNegativeLogLikelihood - gpuPpl.MeanNegativeLogLikelihood);
            _output.WriteLine(
                $"[{quantType}/{backend}] prefill NLL: cpu={cpuPpl.MeanNegativeLogLikelihood:F6} "
                + $"{backend}={gpuPpl.MeanNegativeLogLikelihood:F6} |delta|={natsDelta:F6} nats "
                + $"(bound {NatsTolerance})");

            Assert.True(double.IsFinite(gpuPpl.MeanNegativeLogLikelihood),
                $"[{quantType}/{backend}] {backend} mean NLL was not finite.");
            Assert.True(natsDelta <= NatsTolerance,
                $"[{quantType}/{backend}] prefill NLL delta {natsDelta:F6} nats exceeds bound {NatsTolerance} "
                + $"(cpu={cpuPpl.MeanNegativeLogLikelihood:F6}, {backend}={gpuPpl.MeanNegativeLogLikelihood:F6}).");

            // ── Leg 2: greedy decode (decode/GEMV), per-step logit agreement ──
            double meanOneMinusCos = await ScoreDecodeAgreementAsync(
                cpuModel, gpuModel, gpuDeviceId, tokenizer, DecodeSteps);

            _output.WriteLine(
                $"[{quantType}/{backend}] decode mean(1-cos) over {DecodeSteps} steps: {meanOneMinusCos:E4} "
                + $"(bound {OneMinusCosineTolerance})");

            Assert.True(meanOneMinusCos <= OneMinusCosineTolerance,
                $"[{quantType}/{backend}] decode-step logits diverged: mean(1-cos)={meanOneMinusCos:E4} "
                + $"exceeds bound {OneMinusCosineTolerance}. This is the decode/GEMV path specifically — "
                + "prefill agreeing does not imply decode agrees (#257).");
        }
        finally
        {
            gpuModel.Dispose();
            ownedDevice?.Dispose();
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Fixture resolution
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Resolves a fixture path for <paramref name="quantType"/>: env var override
    /// (<c>DOTLLM_QUANT_FIXTURE_&lt;TYPE&gt;</c>) takes precedence, else the conventional
    /// <c>~/.dotllm/test-cache/quant-fixtures/Llama-3.2-1B-pure/Llama-3.2-1B-pure-&lt;TYPE&gt;.gguf</c>
    /// path documented in QUANT_FIXTURES.md. Returns null (never throws) when neither exists —
    /// callers self-skip.
    /// </summary>
    internal static string? ResolveFixturePath(QuantizationType quantType)
    {
        string envVar = $"DOTLLM_QUANT_FIXTURE_{quantType}";
        string? env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string conventional = Path.Combine(
            home, ".dotllm", "test-cache", "quant-fixtures", "Llama-3.2-1B-pure",
            $"Llama-3.2-1B-pure-{quantType}.gguf");
        return File.Exists(conventional) ? conventional : null;
    }

    private static string FixtureHint(QuantizationType quantType) =>
        $"Set DOTLLM_QUANT_FIXTURE_{quantType} or generate via the recipe in "
        + ".docs/corpora/QUANT_FIXTURES.md into "
        + "~/.dotllm/test-cache/quant-fixtures/Llama-3.2-1B-pure/.";

    private static bool IsTransformerBlockWeight(GgufTensorDescriptor tensor)
    {
        if (!tensor.Name.StartsWith("blk.", StringComparison.Ordinal))
            return false;
        return tensor.Name.EndsWith(".attn_q.weight", StringComparison.Ordinal)
            || tensor.Name.EndsWith(".attn_k.weight", StringComparison.Ordinal)
            || tensor.Name.EndsWith(".attn_v.weight", StringComparison.Ordinal)
            || tensor.Name.EndsWith(".attn_output.weight", StringComparison.Ordinal)
            || tensor.Name.EndsWith(".ffn_gate.weight", StringComparison.Ordinal)
            || tensor.Name.EndsWith(".ffn_up.weight", StringComparison.Ordinal)
            || tensor.Name.EndsWith(".ffn_down.weight", StringComparison.Ordinal);
    }

    // ════════════════════════════════════════════════════════════════════
    // Backend loading
    // ════════════════════════════════════════════════════════════════════

    private static void SkipUnlessBackendAvailable(GateBackend backend, out string? auxDir)
    {
        switch (backend)
        {
            case GateBackend.Cuda:
                Skip.IfNot(DotLLM.Cuda.CudaDevice.IsAvailable(), "No CUDA GPU available.");
                auxDir = ResolvePtxDir();
                Skip.If(auxDir is null, "CUDA PTX kernel directory not found; build the CUDA native kernels.");
                break;

            case GateBackend.Vulkan:
                bool vulkanOk;
                try
                {
                    using var probe = DotLLM.Vulkan.VulkanDevice.Create();
                    vulkanOk = true;
                }
                catch { vulkanOk = false; }
                Skip.IfNot(vulkanOk, "Vulkan runtime not available on this host.");
                auxDir = ResolveSpvDir();
                Skip.If(auxDir is null, "Vulkan SPIR-V directory not found.");
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(backend));
        }
    }

    private static (IModel Model, int DeviceId, IDisposable? OwnedDevice) LoadGpuModel(
        GateBackend backend, GgufFile gguf, ModelConfig config, string auxDir)
    {
        switch (backend)
        {
            case GateBackend.Cuda:
            {
                var (model, _) = DotLLM.Cuda.CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, auxDir);
                return (model, 0, null);
            }

            case GateBackend.Vulkan:
            {
                var device = DotLLM.Vulkan.VulkanDevice.Create();
                var (model, _) = DotLLM.Vulkan.VulkanModelLoader.CreateFromGguf(device, gguf, config, auxDir);
                return (model, 0, device);
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(backend));
        }
    }

    private static string? ResolvePtxDir()
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            string candidate = Path.Combine(dir, "native", "ptx");
            if (Directory.Exists(candidate)) return candidate;
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }

    private static string? ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (string c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        return null;
    }

    // ════════════════════════════════════════════════════════════════════
    // Scoring helpers
    // ════════════════════════════════════════════════════════════════════

    private static PerplexityResult ScoreTeacherForced(
        IModel model, int deviceId, DotLLM.Tokenizers.ITokenizer tokenizer)
    {
        int[] tokens = tokenizer.Encode(Corpus);
        bool returnsAllRows = BackendPerplexityModel.Probe(model, deviceId);
        var perplexityModel = new BackendPerplexityModel(model, deviceId, returnsAllRows);
        var options = new PerplexityOptions(PerplexityMode.TeacherForced, PerplexityContext, PerplexityContext);
        return PerplexityEvaluator.Evaluate(perplexityModel, tokens, options);
    }

    private static async Task<(string Text, int GeneratedTokenCount, FinishReason Finish)> GenerateGreedyAsync(
        IModel model, DotLLM.Tokenizers.ITokenizer tokenizer, int maxTokens)
    {
        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = maxTokens };

        var text = new System.Text.StringBuilder();
        int count = 0;
        FinishReason finish = FinishReason.Length;
        await foreach (var token in generator.GenerateStreamingTokensAsync(Prompt, options))
        {
            text.Append(token.Text);
            if (token.FinishReason is null || token.Text.Length > 0) count++;
            if (token.FinishReason.HasValue) finish = token.FinishReason.Value;
        }
        return (text.ToString(), count, finish);
    }

    /// <summary>
    /// Runs <paramref name="steps"/> greedy decode steps on BOTH models over the identical token
    /// sequence (advanced by the CPU model's argmax, so a backend disagreement never causes the
    /// two sequences to fork and start comparing unrelated positions), and returns the mean
    /// <c>1 - cosine_similarity</c> between their last-row logits at each step. No KV cache: each
    /// step is a fresh growing-prefix forward via the 3-arg <c>IModel.Forward</c>, which every
    /// backend implements uniformly — mirrors the pattern in
    /// <c>RealGgufVulkanParityTests.RunGgufParityTest</c>.
    /// </summary>
    private static async Task<double> ScoreDecodeAgreementAsync(
        IModel cpuModel, IModel gpuModel, int gpuDeviceId, DotLLM.Tokenizers.ITokenizer tokenizer, int steps)
    {
        int[] promptIds = tokenizer.Encode(Prompt);
        var tokens = new List<int>(promptIds);

        double sumOneMinusCos = 0;
        int scored = 0;

        for (int step = 0; step <= steps; step++)
        {
            int[] tokenIds = tokens.ToArray();
            int[] positions = new int[tokenIds.Length];
            for (int i = 0; i < positions.Length; i++) positions[i] = i;

            float[] cpuLogits = LastRowLogits(cpuModel, tokenIds, positions, -1, cpuModel.Config.VocabSize);
            float[] gpuLogits = LastRowLogits(gpuModel, tokenIds, positions, gpuDeviceId, gpuModel.Config.VocabSize);

            double cos = CosineSimilarity(cpuLogits, gpuLogits);
            sumOneMinusCos += 1.0 - cos;
            scored++;

            tokens.Add(Argmax(cpuLogits));
            await Task.Yield();
        }

        return sumOneMinusCos / scored;
    }

    private static unsafe float[] LastRowLogits(
        IModel model, int[] tokens, int[] positions, int deviceId, int vocab)
    {
        using var logits = model.Forward(tokens, positions, deviceId);
        int seqLen = logits.Shape.Rank == 2 ? logits.Shape[0] : 1;
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static int Argmax(float[] xs)
    {
        int best = 0; float bestV = xs[0];
        for (int i = 1; i < xs.Length; i++) if (xs[i] > bestV) { bestV = xs[i]; best = i; }
        return best;
    }
}
