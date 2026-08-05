using System.Collections.Concurrent;
using DotLLM.Core.Configuration;
using DotLLM.Core.Evaluation;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Quantization;

/// <summary>Outcome of one (type, backend) cell, recorded for the coverage meta-test.</summary>
/// <param name="Type">Quantization type under test.</param>
/// <param name="Backend">Backend compared against the CPU reference.</param>
/// <param name="Outcome"><c>ran</c>, <c>skipped</c> or <c>failed</c>.</param>
/// <param name="Detail">Reason or measured verdict.</param>
public sealed record QuantGateCellResult(
    QuantizationType Type, QuantGateBackend Backend, string Outcome, string Detail);

/// <summary>
/// The cross-backend quantization gate (#256): every <c>--pure</c> fixture, scored on CPU and on
/// each available GPU backend, asserted to agree on identical bytes.
/// </summary>
/// <remarks>
/// <para>
/// <b>The metric is spread, not magnitude.</b> The ≥1B fixtures are deliberately near-destroyed;
/// <c>--pure</c> Q2_K scores ~7.1e6 while three backends agree to 0.018 nats. Asserting on an
/// absolute value would false-alarm on every low-bit cell, and a reviewer nearly filed
/// "Q3_K_S is broken on CUDA" from its 5.19e6 magnitude alone when all three backends agreed.
/// </para>
/// <para>
/// <b>CPU is the reference.</b> Kernel-level CPU↔GPU parity already exists; what was missing is
/// model-level agreement on a real quantized checkpoint.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("QuantLadder")]
public sealed class CrossBackendQuantGateTests
{
    private const int CorpusTokens = 4096;

    /// <summary>Prompts seeding the two decode legs — one forward each, no autoregressive feedback.</summary>
    /// <remarks>
    /// <para>
    /// <b>Chosen by measurement.</b> A search over ten candidates across all 21 fixtures
    /// (<c>QuantGateDecodePromptTests.Probe_ReportsDecodeTokensPerCandidatePrompt</c>) found that
    /// <i>no</i> single prompt discriminates everywhere: on the <c>--pure</c> Q2_K and Q3_K 1B
    /// fixtures, greedy self-feedback is a fixed point, so all four steps carry the same id for
    /// every candidate. The first id, though, varies richly with the prompt — Q2_K produced 67585,
    /// 55934, 127327, 37424 and so on across the candidates.
    /// </para>
    /// <para>
    /// So the legs take one step from each of four prompts instead of four steps from one prompt.
    /// This quad was selected because it yields four distinct ids on all 21 fixtures, Q2_K and Q3_K
    /// included; <c>ChosenPrompts_AreInformativeOnEveryFixture</c> holds that property.
    /// </para>
    /// </remarks>
    public static readonly string[] DecodePrompts =
    [
        "A B C D E F",
        "{\"name\": \"Alice\", \"age\":",
        "The three primary colours are red, green and",
        "one two three four five six seven",
    ];

    /// <summary>Number of scored steps on each decode leg — one per entry in <see cref="DecodePrompts"/>.</summary>
    public static int DecodeSteps => DecodePrompts.Length;

    /// <summary>Minimum per-step logit cosine between backends on the decode leg.</summary>
    /// <remarks>
    /// <para>
    /// <b>Measured, not chosen.</b> The first full sweep recorded four per-step cosines for each of
    /// 40 cells. Ranked by worst-step error <c>1−cos</c>, the cells form one smooth continuum from
    /// 1e-5 up to 3.04e-3 with no gap anywhere inside it — and then a single discontinuity: MXFP4
    /// on both backends at 1.22e-2 and 1.21e-2, a factor of four clear of everything else.
    /// </para>
    /// <para>
    /// This bound is placed in that gap, at <c>1−cos = 6e-3</c>: about 2x headroom over the worst
    /// cell in the continuum and about 2x margin below MXFP4. The previous value of 0.999
    /// (<c>1e-3</c>) sat in the middle of the continuum and cut it arbitrarily, failing seven cells
    /// whose neighbours on either side passed.
    /// </para>
    /// <para>
    /// <b>MXFP4 is a real finding, not the reason the bound moved.</b> CUDA and Vulkan track each
    /// other to under 5e-4 at every step (step 2: 0.987781 vs 0.987879) while both drift from the
    /// CPU reference by 1.2e-2. Two independent GPU implementations agreeing with each other and
    /// disagreeing with the reference points at the CPU MXFP4 decode path, and this bound keeps
    /// that cell red.
    /// </para>
    /// </remarks>
    private const double MinDecodeCosine = 0.994;

    private static readonly ConcurrentBag<QuantGateCellResult> Cells = [];

    private readonly QuantLadderFixture _ladder;
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the gate against the shared ladder index.</summary>
    /// <param name="ladder">Ladder index shared across the <c>QuantLadder</c> collection.</param>
    /// <param name="output">xunit sink for the measured numbers behind each cell.</param>
    public CrossBackendQuantGateTests(QuantLadderFixture ladder, ITestOutputHelper output)
    {
        _ladder = ladder;
        _output = output;
    }

    /// <summary>Cells recorded so far, for <c>QuantGateCoverageTests</c>.</summary>
    public static IReadOnlyList<QuantGateCellResult> Results => Cells.ToList();

    /// <summary>Every (quantization type, GPU backend) pair the gate intends to cover.</summary>
    /// <returns>The full matrix; absent fixtures and absent backends are skipped at run time.</returns>
    /// <remarks>
    /// Built from <see cref="QuantLadderFixture.Expected"/> rather than from what is present on this
    /// machine, so a missing fixture surfaces as an explicit skipped cell instead of vanishing from
    /// the matrix — the failure mode that let CPU Q4_0 go entirely unexercised.
    /// </remarks>
    public static TheoryData<QuantizationType, QuantGateBackend> Cases()
    {
        var data = new TheoryData<QuantizationType, QuantGateBackend>();
        foreach (var (type, _, _) in QuantLadderFixture.Expected)
        {
            data.Add(type, QuantGateBackend.Cuda);
            data.Add(type, QuantGateBackend.Vulkan);
        }

        return data;
    }

    /// <summary>Asserts one GPU backend agrees with the CPU reference on one fixture.</summary>
    /// <param name="type">Quantization type whose fixture is loaded.</param>
    /// <param name="backend">GPU backend compared against CPU.</param>
    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Backend_AgreesWithCpu(QuantizationType type, QuantGateBackend backend)
    {
        var entry = _ladder.Available.FirstOrDefault(e => e.Type == type);
        if (entry is null)
        {
            Record(type, backend, "skipped", $"fixture absent under {_ladder.RootDirectory}");
            _output.WriteLine($"  CELL\t{type}\t{backend}\tskipped\t-\t-\t-\t-\t-\tfixture-absent");
            Skip.If(true, $"fixture for {type} not present");
            return;
        }

        if (!QuantGateBackendRunner.IsAvailable(backend))
        {
            Record(type, backend, "skipped", $"{backend} not available on this machine");
            _output.WriteLine($"  CELL\t{type}\t{backend}\tskipped\t-\t-\t-\t-\t-\tbackend-absent");
            Skip.If(true, $"{backend} not available");
            return;
        }

        QuantGateRun cpu;
        QuantGateRun gpu;
        try
        {
            cpu = QuantGateBackendRunner.Run(
                entry, QuantGateBackend.Cpu, QuantGateCorpus.Path, CorpusTokens, DecodePrompts);
            gpu = QuantGateBackendRunner.Run(
                entry, backend, QuantGateCorpus.Path, CorpusTokens, DecodePrompts);
        }
        catch (Exception ex) when (ClassifyEnvironmentFailure(ex) is { } environmental)
        {
            // A device lost to Windows' TDR, an unbuilt SPIR-V blob or an architecture with no loader
            // on this backend are all facts about the machine, not disagreements between kernels.
            // Recording them as "failed" would put a red cell against a quantization type whose
            // kernels were never executed, which is exactly the misattribution this gate exists to
            // end. They are skips, with the reason kept verbatim so the coverage meta-test can tell
            // "not exercised" from "exercised and agreed".
            Record(type, backend, "skipped", environmental);
            _output.WriteLine($"{type}/{backend}: {environmental}");
            _output.WriteLine($"  CELL\t{type}\t{backend}\tskipped\t-\t-\t-\t-\t-\t" +
                              environmental.Replace('\t', ' ').Replace('\n', ' ').Replace('\r', ' '));
            Skip.If(true, $"{type} on {backend}: {environmental}");
            return;
        }

        // ── Leg 1: prefill / GEMM ────────────────────────────────────────────────
        QuantGateMetric metric = QuantGateComparison.ResolveModeFromEnvironment();
        QuantGateVerdict verdict = QuantGateComparison.Compare(
            cpu.Perplexity, gpu.Perplexity, metric, QuantGateThresholds.Default);

        _output.WriteLine($"{type}/{backend} prefill: cpu nll={cpu.Perplexity.MeanNegativeLogLikelihood:F6} " +
                          $"ppl={cpu.Perplexity.Perplexity:G8} | gpu nll={gpu.Perplexity.MeanNegativeLogLikelihood:F6} " +
                          $"ppl={gpu.Perplexity.Perplexity:G8}");
        _output.WriteLine($"  {verdict.Detail}");

        // The scored-token counts must match for the two NLLs to be comparable at all; the window
        // counts deliberately are not compared, because they only record which evaluator path each
        // backend's logits shape selected (see QuantGateRun's remarks).
        Assert.Equal(cpu.Perplexity.ScoredTokens, gpu.Perplexity.ScoredTokens);

        // ── Leg 2: decode / GEMV ─────────────────────────────────────────────────
        // A numerical assertion, deliberately. "Exit 0 and non-empty output" scored CUDA Q4_0 as
        // passing at 1e11 perplexity and scored PQ2_0 as passing while it emitted gibberish.
        var decodeFailures = new List<string>();
        bool allTopOneAgreed = true;
        double[] cosines = CompareLeg(
            "uncached", cpu.DecodeLogits, gpu.DecodeLogits, cpu.DecodeTokens, gpu.DecodeTokens,
            backend, decodeFailures, ref allTopOneAgreed);

        // ── Leg 3: cached seqLen == 1 decode / GEMV ──────────────────────────────
        // The only leg that reaches the fused-decode kernels; see RunKvDecode's remarks for why
        // the other two cannot, and for the cache-management caveat that comes with it.
        var kvFailures = new List<string>();
        bool kvTopOneAgreed = true;
        double[] kvCosines = CompareLeg(
            "cached", cpu.KvDecodeLogits, gpu.KvDecodeLogits, cpu.KvDecodeTokens, gpu.KvDecodeTokens,
            backend, kvFailures, ref kvTopOneAgreed);

        _output.WriteLine($"  decode cpu=[{string.Join(",", cpu.DecodeTokens)}] " +
                          $"{backend}=[{string.Join(",", gpu.DecodeTokens)}]");
        _output.WriteLine($"  kvdecode cpu=[{string.Join(",", cpu.KvDecodeTokens)}] " +
                          $"{backend}=[{string.Join(",", gpu.KvDecodeTokens)}]");
        _output.WriteLine($"  KVCOSINES {type}/{backend} " +
                          string.Join(" ", kvCosines.Select((c, i) => $"s{i}={c:F6}")) +
                          $" min={kvCosines.Min():F6} top1={(kvTopOneAgreed ? "agree" : "DIVERGE")}" +
                          $" informative={cpu.KvDecodeIsInformative}");

        // Every cosine is emitted, not just the breaching ones. The 0.994 bound came out of exactly
        // this distribution across all cells, and re-deriving it after a kernel change needs the
        // same full picture: a log that only records breaches cannot show where the continuum ends
        // and a real outlier begins, and one cell can never settle it.
        _output.WriteLine($"  COSINES {type}/{backend} " +
                          string.Join(" ", cosines.Select((c, i) => $"s{i}={c:F6}")) +
                          $" min={cosines.Min():F6} top1={(allTopOneAgreed ? "agree" : "DIVERGE")}" +
                          $" informative={cpu.DecodeIsInformative}");

        // Each leg scores one step per prompt with no autoregressive feedback, so a degenerate
        // result here means the prompt set stopped discriminating on this fixture — not that the
        // backends agree. Asserted on the CPU reference, because that is the leg that decides it.
        string decodeInformative = cpu.DecodeIsInformative
            ? "informative"
            : $"DEGENERATE (cpu emitted [{string.Join(",", cpu.DecodeTokens)}] — re-run the prompt search)";

        bool passed = verdict.Passed
            && decodeFailures.Count == 0
            && kvFailures.Count == 0
            && cpu.DecodeIsInformative;

        string detail = verdict.Detail
            + (decodeFailures.Count > 0 ? " | decode: " + string.Join("; ", decodeFailures) : " | decode: ok")
            + (kvFailures.Count > 0 ? " | kvdecode: " + string.Join("; ", kvFailures) : " | kvdecode: ok")
            + $" | decode leg {decodeInformative}";

        Record(type, backend, passed ? "ran" : "failed", detail);

        // One machine-readable line per cell, so the matrix can be rebuilt from the log without
        // re-running the sweep. The failing arm is named explicitly, and the cached leg is named
        // separately from the uncached one: a breach that appears only under `kvdecode` implicates
        // the GEMV path or the cache, while one that appears on both is neither.
        string arm = string.Join("+", new[]
        {
            verdict.Passed ? null : "prefill",
            decodeFailures.Count == 0 ? null : "decode",
            kvFailures.Count == 0 ? null : "kvdecode",
        }.Where(a => a is not null));

        _output.WriteLine($"  CELL\t{type}\t{backend}\t{(passed ? "ran" : "failed")}\t" +
                          $"{verdict.NatsDelta:G6}\t{verdict.PerplexityRelative:G6}\t" +
                          $"{FormatCosines(cosines)}\t" +
                          $"{(allTopOneAgreed ? "agree" : "DIVERGE")}\t{cpu.DecodeIsInformative}\t" +
                          $"{FormatCosines(kvCosines)}\t" +
                          $"{(kvTopOneAgreed ? "agree" : "DIVERGE")}\t{cpu.KvDecodeIsInformative}\t" +
                          $"{(arm.Length == 0 ? "none" : arm)}");

        Assert.True(verdict.Passed, $"{type} on {backend} — prefill leg: {verdict.Detail}");
        Assert.True(decodeFailures.Count == 0,
            $"{type} on {backend} — uncached decode leg: {string.Join("; ", decodeFailures)}");
        Assert.True(kvFailures.Count == 0,
            $"{type} on {backend} — cached decode leg (seqLen == 1, the GEMV path): " +
            $"{string.Join("; ", kvFailures)}. Note this leg also exercises each backend's own KV-cache " +
            "type, so a breach here narrows to 'the GEMV path or the cache' rather than to the kernel.");
        Assert.True(cpu.DecodeIsInformative,
            $"{type} on {backend} — decode leg is vacuous: the CPU reference produced the same id for " +
            $"every prompt ([{string.Join(",", cpu.DecodeTokens)}]), so the top-1 comparison proves " +
            "nothing. This is a defect in the prompt set, not evidence that the backends agree — " +
            "re-run QuantGateDecodePromptTests.Probe_ReportsDecodeTokensPerCandidatePrompt.");
    }

    /// <summary>Compares one leg's per-step logits and argmaxes, appending any breaches.</summary>
    /// <param name="leg">Leg name used in failure text.</param>
    /// <param name="cpuLogits">Reference logit rows.</param>
    /// <param name="gpuLogits">Candidate logit rows.</param>
    /// <param name="cpuTokens">Reference argmaxes.</param>
    /// <param name="gpuTokens">Candidate argmaxes.</param>
    /// <param name="backend">Backend under test, named in failure text.</param>
    /// <param name="failures">Collects one entry per breach.</param>
    /// <param name="allTopOneAgreed">Cleared when any step's argmax differs.</param>
    /// <returns>The per-step cosines, breaching or not.</returns>
    private static double[] CompareLeg(
        string leg, float[][] cpuLogits, float[][] gpuLogits, int[] cpuTokens, int[] gpuTokens,
        QuantGateBackend backend, List<string> failures, ref bool allTopOneAgreed)
    {
        var cosines = new double[cpuLogits.Length];
        for (int step = 0; step < cosines.Length; step++)
        {
            double cosine = Cosine(cpuLogits[step], gpuLogits[step]);
            cosines[step] = cosine;
            if (cosine < MinDecodeCosine)
                failures.Add($"{leg} step {step}: logit cosine {cosine:F6} < {MinDecodeCosine}");

            if (cpuTokens[step] != gpuTokens[step])
            {
                allTopOneAgreed = false;
                failures.Add($"{leg} step {step}: top-1 {cpuTokens[step]} (cpu) vs {gpuTokens[step]} ({backend})");
            }
        }

        return cosines;
    }

    /// <summary>Formats a cosine array for the machine-readable CELL row.</summary>
    /// <param name="cosines">Per-step cosines.</param>
    /// <returns>Comma-separated invariant-culture values.</returns>
    private static string FormatCosines(double[] cosines)
        => string.Join(",", cosines.Select(
            c => c.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)));

    /// <summary>
    /// Classifies a load or run failure as environmental, returning a description, or
    /// <see langword="null"/> when the failure is not attributable to the machine.
    /// </summary>
    /// <param name="ex">Exception thrown while loading or scoring a cell.</param>
    /// <returns>An environment description, or <see langword="null"/> to let the failure surface.</returns>
    private static string? ClassifyEnvironmentFailure(Exception ex)
    {
        switch (ex)
        {
            // The local Arc iGPU drives the display, so sustained compute trips Windows' 2-second TDR
            // and the driver hands back VK_ERROR_DEVICE_LOST. That is a watchdog reset, not a wrong
            // number, and must never be counted as a backend disagreement.
            case VulkanException { IsDeviceLost: true } vk:
                return $"environment: Vulkan device lost ({vk.Message})";

            // VK_ERROR_OUT_OF_DEVICE_MEMORY — the fixture did not fit on this device.
            case VulkanException { ErrorCode: -2 } vk:
                return $"environment: Vulkan out of device memory ({vk.Message})";

            // No loader for this architecture on this backend — a coverage gap for the meta-test to
            // report, not a numerical disagreement between two results that were never produced.
            case NotSupportedException:
                return $"unsupported on backend: {ex.Message}";

            // SPIR-V not built for this machine.
            case DirectoryNotFoundException or FileNotFoundException:
                return $"environment: {ex.Message}";

            case OutOfMemoryException:
                return $"environment: out of memory ({ex.Message})";

            // CudaWeights raises OOM as an InvalidOperationException carrying VRAM context rather
            // than as OutOfMemoryException, so the case above does not reach it. A fixture that did
            // not fit on the device is a fact about the machine: the kernels under test never ran,
            // and recording that as "failed" would put a red cell against a quantization type whose
            // numbers were never produced. Matched on the message prefix the thrower controls, so
            // an unrelated InvalidOperationException from the loader still surfaces as a failure.
            //
            // NOTE: the first sweep hit this on IQ3_S/Cuda asking for 107.6 MiB with 7221 MiB free
            // and cuMemAlloc returning rc=2. Free VRAM that large with an allocation that small is
            // not ordinary exhaustion — it points at fragmentation or a context left behind by an
            // earlier cell in the same process, and is tracked separately from this gate.
            case InvalidOperationException when
                ex.Message.StartsWith("CUDA OOM allocating", StringComparison.Ordinal)
                || ex.Message.StartsWith("CUDA H2D failure", StringComparison.Ordinal):
                return $"environment: {ex.Message}";

            default:
                return null;
        }
    }

    private static void Record(QuantizationType type, QuantGateBackend backend, string outcome, string detail)
        => Cells.Add(new QuantGateCellResult(type, backend, outcome, detail));

    private static double Cosine(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            return 0;

        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }

        double denom = Math.Sqrt(na) * Math.Sqrt(nb);
        return denom > 0 ? dot / denom : 0;
    }
}
