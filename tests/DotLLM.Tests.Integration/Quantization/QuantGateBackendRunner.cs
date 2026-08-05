using DotLLM.Core.Configuration;
using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Evaluation;
using DotLLM.Models;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;

namespace DotLLM.Tests.Integration.Quantization;

/// <summary>Backends the gate can score a fixture on.</summary>
public enum QuantGateBackend
{
    /// <summary>CPU reference path.</summary>
    Cpu,

    /// <summary>CUDA, device 0.</summary>
    Cuda,

    /// <summary>Vulkan, default device.</summary>
    Vulkan,
}

/// <summary>Resolves the shared wikitext corpus used by the perplexity leg.</summary>
public static class QuantGateCorpus
{
    /// <summary>Environment variable overriding the corpus path.</summary>
    public const string EnvVar = "DOTLLM_QUANT_GATE_CORPUS";

    /// <summary>
    /// Absolute path to the corpus. Git-ignored and never committed; see
    /// <c>.docs/corpora/QUANT_FIXTURES.md</c> for provenance.
    /// </summary>
    public static string Path =>
        Environment.GetEnvironmentVariable(EnvVar)
        ?? System.IO.Path.GetFullPath(System.IO.Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            ".docs", "corpora", "wikitext2-test.raw"));
}

/// <summary>Both legs of one (fixture, backend) cell.</summary>
/// <param name="Perplexity">Teacher-forced result — the prefill/GEMM leg.</param>
/// <param name="DecodeTokens">Greedy token ids emitted, in order — the decode/GEMV leg.</param>
/// <param name="DecodeLogits">Full logit vector at each decode step.</param>
/// <remarks>
/// <para>
/// <b>Do not compare two runs by record equality, and never compare
/// <see cref="PerplexityResult.WindowCount"/> across backends.</b> The window count is an artefact of
/// which evaluator path the backend's logits shape selected, not of the kernels being measured: a
/// backend returning all rows (the CPU transformer) takes
/// <c>PerplexityEvaluator.TeacherForcedSinglePass</c> and reports <c>WindowCount == 1</c>, while a
/// last-row-only backend (CUDA and Vulkan dense) takes <c>TeacherForcedGrowingPrefix</c> and reports
/// <c>WindowCount == ctx - 1</c>. Comparing it would fail every cell for a reason that has nothing to
/// do with quantization.
/// </para>
/// <para>
/// <b>Comparable across backends:</b> <see cref="PerplexityResult.MeanNegativeLogLikelihood"/>,
/// <see cref="PerplexityResult.Perplexity"/> and <see cref="PerplexityResult.ScoredTokens"/> (the
/// last of which must <i>match</i> for the others to mean anything), plus
/// <see cref="DecodeTokens"/> and <see cref="DecodeLogits"/>.
/// </para>
/// </remarks>
public sealed record QuantGateRun(PerplexityResult Perplexity, int[] DecodeTokens, float[][] DecodeLogits)
{
    /// <summary>
    /// Whether the decode leg emitted more than one distinct token, and so actually discriminates.
    /// </summary>
    /// <remarks>
    /// Decoding is greedy with no repetition penalty (it must stay deterministic to be comparable),
    /// so a small model can emit EOS immediately or lock into a repeat. Every step then carries the
    /// same id, and a token-level cross-backend comparison becomes vacuously equal — passing the
    /// decode/GEMV leg without exercising it, which is the exact failure the two-leg design exists to
    /// prevent. Callers should assert on this rather than assume the leg was informative; a false
    /// value means "choose a different decode prompt", not "the backends agree".
    /// <para>Note that <see cref="DecodeLogits"/> stays discriminating either way — identical tokens
    /// can still sit on materially different logit vectors. This flag is about the token-level
    /// comparison only.</para>
    /// </remarks>
    public bool DecodeIsInformative => DecodeTokens.Distinct().Count() > 1;
}

/// <summary>
/// Loads a ladder fixture on one backend and scores both legs (#256).
/// </summary>
/// <remarks>
/// <para>
/// <b>Two legs, because they exercise different kernels.</b> Perplexity is prefill-dominated and
/// therefore covers GEMM; short greedy generation covers the decode GEMV. The very first BF16 cell
/// measured passed perplexity and failed generation, so a perplexity-only matrix would have
/// reported BF16 as fully covered.
/// </para>
/// <para>
/// <b>Per-architecture loaders, not the plain transformer loader.</b> Hybrid architectures have no
/// <c>attn_output.weight</c> on a GDN layer, so bypassing the dispatch fails with a missing-key
/// error (#259).
/// </para>
/// </remarks>
public static class QuantGateBackendRunner
{
    /// <summary>Reports whether a backend can be used on this machine.</summary>
    /// <param name="backend">Backend to probe.</param>
    /// <returns><see langword="true"/> when the backend is usable.</returns>
    /// <remarks>
    /// Every probe is wrapped: a missing <c>nvcuda.dll</c> or <c>vulkan-1.dll</c> surfaces as a
    /// <see cref="DllNotFoundException"/> from the P/Invoke stub rather than a <see langword="false"/>,
    /// and the gate must report "backend absent" rather than fail the whole matrix.
    /// </remarks>
    public static bool IsAvailable(QuantGateBackend backend)
    {
        switch (backend)
        {
            case QuantGateBackend.Cpu:
                return true;

            case QuantGateBackend.Cuda:
                try
                {
                    return DotLLM.Cuda.CudaDevice.IsAvailable();
                }
                catch (Exception)
                {
                    return false;
                }

            case QuantGateBackend.Vulkan:
                try
                {
                    return DotLLM.Vulkan.VulkanDevice.IsAvailable();
                }
                catch (Exception)
                {
                    return false;
                }

            default:
                return false;
        }
    }

    /// <summary>Scores both legs of one cell.</summary>
    /// <param name="entry">Fixture to load.</param>
    /// <param name="backend">Backend to run on.</param>
    /// <param name="corpusPath">UTF-8 corpus for the perplexity leg.</param>
    /// <param name="corpusTokens">Upper bound on corpus tokens consumed.</param>
    /// <param name="decodePrompt">Prompt for the decode leg.</param>
    /// <param name="decodeSteps">Number of greedy steps to take.</param>
    /// <returns>Both legs' results.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="entry"/> is <see langword="null"/>.</exception>
    public static QuantGateRun Run(
        QuantLadderEntry entry, QuantGateBackend backend, string corpusPath,
        int corpusTokens, string decodePrompt, int decodeSteps)
    {
        ArgumentNullException.ThrowIfNull(entry);
        ArgumentException.ThrowIfNullOrEmpty(corpusPath);
        ArgumentOutOfRangeException.ThrowIfLessThan(decodeSteps, 1);

        using GgufFile gguf = GgufFile.Open(entry.FilePath);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        ITokenizer tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        IModel model;
        IDisposable? ownedDevice = null;
        int deviceId;

        switch (backend)
        {
            case QuantGateBackend.Cuda:
                (model, _) = DotLLM.Cuda.CudaModelLoader.CreateFromGguf(gguf, config, 0);
                deviceId = 0;
                break;

            case QuantGateBackend.Vulkan:
                (model, ownedDevice) = LoadVulkan(gguf, config);
                deviceId = 0;
                break;

            default:
                model = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0));
                deviceId = -1;
                break;
        }

        // Nested rather than a single finally with two disposals: if model.Dispose() throws, a flat
        // finally would skip the device disposal and leak the same handles the catch above exists to
        // protect.
        try
        {
            try
            {
                var tokens = new List<int>();
                using (var reader = new StreamReader(corpusPath))
                {
                    foreach (int id in CorpusReader.StreamTokens(reader, tokenizer, corpusTokens))
                        tokens.Add(id);
                }

                // Probed, not assumed: CUDA returns last-row-only, and so does Vulkan — a type-based
                // assumption here would silently produce a wrong perplexity.
                bool returnsAllRows = BackendPerplexityModel.Probe(model, deviceId);
                var pplModel = new BackendPerplexityModel(model, deviceId, returnsAllRows);

                // Teacher-forced: the only mode comparable across backends, since sliding-window
                // requires all-row logits which the GPU paths do not return.
                int ctx = Math.Min(entry.ContextLength, config.MaxSequenceLength);
                var options = new PerplexityOptions(PerplexityMode.TeacherForced, ctx, ctx);
                PerplexityResult ppl = PerplexityEvaluator.Evaluate(
                    pplModel, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(tokens), options);

                // The recurrent state left by the perplexity run must not leak into decode (#261).
                pplModel.ResetState();

                var (decodeTokens, decodeLogits) = RunDecode(
                    model, tokenizer, deviceId, decodePrompt, decodeSteps);
                return new QuantGateRun(ppl, decodeTokens, decodeLogits);
            }
            finally
            {
                model.Dispose();
            }
        }
        finally
        {
            ownedDevice?.Dispose();
        }
    }

    /// <summary>
    /// Creates a Vulkan device and loads the model onto it, releasing the device if the load fails.
    /// </summary>
    /// <param name="gguf">Open fixture.</param>
    /// <param name="config">Config extracted from <paramref name="gguf"/>.</param>
    /// <returns>The model and the device it owns; the caller disposes both.</returns>
    /// <remarks>
    /// <para><b>The device outlives its own constructor call and must not leak on a failed load.</b>
    /// Load failure is not the exceptional case for a coverage gate — it is the expected outcome for
    /// exactly the cells the gate exists to find: <c>VulkanModelLoader.CreateFromGguf</c> throws
    /// <see cref="NotSupportedException"/> for architectures with no Vulkan loader, plus
    /// missing-tensor and out-of-memory paths, and <c>ResolveSpvDir</c> throws when the SPIR-V has
    /// not been built. Without the catch below, sweeping 21 fixtures leaks a <c>VkDevice</c> and
    /// <c>VkInstance</c> per failing cell until a later cell dies of handle exhaustion and the
    /// failure gets attributed to a kernel.</para>
    /// <para>Extracted into its own method rather than inlined in the switch so the device is
    /// <i>returned</i>, which makes the ownership transfer visible to the caller and to the
    /// IDisposable analyzers.</para>
    /// </remarks>
    private static (IModel Model, IDisposable Device) LoadVulkan(GgufFile gguf, ModelConfig config)
    {
        var device = DotLLM.Vulkan.VulkanDevice.Create();
        try
        {
            (IModel model, _) = DotLLM.Vulkan.VulkanModelLoader.CreateFromGguf(
                device, gguf, config, ResolveSpvDir());
            return (model, device);
        }
        catch (Exception)
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>Greedy-decodes <paramref name="steps"/> tokens, capturing each step's logits.</summary>
    /// <param name="model">Loaded model; not owned here.</param>
    /// <param name="tokenizer">Tokenizer belonging to the same GGUF.</param>
    /// <param name="deviceId">Forward-pass device; <c>-1</c> is CPU.</param>
    /// <param name="prompt">Prompt to seed the context with.</param>
    /// <param name="steps">Number of greedy steps.</param>
    /// <returns>Emitted ids and the full logit vector behind each.</returns>
    private static (int[] Tokens, float[][] Logits) RunDecode(
        IModel model, ITokenizer tokenizer, int deviceId, string prompt, int steps)
    {
        int[] promptTokens = tokenizer.Encode(prompt);
        var context = new List<int>(promptTokens);
        var emitted = new int[steps];
        var logits = new float[steps][];
        int vocab = model.Config.VocabSize;

        for (int step = 0; step < steps; step++)
        {
            var positions = new int[context.Count];
            for (int i = 0; i < positions.Length; i++)
                positions[i] = i;

            // Each step is scored as an independent sequence from position 0, so any recurrent state
            // the previous step left behind must go first — see #261.
            model.ResetSequenceState();

            // Full re-prefill each step rather than an incremental KV step: it is slower, but it
            // is identical across all three backends, and a KV-cache difference would otherwise be
            // scored as a kernel difference.
            using ITensor output = model.Forward(
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(context), positions, deviceId);
            float[] row = LastRowOf(output, vocab);

            int best = 0;
            for (int v = 1; v < row.Length; v++)
            {
                if (row[v] > row[best])
                    best = v;
            }

            logits[step] = row;
            emitted[step] = best;
            context.Add(best);
        }

        return (emitted, logits);
    }

    /// <summary>
    /// Copies the final row of a logits tensor to a managed array, whatever shape the backend
    /// returned it in.
    /// </summary>
    /// <param name="output">Logits from one forward pass.</param>
    /// <param name="vocab">Row length, taken from the model config rather than the tensor shape.</param>
    /// <returns>The last row's <paramref name="vocab"/> floats.</returns>
    /// <remarks>
    /// <para><b>The shape genuinely differs by backend.</b> The CPU transformer returns
    /// <c>[seqLen, vocab]</c> while CUDA and Vulkan return only the final row — the same asymmetry
    /// <see cref="BackendPerplexityModel.Probe"/> exists to measure. Deriving the offset from
    /// <see cref="ITensor.ElementCount"/> rather than branching on row count is therefore correct
    /// for both: the last row always ends at the last element.</para>
    /// <para><b>Read through <see cref="ITensor.DataPointer"/>, as the evaluator does.</b>
    /// <see cref="ITensor"/> exposes no copy-out method; <c>PerplexityEvaluator</c> reads logits by
    /// constructing a span over the pointer, and GPU backends already stage their logits into host
    /// memory before returning, so the same read works unchanged on every backend.</para>
    /// </remarks>
    private static unsafe float[] LastRowOf(ITensor output, int vocab)
    {
        long total = output.ElementCount;
        if (total < vocab)
            throw new InvalidOperationException(
                $"Logits tensor holds {total} elements, fewer than one row of {vocab}.");

        // A padded or otherwise non-vocab row length would make `total - vocab` land mid-row and the
        // read straddle two positions — a wrong measurement that produces no error. Not reachable
        // with today's backends; asserted anyway, because the whole point of this file is never to
        // silently measure the wrong thing.
        if (total % vocab != 0)
            throw new InvalidOperationException(
                $"Logits tensor holds {total} elements, not a whole multiple of the vocabulary ({vocab}); " +
                "the row stride is not the vocabulary size and the last row cannot be located by offset.");

        var row = new float[vocab];
        var source = new ReadOnlySpan<float>((void*)(output.DataPointer + (nint)(total - vocab) * sizeof(float)), vocab);
        source.CopyTo(row);
        return row;
    }

    /// <summary>
    /// Resolves the SPIR-V blob directory for Vulkan: <c>spv/</c> next to the test assembly, falling
    /// back to the in-repo <c>native/vulkan/spv</c> when running from the source tree.
    /// </summary>
    /// <returns>Absolute path to a directory containing compiled SPIR-V.</returns>
    private static string ResolveSpvDir()
    {
        string[] candidates =
        [
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        ];

        foreach (string c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }

        throw new DirectoryNotFoundException(
            "Vulkan SPIR-V directory not found. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");
    }
}
