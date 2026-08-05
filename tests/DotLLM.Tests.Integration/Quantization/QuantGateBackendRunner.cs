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

/// <summary>All three legs of one (fixture, backend) cell.</summary>
/// <param name="Perplexity">Teacher-forced result over the corpus — the bulk prefill/GEMM leg.</param>
/// <param name="DecodeTokens">Argmax id from each prompt's uncached forward — the short-context leg.</param>
/// <param name="DecodeLogits">Full logit vector behind each <see cref="DecodeTokens"/> entry.</param>
/// <param name="KvDecodeTokens">Argmax id from each prompt's single cached <c>seqLen == 1</c> step — the decode/GEMV leg.</param>
/// <param name="KvDecodeLogits">Full logit vector behind each <see cref="KvDecodeTokens"/> entry.</param>
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
/// <see cref="DecodeTokens"/>, <see cref="DecodeLogits"/>, <see cref="KvDecodeTokens"/> and
/// <see cref="KvDecodeLogits"/>.
/// </para>
/// <para>
/// <b>Why three legs and not two.</b> The gate was specified with a prefill leg and a decode leg
/// because the original matrix caught a BF16 defect that passed perplexity and failed generation.
/// But both of the first two legs are GEMM-shaped: the CPU fused-decode/GEMV path is gated on
/// <c>seqLen == 1</c> (<c>TransformerModel.cs:1244</c> and <c>:1606</c>), and neither a corpus
/// window nor a whole-prompt forward ever has a sequence length of one. Without
/// <see cref="KvDecodeTokens"/> the gate would claim two legs and deliver one of them twice,
/// leaving the GEMV path exactly as unexercised as it was before #256.
/// </para>
/// </remarks>
public sealed record QuantGateRun(
    PerplexityResult Perplexity,
    int[] DecodeTokens, float[][] DecodeLogits,
    int[] KvDecodeTokens, float[][] KvDecodeLogits)
{
    /// <summary>
    /// Whether the short-context leg produced more than one distinct token, and so actually
    /// discriminates at the token level.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A token-level comparison across backends is vacuous when every entry carries the same id:
    /// the two sides compare equal for free, and the leg passes without exercising anything.
    /// </para>
    /// <para>
    /// <b>This is why the legs score one step each from several prompts rather than several
    /// autoregressive steps from one prompt.</b> Measured: on the <c>--pure</c> Q2_K and Q3_K 1B
    /// fixtures, greedy self-feedback is a fixed point — feeding the model's own argmax back makes
    /// it re-emit that id forever. A search over ten candidate prompts found the degeneracy on
    /// both fixtures for all ten, so no prompt fixes it. But the <i>first</i> token varies richly
    /// with the prompt (Q2_K produced 67585, 55934, 127327, 37424, … across the candidates), so
    /// independent single steps from distinct prompts discriminate where an autoregressive chain
    /// cannot. The chosen quad yields four distinct ids on all 21 fixtures.
    /// </para>
    /// <para>Note that the logit vectors stay discriminating either way — identical tokens can sit
    /// on materially different logit vectors. This flag is about the token-level comparison only.</para>
    /// </remarks>
    public bool DecodeIsInformative => DecodeTokens.Distinct().Count() > 1;

    /// <summary>Whether the cached <c>seqLen == 1</c> leg produced more than one distinct token.</summary>
    public bool KvDecodeIsInformative => KvDecodeTokens.Distinct().Count() > 1;
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
        int corpusTokens, IReadOnlyList<string> decodePrompts)
    {
        ArgumentNullException.ThrowIfNull(entry);
        ArgumentException.ThrowIfNullOrEmpty(corpusPath);
        ArgumentNullException.ThrowIfNull(decodePrompts);
        ArgumentOutOfRangeException.ThrowIfLessThan(decodePrompts.Count, 1);

        using GgufFile gguf = GgufFile.Open(entry.FilePath);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        ITokenizer tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        IModel model;
        IDisposable? ownedDevice = null;
        int deviceId;

        // Captured rather than discarded: each architecture needs its own concrete cache type and
        // there is no common CreateKvCache interface, so the loaders hand the pairing back here.
        // The cached seqLen == 1 leg cannot be run without it.
        Func<int, DotLLM.Core.Attention.IKvCache> kvCacheFactory;

        switch (backend)
        {
            case QuantGateBackend.Cuda:
                (model, kvCacheFactory) = DotLLM.Cuda.CudaModelLoader.CreateFromGguf(gguf, config, 0);
                deviceId = 0;
                break;

            case QuantGateBackend.Vulkan:
                (model, ownedDevice, kvCacheFactory) = LoadVulkan(gguf, config);
                deviceId = 0;
                break;

            default:
                model = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0));
                deviceId = -1;

                // The CPU transformer has no CreateKvCache of its own; SimpleKvCache is the dense
                // cache its Forward overload consumes, sized from the same config the model loaded.
                kvCacheFactory = maxSeqLen => new DotLLM.Engine.KvCache.SimpleKvCache(
                    DotLLM.Core.Attention.KvGeometry.FromConfig(config), maxSeqLen);
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
                    model, tokenizer, deviceId, decodePrompts);
                var (kvTokens, kvLogits) = RunKvDecode(
                    model, tokenizer, deviceId, kvCacheFactory, decodePrompts);
                return new QuantGateRun(ppl, decodeTokens, decodeLogits, kvTokens, kvLogits);
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
    private static (IModel Model, IDisposable Device, Func<int, DotLLM.Core.Attention.IKvCache> KvCacheFactory)
        LoadVulkan(GgufFile gguf, ModelConfig config)
    {
        var device = DotLLM.Vulkan.VulkanDevice.Create();
        try
        {
            (IModel model, Func<int, DotLLM.Core.Attention.IKvCache> kvCacheFactory) =
                DotLLM.Vulkan.VulkanModelLoader.CreateFromGguf(device, gguf, config, ResolveSpvDir());
            return (model, device, kvCacheFactory);
        }
        catch (Exception)
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Greedy-decodes several candidate prompts against one fixture on the CPU reference path,
    /// loading the model once.
    /// </summary>
    /// <param name="entry">Fixture to load.</param>
    /// <param name="prompts">Candidate decode prompts to try.</param>
    /// <returns>The argmax id each prompt produced, in the order given.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="entry"/> or <paramref name="prompts"/> is <see langword="null"/>.</exception>
    /// <remarks>
    /// <para>
    /// Exists to choose <c>CrossBackendQuantGateTests.DecodePrompts</c> by measurement. The gate
    /// requires the CPU reference to produce more than one distinct id across its steps: when every
    /// step carries the same id, the top-1 arm compares equal for free and "the backends agree"
    /// becomes a statement about the prompts rather than about the kernels.
    /// </para>
    /// <para>
    /// <b>CPU only, deliberately.</b> The reference leg is the one that decides whether a prompt
    /// set discriminates, so a GPU arm here would add load cost and device-lifetime handling
    /// without changing the answer. It also means this probe runs under <c>Category=Fixtures</c>,
    /// on a machine with the ladder but no GPU.
    /// </para>
    /// <para>
    /// The model is loaded once and every prompt scored against it, because loading dominates: a 1B
    /// fixture's cold load runs to minutes while a handful of short forwards take seconds.
    /// </para>
    /// </remarks>
    public static IReadOnlyList<int> ProbeDecodePrompts(
        QuantLadderEntry entry, IReadOnlyList<string> prompts)
    {
        ArgumentNullException.ThrowIfNull(entry);
        ArgumentNullException.ThrowIfNull(prompts);
        ArgumentOutOfRangeException.ThrowIfLessThan(prompts.Count, 1);

        using GgufFile gguf = GgufFile.Open(entry.FilePath);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        ITokenizer tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        IModel model = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0));
        try
        {
            (int[] tokens, _) = RunDecode(model, tokenizer, -1, prompts);
            return tokens;
        }
        finally
        {
            model.Dispose();
        }
    }

    /// <summary>
    /// Scores one uncached forward per prompt and returns each one's argmax and full logit row.
    /// </summary>
    /// <param name="model">Loaded model; not owned here.</param>
    /// <param name="tokenizer">Tokenizer belonging to the same GGUF.</param>
    /// <param name="deviceId">Forward-pass device; <c>-1</c> is CPU.</param>
    /// <param name="prompts">Prompts to score, one forward each.</param>
    /// <returns>The argmax id per prompt and the full logit vector behind each.</returns>
    /// <remarks>
    /// <para>
    /// <b>One step per prompt, not several steps per prompt.</b> Autoregressive greedy decode is a
    /// fixed point on a near-destroyed <c>--pure</c> fixture: feeding the model's own argmax back
    /// makes it re-emit that id forever. Measured over ten candidate prompts, Q2_K and Q3_K on
    /// Llama-3.2-1B were degenerate for every one of them, so no choice of prompt avoids it. The
    /// first id, however, varies richly with the prompt, so independent forwards from distinct
    /// prompts discriminate where a chain cannot.
    /// </para>
    /// <para>
    /// No KV-cache here on purpose — this leg is identical across all three backends, so a
    /// cache-management difference cannot be scored as a kernel difference.
    /// <see cref="RunKvDecode"/> is the leg that deliberately does use one.
    /// </para>
    /// </remarks>
    private static (int[] Tokens, float[][] Logits) RunDecode(
        IModel model, ITokenizer tokenizer, int deviceId, IReadOnlyList<string> prompts)
    {
        var emitted = new int[prompts.Count];
        var logits = new float[prompts.Count][];
        int vocab = model.Config.VocabSize;

        for (int i = 0; i < prompts.Count; i++)
        {
            int[] tokens = tokenizer.Encode(prompts[i]);
            var positions = new int[tokens.Length];
            for (int p = 0; p < positions.Length; p++)
                positions[p] = p;

            // Each prompt is an independent sequence from position 0, so any recurrent state the
            // previous one left behind must go first — see #261.
            model.ResetSequenceState();

            using ITensor output = model.Forward(tokens, positions, deviceId);
            float[] row = LastRowOf(output, vocab);

            logits[i] = row;
            emitted[i] = ArgMax(row);
        }

        return (emitted, logits);
    }

    /// <summary>
    /// Scores one cached single-token step per prompt — the only leg that reaches the decode/GEMV
    /// path.
    /// </summary>
    /// <param name="model">Loaded model; not owned here.</param>
    /// <param name="tokenizer">Tokenizer belonging to the same GGUF.</param>
    /// <param name="deviceId">Forward-pass device; <c>-1</c> is CPU.</param>
    /// <param name="kvCacheFactory">Backend-appropriate cache factory, from the model's loader.</param>
    /// <param name="prompts">Prompts to seed each cache with, one cached step each.</param>
    /// <returns>The argmax id from each cached step and the full logit vector behind it.</returns>
    /// <remarks>
    /// <para>
    /// <b>Why this leg exists.</b> The CPU fused-decode/GEMV kernels are gated on
    /// <c>seqLen == 1</c> (<c>TransformerModel.cs:1244</c> and <c>:1606</c>). Neither a corpus
    /// window nor a whole-prompt forward ever has a sequence length of one, so without this leg the
    /// gate would test GEMM twice and leave GEMV exactly as uncovered as it was before #256 — while
    /// reporting that it had covered both.
    /// </para>
    /// <para>
    /// <b>The cost, stated plainly.</b> Each backend brings its own concrete cache type, so this
    /// leg mixes cache-management behaviour into the comparison in a way the other two legs
    /// deliberately avoid. A disagreement here is therefore weaker evidence about a quantization
    /// kernel than a disagreement on the other legs: it narrows to "the GEMV path or the cache",
    /// not to "the GEMV path". That is still strictly better than not exercising the path at all,
    /// and the prompt-shaped prefill immediately before each step is shared, so a cache that
    /// disagreed about the <i>prompt</i> would show up on the other legs first.
    /// </para>
    /// <para>
    /// One cached step per prompt rather than several, for the same fixed-point reason as
    /// <see cref="RunDecode"/>: distinct prompts give distinct contexts, and a chain does not.
    /// </para>
    /// </remarks>
    private static (int[] Tokens, float[][] Logits) RunKvDecode(
        IModel model, ITokenizer tokenizer, int deviceId,
        Func<int, DotLLM.Core.Attention.IKvCache> kvCacheFactory, IReadOnlyList<string> prompts)
    {
        var emitted = new int[prompts.Count];
        var logits = new float[prompts.Count][];
        int vocab = model.Config.VocabSize;

        for (int i = 0; i < prompts.Count; i++)
        {
            int[] tokens = tokenizer.Encode(prompts[i]);
            var positions = new int[tokens.Length];
            for (int p = 0; p < positions.Length; p++)
                positions[p] = p;

            model.ResetSequenceState();

            // A fresh cache per prompt: reusing one would carry the previous prompt's keys and
            // values into this prompt's attention, which is a different measurement entirely.
            DotLLM.Core.Attention.IKvCache cache = kvCacheFactory(tokens.Length + 1);
            try
            {
                int seed;
                using (ITensor prefill = model.Forward(tokens, positions, deviceId, cache))
                    seed = ArgMax(LastRowOf(prefill, vocab));

                // The one call in the whole gate with seqLen == 1. Everything above this line is
                // setup; this is the measurement.
                //
                // Verified to be load-bearing by sabotage: returning the prefill row here instead
                // of taking this step turns Run_CachedLegIsNotARepeatOfTheUncachedOne red, while
                // every other assertion in the class stays green. That is the failure mode — GEMV
                // coverage reported but not performed — and it is caught.
                using ITensor step = model.Forward(
                    [seed], [tokens.Length], deviceId, cache);
                float[] row = LastRowOf(step, vocab);

                logits[i] = row;
                emitted[i] = ArgMax(row);
            }
            finally
            {
                (cache as IDisposable)?.Dispose();
            }
        }

        return (emitted, logits);
    }

    /// <summary>Index of the largest element.</summary>
    /// <param name="row">Logit row to scan.</param>
    /// <returns>The argmax index.</returns>
    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int v = 1; v < row.Length; v++)
        {
            if (row[v] > row[best])
                best = v;
        }

        return best;
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
