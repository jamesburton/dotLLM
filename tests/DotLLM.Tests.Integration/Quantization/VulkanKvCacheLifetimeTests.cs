using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Quantization;

/// <summary>
/// Whether a Vulkan model's output depends on the <i>lifetime</i> of the KV caches handed to it.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this exists.</b> The #256 gate's cached <c>seqLen == 1</c> leg found every Vulkan cell
/// diverging from CPU — including F16 and BF16, which are not quantized at all, so it cannot be a
/// quantization kernel. The per-step shape pointed somewhere specific: the first prompt was always
/// near-perfect (cosine ~0.9997) and later prompts degraded badly (down to 0.27), even though each
/// prompt gets a freshly constructed cache and is therefore supposed to be independent.
/// </para>
/// <para>
/// <b>The hypothesis.</b> <c>DescriptorSetCache.GetOrCreate</c> keys its cached descriptor sets on
/// the raw <c>VkBuffer</c> handles of the dispatch. Vulkan handles are recyclable: after
/// <c>vkDestroyBuffer</c>, the driver may hand the same numeric handle back for the next
/// allocation. A caller that creates a cache, uses it, destroys it and creates another can
/// therefore hit a descriptor set that was written against the destroyed buffer — a stale binding
/// that reads freed memory, with no error anywhere. CUDA does not have this problem because
/// <c>CudaTransformerModel</c> tracks KV-cache identity explicitly and re-captures when it changes
/// (see its remarks at <c>CudaTransformerModel.cs:192</c> and <c>:1794</c>); Vulkan has no
/// equivalent.
/// </para>
/// <para>
/// <b>Why the test is Vulkan-only.</b> Comparing against CPU would leave the conclusion ambiguous
/// between "Vulkan is wrong" and "the two backends differ". Comparing Vulkan against <i>itself</i>
/// under two cache-lifetime policies removes the second possibility entirely: the same backend,
/// the same weights, the same prompts and the same arithmetic must produce the same logits, and if
/// they do not then cache lifetime is affecting results — which is a defect however the numbers
/// compare to any other backend. This is the "check the reference against its own tier first"
/// discipline that settled the #229 parity question.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("QuantLadder")]
public sealed class VulkanKvCacheLifetimeTests
{
    private static readonly string[] Prompts = CrossBackendQuantGateTests.DecodePrompts;

    private readonly QuantLadderFixture _ladder;
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the probe against the shared ladder index.</summary>
    /// <param name="ladder">Ladder index shared across the <c>QuantLadder</c> collection.</param>
    /// <param name="output">xunit sink for the measured logits.</param>
    public VulkanKvCacheLifetimeTests(QuantLadderFixture ladder, ITestOutputHelper output)
    {
        _ladder = ladder;
        _output = output;
    }

    /// <summary>
    /// Scoring the same prompts with caches destroyed between them, versus kept alive, must give
    /// identical logits.
    /// </summary>
    [SkippableFact]
    public void CachedDecode_DoesNotDependOnCacheLifetime()
    {
        QuantLadderEntry? entry = _ladder.Available.FirstOrDefault(e => e.Type == QuantizationType.F16);
        Skip.If(entry is null, $"F16 fixture not present under {_ladder.RootDirectory}");
        Skip.IfNot(QuantGateBackendRunner.IsAvailable(QuantGateBackend.Vulkan), "Vulkan not available");

        // F16 deliberately: unquantized, so nothing here can be blamed on a quant kernel.
        float[][] churned = ScoreVulkan(entry!, disposeBetweenPrompts: true);
        float[][] retained = ScoreVulkan(entry!, disposeBetweenPrompts: false);

        var mismatches = new List<string>();
        for (int i = 0; i < Prompts.Length; i++)
        {
            double cosine = Cosine(churned[i], retained[i]);
            int churnedArgMax = ArgMax(churned[i]);
            int retainedArgMax = ArgMax(retained[i]);

            _output.WriteLine(
                $"  LIFETIME\tprompt {i}\tcosine={cosine:F6}\t" +
                $"churned argmax={churnedArgMax}\tretained argmax={retainedArgMax}");

            // Bit-identical is the correct bar. Both arms run the same kernels over the same
            // weights with the same reduction order on the same device; the ONLY difference is
            // when the caches are destroyed. Any difference at all is the finding.
            if (!churned[i].AsSpan().SequenceEqual(retained[i]))
                mismatches.Add($"prompt {i}: cosine {cosine:F6}, argmax {churnedArgMax} vs {retainedArgMax}");
        }

        Assert.True(mismatches.Count == 0,
            "Vulkan's cached-decode logits depend on when the KV caches are destroyed: " +
            string.Join("; ", mismatches) + ". The two arms differ only in cache lifetime — same " +
            "device, weights, prompts and kernels — so this is a defect in Vulkan's cache/descriptor " +
            "handling, not a cross-backend difference. Suspect VkBuffer handle recycling colliding " +
            "with DescriptorSetCache's handle-keyed reuse.");
    }

    /// <summary>
    /// Measures how much of the Vulkan/CPU cached-decode gap the cache-lifetime defect accounts
    /// for, by comparing BOTH lifetime policies against the CPU reference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why this is separate from the lifetime test.</b> The lifetime test proves a defect exists
    /// but says nothing about its size relative to the gap the #256 gate actually measured. The
    /// numbers did not obviously reconcile: the two Vulkan arms differ by ~0.98 cosine while Vulkan
    /// differs from CPU by 0.885 on the same fixture. If cache lifetime were the whole story, the
    /// retained arm would land on CPU; if it does not, there is a second and larger defect and
    /// fixing the handle recycling alone would leave the gate red while looking like progress.
    /// </para>
    /// <para>
    /// Diagnostic, not a gate: it always passes and reports the three cosines per prompt, because
    /// its job is to apportion a known gap rather than to decide whether one exists.
    /// </para>
    /// </remarks>
    [SkippableFact]
    public void CachedDecode_ApportionsTheGapAgainstCpu()
    {
        QuantLadderEntry? entry = _ladder.Available.FirstOrDefault(e => e.Type == QuantizationType.F16);
        Skip.If(entry is null, $"F16 fixture not present under {_ladder.RootDirectory}");
        Skip.IfNot(QuantGateBackendRunner.IsAvailable(QuantGateBackend.Vulkan), "Vulkan not available");

        float[][] churned = ScoreVulkan(entry!, disposeBetweenPrompts: true);
        float[][] retained = ScoreVulkan(entry!, disposeBetweenPrompts: false);

        QuantGateRun cpu = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path, 256, Prompts);

        _output.WriteLine("  prompt  churned-vs-cpu  retained-vs-cpu  churned-vs-retained");
        for (int i = 0; i < Prompts.Length; i++)
        {
            _output.WriteLine(
                $"  APPORTION\t{i}\t{Cosine(churned[i], cpu.KvDecodeLogits[i]):F6}\t" +
                $"{Cosine(retained[i], cpu.KvDecodeLogits[i]):F6}\t" +
                $"{Cosine(churned[i], retained[i]):F6}");
        }

        _output.WriteLine(
            "  If retained-vs-cpu is ~1.0 the cache-lifetime defect explains the whole gap. " +
            "If it stays low while churned-vs-retained is high, a second and larger defect remains.");
    }

    /// <summary>Runs the prompt set on Vulkan under one cache-lifetime policy.</summary>
    /// <param name="entry">Fixture to load.</param>
    /// <param name="disposeBetweenPrompts">
    /// When true, each prompt's cache is destroyed before the next is created — the policy the gate
    /// uses. When false, every cache is held until all prompts are scored, so no handle can be
    /// recycled mid-run.
    /// </param>
    /// <returns>The cached step's logit row per prompt.</returns>
    private static float[][] ScoreVulkan(QuantLadderEntry entry, bool disposeBetweenPrompts)
    {
        using GgufFile gguf = GgufFile.Open(entry.FilePath);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        ITokenizer tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        using var device = DotLLM.Vulkan.VulkanDevice.Create();
        (IModel model, Func<int, IKvCache> kvFactory) =
            DotLLM.Vulkan.VulkanModelLoader.CreateFromGguf(device, gguf, config, ResolveSpvDir());

        var held = new List<IDisposable>();
        try
        {
            var rows = new float[Prompts.Length][];
            int vocab = config.VocabSize;

            for (int i = 0; i < Prompts.Length; i++)
            {
                int[] tokens = tokenizer.Encode(Prompts[i]);
                var positions = new int[tokens.Length];
                for (int p = 0; p < positions.Length; p++)
                    positions[p] = p;

                model.ResetSequenceState();
                IKvCache cache = kvFactory(tokens.Length + 1);

                int seed;
                using (ITensor prefill = model.Forward(tokens, positions, 0, cache))
                    seed = ArgMax(LastRow(prefill, vocab));

                using (ITensor step = model.Forward([seed], [tokens.Length], 0, cache))
                    rows[i] = LastRow(step, vocab);

                if (disposeBetweenPrompts)
                    (cache as IDisposable)?.Dispose();
                else if (cache is IDisposable d)
                    held.Add(d);
            }

            return rows;
        }
        finally
        {
            foreach (IDisposable d in held)
                d.Dispose();
            model.Dispose();
        }
    }

    private static string ResolveSpvDir()
        => Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));

    private static unsafe float[] LastRow(ITensor output, int vocab)
    {
        long total = output.ElementCount;
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)output.DataPointer + (total - vocab), vocab).CopyTo(row);
        return row;
    }

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

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }

        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }
}
