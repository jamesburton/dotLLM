using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #331 ask (iii): real-GGUF end-to-end validation of the split-KV
/// (Flash-Decoding) decode path on <see cref="VulkanQwen3MoeHybridTransformerModel"/>
/// — one of the two architectures #346 shipped default-ON with only
/// synthetic-weight forward coverage plus the kernel proven e2e on a *different*
/// architecture.
/// </summary>
/// <remarks>
/// <para>
/// The earlier attempt at this test was written and discarded because a
/// 35B-A3B MoE <i>prefill</i> long enough to reach the then-assumed engagement
/// threshold (ctx &gt; 256) trips the gfx1151 watchdog. Issue #331's finding
/// removes that obstacle: the split path actually engages from
/// <c>seqKv &gt;= 17</c> (see
/// <see cref="VulkanSplitDecodeParityTests.EngagementThreshold_IsSeventeen_NotTwoFiftySix"/>),
/// so a short prompt plus a few dozen decode steps exercises it fully with no
/// long prefill at all.
/// </para>
/// <para>
/// This model reports <c>RequiresPerSequenceState</c> — its GDN (linear-attention)
/// recurrent state is owned by the MODEL instance, not the KV cache — so each arm
/// gets a freshly built model. That is required anyway, because the split gate is
/// read at construction.
/// </para>
/// <para>
/// Self-skips unless the GGUF is staged (env <c>DOTLLM_SPLIT_PARITY_MOE_GGUF</c>
/// or the conventional cache path). Never triggers a download.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class VulkanSplitDecodeMoeParityTests
{
    private const int PromptLen = 8;
    private const int DecodeSteps = 64;   // final decode depth ~72; engagement at 17

    private readonly ITestOutputHelper _output;
    public VulkanSplitDecodeMoeParityTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Qwen3MoeHybrid_SplitDecode_MatchesPerTokenKernel_OnRealGguf()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available.");

        string? modelPath = ResolveModelPath();
        Skip.If(modelPath is null,
            "Qwen3MoeHybrid GGUF not found. Set DOTLLM_SPLIT_PARITY_MOE_GGUF or stage the file "
            + "(no download is triggered — the file is ~20 GB).");

        int numHeads, vocabSize;
        int[] prompt;
        using (var probe = GgufFile.Open(modelPath!))
        {
            var cfg = GgufModelConfigExtractor.Extract(probe.Metadata);
            Assert.Equal(Architecture.Qwen3MoeHybrid, cfg.Architecture);
            numHeads = cfg.NumAttentionHeads;
            vocabSize = cfg.VocabSize;
            int[] enc = GgufBpeTokenizerFactory.Load(probe.Metadata)
                .Encode("The history of science is a long and winding road that");
            prompt = enc.Length >= PromptLen ? enc[..PromptLen] : enc;
        }

        int gate = FirstSplittingSeqKv(numHeads);
        _output.WriteLine(
            $"numHeads={numHeads}; split engages at seqKv={gate}; " +
            $"S at depth {prompt.Length + DecodeSteps} = " +
            $"{VulkanSplitKvAttentionKernel.ComputeSplits(prompt.Length + DecodeSteps, numHeads)}");
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(prompt.Length + DecodeSteps, numHeads),
            "Run is too short to engage the split path — this test would prove nothing.");

        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] split OFF (fresh model)");
        var (tokensOff, logitsOff) = Run(modelPath!, disableSplit: true, prompt, vocabSize);
        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] split ON (fresh model)");
        var (tokensOn, logitsOn) = Run(modelPath!, disableSplit: false, prompt, vocabSize);
        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] complete");

        _output.WriteLine($"split-off tokens: {string.Join(",", tokensOff)}");
        _output.WriteLine($"split-on  tokens: {string.Join(",", tokensOn)}");

        // Below the engagement depth the split kernel cannot be dispatched, so the
        // arms must be bit-identical. Step s is emitted from a forward at depth
        // prompt.Length + s, so steps with depth < gate are pre-engagement.
        int preSteps = Math.Max(0, Math.Min(DecodeSteps, gate - prompt.Length));
        for (int i = 0; i < preSteps; i++)
            Assert.Equal(tokensOff[i], tokensOn[i]);
        _output.WriteLine($"pre-engagement steps checked bit-exact: {preSteps}");

        int firstDivergence = -1;
        for (int i = 0; i < DecodeSteps; i++)
            if (tokensOff[i] != tokensOn[i]) { firstDivergence = i; break; }
        _output.WriteLine(firstDivergence < 0
            ? $"token-for-token EXACT MATCH across all {DecodeSteps} greedy steps."
            : $"FIRST DIVERGENCE at step {firstDivergence} (depth {prompt.Length + firstDivergence}).");

        // Gross-corruption / barrier / scratch-reuse guard: the final-step logits
        // must agree within reduction-order noise and stay finite.
        float maxAbs = 0;
        for (int i = 0; i < logitsOff.Length; i++)
        {
            Assert.True(float.IsFinite(logitsOn[i]), $"non-finite logit at {i} with split-KV on");
            maxAbs = MathF.Max(maxAbs, MathF.Abs(logitsOn[i] - logitsOff[i]));
        }
        _output.WriteLine($"final-step logits L_inf = {maxAbs:G6}");
        Assert.True(maxAbs <= 0.5f,
            $"Split vs per-token decode logits diverged on Qwen3MoeHybrid: L_inf={maxAbs:G6}");
    }

    private static (int[] tokens, float[] lastLogits) Run(
        string modelPath, bool disableSplit, int[] prompt, int vocabSize)
    {
        string? prev = Environment.GetEnvironmentVariable(VulkanTransformerModel.DisableSplitDecodeEnvVar);
        Environment.SetEnvironmentVariable(VulkanTransformerModel.DisableSplitDecodeEnvVar, disableSplit ? "1" : null);
        try
        {
            string spvDir = ResolveSpvDir();
            using var gguf = GgufFile.Open(modelPath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
            using var cache = model.CreateKvCache(prompt.Length + DecodeSteps + 2);

            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;

            int next;
            using (ITensor l = model.Forward(prompt, positions, -1, cache))
                next = Argmax(l, vocabSize);

            var tokens = new int[DecodeSteps];
            float[] lastLogits = Array.Empty<float>();
            int pos = prompt.Length;
            for (int i = 0; i < DecodeSteps; i++)
            {
                tokens[i] = next;
                int[] one = { next };
                int[] p = { pos };
                using ITensor l = model.Forward(one, p, -1, cache);
                next = Argmax(l, vocabSize);
                if (i == DecodeSteps - 1) lastLogits = ToArray(l, vocabSize);
                pos++;
            }
            return (tokens, lastLogits);
        }
        finally
        {
            Environment.SetEnvironmentVariable(VulkanTransformerModel.DisableSplitDecodeEnvVar, prev);
        }
    }

    private static int FirstSplittingSeqKv(int numHeads)
    {
        for (int s = 1; s <= 8192; s++)
            if (VulkanSplitKvAttentionKernel.WouldSplit(s, numHeads)) return s;
        return int.MaxValue;
    }

    private static unsafe int Argmax(ITensor logits, int vocabSize)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
        int idx = 0; float best = span[0];
        for (int i = 1; i < span.Length; i++) if (span[i] > best) { best = span[i]; idx = i; }
        return idx;
    }

    private static unsafe float[] ToArray(ITensor logits, int vocabSize)
        => new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize).ToArray();

    private static string? ResolveModelPath()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_SPLIT_PARITY_MOE_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string conventional = "C:/Development/gguf-cache/Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf";
        return File.Exists(conventional) ? conventional : null;
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException("SPIR-V blobs not found. Run native/vulkan/build.ps1.");
    }
}
