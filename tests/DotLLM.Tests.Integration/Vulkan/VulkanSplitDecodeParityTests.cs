using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// End-to-end integration check for the split-KV (Flash-Decoding) decode path
/// wired into <see cref="VulkanTransformerModel"/>. Decodes a real model at a
/// context long enough to split (<c>seqKv &gt; 256</c>) with the split path
/// ENABLED vs forced OFF (<c>DOTLLM_VULKAN_DISABLE_SPLIT_DECODE=1</c>) and
/// asserts the two agree — same greedy tokens and near-identical logits.
/// </summary>
/// <remarks>
/// The kernel itself is validated against the CPU oracle in the unit tests
/// (<c>VulkanSplitKvAttentionKernelTests</c>). This test covers what those
/// cannot: that the kernel is actually <i>routed</i> at decode, that the
/// per-(head,split) scratch is reused correctly across the model's layers, and
/// that the inter-pass / inter-layer barriers are right — a scratch-corruption
/// or barrier bug would show up as token divergence from the per-token kernel.
///
/// Self-skips when the parity GGUF is absent (env override
/// <c>DOTLLM_SPLIT_PARITY_GGUF</c> or the conventional shared-cache path).
/// </remarks>
[Trait("Category", "GPU")]
public sealed class VulkanSplitDecodeParityTests
{
    private const int Context = 384;     // > 256 so decode splits (SmolLM 9 heads -> S = 2)
    private const int DecodeSteps = 24;
    // Vulkan-vs-Vulkan: only the attention reduction order differs, so drift is
    // small, but it accumulates through ~28 layers + lm_head on a 3B model whose
    // raw logits reach ~±20. The exact-token assertion is the load-bearing check;
    // this guards against gross corruption / NaN (which would be many units off).
    private const float LogitsAbsTol = 0.5f;

    private readonly ITestOutputHelper _output;
    public VulkanSplitDecodeParityTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void SplitDecode_MatchesPerTokenKernel_LongContext()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available.");

        string? modelPath = ResolveModelPath();
        Skip.If(modelPath is null, "Parity model GGUF not found (set DOTLLM_SPLIT_PARITY_GGUF).");

        // Sanity: this shape must actually split, or the test proves nothing.
        int numHeads;
        using (var probe = GgufFile.Open(modelPath!))
            numHeads = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(probe.Metadata).NumAttentionHeads;
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(Context, numHeads),
            $"Context {Context} too short to split for numHeads={numHeads}.");

        var (tokensOn, logitsOn) = RunDecode(modelPath!, disableSplit: false);
        var (tokensOff, logitsOff) = RunDecode(modelPath!, disableSplit: true);

        _output.WriteLine($"split-on  tokens: {string.Join(",", tokensOn)}");
        _output.WriteLine($"split-off tokens: {string.Join(",", tokensOff)}");

        // Greedy decode must follow the identical path under both kernels.
        Assert.Equal(tokensOff, tokensOn);

        // And the final-step logits must agree within reduction-order noise.
        float maxAbs = 0;
        for (int i = 0; i < logitsOn.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(logitsOn[i] - logitsOff[i]));
        _output.WriteLine($"final-step logits L_inf = {maxAbs:G6}");
        Assert.True(maxAbs <= LogitsAbsTol,
            $"Split vs per-token decode logits diverged: L_inf={maxAbs:G6} > {LogitsAbsTol}");
    }

    private (int[] tokens, float[] lastLogits) RunDecode(string modelPath, bool disableSplit)
    {
        // IsSplitDecodeDisabled() is read at model construction, so set the env
        // var before LoadFromGguf and clear it afterwards.
        string? prev = Environment.GetEnvironmentVariable(VulkanTransformerModel.DisableSplitDecodeEnvVar);
        Environment.SetEnvironmentVariable(VulkanTransformerModel.DisableSplitDecodeEnvVar, disableSplit ? "1" : null);
        try
        {
            string spvDir = ResolveSpvDir();
            using var gguf = GgufFile.Open(modelPath);
            var config = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(gguf.Metadata);
            var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);

            int[] baseTok = tokenizer.Encode("The history of science is a long and winding road that ");
            Assert.NotEmpty(baseTok);
            int[] prompt = new int[Context];
            int[] positions = new int[Context];
            for (int i = 0; i < Context; i++) { prompt[i] = baseTok[i % baseTok.Length]; positions[i] = i; }

            int maxSeq = Context + DecodeSteps + 8;
            using var cache = model.CreateKvCache(maxSeq);

            // Chunked prefill keeps each dispatch under the gfx1151 watchdog.
            const int PrefillChunk = 256;
            int nextToken = 0;
            for (int off = 0; off < Context; off += PrefillChunk)
            {
                int len = Math.Min(PrefillChunk, Context - off);
                using var logits = model.Forward(prompt.AsSpan(off, len), positions.AsSpan(off, len), -1, cache);
                if (off + len >= Context) nextToken = Argmax(logits);
            }

            var tokens = new int[DecodeSteps];
            float[] lastLogits = Array.Empty<float>();
            int pos = Context;
            for (int i = 0; i < DecodeSteps; i++)
            {
                int[] s = { nextToken };
                int[] p = { pos };
                using var l = model.Forward(s, p, -1, cache);
                nextToken = Argmax(l);
                tokens[i] = nextToken;
                if (i == DecodeSteps - 1) lastLogits = ToArray(l);
                pos++;
            }
            return (tokens, lastLogits);
        }
        finally
        {
            Environment.SetEnvironmentVariable(VulkanTransformerModel.DisableSplitDecodeEnvVar, prev);
        }
    }

    private static unsafe int Argmax(ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        int idx = 0; float best = span[0];
        for (int i = 1; i < n; i++) if (span[i] > best) { best = span[i]; idx = i; }
        return idx;
    }

    private static unsafe float[] ToArray(ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        return new ReadOnlySpan<float>((void*)logits.DataPointer, n).ToArray();
    }

    private static string? ResolveModelPath()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_SPLIT_PARITY_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string conventional = "C:/Development/gguf-cache/Llama-3.2-3B-Instruct-IQ4_XS.gguf";
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
