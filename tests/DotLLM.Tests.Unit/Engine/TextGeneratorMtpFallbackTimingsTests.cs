using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using static DotLLM.Tests.Unit.Engine.RecurrentPrefixMtpHarness;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// A request whose MTP rounds measure slower than plain decode finishes on the plain loop (#900). Its timings must still report
/// the drafts it made before falling back. The streaming plain tail dropped them, so the server's streamed
/// <c>speculative_draft_tokens</c> read 0 for every such request (seen while measuring #1088 on the T5500 replay).
/// </summary>
public sealed class TextGeneratorMtpFallbackTimingsTests : IDisposable
{
    private readonly string _scratch = Directory.CreateTempSubdirectory("dotllm-mtp-fallback-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RequestThatFallsBackToPlain_StillReportsItsDrafts(bool streaming)
    {
        using var gguf = GgufFile.Open(SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "fallback.gguf"), withMtp: true, contextLength: 512));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var generator = new TextGenerator(model, new TokenIdsTokenizer(config.VocabSize),
            (cfg, size) => new SimpleKvCache(model.AttentionLayerCount, cfg.NumKvHeads, cfg.HeadDim, size), mtpAdaptive: true);
        Skip.If(generator.MtpGate is null, "DOTLLM_MTP_ADAPTIVE=0 disables the adaptive gate");
        // Plain decode "measured" at 1 us/token: the gate still explores MTP once (it has no MTP sample), and the round
        // controller then falls back to plain after its minimum rounds, because no MTP round can beat that.
        generator.MtpGate!.RecordSegment(false, 1000, 1.0);
        var options = new InferenceOptions { Temperature = 0f, RepetitionPenalty = 1f, MaxTokens = 48 };
        string prompt = Csv(MtpLongContextHarness.Prompt(40, config.VocabSize));

        InferenceTimings timings;
        int generated;
        if (streaming)
        {
            InferenceTimings? last = null;
            generated = 0;
            await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options))
            {
                generated++;
                last = token.Timings ?? last;
            }
            timings = last!.Value;
        }
        else
        {
            var response = generator.Generate(prompt, options);
            timings = response.Timings;
            generated = response.GeneratedTokenCount;
        }

        Assert.Equal(48, generated);   // it ran to the limit, so the tail after the fallback was decoded plain
        Assert.True(timings.SpeculativeDraftTokens > 0, "the drafts made before the fallback were not reported");
        Assert.True(timings.SpeculativeAcceptedTokens > 0);
        Assert.True(timings.SpeculativeAcceptedTokens < generated, "every token was accepted from a draft: the request never fell back to plain");
    }
}
