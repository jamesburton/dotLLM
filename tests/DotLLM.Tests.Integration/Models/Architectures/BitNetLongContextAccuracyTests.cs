using DotLLM.Core.Evaluation;
using DotLLM.Engine.Evaluation;
using DotLLM.Models.Architectures;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Regression gate for issue #247 — BitNet b1.58 (I2_S) quality degraded with sequence length
/// (PPL 384 vs Llama-3.2-1B's 16.3 on wikitext-2, teacher-forced ratio growing from ~1.1x at
/// ctx=16 to ~6.5x at ctx=256). Root cause: <c>GgufModelConfigExtractor</c> defaulted every
/// non-Qwen/Phi/GptOss architecture to <c>RoPEType.Norm</c> (interleaved pairing) on the
/// assumption that the GGUF writer applies llama.cpp's Llama-style Q/K permute. Microsoft's own
/// <c>bitnet.cpp</c> converter (<c>BitnetModel.modify_tensors</c>) does not call that permute, so
/// BitNet's Q/K weights stay in HuggingFace (<c>rotate_half</c>) order and need
/// <c>RoPEType.NeoX</c> — same as the already-correct safetensors path
/// (<see cref="DotLLM.Models.SafeTensors.HfConfigExtractor"/> hardcodes NeoX unconditionally).
/// </summary>
/// <remarks>
/// <para><b>Why this test would have caught the bug and the existing ones didn't.</b>
/// <see cref="BitNetReferenceParityTests"/> checks greedy argmax on 5 short prompts — argmax
/// survives any monotonic distortion of the logits, and short prompts never reach the context
/// depth where a wrong RoPE pairing's effect (near-identity rotation angles at small positions)
/// becomes visible. <see cref="BitNetAccuracyTests.Cpu_Perplexity_OnFixedPassage_IsSane"/> scores
/// a real NLL (not argmax) but on a ~120-token passage — below where the degradation ramps up.
/// This test scores a real NLL at a context length of at least 512, and additionally checks a
/// long-context/short-context <i>ratio</i>, which is what actually distinguishes "genuinely a
/// harder/longer window" (both go up together) from "quality degrades disproportionately with
/// position" (the bug's specific signature).</para>
/// <para>Measured before/after the fix on this exact passage (teacher-forced, CPU, single-pass):
/// pre-fix ctx=512 landed far outside the bound below (mirroring the issue's reproduced
/// teacher-forced ctx=512 figure of 176.69 on wikitext-2); post-fix it is back in the low
/// double digits and no longer grows faster than the short-context figure.</para>
/// </remarks>
public sealed class BitNetLongContextAccuracyTests
{
    private readonly ITestOutputHelper _output;
    public BitNetLongContextAccuracyTests(ITestOutputHelper output) => _output = output;

    private static string? ModelPath =>
        Environment.GetEnvironmentVariable("DOTLLM_BITNET_GGUF");

    // ~700 words of varied, non-repetitive factual prose spanning several unrelated topics
    // (deliberately encyclopedic/wikitext-like, not a single easy narrative) — real wikitext-2
    // is what actually exposed the bug; a single short, topically-narrow passage did not.
    private const string Passage =
        "The city of Constantinople served for over a thousand years as the capital of the "
        + "Byzantine Empire, controlling trade routes between Europe and Asia across the "
        + "Bosphorus strait. Its massive Theodosian land walls, first completed in the fifth "
        + "century, withstood dozens of sieges before the Ottoman conquest of fourteen fifty "
        + "three finally breached them with early gunpowder artillery. "
        + "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy "
        + "absorbed by chlorophyll molecules inside chloroplasts. The light dependent reactions "
        + "occur in the thylakoid membrane, generating ATP and NADPH, while the Calvin cycle in "
        + "the stroma fixes carbon into three carbon sugars using the enzyme rubisco, one of the "
        + "most abundant proteins on the planet. "
        + "The Bretton Woods conference of nineteen forty four established a system of fixed "
        + "exchange rates pegged to the United States dollar, which was itself convertible into "
        + "gold at thirty five dollars per ounce. This arrangement underpinned two decades of "
        + "post war economic expansion before the United States suspended gold convertibility in "
        + "nineteen seventy one, an event later called the Nixon shock, after which major "
        + "currencies floated freely against one another. "
        + "Saturn's rings are composed almost entirely of water ice particles ranging from "
        + "microscopic dust grains to boulders several meters across, held in a thin disk by "
        + "orbital resonances with nearby moons such as Mimas and Pandora. Despite spanning "
        + "hundreds of thousands of kilometers in diameter, the main rings are remarkably thin, "
        + "in places less than ten meters thick, a proportion often compared to a sheet of paper "
        + "spread across a football field. "
        + "The printing press developed by Johannes Gutenberg around fourteen forty combined "
        + "movable metal type, oil based ink, and a modified wine press to make books dramatically "
        + "cheaper to produce than hand copied manuscripts. Within fifty years presses had spread "
        + "to hundreds of European cities, and the resulting flood of printed material is widely "
        + "credited with accelerating the Reformation and the broader spread of literacy across "
        + "the continent. "
        + "Coral reefs form when colonies of tiny marine invertebrates called polyps secrete "
        + "calcium carbonate skeletons over many generations, eventually building structures large "
        + "enough to be seen from orbit. Rising ocean temperatures cause the polyps to expel the "
        + "symbiotic algae living in their tissues, a process known as bleaching, which leaves the "
        + "coral skeleton visible through its now transparent flesh and, if prolonged, can kill "
        + "the colony outright. "
        + "The Antikythera mechanism, recovered from a shipwreck off a Greek island in nineteen "
        + "hundred one, is a geared bronze device dating to roughly one hundred fifty years before "
        + "the common era that modeled the movements of the sun, moon, and known planets and could "
        + "predict eclipses years in advance. Nothing of comparable mechanical complexity is known "
        + "to have been built again in Europe until astronomical clocks appeared more than a "
        + "thousand years later. "
        + "Vaccination works by exposing the immune system to a harmless version or fragment of a "
        + "pathogen, prompting the production of memory B cells and T cells that can recognize the "
        + "real pathogen far more quickly on subsequent exposure. Edward Jenner's observation in "
        + "seventeen ninety six that milkmaids exposed to cowpox rarely contracted smallpox led "
        + "directly to the first deliberate vaccination and, eventually, to the global eradication "
        + "of smallpox two centuries later.";

    [Fact]
    public void Cpu_Perplexity_AtLongContext_StaysWithinBoundAndDoesNotDegradeWithLength()
    {
        if (ModelPath is null || !File.Exists(ModelPath))
        {
            _output.WriteLine("SKIP: BitNet GGUF not available (set DOTLLM_BITNET_GGUF).");
            return;
        }

        using var gguf = GgufFile.Open(ModelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        int[] tokenIds = tokenizer.Encode(Passage);
        _output.WriteLine($"passage tokens: {tokenIds.Length}");
        Assert.True(tokenIds.Length >= 512,
            $"passage must tokenize to >= 512 tokens to exercise the long-context path (got {tokenIds.Length}).");

        bool returnsAllRows = BackendPerplexityModel.Probe(model, deviceId: -1);
        Assert.True(returnsAllRows, "CPU TransformerModel is expected to return logits for every row.");
        var perplexityModel = new BackendPerplexityModel(model, deviceId: -1, returnsAllRows);

        // Short context: establishes the "this corpus/model isn't just universally bad" baseline.
        var shortOptions = new PerplexityOptions(PerplexityMode.TeacherForced, ContextLength: 64, Stride: 64);
        PerplexityResult shortResult = PerplexityEvaluator.Evaluate(perplexityModel, tokenIds, shortOptions);

        // Long context: the regime the original bug corrupted (ratio to a healthy model grew from
        // ~1.1x at ctx=16 to ~6.5x at ctx=256 on wikitext-2; teacher-forced ctx=512 landed at 176.69,
        // reproduced exactly on this box before the RoPE-pairing fix below).
        var longOptions = new PerplexityOptions(PerplexityMode.TeacherForced, ContextLength: 512, Stride: 512);
        PerplexityResult longResult = PerplexityEvaluator.Evaluate(perplexityModel, tokenIds, longOptions);

        _output.WriteLine(
            $"ctx=64  PPL={shortResult.Perplexity:F4} (scored {shortResult.ScoredTokens})");
        _output.WriteLine(
            $"ctx=512 PPL={longResult.Perplexity:F4} (scored {longResult.ScoredTokens})");

        // Absolute bound: a capable 2B model, teacher-forced, at ctx=512 on ordinary prose should sit
        // in the low double digits (Microsoft reports ~9-13 on wikitext-2; this passage's shorter,
        // narrower corpus carries a wider error bar, hence the generous margin). The pre-fix bug
        // landed at 176.69 on the real wikitext-2 corpus at this same context length — nowhere near
        // this bound.
        Assert.True(longResult.Perplexity < 40.0,
            $"ctx=512 perplexity {longResult.Perplexity:F4} exceeds the sane bound (40) — "
            + "possible regression of issue #247 (RoPE pairing / long-context quality).");

        // Ratio gate: this is the actual signature of issue #247. A healthy model's teacher-forced
        // perplexity does not grow faster than context — more context is never-decreasing evidence,
        // so ctx=512 should be no worse than a modest multiple of ctx=64, not several times worse.
        double ratio = longResult.Perplexity / shortResult.Perplexity;
        _output.WriteLine($"ctx512/ctx64 ratio = {ratio:F3}");
        Assert.True(ratio < 2.0,
            $"ctx=512 perplexity ({longResult.Perplexity:F4}) is {ratio:F2}x ctx=64's "
            + $"({shortResult.Perplexity:F4}) — quality degrading with sequence length, "
            + "the exact symptom of issue #247.");
    }
}
