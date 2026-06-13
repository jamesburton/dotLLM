using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// bf16-vs-inner Q8_0 perplexity over a longer passage on Llama-3.2-1B, so the headline accuracy cost
/// averages out the sample noise of the 32-token A/B used in <see cref="Llama32OuterProductBf16AccuracyTests"/>.
/// This is a ~several-hundred-token passage (larger than the short sample, NOT a standard benchmark corpus
/// such as wikitext-2); the A/B/C compares the three reductions on the identical tokens, so only the
/// relative deltas matter. Skips unless AVX512-BF16 is present (net11 on Zen4/Zen5/Strix).
/// </summary>
[Collection("Llama32Instruct")]
public class Llama32Bf16PerplexityCorpusTests
{
    private readonly Llama32InstructFixture _fixture;

    public Llama32Bf16PerplexityCorpusTests(Llama32InstructFixture fixture)
    {
        _fixture = fixture;
    }

    // Neutral encyclopedic prose, a few hundred tokens — enough that the bf16-vs-inner delta reflects the
    // averaged rounding cost rather than the luck of a single short sentence. Self-authored (no license
    // entanglement); content is irrelevant beyond giving the model varied, in-distribution text to score.
    private const string Corpus =
        "The history of computing stretches back long before the electronic age. Early civilizations " +
        "devised mechanical aids for counting and calculation, from the abacus to geared astronomical " +
        "instruments. In the nineteenth century, Charles Babbage designed the Analytical Engine, a " +
        "general-purpose mechanical computer that anticipated many features of modern machines, and Ada " +
        "Lovelace described how such an engine might be programmed to follow a sequence of operations. " +
        "The theoretical foundations were laid in the twentieth century by Alan Turing, whose abstract " +
        "model of computation clarified what it means for a problem to be solvable by a machine. " +
        "During the Second World War, electromechanical and then electronic devices were built to break " +
        "ciphers and to compute artillery tables, and these efforts accelerated the development of " +
        "programmable digital computers. After the war, the invention of the transistor and later the " +
        "integrated circuit shrank room-sized machines into devices that could sit on a desk. " +
        "Software evolved in parallel with hardware. Programming languages rose in level of abstraction, " +
        "from machine code and assembly to compiled and interpreted languages that let people express " +
        "algorithms in terms closer to human reasoning. Operating systems managed the growing complexity " +
        "of shared resources, scheduling many tasks across limited memory and processing time. " +
        "Networks connected isolated machines, first within institutions and then across the globe, " +
        "giving rise to electronic mail, the world wide web, and the vast interconnected systems that " +
        "underpin commerce, science, and everyday communication today. " +
        "The most recent chapter concerns machine learning, in which programs improve their behavior by " +
        "finding patterns in large collections of data rather than following rules written by hand. " +
        "Models built from many simple units, trained on text, images, and other signals, can now " +
        "translate languages, recognize objects, and generate fluent prose. These systems demand " +
        "enormous computation, which has renewed interest in efficient numerical kernels, specialized " +
        "hardware, and careful management of memory bandwidth, the very concerns that have shaped " +
        "computing from its earliest mechanical beginnings.";

    [SkippableFact]
    public void Bf16Perplexity_OverLongerCorpus_ReportsCost()
        => OuterProductBf16Accuracy.AssertPerplexityOverCorpus(_fixture.FilePath, Corpus);
}
