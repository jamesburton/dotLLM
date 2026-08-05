using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Quantization;

/// <summary>
/// Asserts each ladder fixture actually contains the block type it is indexed under (#256).
/// </summary>
/// <remarks>
/// llama.cpp <i>file-types</i> do not map 1:1 to ggml <i>block types</i>, so a filename overstates
/// coverage: ftype <c>IQ2_S</c> emits IQ2_XS blocks (ftype <c>IQ2_M</c> is what emits IQ2_S), and a
/// stock <c>Q4_K_M</c> mixes Q4_K with Q6_K — which is why the ladder is <c>--pure</c>. Until this
/// passes, no cell may be counted as coverage for its type.
/// </remarks>
// Category=Fixtures: needs the local ~/.dotllm/quant-ladder/ fixtures but no GPU — this test
// only mmaps a GGUF and counts tensor descriptors. Do not tag it GPU: that category means
// "requires an NVIDIA GPU, 20-30 min" (README.md / CONTRIBUTING.md), and a contributor
// filtering "Category!=GPU" on a CPU box would silently lose this gate.
[Trait("Category", "Fixtures")]
[Collection("QuantLadder")]
public sealed class QuantFixtureBlockTypeTests
{
    private readonly QuantLadderFixture _ladder;
    private readonly ITestOutputHelper _output;

    public QuantFixtureBlockTypeTests(QuantLadderFixture ladder, ITestOutputHelper output)
    {
        _ladder = ladder;
        _output = output;
    }

    /// <summary>
    /// Theory data of every type the ladder is expected to cover — one case per row of
    /// <see cref="QuantLadderFixture.Expected"/>, so a new ladder entry is verified automatically
    /// without touching this class.
    /// </summary>
    public static TheoryData<QuantizationType> ExpectedTypes()
    {
        var data = new TheoryData<QuantizationType>();
        foreach (var (type, _, _) in QuantLadderFixture.Expected)
            data.Add(type);
        return data;
    }

    /// <summary>
    /// Proves the fixture indexed under <paramref name="type"/> actually contains that ggml block
    /// type on real weight matrices — not merely that a file with a matching ftype exists. This is
    /// the check that turns "we cover N quantization types" from a filename convention into a
    /// measured claim; see the class remarks for why filenames cannot be trusted.
    /// </summary>
    /// <param name="type">Block type the ladder claims this fixture exercises.</param>
    [SkippableTheory]
    [MemberData(nameof(ExpectedTypes))]
    public void Fixture_ContainsTheBlockTypeItClaims(QuantizationType type)
    {
        QuantLadderEntry? entry = _ladder.Available.FirstOrDefault(e => e.Type == type);
        Skip.If(entry is null, $"fixture for {type} not present under {_ladder.RootDirectory}");

        using GgufFile gguf = GgufFile.Open(entry!.FilePath);

        // Count, and report, every block type present. A --pure fixture still legitimately holds
        // F32 norms/biases and may hold a Q8_0 token embedding (the IQ types require an imatrix
        // for every tensor they touch, and imatrix collection produces no entry for token_embd),
        // so the assertion is that the claimed type IS present on real weight matrices — not that
        // it is the only type in the file.
        var histogram = gguf.Tensors
            .GroupBy(t => t.QuantizationType)
            .ToDictionary(g => g.Key, g => g.Count());

        foreach (var (blockType, count) in histogram.OrderByDescending(kv => kv.Value))
            _output.WriteLine($"{blockType,-10} {count,4} tensors");

        Assert.True(histogram.ContainsKey(type),
            $"{type}: fixture {Path.GetFileName(entry.FilePath)} contains no {type} tensors. " +
            $"Present: {string.Join(", ", histogram.Keys.OrderBy(k => k.ToString()))}. " +
            "Coverage must be read from observed block types, never filenames.");

        // The claimed type must carry real weight matrices, not one stray tensor.
        int claimed = histogram[type];
        Assert.True(claimed >= 4,
            $"{type}: only {claimed} tensor(s) carry this block type — too few to exercise the kernel.");
    }
}
