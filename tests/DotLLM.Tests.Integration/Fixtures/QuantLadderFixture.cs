using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>One quantization fixture from the <c>--pure</c> ladder.</summary>
/// <param name="Type">Block type the fixture is expected to contain.</param>
/// <param name="FilePath">Absolute path to the GGUF.</param>
/// <param name="ContextLength">Scoring context, fixed by the fixture's base model.</param>
public sealed record QuantLadderEntry(QuantizationType Type, string FilePath, int ContextLength);

/// <summary>
/// Indexes the <c>--pure</c> quantization ladder for the cross-backend gate (#256).
/// </summary>
/// <remarks>
/// <para>
/// Fixtures live in <c>~/.dotllm/quant-ladder/</c> and are never committed — a single GGUF is
/// 0.1–50 GB and git keeps a committed blob forever, even after deletion. Override the root with
/// <see cref="DirEnvVar"/>. Provenance and regeneration: <c>.docs/corpora/QUANT_FIXTURES.md</c>.
/// </para>
/// <para>
/// <b>Two base models, deliberately.</b> 256-superblock types require <c>ne[0] % 256 == 0</c>;
/// SmolLM2-135M has hidden 576, which fails <c>GGML_ASSERT</c> on <c>token_embd</c>, so K-quants
/// and the IQ family use Llama-3.2-1B instead.
/// </para>
/// <para>
/// <b>The IQ2_S entry is not a typo.</b> llama.cpp file-types do not map 1:1 to ggml block types:
/// ftype <c>IQ2_S</c> emits IQ2_XS blocks, and ftype <c>IQ2_M</c> is what emits IQ2_S. Coverage is
/// verified from observed block types, never filenames — see <c>QuantFixtureBlockTypeTests</c>.
/// </para>
/// </remarks>
public sealed class QuantLadderFixture
{
    /// <summary>Environment variable overriding the ladder root directory.</summary>
    public const string DirEnvVar = "DOTLLM_QUANT_LADDER";

    /// <summary>
    /// Minimum byte size for a fixture to be classified as <see cref="Available"/> rather than
    /// <see cref="Missing"/>. Existence alone does not prove a fixture is usable: a crashed
    /// <c>llama-quantize</c> leaves a truncated stub behind, and a first sweep once reported
    /// "21/21 produced" where 12 were 1.7 MB stubs.
    /// </summary>
    public const long MinFixtureBytes = 1_000_000;

    private const int Ctx135M = 512;
    private const int Ctx1B = 128;

    /// <summary>Every fixture the gate expects, with its path relative to the ladder root.</summary>
    public static IReadOnlyList<(QuantizationType Type, string RelativePath, int ContextLength)> Expected { get; } =
    [
        (QuantizationType.F16,     @"SmolLM2-135M/SmolLM2-135M-pure-F16.gguf",        Ctx135M),
        (QuantizationType.BF16,    @"SmolLM2-135M/SmolLM2-135M-pure-BF16.gguf",       Ctx135M),
        (QuantizationType.Q8_0,    @"SmolLM2-135M/SmolLM2-135M-pure-Q8_0.gguf",       Ctx135M),
        (QuantizationType.Q4_0,    @"SmolLM2-135M/SmolLM2-135M-pure-Q4_0.gguf",       Ctx135M),
        (QuantizationType.Q4_1,    @"SmolLM2-135M/SmolLM2-135M-pure-Q4_1.gguf",       Ctx135M),
        (QuantizationType.Q5_0,    @"SmolLM2-135M/SmolLM2-135M-pure-Q5_0.gguf",       Ctx135M),
        (QuantizationType.Q5_1,    @"SmolLM2-135M/SmolLM2-135M-pure-Q5_1.gguf",       Ctx135M),
        (QuantizationType.IQ4_NL,  @"SmolLM2-135M/SmolLM2-135M-pure-IQ4_NL.gguf",     Ctx135M),
        (QuantizationType.MXFP4,   @"SmolLM2-135M/SmolLM2-135M-pure-MXFP4_MOE.gguf",  Ctx135M),
        (QuantizationType.Q2_K,    @"Llama-3.2-1B/Llama-3.2-1B-pure-Q2_K.gguf",       Ctx1B),
        (QuantizationType.Q3_K,    @"Llama-3.2-1B/Llama-3.2-1B-pure-Q3_K_S.gguf",     Ctx1B),
        (QuantizationType.Q4_K,    @"Llama-3.2-1B/Llama-3.2-1B-pure-Q4_K_S.gguf",     Ctx1B),
        (QuantizationType.Q5_K,    @"Llama-3.2-1B/Llama-3.2-1B-pure-Q5_K_S.gguf",     Ctx1B),
        (QuantizationType.Q6_K,    @"Llama-3.2-1B/Llama-3.2-1B-pure-Q6_K.gguf",       Ctx1B),
        (QuantizationType.IQ4_XS,  @"Llama-3.2-1B/Llama-3.2-1B-pure-IQ4_XS.gguf",     Ctx1B),
        (QuantizationType.IQ3_S,   @"Llama-3.2-1B/Llama-3.2-1B-pure-IQ3_S.gguf",      Ctx1B),
        (QuantizationType.IQ3_XXS, @"Llama-3.2-1B/Llama-3.2-1B-pure-IQ3_XXS.gguf",    Ctx1B),
        (QuantizationType.IQ2_S,   @"Llama-3.2-1B/Llama-3.2-1B-pure-IQ2_M.gguf",      Ctx1B),
        (QuantizationType.IQ2_XS,  @"Llama-3.2-1B/Llama-3.2-1B-pure-IQ2_XS.gguf",     Ctx1B),
        (QuantizationType.IQ2_XXS, @"Llama-3.2-1B/Llama-3.2-1B-pure-IQ2_XXS.gguf",    Ctx1B),
        (QuantizationType.IQ1_S,   @"Llama-3.2-1B/Llama-3.2-1B-pure-IQ1_S.gguf",      Ctx1B),
    ];

    /// <summary>Fixtures found on this machine.</summary>
    public IReadOnlyList<QuantLadderEntry> Available { get; }

    /// <summary>Expected types with no fixture present here.</summary>
    public IReadOnlyList<QuantizationType> Missing { get; }

    /// <summary>Root directory the ladder was resolved from.</summary>
    public string RootDirectory { get; }

    /// <summary>Indexes the ladder, classifying each expected fixture as available or missing.</summary>
    public QuantLadderFixture()
    {
        RootDirectory = Environment.GetEnvironmentVariable(DirEnvVar)
            ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "quant-ladder");

        var available = new List<QuantLadderEntry>();
        var missing = new List<QuantizationType>();

        foreach (var (type, relativePath, ctx) in Expected)
        {
            string full = Path.GetFullPath(Path.Combine(RootDirectory, relativePath));

            // Size is checked, not just existence — see MinFixtureBytes for why.
            if (File.Exists(full) && new FileInfo(full).Length > MinFixtureBytes)
                available.Add(new QuantLadderEntry(type, full, ctx));
            else
                missing.Add(type);
        }

        Available = available;
        Missing = missing;
    }
}

/// <summary>Shares one ladder index across the gate's test classes.</summary>
[CollectionDefinition("QuantLadder")]
public class QuantLadderCollection : ICollectionFixture<QuantLadderFixture>;
