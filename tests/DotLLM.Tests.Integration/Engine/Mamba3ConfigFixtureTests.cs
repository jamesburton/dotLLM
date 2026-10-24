using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Integration-level sanity check: the <c>ib_ssm_config.json</c> fixture
/// checked in under <c>Fixtures/Mamba3/</c> parses through
/// <see cref="Mamba3ConfigExtractor"/> end-to-end and yields the expected
/// Mamba-3 configuration for <c>ib-ssm/mamba3-370M-10BT</c> (commit
/// <c>02943831ad63d36783f41fa872f08cc8631538ee</c>, 2026-04-15).
/// </summary>
/// <remarks>
/// The embedded-JSON variants in <c>DotLLM.Tests.Unit</c> own the
/// parsing-logic test surface; this test exists purely to catch fixture
/// corruption, missing copy-to-output, or accidental field edits in the
/// checked-in file.
/// </remarks>
public class Mamba3ConfigFixtureTests
{
    private static string FixturePath() =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "Mamba3", "ib_ssm_config.json");

    [Fact]
    public void IbSsmConfigFixture_ParsesToMamba3Config()
    {
        string path = Path.GetFullPath(FixturePath());
        Assert.True(File.Exists(path), $"Mamba-3 config fixture not found at {path}");

        string json = File.ReadAllText(path);
        var config = Mamba3ConfigExtractor.Extract(json);

        Assert.Equal(Architecture.Mamba3, config.Architecture);
        Assert.NotNull(config.Mamba3Config);
        Assert.Equal(48, config.NumLayers);
        Assert.Equal(1024, config.HiddenSize);
        Assert.Equal(32000, config.VocabSize);
        Assert.Equal(128, config.Mamba3Config!.StateSize);
        Assert.Equal(4480, config.Mamba3Config.InputProjectionDim);
    }
}
