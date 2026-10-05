using DotLLM.HuggingFace;
using DotLLM.Server;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.HuggingFace;

/// <summary>Issue #716: persisted model profiles (the Modelfile equivalent) and their server-side effects.</summary>
[Collection("SequentialFileIO")]   // DOTLLM_PROFILES_DIR is process-global
public sealed class ModelProfileTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-716-" + Guid.NewGuid().ToString("N"));
    private readonly string? _prev = Environment.GetEnvironmentVariable("DOTLLM_PROFILES_DIR");

    public ModelProfileTests() => Environment.SetEnvironmentVariable("DOTLLM_PROFILES_DIR", _dir);

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("DOTLLM_PROFILES_DIR", _prev);
        try { if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true); } catch { }
    }

    [Theory]
    [InlineData("coder", "coder")]
    [InlineData("coder:latest", "coder")]
    [InlineData("Coder:7b", "Coder:7b")]
    [InlineData("my-model_1.5", "my-model_1.5")]
    [InlineData("a/b", null)]
    [InlineData("a\\b", null)]
    [InlineData("..", null)]
    [InlineData("-x", null)]
    [InlineData("a:b:c", null)]
    [InlineData("", null)]
    [InlineData("con fig", null)]
    public void NormalizeName_AcceptsOnlyFileSafeNames(string input, string? expected) =>
        Assert.Equal(expected, ModelProfileStore.NormalizeName(input));

    [Fact]
    public void SaveGetListDelete_RoundTrip_AndNamesAreCaseInsensitiveWithLatest()
    {
        ModelProfileStore.Save("Coder:7b", new ModelProfile { From = "owner/repo:Q4_K_M", System = "Be terse.", Temperature = 0.2f, Stop = ["###"], KeepAlive = -1 });
        ModelProfileStore.Save("helper", new ModelProfile { From = "coder:7b" });

        var p = ModelProfileStore.TryGet("coder:7B")!;
        Assert.Equal("owner/repo:Q4_K_M", p.From);
        Assert.Equal(0.2f, p.Temperature);
        Assert.Equal(["###"], p.Stop!);
        Assert.Equal(-1, p.KeepAlive);
        Assert.Equal(["Coder:7b", "helper"], ModelProfileStore.List().Select(x => x.Name));
        Assert.NotNull(ModelProfileStore.TryGet("helper:latest"));

        Assert.True(ModelProfileStore.Delete("HELPER"));
        Assert.False(ModelProfileStore.Delete("helper"));
        Assert.Null(ModelProfileStore.TryGet("helper"));
    }

    [Fact]
    public void Save_RejectsBadNamesAndMissingFrom_AndCorruptFilesReadAsAbsent()
    {
        Assert.Throws<ArgumentException>(() => ModelProfileStore.Save("a/b", new ModelProfile { From = "x" }));
        Assert.Throws<ArgumentException>(() => ModelProfileStore.Save("ok", new ModelProfile { From = " " }));
        Directory.CreateDirectory(_dir);
        File.WriteAllText(Path.Combine(_dir, "broken.json"), "{ not json");
        File.WriteAllText(Path.Combine(_dir, "nofrom.json"), "{\"system\":\"x\"}");
        Assert.Null(ModelProfileStore.TryGet("broken"));
        Assert.Null(ModelProfileStore.TryGet("nofrom"));
        Assert.Empty(ModelProfileStore.List());
    }

    [Fact]
    public void Resolve_FollowsFromChains_OuterOverridesInner_AndCutsCycles()
    {
        ModelProfileStore.Save("base", new ModelProfile { From = "owner/repo", Temperature = 0.7f, System = "inner", Device = "vulkan" });
        ModelProfileStore.Save("outer", new ModelProfile { From = "base", Temperature = 0.1f });

        var (reference, chain) = ModelProfileStore.Resolve("outer")!.Value;
        Assert.Equal("owner/repo", reference);
        Assert.Equal(2, chain.Count);
        var merged = ModelProfileStore.Merge(chain);
        Assert.Equal(0.1f, merged.Temperature);   // outer wins
        Assert.Equal("inner", merged.System);     // inherited
        Assert.Equal("vulkan", merged.Device);
        Assert.Equal("owner/repo", merged.From);

        ModelProfileStore.Save("a", new ModelProfile { From = "b" });
        ModelProfileStore.Save("b", new ModelProfile { From = "a" });
        Assert.Null(ModelProfileStore.Resolve("a"));
        Assert.Null(ModelProfileStore.Resolve("not-a-profile"));
    }

    [Fact]
    public void ResolveModelPath_ProfileNameResolvesToItsBaseModel_AndModelIdIsTheAlias()
    {
        string gguf = Path.Combine(_dir, "base-model.gguf");
        Directory.CreateDirectory(_dir);
        File.WriteAllText(gguf, "g");
        ModelProfileStore.Save("tuned", new ModelProfile { From = gguf });

        Assert.Equal(Path.GetFullPath(gguf), ServerStartup.ResolveModelPath("tuned", null));
        Assert.Equal("tuned", ServerStartup.ModelIdFor("tuned", gguf));
        Assert.Equal("tuned", ServerStartup.ModelIdFor("tuned:latest", gguf));
        Assert.Equal("base-model", ServerStartup.ModelIdFor(gguf, gguf));   // not a profile -> the file stem
    }

    [Fact]
    public void SamplingDefaults_OverlayOnlyWhatTheProfileSets_AndNeverMutatesTheGlobalDefaults()
    {
        var global = new SamplingDefaults { Temperature = 0.5f, TopP = 0.9f, MaxTokens = 512, TopK = 40 };
        var profile = new ModelProfile { From = "x", Temperature = 0.1f, MaxTokens = 64, Stop = ["END"] };

        var eff = global.OverlayProfile(profile);

        Assert.Equal(0.1f, eff.Temperature);
        Assert.Equal(64, eff.MaxTokens);
        Assert.Equal(0.9f, eff.TopP);   // untouched: the profile did not set it
        Assert.Equal(40, eff.TopK);
        Assert.Equal(["END"], eff.StopSequences!);
        Assert.Equal(0.5f, global.Temperature);
        Assert.Null(global.StopSequences);
    }

    [Fact]
    public void ProfileSystemPrompt_IsPrependedOnlyWhenTheRequestHasNoSystemMessage()
    {
        var user = new ChatMessage { Role = "user", Content = "hi" };

        var withProfile = ProfileSystemPrompt.Apply("Be terse.", [user]);
        Assert.Equal(["system", "user"], withProfile.Select(m => m.Role));
        Assert.Equal("Be terse.", withProfile[0].Content);

        var ownSystem = new ChatMessage { Role = "system", Content = "mine" };
        var kept = ProfileSystemPrompt.Apply("Be terse.", [ownSystem, user]);
        Assert.Equal("mine", kept[0].Content);
        Assert.Equal(2, kept.Length);

        Assert.Same(ProfileSystemPrompt.Apply(null, [user])[0], user);
        Assert.Single(ProfileSystemPrompt.Apply("  ", [user]));
    }
}
