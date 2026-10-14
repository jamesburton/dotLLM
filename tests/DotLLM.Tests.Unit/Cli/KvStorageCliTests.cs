using DotLLM.Cli.Commands;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Configuration;
using Spectre.Console.Cli;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// Issue #967: KV-storage selection for the CUDA hybrid-dense model. All CPU-only: precedence is a pure
/// function, flag parsing goes through Spectre's real parser, and the warn/reject matrix is data.
/// </summary>
public sealed class KvStorageCliTests
{
    private const KvCacheDType F = KvCacheDType.F32;
    private const KvCacheDType Q8 = KvCacheDType.Q8_0;
    private const KvCacheDType Q4 = KvCacheDType.Q4_0;

    // ── precedence: explicit flag > DOTLLM_HYBRID_KV > f16, per side ──

    [Theory]
    [InlineData(null, null, null, F, F)]          // default
    [InlineData(null, null, "", F, F)]
    [InlineData(null, null, "q8_0", Q8, Q8)]      // env only
    [InlineData(null, null, "k8v4", Q8, Q4)]
    [InlineData(null, null, "k4v8", Q4, Q8)]
    [InlineData(null, null, "F16", F, F)]
    [InlineData("q4_0", "q4_0", "q8_0", Q4, Q4)]  // flag beats env on both sides
    [InlineData("q8_0", null, "q4_0", Q8, Q4)]    // per side: key from flag, value from env
    [InlineData(null, "q8_0", "q4_0", Q4, Q8)]
    [InlineData("f16", "f16", "q4_0", F, F)]      // explicit f16 beats a quantizing env
    [InlineData("q8_0", "q4_0", null, Q8, Q4)]    // flag only (k8v4)
    public void Resolve_FlagBeatsEnvBeatsDefault(string? k, string? v, string? env, KvCacheDType wantK, KvCacheDType wantV)
    {
        var (key, value) = HybridKvStorage.Resolve(HybridKvStorage.ParseFlag(k), HybridKvStorage.ParseFlag(v), env);
        Assert.Equal(wantK, key);
        Assert.Equal(wantV, value);
    }

    [Fact]
    public void Resolve_BadEnv_Throws_AndBadFlag_Throws()
    {
        Assert.Throws<InvalidOperationException>(() => HybridKvStorage.Resolve(null, null, "q5_0"));
        Assert.Throws<ArgumentException>(() => HybridKvStorage.ParseFlag("q5_0"));
    }

    [Fact]
    public void ServerOption_F32DefaultIsUnspecified_F16IsExplicit()
    {
        Assert.Null(HybridKvStorage.FromServerOption("f32"));
        Assert.Null(HybridKvStorage.FromServerOption(null));
        Assert.Equal(F, HybridKvStorage.FromServerOption("f16"));
        Assert.Equal(Q8, HybridKvStorage.FromServerOption("q8_0"));
    }

    [Fact]
    public void Label_And_Bytes_MatchTheDocumentedTable()
    {
        Assert.Equal("f16", HybridKvStorage.Label(F, F));
        Assert.Equal("q8_0", HybridKvStorage.Label(Q8, Q8));
        Assert.Equal("k8v4", HybridKvStorage.Label(Q8, Q4));
        Assert.Equal("k4v8", HybridKvStorage.Label(Q4, Q8));

        // Bonsai-2: 16 attention slots, 4 KV heads x 256 dim, 64k tokens.
        const int slots = 16, elems = 4 * 256, tokens = 65536;
        const double gib = 1024.0 * 1024 * 1024;
        Assert.Equal(4.0, HybridKvStorage.TotalBytes(slots, tokens, elems, F, F) / gib, 3);
        Assert.Equal(2.125, HybridKvStorage.TotalBytes(slots, tokens, elems, Q8, Q8) / gib, 3);
        Assert.Equal(1.125, HybridKvStorage.TotalBytes(slots, tokens, elems, Q4, Q4) / gib, 3);
        Assert.Equal(1.625, HybridKvStorage.TotalBytes(slots, tokens, elems, Q8, Q4) / gib, 3);
    }

    // ── validation ──

    [Theory]
    [InlineData(Q8, Q4, 1024, null)]
    [InlineData(F, F, 1000, null)]                       // unquantized has no row-width constraint
    [InlineData(Q8, F, 1000, "multiple of 32")]
    [InlineData(F, Q4, 33, "multiple of 32")]
    [InlineData(KvCacheDType.TurboQuant, F, 1024, "not supported")]
    [InlineData(F, KvCacheDType.TurboQuant, 1024, "not supported")]
    public void Validate_RejectsWrongCombinations(KvCacheDType k, KvCacheDType v, int elems, string? errorFragment)
    {
        string? error = HybridKvStorage.Validate(k, v, elems);
        if (errorFragment is null) Assert.Null(error);
        else Assert.Contains(errorFragment, error);
    }

    // ── request building: synonyms and conflicts ──

    [Fact]
    public void TryBuild_NoFlags_IsUnspecified()
    {
        Assert.True(KvStorageCli.TryBuild(null, null, null, null, 0, out var r, out var e));
        Assert.Null(e);
        Assert.False(r.IsSpecified);
    }

    [Fact]
    public void TryBuild_KvQuantAndCacheTypeAreSynonyms()
    {
        Assert.True(KvStorageCli.TryBuild("q8_0", null, null, "q4_0", 0, out var r, out _));
        Assert.Equal(Q8, r.Key);
        Assert.Equal(Q4, r.Value);
        Assert.Equal("--cache-type-k", r.KeyFlag);
        Assert.Equal("--kv-quant-v", r.ValueFlag);

        Assert.True(KvStorageCli.TryBuild("q8_0", null, "q8", null, 0, out var same, out _));
        Assert.Equal(Q8, same.Key);
    }

    [Fact]
    public void TryBuild_ConflictingSynonymsAndUnknownTypes_AreErrors()
    {
        Assert.False(KvStorageCli.TryBuild("q8_0", null, "q4_0", null, 0, out _, out var e1));
        Assert.Contains("--kv-quant-k", e1);
        Assert.False(KvStorageCli.TryBuild(null, null, "q5_1", null, 0, out _, out var e2));
        Assert.Contains("q5_1", e2);
    }

    // ── warning matrix: never silently ignore ──

    [Theory]
    [InlineData(KvFlagTarget.CudaHybridDense, true, 0)]       // honoured
    [InlineData(KvFlagTarget.CudaHybridDense, false, 0)]      // bench still honours it
    [InlineData(KvFlagTarget.CudaPlain, true, 0)]
    [InlineData(KvFlagTarget.Cpu, true, 0)]
    [InlineData(KvFlagTarget.CudaPlain, false, 2)]            // bench: SimpleKvCache, flags ignored
    [InlineData(KvFlagTarget.Cpu, false, 2)]
    [InlineData(KvFlagTarget.Vulkan, true, 2)]
    [InlineData(KvFlagTarget.OwnCacheUnsupported, true, 2)]
    public void Warnings_OnePerIgnoredFlag(KvFlagTarget target, bool engineCacheHonoured, int expected)
    {
        KvStorageCli.TryBuild(null, null, "q8_0", "q4_0", 0, out var req, out _);
        var warnings = KvStorageCli.Warnings(req, target, "Qwen3HybridDense", "cuda", engineCacheHonoured);
        Assert.Equal(expected, warnings.Count);
        foreach (var w in warnings)
        {
            Assert.Contains("--kv-quant-", w);          // names the flag
            Assert.Contains("Qwen3HybridDense", w);     // names the model
            Assert.Contains("cuda", w);                 // names the backend
        }
    }

    [Fact]
    public void Warnings_OnlyTheGivenSideIsReported()
    {
        KvStorageCli.TryBuild(null, null, "q8_0", null, 0, out var req, out _);
        var w = Assert.Single(KvStorageCli.Warnings(req, KvFlagTarget.Vulkan, "Gemma3", "vulkan", true));
        Assert.Contains("--kv-quant-k", w);
        Assert.DoesNotContain("--kv-quant-v", w);
    }

    [Fact]
    public void Warnings_CacheWindowIsIgnoredByHybridDense()
    {
        KvStorageCli.TryBuild(null, null, "q8_0", null, 128, out var req, out _);
        var w = Assert.Single(KvStorageCli.Warnings(req, KvFlagTarget.CudaHybridDense, "Qwen3HybridDense", "cuda", true));
        Assert.Contains("--cache-window 128", w);
    }

    [Fact]
    public void Warnings_NoFlags_NoWarnings()
    {
        Assert.Empty(KvStorageCli.Warnings(KvStorageRequest.None, KvFlagTarget.Vulkan, "m", "vulkan", true));
    }

    // ── option parsing through Spectre's real parser ──

    private sealed class Capture<T> : Command<T> where T : CommandSettings
    {
        public static T? Last;
        public override int Execute(CommandContext context, T settings) { Last = settings; return 0; }
    }

    private static T Parse<T>(params string[] args) where T : CommandSettings
    {
        Capture<T>.Last = null;
        var app = new CommandApp();
        app.Configure(c => c.AddCommand<Capture<T>>("x"));
        int rc = app.Run(new[] { "x" }.Concat(args).ToArray());
        Assert.Equal(0, rc);
        return Capture<T>.Last!;
    }

    [Fact]
    public void Run_ParsesKvQuantFlags_AndCacheTypeStaysUnsetByDefault()
    {
        var s = Parse<RunCommand.Settings>("model.gguf", "--kv-quant-k", "q8_0", "--kv-quant-v", "q4_0");
        Assert.Equal("q8_0", s.KvQuantK);
        Assert.Equal("q4_0", s.KvQuantV);
        Assert.Null(s.CacheTypeK);

        var d = Parse<RunCommand.Settings>("model.gguf");
        Assert.Null(d.KvQuantK);
        Assert.Null(d.CacheTypeV);
    }

    [Fact]
    public void Bench_ParsesKvQuantFlags()
    {
        var s = Parse<BenchCommand.Settings>("model.gguf", "--kv-quant-k", "q4_0");
        Assert.Equal("q4_0", s.KvQuantK);
        Assert.Null(s.KvQuantV);
    }

    [Fact]
    public void Serve_ParsesKvQuantFlags_AndCacheTypeFlags()
    {
        var s = Parse<ServeCommand.Settings>("model.gguf", "--kv-quant-v", "q8_0", "--cache-type-k", "q4_0");
        Assert.Equal("q8_0", s.KvQuantV);
        Assert.Equal("q4_0", s.CacheTypeK);
    }
}
