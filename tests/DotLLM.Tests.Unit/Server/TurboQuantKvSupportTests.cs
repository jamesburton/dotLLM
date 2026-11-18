using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Server;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1888 (#1887 step 1): a TurboQuant KV request runs TurboQuant for real or is refused with an actionable reason - never served
/// by another cache. The only real path in step 1 is CPU serve --no-paged on the plain transformer.
/// </summary>
public class TurboQuantKvSupportTests
{
    private static ModelConfig Dense(int headDim = 64, Architecture arch = Architecture.Llama) => new()
    {
        Architecture = arch,
        VocabSize = 1000, HiddenSize = 1024, IntermediateSize = 2048,
        NumLayers = 4, NumAttentionHeads = 16, NumKvHeads = 4, HeadDim = headDim,
        MaxSequenceLength = 2048,
    };

    private static readonly KvCacheConfig Tq4 = new(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, TurboQuantBits: 4);

    /// <summary>The one real step-1 context: CPU serve, not paged, plain transformer, no draft/MTP.</summary>
    private static TurboQuantContext Real(ModelConfig? config = null) => new(
        KvCommand.Serve, KvBackendKind.Cpu, Paged: false, config ?? Dense(), HostIsPlainCpuTransformer: true,
        HasDraftModel: false, MtpWillAttach: false);

    public static TheoryData<string, KvCacheConfig, TurboQuantContext, string> RefusalCases() => new()
    {
        { "mixed K tq / V q8_0", new(KvCacheDType.TurboQuant, KvCacheDType.Q8_0), Real(), "step 6" },
        { "V-only tq", new(KvCacheDType.F32, KvCacheDType.TurboQuant), Real(), "step 6" },
        // Mixed is unsupported on every backend: say so first, so moving to the CPU does not hit a second refusal.
        { "mixed on vulkan", new(KvCacheDType.TurboQuant, KvCacheDType.Q4_0), Real() with { Backend = KvBackendKind.Vulkan }, "step 6" },
        { "vulkan", Tq4, Real() with { Backend = KvBackendKind.Vulkan }, "vulkan" },
        { "cuda", Tq4, Real() with { Backend = KvBackendKind.Cuda }, "cuda" },
        { "hip", Tq4, Real() with { Backend = KvBackendKind.Hip }, "hip" },
        { "run", Tq4, Real() with { Command = KvCommand.Run }, "dotllm serve --device cpu --no-paged" },
        { "chat", Tq4, Real() with { Command = KvCommand.Chat }, "dotllm serve --device cpu --no-paged" },
        { "paged", Tq4, Real() with { Paged = true }, "--no-paged" },
        { "head dim 80", Tq4, Real(Dense(headDim: 80)), "head dim 80" },
        { "head dim 512", Tq4, Real(Dense(headDim: 512)), "head dim 512" },
        { "draft", Tq4, Real() with { HasDraftModel = true }, "draft" },
        { "mtp", Tq4, Real() with { MtpWillAttach = true }, "--no-mtp" },
        { "hybrid host", Tq4, Real(Dense(arch: Architecture.Qwen3HybridDense)) with { HostIsPlainCpuTransformer = false }, "Qwen3HybridDense" },
    };

    [Theory]
    [MemberData(nameof(RefusalCases))]
    public void Refuses(string name, KvCacheConfig kv, TurboQuantContext ctx, string expected)
    {
        string? reason = TurboQuantKvSupport.RefusalReason(kv, ctx);
        Assert.NotNull(reason);
        Assert.Contains(expected, reason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains($"#{TurboQuantKvSupport.EpicIssue}", reason);
        Assert.False(string.IsNullOrEmpty(name));
    }

    /// <summary>Every refusal names the ticket that closes its gap (each carries the checklist to remove this guidance), not only the epic.</summary>
    public static TheoryData<string, KvCacheConfig, TurboQuantContext, int> GapTicketCases() => new()
    {
        { "mixed", new(KvCacheDType.TurboQuant, KvCacheDType.Q8_0), Real(), 1895 },
        { "vulkan", Tq4, Real() with { Backend = KvBackendKind.Vulkan }, 1892 },
        { "cuda", Tq4, Real() with { Backend = KvBackendKind.Cuda }, 1892 },
        { "hip", Tq4, Real() with { Backend = KvBackendKind.Hip }, 1896 },
        { "run", Tq4, Real() with { Command = KvCommand.Run }, 1892 },
        { "bench", Tq4, Real() with { Command = KvCommand.Bench }, 1892 },
        { "paged", Tq4, Real() with { Paged = true }, 1893 },
        { "head dim", Tq4, Real(Dense(headDim: 80)), 1897 },
        { "draft", Tq4, Real() with { HasDraftModel = true }, 1901 },
        { "mtp", Tq4, Real() with { MtpWillAttach = true }, 1901 },
        { "host", Tq4, Real() with { HostIsPlainCpuTransformer = false }, 1900 },
    };

    [Theory]
    [MemberData(nameof(GapTicketCases))]
    public void EachRefusal_CitesItsGapTicket(string name, KvCacheConfig kv, TurboQuantContext ctx, int ticket)
    {
        string? reason = TurboQuantKvSupport.RefusalReason(kv, ctx);
        Assert.Contains($"#{ticket}", reason);
        Assert.Contains($"#{TurboQuantKvSupport.EpicIssue}", reason);
        Assert.False(string.IsNullOrEmpty(name));
    }

    [Fact]
    public void MixedFormatRefusal_CitesStep6()
        => Assert.Contains("#1895", TurboQuantKvSupport.MixedFormatRefusal("tq3", "tq4"));

    [Theory]
    [InlineData("tq4")]
    [InlineData("tq4q")]
    [InlineData("turboquant")]
    [InlineData("tq")]
    [InlineData("tq2")]
    public void AllowsRealCpuPath(string flag)
    {
        var dtype = KvCacheConfig.ParseDType(flag, out int bits, out bool qjl);
        var kv = new KvCacheConfig(dtype, dtype, TurboQuantBits: bits > 0 ? bits : 4, TurboQuantUseQjl: qjl);
        Assert.Null(TurboQuantKvSupport.RefusalReason(kv, Real()));
    }

    [Theory]
    [InlineData(KvCacheDType.F32, KvCacheDType.F32)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0, KvCacheDType.Q8_0)]
    public void NotTurboQuant_IsNeverRefused(KvCacheDType k, KvCacheDType v)
    {
        var kv = new KvCacheConfig(k, v);
        foreach (var backend in Enum.GetValues<KvBackendKind>())
        foreach (var cmd in Enum.GetValues<KvCommand>())
            Assert.Null(TurboQuantKvSupport.RefusalReason(kv, Real() with
            {
                Backend = backend, Command = cmd, Paged = true, HasDraftModel = true, MtpWillAttach = true,
                HostIsPlainCpuTransformer = false,
            }));
    }

    [Fact]
    public void PreLoadPass_SkipsHostAndBackendRules()
    {
        // --device auto before load: backend and host unknown. Nothing else wrong -> not refused yet (the post-load pass decides).
        var preLoad = Real() with { Backend = KvBackendKind.Unknown, HostIsPlainCpuTransformer = null };
        Assert.Null(TurboQuantKvSupport.RefusalReason(Tq4, preLoad));
        // ...but the rules that do not depend on the loaded model still refuse before loading.
        Assert.Contains("--no-paged", TurboQuantKvSupport.RefusalReason(Tq4, preLoad with { Paged = true }));
    }

    [Fact]
    public void BackendOf_PlainCpuTransformer_IsCpu()
    {
        using var tiny = new PrefixCacheFailedRequestTests.TinyQwen();
        Assert.Equal(KvBackendKind.Cpu, TurboQuantKvSupport.BackendOf(tiny.Model));
    }
}
