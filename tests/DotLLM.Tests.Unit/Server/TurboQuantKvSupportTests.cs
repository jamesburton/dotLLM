using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Server;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1888 (#1887 step 1): a TurboQuant KV request runs TurboQuant for real or is refused with an actionable reason - never served
/// by another cache. Since #1892 the MSE stage runs in place on every host that has the full twin set.
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

    /// <summary>The real in-place context: CPU serve, not paged, a host that runs it in place, no draft/MTP.</summary>
    private static TurboQuantContext Real(ModelConfig? config = null) => new(
        KvCommand.Serve, KvBackendKind.Cpu, Paged: false, config ?? Dense(), HostSupportsInPlace: true,
        HasDraftModel: false, MtpWillAttach: false, CacheWindow: 0, HostIsPlainCpuTransformer: true);

    private static readonly KvCacheConfig Tq4Qjl = new(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, TurboQuantBits: 4, TurboQuantUseQjl: true);

    public static TheoryData<string, KvCacheConfig, TurboQuantContext, string> RefusalCases() => new()
    {
        { "mixed K tq / V q8_0", new(KvCacheDType.TurboQuant, KvCacheDType.Q8_0), Real(), "step 6" },
        { "V-only tq", new(KvCacheDType.F32, KvCacheDType.TurboQuant), Real(), "step 6" },
        // Mixed is unsupported on every backend: say so first, so moving to the CPU does not hit a second refusal.
        { "mixed on vulkan", new(KvCacheDType.TurboQuant, KvCacheDType.Q4_0), Real() with { Backend = KvBackendKind.Vulkan }, "step 6" },
        { "hip", Tq4, Real() with { Backend = KvBackendKind.Hip }, "hip" },
        { "qjl on vulkan", Tq4Qjl, Real() with { Backend = KvBackendKind.Vulkan, HostSupportsInPlace = null }, "QJL" },
        { "qjl on cuda", Tq4Qjl, Real() with { Backend = KvBackendKind.Cuda, HostSupportsInPlace = null }, "QJL" },
        { "cache window", Tq4 with { MixedPrecisionWindowSize = 64 }, Real() with { CacheWindow = 64 }, "--cache-window 64" },
        { "paged", Tq4, Real() with { Paged = true }, "--no-paged" },
        { "head dim 80", Tq4, Real(Dense(headDim: 80)), "head dim 80" },
        { "head dim 512", Tq4, Real(Dense(headDim: 512)), "head dim 512" },
        { "distinct V head dim", Tq4, Real(Dense() with { ValueHeadDim = 32 }), "V head dim" },
        { "draft", Tq4, Real() with { HasDraftModel = true }, "draft" },
        { "mtp", Tq4, Real() with { MtpWillAttach = true }, "--no-mtp" },
        { "hybrid host (CPU)", Tq4, Real(Dense(arch: Architecture.Qwen3HybridDense)) with { HostSupportsInPlace = false, HostIsPlainCpuTransformer = false }, "Qwen3HybridDense" },
        { "host without the full twin set (vulkan)", Tq4, Real() with { Backend = KvBackendKind.Vulkan, HostSupportsInPlace = false }, "no TurboQuant twin" },
        { "host without the full twin set (cuda)", Tq4, Real() with { Backend = KvBackendKind.Cuda, HostSupportsInPlace = false }, "no TurboQuant twin" },
        { "qjl on a non-plain cpu host", Tq4Qjl, Real() with { HostSupportsInPlace = false, HostIsPlainCpuTransformer = false }, "no TurboQuant twin" },
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
        { "hip", Tq4, Real() with { Backend = KvBackendKind.Hip }, 1896 },
        { "qjl on vulkan", Tq4Qjl, Real() with { Backend = KvBackendKind.Vulkan, HostSupportsInPlace = null }, 1894 },
        { "qjl on cuda", Tq4Qjl, Real() with { Backend = KvBackendKind.Cuda, HostSupportsInPlace = null }, 1894 },
        { "paged", Tq4, Real() with { Paged = true }, 1893 },
        { "head dim", Tq4, Real(Dense(headDim: 80)), 1897 },
        { "distinct V head dim", Tq4, Real(Dense() with { ValueHeadDim = 32 }), 1898 },
        { "draft", Tq4, Real() with { HasDraftModel = true }, 1901 },
        { "mtp", Tq4, Real() with { MtpWillAttach = true }, 1901 },
        { "host (cpu)", Tq4, Real() with { HostSupportsInPlace = false }, 1900 },
        { "host (vulkan)", Tq4, Real() with { Backend = KvBackendKind.Vulkan, HostSupportsInPlace = false }, 1900 },
        { "host (cuda)", Tq4, Real() with { Backend = KvBackendKind.Cuda, HostSupportsInPlace = false }, 1900 },
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

    /// <summary>
    /// #1892: the rules that stopped TurboQuant on CUDA, Vulkan and run / chat / bench are GONE. A host that runs it in place is accepted
    /// wherever it is, for every command, and the refusals that remain never send the user to the CPU path or to the scratch cache.
    /// </summary>
    [Theory]
    [InlineData(KvBackendKind.Cpu, KvCommand.Serve)]
    [InlineData(KvBackendKind.Cpu, KvCommand.Run)]
    [InlineData(KvBackendKind.Cpu, KvCommand.Chat)]
    [InlineData(KvBackendKind.Cpu, KvCommand.Bench)]
    [InlineData(KvBackendKind.Cuda, KvCommand.Serve)]
    [InlineData(KvBackendKind.Cuda, KvCommand.Run)]
    [InlineData(KvBackendKind.Cuda, KvCommand.Chat)]
    [InlineData(KvBackendKind.Cuda, KvCommand.Bench)]
    [InlineData(KvBackendKind.Vulkan, KvCommand.Serve)]
    [InlineData(KvBackendKind.Vulkan, KvCommand.Run)]
    [InlineData(KvBackendKind.Vulkan, KvCommand.Chat)]
    [InlineData(KvBackendKind.Vulkan, KvCommand.Bench)]
    public void InPlaceHost_IsAccepted_OnEveryBackendAndCommand(KvBackendKind backend, KvCommand command)
        => Assert.Null(TurboQuantKvSupport.RefusalReason(Tq4, Real() with { Backend = backend, Command = command }));

    [Fact]
    public void QjlOnThePlainCpuHost_StaysOnTheScratchPath()
    {
        // tq*q is not in place until step 5: the plain CPU transformer serves it on the scratch cache, so its capability being false is no refusal.
        Assert.Null(TurboQuantKvSupport.RefusalReason(Tq4Qjl, Real() with { HostSupportsInPlace = false, HostIsPlainCpuTransformer = true }));
    }

    /// <summary>
    /// A cache window with TurboQuant is a refusal by design: no ticket to close and no "use q8_0 instead" suffix, and it fires before the
    /// geometry / host rules so the CPU window guard in <c>ExecuteQuantizedCore</c> is never reached.
    /// </summary>
    [Fact]
    public void CacheWindow_WithTq_Refuses()
    {
        string? reason = TurboQuantKvSupport.RefusalReason(Tq4 with { MixedPrecisionWindowSize = 128 }, Real() with { CacheWindow = 128 });
        Assert.NotNull(reason);
        Assert.Contains("--cache-window 128", reason);
        Assert.DoesNotContain("Closing this gap", reason);
        Assert.DoesNotContain("for now", reason);
        // Ahead of the rules that need the model: a pre-load pass already refuses it.
        Assert.NotNull(TurboQuantKvSupport.RefusalReason(Tq4, Real() with { CacheWindow = 128, Config = null, HostSupportsInPlace = null }));
    }

    /// <summary>
    /// A Vulkan host whose twin set is incomplete (<c>SupportsInPlaceTurboQuant</c> false) is refused after the load with the host gap, on
    /// <c>--device auto</c> as much as on an explicit device: never served by another cache.
    /// </summary>
    [Fact]
    public void Auto_OnAHostWithoutTheFullTwinSet_Refuses()
    {
        var model = new DotLLM.Vulkan.TestDoubles.FakeVulkanHost(Dense(), supportsInPlace: false);
        Assert.Equal(KvBackendKind.Vulkan, TurboQuantKvSupport.BackendOf(model));
        string? reason = TurboQuantKv.PostLoadRefusal(Tq4, formatMismatch: null, KvCommand.Serve, paged: false, model.Config, model,
            hasDraftModel: false, mtpWillAttach: false);
        Assert.NotNull(reason);
        Assert.Contains("#1900", reason);
        Assert.Contains("vulkan", reason, StringComparison.OrdinalIgnoreCase);
        // The same host with the full set is accepted: the refusal is the capability, nothing else.
        var full = new DotLLM.Vulkan.TestDoubles.FakeVulkanHost(Dense(), supportsInPlace: true);
        Assert.Null(TurboQuantKv.PostLoadRefusal(Tq4, null, KvCommand.Serve, paged: false, full.Config, full, false, false));
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
        // tq*q has no in-place cache: the plain CPU host runs it on the scratch cache, which its capability (false for QJL) must not refuse.
        Assert.Null(TurboQuantKvSupport.RefusalReason(kv, Real() with { HostSupportsInPlace = !qjl }));
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
                HostSupportsInPlace = false, HostIsPlainCpuTransformer = false, CacheWindow = 128,
            }));
    }

    [Fact]
    public void PreLoadPass_SkipsHostAndBackendRules()
    {
        // --device auto before load: backend and host unknown. Nothing else wrong -> not refused yet (the post-load pass decides).
        var preLoad = Real() with { Backend = KvBackendKind.Unknown, HostSupportsInPlace = null, HostIsPlainCpuTransformer = null };
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

