using DotLLM.Tests.Unit.Engine;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1919 T5: the <see cref="GenerationSessionBackendSuite"/> on the Vulkan backend for all three families (attention-only Llama,
/// GDN-hybrid dense, GDN-hybrid MoE) over their synthetic GGUFs. Logits within 5e-4 of a fresh-prefill oracle on the same device.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class GenerationSessionVulkanTests : GenerationSessionBackendSuite
{
    internal override GenerationSessionFixture Open(string family, int? headDim = null)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
        return GenerationSessionFixture.Create(family, headDim, SessionBackend.Vulkan);
    }

    /// <summary>
    /// Disposal returns the device to its post-first-session footprint: a second identical open/prefill/generate/dispose cycle
    /// holds exactly the bytes the first left (the pooled spare checkpoint), so no KV buffer or checkpoint leaks per session.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(GenerationSessionFixture.Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task Dispose_DoesNotLeakDeviceMemory(string family)
    {
        using var fx = Open(family);
        var gen = fx.NewGenerator();
        async Task<long> Cycle()
        {
            var s = gen.CreateSession(GenerationSessionFixture.Greedy(), GenerationSessionFixture.Tag());
            await s.PrefillAsync(fx.Prompt);
            await s.GenerateUntilBoundaryAsync(new GenerationSessionFixture.StopAfter(4));
            await s.DisposeAsync();
            return VulkanModelLoader.SharedDevice.TotalLiveBytes();
        }
        await Cycle();                     // warm: scratch, pipelines, the pooled spare checkpoint
        long after1 = await Cycle();
        long after2 = await Cycle();
        Assert.Equal(after1, after2);
    }
}
