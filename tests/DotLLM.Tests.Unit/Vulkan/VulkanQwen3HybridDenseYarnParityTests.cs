using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1101: the Vulkan Qwen3.5/3.8 hybrid host rotates exactly like the CPU oracle with <c>--rope-scaling yarn</c>,
/// at positions past the native window: trunk prefill and decode, the MTP head, and Bonsai-2's own RoPE at positions past
/// 262,144. See <see cref="HybridYarnParity"/> for why the fixture discriminates.
/// </summary>
/// <remarks>
/// Mutant runs (recorded in the PR): not attaching the YaRN rotation (the pre-#1101 host) and attaching it with
/// <c>mscale = 1</c> both fail these tests by orders of magnitude over the tolerance.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseYarnParityTests : IDisposable
{
    // F32 fixture: measured CPU-vs-GPU gap is ~1e-4 (cached, CUDA F16 KV rows) and ~1e-7 (cacheless). 1e-3 keeps a ~9x
    // margin over that while staying far below the ~2.5e-2 YaRN on/off difference (HybridYarnParity.SignalFloor).
    private const float AbsTol = HybridYarnParity.AbsTol;
    private const float RelTol = HybridYarnParity.RelTol;

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-qwen35-yarn-{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _out;

    public VulkanQwen3HybridDenseYarnParityTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>600-token prefill (past the 512 native window), 4 decode steps and 3 MTP drafts match the CPU oracle.</summary>
    [SkippableFact]
    public void Cached_PastNativeWindow_MatchesCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = HybridYarnParity.WriteDiscriminating(_scratch);
        const int Prompt = 600, Decode = 4, Mtp = 3;

        HybridYarnParity.Result cpu;
        using (var gguf = GgufFile.Open(path))
        {
            var config = HybridYarnParity.Config(gguf, yarn: true);
            using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            cpu = HybridYarnParity.RunCached(model, kv, Prompt, Decode, Mtp);
        }

        HybridYarnParity.Result vk;
        using (var gguf = GgufFile.Open(path))
        {
            var config = HybridYarnParity.Config(gguf, yarn: true);
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
            using var kv = model.CreateKvCache(config.MaxSequenceLength);
            vk = HybridYarnParity.RunCached(model, kv, Prompt, Decode, Mtp);
        }

        float trunk = HybridYarnParity.AssertClose(cpu.Trunk, vk.Trunk, AbsTol, RelTol, "trunk");
        float mtp = HybridYarnParity.AssertClose(cpu.Mtp, vk.Mtp, AbsTol, RelTol, "mtp");
        _out.WriteLine($"max |cpu - vulkan|: trunk {trunk:E3}, mtp {mtp:E3}");
    }

    /// <summary>Bonsai-2's own RoPE with YaRN (window 1,048,576), cacheless at positions past 262,144.</summary>
    [SkippableTheory]
    [InlineData(300_000, 16)]
    [InlineData(1_048_000, 16)]
    public void Uncached_BonsaiRope_PastNativeWindow_MatchesCpu(int start, int length)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = HybridYarnParity.WriteBonsaiLike(_scratch);

        List<float[]> cpu;
        using (var gguf = GgufFile.Open(path))
        {
            var config = HybridYarnParity.Config(gguf, yarn: true);
            Assert.Equal(1_048_576, config.MaxSequenceLength);
            using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
            cpu = HybridYarnParity.RunUncached(model, start, length);
        }

        List<float[]> vk;
        using (var gguf = GgufFile.Open(path))
        {
            var config = HybridYarnParity.Config(gguf, yarn: true);
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
            vk = HybridYarnParity.RunUncached(model, start, length);
        }

        float worst = HybridYarnParity.AssertClose(cpu, vk, AbsTol, RelTol, $"positions {start}..{start + length - 1}");
        _out.WriteLine($"max |cpu - vulkan| at {start}: {worst:E3}");
    }
}
