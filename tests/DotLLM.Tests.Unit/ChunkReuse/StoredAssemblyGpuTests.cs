using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// GPU half of the backend tag tests (Phase I Task 1, L4): store-backed assembly on the Vulkan backend with the synthetic F32 model.
/// CPU and Vulkan entries never mix: a store filled through one backend gives no hit to the other. This shows key separation on the
/// real device (0 rejects: the header check is not reached) plus a positive control into a device cache; the header check in both
/// directions is covered on the CPU (StoredAssembly_EntryBackendDiffers_IsRejectedAndReencoded and
/// StoredAssembly_OtherBackend_FindsCpuTaggedEntryUnderItsOwnKey_IsRejectedAndReencoded).
/// Kept small (three short chunks, one tiny model, no recompute) because the box has had device-lost resets on long submits.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class StoredAssemblyTestsGpu : IDisposable
{
    private const string Tokenizer = "tok-tiny";
    private const string ModelId = "tiny-qwen3-default-seed";
    private static readonly int[] TokA = [3, 17, 5, 29, 11, 40];
    private static readonly int[] TokB = [8, 22, 1, 35];
    private static readonly int[] TokC = [12, 7, 44, 2, 19];
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-storedasm-gpu-" + Guid.NewGuid().ToString("N"));

    public StoredAssemblyTestsGpu() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static IReadOnlyList<Chunk> Context3 =>
        [new("a.cs", 1, 1, (int[])TokA.Clone()), new("b.cs", 1, 1, (int[])TokB.Clone()), new("c.cs", 1, 1, (int[])TokC.Clone())];

    [SkippableFact]
    public void StoredAssembly_Vulkan_CpuEntryNeverHits()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        using VulkanDevice device = VulkanDevice.Create();
        using IModel vkModel = VulkanTransformerModel.LoadFromGguf(device, tiny.Gguf, tiny.Config, spvDir);
        using var vkBackend = new VulkanChunkBackend(device, spvDir, tiny.Config, ChunkModelFeatures.FromGguf(tiny.Gguf));
        var cpu = StoredContextAssembler.ForModel(tiny.Config, ModelId, "F32", Tokenizer);
        var vk = StoredContextAssembler.ForModel(tiny.Config, ModelId, "F32", Tokenizer, vkBackend);
        Assert.Equal("cpu", cpu.Backend);
        Assert.Equal("vulkan", vk.Backend);
        int tokens = TokA.Length + TokB.Length + TokC.Length;

        foreach (bool cpuFirst in new[] { true, false })
        {
            string root = Path.Combine(_scratch, cpuFirst ? "cpu-first" : "vulkan-first");
            (StoredContextAssembler first, IModel firstModel) = cpuFirst ? (cpu, (IModel)tiny.Model) : (vk, vkModel);
            (StoredContextAssembler second, IModel secondModel) = cpuFirst ? (vk, vkModel) : (cpu, (IModel)tiny.Model);

            Assert.Equal(new AssemblyStats(0, 3, tokens, 0), first.Assemble(Context3, firstModel, new FileChunkStore(root)).Stats);
            StoredAssembly cross = second.Assemble(Context3, secondModel, new FileChunkStore(root));
            Assert.Equal(new AssemblyStats(0, 3, tokens, 0), cross.Stats);
            Assert.Equal(0, cross.Rejected);
            Assert.All(cross.Entries, e => Assert.Equal(second.Backend, e.Backend));

            // Both sets of entries are on disk with their own header tag, and each backend now hits its own.
            var store = new FileChunkStore(root);
            Assert.Equal(6, store.Keys().Count());
            foreach (StoredContextAssembler a in new[] { cpu, vk })
            {
                Assert.True(store.TryGet(a.KeyOf(tiny.Config, TokB), out ChunkKvEntry e));
                Assert.Equal(a.Backend, e.Backend);
            }
            Assert.Equal(new AssemblyStats(3, 0, 0, 0), first.Assemble(Context3, firstModel, new FileChunkStore(root)).Stats);

            // Vulkan warm assembly into a device cache (positive control on the device path).
            using IKvCache destination = vkBackend.CreateCache(SyntheticDenseGguf.Context);
            StoredAssembly warmVk = vk.Assemble(Context3, vkModel, new FileChunkStore(root), destination);
            Assert.Equal(new AssemblyStats(3, 0, 0, 0), warmVk.Stats);
            Assert.Equal(tokens, destination.CurrentLength);
        }
    }
}