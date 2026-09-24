using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration;

/// <summary>
/// xUnit collection that serializes every integration test class touching a GPU (CUDA or
/// Vulkan). By default xUnit makes each test class its own collection and runs collections in
/// parallel; on a single-GPU box that lets several GPU classes contend for one device, and a
/// VRAM-starved kernel can stall indefinitely (#483: <c>IbSsmMamba3CudaParityTests</c> hung for
/// 28+ minutes at 100% GPU with all 12 GB of an RTX 3060 in use, but passes in ~3 s alone).
/// </summary>
/// <remarks>
/// <para>
/// With <c>DisableParallelization = true</c>, xUnit (v2) runs this collection only after the
/// parallel batch has finished, and one non-parallel collection at a time, so GPU classes never
/// overlap each other or anything else. CPU-only classes stay in the parallel batch.
/// </para>
/// <para>
/// A GPU class that also needs a shared model fixture cannot join this collection (a class
/// belongs to exactly one collection), so each such fixture gets its own GPU-only collection
/// definition, also with <c>DisableParallelization = true</c> — see
/// <see cref="SmallModelGpuCollection"/> and <see cref="VulkanResidencyReportCollection"/>.
/// Non-parallel collections are mutually exclusive, so they serialize against this one too.
/// </para>
/// <para>
/// <c>GpuCollectionGuardTests</c> fails the build's test run if a GPU test class is left in
/// a parallel collection, so new tests cannot silently reintroduce the contention.
/// </para>
/// </remarks>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GpuCollection
{
    /// <summary>Collection name referenced by <c>[Collection(GpuCollection.Name)]</c>.</summary>
    public const string Name = "GPU";
}

/// <summary>
/// GPU-only twin of the <c>SmallModel</c> collection: shares the same
/// <see cref="SmallModelFixture"/> (a cached download path, so a second instance is free) but
/// runs non-parallel, like <see cref="GpuCollection"/>. CPU-only <c>SmallModel</c> classes
/// stay in the parallel <c>SmallModel</c> collection.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class SmallModelGpuCollection : ICollectionFixture<SmallModelFixture>
{
    /// <summary>Collection name referenced by <c>[Collection(SmallModelGpuCollection.Name)]</c>.</summary>
    public const string Name = "SmallModelGpu";
}

/// <summary>
/// Serializes the Vulkan classes that read the mutable static
/// <c>VulkanWeights.LastResidencyReport</c> after a load (a concurrent load would overwrite
/// the report and can produce a false pass). Previously an implicit (undefined) collection,
/// which xUnit ran in parallel with every other GPU class; now non-parallel, so it is also
/// mutually exclusive with <see cref="GpuCollection"/>.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class VulkanResidencyReportCollection
{
    /// <summary>Collection name referenced by <c>[Collection(VulkanResidencyReportCollection.Name)]</c>.</summary>
    public const string Name = "VulkanResidencyReport";
}
