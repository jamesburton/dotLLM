using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// xUnit collection that serializes CUDA tests. By default each test class is its own collection
/// and collections run in parallel — but the CUDA tests share a single GPU and CUDA context, so
/// running them concurrently causes intermittent batch failures. Placing a CUDA test class in
/// this collection (with parallelization disabled) forces it to run one at a time and never
/// alongside another collection.
/// </summary>
/// <remarks>
/// <para>
/// <b>Every GPU-traited class under <c>Cuda/</c> must be serialized, but not necessarily by this
/// collection.</b> xUnit v2 runs the <c>DisableParallelization = true</c> collections one at a
/// time, after the parallel batch, so they are mutually exclusive with each other as well: the
/// nine direct-kernel parity classes in the separate <c>CudaKernels</c> collection
/// (<c>TestCollections.cs</c>) are equally safe. <c>CudaTestCollectionContractTests</c> enforces
/// the rule in that form — "in some non-parallel collection" — and fails on any GPU-traited class
/// that is left parallel.
/// </para>
/// <para>
/// This was aspirational until #503: only 43 of ~113 classes carried the attribute, which is what
/// produced #502 (a sibling's <c>Dispose</c> un-pinning a process-wide static mid-test, read as a
/// numeric regression) and #484 (a device-wide <c>cuMemGetInfo</c> assertion racing a sibling's
/// allocations, read as a 30 MB leak). The genuinely CPU-only classes that happen to live in the
/// same directory — <c>CpuPrefillLastTokenLogitTest</c>,
/// <c>CpuLlamaCppLogitsParitySidecarTests</c>, <c>PtxIsaVersionTests</c>, the
/// <c>PQ2_0*LayoutEmulationTests</c>, <c>CudaKernelsPtxValidationTests</c>,
/// <c>CudaQ3KKernelSurfaceTests</c>, <c>CudaPerLayerSlidingWindowTests</c>,
/// <c>HybridKvCacheTests</c>, <c>HybridKvLengthBookkeepingTests</c> and
/// <c>CudaQwen3HybridForwardStateCapacityGranularityTests</c> — stay parallel on purpose.
/// </para>
/// </remarks>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class CudaCollection
{
    /// <summary>Collection name referenced by <c>[Collection(CudaCollection.Name)]</c> on each CUDA test class.</summary>
    public const string Name = "Cuda";
}
