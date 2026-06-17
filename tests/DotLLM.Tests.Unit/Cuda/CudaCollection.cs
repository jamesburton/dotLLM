using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// xUnit collection that serializes all CUDA tests. By default each test class is its own
/// collection and collections run in parallel — but the CUDA tests share a single GPU/CUDA
/// context, so running them concurrently causes intermittent batch failures. Placing every
/// CUDA test class in this collection (with parallelization disabled) forces them to run one
/// at a time and never alongside other collections.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class CudaCollection
{
    /// <summary>Collection name referenced by <c>[Collection(CudaCollection.Name)]</c> on each CUDA test class.</summary>
    public const string Name = "Cuda";
}
