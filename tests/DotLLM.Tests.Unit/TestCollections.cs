using Xunit;

namespace DotLLM.Tests.Unit;

/// <summary>
/// Marker for CUDA direct-kernel parity tests that each construct their own
/// <c>CudaContext</c> / <c>CudaKernels</c> / <c>CudaStream</c>. Concurrent context
/// creation on the same device can race during PTX module load, surfacing as
/// intermittent "PTX files not found" skips. Apply
/// <c>[Collection("CudaKernels")]</c> to any test class that uses
/// <c>CudaKernelTestHarness</c> to force sequential execution between such classes.
/// </summary>
[CollectionDefinition("CudaKernels", DisableParallelization = true)]
public class CudaKernelsCollection
{
}
