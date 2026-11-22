using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1919 T5: the <see cref="GenerationSessionBackendSuite"/> on the CUDA backend for all three families. Written here, run only on a
/// CUDA box (the T5500 / framework, with the owner's go): the Strix Halo dev box has no NVIDIA GPU, so every test skips.
/// The Qwen3MoeHybrid rollback tests are the ones that need <c>CudaQwen3MoeHybridTransformerModel.RestoreRecurrentState</c>.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class GenerationSessionCudaTests : GenerationSessionBackendSuite
{
    internal override GenerationSessionFixture Open(string family, int? headDim = null)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        return GenerationSessionFixture.Create(family, headDim, SessionBackend.Cuda);
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }
}
