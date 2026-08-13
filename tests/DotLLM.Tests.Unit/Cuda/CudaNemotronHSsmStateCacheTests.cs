using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
public class CudaNemotronHSsmStateCacheTests
{
    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    [SkippableFact]
    public void Construct_ZeroInitializesBothBuffers()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        var ssm = new MambaSsmConfig(DConv: 4, DInner: 16, DState: 8, NGroup: 2, NHead: 2);
        using var cache = new CudaNemotronHSsmStateCache(ssm, numSsmLayers: 3);

        Assert.Equal(3, cache.NumSsmLayers);
        Assert.Equal((4 - 1) * ssm.ConvDim, cache.ConvStateElements);
        Assert.Equal(16 * 8, cache.SsmStateElements);

        for (int layer = 0; layer < 3; layer++)
        {
            float[] conv = Download(cache.GetConvStatePtr(layer), cache.ConvStateElements);
            float[] state = Download(cache.GetSsmStatePtr(layer), cache.SsmStateElements);
            Assert.All(conv, v => Assert.Equal(0f, v));
            Assert.All(state, v => Assert.Equal(0f, v));
        }
    }

    [SkippableFact]
    public void Reset_ZeroesNonZeroState()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        var ssm = new MambaSsmConfig(DConv: 4, DInner: 8, DState: 4, NGroup: 1, NHead: 2);
        using var cache = new CudaNemotronHSsmStateCache(ssm, numSsmLayers: 1);

        float[] ones = new float[cache.SsmStateElements];
        Array.Fill(ones, 1.0f);
        unsafe
        {
            fixed (float* p = ones)
                CudaDriverApi.cuMemcpyHtoD_v2(cache.GetSsmStatePtr(0), (nint)p,
                    (nuint)(ones.Length * sizeof(float))).ThrowOnError();
        }

        float[] twos = new float[cache.ConvStateElements];
        Array.Fill(twos, 2.0f);
        unsafe
        {
            fixed (float* p = twos)
                CudaDriverApi.cuMemcpyHtoD_v2(cache.GetConvStatePtr(0), (nint)p,
                    (nuint)(twos.Length * sizeof(float))).ThrowOnError();
        }

        cache.Reset();
        float[] afterReset = Download(cache.GetSsmStatePtr(0), cache.SsmStateElements);
        Assert.All(afterReset, v => Assert.Equal(0f, v));

        float[] convAfterReset = Download(cache.GetConvStatePtr(0), cache.ConvStateElements);
        Assert.All(convAfterReset, v => Assert.Equal(0f, v));
    }

    private static unsafe float[] Download(nint devicePtr, int count)
    {
        float[] result = new float[count];
        fixed (float* p = result)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(count * sizeof(float))).ThrowOnError();
        return result;
    }
}
