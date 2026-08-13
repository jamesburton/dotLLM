using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
public class CudaNemotronHKvCacheTests
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
    public unsafe void Update_ThenGetKeysRef_RoundTripsWrittenRows()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        const int numKvHeads = 2, headDim = 4, maxSeqLen = 16;
        int kvStride = numKvHeads * headDim;
        using var cache = new CudaNemotronHKvCache(attentionLayerCount: 1, numKvHeads, headDim, maxSeqLen, deviceId: 0);

        // Prefill: write 3 rows at positions [0,1,2).
        float[] kHost = new float[3 * kvStride];
        for (int i = 0; i < kHost.Length; i++) kHost[i] = i + 1;
        nint dK = 0, dV = 0;
        try
        {
            long bytes = (long)kHost.Length * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)bytes).ThrowOnError();
            fixed (float* p = kHost)
            {
                CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)bytes).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)bytes).ThrowOnError();
            }

            var kRef = new TensorRef(3, kvStride, DType.Float32, 0, dK);
            var vRef = new TensorRef(3, kvStride, DType.Float32, 0, dV);
            cache.Update(kRef, vRef, new[] { 0, 1, 2 }, layerIndex: 0);

            Assert.Equal(3, cache.CurrentLength);
            TensorRef stored = cache.GetKeysRef(0);
            Assert.Equal(3, stored.Dim0);
            Assert.Equal(kvStride, stored.Dim1);

            float[] readBack = new float[3 * kvStride];
            fixed (float* p = readBack)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, stored.DataPointer, (nuint)bytes).ThrowOnError();
            Assert.Equal(kHost, readBack);
        }
        finally
        {
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
        }
    }

    [SkippableFact]
    public void Update_NonContiguousPositions_Throws()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        using var cache = new CudaNemotronHKvCache(attentionLayerCount: 1, numKvHeads: 1, headDim: 4, maxSeqLen: 8, deviceId: 0);
        var kRef = new TensorRef(2, 4, DType.Float32, 0, (nint)1);
        var vRef = new TensorRef(2, 4, DType.Float32, 0, (nint)1);
        Assert.Throws<NotSupportedException>(() => cache.Update(kRef, vRef, new[] { 0, 2 }, layerIndex: 0));
    }

    [SkippableFact]
    public void Rollback_ReducesCurrentLength()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);

        using var cache = new CudaNemotronHKvCache(attentionLayerCount: 1, numKvHeads: 1, headDim: 4, maxSeqLen: 8, deviceId: 0);
        // Directly exercise Rollback's bound check without a real Update (0-length cache).
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Rollback(1));
        cache.Rollback(0);
        Assert.Equal(0, cache.CurrentLength);
    }
}
