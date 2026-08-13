using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
public class CudaMamba3StateCacheTests
{
    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static Mamba3Config TinySisoConfig() => new()
    {
        StateSize = 8, NumHeads = 4, HeadDim = 4, Expand = 2, NumGroups = 1,
        ChunkSize = 64, IsMimo = false, MimoRank = 4, AFloor = 1e-4f,
        DtInitFloor = 1e-4f, DtMin = 1e-3f, DtMax = 0.1f, UseL2Warp = false,
        RopeFraction = 0.5f, IsOutProjNorm = false, RescalePrenormResidual = true,
        ResidualInFp32 = true,
    };

    private static Mamba3Config TinyMimoConfig() => TinySisoConfig() with { IsMimo = true, MimoRank = 3 };

    [SkippableFact]
    public void Constructor_ZeroInitializes_AllFourBuffers()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var m3 = TinySisoConfig();
        using var cache = new CudaMamba3StateCache(m3, numLayers: 2);

        Assert.Equal(2, cache.NumLayers);
        Assert.Equal(m3.NumHeads * m3.HeadDim * m3.StateSize, cache.SsmStateElementsPerLayer);
        Assert.Equal(m3.NumHeads * m3.NumRopeAngles, cache.CumAngleElementsPerLayer);
        Assert.Equal(m3.NumHeads * m3.StateSize, cache.KStateElementsPerLayer); // SISO: kRank=1
        Assert.Equal(m3.NumHeads * m3.HeadDim, cache.VStateElementsPerLayer);

        // Cover all four buffers, both layers where feasible — a zero-init check that
        // only touches ssm_state cannot catch a missing memset on a sibling buffer.
        AssertAllZero(cache.GetSsmStatePtr(0), cache.SsmStateElementsPerLayer);
        AssertAllZero(cache.GetSsmStatePtr(1), cache.SsmStateElementsPerLayer);
        AssertAllZero(cache.GetCumAnglePtr(0), cache.CumAngleElementsPerLayer);
        AssertAllZero(cache.GetCumAnglePtr(1), cache.CumAngleElementsPerLayer);
        AssertAllZero(cache.GetKStatePtr(0), cache.KStateElementsPerLayer);
        AssertAllZero(cache.GetKStatePtr(1), cache.KStateElementsPerLayer);
        AssertAllZero(cache.GetVStatePtr(0), cache.VStateElementsPerLayer);
        AssertAllZero(cache.GetVStatePtr(1), cache.VStateElementsPerLayer);
    }

    [SkippableFact]
    public void MimoConfig_KStateElementsPerLayer_IncludesRankAxis()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var m3 = TinyMimoConfig();
        using var cache = new CudaMamba3StateCache(m3, numLayers: 1);
        Assert.Equal(m3.MimoRank * m3.NumHeads * m3.StateSize, cache.KStateElementsPerLayer);
    }

    [SkippableFact]
    public void Reset_ZeroesNonZeroState()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var m3 = TinySisoConfig();
        using var cache = new CudaMamba3StateCache(m3, numLayers: 1);

        // Dirty ALL FOUR buffers before Reset() — a Reset test that only dirties/verifies
        // one buffer cannot catch a Reset() that drops a memset on a sibling buffer (this
        // exact bug shape was a real finding on the GDN-sibling state cache).
        FillWithOnes(cache.GetSsmStatePtr(0), cache.SsmStateElementsPerLayer);
        FillWithOnes(cache.GetCumAnglePtr(0), cache.CumAngleElementsPerLayer);
        FillWithOnes(cache.GetKStatePtr(0), cache.KStateElementsPerLayer);
        FillWithOnes(cache.GetVStatePtr(0), cache.VStateElementsPerLayer);

        cache.Reset();

        AssertAllZero(cache.GetSsmStatePtr(0), cache.SsmStateElementsPerLayer);
        AssertAllZero(cache.GetCumAnglePtr(0), cache.CumAngleElementsPerLayer);
        AssertAllZero(cache.GetKStatePtr(0), cache.KStateElementsPerLayer);
        AssertAllZero(cache.GetVStatePtr(0), cache.VStateElementsPerLayer);
    }

    [SkippableFact]
    public void CopyTo_DeepCopiesState_IndependentOfSource()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var m3 = TinySisoConfig();
        using var src = new CudaMamba3StateCache(m3, numLayers: 1);
        using var dst = new CudaMamba3StateCache(m3, numLayers: 1);

        // Seed all four buffers with distinct values so a CopyTo() that misses one
        // buffer is caught (mirrors the Reset-coverage lesson above; Clone/CopyTo is
        // the Task-12 speculative-decoding checkpoint/rollback primitive).
        float[] ssmVals = SeedSequential(src.SsmStateElementsPerLayer, baseValue: 1f);
        float[] cumVals = SeedSequential(src.CumAngleElementsPerLayer, baseValue: 101f);
        float[] kVals = SeedSequential(src.KStateElementsPerLayer, baseValue: 201f);
        float[] vVals = SeedSequential(src.VStateElementsPerLayer, baseValue: 301f);

        CopyHostToDevice(ssmVals, src.GetSsmStatePtr(0));
        CopyHostToDevice(cumVals, src.GetCumAnglePtr(0));
        CopyHostToDevice(kVals, src.GetKStatePtr(0));
        CopyHostToDevice(vVals, src.GetVStatePtr(0));

        src.CopyTo(dst);

        Assert.Equal(ssmVals, CopyDeviceToHost(dst.GetSsmStatePtr(0), dst.SsmStateElementsPerLayer));
        Assert.Equal(cumVals, CopyDeviceToHost(dst.GetCumAnglePtr(0), dst.CumAngleElementsPerLayer));
        Assert.Equal(kVals, CopyDeviceToHost(dst.GetKStatePtr(0), dst.KStateElementsPerLayer));
        Assert.Equal(vVals, CopyDeviceToHost(dst.GetVStatePtr(0), dst.VStateElementsPerLayer));

        // Mutate source after copy; destination must be unaffected (independent allocation).
        src.Reset();
        Assert.Equal(ssmVals, CopyDeviceToHost(dst.GetSsmStatePtr(0), dst.SsmStateElementsPerLayer));
        Assert.Equal(cumVals, CopyDeviceToHost(dst.GetCumAnglePtr(0), dst.CumAngleElementsPerLayer));
        Assert.Equal(kVals, CopyDeviceToHost(dst.GetKStatePtr(0), dst.KStateElementsPerLayer));
        Assert.Equal(vVals, CopyDeviceToHost(dst.GetVStatePtr(0), dst.VStateElementsPerLayer));
    }

    [SkippableFact]
    public void Clone_ProducesIndependentCache_WithSameShape()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        var m3 = TinySisoConfig();
        using var src = new CudaMamba3StateCache(m3, numLayers: 2);

        float[] ssmVals = SeedSequential(src.SsmStateElementsPerLayer, baseValue: 5f);
        CopyHostToDevice(ssmVals, src.GetSsmStatePtr(0));

        using var clone = src.Clone();
        Assert.Equal(src.NumLayers, clone.NumLayers);
        Assert.Equal(ssmVals, CopyDeviceToHost(clone.GetSsmStatePtr(0), clone.SsmStateElementsPerLayer));

        // Mutating the clone must not affect the source (independent allocation).
        clone.Reset();
        Assert.Equal(ssmVals, CopyDeviceToHost(src.GetSsmStatePtr(0), src.SsmStateElementsPerLayer));
    }

    private static float[] SeedSequential(int length, float baseValue)
    {
        float[] vals = new float[length];
        for (int i = 0; i < vals.Length; i++) vals[i] = baseValue + i;
        return vals;
    }

    private static unsafe void FillWithOnes(nint devicePtr, int elementCount)
    {
        float[] ones = new float[elementCount];
        Array.Fill(ones, 1.0f);
        CopyHostToDevice(ones, devicePtr);
    }

    private static unsafe void CopyHostToDevice(float[] values, nint devicePtr)
    {
        fixed (float* p = values)
            CudaDriverApi.cuMemcpyHtoD_v2(devicePtr, (nint)p, (nuint)(values.Length * sizeof(float))).ThrowOnError();
    }

    private static unsafe float[] CopyDeviceToHost(nint devicePtr, int elementCount)
    {
        float[] host = new float[elementCount];
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(elementCount * sizeof(float))).ThrowOnError();
        return host;
    }

    private static unsafe void AssertAllZero(nint devicePtr, int elementCount)
    {
        float[] host = new float[elementCount];
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(elementCount * sizeof(float))).ThrowOnError();
        foreach (float v in host) Assert.Equal(0f, v);
    }
}
