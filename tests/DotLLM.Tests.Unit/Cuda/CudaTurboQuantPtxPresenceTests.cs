using System;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1892 Task 3 follow-up (CI assert): the TurboQuant CUDA suites skip when <c>TqRotateAvailable</c> (or another TurboQuant kernel flag) is false,
/// so a missing <c>turboquant_rotate.ptx</c> would turn the whole suite into skips and CI would stay green. On a host where the CUDA
/// kernels DO load (the older attention kernels resolve), the TurboQuant ones must too: this test FAILS, it does not skip.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaTurboQuantPtxPresenceTests
{
    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    [SkippableFact]
    public void TurboQuantKernels_AreLoaded_WheneverTheOtherCudaKernelsAre()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var kernels = new CudaKernels(ptxDir!);
        // Precondition (a skip, not a failure: this host has no usable CUDA kernel set at all).
        Skip.IfNot(kernels.HasAttentionSoftmaxCausal, "the base CUDA kernels did not load on this host");

        // From here a missing TurboQuant kernel is a defect: the PTX set is incomplete or stale.
        Assert.True(kernels.TurboQuantF16Available, "turboquant.ptx (encode/dequant F16) did not load although the base kernels did");
        Assert.True(kernels.TqRotateAvailable, "turboquant_rotate.ptx (Q rotation) did not load although the base kernels did");
        Assert.True(kernels.TqRotateOF16Available, "turboquant_rotate.ptx (O rotation F16) did not load although the base kernels did");
        Assert.True(kernels.HasAttentionKvTq, "attention_kvtq.ptx did not load although the base kernels did");
    }
}
