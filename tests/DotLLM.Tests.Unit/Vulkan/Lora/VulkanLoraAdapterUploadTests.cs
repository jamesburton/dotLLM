using System.Runtime.InteropServices;
using DotLLM.Core.Lora;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan.Lora;

/// <summary>
/// Smoke tests for <see cref="VulkanLoraAdapter"/>: confirm that a synthetic
/// adapter uploads cleanly, every (layer, proj) entry round-trips with the
/// alpha/rank scaling pre-folded into B, and the cache returns the same
/// instance on repeat lookup.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanLoraAdapterUploadTests
{
    [SkippableFact]
    public unsafe void Upload_ScalesB_AndPreservesA()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        const int rank = 4;
        const int inputDim = 8;
        const int outputDim = 12;
        const float alpha = 8.0f;
        const float scale = alpha / rank;

        // Build a deterministic synthetic adapter with one (layer=0, proj=q_proj) entry.
        using var adapter = new LoraAdapter("test", rank, alpha, ["q_proj"]);
        nint bHandle = LoraAdapter.AllocAligned((long)rank * inputDim);
        nint aHandle = LoraAdapter.AllocAligned((long)outputDim * rank);
        var bSrc = new float[rank * inputDim];
        var aSrc = new float[outputDim * rank];
        for (int i = 0; i < bSrc.Length; i++) bSrc[i] = i * 0.1f - 0.4f;
        for (int i = 0; i < aSrc.Length; i++) aSrc[i] = i * -0.05f + 0.15f;
        bSrc.AsSpan().CopyTo(new Span<float>((void*)bHandle, bSrc.Length));
        aSrc.AsSpan().CopyTo(new Span<float>((void*)aHandle, aSrc.Length));
        adapter.AddLayerWeights(0, "q_proj",
            new LoraLayerWeights(AHandle: aHandle, BHandle: bHandle, InputDim: inputDim, OutputDim: outputDim));

        using var device = VulkanDevice.Create();
        using var vkLora = VulkanLoraAdapter.Upload(device, adapter);

        Assert.Equal(rank, vkLora.Rank);
        Assert.Equal(scale, vkLora.Scale);
        Assert.Equal(outputDim, vkLora.MaxOutputDim);

        var lb = vkLora.Get(0, "q_proj");
        Assert.NotNull(lb);
        Assert.Equal(inputDim, lb!.Value.InputDim);
        Assert.Equal(outputDim, lb.Value.OutputDim);
        Assert.Equal(rank, lb.Value.Rank);

        // Round-trip B through host: download via a host-visible buffer copy,
        // confirm scale is folded into B and A is verbatim. The device buffer
        // is device-local so we stage through a host-visible buffer.
        var bDl = new float[rank * inputDim];
        var aDl = new float[outputDim * rank];
        DownloadDeviceLocal(device, lb.Value.B, MemoryMarshal.AsBytes(bDl.AsSpan()));
        DownloadDeviceLocal(device, lb.Value.A, MemoryMarshal.AsBytes(aDl.AsSpan()));

        for (int i = 0; i < bSrc.Length; i++)
        {
            float expected = bSrc[i] * scale;
            float diff = MathF.Abs(expected - bDl[i]);
            Assert.True(diff < 1e-6f,
                $"B[{i}] expected {expected} (= {bSrc[i]} * scale {scale}) but got {bDl[i]} (diff {diff}).");
        }
        for (int i = 0; i < aSrc.Length; i++)
        {
            Assert.Equal(aSrc[i], aDl[i]);
        }
    }

    [SkippableFact]
    public unsafe void Cache_ReturnsSameInstance_OnRepeatLookup()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        const int rank = 4, inputDim = 8, outputDim = 12;
        using var adapter = new LoraAdapter("test", rank, alpha: 8f, ["q_proj"]);
        nint b = LoraAdapter.AllocAligned((long)rank * inputDim);
        nint a = LoraAdapter.AllocAligned((long)outputDim * rank);
        new Span<float>((void*)b, rank * inputDim).Clear();
        new Span<float>((void*)a, outputDim * rank).Clear();
        adapter.AddLayerWeights(0, "q_proj",
            new LoraLayerWeights(AHandle: a, BHandle: b, InputDim: inputDim, OutputDim: outputDim));

        using var device = VulkanDevice.Create();
        using var cache = new VulkanLoraAdapterCache(device);

        var first = cache.GetOrAdd(adapter);
        var second = cache.GetOrAdd(adapter);

        // Reference equality — same upload returned, no re-upload on second call.
        Assert.Same(first, second);
        Assert.Equal(1, cache.Count);
    }

    [SkippableFact]
    public unsafe void Upload_IncludesMlaAndPerExpertProjections_ButStripsUnknownNames()
    {
        // MLA-specific and per-expert MoE projection names are runtime
        // supported and must upload. Truly unknown target names should still
        // be silently skipped at upload.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        const int rank = 4, inputDim = 8, outputDim = 12;
        using var adapter = new LoraAdapter("test", rank, alpha: 8f,
            ["q_proj", "q_a_proj", "mlp.experts.0.gate_proj", "not_a_projection"]);

        nint b1 = LoraAdapter.AllocAligned((long)rank * inputDim);
        nint a1 = LoraAdapter.AllocAligned((long)outputDim * rank);
        new Span<float>((void*)b1, rank * inputDim).Clear();
        new Span<float>((void*)a1, outputDim * rank).Clear();
        adapter.AddLayerWeights(0, "q_proj",
            new LoraLayerWeights(AHandle: a1, BHandle: b1, InputDim: inputDim, OutputDim: outputDim));

        // MLA-specific name — must upload.
        nint b2 = LoraAdapter.AllocAligned((long)rank * inputDim);
        nint a2 = LoraAdapter.AllocAligned((long)outputDim * rank);
        new Span<float>((void*)b2, rank * inputDim).Clear();
        new Span<float>((void*)a2, outputDim * rank).Clear();
        adapter.AddLayerWeights(0, "q_a_proj",
            new LoraLayerWeights(AHandle: a2, BHandle: b2, InputDim: inputDim, OutputDim: outputDim));

        // Per-expert MoE name — must upload.
        nint b3 = LoraAdapter.AllocAligned((long)rank * inputDim);
        nint a3 = LoraAdapter.AllocAligned((long)outputDim * rank);
        new Span<float>((void*)b3, rank * inputDim).Clear();
        new Span<float>((void*)a3, outputDim * rank).Clear();
        adapter.AddLayerWeights(0, "mlp.experts.0.gate_proj",
            new LoraLayerWeights(AHandle: a3, BHandle: b3, InputDim: inputDim, OutputDim: outputDim));

        // Unknown name — must be ignored by upload.
        nint b4 = LoraAdapter.AllocAligned((long)rank * inputDim);
        nint a4 = LoraAdapter.AllocAligned((long)outputDim * rank);
        new Span<float>((void*)b4, rank * inputDim).Clear();
        new Span<float>((void*)a4, outputDim * rank).Clear();
        adapter.AddLayerWeights(0, "not_a_projection",
            new LoraLayerWeights(AHandle: a4, BHandle: b4, InputDim: inputDim, OutputDim: outputDim));

        using var device = VulkanDevice.Create();
        using var vkLora = VulkanLoraAdapter.Upload(device, adapter);

        Assert.NotNull(vkLora.Get(0, "q_proj"));
        Assert.NotNull(vkLora.Get(0, "q_a_proj"));
        Assert.NotNull(vkLora.Get(0, "mlp.experts.0.gate_proj"));
        Assert.Null(vkLora.Get(0, "not_a_projection"));
    }

    /// <summary>
    /// Stages a device-local buffer into host memory by copying through a
    /// host-visible staging buffer + <c>vkCmdCopyBuffer</c>. Used by the
    /// upload round-trip test to verify the bytes that landed on device.
    /// </summary>
    private static unsafe void DownloadDeviceLocal(VulkanDevice device, VulkanDevice.Buffer src, Span<byte> dest)
    {
        long bytes = dest.Length;
        if (bytes > src.Size)
            throw new ArgumentException("Destination larger than source buffer.", nameof(dest));

        using var staging = device.Allocate(bytes);
        device.CopyBufferSynchronous(src, staging, (ulong)bytes);

        // staging is host-visible host-coherent — map and copy out.
        var floatDst = MemoryMarshal.Cast<byte, float>(dest);
        device.Download(staging, floatDst);
    }
}
