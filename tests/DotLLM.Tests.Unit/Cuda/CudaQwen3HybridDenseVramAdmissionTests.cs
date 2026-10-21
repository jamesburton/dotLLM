using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1068: every device buffer of the CUDA hybrid that grows with the context is admitted against free VRAM, and a growth
/// that would not leave <see cref="CudaQwen3HybridDenseTransformerModel.VramReserveBytes"/> free throws
/// <see cref="DeviceMemoryAdmissionException"/> instead of letting WDDM back it with shared system memory. Driven with a fake
/// small free-VRAM figure (<see cref="CudaQwen3HybridDenseTransformerModel.FreeVramOverride"/>), so no real exhaustion is needed.
/// </summary>
/// <remarks>
/// Mutants (each fails its test): deleting the KV admission in <c>EnsureF16KvCache</c>; deleting the staging admission in
/// <c>EnsureF32KvStaging</c>; dropping <c>admitGrowth: AdmitDeviceGrowth</c> from <c>CreateMtpState</c>; restoring the old
/// "<c>_f16KCache is null</c> admits everything" short-circuit in <c>CanGrowKvStorageTo</c>.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseVramAdmissionTests : IDisposable
{
    private static readonly long Reserve = CudaQwen3HybridDenseTransformerModel.VramReserveBytes;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-vram-admit-{Guid.NewGuid():N}");

    public CudaQwen3HybridDenseVramAdmissionTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = null;
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void FreshKvAllocation_IsRefused_WhenItDoesNotFit_AndSucceedsOnceItDoes()
    {
        // The miss path (a new handle) used to allocate with no admission at all: an agent prompt plus its max_tokens went
        // straight into shared memory.
        using var l = Load(withMtp: false);
        var model = l.Model;
        using var kv = model.CreateKvCache(16);
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve + model.ComputeKvBytes(16) - 1;
        var ex = Assert.Throws<DeviceMemoryAdmissionException>(
            () => model.Forward([1, 4, 7], [0, 1, 2], deviceId: -1, kv, lastTokenLogitsOnly: true));
        Assert.Contains("KV cache", ex.Message);
        Assert.Contains("for 16 tokens", ex.Message);
        Assert.Equal(0, model.KvAllocatedBytes);

        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = null;
        using (ITensor _ = model.Forward([1, 4, 7], [0, 1, 2], deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        Assert.Equal(model.ComputeKvBytes(16), model.KvAllocatedBytes);
    }

    [SkippableFact]
    public void AttentionStaging_IsRefused_WhenTheKvFitsButTheStagingDoesNot()
    {
        // The F32 staging pair (and, for quantized KV, the F16 dequant scratch) is sized seqKv * kvElems and grew with no check.
        // The synthetic model's head_dim is 16, so its prefill takes the F32 staging tier.
        using var l = Load(withMtp: false);
        var model = l.Model;
        using var kv = model.CreateKvCache(16);
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve + model.ComputeKvBytes(16) + 1024;
        var ex = Assert.Throws<DeviceMemoryAdmissionException>(
            () => model.Forward([1, 4, 7], [0, 1, 2], deviceId: -1, kv, lastTokenLogitsOnly: true));
        Assert.Contains("F32 attention staging", ex.Message);
    }

    [SkippableFact]
    public void KvGrowth_IsRefused_ForAnyStorage_WhenTheCopyWouldNotFit()
    {
        // The handle's growth check: also before the first forward (no storage yet), where it used to admit unconditionally.
        using var l = Load(withMtp: false);
        var model = l.Model;
        using var kv = model.CreateKvCache(6);
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve;
        Assert.False(kv.TryGrowTo(16));
        Assert.Equal(6, kv.MaxLength);

        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve + model.ComputeKvBytes(16);
        Assert.True(kv.TryGrowTo(16));
    }

    [SkippableFact]
    public void MtpHeadKvGrowth_IsAdmittedByTheModel()
    {
        // The head's F32 KV grows 1024 positions at a time with the context (8 KiB/position on Bonsai 2).
        using var l = Load(withMtp: true, contextLength: 4096);
        var model = l.Model;
        using var mtp = (CudaMtpState)model.CreateMtpState(4096)!;
        Assert.Equal(1024, mtp.Capacity);

        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve;
        var ex = Assert.Throws<DeviceMemoryAdmissionException>(() => mtp.EnsureCapacity(2000, stream: 0));
        Assert.Contains("MTP head KV cache for 2048 positions", ex.Message);
        Assert.Equal(1024, mtp.Capacity);   // the refused state is still valid at its old size

        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = null;
        mtp.EnsureCapacity(2000, stream: 0);
        Assert.Equal(2048, mtp.Capacity);
    }

    [SkippableFact]
    public void RecurrentCheckpoint_IsAdmitted()
    {
        using var l = Load(withMtp: false);
        var model = l.Model;
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve;
        var ex = Assert.Throws<DeviceMemoryAdmissionException>(() => model.CheckpointRecurrentState());
        Assert.Contains("checkpoint", ex.Message);
    }

    [SkippableFact]
    public void PrefixSnapshots_UseTheSharedAdmission_ForDeviceCopiesAndRequestHeadroom()
    {
        // Rebase onto the #1033/#1034 prefix cache: a device snapshot and the restore headroom check each kept their own 512 MiB
        // margin and read cuMemGetInfo directly. Both now go through the shared reserve, so the override drives them too.
        using var l = Load(withMtp: false, contextLength: 64);
        var model = l.Model;
        var provider = (DotLLM.Core.Models.IRecurrentPrefixSnapshotProvider)model;
        using var kv = model.CreateKvCache(16);
        using (ITensor _ = model.Forward([1, 4, 7], [0, 1, 2], deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        using var larger = model.CreateKvCache(64);

        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve;
        Assert.Throws<DeviceMemoryAdmissionException>(
            () => provider.CapturePrefixSnapshot(kv, 3, DotLLM.Core.Models.RecurrentPrefixSnapshotStorage.Device));
        Assert.False(model.HasPrefixSnapshotRequestHeadroom(larger));

        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve + model.ComputeKvBytes(64);
        Assert.True(model.HasPrefixSnapshotRequestHeadroom(larger));
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = null;   // the real card has room for a few KiB
        using var snapshot = provider.CapturePrefixSnapshot(kv, 3, DotLLM.Core.Models.RecurrentPrefixSnapshotStorage.Device);
        Assert.Equal(3, snapshot.PrefixLength);
    }

    private Loaded Load(bool withMtp, int contextLength = 16)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, $"m-{withMtp}-{contextLength}.gguf"), withMtp: withMtp, contextLength: contextLength);
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        return new Loaded(gguf, CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!));
    }

    /// <summary>The model and the GGUF mapping it reads, disposed together.</summary>
    private sealed record Loaded(GgufFile Gguf, CudaQwen3HybridDenseTransformerModel Model) : IDisposable
    {
        public void Dispose()
        {
            Model.Dispose();
            Gguf.Dispose();
        }
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
