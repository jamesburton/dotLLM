using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
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
        // The free-VRAM override is a fixed figure, so the KV must cost less than the staging: the copying KV (exact bytes) does,
        // the in-place KV (whole 2 MiB pages, issue #1080) does not.
        using var l = Load(withMtp: false);
        var model = l.Model;
        model.UseCopyingKvGrowthForTest();
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

        // Issue #1080: what a growth needs is the storage it adds (whole granules of each slot's K and V range when it grows in
        // place, the whole new storage when it copies).
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve + model.KvStorageGrowthBytes(16);
        Assert.True(kv.TryGrowTo(16));
    }

    [SkippableFact]
    public void MtpHeadKvGrowth_IsAdmittedByTheModel()
    {
        // The head's KV grows 1024 positions at a time with the context. The copying growth (no CUDA virtual memory) admits
        // the whole new storage, since the old buffers are still held during the copy.
        using var l = Load(withMtp: true, contextLength: 4096);
        var model = l.Model;
        model.UseCopyingKvGrowthForTest();
        using var mtp = (CudaMtpState)model.CreateMtpState(4096)!;
        Assert.False(mtp.GrowsInPlace);
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
    public unsafe void MtpHeadKvGrowth_InPlace_KeepsRowsAndPointer_AndIsChargedOnlyTheAddedPages()
    {
        // Issue #1095: the head's K, V and iota map pages behind their rows. A growth that stays inside the mapped granules
        // needs nothing; one that crosses a granule is charged exactly the pages it maps, and no row or pointer moves.
        using var l = Load(withMtp: true, contextLength: 4096);
        var model = l.Model;
        Skip.IfNot(model.KvGrowsInPlace, "CUDA virtual memory unavailable on this device");
        const int Ceiling = 1 << 18;
        using var mtp = (CudaMtpState)model.CreateMtpState(Ceiling)!;
        Assert.True(mtp.GrowsInPlace);
        Assert.Equal(CudaMtpKvFormat.F16, mtp.KeyFormat);   // the synthetic trunk is F16; never wider (#1107)
        long granule = CudaDriverVirtualMemory.TryCreate(0)!.Granularity;

        // A recognisable pattern in a committed row, read back after the growths.
        const int Row = 5;
        byte[] pattern = new byte[mtp.KeyRowBytes];
        for (int i = 0; i < pattern.Length; i++) pattern[i] = (byte)(i * 37 + 11);
        fixed (byte* p = pattern)
            CudaDriverApi.cuMemcpyHtoD_v2(mtp.GetKeyRowDevicePtr(Row), (nint)p, (nuint)pattern.Length).ThrowOnError();
        nint keyPtr = mtp.KeyCacheDevicePtr, valuePtr = mtp.ValueCacheDevicePtr;

        // Inside the first granule: free VRAM at exactly the reserve still admits it.
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve;
        mtp.EnsureCapacity(2000, stream: 0);
        Assert.Equal(2048, mtp.Capacity);

        // Past it: K and V each map whole granules for the rows they add; the iota (4 B/row) still fits its first granule.
        int target = (int)(granule / mtp.KeyRowBytes) + 1;
        int targetRounded = (target + 1023) / 1024 * 1024;
        Assert.True((long)targetRounded * sizeof(int) <= granule, "the iota must stay in its first granule for this count");
        long added = RoundUp(targetRounded * mtp.KeyRowBytes, granule) - granule
                   + RoundUp(targetRounded * mtp.ValueRowBytes, granule) - granule;
        long before = mtp.AllocatedBytes;
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve + added - 1;
        var ex = Assert.Throws<DeviceMemoryAdmissionException>(() => mtp.EnsureCapacity(target, stream: 0));
        Assert.Contains($"MTP head KV cache for {targetRounded} positions", ex.Message);
        Assert.Equal(2048, mtp.Capacity);
        CudaQwen3HybridDenseTransformerModel.FreeVramOverride = Reserve + added;
        mtp.EnsureCapacity(target, stream: 0);
        Assert.Equal(targetRounded, mtp.Capacity);
        Assert.Equal(added, mtp.AllocatedBytes - before);

        Assert.Equal(keyPtr, mtp.KeyCacheDevicePtr);
        Assert.Equal(valuePtr, mtp.ValueCacheDevicePtr);
        byte[] back = new byte[pattern.Length];
        fixed (byte* p = back)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, mtp.GetKeyRowDevicePtr(Row), (nuint)back.Length).ThrowOnError();
        Assert.Equal(pattern, back);

        static long RoundUp(long bytes, long g) => (bytes + g - 1) / g * g;
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
        model.UseCopyingKvGrowthForTest();   // issue #1080: in place, growing 16 -> 64 tokens stays inside the first page and needs nothing
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
