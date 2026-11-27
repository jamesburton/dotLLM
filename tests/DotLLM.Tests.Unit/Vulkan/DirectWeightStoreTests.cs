using System.Runtime.InteropServices;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.DirectStorage;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1304: Windows DirectStorage -&gt; shared D3D12 heap -&gt; Vulkan import of as-is weight tensors.
/// The layout tests are pure CPU; the load tests need a Windows GPU with <c>dstorage.dll</c> and skip otherwise.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class DirectWeightStoreTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _output = output;

    /// <summary>The hand-written interop structs must match the SDK headers byte for byte (sizes/offsets measured with a C++ probe).</summary>
    [Fact]
    public void InteropStructs_MatchSdkLayouts()
    {
        Assert.Equal(104, Marshal.SizeOf<DStorageNative.Request>());
        Assert.Equal(16, (int)Marshal.OffsetOf<DStorageNative.Request>(nameof(DStorageNative.Request.SourceFile)));
        Assert.Equal(40, (int)Marshal.OffsetOf<DStorageNative.Request>(nameof(DStorageNative.Request.DestResource)));
        Assert.Equal(80, (int)Marshal.OffsetOf<DStorageNative.Request>(nameof(DStorageNative.Request.UncompressedSize)));
        Assert.Equal(96, (int)Marshal.OffsetOf<DStorageNative.Request>(nameof(DStorageNative.Request.Name)));
        Assert.Equal(32, Marshal.SizeOf<DStorageNative.QueueDesc>());
        Assert.Equal(48, Marshal.SizeOf<D3dNative.HeapDesc>());
        Assert.Equal(56, Marshal.SizeOf<D3dNative.ResourceDesc>());
        Assert.Equal(0xC0, D3dNative.HeapFlagAllowOnlyBuffers); // D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS (0x200 is not it: CreateHeap -> E_INVALIDARG)
    }

    /// <summary>A pointer into a mapped tensor maps back to the file offset that holds the same bytes.</summary>
    [Fact]
    public void TryResolveFileRange_MapsTensorPointerToFileOffset()
    {
        byte[] data = RandomBytes(4096, seed: 1);
        string path = WriteSingleTensorGguf(data);
        try
        {
            using var gguf = GgufFile.Open(path);
            var desc = gguf.TensorsByName["w"];
            nint ptr = gguf.TensorDataPointer(desc);

            Assert.True(GgufFile.TryResolveFileRange(ptr, data.Length, out string resolved, out long offset));
            Assert.Equal(Path.GetFullPath(path), Path.GetFullPath(resolved));
            byte[] onDisk = new byte[data.Length];
            using (var fs = new FileStream(resolved, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            {
                fs.Seek(offset, SeekOrigin.Begin);
                fs.ReadExactly(onDisk);
            }
            Assert.Equal(data, onDisk);

            // A range running past the mapped data, and a foreign pointer, are refused.
            Assert.False(GgufFile.TryResolveFileRange(ptr, gguf.DataSectionLength + 1, out _, out _));
            Assert.False(GgufFile.TryResolveFileRange(1, 16, out _, out _));
        }
        finally { File.Delete(path); }
    }

    /// <summary>
    /// The end-to-end guard: tensors loaded through DirectStorage (two slab-sized ones sharing a heap, one large dedicated one)
    /// read back through Vulkan byte-identical to the file, and the heaps are released on dispose.
    /// </summary>
    [SkippableFact]
    public void TryLoad_ReadsBackByteIdentical()
    {
        Skip.IfNot(OperatingSystem.IsWindows(), "DirectStorage is Windows-only.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryWin32, "VK_KHR_external_memory_win32 absent.");
        var store = DirectWeightStore.For(device);
        Skip.If(store is null, "DirectStorage unavailable: " + DirectWeightStore.UnavailableReason(device));

        byte[] a = RandomBytes(1 << 20, seed: 11);          // slab
        byte[] b = RandomBytes((2 << 20) + 4096, seed: 12); // slab, same heap as a
        byte[] c = RandomBytes(20 << 20, seed: 13);         // >= DedicatedThresholdBytes: its own heap
        string path = WriteGguf(("a", a), ("b", b), ("c", c));
        try
        {
            using var gguf = GgufFile.Open(path);
            var loaded = new List<(byte[] Data, VulkanDevice.Buffer Buf)>();
            foreach (var (name, data) in new[] { ("a", a), ("b", b), ("c", c) })
            {
                nint ptr = gguf.TensorDataPointer(gguf.TensorsByName[name]);
                Assert.True(store!.TryLoad(ptr, data.Length, out var buf), $"{name}: {store.LastFallbackReason}");
                Assert.NotNull(buf);
                Assert.False(buf!.IsHostVisible);
                loaded.Add((data, buf));
            }

            store!.WaitAll();

            foreach (var (data, buf) in loaded)
            {
                var floats = new float[data.Length / 4];
                device.Download(buf, floats);
                byte[] back = new byte[floats.Length * 4];
                Buffer.BlockCopy(floats, 0, back, 0, back.Length);
                Assert.True(data.AsSpan(0, back.Length).SequenceEqual(back), "bytes read back from device differ from the file");
            }

            long liveBefore = device.LiveBytesTotal();
            store.Seal();
            foreach (var (_, buf) in loaded) buf.Dispose();
            Assert.True(device.LiveBytesTotal() < liveBefore, "imported heaps were not released");
            _output.WriteLine($"direct tensors={store.LoadedTensorCount} bytes={store.LoadedBytes} live {liveBefore} -> {device.LiveBytesTotal()}");
        }
        finally { File.Delete(path); }
    }

    /// <summary>
    /// #2044: a shared heap is charged by D3D12 AND by Vulkan's import of it, which removed the device at ~85 GiB of D3D12 "local"
    /// usage (a 49 GiB model, 2x). The D3D12 side is dropped once the reads landed, so the weights must stay valid and byte-identical
    /// after <see cref="DirectWeightStore.Seal"/> (dedicated heap and slab heap both), with the D3D12 heap objects already gone.
    /// </summary>
    [SkippableFact]
    public void Seal_ReleasesD3d12Side_WeightsStayByteIdentical()
    {
        Skip.IfNot(OperatingSystem.IsWindows(), "DirectStorage is Windows-only.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryWin32, "VK_KHR_external_memory_win32 absent.");
        var store = DirectWeightStore.For(device);
        Skip.If(store is null, "DirectStorage unavailable: " + DirectWeightStore.UnavailableReason(device));

        byte[] a = RandomBytes(1 << 20, seed: 21);   // slab heap
        byte[] c = RandomBytes(20 << 20, seed: 23);  // dedicated heap
        string path = WriteGguf(("a", a), ("c", c));
        try
        {
            using var gguf = GgufFile.Open(path);
            var loaded = new List<(byte[] Data, VulkanDevice.Buffer Buf)>();
            foreach (var (name, data) in new[] { ("a", a), ("c", c) })
            {
                Assert.True(store!.TryLoad(gguf.TensorDataPointer(gguf.TensorsByName[name]), data.Length, out var buf), store.LastFallbackReason);
                loaded.Add((data, buf!));
            }
            long d3dBefore = store!.D3d12LocalUsageBytes();
            int heapsBefore = DirectWeightStore.LiveD3d12Heaps;
            Assert.True(heapsBefore >= 2, "expected a slab heap and a dedicated heap to be live on the D3D12 side");
            store.Seal();   // waits for the reads, then drops every D3D12 heap + resource
            _output.WriteLine($"D3D12 local usage {d3dBefore >> 20} -> {store.D3d12LocalUsageBytes() >> 20} MiB; live D3D12 heaps {heapsBefore} -> {DirectWeightStore.LiveD3d12Heaps}");
            Assert.True(DirectWeightStore.LiveD3d12Heaps <= heapsBefore - 2,
                "Seal did not release the D3D12 side of the loaded heaps (each stays charged twice: D3D12 + Vulkan import)");

            foreach (var (data, buf) in loaded)
            {
                var floats = new float[data.Length / 4];
                device.Download(buf, floats);
                byte[] back = new byte[floats.Length * 4];
                Buffer.BlockCopy(floats, 0, back, 0, back.Length);
                Assert.True(data.AsSpan(0, back.Length).SequenceEqual(back), "weights changed after the D3D12 side was released");
                buf.Dispose();
            }
        }
        finally { File.Delete(path); }
    }

    /// <summary>
    /// #2044: the fix that matters for a large model is the retire DURING the load. Bolt failed at heap ~#241, long before Seal. Once
    /// the retire window is passed, the next heap must first wait for the reads and then drop the D3D12 side of the finished dedicated
    /// heaps. So at most the last heap is still live on the D3D12 side before Seal, and every weight reads back byte-identical.
    /// Mutants: deleting the release loop in DrainAndRetire leaves all 6 heaps live; releasing before WaitAll corrupts the weights.
    /// </summary>
    [SkippableFact]
    public void MidLoadRetire_ReleasesFinishedHeapsBeforeSeal_WeightsStayByteIdentical()
    {
        Skip.IfNot(OperatingSystem.IsWindows(), "DirectStorage is Windows-only.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryWin32, "VK_KHR_external_memory_win32 absent.");
        var store = DirectWeightStore.For(device);
        Skip.If(store is null, "DirectStorage unavailable: " + DirectWeightStore.UnavailableReason(device));
        store!.RetireWindowBytes = 16L << 20; // every 24 MiB tensor passes the window, so each new heap retires the previous ones

        var tensors = Enumerable.Range(0, 6).Select(i => ($"t{i}", RandomBytes(24 << 20, seed: 100 + i))).ToArray();
        string path = WriteGguf(tensors);
        try
        {
            using var gguf = GgufFile.Open(path);
            int live0 = DirectWeightStore.LiveD3d12Heaps;
            var loaded = new List<(byte[] Data, VulkanDevice.Buffer Buf)>();
            foreach (var (name, data) in tensors)
            {
                Assert.True(store.TryLoad(gguf.TensorDataPointer(gguf.TensorsByName[name]), data.Length, out var buf), store.LastFallbackReason);
                loaded.Add((data, buf!));
            }
            int liveBeforeSeal = DirectWeightStore.LiveD3d12Heaps - live0;
            store.Seal();
            _output.WriteLine($"live D3D12 heaps before Seal: {liveBeforeSeal} of {tensors.Length}");
            Assert.True(liveBeforeSeal <= 1, $"{liveBeforeSeal} of {tensors.Length} D3D12 heaps still live before Seal: the mid-load retire did not run");

            foreach (var (data, buf) in loaded)
            {
                var floats = new float[data.Length / 4];
                device.Download(buf, floats);
                byte[] back = new byte[floats.Length * 4];
                Buffer.BlockCopy(floats, 0, back, 0, back.Length);
                Assert.True(data.AsSpan(0, back.Length).SequenceEqual(back), "weights changed after the mid-load retire");
                buf.Dispose();
            }
        }
        finally
        {
            store.Seal(); // closes the DirectStorage file handle even when an assertion fired, so the delete below cannot mask it
            File.Delete(path);
        }
    }

    /// <summary>
    /// #2044 admission rule. The first row is the measured failure: D3D12 local usage 85114 MiB of a 103727 MiB budget when the 886 MiB
    /// expert bank was refused. The second row pins the 2x charge (heap + Vulkan import): a 1x projection (60000) would admit it.
    /// </summary>
    [Theory]
    [InlineData(85114L, 103727L, 886L, false)]
    [InlineData(40000L, 103727L, 20000L, false)]
    [InlineData(30000L, 103727L, 20000L, true)]
    [InlineData(50000L, 0L, 99999L, true)]
    public void AdmitHeap_ChargesBothSidesAndRefusesPastTheFraction(long usageMiB, long budgetMiB, long heapMiB, bool expected)
        => Assert.Equal(expected, DirectWeightStore.AdmitHeap(usageMiB << 20, budgetMiB << 20, heapMiB << 20, 0.75));

    /// <summary>Anything that is not file-backed GGUF data is declined with a reason, never an exception.</summary>
    [SkippableFact]
    public void TryLoad_DeclinesNonFileBackedPointer()
    {
        Skip.IfNot(OperatingSystem.IsWindows(), "DirectStorage is Windows-only.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        var store = DirectWeightStore.For(device);
        Skip.If(store is null, "DirectStorage unavailable.");

        unsafe
        {
            void* p = NativeMemory.AlignedAlloc(1 << 20, 4096);
            try
            {
                Assert.False(store!.TryLoad((nint)p, 1 << 20, out var buf));
                Assert.Null(buf);
                Assert.Equal("not_file_backed", store.LastFallbackReason);
            }
            finally { NativeMemory.AlignedFree(p); }
        }
    }

    private static byte[] RandomBytes(int n, int seed)
    {
        var b = new byte[n];
        new Random(seed).NextBytes(b);
        return b;
    }

    private static string WriteSingleTensorGguf(byte[] data) => WriteGguf(("w", data));

    // F32 tensors (quant type 0) with raw random payloads: the loader treats them as opaque bytes.
    private static string WriteGguf(params (string Name, byte[] Data)[] tensors)
    {
        using var g = new GgufTestData(version: 3);
        g.AddString("general.architecture", "llama");
        foreach (var (name, data) in tensors)
            g.AddTensor(name, [data.Length / 4], 0, data);
        return g.WriteToTempFile();
    }
}
