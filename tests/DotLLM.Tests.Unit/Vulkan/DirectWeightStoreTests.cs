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
