using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #874: the multi-threaded staging path and the background bank pre-allocator. Correctness is the contract - the parallel path must
/// deliver byte-identical device contents to the serial one for every chunk geometry (tail chunk, zero-pad tail, tensors smaller
/// than two slots), and must leave the buffer usable by the serial path afterwards.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanParallelStagingUploadTests
{
    private const long MiB = 1L << 20;

    private static float[] RandomFloats(int count, int seed)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static void AssertDeviceEquals(VulkanDevice device, VulkanDevice.Buffer buf, float[] expected, int atFloat = 0)
    {
        var back = new float[expected.Length];
        device.Download(buf, back);
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(back[i]) != BitConverter.SingleToInt32Bits(expected[i]))
                Assert.Fail($"mismatch at float {atFloat + i}: device {back[i]} vs source {expected[i]}");
    }

    [SkippableTheory]
    [InlineData(40 * MiB, 0)]            // 10 chunks of 4 MiB, exact
    [InlineData(40 * MiB + 4096, 0)]     // a short tail chunk
    [InlineData(23 * MiB + 100, 0)]      // not a multiple of the slot size
    [InlineData(9 * MiB - 2, 2)]         // odd byte count with the 2-byte zero pad folded into the last chunk
    [InlineData(8 * MiB, 0)]             // exactly two slots: smallest size that takes the parallel path
    [InlineData(5 * MiB, 0)]             // below two slots: falls back to the serial path inside the same buffer
    public void ParallelUpload_IsByteIdenticalToTheSource(long bytes, long zeroTail)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        long total = bytes + zeroTail;
        Assert.Equal(0, total % 4);
        float[] src = RandomFloats((int)(total / 4), seed: 7);
        var expected = (float[])src.Clone();
        // The pad bytes are zeros on the device: clear the final float's pad region in the expectation.
        if (zeroTail > 0)
        {
            var bits = BitConverter.SingleToUInt32Bits(expected[^1]);
            bits &= zeroTail == 2 ? 0x0000FFFFu : 0u;
            expected[^1] = BitConverter.UInt32BitsToSingle(bits);
        }

        using var staging = VulkanStagingBuffer.CreateParallel(device, slotBytes: 4 * MiB, slotCount: 6);
        Assert.True(staging.IsParallel);
        using var dst = device.AllocateDeviceLocal(total);
        fixed (float* p = src)
            staging.UploadBytes((nint)p, bytes, dst, 0, zeroTail);
        staging.WaitAll();
        AssertDeviceEquals(device, dst, expected);
    }

    [SkippableFact]
    public void SerialPathStillWorks_AfterAParallelUpload_OnTheSameBuffer()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        using var staging = VulkanStagingBuffer.CreateParallel(device, slotBytes: 2 * MiB, slotCount: 5);

        float[] big = RandomFloats((int)(30 * MiB / 4), 1);   // 15 slots worth: wraps the ring several times
        float[] small = RandomFloats(1000, 2);
        float[] tiny = RandomFloats(64, 3);
        using var d1 = device.AllocateDeviceLocal(30 * MiB);
        using var d2 = device.AllocateDeviceLocal(4000);
        using var d3 = device.AllocateDeviceLocal(256);

        fixed (float* p = big) staging.UploadBytes((nint)p, 30 * MiB, d1);
        staging.UploadFloats(small, d2);                       // serial path, immediately after
        fixed (float* p = tiny) staging.UploadBytes((nint)p, 256, d3);
        staging.WaitAll();

        AssertDeviceEquals(device, d1, big);
        AssertDeviceEquals(device, d2, small);
        AssertDeviceEquals(device, d3, tiny);
        Assert.NotEqual(0, staging.Mapped);                   // lazily created slot after rotation
        Assert.True(staging.Submits >= 15);
    }

    [SkippableFact]
    public void ParallelUpload_RefusesHostOnlyRanges_LikeTheSerialPath()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        using var staging = VulkanStagingBuffer.CreateParallel(device, 2 * MiB, 4);
        byte* host = (byte*)NativeMemory.AlignedAlloc((nuint)(16 * MiB), 4096);
        try
        {
            VulkanWeightImportPolicy.RegisterHostOnly((nint)host, 16 * MiB);
            using var dst = device.AllocateDeviceLocal(16 * MiB);
            Assert.Throws<InvalidOperationException>(() => staging.UploadBytes((nint)host, 16 * MiB, dst));
        }
        finally
        {
            VulkanWeightImportPolicy.UnregisterHostOnly((nint)host);
            NativeMemory.AlignedFree(host);
        }
    }

    [SkippableFact]
    public void BankPrealloc_HandsOutScheduledSizesFifo_AndAllocatesInlineWhenUnscheduled()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        using var pool = new VulkanBankPrealloc(device);
        pool.Schedule([3 * MiB, 5 * MiB, 3 * MiB]);

        using var a = pool.Take(3 * MiB)!;
        using var b = pool.Take(5 * MiB)!;
        using var c = pool.Take(3 * MiB)!;
        Assert.Equal(3 * MiB, a.Size);
        Assert.Equal(5 * MiB, b.Size);
        Assert.Equal(3 * MiB, c.Size);
        Assert.NotEqual(a.Handle, c.Handle);
        Assert.Null(pool.Take(3 * MiB));           // exhausted: caller allocates inline
        Assert.Null(pool.Take(7 * MiB));           // never scheduled
        Assert.Equal(0, pool.PendingCount);
    }

    [SkippableFact]
    public void BankPrealloc_Dispose_ReleasesUntakenBuffers()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        long before = device.LiveBytesOnHeap(0) + device.LiveBytesOnHeap(1);
        var pool = new VulkanBankPrealloc(device);
        pool.Schedule([64 * MiB, 64 * MiB]);
        pool.Dispose();                             // waits for the in-flight allocations, then frees them
        Assert.Equal(before, device.LiveBytesOnHeap(0) + device.LiveBytesOnHeap(1));
    }
}
