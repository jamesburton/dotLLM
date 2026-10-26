using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Interop;

/// <summary>
/// Issue #1080: the in-place growable device buffer's bookkeeping, through a fake virtual-memory API (no GPU). A growth maps
/// only the added bytes behind the same pointer; a growth past the reservation moves the same physical chunks to a new range;
/// a failure undoes itself; disposal returns everything.
/// </summary>
public sealed class CudaGrowableDeviceBufferTests
{
    private const long G = 2L << 20;   // the RTX 3060's minimum granularity (2 MiB)

    [Fact]
    public void Growth_MapsOnlyTheAddedBytes_BehindTheSamePointer()
    {
        var vm = new FakeVm();
        using var buffer = new CudaGrowableDeviceBuffer(vm, 10 * G);
        nint pointer = buffer.Pointer;

        Assert.False(buffer.EnsureCapacity(3 * G - 5));
        Assert.Equal(3 * G, buffer.MappedBytes);
        Assert.Equal(2 * G, buffer.GrowthBytes(5 * G));
        Assert.False(buffer.EnsureCapacity(5 * G));

        Assert.Equal(pointer, buffer.Pointer);
        Assert.Equal(5 * G, buffer.MappedBytes);
        Assert.Equal(5 * G, vm.LivePhysicalBytes);   // never old + new at once
        Assert.Equal([(pointer, 3 * G), (pointer + (nint)(3 * G), 2 * G)], vm.Mappings.Select(m => (m.Address, m.Bytes)));
        Assert.Equal(0, buffer.GrowthBytes(5 * G));
        Assert.False(buffer.EnsureCapacity(4 * G));   // never shrinks
        Assert.Equal(2, buffer.ChunkCount);
    }

    [Fact]
    public void GrowthPastTheReservation_RemapsTheSameChunks_AndFreesTheOldRange()
    {
        var vm = new FakeVm();
        using var buffer = new CudaGrowableDeviceBuffer(vm, 2 * G);
        buffer.EnsureCapacity(2 * G);
        ulong first = vm.Mappings.Single().Handle;
        nint oldBase = buffer.Pointer;

        Assert.True(buffer.EnsureCapacity(3 * G));

        Assert.NotEqual(oldBase, buffer.Pointer);
        Assert.Equal(3 * G, buffer.ReservedBytes);
        Assert.Equal(3 * G, vm.LivePhysicalBytes);
        Assert.Equal([(buffer.Pointer, 2 * G, first), (buffer.Pointer + (nint)(2 * G), G, vm.Mappings[1].Handle)],
            vm.Mappings.Select(m => (m.Address, m.Bytes, m.Handle)));
        Assert.DoesNotContain(oldBase, vm.Reservations.Keys);
    }

    [Fact]
    public void FailedMap_ReleasesTheNewChunk_AndLeavesTheBufferAsItWas()
    {
        var vm = new FakeVm();
        using var buffer = new CudaGrowableDeviceBuffer(vm, 10 * G);
        buffer.EnsureCapacity(G);
        vm.FailMapAfter = 0;

        Assert.Throws<InvalidOperationException>(() => buffer.EnsureCapacity(4 * G));

        Assert.Equal(G, buffer.MappedBytes);
        Assert.Equal(G, vm.LivePhysicalBytes);
        Assert.Single(vm.Mappings);
        vm.FailMapAfter = null;
        buffer.EnsureCapacity(4 * G);   // and it can still grow
        Assert.Equal(4 * G, vm.LivePhysicalBytes);
    }

    [Fact]
    public void FailedRereservation_KeepsTheOldRangeInUse_AndFreesTheNewOne()
    {
        var vm = new FakeVm();
        using var buffer = new CudaGrowableDeviceBuffer(vm, 2 * G);
        buffer.EnsureCapacity(G);
        buffer.EnsureCapacity(2 * G);
        nint oldBase = buffer.Pointer;
        vm.FailMapAfter = 1;   // the second chunk's remap fails

        Assert.Throws<InvalidOperationException>(() => buffer.EnsureCapacity(3 * G));

        Assert.Equal(oldBase, buffer.Pointer);
        Assert.Equal(2 * G, buffer.MappedBytes);
        Assert.Equal([oldBase], vm.Reservations.Keys);
        Assert.All(vm.Mappings, m => Assert.True(m.Address >= oldBase && m.Address < oldBase + (nint)(2 * G)));
        Assert.Equal(2, vm.Mappings.Count);
    }

    [Fact]
    public void Dispose_UnmapsReleasesAndFreesEverything()
    {
        var vm = new FakeVm();
        var buffer = new CudaGrowableDeviceBuffer(vm, 4 * G);
        buffer.EnsureCapacity(G);
        buffer.EnsureCapacity(3 * G);
        buffer.Dispose();
        buffer.Dispose();   // idempotent

        Assert.Empty(vm.Mappings);
        Assert.Equal(0, vm.LivePhysicalBytes);
        Assert.Empty(vm.Reservations);
        Assert.Throws<ObjectDisposedException>(() => buffer.EnsureCapacity(G));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, G)]
    [InlineData(G, G)]
    [InlineData(G + 1, 2 * G)]
    public void RoundUp_GoesToTheNextGranule(long bytes, long expected)
        => Assert.Equal(expected, CudaGrowableDeviceBuffer.RoundUp(bytes, G));

    /// <summary>Tracks reservations, physical allocations and mappings, and checks every call is legal.</summary>
    private sealed class FakeVm : ICudaVirtualMemory
    {
        private nint _nextAddress = 0x1000_0000;
        private ulong _nextHandle = 1;
        private readonly Dictionary<ulong, long> _physical = [];
        private readonly HashSet<ulong> _released = [];

        public long Granularity => G;
        public Dictionary<nint, long> Reservations { get; } = [];
        public List<(nint Address, long Bytes, ulong Handle)> Mappings { get; } = [];
        public int? FailMapAfter { get; set; }

        /// <summary>Physical bytes still alive: not released, or released but still mapped somewhere.</summary>
        public long LivePhysicalBytes => _physical.Where(p => !_released.Contains(p.Key) || Mappings.Any(m => m.Handle == p.Key))
            .Sum(p => p.Value);

        public nint Reserve(long bytes)
        {
            Assert.Equal(0, bytes % G);
            nint address = _nextAddress;
            _nextAddress += (nint)(bytes + 64 * G);
            Reservations.Add(address, bytes);
            return address;
        }

        public void FreeReservation(nint address, long bytes)
        {
            Assert.Equal(bytes, Reservations[address]);
            Assert.DoesNotContain(Mappings, m => m.Address >= address && m.Address < address + (nint)bytes);
            Reservations.Remove(address);
        }

        public ulong CreatePhysical(long bytes)
        {
            Assert.Equal(0, bytes % G);
            ulong handle = _nextHandle++;
            _physical.Add(handle, bytes);
            return handle;
        }

        public void ReleasePhysical(ulong handle) => Assert.True(_released.Add(handle));

        public void Map(nint address, long bytes, ulong handle)
        {
            if (FailMapAfter is int left)
            {
                if (left <= 0) throw new InvalidOperationException("injected map failure");
                FailMapAfter = left - 1;
            }
            Assert.Equal(_physical[handle], bytes);
            Assert.Contains(Reservations, r => address >= r.Key && address + (nint)bytes <= r.Key + (nint)r.Value);
            Assert.DoesNotContain(Mappings, m => address < m.Address + (nint)m.Bytes && m.Address < address + (nint)bytes);
            Mappings.Add((address, bytes, handle));
        }

        public void Unmap(nint address, long bytes)
            => Assert.Equal(1, Mappings.RemoveAll(m => m.Address == address && m.Bytes == bytes));
    }
}
