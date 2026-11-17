using System.Runtime.InteropServices;
using System.Security.Cryptography;
using DotLLM.Hip;
using DotLLM.Hip.DirectStorage;
using DotLLM.Hip.Interop;

namespace DotLLM.Tests.LinuxHipProbe;

static partial class Probe
{
    [LibraryImport("amdhip64", EntryPoint = "probe_mode")] private static partial void Mode(int value);
    [LibraryImport("amdhip64", EntryPoint = "probe_stat")] private static partial int Stat(int index);
    static int checks;
    static void Check(bool condition, string name) { if (!condition) throw new Exception(name); checks++; }
    static void Throws<T>(Action action) where T : Exception
    {
        try { action(); } catch (T) { checks++; return; }
        throw new Exception("Expected " + typeof(T).Name);
    }
    static void Clean() => Check(Stat(0) == Stat(1) && Stat(3) == 0 && Stat(4) == 0, "allocation/ABI balance");

    static void Main()
    {
        Check(OperatingSystem.IsLinux() && RuntimeInformation.ProcessArchitecture == Architecture.X64, "Linux x64");
        Check(Marshal.SizeOf<HipLinuxDirectIo.AmdFileHandle>() == 8, "handle layout");
        Check(Marshal.OffsetOf<HipLinuxDirectIo.AmdFileHandle>("FileDescriptor") == 0, "fd offset");
        Check(Marshal.SizeOf<HipLinuxDirectIo.FileStat>() == 256, "statx size");
        Check(Marshal.OffsetOf<HipLinuxDirectIo.FileStat>("Size") == 40, "statx size offset");
        Check(Marshal.OffsetOf<HipLinuxDirectIo.FileStat>("MemoryAlignment") == 152, "DIO memory offset");
        Check(Marshal.OffsetOf<HipLinuxDirectIo.FileStat>("OffsetAlignment") == 156, "DIO file offset");
        Check(Marshal.SizeOf<HipLinuxDirectIo.FileSystemStat>() == 120, "statfs size");
        foreach (var test in new (long File, long Offset, long Length, uint Memory, uint OffsetAlign, long Start, long Read, long Required, long Displacement, long Allocation)[] {
            (260,3,257,512,512,0,4096,260,3,8191), (16384,8192,8192,4096,4096,8192,8192,8192,0,12287),
            (1000000,65537,65536,65536,512,65536,131072,65537,1,196607), (4097,4096,1,0,0,4096,4096,1,0,8191) })
        {
            var p = HipLinuxDirectIo.Plan(test.File, test.Offset, test.Length, test.Memory, test.OffsetAlign);
            Check(p.FileOffset == test.Start && p.ReadBytes == test.Read && p.RequiredBytes == test.Required
                && p.TensorOffset == test.Displacement && p.AllocationBytes == test.Allocation, "exact plan");
        }
        foreach (var pair in new (uint M, uint O)[] { (3,512),(4096,511),(67108864,4096) })
            Throws<NotSupportedException>(() => HipLinuxDirectIo.Plan(100,0,100,pair.M,pair.O));
        Throws<ArgumentOutOfRangeException>(() => HipLinuxDirectIo.Plan(100,99,2,0,0));
        Throws<OverflowException>(() => HipLinuxDirectIo.Plan(long.MaxValue,long.MaxValue-1,1,0,0));
        Throws<OverflowException>(() => HipLinuxDirectIo.Plan(long.MaxValue,0,long.MaxValue,0,0));
        HipLinuxDirectIo.ValidateRead(0,0,260,4096,260);
        HipLinuxDirectIo.ValidateRead(0,0,4096,4096,260);
        foreach (ulong count in new ulong[] { 0,259,4097 })
            Throws<EndOfStreamException>(() => HipLinuxDirectIo.ValidateRead(0,0,count,4096,260));
        Throws<IOException>(() => HipLinuxDirectIo.ValidateRead(1,0,260,4096,260));
        Throws<IOException>(() => HipLinuxDirectIo.ValidateRead(0,-5,260,4096,260));

        string path = Path.Combine(Path.GetTempPath(), "dotllm-hip-é-" + Guid.NewGuid().ToString("N") + ".bin");
        byte[] data = new byte[32 * 1024 * 1024 + 257];
        new Random(1797).NextBytes(data);
        try
        {
            File.WriteAllBytes(path,data);
            byte[] hash = SHA256.HashData(data);
            using (var file = HipLinuxDirectIo.OpenFile(path,out var stat))
                Check(!file.IsInvalid && stat.Size == (ulong)data.Length, "actual libc UTF8 statx/fstatfs");
            using var context = HipContext.Create(0);
            Mode(7);
            Throws<PlatformNotSupportedException>(() => new HipDirectWeightStore(context));
            Clean(); Mode(0);
            using var store = new HipDirectWeightStore(context);
            using (var tail = store.Load(path,data.Length-257,257))
            {
                byte[] actual = new byte[257]; Marshal.Copy(tail.Pointer,actual,0,actual.Length);
                Check(actual.AsSpan().SequenceEqual(data.AsSpan(data.Length-257)), "unaligned EOF tail");
                Check(tail.ByteLength == 257 && tail.AllocatedBytes == 8191, "accounting exact tail");
            }
            Clean(); Mode(0);
            using (var multi = store.Load(path,3,data.Length-3))
            {
                Check(Stat(2) == 2, "32MiB chunk boundary");
#pragma warning disable IDISP016 // Deliberately verify independent ownership and idempotent disposal.
                store.Dispose();
                byte[] actual = new byte[data.Length-3]; Marshal.Copy(multi.Pointer,actual,0,actual.Length);
                Check(actual.AsSpan().SequenceEqual(data.AsSpan(3)), "displaced view outlives store");
                Throws<ObjectDisposedException>(() => store.Load(path,0,1));
                context.Dispose();
                multi.Dispose(); multi.Dispose();
#pragma warning restore IDISP016
                Throws<ObjectDisposedException>(() => { _ = multi.Pointer; });
            }
            Clean();
            using var context2 = HipContext.Create(0);
            using var store2 = new HipDirectWeightStore(context2);
            Throws<ArgumentOutOfRangeException>(() => store2.Load(path,data.Length-1,2)); Clean();
            Throws<IOException>(() => store2.Load(path+"missing",0,1)); Clean();
            for (int fault=1; fault<=8; fault++)
            {
                if (fault==7) continue;
                Mode(fault);
                if (fault==5) Throws<OutOfMemoryException>(() => store2.Load(path,3,data.Length-3));
                else if (fault==6) Throws<HipException>(() => store2.Load(path,3,data.Length-3));
                else Throws<IOException>(() => store2.Load(path,3,data.Length-3));
                Clean();
            }
            Mode(0);
            Check(SHA256.HashData(File.ReadAllBytes(path)).AsSpan().SequenceEqual(hash), "original file unchanged");
            Clean();
            Console.WriteLine($"PASS {checks} Linux managed ABI/lifecycle/fault checks; allocations={Stat(0)}, frees={Stat(1)}, badFree={Stat(3)}, badAbi={Stat(4)}. SIMULATED HIP allocations/read function, actual libc O_DIRECT pread, no GPU qualification.");
        }
        finally { File.Delete(path); }
    }
}
