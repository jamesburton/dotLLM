// IoProbe (#937, parent #825): raw NVMe -> GPU-visible-RAM throughput / latency / page-cache-bloat probe for expert tiering.
// Diagnostics tool, not shipped. Windows only.
//
//   io  <file> <un|buf> <blockKiB> <qd> <totalMiB> <startMiB> <target> [regionMiB=1024] [rand] [passes=1]
//        un  = FILE_FLAG_NO_BUFFERING|OVERLAPPED (+IOCP), bypasses the page cache; buf = OVERLAPPED but cached.
//        target = ring (QD blocks, data discarded) | valloc (VirtualAlloc region) | vkhost (Vulkan HOST_VISIBLE, mapped)
//                 | vkdl (Vulkan DEVICE_LOCAL, mapped when host-visible) | import (VirtualAlloc region wrapped by VK_EXT_external_memory_host)
//        rand = random 4 KiB-aligned offsets inside [start, start+total) instead of sequential.
//   mm  <file> <totalMiB> <startMiB> <threads> <target> [regionMiB=1024] [touch]
//        mmap read-only view + N threads copying 2 MiB chunks to the target (touch = fault pages only, no copy).
//   cache  (prints the system-cache / available-RAM counters)
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;

unsafe
{
    if (args.Length == 0) { Console.WriteLine("usage: see Program.cs header"); return; }
    switch (args[0])
    {
        case "cache": Mem.Print("now"); return;
        case "io": Io.Run(args); return;
        case "mm": MmapArm.Run(args); return;
        case "gpucopy": GpuCopy.Run(args); return;
        default: Console.WriteLine("unknown command"); return;
    }
}

static unsafe class Mem
{
    [StructLayout(LayoutKind.Sequential)]
    struct PerfInfo
    {
        public uint cb; public nuint CommitTotal, CommitLimit, CommitPeak, PhysicalTotal, PhysicalAvailable, SystemCache,
            KernelTotal, KernelPaged, KernelNonpaged, PageSize; public uint HandleCount, ProcessCount, ThreadCount;
    }
    [DllImport("psapi.dll")] static extern bool GetPerformanceInfo(out PerfInfo pi, uint cb);

    public static (double CacheGiB, double AvailGiB) Read()
    {
        GetPerformanceInfo(out var p, (uint)sizeof(PerfInfo));
        double ps = (double)p.PageSize / (1 << 30);
        return (p.SystemCache * ps, p.PhysicalAvailable * ps);
    }
    public static void Print(string tag)
    {
        var (c, a) = Read();
        Console.WriteLine($"[mem] {tag}: systemCache {c:F2} GiB, availablePhys {a:F2} GiB");
    }
}

sealed unsafe class Target : IDisposable
{
    public nint Base; public long Size; public string Kind = "";
    VulkanDevice? _shared; VulkanDevice? _dev; VulkanDevice.Buffer? _buf; VulkanDevice.Buffer? _imp; bool _virt;

    [DllImport("kernel32.dll")] static extern nint VirtualAlloc(nint a, nuint sz, uint type, uint prot);
    [DllImport("kernel32.dll")] static extern bool VirtualFree(nint a, nuint sz, uint type);

    public VulkanDevice.Buffer? Buf => _imp ?? _buf;
    public static Target Create(string kind, long size, VulkanDevice? shared = null)
    {
        var t = new Target { Kind = kind, Size = size, _shared = shared };
        switch (kind)
        {
            case "ring":
            case "valloc":
            case "import":
                t.Base = VirtualAlloc(0, (nuint)size, 0x3000, 0x04); t._virt = true;
                if (t.Base == 0) throw new OutOfMemoryException("VirtualAlloc");
                if (kind == "import")
                {
                    t._dev = shared ?? VulkanDevice.Create();
                    t._imp = t._dev.TryWrapHostVisible(t.Base, size);
                    Console.WriteLine($"[target] import: {(t._imp is null ? "REFUSED" : "OK")} ({size >> 20} MiB)");
                    if (t._imp is null) throw new InvalidOperationException("import refused");
                }
                break;
            case "vkhost":
                t._dev = shared ?? VulkanDevice.Create();
                t._buf = t._dev.Allocate(size);
                t.Base = t._dev.MapMemoryWithRetry(t._buf.Memory, 0, (ulong)size, "probe");
                break;
            case "vkdl":
                t._dev = shared ?? VulkanDevice.Create();
                t._buf = t._dev.AllocateDeviceLocal(size);
                Console.WriteLine($"[target] vkdl host-visible={t._buf.IsHostVisible}");
                if (!t._buf.IsHostVisible) throw new InvalidOperationException("device-local not host visible");
                t.Base = t._dev.MapMemoryWithRetry(t._buf.Memory, 0, (ulong)size, "probe");
                break;
            default: throw new ArgumentException(kind);
        }
        return t;
    }

    public void Dispose()
    {
        _imp?.Dispose(); _buf?.Dispose();
        if (_virt && Base != 0) VirtualFree(Base, 0, 0x8000);
        if (_shared is null) _dev?.Dispose();
    }
}

static unsafe class Io
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern nint CreateFileW(string name, uint access, uint share, nint sec, uint disp, uint flags, nint tmpl);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool ReadFile(nint h, void* buf, uint n, out uint read, void* ov);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern nint CreateIoCompletionPort(nint file, nint port, nuint key, uint threads);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetQueuedCompletionStatus(nint port, out uint bytes, out nuint key, out void* ov, uint ms);
    [DllImport("kernel32.dll")] static extern bool GetFileSizeEx(nint h, out long size);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(nint h);

    [StructLayout(LayoutKind.Sequential)]
    struct Ov { public nuint Internal, InternalHigh; public uint Offset, OffsetHigh; public nint hEvent; }

    // One stream = one handle + one IOCP + QD outstanding requests over [start, start+total) into dst..dst+region.
    static (long Bytes, List<double> Lat) Stream(string path, bool unbuf, long blk, int qd, long total, long start, byte* dst, long region,
        bool ring, bool rand, int seed)
    {
        uint flags = 0x40000000 | (unbuf ? 0x20000000u : 0u); // OVERLAPPED | NO_BUFFERING
        nint h = CreateFileW(path, 0x80000000, 1, 0, 3, flags, 0);
        if (h == -1) throw new IOException("open " + Marshal.GetLastWin32Error());
        nint port = CreateIoCompletionPort(h, 0, 0, 1);
        var ovs = (Ov*)NativeMemory.AllocZeroed((nuint)(sizeof(Ov) * qd));
        var t0s = new long[qd]; var lat = new List<double>(1 << 14);
        long nreq = total / blk; var rng = new Random(seed);
        long spanBytes = long.TryParse(Environment.GetEnvironmentVariable("IO_SPAN_MIB"), out long sm) ? sm << 20 : total;
        long span4k = (spanBytes - blk) / 4096;
        long issued = 0, done = 0, bytes = 0;
        void Issue(int slot)
        {
            long off = rand ? start + rng.NextInt64(span4k + 1) * 4096 : start + issued * blk;
            long d = ring ? (long)slot * blk : (issued * blk) % region;
            ovs[slot] = default; ovs[slot].Offset = (uint)(off & 0xffffffff); ovs[slot].OffsetHigh = (uint)(off >> 32);
            t0s[slot] = Stopwatch.GetTimestamp(); issued++;
            if (!ReadFile(h, dst + d, (uint)blk, out _, &ovs[slot]))
            {
                int e = Marshal.GetLastWin32Error();
                if (e != 997) throw new IOException("ReadFile err " + e + " off " + off);
            }
        }
        int init = (int)Math.Min(qd, nreq);
        for (int i = 0; i < init; i++) Issue(i);
        while (done < nreq)
        {
            if (!GetQueuedCompletionStatus(port, out uint nb, out _, out void* pov, uint.MaxValue) && pov == null)
                throw new IOException("gqcs " + Marshal.GetLastWin32Error());
            int slot = (int)((Ov*)pov - ovs);
            lat.Add((Stopwatch.GetTimestamp() - t0s[slot]) * 1000.0 / Stopwatch.Frequency);
            bytes += nb; done++;
            if (issued < nreq) Issue(slot);
        }
        CloseHandle(h); NativeMemory.Free(ovs);
        return (bytes, lat);
    }

    public static void Run(string[] a)
    {
        string path = a[1]; bool unbuf = a[2] == "un"; long blk = long.Parse(a[3]) << 10; int qd = int.Parse(a[4]);
        long total = long.Parse(a[5]) << 20, start = long.Parse(a[6]) << 20; string tk = a[7];
        long region = a.Length > 8 ? long.Parse(a[8]) << 20 : 1L << 30;
        bool rand = a.Length > 9 && a[9] == "rand"; int passes = a.Length > 10 ? int.Parse(a[10]) : 1;
        int streams = int.TryParse(Environment.GetEnvironmentVariable("IO_STREAMS"), out int sv) ? sv : 1;
        bool ring = tk == "ring";
        if (ring) region = blk * qd * streams;
        region = Math.Max(region, blk * qd * streams);
        nint h0 = CreateFileW(path, 0x80000000, 1, 0, 3, 0, 0); GetFileSizeEx(h0, out long fsz); CloseHandle(h0);
        long spanChk = long.TryParse(Environment.GetEnvironmentVariable("IO_SPAN_MIB"), out long sm2) ? sm2 << 20 : total;
        if (start + spanChk > fsz) { Console.WriteLine("span past EOF"); return; }
        using var tgt = Target.Create(tk, region);
        long perStream = total / streams / blk * blk, regionPer = region / streams / 4096 * 4096;

        Mem.Print("before");
        var cpu0 = Process.GetCurrentProcess().TotalProcessorTime;
        for (int pass = 0; pass < passes; pass++)
        {
            var sw = Stopwatch.StartNew();
            var res = new (long Bytes, List<double> Lat)[streams]; var th = new Thread[streams];
            for (int i = 0; i < streams; i++)
            {
                int k = i;
                th[k] = new Thread(() => res[k] = Stream(path, unbuf, blk, qd, perStream, start + (long)k * perStream + (Environment.GetEnvironmentVariable("IO_ADVANCE") == "1" ? (long)pass * perStream * streams : 0),
                    (byte*)tgt.Base + (long)k * regionPer, regionPer, ring, rand, 1234 + k));
                th[k].Start();
            }
            foreach (var t in th) t.Join();
            double s = sw.Elapsed.TotalSeconds; long bytes = res.Sum(r => r.Bytes);
            var lat = res.SelectMany(r => r.Lat).OrderBy(x => x).ToList();
            double P(double q) => lat[(int)Math.Min(lat.Count - 1, q * lat.Count)];
            var cpu = Process.GetCurrentProcess().TotalProcessorTime - cpu0; cpu0 = Process.GetCurrentProcess().TotalProcessorTime;
            Console.WriteLine($"[io] {a[2]} {(rand ? "rand" : "seq ")} blk={blk >> 10}KiB qd={qd}x{streams} tgt={tk} pass{pass}: {bytes / s / 1e9:F2} GB/s, {lat.Count / s:F0} req/s, lat ms p50={P(.5):F2} p99={P(.99):F2} max={lat[^1]:F2}, cpu={cpu.TotalSeconds / s:F2} cores");
        }
        Mem.Print("after ");
    }
}

static unsafe class MmapArm
{
    public static void Run(string[] a)
    {
        string path = a[1]; long total = long.Parse(a[2]) << 20, start = long.Parse(a[3]) << 20; int threads = int.Parse(a[4]);
        string tk = a[5]; long region = a.Length > 6 ? long.Parse(a[6]) << 20 : 1L << 30; bool touch = a.Length > 7 && a[7] == "touch";
        const long Chunk = 2 << 20;
        using var tgt = Target.Create(tk, region);
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* p = null; view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        p += view.PointerOffset;
        long nchunks = total / Chunk; long next = -1;
        Mem.Print("before");
        var cpu0 = Process.GetCurrentProcess().TotalProcessorTime; var sw = Stopwatch.StartNew();
        var ts = new Thread[threads]; long sink = 0;
        for (int t = 0; t < threads; t++)
        {
            ts[t] = new Thread(() =>
            {
                long local = 0;
                while (true)
                {
                    long c = Interlocked.Increment(ref next); if (c >= nchunks) break;
                    byte* src = p + start + c * Chunk;
                    if (touch) { for (long o = 0; o < Chunk; o += 4096) local += src[o]; }
                    else Buffer.MemoryCopy(src, (byte*)tgt.Base + (c * Chunk) % (region - Chunk + 1) / 4096 * 4096, Chunk, Chunk);
                }
                Interlocked.Add(ref sink, local);
            });
            ts[t].Start();
        }
        foreach (var t in ts) t.Join();
        double s = sw.Elapsed.TotalSeconds; var cpu = Process.GetCurrentProcess().TotalProcessorTime - cpu0;
        Console.WriteLine($"[mm] threads={threads} tgt={tk} {(touch ? "touch" : "copy")}: {nchunks * Chunk / s / 1e9:F2} GB/s, cpu={cpu.TotalSeconds / s:F2} cores");
        Mem.Print("after ");
        view.SafeMemoryMappedViewHandle.ReleasePointer();
    }
}

static unsafe class GpuCopy
{
    // gpucopy <sizeMiB>: GPU vkCmdCopyBuffer read bandwidth FROM each memory kind into a device-local buffer (proxy for how fast a shader can
    // stream a slot pool living in that memory). Source is filled first so pages are resident.
    public static void Run(string[] a)
    {
        long size = long.Parse(a[1]) << 20;
        using var dev = VulkanDevice.Create();
        using var dst = dev.AllocateDeviceLocal(size);
        foreach (var kind in new[] { "vkdl", "vkhost", "import" })
        {
            try
            {
                using var t = Target.Create(kind, size, dev);
                new Span<byte>((void*)t.Base, (int)Math.Min(size, int.MaxValue)).Fill(7);
                if (size > int.MaxValue) new Span<byte>((byte*)t.Base + int.MaxValue, (int)(size - int.MaxValue)).Fill(7);
                var src = t.Buf!;
                var times = new List<double>();
                for (int i = 0; i < 6; i++)
                {
                    var sw = Stopwatch.StartNew();
                    dev.CopyBufferSynchronous(src, dst, (ulong)size);
                    times.Add(sw.Elapsed.TotalSeconds);
                }
                times.RemoveAt(0); times.Sort();
                Console.WriteLine($"[gpucopy] {kind}: read {size / times[times.Count / 2] / 1e9:F1} GB/s (median of 5; best {size / times[0] / 1e9:F1})");
            }
            catch (Exception e) { Console.WriteLine($"[gpucopy] {kind}: {e.GetType().Name} {e.Message}"); }
        }
    }
}
