using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Does <c>VK_EXT_external_memory_host</c> accept the pages a GGUF is actually mapped
/// from? The existing import tests all pass, and on real hardware the import engaged on
/// <b>zero</b> tensors of every real model — because they import
/// <see cref="NativeMemory.AlignedAlloc"/> memory, which is private and read-write,
/// while <c>GgufFile</c> maps the weights with
/// <see cref="MemoryMappedFileAccess.Read"/>.
/// </summary>
/// <remarks>
/// <para>
/// This class imports the SAME BYTES through three different host mappings and reports
/// the driver's own verdict for each (<see cref="HostVisibleBuffer.LastImportFailureStage"/>
/// plus the raw <c>VkResult</c>), so the cause is a measurement rather than a guess:
/// </para>
/// <list type="number">
///   <item><b>Read-write anonymous memory</b> — what every pre-existing import test uses.
///   The control.</item>
///   <item><b>Read-only file mapping</b> — exactly what <c>GgufFile</c> does today
///   (<c>MemoryMappedFile.CreateFromFile(..., MemoryMappedFileAccess.Read)</c> +
///   a <c>Read</c> view accessor).</item>
///   <item><b>Copy-on-write file mapping</b> — the candidate fix: still no writes reach
///   the file, but the pages are writable, so a driver that probe-and-locks for write
///   may accept them.</item>
/// </list>
/// <para>
/// <b>These tests deliberately do not hard-assert a particular driver's behaviour.</b>
/// They assert the control works, and then record what the file-backed modes do, so the
/// matrix is in the test output of whatever machine runs them. The behavioural guard
/// that the import must actually engage on a real load lives in
/// <see cref="VulkanRealLoadHostImportTests"/>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanHostImportMmapAccessModeTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _output = output;

    private static string WritePayload(long bytes)
    {
        string path = Path.Combine(Path.GetTempPath(), $"dotllm-import-probe-{Guid.NewGuid():N}.bin");
        var buf = new byte[bytes];
        new Random(0x508).NextBytes(buf);
        File.WriteAllBytes(path, buf);
        return path;
    }

    private static string Verdict(HostVisibleBuffer? b)
        => b is not null
            ? "ACCEPTED"
            : $"REFUSED at {HostVisibleBuffer.LastImportFailureStage} " +
              $"(VkResult {HostVisibleBuffer.LastImportFailureCode})";

    /// <summary>
    /// The whole matrix in one test, so the three verdicts are directly comparable on the
    /// same device, same bytes, same size, same alignment — the only variable is how the
    /// host pages were obtained.
    /// </summary>
    [SkippableFact]
    public unsafe void ImportVerdict_ByHostMappingMode()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost,
            "Driver does not expose VK_EXT_external_memory_host on this host.");

        ulong alignment = Math.Max(4096UL, device.MinImportedHostPointerAlignment);
        long bytes = (long)alignment * 4;

        _output.WriteLine($"device type={device.PhysicalDeviceTypeValue} " +
                          $"minImportedHostPointerAlignment={alignment} size={bytes}");

        // ── 1. Control: private read-write memory (what every existing test uses).
        bool rwAccepted;
        string rwVerdict;
        void* anon = NativeMemory.AlignedAlloc((nuint)bytes, (nuint)alignment);
        try
        {
            new Span<byte>(anon, (int)bytes).Clear();
            using var b = HostVisibleBuffer.TryCreate(device, (nint)anon, bytes);
            rwAccepted = b is not null;
            rwVerdict = Verdict(b);
        }
        finally
        {
            NativeMemory.AlignedFree(anon);
        }
        _output.WriteLine($"  [1] anonymous read-write : {rwVerdict}");

        string path = WritePayload(bytes);
        try
        {
            // ── 2. Read-only file mapping — byte-for-byte what GgufFile.Open does.
            string roVerdict;
            bool roAccepted;
            {
                using var mmf = MemoryMappedFile.CreateFromFile(
                    path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
                using var acc = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
                byte* basePtr = null;
                acc.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
                try
                {
                    using var b = HostVisibleBuffer.TryCreate(device, (nint)(basePtr + acc.PointerOffset), bytes);
                    roAccepted = b is not null;
                    roVerdict = Verdict(b);
                }
                finally { acc.SafeMemoryMappedViewHandle.ReleasePointer(); }
            }
            _output.WriteLine($"  [2] mmap READ-ONLY       : {roVerdict}   <-- what GgufFile uses");

            // ── 3. Copy-on-write file mapping — writable pages, no writes reach the file.
            string cowVerdict;
            bool cowAccepted;
            {
                using var mmf = MemoryMappedFile.CreateFromFile(
                    path, FileMode.Open, null, 0, MemoryMappedFileAccess.CopyOnWrite);
                using var acc = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.CopyOnWrite);
                byte* basePtr = null;
                acc.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
                try
                {
                    using var b = HostVisibleBuffer.TryCreate(device, (nint)(basePtr + acc.PointerOffset), bytes);
                    cowAccepted = b is not null;
                    cowVerdict = Verdict(b);
                }
                finally { acc.SafeMemoryMappedViewHandle.ReleasePointer(); }
            }
            _output.WriteLine($"  [3] mmap COPY-ON-WRITE   : {cowVerdict}");

            _output.WriteLine(
                $"CONCLUSION: rw={(rwAccepted ? "ok" : "refused")} " +
                $"readonly={(roAccepted ? "ok" : "refused")} " +
                $"cow={(cowAccepted ? "ok" : "refused")}");
            if (rwAccepted && !roAccepted)
            {
                _output.WriteLine(
                    "  => PAGE PROTECTION IS THE DISCRIMINATOR. The zero-copy import cannot " +
                    "engage on any GGUF while GgufFile maps MemoryMappedFileAccess.Read." +
                    (cowAccepted
                        ? " CopyOnWrite is accepted and is the candidate fix (measure private "
                          + "working set before adopting -- CoW pages that get dirtied duplicate "
                          + "the mapping and defeat the purpose)."
                        : " CopyOnWrite is ALSO refused, so no read-only-safe mapping mode works "
                          + "on this driver and #438's release-after-upload is the whole answer on UMA."));
            }
            else if (rwAccepted && roAccepted)
            {
                _output.WriteLine(
                    "  => page protection is NOT the discriminator; the production refusal has " +
                    "another cause (size, allocation count, or the tensor's sub-page offset). " +
                    "Re-check with a production-sized import.");
            }

            // The control MUST work, or this whole probe says nothing.
            Assert.True(rwAccepted,
                $"Control import of private read-write memory failed: {rwVerdict}. " +
                "The probe cannot discriminate mapping modes if even the control is refused.");
        }
        finally
        {
            try { File.Delete(path); } catch (IOException) { /* best effort */ }
        }
    }

    /// <summary>
    /// What does <see cref="MemoryMappedFileAccess.CopyOnWrite"/> actually COST? It is only a
    /// fix for the read-only refusal if the pages stay shared with the page cache. If the
    /// driver pins them for write at import, every page breaks copy-on-write and the mapping
    /// is duplicated into private memory — which is precisely the second resident copy #508
    /// exists to remove, so the "fix" would buy nothing.
    /// </summary>
    /// <remarks>
    /// Reports Windows <i>commit charge</i> (<c>PrivateMemorySize64</c> — CoW copies land here)
    /// and working set across: baseline, after mapping, after reading every page, and after the
    /// import. A CoW break shows up as private commit growing by roughly the mapped size.
    /// </remarks>
    [SkippableFact]
    public unsafe void CopyOnWriteImport_PrivateCommitCost()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost,
            "Driver does not expose VK_EXT_external_memory_host on this host.");

        const long bytes = 256L * 1024 * 1024;
        string path = WritePayload(bytes);
        var proc = System.Diagnostics.Process.GetCurrentProcess();

        static long Mib(long b) => b / (1024 * 1024);
        long Commit() { proc.Refresh(); return proc.PrivateMemorySize64; }
        long Ws() { proc.Refresh(); return proc.WorkingSet64; }

        try
        {
            long commit0 = Commit(), ws0 = Ws();
            _output.WriteLine($"mapped size            : {Mib(bytes)} MiB");
            _output.WriteLine($"[0] baseline           : commit {Mib(commit0)} MiB, ws {Mib(ws0)} MiB");

            using var mmf = MemoryMappedFile.CreateFromFile(
                path, FileMode.Open, null, 0, MemoryMappedFileAccess.CopyOnWrite);
            using var acc = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.CopyOnWrite);
            byte* basePtr = null;
            acc.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            try
            {
                long commit1 = Commit();
                _output.WriteLine($"[1] after CoW map      : commit {Mib(commit1)} MiB (+{Mib(commit1 - commit0)}), ws {Mib(Ws())} MiB");

                // Read every page. Reads must NOT break copy-on-write.
                byte* p = basePtr + acc.PointerOffset;
                long sink = 0;
                for (long off = 0; off < bytes; off += 4096) sink += p[off];
                Assert.True(sink >= 0);
                long commit2 = Commit();
                _output.WriteLine($"[2] after reading all  : commit {Mib(commit2)} MiB (+{Mib(commit2 - commit1)}), ws {Mib(Ws())} MiB");

                using var buf = HostVisibleBuffer.TryCreate(device, (nint)p, bytes);
                long commit3 = Commit();
                _output.WriteLine($"[3] after import       : {Verdict(buf)}");
                _output.WriteLine($"                         commit {Mib(commit3)} MiB (+{Mib(commit3 - commit2)}), ws {Mib(Ws())} MiB");

                Skip.If(buf is null, "CopyOnWrite import refused on this device — nothing to cost.");

                long brokenPages = commit3 - commit2;
                _output.WriteLine(brokenPages > bytes / 2
                    ? $"  => IMPORT BREAKS COPY-ON-WRITE: +{Mib(brokenPages)} MiB private for a "
                      + $"{Mib(bytes)} MiB mapping. CoW duplicates the weights and is NOT a fix."
                    : $"  => import added {Mib(brokenPages)} MiB commit for a {Mib(bytes)} MiB mapping. Commit cannot "
                      + "see a copy-on-write break (the view's commit is charged at map time): see "
                      + "Import_PrivateWorkingSetCost_ByMapping for the physical cost (#1190).");
            }
            finally { acc.SafeMemoryMappedViewHandle.ReleasePointer(); }
        }
        finally
        {
            try { File.Delete(path); } catch (IOException) { /* best effort */ }
        }
    }

    /// <summary>
    /// The physical cost of an import, per host mapping (#1190): private working set (pages only this process holds)
    /// before and after the import of the same 256 MiB, already read. Commit charge cannot see a copy-on-write break —
    /// the view's commit is charged at map time — so <see cref="CopyOnWriteImport_PrivateCommitCost"/> reports "0 MiB"
    /// whether or not the driver privatised every page.
    /// </summary>
    [SkippableFact]
    public unsafe void Import_PrivateWorkingSetCost_ByMapping()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        Skip.IfNot(OperatingSystem.IsWindows(), "Private working set probe is Windows-only.");
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost, "Driver does not expose VK_EXT_external_memory_host on this host.");

        const long bytes = 256L * 1024 * 1024;
        string path = WritePayload(bytes);
        try
        {
            void Probe(string label, MemoryMappedFileAccess mmfAccess, MemoryMappedFileAccess viewAccess, FileAccess fileAccess)
            {
                File.SetLastWriteTimeUtc(path, new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc));
                long modified0 = ModifiedFileBytes();
                ProbeOnce(label, mmfAccess, viewAccess, fileAccess);
                _output.WriteLine($"{"",-34}  modified (file-backed) page list after unmap: {(ModifiedFileBytes() - modified0) / (1024 * 1024):+#;-#;0} MiB");
                // A driver that locks the pages for write marks them dirty on release; on a shared writable mapping
                // that is a write-back of the whole range to the checkpoint (same bytes, but real disk writes).
                _output.WriteLine($"{"",-34}  file mtime after unmap: {(File.GetLastWriteTimeUtc(path).Year == 2020 ? "unchanged" : "CHANGED")}");
            }

            void ProbeOnce(string label, MemoryMappedFileAccess mmfAccess, MemoryMappedFileAccess viewAccess, FileAccess fileAccess)
            {
                using var fs = new FileStream(path, FileMode.Open, fileAccess, FileShare.ReadWrite);
                using var mmf = MemoryMappedFile.CreateFromFile(fs, null, 0, mmfAccess, HandleInheritability.None, leaveOpen: false);
                using var acc = mmf.CreateViewAccessor(0, bytes, viewAccess);
                byte* basePtr = null;
                acc.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
                try
                {
                    byte* p = basePtr + acc.PointerOffset;
                    long sink = 0;
                    for (long off = 0; off < bytes; off += 4096) sink += p[off];
                    Assert.True(sink >= 0);
                    long pws0 = PrivateWorkingSet();
                    using var buf = HostVisibleBuffer.TryCreate(device, (nint)p, bytes);
                    long pws1 = PrivateWorkingSet();
                    _output.WriteLine($"{label,-34}: {Verdict(buf),-50} private WS +{(pws1 - pws0) / (1024 * 1024)} MiB of {bytes / (1024 * 1024)}");
                }
                finally { acc.SafeMemoryMappedViewHandle.ReleasePointer(); }
            }

            Probe("CoW mapping, CoW view", MemoryMappedFileAccess.CopyOnWrite, MemoryMappedFileAccess.CopyOnWrite, FileAccess.Read);
            Probe("read-only mapping, CoW view", MemoryMappedFileAccess.Read, MemoryMappedFileAccess.CopyOnWrite, FileAccess.Read);
            Probe("read-write mapping, read view", MemoryMappedFileAccess.ReadWrite, MemoryMappedFileAccess.Read, FileAccess.ReadWrite);
            Probe("read-write mapping, read-write view", MemoryMappedFileAccess.ReadWrite, MemoryMappedFileAccess.ReadWrite, FileAccess.ReadWrite);
        }
        finally
        {
            try { File.Delete(path); } catch (IOException) { /* best effort */ }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemCountersEx2
    {
        public uint Cb, PageFaultCount;
        public nuint PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage, QuotaPeakNonPagedPoolUsage,
            QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage, PrivateUsage, PrivateWorkingSetSize;
        public ulong SharedCommitUsage;
    }

    [DllImport("psapi.dll")]
    private static extern bool GetProcessMemoryInfo(nint process, ref MemCountersEx2 counters, uint cb);

    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct MemoryListInfo
    {
        public nuint ZeroPageCount, FreePageCount, ModifiedPageCount, ModifiedNoWritePageCount, BadPageCount;
        public fixed ulong PageCountByPriority[8];
        public fixed ulong RepurposedPagesByPriority[8];
        public nuint ModifiedPageCountPageFile;
    }

    [DllImport("ntdll.dll")]
    private static extern int NtQuerySystemInformation(int infoClass, ref MemoryListInfo info, int length, out int returnLength);

    /// <summary>System-wide modified pages bound for a mapped file (not the pagefile), in bytes; -1 when unavailable.</summary>
    private static unsafe long ModifiedFileBytes()
    {
        var info = default(MemoryListInfo);
        return NtQuerySystemInformation(80 /* SystemMemoryListInformation */, ref info, sizeof(MemoryListInfo), out _) == 0
            ? (long)(info.ModifiedPageCount - info.ModifiedPageCountPageFile) * 4096
            : -1;
    }

    private static long PrivateWorkingSet()
    {
        var c = new MemCountersEx2 { Cb = (uint)Marshal.SizeOf<MemCountersEx2>() };
        return GetProcessMemoryInfo(System.Diagnostics.Process.GetCurrentProcess().Handle, ref c, c.Cb) ? (long)c.PrivateWorkingSetSize : -1;
    }

    /// <summary>
    /// Production tensors are multi-MB and start at arbitrary sub-page offsets inside the
    /// mapping. Repeats the read-only probe at a realistic size and a non-zero offset so a
    /// refusal that only appears at scale cannot hide behind the 16 KiB case above.
    /// </summary>
    [SkippableFact]
    public unsafe void ImportVerdict_ReadOnlyMmap_AtProductionSizeAndOffset()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost,
            "Driver does not expose VK_EXT_external_memory_host on this host.");

        const long bytes = 64L * 1024 * 1024;
        const int subPageOffset = 544;   // the sort of offset a GGUF tensor actually lands on
        string path = WritePayload(bytes + 4096);
        try
        {
            using var mmf = MemoryMappedFile.CreateFromFile(
                path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            using var acc = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* basePtr = null;
            acc.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            try
            {
                using var b = HostVisibleBuffer.TryCreate(
                    device, (nint)(basePtr + acc.PointerOffset + subPageOffset), bytes);
                _output.WriteLine($"64 MiB read-only mmap at +{subPageOffset}: {Verdict(b)}");
            }
            finally { acc.SafeMemoryMappedViewHandle.ReleasePointer(); }
        }
        finally
        {
            try { File.Delete(path); } catch (IOException) { /* best effort */ }
        }
    }
}
