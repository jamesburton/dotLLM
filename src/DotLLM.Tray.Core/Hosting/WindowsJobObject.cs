using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace DotLLM.Tray.Hosting;

/// <summary>
/// A Win32 job object with <c>JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE</c>, used to guarantee that the
/// tray never orphans a server process.
/// </summary>
/// <remarks>
/// <para>
/// <b>The tray puts itself in the job, not its children.</b> Children started with
/// <see cref="Process"/> inherit their parent's job membership automatically, so a single
/// assignment at startup covers every server the tray ever launches — and, crucially, leaves no
/// window between <c>CreateProcess</c> returning and an <c>AssignProcessToJobObject</c> call in
/// which a child exists outside the job. Closing that window per-child would need
/// <c>CREATE_SUSPENDED</c>, which <see cref="ProcessStartInfo"/> cannot express.
/// </para>
/// <para>
/// The tray holds the only handle. When the tray exits — cleanly, by crash, or by
/// <c>taskkill /F</c> — the kernel closes the handle, the job's last handle goes away, and every
/// remaining process in it is terminated. That is the orphan guarantee, and it is enforced by
/// Windows rather than by any code path the tray has to remember to run.
/// </para>
/// <para>
/// Nesting is fine: since Windows 8 a process can belong to several jobs, so running the tray
/// under an outer job (a CI harness, a terminal's job, Visual Studio) does not make this fail.
/// </para>
/// </remarks>
[SupportedOSPlatform("windows")]
public sealed partial class WindowsJobObject : IDisposable
{
    private const int JobObjectExtendedLimitInformation = 9;
    private const uint JobObjectLimitKillOnJobClose = 0x00002000;

    private nint _handle;

    private WindowsJobObject(nint handle) => _handle = handle;

    /// <summary>Whether the job was created and the current process assigned to it.</summary>
    public bool IsActive => _handle != 0;

    /// <summary>
    /// Creates a kill-on-close job and puts the current process in it. Returns null when the OS
    /// refuses, which the caller must treat as "orphan protection is unavailable", not as fatal.
    /// </summary>
    public static WindowsJobObject? AssignCurrentProcess()
    {
        var handle = CreateJobObjectW(0, null);
        if (handle == 0)
            return null;

        var limits = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        limits.BasicLimitInformation.LimitFlags = JobObjectLimitKillOnJobClose;

        var size = Marshal.SizeOf<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
        var buffer = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(limits, buffer, fDeleteOld: false);
            if (!SetInformationJobObject(handle, JobObjectExtendedLimitInformation, buffer, (uint)size))
            {
                CloseHandle(handle);
                return null;
            }
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }

        if (!AssignProcessToJobObject(handle, GetCurrentProcess()))
        {
            CloseHandle(handle);
            return null;
        }

        return new WindowsJobObject(handle);
    }

    /// <summary>
    /// Releases the job handle, terminating every process still in it.
    /// </summary>
    /// <remarks>
    /// Calling this is an optimization, not a requirement — process exit closes the handle either
    /// way. It exists so a clean shutdown reaps the server promptly instead of leaving it alive
    /// for the moments between the last window closing and the process record going away.
    /// </remarks>
    public void Dispose()
    {
        var handle = Interlocked.Exchange(ref _handle, 0);
        if (handle != 0)
            CloseHandle(handle);
    }

    [LibraryImport("kernel32.dll", SetLastError = true, StringMarshalling = StringMarshalling.Utf16)]
    private static partial nint CreateJobObjectW(nint securityAttributes, string? name);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool SetInformationJobObject(
        nint job, int infoClass, nint info, uint infoLength);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool AssignProcessToJobObject(nint job, nint process);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool CloseHandle(nint handle);

    [LibraryImport("kernel32.dll")]
    private static partial nint GetCurrentProcess();

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public nuint MinimumWorkingSetSize;
        public nuint MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public nuint Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public nuint ProcessMemoryLimit;
        public nuint JobMemoryLimit;
        public nuint PeakProcessMemoryUsed;
        public nuint PeakJobMemoryUsed;
    }
}
