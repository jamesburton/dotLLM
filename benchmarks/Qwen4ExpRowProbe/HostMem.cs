using System.Runtime.InteropServices;

/// <summary>Host physical-memory probe for the cap measurements (#820).</summary>
internal static class HostMem
{
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatusEx
    {
        public uint dwLength, dwMemoryLoad;
        public ulong ullTotalPhys, ullAvailPhys, ullTotalPageFile, ullAvailPageFile, ullTotalVirtual, ullAvailVirtual, ullAvailExtendedVirtual;
    }

    [DllImport("kernel32.dll")]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx s);

    /// <summary>Physical memory the OS reports as available (free + standby), bytes; -1 when unknown.</summary>
    public static long AvailableBytes()
    {
        var s = new MemoryStatusEx { dwLength = (uint)Marshal.SizeOf<MemoryStatusEx>() };
        return GlobalMemoryStatusEx(ref s) ? (long)s.ullAvailPhys : -1;
    }
}
