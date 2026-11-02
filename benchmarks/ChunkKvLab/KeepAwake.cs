using System.Runtime.InteropServices;

namespace ChunkKvLab;

/// <summary>
/// Keeps a long lab run going on a Modern Standby (S0ix) Windows box. G1 lost ~74 min (and later ~36 min) when the box entered
/// standby mid-run and the Desktop Activity Moderator suspended the process; a thread-level <c>ES_SYSTEM_REQUIRED</c> request in
/// the driving script did not stop it. This holds, for the life of the process: a <c>PowerRequestExecutionRequired</c> request (the
/// process keeps running even if the system enters standby) and <c>PowerRequestSystemRequired</c> plus
/// <c>PowerRequestDisplayRequired</c> (the idle timeout does not turn the display off, which is what starts Modern Standby).
/// No-op off Windows or if the requests cannot be created; the run proceeds either way.
/// </summary>
internal sealed class KeepAwake : IDisposable
{
    private const int PowerRequestDisplayRequired = 0;
    private const int PowerRequestSystemRequired = 1;
    private const int PowerRequestExecutionRequired = 3;
    private const uint PowerRequestContextVersion = 0;
    private const uint PowerRequestContextSimpleString = 0x1;

    private readonly nint _request;
    private readonly int[] _types = [];

    private KeepAwake(nint request, int[] types)
    {
        _request = request;
        _types = types;
    }

    /// <summary>Creates and sets the power requests; <see cref="Description"/> says what is held.</summary>
    public static KeepAwake Hold(string reason)
    {
        if (!OperatingSystem.IsWindows()) return new KeepAwake(0, []) { Description = "not Windows: no power request" };
        nint text = Marshal.StringToHGlobalUni(reason);
        try
        {
            var ctx = new ReasonContext { Version = PowerRequestContextVersion, Flags = PowerRequestContextSimpleString, SimpleReasonString = text };
            nint h = PowerCreateRequest(ref ctx);
            if (h == 0 || h == -1) return new KeepAwake(0, []) { Description = $"PowerCreateRequest failed ({Marshal.GetLastWin32Error()})" };
            var held = new List<int>();
            foreach (int t in new[] { PowerRequestExecutionRequired, PowerRequestSystemRequired, PowerRequestDisplayRequired })
                if (PowerSetRequest(h, t)) held.Add(t);
            return new KeepAwake(h, held.ToArray())
            {
                Description = $"power requests held: {string.Join(", ", held.Select(Name))}"
                    + (held.Count < 3 ? " (some could not be set)" : ""),
            };
        }
        finally
        {
            Marshal.FreeHGlobal(text);
        }
    }

    /// <summary>What was held, for the run log.</summary>
    public string Description { get; private init; } = "";

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_request == 0) return;
        foreach (int t in _types) PowerClearRequest(_request, t);
        CloseHandle(_request);
    }

    private static string Name(int t) => t switch
    {
        PowerRequestExecutionRequired => "ExecutionRequired",
        PowerRequestSystemRequired => "SystemRequired",
        PowerRequestDisplayRequired => "DisplayRequired",
        _ => t.ToString(System.Globalization.CultureInfo.InvariantCulture),
    };

    [StructLayout(LayoutKind.Sequential)]
    private struct ReasonContext
    {
        public uint Version;
        public uint Flags;
        public nint SimpleReasonString;

        // REASON_CONTEXT's union is as large as its Detailed member (HMODULE, ULONG, ULONG, LPWSTR*): pad to that size.
        public nint Pad1;
        public nint Pad2;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern nint PowerCreateRequest(ref ReasonContext context);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool PowerSetRequest(nint powerRequest, int requestType);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool PowerClearRequest(nint powerRequest, int requestType);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CloseHandle(nint handle);
}
