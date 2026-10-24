using System.Runtime.Versioning;
using System.Windows.Forms;
using DotLLM.Tray.Hosting;
using DotLLM.Tray.Ui;

namespace DotLLM.Tray;

/// <summary>Entry point for the dotLLM system-tray application.</summary>
[SupportedOSPlatform("windows")]
internal static class Program
{
    private const string SingleInstanceMutexName = @"Local\dotLLM.Tray.SingleInstance";

    [STAThread]
    private static void Main()
    {
        // One tray per session. A second icon would fight the first over the same server and the
        // same autostart entry.
        using var singleInstance = new Mutex(initiallyOwned: true, SingleInstanceMutexName, out var isFirst);
        if (!isFirst)
            return;

        // The orphan guarantee, established before anything can be launched: the tray joins a
        // kill-on-close job object, and every child it starts inherits the membership. See
        // WindowsJobObject for why the tray, and not each child, is the one assigned.
        //
        // A null result means the OS refused. That is degraded, not fatal — ServerSupervisor's
        // own Dispose/Kill paths still reap a child on every ordinary exit; what is lost is the
        // guarantee under `taskkill /F`. The status form reports it rather than hiding it.
        using var jobObject = WindowsJobObject.AssignCurrentProcess();

        ApplicationConfiguration.Initialize();
        Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);

        using var context = new TrayApplicationContext(jobObject is { IsActive: true });
        Application.Run(context);
    }
}
