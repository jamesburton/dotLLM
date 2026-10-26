using DotLLM.Vulkan.DirectStorage;
using Spectre.Console;

namespace DotLLM.Cli.Helpers;

/// <summary>
/// Startup offer (#1304): on Windows, when Microsoft's DirectStorage DLLs are missing, offer to fetch them from NuGet after the
/// user accepts Microsoft's licence. Interactive terminals only, asked once (a "no" is remembered), and never for commands
/// that do not load a model.
/// </summary>
internal static class DirectStoragePrompt
{
    private static readonly HashSet<string> s_modelCommands = new(StringComparer.OrdinalIgnoreCase) { "run", "chat", "bench", "serve" };

    /// <summary>Makes the offer when every condition holds; never throws and never blocks a non-interactive run.</summary>
    public static void OfferIfMissing(string[] args)
    {
        try
        {
            if (!ShouldOffer(args)) return;

            AnsiConsole.MarkupLine("[yellow]DirectStorage is not installed.[/] dotLLM can use Microsoft's DirectStorage to load model weights");
            AnsiConsole.MarkupLine("straight from disk into GPU memory (Vulkan, Windows) without an extra copy in system RAM.");
            AnsiConsole.MarkupLine("The two DLLs are Microsoft's, not dotLLM's: they are fetched from NuGet ([blue]Microsoft.Direct3D.DirectStorage "
                + DirectStorageInstaller.PackageVersion + "[/]) into [blue]" + Markup.Escape(DirectStorageInstaller.InstallDirectory!) + "[/].");
            if (!AnsiConsole.Confirm("Download them? You will be shown Microsoft's licence first.", defaultValue: false))
            {
                Decline();
                return;
            }

            string? dir = DirectStorageInstaller.InstallAsync(ConfirmLicense).GetAwaiter().GetResult();
            if (dir is null)
            {
                Decline();
                AnsiConsole.MarkupLine("[grey]Licence not accepted; nothing was installed. dotLLM will not ask again.[/]");
                return;
            }
            AnsiConsole.MarkupLine("[green]Installed[/] to " + Markup.Escape(dir) + ".");
            AnsiConsole.MarkupLine("DirectStorage weight loading is still opt-in: set [blue]DOTLLM_VULKAN_DIRECT_STORAGE=1[/] to use it.");
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidOperationException or PlatformNotSupportedException)
        {
            // A failed offer must never stop the command the user actually asked for.
            AnsiConsole.MarkupLine("[yellow]DirectStorage install skipped:[/] " + Markup.Escape(e.Message));
        }
    }

    private static bool ShouldOffer(string[] args)
    {
        if (!DirectStorageInstaller.IsSupportedPlatform) return false;
        if (args.Length == 0 || !s_modelCommands.Contains(args[0])) return false;
        if (Console.IsInputRedirected || Console.IsOutputRedirected || Console.IsErrorRedirected) return false;
        if (string.Equals(Environment.GetEnvironmentVariable("DOTLLM_NO_PROMPT"), "1", StringComparison.Ordinal)) return false;
        if (File.Exists(DirectStorageInstaller.DeclinedMarkerPath)) return false;
        return !DirectStorageInstaller.IsAvailable();
    }

    private static bool ConfirmLicense(string licenseText)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold]Microsoft DirectStorage SDK licence[/]").LeftJustified());
        Console.WriteLine(licenseText);
        AnsiConsole.Write(new Rule());
        AnsiConsole.MarkupLine("A copy is saved next to the DLLs as LICENSE.txt. Note the licence's distribution terms (section 2): they apply to anyone who redistributes these files.");
        AnsiConsole.Markup("Type [bold]yes[/] to accept these terms and install: ");
        return string.Equals(Console.ReadLine()?.Trim(), "yes", StringComparison.OrdinalIgnoreCase);
    }

    private static void Decline()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(DirectStorageInstaller.DeclinedMarkerPath)!);
            File.WriteAllText(DirectStorageInstaller.DeclinedMarkerPath, DateTime.UtcNow.ToString("O"));
            AnsiConsole.MarkupLine("[grey]Not installing. dotLLM will not ask again (delete " + Markup.Escape(DirectStorageInstaller.DeclinedMarkerPath) + " to be asked).[/]");
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
    }
}
