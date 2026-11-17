using DotLLM.Cli.Helpers;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>Explicit installation using the same pinned package and license flow as the startup offer.</summary>
public sealed class DirectStorageInstallCommand : Command<DirectStorageInstallCommand.Settings>
{
    /// <summary>This interactive installer has no command options.</summary>
    public sealed class Settings : CommandSettings;

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
        => DirectStoragePrompt.Install();
}
