using System.ComponentModel;
using DotLLM.Cli.Helpers;
using DotLLM.HuggingFace;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary><c>dotllm model create</c>: saves a named profile (the Modelfile equivalent) - a base model plus defaults (issue #716).</summary>
internal sealed class ModelCreateCommand : Command<ModelCreateCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<name>")]
        [Description("Name of the new model (letters, digits, . _ - and an optional ':tag'). Requests for this name load the base model with these defaults.")]
        public string Name { get; set; } = string.Empty;

        [CommandOption("--from|-f")]
        [Description("Base model: a .gguf path, 'owner/repo[:quant]', a local model name, or another profile.")]
        public string From { get; set; } = string.Empty;

        [CommandOption("--system")]
        [Description("System prompt prepended when a chat request has none.")]
        public string? System { get; set; }

        [CommandOption("--system-file")]
        [Description("Read the system prompt from a file (instead of --system).")]
        public string? SystemFile { get; set; }

        [CommandOption("--temperature")] public float? Temperature { get; set; }
        [CommandOption("--top-p")] public float? TopP { get; set; }
        [CommandOption("--top-k")] public int? TopK { get; set; }
        [CommandOption("--min-p")] public float? MinP { get; set; }
        [CommandOption("--repeat-penalty")] public float? RepeatPenalty { get; set; }
        [CommandOption("--max-tokens")] public int? MaxTokens { get; set; }
        [CommandOption("--seed")] public int? Seed { get; set; }

        [CommandOption("--stop")]
        [Description("Stop sequence (repeatable).")]
        public string[]? Stop { get; set; }

        [CommandOption("--device|-d")]
        [Description("Device this model loads on: auto, cpu, vulkan, gpu[:N] / cuda[:N].")]
        public string? Device { get; set; }

        [CommandOption("--gpu-layers")] public int? GpuLayers { get; set; }

        [CommandOption("--keep-alive")]
        [Description("Seconds to keep the model loaded after its last request (0 = unload after each request, negative = never).")]
        public double? KeepAlive { get; set; }

        [CommandOption("--description")] public string? Description { get; set; }
    }

    public override int Execute(CommandContext context, Settings s)
    {
        if (ModelProfileStore.NormalizeName(s.Name) is not { } name)
        {
            AnsiConsole.MarkupLine($"[red]'{s.Name.EscapeMarkup()}' is not a valid model name.[/] Use letters, digits, '.', '_', '-' and at most one ':tag'.");
            return 1;
        }
        if (string.IsNullOrWhiteSpace(s.From))
        {
            AnsiConsole.MarkupLine("[red]--from <base model> is required.[/]");
            return 1;
        }

        // A profile's device is replayed at load time; a typo must not be saved and later mean the CPU (#790).
        if (s.Device is not null && DeviceCli.Validate(s.Device) is { } deviceError)
        {
            AnsiConsole.MarkupLine($"[red]{deviceError.EscapeMarkup()}[/]");
            return 1;
        }

        string? system = s.System;
        if (s.SystemFile is not null)
        {
            if (!File.Exists(s.SystemFile)) { AnsiConsole.MarkupLine($"[red]System prompt file not found:[/] {s.SystemFile.EscapeMarkup()}"); return 1; }
            system = File.ReadAllText(s.SystemFile).TrimEnd();
        }

        var profile = new ModelProfile
        {
            From = s.From, Description = s.Description, System = system,
            Temperature = s.Temperature, TopP = s.TopP, TopK = s.TopK, MinP = s.MinP, RepeatPenalty = s.RepeatPenalty,
            MaxTokens = s.MaxTokens, Seed = s.Seed, Stop = s.Stop is { Length: > 0 } ? s.Stop : null,
            Device = s.Device, GpuLayers = s.GpuLayers, KeepAlive = s.KeepAlive,
        };

        var previous = ModelProfileStore.TryGet(name);
        ModelProfileStore.Save(name, profile);
        if (ModelProfileStore.Resolve(name) is null)
        {
            // A cycle (a -> b -> a) or a chain deeper than 8: undo rather than leave an unloadable name behind.
            if (previous is not null) ModelProfileStore.Save(name, previous); else ModelProfileStore.Delete(name);
            AnsiConsole.MarkupLine("[red]That would make the profile chain loop or exceed 8 levels.[/]");
            return 1;
        }

        var resolved = ModelProfileStore.Resolve(name)!.Value.BaseReference;
        bool local = ModelResolver.ResolveLocal(resolved) is not null;
        AnsiConsole.MarkupLine($"[green]Created[/] [bold]{name.EscapeMarkup()}[/] from {s.From.EscapeMarkup()}" +
                               (local ? "" : " [yellow](base model is not downloaded yet; it is pulled on first 'run'/'serve', or use --auto-pull on a server)[/]"));
        return 0;
    }
}

/// <summary><c>dotllm model show</c>: a profile's settings and where its base model resolves, or a model's file and size.</summary>
internal sealed class ModelShowCommand : Command<ModelShowCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("A profile name, 'owner/repo[:quant]', or a local model name.")]
        public string Model { get; set; } = string.Empty;
    }

    public override int Execute(CommandContext context, Settings s)
    {
        var resolvedProfile = ModelProfileStore.Resolve(s.Model);
        string reference = resolvedProfile?.BaseReference ?? s.Model;
        string? path = ModelResolver.ResolveLocal(reference);

        var table = new Table().Border(TableBorder.Rounded).HideHeaders().AddColumn("k").AddColumn("v");
        table.AddRow("[bold]Name[/]", s.Model.EscapeMarkup());
        if (resolvedProfile is { } rp)
        {
            var m = ModelProfileStore.Merge(rp.Chain);
            table.AddRow("[bold]Kind[/]", "profile" + (rp.Chain.Count > 1 ? $" ({rp.Chain.Count} levels)" : ""));
            table.AddRow("[bold]Base[/]", reference.EscapeMarkup());
            void Row(string k, object? v) { if (v is not null) table.AddRow($"[bold]{k}[/]", (v is string[] a ? string.Join(", ", a) : v.ToString() ?? "").EscapeMarkup()); }
            Row("Description", m.Description); Row("System", m.System); Row("Temperature", m.Temperature); Row("Top-P", m.TopP); Row("Top-K", m.TopK);
            Row("Min-P", m.MinP); Row("Repeat penalty", m.RepeatPenalty); Row("Max tokens", m.MaxTokens); Row("Seed", m.Seed); Row("Stop", m.Stop);
            Row("Device", m.Device); Row("GPU layers", m.GpuLayers); Row("Keep-alive (s)", m.KeepAlive);
        }
        else table.AddRow("[bold]Kind[/]", "model");

        if (path is not null)
        {
            table.AddRow("[bold]File[/]", path.EscapeMarkup());
            table.AddRow("[bold]Size[/]", FormatHelpers.FormatSize(ModelResolver.FileLength(path)));
        }
        else
        {
            table.AddRow("[bold]File[/]", "[yellow]not downloaded[/]");
            if (resolvedProfile is null && ModelResolver.Parse(reference) is { IsRepo: false })
            {
                AnsiConsole.MarkupLine($"[red]No profile or local model named '{s.Model.EscapeMarkup()}'.[/]");
                return 1;
            }
        }
        AnsiConsole.Write(table);
        return 0;
    }
}

/// <summary><c>dotllm model cp</c>: copies a profile, or creates a profile that points at a model.</summary>
internal sealed class ModelCopyCommand : Command<ModelCopyCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<source>")]
        [Description("A profile to copy, or a model reference to alias.")]
        public string Source { get; set; } = string.Empty;

        [CommandArgument(1, "<destination>")]
        [Description("The new profile name.")]
        public string Destination { get; set; } = string.Empty;
    }

    public override int Execute(CommandContext context, Settings s)
    {
        if (ModelProfileStore.NormalizeName(s.Destination) is not { } dest)
        {
            AnsiConsole.MarkupLine($"[red]'{s.Destination.EscapeMarkup()}' is not a valid model name.[/]");
            return 1;
        }
        var source = ModelProfileStore.TryGet(s.Source);
        var copy = source is not null ? source : new ModelProfile { From = s.Source };
        if (source is null && ModelResolver.ResolveLocal(s.Source) is null && !ModelResolver.Parse(s.Source).IsRepo)
        {
            AnsiConsole.MarkupLine($"[red]No profile or local model named '{s.Source.EscapeMarkup()}'.[/]");
            return 1;
        }
        ModelProfileStore.Save(dest, copy);
        AnsiConsole.MarkupLine($"[green]Copied[/] {s.Source.EscapeMarkup()} -> [bold]{dest.EscapeMarkup()}[/]");
        return 0;
    }
}

/// <summary><c>dotllm model add</c>: imports a local GGUF (linked, not copied) or downloads a direct URL into the model store.</summary>
internal sealed class ModelAddCommand : AsyncCommand<ModelAddCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<source>")]
        [Description("A path to a .gguf file, or an http(s) URL of one.")]
        public string Source { get; set; } = string.Empty;

        [CommandOption("--name|-n")]
        [Description("Name to register it under (default: the file name without extension). Listed as 'local/<name>'.")]
        public string? Name { get; set; }
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings s)
    {
        bool isUrl = Uri.TryCreate(s.Source, UriKind.Absolute, out var uri) && uri.Scheme is "http" or "https";
        string fileName = isUrl ? Path.GetFileName(uri!.LocalPath) : Path.GetFileName(s.Source);
        if (!fileName.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            AnsiConsole.MarkupLine("[red]Only .gguf files can be added.[/]");
            return 1;
        }
        if (!isUrl && !File.Exists(s.Source))
        {
            AnsiConsole.MarkupLine($"[red]File not found:[/] {s.Source.EscapeMarkup()}");
            return 1;
        }

        string name = (s.Name ?? Path.GetFileNameWithoutExtension(fileName));
        if (ModelProfileStore.NormalizeName(name) is null || name.Contains(':'))
        {
            AnsiConsole.MarkupLine($"[red]'{name.EscapeMarkup()}' is not a valid name.[/] Use letters, digits, '.', '_', '-'.");
            return 1;
        }

        string dir = Path.Combine(HuggingFaceDownloader.DefaultModelsDirectory, "local", name);
        string target = Path.Combine(dir, fileName);
        Directory.CreateDirectory(dir);

        if (isUrl)
        {
            string part = target + ".part";
            using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
            using var response = await http.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead);
            if (!response.IsSuccessStatusCode)
            {
                AnsiConsole.MarkupLine($"[red]Download failed:[/] HTTP {(int)response.StatusCode}");
                return 1;
            }
            long? total = response.Content.Headers.ContentLength;
            await AnsiConsole.Progress().AutoClear(false)
                .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new TransferSpeedColumn())
                .StartAsync(async ctx =>
                {
                    var task = ctx.AddTask($"[green]{fileName.EscapeMarkup()}[/]", maxValue: total ?? 1);
                    await using var src = await response.Content.ReadAsStreamAsync();
                    await using var dst = File.Create(part);
                    var buf = new byte[1 << 20];
                    int n;
                    while ((n = await src.ReadAsync(buf)) > 0)
                    {
                        await dst.WriteAsync(buf.AsMemory(0, n));
                        if (total.HasValue) task.Increment(n);
                    }
                });
            File.Move(part, target, overwrite: true);
        }
        else if (!HubCache.LinkOrCopy(ModelResolver.ResolveLinks(Path.GetFullPath(s.Source)), target))
        {
            AnsiConsole.MarkupLine("[red]Could not link or copy the file into the model store.[/]");
            return 1;
        }

        // A GGUF starts with the magic "GGUF"; anything else is the wrong file (a failed download is usually an HTML error page).
        byte[] magic = new byte[4];
        using (var fs = File.OpenRead(target)) fs.ReadExactly(magic);
        if (System.Text.Encoding.ASCII.GetString(magic) != "GGUF")
        {
            File.Delete(target);
            AnsiConsole.MarkupLine("[red]That is not a GGUF file (bad magic); removed.[/]");
            return 1;
        }

        AnsiConsole.MarkupLine($"[green]Added[/] [bold]{name.EscapeMarkup()}[/] ({FormatHelpers.FormatSize(ModelResolver.FileLength(target))}) as local/{name.EscapeMarkup()}" +
                               (isUrl ? "" : " [grey](hard-linked, no extra disk used)[/]"));
        AnsiConsole.MarkupLine($"[grey]Run it with:[/] dotllm run {name.EscapeMarkup()}");
        return 0;
    }
}

/// <summary><c>dotllm model import-ollama</c>: turns the models of an existing ollama installation into dotLLM profiles (the blobs are used in place).</summary>
internal sealed class ModelImportOllamaCommand : Command<ModelImportOllamaCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "[model]")]
        [Description("One ollama model (e.g. 'llama3.2:3b'). Omit to import every model in the store.")]
        public string? Model { get; set; }

        [CommandOption("--force")]
        [Description("Overwrite an existing profile of the same name.")]
        [DefaultValue(false)]
        public bool Force { get; set; }
    }

    public override int Execute(CommandContext context, Settings s)
    {
        var all = OllamaStore.ListAll();
        if (all.Count == 0)
        {
            AnsiConsole.MarkupLine($"[yellow]No ollama models found in[/] {OllamaStore.DefaultRoot.EscapeMarkup()} [grey](set OLLAMA_MODELS if it lives elsewhere)[/]");
            return 0;
        }
        if (s.Model is not null)
        {
            var want = OllamaRef.TryParse(s.Model);
            all = all.Where(m => want is { } w && m.Ref.Name == w.Name && m.Ref.Tag == w.Tag).ToList();
            if (all.Count == 0) { AnsiConsole.MarkupLine($"[red]'{s.Model.EscapeMarkup()}' is not in the ollama store.[/]"); return 1; }
        }

        int made = 0, skipped = 0;
        foreach (var m in all)
        {
            string name = m.Ref.ToString();
            if (ModelProfileStore.NormalizeName(name) is null) { skipped++; continue; }
            if (!s.Force && ModelProfileStore.TryGet(name) is not null) { skipped++; continue; }
            ModelProfileStore.Save(name, OllamaStore.ToProfile(m, m.BlobPath));
            made++;
            AnsiConsole.MarkupLine($"[green]Imported[/] {name.EscapeMarkup()} [grey]({FormatHelpers.FormatSize(m.SizeBytes)}, blob used in place)[/]");
        }
        AnsiConsole.MarkupLine($"{made} imported, {skipped} skipped (already a profile)." );
        return 0;
    }
}
