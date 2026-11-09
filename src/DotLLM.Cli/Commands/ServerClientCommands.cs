using System.ComponentModel;
using System.Text.Json;
using DotLLM.Cli.Helpers;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>Settings shared by the thin clients of a running server (<c>ps</c>, <c>stop</c>).</summary>
internal class ServerClientSettings : CommandSettings
{
    [CommandOption("--url")]
    [Description("Base URL of the running dotllm server. Default: the DOTLLM_URL environment variable, else http://localhost:8080 (the default of dotllm serve --port).")]
    public string? Url { get; set; }

    internal string BaseUrl => (Url ?? Environment.GetEnvironmentVariable("DOTLLM_URL") ?? "http://localhost:8080").TrimEnd('/');
}

/// <summary><c>dotllm ps</c>: the models a running server has resident, with idle time and auto-unload countdown (ollama <c>ps</c>).</summary>
internal sealed class PsCommand : AsyncCommand<PsCommand.Settings>
{
    public sealed class Settings : ServerClientSettings
    {
        [CommandOption("--json")]
        [Description("Print one JSON object {\"version\": 1, \"url\", \"models\": array, \"device_warning\"} instead of a table (model: id, active, device, size_bytes, idle_seconds, expires_in_seconds). A failure prints {\"error\": {code, message, exit_code}} on stdout. Nothing else is written to stdout.")]
        [DefaultValue(false)]
        public bool Json { get; set; }
    }

    /// <summary>
    /// Builds the resident-model rows from a <c>/v1/models</c> response (the "none" placeholder and known-but-unloaded entries are skipped).
    /// Shared by the table and <c>--json</c> so they cannot disagree.
    /// </summary>
    internal static PsJsonResult Build(JsonElement modelsResponse, string baseUrl, string? activeDevice, string? deviceWarning)
    {
        var rows = new List<PsJsonModel>();
        foreach (var m in modelsResponse.GetProperty("data").EnumerateArray())
        {
            string id = m.GetProperty("id").GetString() ?? "";
            if (id == "none") continue;   // the placeholder entry of a server with no model loaded
            long size = m.TryGetProperty("size_bytes", out var sz) && sz.ValueKind == JsonValueKind.Number ? sz.GetInt64() : 0;
            bool active = m.TryGetProperty("is_active", out var a) && a.ValueKind == JsonValueKind.True;
            if (!active && size == 0) continue;   // known but not resident (unloaded): not part of ps
            rows.Add(new PsJsonModel
            {
                Id = id,
                Active = active,
                Device = active ? activeDevice : null,
                SizeBytes = size,
                IdleSeconds = m.TryGetProperty("idle_seconds", out var i) && i.ValueKind == JsonValueKind.Number ? i.GetDouble() : null,
                ExpiresInSeconds = m.TryGetProperty("expires_in_seconds", out var e) && e.ValueKind == JsonValueKind.Number ? e.GetDouble() : null,
            });
        }
        return new PsJsonResult { Version = CliJson.SchemaVersion, Url = baseUrl, Models = rows.ToArray(), DeviceWarning = deviceWarning };
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings s)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
        JsonDocument doc;
        try { doc = JsonDocument.Parse(await http.GetStringAsync($"{s.BaseUrl}/v1/models")); }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException)
        {
            if (s.Json) return CliJson.WriteError("error", $"No dotllm server answering at {s.BaseUrl} ({ex.Message}). Start one with 'dotllm serve'.", 1);
            AnsiConsole.MarkupLine($"[red]No dotllm server answering at {s.BaseUrl.EscapeMarkup()}[/] ({ex.Message.EscapeMarkup()}). Start one with 'dotllm serve'.");
            return 1;
        }

        // Device of the active model, from /props - the same fields the server reports there (#790): requested -> resolved + any CPU warning.
        string? activeDevice = null, deviceWarning = null;
        try
        {
            using var props = JsonDocument.Parse(await http.GetStringAsync($"{s.BaseUrl}/props"));
            var root = props.RootElement;
            string? requested = root.TryGetProperty("device", out var rq) && rq.ValueKind == JsonValueKind.String ? rq.GetString() : null;
            string? resolved = root.TryGetProperty("resolved_device", out var rs) && rs.ValueKind == JsonValueKind.String ? rs.GetString() : null;
            activeDevice = resolved is null ? requested
                : string.Equals(requested, resolved, StringComparison.OrdinalIgnoreCase) || requested is null ? resolved : $"{requested} -> {resolved}";
            deviceWarning = root.TryGetProperty("device_fallback_warning", out var dw) && dw.ValueKind == JsonValueKind.String ? dw.GetString() : null;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException) { /* /props is optional for ps */ }

        using (doc)
        {
            PsJsonResult result;
            try { result = Build(doc.RootElement, s.BaseUrl, activeDevice, deviceWarning); }
            catch (Exception ex) when (ex is KeyNotFoundException or InvalidOperationException)
            {
                if (s.Json) return CliJson.WriteError("error", $"{s.BaseUrl}/v1/models did not return the expected shape: {ex.Message}", 1);
                AnsiConsole.MarkupLine($"[red]{s.BaseUrl.EscapeMarkup()}/v1/models did not return the expected shape:[/] {ex.Message.EscapeMarkup()}");
                return 1;
            }

            if (s.Json)
            {
                Console.WriteLine(JsonSerializer.Serialize(result, CliJsonContext.Default.PsJsonResult));
                return 0;
            }

            if (result.Models.Length == 0) { AnsiConsole.MarkupLine("[yellow]No models are loaded.[/]"); return 0; }
            var table = new Table().Border(TableBorder.Rounded);
            table.AddColumn("Model"); table.AddColumn("Active"); table.AddColumn("Device"); table.AddColumn(new TableColumn("Size").RightAligned());
            table.AddColumn(new TableColumn("Idle").RightAligned()); table.AddColumn(new TableColumn("Unloads in").RightAligned());
            foreach (var m in result.Models)
                table.AddRow(m.Id.EscapeMarkup(), m.Active ? "[green]yes[/]" : "-", m.Device is not null ? m.Device.EscapeMarkup() : "-",
                    m.SizeBytes > 0 ? FormatHelpers.FormatSize(m.SizeBytes) : "-",
                    m.IdleSeconds is { } i ? $"{i:F0}s" : "-", m.ExpiresInSeconds is { } e ? $"{e:F0}s" : "never");
            AnsiConsole.Write(table);
            if (result.DeviceWarning is not null)
                AnsiConsole.MarkupLine($"[bold yellow]WARNING: {result.DeviceWarning.EscapeMarkup()}[/]");
        }
        return 0;
    }
}

/// <summary><c>dotllm stop</c>: unloads a model (or every model) from a running server. Needs a server started with --allow-model-admin.</summary>
internal sealed class StopCommand : AsyncCommand<StopCommand.Settings>
{
    public sealed class Settings : ServerClientSettings
    {
        [CommandArgument(0, "[model]")]
        [Description("Model to unload (as shown by 'dotllm ps'). Omit with --all to unload everything.")]
        public string? Model { get; set; }

        [CommandOption("--all")]
        [Description("Unload every resident model.")]
        [DefaultValue(false)]
        public bool All { get; set; }

        [CommandOption("--server")]
        [Description("Stop the whole server process gracefully (POST /v1/admin/shutdown).")]
        [DefaultValue(false)]
        public bool Server { get; set; }
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings s)
    {
        if (string.IsNullOrWhiteSpace(s.Model) && !s.All && !s.Server)
        {
            AnsiConsole.MarkupLine("[red]Name a model to stop, or pass --all (every model) or --server (the whole server).[/]");
            return 1;
        }
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        try
        {
            if (s.Server)
            {
                using var empty = new StringContent(string.Empty, System.Text.Encoding.UTF8, "application/json");
                using var shut = await http.PostAsync($"{s.BaseUrl}/v1/admin/shutdown", empty);
                if (shut.StatusCode == System.Net.HttpStatusCode.Forbidden)
                {
                    AnsiConsole.MarkupLine("[red]The server refused: model administration is off.[/] Restart it with [bold]--allow-model-admin[/].");
                    return 1;
                }
                AnsiConsole.MarkupLine(shut.IsSuccessStatusCode ? "[green]Server is shutting down.[/]" : $"[red]Shutdown failed:[/] HTTP {(int)shut.StatusCode}");
                return shut.IsSuccessStatusCode ? 0 : 1;
            }

            // Hand-built body: this assembly is trim-analysed, so no reflection-based serialisation.
            string json = s.All ? "{\"all\":true}" : $"{{\"model\":{JsonSerializer.Serialize(s.Model, JsonStringContext.Default.String)},\"all\":false}}";
            using var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
            using var response = await http.PostAsync($"{s.BaseUrl}/v1/models/unload", content);
            string body = await response.Content.ReadAsStringAsync();
            if (response.StatusCode == System.Net.HttpStatusCode.Forbidden)
            {
                AnsiConsole.MarkupLine("[red]The server refused: model administration is off.[/] Restart it with [bold]--allow-model-admin[/].");
                return 1;
            }
            if (!response.IsSuccessStatusCode)
            {
                AnsiConsole.MarkupLine($"[red]Stop failed:[/] HTTP {(int)response.StatusCode} {body.EscapeMarkup()}");
                return 1;
            }
            AnsiConsole.MarkupLine(s.All ? "[green]Unloaded all models.[/]" : $"[green]Unloaded[/] {s.Model!.EscapeMarkup()}");
            return 0;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            AnsiConsole.MarkupLine($"[red]No dotllm server answering at {s.BaseUrl.EscapeMarkup()}[/] ({ex.Message.EscapeMarkup()}).");
            return 1;
        }
    }
}

[System.Text.Json.Serialization.JsonSerializable(typeof(string))]
internal partial class JsonStringContext : System.Text.Json.Serialization.JsonSerializerContext;
