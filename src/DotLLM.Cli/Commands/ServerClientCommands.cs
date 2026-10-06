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
    [Description("Server base URL (default: $DOTLLM_URL or http://localhost:8080).")]
    public string? Url { get; set; }

    internal string BaseUrl => (Url ?? Environment.GetEnvironmentVariable("DOTLLM_URL") ?? "http://localhost:8080").TrimEnd('/');
}

/// <summary><c>dotllm ps</c>: the models a running server has resident, with idle time and auto-unload countdown (ollama <c>ps</c>).</summary>
internal sealed class PsCommand : AsyncCommand<ServerClientSettings>
{
    public override async Task<int> ExecuteAsync(CommandContext context, ServerClientSettings s)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
        JsonDocument doc;
        try { doc = JsonDocument.Parse(await http.GetStringAsync($"{s.BaseUrl}/v1/models")); }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            AnsiConsole.MarkupLine($"[red]No dotllm server answering at {s.BaseUrl.EscapeMarkup()}[/] ({ex.Message.EscapeMarkup()}). Start one with 'dotllm serve'.");
            return 1;
        }

        using (doc)
        {
            var table = new Table().Border(TableBorder.Rounded);
            table.AddColumn("Model"); table.AddColumn("Active"); table.AddColumn(new TableColumn("Size").RightAligned());
            table.AddColumn(new TableColumn("Idle").RightAligned()); table.AddColumn(new TableColumn("Unloads in").RightAligned());
            int rows = 0;
            foreach (var m in doc.RootElement.GetProperty("data").EnumerateArray())
            {
                string id = m.GetProperty("id").GetString() ?? "";
                if (id == "none") continue;   // the placeholder entry of a server with no model loaded
                long size = m.TryGetProperty("size_bytes", out var sz) && sz.ValueKind == JsonValueKind.Number ? sz.GetInt64() : 0;
                bool active = m.TryGetProperty("is_active", out var a) && a.ValueKind == JsonValueKind.True;
                if (!active && size == 0) continue;   // known but not resident (unloaded): not part of ps
                string idle = m.TryGetProperty("idle_seconds", out var i) && i.ValueKind == JsonValueKind.Number ? $"{i.GetDouble():F0}s" : "-";
                string expires = m.TryGetProperty("expires_in_seconds", out var e) && e.ValueKind == JsonValueKind.Number ? $"{e.GetDouble():F0}s" : "never";
                table.AddRow(id.EscapeMarkup(), active ? "[green]yes[/]" : "-", size > 0 ? FormatHelpers.FormatSize(size) : "-", idle, expires);
                rows++;
            }
            if (rows == 0) { AnsiConsole.MarkupLine("[yellow]No models are loaded.[/]"); return 0; }
            AnsiConsole.Write(table);
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
