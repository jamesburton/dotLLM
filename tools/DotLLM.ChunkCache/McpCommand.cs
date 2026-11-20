using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ModelContextProtocol;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace DotLLM.ChunkCache;

/// <summary>
/// <c>chunkcache mcp [--store DIR] [--server URL] [--model FILE] [--threads N]</c>: a stdio MCP server whose six tools run the
/// same verb handlers as <c>chunkcache &lt;verb&gt; --json</c> and nothing else (<see cref="McpTools"/>). stdout carries only the
/// protocol's JSON-RPC frames; progress, warnings and the server's own diagnostics go to stderr. Tool calls run one at a time.
/// </summary>
internal static class McpCommand
{
    public const string Help = """
        usage: chunkcache mcp [--store DIR] [--server URL] [--model FILE.gguf] [--threads N]

        Runs a Model Context Protocol server on stdin/stdout (newline-delimited JSON-RPC, MCP 2025-06-18 and later) with the tools
          chunkcache_build, chunkcache_select, chunkcache_status, chunkcache_evict, chunkcache_prewarm, chunkcache_explain
        Each tool runs the matching `chunkcache <verb> --json` and returns its result as structuredContent (projected onto the tool's
        outputSchema) plus a serialised text copy. A failing verb is a tool result with isError (a {"error": {"code", "message"}}
        object), not a protocol error. chunkcache_evict is a dry run and chunkcache_build only indexes unless the call says otherwise.

          --store DIR     default store of a call that gives none
          --server URL    default dotLLM server (the server owns the model identity and all K/V writes). The API key is read from
                          CHUNKCACHE_API_KEY, never from an argument.
          --model FILE    the GGUF whose tokenizer chunks the tree (build, select, status; the file the server serves).
                          Default for a call that gives no `model` of its own.
          --threads N     CPU threads the model is opened with (default min(8, cores))
          --help          this text

        Claude Code (.mcp.json):
          {"mcpServers": {"chunkcache": {"command": "chunkcache", "args": ["mcp", "--store", "./store", "--model", "M.gguf", "--server", "http://localhost:8080"]}}}

        exit codes: 0 stdin closed, 2 usage, 1 error.
        """;

    private static readonly Dictionary<string, OptionKind> Spec = new(StringComparer.Ordinal)
    {
        ["store"] = OptionKind.Single, ["server"] = OptionKind.Single, ["model"] = OptionKind.Single, ["threads"] = OptionKind.Single, ["help"] = OptionKind.Flag,
        ["verbose"] = OptionKind.Flag,
    };

    /// <summary>
    /// Runs the server until <paramref name="input"/> ends or <paramref name="ct"/> is cancelled.
    /// </summary>
    /// <param name="args">Arguments after <c>mcp</c>.</param>
    /// <param name="input">The client's frames (stdin).</param>
    /// <param name="output">The server's frames (stdout).</param>
    /// <param name="error">Diagnostics (stderr).</param>
    /// <param name="helpOutput">Where <c>--help</c> prints (stdout of the process; no frame is ever written there).</param>
    /// <param name="workingDirectory">Directory relative tool paths resolve against.</param>
    /// <param name="httpHandler">Handler of the server calls of the tools (tests; null: the default).</param>
    /// <param name="getEnv">Environment lookup (null: the process's); the API key is read through it.</param>
    /// <param name="ct">Cancellation.</param>
    /// <returns>The exit code.</returns>
    public static async Task<int> RunAsync(string[] args, Stream input, Stream output, TextWriter error, TextWriter helpOutput,
        string workingDirectory, HttpMessageHandler? httpHandler, Func<string, string?>? getEnv, CancellationToken ct)
    {
        McpDefaults defaults;
        try
        {
            CliArgs parsed = CliArgs.Parse("mcp", args, Spec);
            if (parsed.Flag("help"))
            {
                helpOutput.WriteLine(Help);
                return ExitCodes.Ok;
            }
            defaults = new McpDefaults(parsed.Value("store"), parsed.Value("server"), parsed.Value("model"), parsed.IntOrNull("threads", 1, 1024));
        }
        catch (UsageException ex)
        {
            error.WriteLine($"usage error: {ex.Message}");
            error.WriteLine(Help);
            return ExitCodes.Usage;
        }

        getEnv ??= Environment.GetEnvironmentVariable;
        using var gate = new SemaphoreSlim(1, 1);
        var options = new McpServerOptions
        {
            ServerInfo = new Implementation { Name = "chunkcache", Version = typeof(McpCommand).Assembly.GetName().Version?.ToString() ?? "0" },
            Capabilities = new ServerCapabilities { Tools = new ToolsCapability() },
            ServerInstructions = "Manage the dotLLM chunk-level KV cache store of a .NET source tree: index it, select context chunks, check status, evict, prewarm and explain.",
            Handlers = new McpServerHandlers
            {
                ListToolsHandler = (_, _) => ValueTask.FromResult(new ListToolsResult { Tools = [.. McpTools.All.Select(ToProtocol)] }),
                CallToolHandler = async (request, token) =>
                {
                    McpToolDef tool = McpTools.Find(request.Params!.Name)
                        ?? throw new McpProtocolException($"Unknown tool '{request.Params.Name}'.", McpErrorCode.InvalidParams);
                    JsonObject arguments = new();
                    if (request.Params.Arguments is { } dict)
                        foreach ((string k, JsonElement v) in dict) arguments[k] = JsonNode.Parse(v.GetRawText());
                    await gate.WaitAsync(token).ConfigureAwait(false);
                    try
                    {
                        return await Task.Run(() => CallTool(tool, arguments, defaults, error, workingDirectory, httpHandler, getEnv), token).ConfigureAwait(false);
                    }
                    finally
                    {
                        gate.Release();
                    }
                },
            },
        };

        try
        {
            await using var transport = new StreamServerTransport(input, output, "chunkcache", null);
            await using McpServer server = McpServer.Create(transport, options);
            await server.RunAsync(ct).ConfigureAwait(false);
            return ExitCodes.Ok;
        }
        catch (OperationCanceledException)
        {
            return ExitCodes.Ok;
        }
        catch (Exception ex) when (ex is IOException or InvalidOperationException or McpException)
        {
            error.WriteLine($"error: {ex.Message}");
            return ExitCodes.Error;
        }
    }

    private static Tool ToProtocol(McpToolDef t) => new()
    {
        Name = t.Name,
        Description = t.Description,
        InputSchema = JsonSerializer.SerializeToElement(t.InputSchema),
        OutputSchema = JsonSerializer.SerializeToElement(t.OutputSchema),
        Annotations = new ToolAnnotations { ReadOnlyHint = t.ReadOnly, DestructiveHint = t.Destructive, OpenWorldHint = false },
    };

    /// <summary>Validates the arguments, runs the verb in process and turns its JSON into a tool result.</summary>
    internal static CallToolResult CallTool(McpToolDef tool, JsonObject arguments, McpDefaults defaults, TextWriter error, string workingDirectory,
        HttpMessageHandler? http, Func<string, string?> getEnv)
    {
        List<string> problems = McpSchema.Validate(arguments, tool.InputSchema, "arguments");
        if (problems.Count > 0) return Failure("invalid_arguments", $"{tool.Name}: " + string.Join("; ", problems));

        McpCall call;
        try
        {
            call = tool.Plan(arguments, defaults);
        }
        catch (McpArgumentException ex)
        {
            return Failure("invalid_arguments", ex.Message);
        }

        try
        {
            var o = new StringWriter();
            var e = new StringWriter();
            int code = ChunkCacheCli.Run([.. call.Argv], o, e, workingDirectory, http, getEnv);
            string progress = e.ToString();
            if (progress.Length > 0) error.Write(progress); // diagnostics stay off the protocol channel
            JsonNode? doc;
            try
            {
                doc = JsonNode.Parse(o.ToString());
            }
            catch (JsonException)
            {
                return Failure("internal", $"{tool.Name}: the verb printed no JSON document (exit {code}).");
            }
            if (code != ExitCodes.Ok)
            {
                if (doc?["error"] is JsonObject err)
                    return Failure(err["code"]?.GetValue<string>() ?? "error", err["message"]?.GetValue<string>() ?? $"exit {code}");
                return Failure("error", $"{tool.Name}: exit {code}.");
            }

            JsonNode projected = McpSchema.Project(doc!, tool.OutputSchema);
            List<string> output = McpSchema.Validate(projected, tool.OutputSchema, "result");
            if (output.Count > 0) return Failure("internal", $"{tool.Name}: the verb's result does not match the tool's output schema: {string.Join("; ", output)}");
            string text = projected.ToJsonString();
            return new CallToolResult
            {
                Content = [new TextContentBlock { Text = text }],
                StructuredContent = JsonSerializer.SerializeToElement(projected),
            };
        }
        finally
        {
            foreach (string dir in call.TempFiles)
            {
                try { Directory.Delete(dir, recursive: true); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
    }

    private static CallToolResult Failure(string code, string message) => new()
    {
        IsError = true,
        Content = [new TextContentBlock { Text = new JsonObject { ["error"] = new JsonObject { ["code"] = code, ["message"] = message } }.ToJsonString() }],
    };
}
