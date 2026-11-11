using System.Net;
using Spectre.Console;
using Spectre.Console.Cli;
using System.Text;
using System.Text.Json;
using DotLLM.Cli;
using DotLLM.Cli.Commands;
using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>Serialises the tests that swap <see cref="Console.Out"/> and process environment variables.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class CliConsoleCollection
{
    public const string Name = "CliConsole";
}

/// <summary>
/// Issue #1721: the <c>--json</c> outputs of <c>list</c>/<c>model list</c>, <c>model info</c> and <c>ps</c> parse and match the schemas in
/// <c>Cli/Fixtures/v1/schemas</c>; a failure prints the <c>{"error":{...}}</c> document; the documented exit codes (0 ok, 1 runtime error,
/// 2 usage) are honoured by the real command tree.
/// </summary>
[Collection(CliConsoleCollection.Name)]
public sealed class CliJsonTests
{
    // ------------------------------------------------------------------------------------------------ harness

    /// <summary>Runs the real command tree and returns the exit code plus everything written to <see cref="Console.Out"/> (where --json goes).</summary>
    private static (int Code, string Stdout) Run(params string[] args)
    {
        var original = Console.Out;
        var sw = new StringWriter();
        Console.SetOut(sw);
        try { return (CliCommandModel.Run(args).ExitCode, sw.ToString()); }
        finally { Console.SetOut(original); }
    }

    /// <summary>Violations of <paramref name="json"/> against <c>Cli/Fixtures/v1/schemas/&lt;name&gt;.json</c> (empty = valid). The text must be ONE JSON document.</summary>
    private static List<string> SchemaErrors(string name, string json)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Cli", "Fixtures", "v1", "schemas", name + ".json");
        using JsonDocument schema = JsonDocument.Parse(File.ReadAllText(path));
        using JsonDocument doc = JsonDocument.Parse(json);
        var errors = new List<string>();
        Validate(doc.RootElement, schema.RootElement, "$", errors);
        return errors;
    }

    // The subset of JSON Schema the fixtures use: type, enum, required, properties, additionalProperties: false, items.
    private static void Validate(JsonElement value, JsonElement schema, string path, List<string> errors)
    {
        if (schema.TryGetProperty("type", out JsonElement type))
        {
            string t = type.GetString()!;
            bool ok = t switch
            {
                "object" => value.ValueKind == JsonValueKind.Object,
                "array" => value.ValueKind == JsonValueKind.Array,
                "string" => value.ValueKind == JsonValueKind.String,
                "boolean" => value.ValueKind is JsonValueKind.True or JsonValueKind.False,
                "number" => value.ValueKind == JsonValueKind.Number,
                "integer" => value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out _),
                _ => throw new InvalidOperationException("schema type " + t),
            };
            if (!ok) { errors.Add($"{path}: expected {t}, found {value.ValueKind}"); return; }
        }
        if (schema.TryGetProperty("enum", out JsonElement allowed) && !allowed.EnumerateArray().Any(a => a.ToString() == value.ToString()))
            errors.Add($"{path}: '{value}' is not one of {allowed}");
        if (value.ValueKind == JsonValueKind.Object)
        {
            if (schema.TryGetProperty("required", out JsonElement required))
                foreach (JsonElement name in required.EnumerateArray())
                    if (!value.TryGetProperty(name.GetString()!, out _)) errors.Add($"{path}: missing required property {name.GetString()}");
            schema.TryGetProperty("properties", out JsonElement props);
            foreach (JsonProperty p in value.EnumerateObject())
            {
                if (props.ValueKind == JsonValueKind.Object && props.TryGetProperty(p.Name, out JsonElement sub)) Validate(p.Value, sub, path + "." + p.Name, errors);
                else if (schema.TryGetProperty("additionalProperties", out JsonElement extra) && extra.ValueKind == JsonValueKind.False)
                    errors.Add($"{path}: unexpected property {p.Name}");
            }
        }
        if (value.ValueKind == JsonValueKind.Array && schema.TryGetProperty("items", out JsonElement items))
        {
            int i = 0;
            foreach (JsonElement item in value.EnumerateArray()) Validate(item, items, $"{path}[{i++}]", errors);
        }
    }

    /// <summary>A throwaway HTTP server answering <c>/v1/models</c> and <c>/props</c> like <c>dotllm serve</c>.</summary>
    private sealed class FakeServer : IDisposable
    {
        private readonly HttpListener _listener = new();
        public string Url { get; }

        public FakeServer(string modelsJson, string propsJson)
        {
            int port;
            using (var probe = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0))
            {
                probe.Start();
                port = ((IPEndPoint)probe.LocalEndpoint).Port;
            }
            Url = $"http://127.0.0.1:{port}";
            _listener.Prefixes.Add(Url + "/");
            _listener.Start();
            _ = Task.Run(async () =>
            {
                while (_listener.IsListening)
                {
                    HttpListenerContext ctx;
                    try { ctx = await _listener.GetContextAsync(); } catch { return; }
                    string body = ctx.Request.Url!.AbsolutePath == "/v1/models" ? modelsJson : ctx.Request.Url.AbsolutePath == "/props" ? propsJson : "{}";
                    byte[] bytes = Encoding.UTF8.GetBytes(body);
                    ctx.Response.ContentType = "application/json";
                    await ctx.Response.OutputStream.WriteAsync(bytes);
                    ctx.Response.Close();
                }
            });
        }

        public void Dispose() => _listener.Close();
    }

    private const string ModelsResponse = """
        {"object":"list","data":[
          {"id":"none","is_active":false},
          {"id":"qwen3:0.6b","is_active":true,"size_bytes":639631360,"idle_seconds":12.4,"expires_in_seconds":287.6},
          {"id":"smol","is_active":false,"size_bytes":145000000,"idle_seconds":3},
          {"id":"unloaded-one","is_active":false}
        ]}
        """;

    private const string PropsResponse = """{"device":"auto","resolved_device":"vulkan:0 (Radeon)","device_fallback_warning":"running on CPU"}""";

    // ------------------------------------------------------------------------------------------------ ps

    [Fact]
    public void Ps_Json_MatchesSchema_AndListsOnlyResidentModels()
    {
        using var server = new FakeServer(ModelsResponse, PropsResponse);
        var (code, stdout) = Run("ps", "--json", "--url", server.Url);

        Assert.Equal(0, code);
        Assert.Empty(SchemaErrors("ps", stdout));
        using var doc = JsonDocument.Parse(stdout);
        JsonElement root = doc.RootElement;
        Assert.Equal(1, root.GetProperty("version").GetInt32());
        Assert.Equal(server.Url, root.GetProperty("url").GetString());
        Assert.Equal("running on CPU", root.GetProperty("device_warning").GetString());

        JsonElement[] models = root.GetProperty("models").EnumerateArray().ToArray();
        Assert.Equal(["qwen3:0.6b", "smol"], models.Select(m => m.GetProperty("id").GetString()));   // "none" and the unloaded entry are skipped
        Assert.True(models[0].GetProperty("active").GetBoolean());
        Assert.Equal("auto -> vulkan:0 (Radeon)", models[0].GetProperty("device").GetString());
        Assert.Equal(639631360, models[0].GetProperty("size_bytes").GetInt64());
        Assert.Equal(12.4, models[0].GetProperty("idle_seconds").GetDouble());
        Assert.Equal(287.6, models[0].GetProperty("expires_in_seconds").GetDouble());
        Assert.False(models[1].GetProperty("active").GetBoolean());
        Assert.False(models[1].TryGetProperty("device", out _));           // device is only known for the active model
        Assert.False(models[1].TryGetProperty("expires_in_seconds", out _)); // never auto-unloads: field absent
    }

    [Fact]
    public void Ps_Json_NoModelsLoaded_IsAnEmptyArray_ExitZero()
    {
        using var server = new FakeServer("""{"data":[{"id":"none"}]}""", "{}");
        var (code, stdout) = Run("ps", "--json", "--url", server.Url);
        Assert.Equal(0, code);
        Assert.Empty(SchemaErrors("ps", stdout));
        Assert.Equal(0, JsonDocument.Parse(stdout).RootElement.GetProperty("models").GetArrayLength());
    }

    [Fact]
    public void Ps_Json_NoServer_PrintsErrorDocument_ExitOne()
    {
        var (code, stdout) = Run("ps", "--json", "--url", "http://127.0.0.1:1");
        Assert.Equal(1, code);
        Assert.Empty(SchemaErrors("error", stdout));
        JsonElement err = JsonDocument.Parse(stdout).RootElement.GetProperty("error");
        Assert.Equal("server_unreachable", err.GetProperty("code").GetString());
        Assert.Equal(1, err.GetProperty("exit_code").GetInt32());
        Assert.Contains("No dotllm server answering", err.GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ model list / list

    [Theory]
    [InlineData("list")]
    [InlineData("model list")]
    public void List_Json_MatchesSchema_AndIncludesLocalFilesAndProfiles(string command)
    {
        string root = Path.Combine(Path.GetTempPath(), "dotllm-cli-json-" + Guid.NewGuid().ToString("N"));
        string models = Path.Combine(root, "models"), profiles = Path.Combine(root, "profiles");
        string repoDir = Path.Combine(models, "acme", "tiny-GGUF");
        Directory.CreateDirectory(repoDir);
        Directory.CreateDirectory(Path.Combine(root, "hub"));
        Directory.CreateDirectory(Path.Combine(root, "ollama"));
        File.WriteAllBytes(Path.Combine(repoDir, "tiny.Q4_K_M.gguf"), new byte[4096]);
        ModelProfileStore.Save("my-tiny", new ModelProfile { From = "acme/tiny-GGUF", Device = "cpu" }, profiles);

        var saved = new Dictionary<string, string?>();
        foreach (var (k, v) in new[] { ("DOTLLM_MODELS_DIR", models), ("DOTLLM_PROFILES_DIR", profiles), ("HF_HUB_CACHE", Path.Combine(root, "hub")), ("OLLAMA_MODELS", Path.Combine(root, "ollama")) })
        {
            saved[k] = Environment.GetEnvironmentVariable(k);
            Environment.SetEnvironmentVariable(k, v);
        }
        try
        {
            var (code, stdout) = Run([.. command.Split(' '), "--json"]);
            Assert.Equal(0, code);
            Assert.Empty(SchemaErrors("model-list", stdout));
            JsonElement doc = JsonDocument.Parse(stdout).RootElement;
            Assert.Equal(1, doc.GetProperty("version").GetInt32());
            JsonElement m = Assert.Single(doc.GetProperty("models").EnumerateArray());
            Assert.Equal("tiny.Q4_K_M.gguf", m.GetProperty("filename").GetString());
            Assert.Equal("Q4_K_M", m.GetProperty("quant").GetString());
            Assert.Equal(4096, m.GetProperty("size_bytes").GetInt64());
            JsonElement p = Assert.Single(doc.GetProperty("profiles").EnumerateArray());
            Assert.Equal("my-tiny", p.GetProperty("name").GetString());
            Assert.Equal("acme/tiny-GGUF", p.GetProperty("from").GetString());
            Assert.Equal("cpu", p.GetProperty("device").GetString());
        }
        finally
        {
            foreach (var (k, v) in saved) Environment.SetEnvironmentVariable(k, v);
            try { Directory.Delete(root, recursive: true); } catch (IOException) { }
        }
    }

    // ------------------------------------------------------------------------------------------------ model info

    [Fact]
    public void ModelInfo_Json_MatchesSchema_AndMarksLocalFiles()
    {
        var model = new HuggingFaceModelInfo
        {
            Id = "acme/tiny-GGUF", Author = "acme", Downloads = 1234, Likes = 5, PipelineTag = "text-generation", Tags = ["gguf", "llama"],
        };
        var files = new List<RepoFileEntry>
        {
            new() { Path = "tiny.Q8_0.gguf", Size = 800 },
            new() { Path = "tiny.Q4_K_M.gguf", Size = 500 },
        };
        var local = new List<LocalModel> { new("acme/tiny-GGUF", "tiny.Q4_K_M.gguf", Path.Combine(Path.GetTempPath(), "m", "tiny.Q4_K_M.gguf"), 500, DateTimeOffset.UnixEpoch) };

        string json = JsonSerializer.Serialize(ModelInfoCommand.BuildJson(model, files, local), CliJsonContext.Default.ModelInfoJsonResult);

        Assert.Empty(SchemaErrors("model-info", json));
        JsonElement doc = JsonDocument.Parse(json).RootElement;
        Assert.Equal(1, doc.GetProperty("version").GetInt32());
        Assert.Equal("acme/tiny-GGUF", doc.GetProperty("repo").GetString());
        Assert.Equal(1234, doc.GetProperty("downloads").GetInt64());
        Assert.Equal("text-generation", doc.GetProperty("pipeline").GetString());
        Assert.Equal(Path.Combine(Path.GetTempPath(), "m"), doc.GetProperty("local_path").GetString());
        JsonElement[] f = doc.GetProperty("gguf_files").EnumerateArray().ToArray();
        Assert.Equal(["tiny.Q4_K_M.gguf", "tiny.Q8_0.gguf"], f.Select(x => x.GetProperty("filename").GetString()));   // sorted by name
        Assert.Equal(["Q4_K_M", "Q8_0"], f.Select(x => x.GetProperty("quant").GetString()));
        Assert.Equal([true, false], f.Select(x => x.GetProperty("local").GetBoolean()));
    }

    [Fact]
    public void ModelInfo_Json_NothingLocal_OmitsLocalPath()
    {
        var model = new HuggingFaceModelInfo { Id = "acme/x", Tags = [] };
        string json = JsonSerializer.Serialize(ModelInfoCommand.BuildJson(model, [], []), CliJsonContext.Default.ModelInfoJsonResult);
        Assert.Empty(SchemaErrors("model-info", json));
        Assert.False(JsonDocument.Parse(json).RootElement.TryGetProperty("local_path", out _));
    }

    // ------------------------------------------------------------------------------------------------ the schema checker itself, and exit codes

    [Fact]
    public void SchemaChecker_RejectsMissingVersion_ExtraFields_AndWrongTypes()
    {
        Assert.NotEmpty(SchemaErrors("ps", """{"url":"u","models":[]}"""));
        Assert.NotEmpty(SchemaErrors("ps", """{"version":1,"url":"u","models":[],"surprise":1}"""));
        Assert.NotEmpty(SchemaErrors("ps", """{"version":1,"url":"u","models":[{"id":"a","active":"yes","size_bytes":1}]}"""));
        Assert.NotEmpty(SchemaErrors("ps", """{"version":2,"url":"u","models":[]}"""));
        Assert.NotEmpty(SchemaErrors("error", """{"error":{"code":"x","message":"m"}}"""));   // exit_code is required
    }

    [Fact]
    public void ExitCodes_UsageErrorsAreTwo_HelpIsZero()
    {
        Assert.Equal(0, Run("--help").Code);
        Assert.Equal(0, Run("ps", "--help").Code);
        Assert.Equal(0, Run("model", "info", "--help").Code);
        Assert.Equal(2, Run("frobnicate").Code);                        // unknown command
        Assert.Equal(2, Run("ps", "--no-such-option").Code);            // unknown option
        Assert.Equal(2, Run("list", "--no-such-option", "--json").Code);
        Assert.Equal(2, Run("model", "info").Code);                     // missing required argument
        Assert.Equal(2, Run("run").Code);                               // missing required argument
        Assert.Equal(1, Run("ps", "--url", "http://127.0.0.1:1").Code); // runtime error: nothing listening
    }

    // ------------------------------------------------------------------------------------------------ failures

    /// <summary>A command that fails at run time by throwing Spectre's <see cref="Spectre.Console.Cli.CommandRuntimeException"/>.</summary>
    private sealed class ThrowingSettings : Spectre.Console.Cli.CommandSettings { }

    private sealed class ThrowingCommand : Spectre.Console.Cli.Command<ThrowingSettings>
    {
        public override int Execute(Spectre.Console.Cli.CommandContext context, ThrowingSettings settings)
            => throw MakeRuntimeException("boom at run time");

        // The constructors are internal to Spectre (it throws the type itself); build one the way a runtime fault inside Spectre would.
        private static Exception MakeRuntimeException(string message)
        {
            var ctor = typeof(Spectre.Console.Cli.CommandRuntimeException).GetConstructors(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic)
                .OrderBy(c => c.GetParameters().Length).First();
            object?[] args = ctor.GetParameters().Select(p => p.ParameterType == typeof(string) ? message : p.ParameterType.IsValueType ? Activator.CreateInstance(p.ParameterType) : null).ToArray();
            return (Exception)ctor.Invoke(args);
        }
    }

    [Fact]
    public void CommandRuntimeException_ThrownByARunningCommand_IsExitOne_ButAParseErrorIsExitTwo()
    {
        static int Go(params string[] args)
        {
            var sw = new StringWriter();
            var console = Spectre.Console.AnsiConsole.Create(new Spectre.Console.AnsiConsoleSettings { Out = new Spectre.Console.AnsiConsoleOutput(sw) });
            var app = new Spectre.Console.Cli.CommandApp();
            app.Configure(c =>
            {
                CliConfiguration.Configure(c);
                c.AddCommand<ThrowingCommand>("boom-test");
                c.ConfigureConsole(console);
            });
            return app.Run(args);
        }
        Assert.Equal(1, Go("boom-test"));              // runtime failure: same exception type as a usage error, but the command had started
        Assert.Equal(2, Go("boom-test", "--nope"));    // parse error on the same command: usage
        Assert.Equal(2, Go("no-such-command"));
    }

    [Fact]
    public async Task ModelInfo_Json_Failures_PrintTheErrorDocument_WithAKindCode_ExitOne()
    {
        static Task<(HuggingFaceModelInfo, IReadOnlyList<RepoFileEntry>)> Throw(Exception ex) => Task.FromException<(HuggingFaceModelInfo, IReadOnlyList<RepoFileEntry>)>(ex);
        var cases = new (Exception Ex, string Code)[]
        {
            (new HttpRequestException("404 not found"), "hub_unreachable"),
            (new TaskCanceledException("timeout"), "hub_unreachable"),
            (new JsonException("bad json"), "hub_bad_response"),
            (new InvalidOperationException("surprise"), "internal_error"),     // used to escape as a stack trace
            (new IOException("disk"), "internal_error"),
        };
        foreach (var (ex, code) in cases)
        {
            var original = Console.Out;
            var sw = new StringWriter();
            Console.SetOut(sw);
            int rc;
            try { rc = await ModelInfoCommand.RunJsonAsync("acme/none", () => Throw(ex)); }
            finally { Console.SetOut(original); }

            Assert.Equal(1, rc);
            Assert.Empty(SchemaErrors("error", sw.ToString()));
            using var doc = JsonDocument.Parse(sw.ToString());
            JsonElement e = doc.RootElement.GetProperty("error");
            Assert.Equal(code, e.GetProperty("code").GetString());
            Assert.Equal(1, e.GetProperty("exit_code").GetInt32());
            Assert.Contains("acme/none", e.GetProperty("message").GetString());
        }
    }

    [Fact]
    public void Ps_Json_NoServer_CodeIsServerUnreachable()
    {
        var (_, stdout) = Run("ps", "--json", "--url", "http://127.0.0.1:1");
        using var doc = JsonDocument.Parse(stdout);
        Assert.Equal("server_unreachable", doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }
}
