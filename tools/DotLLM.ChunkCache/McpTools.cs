using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace DotLLM.ChunkCache;

/// <summary>Server-level defaults of <c>chunkcache mcp</c>: what a tool call falls back to for the parameters the §7.2 schemas leave to configuration.</summary>
internal sealed record McpDefaults(string? Store, string? Server, string? Model, int? Threads);

/// <summary>A tool argument that is invalid in a way the schema cannot say (reported as a tool error, nothing runs).</summary>
internal sealed class McpArgumentException(string message) : Exception(message);

/// <summary>One command line for the CLI and the temp files it reads (deleted after the run).</summary>
internal sealed record McpCall(IReadOnlyList<string> Argv, IReadOnlyList<string> TempFiles);

/// <summary>One MCP tool: its schemas and how its arguments become the equivalent <c>chunkcache &lt;verb&gt; --json</c> command line.</summary>
internal sealed class McpToolDef(string name, string description, JsonObject inputSchema, JsonObject outputSchema, Func<JsonObject, McpDefaults, McpCall> plan, bool readOnly, bool destructive)
{
    public string Name { get; } = name;

    public string Description { get; } = description;

    public JsonObject InputSchema { get; } = inputSchema;

    public JsonObject OutputSchema { get; } = outputSchema;

    public Func<JsonObject, McpDefaults, McpCall> Plan { get; } = plan;

    public bool ReadOnly { get; } = readOnly;

    public bool Destructive { get; } = destructive;
}

/// <summary>
/// The six <c>chunkcache_*</c> tools (integration spec §7.2). A tool is a thin wrapper: its arguments are validated against the
/// input schema, turned into the argument list of the matching verb with <c>--json</c> and run in process through
/// <see cref="ChunkCacheCli.Run"/>; the verb's JSON document is projected onto the output schema (the CLI may print more) and
/// returned as <c>structuredContent</c>. Nothing else happens here.
/// </summary>
internal static class McpTools
{
    private const string ChunkerProp = """{"enum": ["symbol-512", "lines-512"], "default": "symbol-512", "description": "the wire-eligible chunkers this build accepts"}""";

    private static readonly Lazy<IReadOnlyList<McpToolDef>> AllTools = new(Build);

    /// <summary>The tools, in the order <c>tools/list</c> returns them.</summary>
    public static IReadOnlyList<McpToolDef> All => AllTools.Value;

    /// <summary>The tool named <paramref name="name"/>, or null.</summary>
    public static McpToolDef? Find(string name) => All.FirstOrDefault(t => string.Equals(t.Name, name, StringComparison.Ordinal));

    private static IReadOnlyList<McpToolDef> Build() =>
    [
        new("chunkcache_build",
            "Scan a solution or folder, chunk it with a wire-eligible chunker and update the store's index (tokenizer only; nothing is encoded unless encode is true, which queues the missing chunks on the server).",
            Input($$$$"""
                {"required": ["path"], "properties": {
                  "store": {"type": "string"}, "server": {"type": "string"}, "model": {"type": "string", "description": "GGUF for the tokenizer and chunk keys; default: the --model chunkcache mcp was started with"},
                  "path": {"type": "string", "description": ".sln/.slnx/.csproj or folder"},
                  "chunker": {{{{ChunkerProp}}}},
                  "configuration": {"type": "string", "default": "Release"},
                  "encode": {"type": "boolean", "default": false, "description": "queue all missing chunks via the server's prewarm"},
                  "max_encode_tokens": {"type": "integer", "minimum": 0}}}
                """),
            Output("build"), PlanBuild, readOnly: false, destructive: false),
        new("chunkcache_select",
            "Choose the chunks for a context (paths/globs, a symbol's neighbourhood, git changes) under a token budget, from an index and its reference graph.",
            Input($$$$"""
                {"properties": {
                  "store": {"type": "string"}, "server": {"type": "string"}, "model": {"type": "string", "description": "GGUF for the tokenizer and chunk keys; default: the --model chunkcache mcp was started with"},
                  "path": {"type": "string", "description": "solution/folder whose index to use"},
                  "chunker": {{{{ChunkerProp}}}},
                  "paths": {"type": "array", "items": {"type": "string"}}, "exclude": {"type": "array", "items": {"type": "string"}},
                  "symbol": {"type": "string"}, "hops": {"type": "integer", "default": 1, "minimum": 0, "maximum": 100},
                  "changed_since": {"type": "string"}, "token_budget": {"type": "integer", "minimum": 0},
                  "include_text": {"type": "boolean", "default": false},
                  "order": {"enum": ["rank", "document"], "default": "document"}}}
                """),
            Output("select"), PlanSelect, readOnly: true, destructive: false),
        new("chunkcache_status",
            "Which chunks are missing or stale for a tree (by file and line range), and the store's statistics.",
            Input($$$$"""
                {"properties": {"store": {"type": "string"}, "server": {"type": "string"}, "model": {"type": "string", "description": "GGUF for the tokenizer and chunk keys; default: the --model chunkcache mcp was started with"}, "path": {"type": "string"},
                  "chunker": {{{{ChunkerProp}}}},
                  "list_missing": {"type": "boolean", "default": false}}}
                """),
            Output("status"), PlanStatus, readOnly: true, destructive: false),
        new("chunkcache_evict",
            "Evict chunk entries through the server (or, with only a store, from that store directly): orphans, a path glob, a namespace, or all. A dry run unless dry_run is false.",
            Input("""
                {"properties": {"store": {"type": "string"}, "server": {"type": "string"}, "model": {"type": "string", "description": "GGUF for the tokenizer and chunk keys; default: the --model chunkcache mcp was started with"},
                  "mode": {"enum": ["orphans", "path", "namespace", "all"], "default": "orphans"},
                  "path": {"type": "string", "description": "glob; mode=path: forget every entry first seen under it"},
                  "namespace": {"type": "string", "description": "admin keys only; others act on their own namespace"},
                  "dry_run": {"type": "boolean", "default": true}}}
                """),
            Output("evict"), PlanEvict, readOnly: false, destructive: true),
        new("chunkcache_prewarm",
            "Queue isolated encodes of chunk texts on the server (text only: a chunkcache_select input read locally, or explicit texts).",
            Input("""
                {"properties": {"store": {"type": "string"}, "server": {"type": "string"}, "model": {"type": "string", "description": "GGUF for the tokenizer and chunk keys; default: the --model chunkcache mcp was started with"},
                  "select": {"type": "object", "additionalProperties": true, "description": "a chunkcache_select input; text is read locally"},
                  "texts": {"type": "array", "items": {"type": "object", "required": ["id", "text"], "additionalProperties": false,
                     "properties": {"id": {"type": "string"}, "text": {"type": "string"}}}},
                  "max_tokens": {"type": "integer", "minimum": 1}}}
                """),
            Output("prewarm"), PlanPrewarm, readOnly: false, destructive: false),
        new("chunkcache_explain",
            "A dry run of the chunk_cache plan for a request body on the server: which chunks would be prefix, hit, miss or ignored, and why.",
            Input("""
                {"required": ["request"], "properties": {"server": {"type": "string"},
                  "request": {"type": "object", "additionalProperties": true}}}
                """),
            Output("explain"), PlanExplain, readOnly: true, destructive: false),
    ];

    // ---------------------------------------------------------------------------------------------------- schemas

    /// <summary>An input schema: the §7.2 body, <c>type: object</c>, no properties beyond the listed ones.</summary>
    private static JsonObject Input(string body)
    {
        JsonObject schema = JsonNode.Parse(body)!.AsObject();
        var result = new JsonObject { ["type"] = "object" };
        foreach ((string k, JsonNode? v) in schema.ToArray())
        {
            schema.Remove(k);
            result[k] = v;
        }
        result["additionalProperties"] = false;
        return result;
    }

    /// <summary>The §7.2 output schema of <paramref name="verb"/> (an embedded copy of the contract's), closed to extra properties.</summary>
    private static JsonObject Output(string verb)
    {
        Assembly asm = typeof(McpTools).Assembly;
        string resource = asm.GetManifestResourceNames().Single(n => n.EndsWith($".{verb}.output.json", StringComparison.Ordinal));
        using Stream s = asm.GetManifestResourceStream(resource)!;
        JsonObject schema = JsonNode.Parse(s)!.AsObject();
        schema.Remove("$schema");
        Close(schema);
        return schema;
    }

    private static void Close(JsonNode? node)
    {
        if (node is JsonObject o)
        {
            if (o["properties"] is JsonObject props)
            {
                o["additionalProperties"] = false;
                foreach ((_, JsonNode? v) in props) Close(v);
            }
            if (o["items"] is { } items) Close(items);
        }
    }

    // ---------------------------------------------------------------------------------------------------- argument -> command line

    private sealed class Argv
    {
        public List<string> Items { get; } = [];

        public List<string> Temp { get; } = [];

        public void Flag(string name) => Items.Add("--" + name);

        /// <summary>
        /// Adds <c>--name=value</c> as ONE token. A value is caller-controlled text: as a token of its own it could be read as an
        /// option (and the CLI strips a bare <c>--verbose</c> anywhere, which would shift the next value into a flag such as
        /// <c>--apply</c>). The inline form keeps it a value whatever it contains.
        /// </summary>
        public void Add(string name, string? value)
        {
            if (value is null) return;
            Items.Add("--" + name + "=" + value);
        }

        public McpCall Done() => new([.. Items, "--json"], Temp);
    }

    private static string? Str(JsonObject a, string name) => a[name] is JsonValue v && v.TryGetValue(out string? s) ? s : null;

    private static bool Bool(JsonObject a, string name, bool fallback) => a[name] is JsonValue v && v.TryGetValue(out bool b) ? b : fallback;

    /// <summary>
    /// An integer argument as the CLI spells it. A JSON integer may be written <c>100</c>, <c>100.0</c> or <c>1e2</c> (all pass the
    /// schema's <c>integer</c>); each must reach the verb, because a dropped <c>max_encode_tokens</c> is an unbounded encode.
    /// </summary>
    private static string? Num(JsonObject a, string name)
    {
        if (a[name] is not JsonValue v) return null;
        if (v.TryGetValue(out long n)) return n.ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (v.TryGetValue(out double d) && d == Math.Floor(d) && Math.Abs(d) < 9e15) return ((long)d).ToString(System.Globalization.CultureInfo.InvariantCulture);
        throw new McpArgumentException($"{name}: not an integer in range.");
    }

    private static IEnumerable<string> Strings(JsonObject a, string name) =>
        a[name] is JsonArray arr ? arr.Select(x => x!.GetValue<string>()) : [];

    /// <summary><c>symbol-512</c> becomes <c>--chunker symbol --chunk-tokens 512</c>.</summary>
    private static void Chunker(Argv v, JsonObject a)
    {
        string? name = Str(a, "chunker");
        if (name is null) return;
        int dash = name.LastIndexOf('-');
        v.Add("chunker", name[..dash]);
        v.Add("chunk-tokens", name[(dash + 1)..]);
    }

    /// <summary>A <c>.sln</c>, <c>.slnx</c> or <c>.csproj</c> is a solution; anything else is a directory.</summary>
    private static void Source(Argv v, string path)
    {
        string ext = Path.GetExtension(path);
        bool solution = ext.Equals(".sln", StringComparison.OrdinalIgnoreCase) || ext.Equals(".slnx", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".csproj", StringComparison.OrdinalIgnoreCase);
        v.Add(solution ? "solution" : "directory", path);
    }

    private static string? Server(JsonObject a, McpDefaults d) => Str(a, "server") ?? d.Server;

    private static void ModelOptions(Argv v, JsonObject a, McpDefaults d)
    {
        v.Add("model", Str(a, "model") ?? d.Model); // a per-call GGUF, else the one the server was started with
        v.Add("threads", d.Threads?.ToString(System.Globalization.CultureInfo.InvariantCulture));
    }

    private static McpCall PlanBuild(JsonObject a, McpDefaults d)
    {
        var v = new Argv { Items = { "build" } };
        string? server = Server(a, d);
        v.Add("store", Str(a, "store") ?? d.Store);
        Source(v, Str(a, "path")!);
        Chunker(v, a);
        v.Add("configuration", Str(a, "configuration"));
        ModelOptions(v, a, d);
        v.Add("server", server);
        if (Bool(a, "encode", false))
        {
            if (server is null) throw new McpArgumentException("encode: true needs a server (the server owns the model identity and every K/V write): pass server or start 'chunkcache mcp --server URL'.");
            v.Flag("encode");
        }
        else
        {
            v.Flag("index-only"); // the default: tokenizer only, nothing is queued or encoded
        }
        v.Add("max-encode-tokens", Num(a, "max_encode_tokens")); // tokens, never --max-chunks
        return v.Done();
    }

    private static void SelectOptions(Argv v, JsonObject a, bool outputOptions)
    {
        foreach (string p in Strings(a, "paths")) v.Add("path", p);
        foreach (string p in Strings(a, "exclude")) v.Add("exclude", p);
        v.Add("symbol", Str(a, "symbol"));
        v.Add("hops", Num(a, "hops"));
        v.Add("since", Str(a, "changed_since"));
        v.Add("budget", Num(a, "token_budget"));
        if (outputOptions)
        {
            v.Add("order", Str(a, "order"));
            if (Bool(a, "include_text", false)) v.Flag("include-text");
        }
    }

    private static McpCall PlanSelect(JsonObject a, McpDefaults d)
    {
        var v = new Argv { Items = { "select" } };
        v.Add("store", Str(a, "store") ?? d.Store);
        if (Str(a, "path") is { } path) Source(v, path);
        Chunker(v, a);
        ModelOptions(v, a, d);
        v.Add("server", Server(a, d));
        SelectOptions(v, a, outputOptions: true);
        return v.Done();
    }

    private static McpCall PlanStatus(JsonObject a, McpDefaults d)
    {
        var v = new Argv { Items = { "status" } };
        v.Add("store", Str(a, "store") ?? d.Store);
        if (Str(a, "path") is { } path) Source(v, path);
        else v.Flag("no-scan"); // without a tree: the store against its index
        Chunker(v, a);
        ModelOptions(v, a, d);
        v.Add("server", Server(a, d));
        if (Bool(a, "list_missing", false)) v.Flag("list-missing");
        return v.Done();
    }

    private static McpCall PlanEvict(JsonObject a, McpDefaults d)
    {
        var v = new Argv { Items = { "evict" } };
        string mode = Str(a, "mode") ?? "orphans";
        string? server = Server(a, d);
        v.Add("mode", mode);
        v.Add("path", Str(a, "path"));
        v.Add("namespace", Str(a, "namespace"));
        if (server is not null) v.Add("server", server);
        else
        {
            v.Flag("offline");
            v.Add("store", Str(a, "store") ?? d.Store);
            if (mode == "all") ModelOptions(v, a, d);
        }
        if (!Bool(a, "dry_run", true)) v.Flag("apply"); // a dry run unless dry_run is false
        return v.Done();
    }

    private static McpCall PlanPrewarm(JsonObject a, McpDefaults d)
    {
        var v = new Argv { Items = { "prewarm" } };
        v.Add("server", Server(a, d) ?? throw new McpArgumentException("prewarm needs a server: pass server or start 'chunkcache mcp --server URL'."));
        bool hasTexts = a["texts"] is JsonArray, hasSelect = a["select"] is JsonObject;
        if (hasTexts == hasSelect) throw new McpArgumentException("prewarm: give exactly one of texts or select.");
        v.Add("max-tokens", Num(a, "max_tokens"));
        if (hasTexts)
        {
            string file = TempFile(v, a["texts"]!.ToJsonString());
            v.Add("texts", file);
        }
        else
        {
            JsonObject sel = a["select"]!.AsObject();
            McpToolDef select = Find("chunkcache_select")!;
            string? error = McpSchema.Validate(sel, select.InputSchema, "select").FirstOrDefault();
            if (error is not null) throw new McpArgumentException(error);
            if (Str(sel, "path") is not { } path) throw new McpArgumentException("prewarm: select.path (the solution or folder to read the chunk text from) is required.");
            Source(v, path);
            Chunker(v, sel);
            ModelOptions(v, Str(a, "model") is null && Str(sel, "model") is { } m ? new JsonObject { ["model"] = m } : a, d); // select.model counts too
            SelectOptions(v, sel, outputOptions: false);
        }
        return v.Done();
    }

    private static McpCall PlanExplain(JsonObject a, McpDefaults d)
    {
        var v = new Argv { Items = { "explain" } };
        v.Add("server", Server(a, d) ?? throw new McpArgumentException("explain needs a server: pass server or start 'chunkcache mcp --server URL'."));
        v.Add("request", TempFile(v, a["request"]!.ToJsonString()));
        return v.Done();
    }

    private static string TempFile(Argv v, string content)
    {
        string dir = Path.Combine(Path.GetTempPath(), "chunkcache-mcp-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string file = Path.Combine(dir, "arg.json");
        File.WriteAllText(file, content, new System.Text.UTF8Encoding(false));
        v.Temp.Add(dir);
        return file;
    }
}

/// <summary>The subset of JSON Schema the tool schemas use: type, enum, required, properties, additionalProperties, items, minimum, maximum.</summary>
internal static class McpSchema
{
    /// <summary>Violations of <paramref name="value"/> against <paramref name="schema"/> (empty = valid), each naming the property.</summary>
    public static List<string> Validate(JsonNode? value, JsonNode schema, string path)
    {
        var errors = new List<string>();
        Check(value, schema, path, errors);
        return errors;
    }

    private static void Check(JsonNode? value, JsonNode schema, string path, List<string> errors)
    {
        if (schema["type"]?.GetValue<string>() is { } type)
        {
            JsonValueKind kind = value is null ? JsonValueKind.Null : value.GetValueKind();
            bool ok = type switch
            {
                "object" => kind == JsonValueKind.Object,
                "array" => kind == JsonValueKind.Array,
                "string" => kind == JsonValueKind.String,
                "boolean" => kind is JsonValueKind.True or JsonValueKind.False,
                "number" => kind == JsonValueKind.Number,
                "integer" => kind == JsonValueKind.Number && value is JsonValue jv && (jv.TryGetValue(out long _) || (jv.TryGetValue(out double dd) && dd == Math.Floor(dd))),
                _ => true,
            };
            if (!ok)
            {
                errors.Add($"{path}: expected {type}, found {kind}");
                return;
            }
        }
        if (schema["enum"] is JsonArray allowed && !allowed.Any(a => a?.ToJsonString() == value?.ToJsonString()))
            errors.Add($"{path}: {value?.ToJsonString() ?? "null"} is not one of {allowed.ToJsonString()}");
        if (value is JsonValue num && value.GetValueKind() == JsonValueKind.Number && num.TryGetValue(out double d))
        {
            if (schema["minimum"] is { } min && d < min.GetValue<double>()) errors.Add($"{path}: {d} is below the minimum {min}");
            if (schema["maximum"] is { } max && d > max.GetValue<double>()) errors.Add($"{path}: {d} is above the maximum {max}");
        }
        if (value is JsonObject obj)
        {
            if (schema["required"] is JsonArray required)
                foreach (JsonNode? r in required)
                    if (!obj.ContainsKey(r!.GetValue<string>())) errors.Add($"{path}: missing required property {r.GetValue<string>()}");
            JsonObject? props = schema["properties"] as JsonObject;
            bool closed = schema["additionalProperties"] is JsonValue ap && ap.GetValueKind() == JsonValueKind.False;
            foreach ((string k, JsonNode? v) in obj)
            {
                if (props is not null && props[k] is { } sub) Check(v, sub, path + "." + k, errors);
                else if (closed) errors.Add($"{path}: unknown property {k} (the schema allows {string.Join(", ", props?.Select(p => p.Key) ?? [])})");
            }
        }
        if (value is JsonArray arr && schema["items"] is { } items)
        {
            int i = 0;
            foreach (JsonNode? item in arr) Check(item, items, $"{path}[{i++}]", errors);
        }
    }

    /// <summary>Copies of <paramref name="value"/> keeping only the properties <paramref name="schema"/> lists, recursively.</summary>
    public static JsonNode Project(JsonNode value, JsonNode schema)
    {
        if (value is JsonObject obj && schema["properties"] is JsonObject props)
        {
            var result = new JsonObject();
            foreach ((string k, JsonNode? v) in obj)
                if (props[k] is { } sub && v is not null) result[k] = Project(v, sub);
            return result;
        }
        if (value is JsonArray arr && schema["items"] is { } items)
        {
            var result = new JsonArray();
            foreach (JsonNode? v in arr) result.Add(v is null ? null : Project(v, items));
            return result;
        }
        return value.DeepClone();
    }
}
