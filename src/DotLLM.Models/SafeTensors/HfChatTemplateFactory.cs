using System.Text.Json;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

namespace DotLLM.Models.SafeTensors;

/// <summary>Loads a checkpoint's HuggingFace chat template without changing its tokenizer.</summary>
public static class HfChatTemplateFactory
{
    /// <summary>
    /// Loads a string or named template collection from tokenizer_config.json. Standalone
    /// chat_template.jinja and additional_chat_templates/*.jinja override embedded entries.
    /// Selects tool_use when tools are requested and that entry exists, otherwise default.
    /// Returns null only when no template is declared; malformed or ambiguous declarations fail.
    /// </summary>
    /// <param name="directory">HuggingFace checkpoint directory.</param>
    /// <param name="tokenizer">Existing tokenizer, used for BOS/EOS and control-token guarding.</param>
    /// <param name="useTools">Whether tool definitions will be supplied to the selected template.</param>
    public static JinjaChatTemplate? TryCreate(string directory, ITokenizer tokenizer, bool useTools = false)
    {
        ArgumentNullException.ThrowIfNull(directory);
        ArgumentNullException.ThrowIfNull(tokenizer);
        var templates = new Dictionary<string, string>(StringComparer.Ordinal);
        string configPath = Path.Combine(directory, "tokenizer_config.json");
        if (File.Exists(configPath))
        {
            try
            {
                using var stream = File.OpenRead(configPath);
                using var document = JsonDocument.Parse(stream);
                if (document.RootElement.ValueKind != JsonValueKind.Object)
                    throw Invalid(configPath, "expected a JSON object");
                if (document.RootElement.TryGetProperty("chat_template", out var declaration)
                    && declaration.ValueKind != JsonValueKind.Null)
                    ReadTemplates(declaration, templates, configPath);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"Invalid HuggingFace chat template configuration '{configPath}': {ex.Message}", ex);
            }
        }

        string standalonePath = Path.Combine(directory, "chat_template.jinja");
        if (File.Exists(standalonePath))
            templates["default"] = ReadStandalone(standalonePath);
        string additionalPath = Path.Combine(directory, "additional_chat_templates");
        if (Directory.Exists(additionalPath))
            foreach (string path in Directory.EnumerateFiles(additionalPath, "*.jinja", SearchOption.TopDirectoryOnly))
                templates[Path.GetFileNameWithoutExtension(path)] = ReadStandalone(path);

        if (templates.Count == 0)
            return null;
        string name = useTools && templates.ContainsKey("tool_use") ? "tool_use" : "default";
        if (!templates.TryGetValue(name, out string? source))
            throw Invalid(configPath, "named templates require a 'default' entry (or 'tool_use' when tools are requested)");

        return new JinjaChatTemplate(source,
            DecodeSpecial(tokenizer, tokenizer.BosTokenId),
            DecodeSpecial(tokenizer, tokenizer.EosTokenId), tokenizer);
    }

    private static string DecodeSpecial(ITokenizer tokenizer, int id)
        => id >= 0 && id < tokenizer.VocabSize ? tokenizer.DecodeToken(id) : string.Empty;

    private static string ReadStandalone(string path)
    {
        string source = File.ReadAllText(path);
        if (string.IsNullOrWhiteSpace(source))
            throw Invalid(path, "template must not be empty");
        return source;
    }

    private static void ReadTemplates(JsonElement declaration, Dictionary<string, string> templates, string path)
    {
        if (declaration.ValueKind == JsonValueKind.String)
        {
            if (declaration.GetString()!.Length == 0) return;
            Add("default", declaration);
            return;
        }
        if (declaration.ValueKind == JsonValueKind.Object)
        {
            foreach (var entry in declaration.EnumerateObject()) Add(entry.Name, entry.Value);
        }
        else if (declaration.ValueKind == JsonValueKind.Array)
        {
            foreach (var entry in declaration.EnumerateArray())
            {
                if (entry.ValueKind != JsonValueKind.Object
                    || !entry.TryGetProperty("name", out var name) || name.ValueKind != JsonValueKind.String
                    || !entry.TryGetProperty("template", out var source))
                    throw Invalid(path, "each named template must have string 'name' and 'template' fields");
                Add(name.GetString()!, source);
            }
        }
        else
        {
            throw Invalid(path, "chat_template must be a string, a name-to-template object, or a list of named templates");
        }
        if (templates.Count == 0)
            throw Invalid(path, "named template collection must not be empty");

        void Add(string name, JsonElement value)
        {
            if (string.IsNullOrWhiteSpace(name) || value.ValueKind != JsonValueKind.String
                || string.IsNullOrWhiteSpace(value.GetString()))
                throw Invalid(path, "template names and sources must be nonempty strings");
            if (!templates.TryAdd(name, value.GetString()!))
                throw Invalid(path, $"duplicate template name '{name}'");
        }
    }

    private static InvalidDataException Invalid(string path, string reason)
        => new($"Invalid HuggingFace chat template configuration '{path}': {reason}.");
}
