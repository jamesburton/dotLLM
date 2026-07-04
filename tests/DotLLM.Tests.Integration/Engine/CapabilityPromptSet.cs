using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Objective, cheap scoring rule for one capability prompt: case-insensitive keyword
/// containment (<c>anyOf</c> / <c>allOf</c>) or a case-insensitive regex match against
/// the generated text. Deliberately NO LLM-judging — pass/fail must be reproducible
/// and free.
/// </summary>
public sealed record CapabilityRule
{
    /// <summary>Rule kind: <c>"keyword"</c> (containment) or <c>"regex"</c> (pattern match).</summary>
    [JsonPropertyName("type")]
    public required string Type { get; init; }

    /// <summary>Keyword rule: pass requires AT LEAST ONE of these substrings (case-insensitive).</summary>
    [JsonPropertyName("anyOf")]
    public string[]? AnyOf { get; init; }

    /// <summary>Keyword rule: pass requires ALL of these substrings (case-insensitive).</summary>
    [JsonPropertyName("allOf")]
    public string[]? AllOf { get; init; }

    /// <summary>Regex rule: pass requires <see cref="Regex.IsMatch(string)"/> (IgnoreCase, Singleline).</summary>
    [JsonPropertyName("pattern")]
    public string? Pattern { get; init; }

    /// <summary>Scores generated text against this rule. Null/empty text always fails.</summary>
    /// <param name="output">The model's generated text (prompt excluded).</param>
    /// <returns>True when the objective rule is satisfied.</returns>
    public bool Score(string? output)
    {
        if (string.IsNullOrWhiteSpace(output))
            return false;

        if (Type == "regex")
            return Regex.IsMatch(output, Pattern!,
                RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.CultureInvariant,
                TimeSpan.FromSeconds(1));

        // keyword rule: allOf (when present) must ALL appear; anyOf (when present) needs >= 1.
        if (AllOf is { Length: > 0 })
            foreach (string needle in AllOf)
                if (!output.Contains(needle, StringComparison.OrdinalIgnoreCase))
                    return false;

        if (AnyOf is { Length: > 0 })
        {
            foreach (string needle in AnyOf)
                if (output.Contains(needle, StringComparison.OrdinalIgnoreCase))
                    return true;
            return false;
        }

        return AllOf is { Length: > 0 };
    }

    /// <summary>Human-readable one-line description of the rule (for the report).</summary>
    public string Describe() => Type switch
    {
        "regex" => $"regex `{Pattern}`",
        _ when AllOf is { Length: > 0 } && AnyOf is { Length: > 0 }
            => $"all of [{string.Join(", ", AllOf)}] + any of [{string.Join(", ", AnyOf)}]",
        _ when AllOf is { Length: > 0 } => $"contains all of [{string.Join(", ", AllOf)}]",
        _ => $"contains any of [{string.Join(", ", AnyOf ?? [])}]",
    };

    /// <summary>Validates structural consistency; throws with the prompt id on malformed rules.</summary>
    /// <param name="promptId">Owning prompt id, used in the exception message.</param>
    public void Validate(string promptId)
    {
        switch (Type)
        {
            case "regex":
                if (string.IsNullOrWhiteSpace(Pattern))
                    throw new InvalidDataException($"Prompt '{promptId}': regex rule requires a 'pattern'.");
                _ = new Regex(Pattern); // throws on invalid pattern
                break;
            case "keyword":
                if (AnyOf is not { Length: > 0 } && AllOf is not { Length: > 0 })
                    throw new InvalidDataException($"Prompt '{promptId}': keyword rule requires 'anyOf' and/or 'allOf'.");
                break;
            default:
                throw new InvalidDataException($"Prompt '{promptId}': unknown rule type '{Type}' (expected 'keyword' or 'regex').");
        }
    }
}

/// <summary>
/// One fixed capability prompt: id, task family (<c>qa</c> / <c>typo</c> / <c>code</c>),
/// the exact prompt text, and its objective <see cref="CapabilityRule"/>.
/// </summary>
public sealed record CapabilityPrompt
{
    /// <summary>Stable unique id (used as the row key in reports).</summary>
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    /// <summary>Task family: <c>qa</c> (short factual QA), <c>typo</c> (OCR/typo correction), <c>code</c> (small code-gen).</summary>
    [JsonPropertyName("family")]
    public required string Family { get; init; }

    /// <summary>Exact prompt text fed to the model (completion-shaped; no chat template).</summary>
    [JsonPropertyName("prompt")]
    public required string Prompt { get; init; }

    /// <summary>Objective pass/fail scoring rule for the generated text.</summary>
    [JsonPropertyName("rule")]
    public required CapabilityRule Rule { get; init; }
}

/// <summary>
/// The FIXED prompt set for the diffusion-vs-AR capability harness (#33), loaded from
/// <c>TestData/capability-prompts.json</c>. The set is checked into the repo so both
/// engines and future runs always score the exact same prompts.
/// </summary>
public sealed class CapabilityPromptSet
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    /// <summary>The prompts, in file order.</summary>
    public IReadOnlyList<CapabilityPrompt> Prompts { get; }

    private CapabilityPromptSet(IReadOnlyList<CapabilityPrompt> prompts) => Prompts = prompts;

    /// <summary>Loads the checked-in default prompt set from the test output directory.</summary>
    public static CapabilityPromptSet LoadDefault()
        => Load(Path.Combine(AppContext.BaseDirectory, "TestData", "capability-prompts.json"));

    /// <summary>Loads and validates a prompt set from a JSON file (top-level array of prompts).</summary>
    /// <param name="path">Path to the JSON prompt-set file.</param>
    /// <exception cref="InvalidDataException">Empty set, duplicate ids, or a malformed rule.</exception>
    public static CapabilityPromptSet Load(string path)
    {
        using FileStream fs = File.OpenRead(path);
        var prompts = JsonSerializer.Deserialize<List<CapabilityPrompt>>(fs, JsonOptions)
            ?? throw new InvalidDataException($"Prompt set '{path}' deserialized to null.");

        if (prompts.Count == 0)
            throw new InvalidDataException($"Prompt set '{path}' is empty.");

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (CapabilityPrompt p in prompts)
        {
            if (!ids.Add(p.Id))
                throw new InvalidDataException($"Prompt set '{path}': duplicate id '{p.Id}'.");
            p.Rule.Validate(p.Id);
        }

        return new CapabilityPromptSet(prompts);
    }
}
