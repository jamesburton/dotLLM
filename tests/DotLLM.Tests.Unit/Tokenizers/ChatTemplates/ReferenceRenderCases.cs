using System.Text.Json;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Replays a <c>*-render-cases.json</c> fixture (written by a Python reference renderer under <c>Fixtures/</c>) against a chat
/// template: each case holds the template context and either the expected render or the expected error text.
/// </summary>
internal static class ReferenceRenderCases
{
    /// <summary>Reads a fixture from the test output directory.</summary>
    /// <param name="name">File name under <c>Tokenizers/ChatTemplates/Fixtures</c>.</param>
    /// <returns>The file text.</returns>
    public static string Fixture(string name) =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    /// <summary>Case names in <paramref name="casesFile"/>, as xUnit member data.</summary>
    /// <param name="casesFile">The render-cases JSON fixture.</param>
    /// <returns>One row per case.</returns>
    public static IEnumerable<object[]> Names(string casesFile)
    {
        using var doc = JsonDocument.Parse(Fixture(casesFile));
        foreach (var c in doc.RootElement.EnumerateArray())
            yield return [c.GetProperty("name").GetString()!];
    }

    /// <summary>Renders case <paramref name="caseName"/> and asserts it equals the reference output (or error).</summary>
    /// <param name="templateFile">The Jinja template fixture.</param>
    /// <param name="casesFile">The render-cases JSON fixture.</param>
    /// <param name="caseName">The case to replay.</param>
    public static void AssertMatches(string templateFile, string casesFile, string caseName)
    {
        using var doc = JsonDocument.Parse(Fixture(casesFile));
        var c = doc.RootElement.EnumerateArray().Single(e => e.GetProperty("name").GetString() == caseName);

        var ctx = (Dictionary<string, object?>)Convert(c.GetProperty("context"))!;
        var ast = new JinjaParser(new JinjaLexer(Fixture(templateFile)).Tokenize()).Parse();
        var evaluator = new JinjaEvaluator(ctx);

        if (c.TryGetProperty("expectedError", out var err))
        {
            var ex = Assert.ThrowsAny<Exception>(() => evaluator.Evaluate(ast));
            Assert.Contains(err.GetString()!, ex.Message, StringComparison.Ordinal);
        }
        else
        {
            Assert.Equal(c.GetProperty("expected").GetString(), evaluator.Evaluate(ast));
        }
    }

    private static object? Convert(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.Object => e.EnumerateObject().ToDictionary(p => p.Name, p => Convert(p.Value)),
        JsonValueKind.Array => e.EnumerateArray().Select(Convert).ToList(),
        JsonValueKind.String => e.GetString(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Number => e.TryGetInt64(out var l) ? l : e.GetDouble(),
        _ => null,
    };
}
