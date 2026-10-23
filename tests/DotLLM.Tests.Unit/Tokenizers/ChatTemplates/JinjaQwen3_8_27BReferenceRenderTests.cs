using System.Text.Json;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Renders the real Qwen3.x-style chat template (byte-identical to the one embedded in
/// <c>ggml-org/Qwen3.8-27B-GGUF</c> and <c>prism-ml/Ternary-Bonsai-2-27B-gguf</c>) and compares it
/// against golden output produced by reference Python Jinja2 (see
/// <c>Fixtures/generate_render_cases.py</c>; HF-style sandboxed env, trim_blocks + lstrip_blocks).
/// Covers tool-less chat with reasoning on/off/effort levels, multi-turn, tools and tool calls.
/// </summary>
public class JinjaQwen3_8_27BReferenceRenderTests
{
    private static string Fixture(string name) =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    public static IEnumerable<object[]> CaseNames()
    {
        using var doc = JsonDocument.Parse(Fixture("qwen3.8-27b-render-cases.json"));
        foreach (var c in doc.RootElement.EnumerateArray())
            yield return [c.GetProperty("name").GetString()!];
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesReferenceJinja2(string caseName)
    {
        using var doc = JsonDocument.Parse(Fixture("qwen3.8-27b-render-cases.json"));
        var c = doc.RootElement.EnumerateArray().Single(e => e.GetProperty("name").GetString() == caseName);

        var ctx = (Dictionary<string, object?>)Convert(c.GetProperty("context"))!;
        var ast = new JinjaParser(new JinjaLexer(Fixture("qwen3.8-27b-chat-template.jinja")).Tokenize()).Parse();
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
