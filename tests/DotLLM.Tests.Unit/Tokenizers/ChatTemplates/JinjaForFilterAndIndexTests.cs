using System.Text.Json;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Three Jinja behaviours the Nemotron 3.5 Lightning template depends on (issue #928): negative list/string
/// indices, the <c>for x in seq if cond</c> filter, and single-variable iteration over a dict (keys).
/// The golden class below renders the real template and compares byte for byte with llama.cpp's minja.
/// </summary>
public class JinjaForFilterAndIndexTests
{
    private static string Eval(string template, Dictionary<string, object?>? vars = null)
    {
        var ast = new JinjaParser(new JinjaLexer(template).Tokenize()).Parse();
        return new JinjaEvaluator(vars ?? new Dictionary<string, object?>(StringComparer.Ordinal)).Evaluate(ast);
    }

    [Theory]
    [InlineData("{{ [1,2,3][-1] }}", "3")]
    [InlineData("{{ [1,2,3][-3] }}", "1")]
    [InlineData("{{ 'abc'[-1] }}", "c")]
    [InlineData("{{ 'a</think>b'.split('</think>')[-1] }}", "b")]
    [InlineData("{% set l = [10,20,30] %}{{ l[-2] }}", "20")]
    [InlineData("{{ -3 + 5 }}", "2")]
    public void NegativeNumbersStayIntegersSoNegativeIndexingWorks(string template, string expected)
        => Assert.Equal(expected, Eval(template));

    [Fact]
    public void ForLoopIfFilter_DropsNonMatchingItems()
        => Assert.Equal("[3][4]", Eval("{% for x in [1,2,3,4] if x > 2 %}[{{ x }}]{% endfor %}"));

    [Fact]
    public void ForLoopIfFilter_LoopVariablesCountOnlyTheSurvivors()
        => Assert.Equal("0:3 1:4 last", Eval(
            "{% for x in [1,2,3,4] if x > 2 %}{{ loop.index0 }}:{{ x }}{% if loop.last %} last{% else %} {% endif %}{% endfor %}"));

    [Fact]
    public void ForLoopIfFilter_AllFilteredOut_RunsTheElseBlock()
        => Assert.Equal("none", Eval("{% for x in [1,2] if x > 5 %}{{ x }}{% else %}none{% endfor %}"));

    [Fact]
    public void ForLoopIfFilter_OverDictKeys_NotInHandledList()
    {
        // The exact shape of render_extra_keys in the Nemotron template.
        var vars = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["d"] = new Dictionary<string, object?>(StringComparer.Ordinal) { ["type"] = "string", ["x"] = 1, ["y"] = 2 },
            ["handled"] = new List<object?> { "type" },
        };
        Assert.Equal("[x=1][y=2]", Eval("{% for k in d if k not in handled %}[{{ k }}={{ d[k] }}]{% endfor %}", vars));
    }

    [Fact]
    public void DictSingleVariableIteration_YieldsKeys()
    {
        var vars = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["d"] = new Dictionary<string, object?>(StringComparer.Ordinal) { ["a"] = 1, ["b"] = 2 },
        };
        Assert.Equal("[a][b]", Eval("{% for k in d %}[{{ k }}]{% endfor %}", vars));
        // Two variables over items() and over the bare dict still unpack key/value.
        Assert.Equal("[a=1][b=2]", Eval("{% for k, v in d|items %}[{{ k }}={{ v }}]{% endfor %}", vars));
        Assert.Equal("[a=1][b=2]", Eval("{% for k, v in d %}[{{ k }}={{ v }}]{% endfor %}", vars));
    }

    // trim_blocks + lstrip_blocks (HF and llama.cpp minja both enable them; templates are written against them).
    [Theory]
    [InlineData("a\n{% if true %}\nb\n{% endif %}\nc", "a\nb\nc")]                       // newline after a block tag is dropped
    [InlineData("x\n    {% if true %}y{% endif %}\nz", "x\nyz")]                          // indentation before a block tag is dropped
    [InlineData("x\n    {%+ if true %}y{% endif %}", "x\n    y")]                         // {%+ keeps it
    [InlineData("a{{ 1 }}\nb", "a1\nb")]                                                // variable tags are never trimmed
    [InlineData("a {% if true %}b{% endif %}", "a b")]                                  // a tag mid-line keeps the space before it
    [InlineData("{% macro m() %}\n{%- if true %}x{%- endif %}\n{% endmacro %}[{{ m() }}]", "[x]")]  // the Nemotron render_extra_keys shape
    [InlineData("  {% if true %}\nk\n  {% endif %}\n", "k\n")]                           // indented block at the very start of the template
    public void TrimBlocksAndLstripBlocks(string template, string expected)
        => Assert.Equal(expected, Eval(template));

    [Fact]
    public void TrimAndLstripCanBeSwitchedOff()
    {
        var ast = new JinjaParser(new JinjaLexer("a\n  {% if true %}\nb{% endif %}", trimBlocks: false, lstripBlocks: false).Tokenize()).Parse();
        Assert.Equal("a\n  \nb", new JinjaEvaluator(new Dictionary<string, object?>(StringComparer.Ordinal)).Evaluate(ast));
    }

    [Fact]
    public void ParenthesisedTernaryStillWorksAsTheIterable()
        => Assert.Equal("[1][2]", Eval("{% for x in ([1,2] if true else [9]) %}[{{ x }}]{% endfor %}"));
}

/// <summary>
/// Renders the real Nemotron 3.5 Lightning chat template (from the ollama / ggml-org GGUF) and compares with the output
/// of llama.cpp b8683's minja (<c>llama-server --jinja</c>, <c>/apply-template</c>) on the same messages and tools.
/// Before #928 the assistant history content was dropped (<c>split(...)[-1]</c>) and the tool schema emitted
/// <c>&lt;[type, string]&gt;</c> junk (<c>for k in d if k not in handled</c>).
/// </summary>
public class JinjaNemotronLightningGoldenTests
{
    private static string Fixture(string name) =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    public static IEnumerable<object[]> CaseNames()
    {
        using var doc = JsonDocument.Parse(Fixture("nemotron-3.5-lightning-render-cases.json"));
        foreach (var c in doc.RootElement.EnumerateArray())
            yield return [c.GetProperty("name").GetString()!];
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesLlamaCppMinja(string caseName)
    {
        using var doc = JsonDocument.Parse(Fixture("nemotron-3.5-lightning-render-cases.json"));
        var c = doc.RootElement.EnumerateArray().Single(e => e.GetProperty("name").GetString() == caseName);
        var ctx = (Dictionary<string, object?>)Convert(c.GetProperty("context"))!;
        var ast = new JinjaParser(new JinjaLexer(Fixture("nemotron-3.5-lightning-chat-template.jinja")).Tokenize()).Parse();
        Assert.Equal(c.GetProperty("expected").GetString(), new JinjaEvaluator(ctx).Evaluate(ast));
    }

    private static object? Convert(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.Object => e.EnumerateObject().ToDictionary(p => p.Name, p => Convert(p.Value)),
        JsonValueKind.Array => e.EnumerateArray().Select(Convert).ToList(),
        JsonValueKind.String => e.GetString(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Number => e.TryGetInt64(out var l) ? (int)l : e.GetDouble(),
        _ => null,
    };
}
