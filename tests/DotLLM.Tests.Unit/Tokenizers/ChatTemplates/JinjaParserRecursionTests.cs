using System.Text;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Regression tests for the recursion-depth guard added to <see cref="JinjaParser"/>.
/// Without the guard, an adversarial chat template with deep paren / bracket nesting
/// blows the stack with an uncatchable <see cref="StackOverflowException"/> — the
/// model loader simply crashes the process. See upstream issue #107 item 4.
/// </summary>
public class JinjaParserRecursionTests
{
    private static JinjaTemplate Parse(string source)
    {
        var tokens = new JinjaLexer(source).Tokenize();
        return new JinjaParser(tokens).Parse();
    }

    /// <summary>
    /// A template with paren nesting well past the depth limit must raise a catchable
    /// <see cref="JinjaException"/>, not crash the process with a stack overflow.
    /// The chosen depth (5000) is well beyond <c>MaxRecursionDepth = 100</c> and
    /// enough to overflow the default thread stack on unfixed code.
    /// </summary>
    [Fact]
    public void DeeplyNestedParens_ThrowsJinjaException_NotStackOverflow()
    {
        const int depth = 5000;
        var sb = new StringBuilder(depth * 2 + 8);
        sb.Append("{{ ");
        for (int i = 0; i < depth; i++) sb.Append('(');
        sb.Append('1');
        for (int i = 0; i < depth; i++) sb.Append(')');
        sb.Append(" }}");

        var ex = Assert.Throws<JinjaException>(() => Parse(sb.ToString()));
        Assert.Contains("recursion depth", ex.Message);
    }

    /// <summary>
    /// Nested if blocks (template-body recursion) must also be guarded.
    /// </summary>
    [Fact]
    public void DeeplyNestedIfBlocks_ThrowsJinjaException_NotStackOverflow()
    {
        const int depth = 5000;
        var sb = new StringBuilder(depth * 12);
        for (int i = 0; i < depth; i++) sb.Append("{% if true %}");
        sb.Append("x");
        for (int i = 0; i < depth; i++) sb.Append("{% endif %}");

        var ex = Assert.Throws<JinjaException>(() => Parse(sb.ToString()));
        Assert.Contains("recursion depth", ex.Message);
    }

    /// <summary>
    /// Realistic chat templates with normal nesting (a couple of layers) parse fine.
    /// Verifies the guard does not regress legitimate templates.
    /// </summary>
    [Fact]
    public void NormalNesting_ParsesWithoutError()
    {
        const string template =
            "{% for msg in messages %}" +
            "  {% if msg.role == 'user' %}{{ msg.content }}{% endif %}" +
            "{% endfor %}";

        var ast = Parse(template);
        Assert.NotNull(ast);
        Assert.NotEmpty(ast.Nodes);
    }

    /// <summary>
    /// The configured depth limit is high enough that templates with merely
    /// moderate nesting (well within real-world Jinja templates) still parse.
    /// </summary>
    [Fact]
    public void NestingBelowLimit_ParsesWithoutError()
    {
        // 50 paren-levels — under the 100 limit. Should succeed.
        var sb = new StringBuilder();
        sb.Append("{{ ");
        const int depth = 50;
        for (int i = 0; i < depth; i++) sb.Append('(');
        sb.Append('1');
        for (int i = 0; i < depth; i++) sb.Append(')');
        sb.Append(" }}");

        var ast = Parse(sb.ToString());
        Assert.NotNull(ast);
        Assert.Single(ast.Nodes);
    }
}
