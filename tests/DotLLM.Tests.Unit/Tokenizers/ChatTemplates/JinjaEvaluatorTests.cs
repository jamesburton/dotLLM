using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

public class JinjaEvaluatorTests
{
    private static string Eval(string template, Dictionary<string, object?>? vars = null)
    {
        var lexer = new JinjaLexer(template);
        var tokens = lexer.Tokenize();
        var parser = new JinjaParser(tokens);
        var ast = parser.Parse();
        var evaluator = new JinjaEvaluator(vars ?? new Dictionary<string, object?>(StringComparer.Ordinal));
        return evaluator.Evaluate(ast);
    }

    // ── Variable lookup ──

    [Fact]
    public void VariableLookup_Defined()
    {
        var result = Eval("{{ name }}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["name"] = "Alice" });
        Assert.Equal("Alice", result);
    }

    [Fact]
    public void VariableLookup_Undefined_EmptyString()
    {
        var result = Eval("{{ unknown }}");
        Assert.Equal("", result);
    }

    [Fact]
    public void VariableLookup_Null_EmptyString()
    {
        var result = Eval("{{ x }}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = null });
        Assert.Equal("", result);
    }

    // ── Literals ──

    [Fact]
    public void StringLiteral()
    {
        Assert.Equal("hello", Eval("{{ 'hello' }}"));
    }

    [Fact]
    public void IntLiteral()
    {
        Assert.Equal("42", Eval("{{ 42 }}"));
    }

    [Fact]
    public void BoolLiteral()
    {
        Assert.Equal("True", Eval("{{ true }}"));
        Assert.Equal("False", Eval("{{ false }}"));
    }

    // ── Slicing with step (#105: Qwen GGUF template uses messages[::-1]) ──

    [Fact]
    public void Slice_String_Reverse()
    {
        Assert.Equal("dcba", Eval("{{ 'abcd'[::-1] }}"));
    }

    [Fact]
    public void Slice_String_PositiveStep()
    {
        Assert.Equal("ace", Eval("{{ 'abcdef'[::2] }}"));
    }

    [Fact]
    public void Slice_String_StartStopStep()
    {
        Assert.Equal("bd", Eval("{{ 'abcdef'[1:5:2] }}"));
    }

    [Fact]
    public void Slice_String_StartStop_NoStep_StillWorks()
    {
        Assert.Equal("bc", Eval("{{ 'abcd'[1:3] }}"));
    }

    [Fact]
    public void Slice_List_Reverse_InForLoop()
    {
        var vars = new Dictionary<string, object?>(StringComparer.Ordinal) { ["xs"] = new List<object?> { 1, 2, 3 } };
        Assert.Equal("321", Eval("{% for x in xs[::-1] %}{{ x }}{% endfor %}", vars));
    }

    [Fact]
    public void Slice_List_NegativeStep_StartStop()
    {
        // Python: [10,20,30,40,50][4:1:-1] == [50,40,30]
        var vars = new Dictionary<string, object?>(StringComparer.Ordinal) { ["xs"] = new List<object?> { 10, 20, 30, 40, 50 } };
        Assert.Equal("50,40,30,", Eval("{% for x in xs[4:1:-1] %}{{ x }},{% endfor %}", vars));
    }

    [Fact]
    public void Slice_QwenTemplate_ReversedMessagesLoop()
    {
        // The exact construct from the Qwen3 GGUF embedded template (#105 line 18).
        var vars = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["messages"] = new List<object?>
            {
                new Dictionary<string, object?>(StringComparer.Ordinal) { ["role"] = "system", ["content"] = "s" },
                new Dictionary<string, object?>(StringComparer.Ordinal) { ["role"] = "user", ["content"] = "u" },
            }
        };
        Assert.Equal("us", Eval("{%- for m in messages[::-1] %}{{- m.content }}{%- endfor %}", vars));
    }

    // ── String concatenation ──

    [Fact]
    public void StringConcatenation_Plus()
    {
        Assert.Equal("ab", Eval("{{ 'a' + 'b' }}"));
    }

    [Fact]
    public void StringConcatenation_Tilde()
    {
        Assert.Equal("ab", Eval("{{ 'a' ~ 'b' }}"));
    }

    // ── Comparisons ──

    [Fact]
    public void Comparison_Eq()
    {
        Assert.Equal("True", Eval("{{ 1 == 1 }}"));
        Assert.Equal("False", Eval("{{ 1 == 2 }}"));
    }

    [Fact]
    public void Comparison_Ne()
    {
        Assert.Equal("True", Eval("{{ 1 != 2 }}"));
    }

    [Fact]
    public void Comparison_StringEq()
    {
        Assert.Equal("True", Eval("{{ x == 'user' }}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = "user" }));
    }

    // ── Logical operators ──

    [Fact]
    public void LogicalAnd()
    {
        Assert.Equal("True", Eval("{{ true and true }}"));
        Assert.Equal("False", Eval("{{ true and false }}"));
    }

    [Fact]
    public void LogicalOr()
    {
        Assert.Equal("True", Eval("{{ false or true }}"));
        Assert.Equal("False", Eval("{{ false or false }}"));
    }

    [Fact]
    public void LogicalNot()
    {
        Assert.Equal("True", Eval("{{ not false }}"));
        Assert.Equal("False", Eval("{{ not true }}"));
    }

    // ── For loops ──

    [Fact]
    public void ForLoop_BasicIteration()
    {
        var result = Eval(
            "{% for x in items %}{{ x }}{% endfor %}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a", "b", "c" } });
        Assert.Equal("abc", result);
    }

    [Fact]
    public void ForLoop_LoopIndex()
    {
        var result = Eval(
            "{% for x in items %}{{ loop.index }}{% endfor %}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a", "b", "c" } });
        Assert.Equal("123", result);
    }

    [Fact]
    public void ForLoop_LoopIndex0()
    {
        var result = Eval(
            "{% for x in items %}{{ loop.index0 }}{% endfor %}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a", "b" } });
        Assert.Equal("01", result);
    }

    [Fact]
    public void ForLoop_LoopFirst()
    {
        var result = Eval(
            "{% for x in items %}{% if loop.first %}F{% endif %}{{ x }}{% endfor %}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a", "b" } });
        Assert.Equal("Fab", result);
    }

    [Fact]
    public void ForLoop_LoopLast()
    {
        var result = Eval(
            "{% for x in items %}{{ x }}{% if loop.last %}L{% endif %}{% endfor %}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a", "b" } });
        Assert.Equal("abL", result);
    }

    [Fact]
    public void ForLoop_LoopLength()
    {
        var result = Eval(
            "{% for x in items %}{{ loop.length }}{% endfor %}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a", "b", "c" } });
        Assert.Equal("333", result);
    }

    [Fact]
    public void ForLoop_EmptyList()
    {
        var result = Eval(
            "{% for x in items %}{{ x }}{% endfor %}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?>() });
        Assert.Equal("", result);
    }

    // ── Scoping ──

    [Fact]
    public void ForLoop_VariableDoesNotLeak()
    {
        // After for loop, 'x' should not be accessible
        var result = Eval(
            "{% for x in items %}{% endfor %}{{ x }}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a" } });
        Assert.Equal("", result);
    }

    [Fact]
    public void SetInLoop_DoesNotLeakToOuterScope()
    {
        var result = Eval(
            "{% set y = 'before' %}{% for x in items %}{% set y = x %}{% endfor %}{{ y }}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a", "b" } });
        Assert.Equal("before", result);
    }

    // ── Namespace (mutable loop state) ──

    [Fact]
    public void Namespace_MutationInLoop()
    {
        var result = Eval(
            "{% set ns = namespace(found=false) %}{% for x in items %}{% if x == 'target' %}{% set ns.found = true %}{% endif %}{% endfor %}{{ ns.found }}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a", "target", "c" } });
        Assert.Equal("True", result);
    }

    [Fact]
    public void Namespace_DefaultValue()
    {
        var result = Eval(
            "{% set ns = namespace(count=0) %}{{ ns.count }}");
        Assert.Equal("0", result);
    }

    // ── Filters ──

    [Fact]
    public void Filter_Trim()
    {
        Assert.Equal("hello", Eval("{{ '  hello  ' | trim }}"));
    }

    [Fact]
    public void Filter_Tojson_String()
    {
        Assert.Equal("\"hello\"", Eval("{{ 'hello' | tojson }}"));
    }

    [Fact]
    public void Filter_Tojson_Dict()
    {
        var result = Eval(
            "{{ data | tojson }}",
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["data"] = new Dictionary<string, object?>(StringComparer.Ordinal) { ["key"] = "value" }
            });
        Assert.Equal("{\"key\": \"value\"}", result);
    }

    [Fact]
    public void Filter_Length_String()
    {
        Assert.Equal("5", Eval("{{ 'hello' | length }}"));
    }

    [Fact]
    public void Filter_Length_List()
    {
        var result = Eval(
            "{{ items | length }}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { 1, 2, 3 } });
        Assert.Equal("3", result);
    }

    [Fact]
    public void Filter_Default_UndefinedVariable()
    {
        Assert.Equal("fallback", Eval("{{ x | default('fallback') }}"));
    }

    [Fact]
    public void Filter_Default_DefinedVariable()
    {
        Assert.Equal("value", Eval("{{ x | default('fallback') }}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = "value" }));
    }

    // ── Is defined / is not defined ──

    [Fact]
    public void IsDefined_True()
    {
        Assert.Equal("True", Eval("{{ x is defined }}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = "val" }));
    }

    [Fact]
    public void IsDefined_False()
    {
        Assert.Equal("False", Eval("{{ x is defined }}"));
    }

    [Fact]
    public void IsNotDefined_True()
    {
        Assert.Equal("True", Eval("{{ x is not defined }}"));
    }

    [Fact]
    public void IsNotDefined_False()
    {
        Assert.Equal("False", Eval("{{ x is not defined }}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = "val" }));
    }

    // ── raise_exception ──

    [Fact]
    public void RaiseException_Throws()
    {
        var ex = Assert.Throws<JinjaException>(() => Eval("{{ raise_exception('boom') }}"));
        Assert.Contains("boom", ex.Message, StringComparison.Ordinal);
    }

    // ── Truthiness (Python semantics) ──

    [Fact]
    public void Truthiness_NullIsFalsy()
    {
        Assert.Equal("no", Eval("{% if x %}yes{% else %}no{% endif %}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = null }));
    }

    [Fact]
    public void Truthiness_FalseIsFalsy()
    {
        Assert.Equal("no", Eval("{% if x %}yes{% else %}no{% endif %}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = false }));
    }

    [Fact]
    public void Truthiness_ZeroIsFalsy()
    {
        Assert.Equal("no", Eval("{% if x %}yes{% else %}no{% endif %}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = 0 }));
    }

    [Fact]
    public void Truthiness_EmptyStringIsFalsy()
    {
        Assert.Equal("no", Eval("{% if x %}yes{% else %}no{% endif %}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = "" }));
    }

    [Fact]
    public void Truthiness_EmptyListIsFalsy()
    {
        Assert.Equal("no", Eval("{% if x %}yes{% else %}no{% endif %}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = new List<object?>() }));
    }

    [Fact]
    public void Truthiness_NonEmptyStringIsTruthy()
    {
        Assert.Equal("yes", Eval("{% if x %}yes{% else %}no{% endif %}", new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = "hi" }));
    }

    // ── Conditional (ternary) ──

    [Fact]
    public void Ternary_TrueCase()
    {
        Assert.Equal("yes", Eval("{{ 'yes' if true else 'no' }}"));
    }

    [Fact]
    public void Ternary_FalseCase()
    {
        Assert.Equal("no", Eval("{{ 'yes' if false else 'no' }}"));
    }

    // ── Member access ──

    [Fact]
    public void MemberAccess_DictProperty()
    {
        var result = Eval(
            "{{ msg.role }}",
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["msg"] = new Dictionary<string, object?>(StringComparer.Ordinal) { ["role"] = "user" }
            });
        Assert.Equal("user", result);
    }

    [Fact]
    public void MemberAccess_BracketNotation()
    {
        var result = Eval(
            "{{ msg['content'] }}",
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["msg"] = new Dictionary<string, object?>(StringComparer.Ordinal) { ["content"] = "hello" }
            });
        Assert.Equal("hello", result);
    }

    // ── In operator ──

    [Fact]
    public void InOperator_ListContains()
    {
        var result = Eval(
            "{{ 'a' in items }}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["items"] = new List<object?> { "a", "b", "c" } });
        Assert.Equal("True", result);
    }

    [Fact]
    public void InOperator_StringContains()
    {
        Assert.Equal("True", Eval("{{ 'world' in 'hello world' }}"));
    }

    // ── Strip method ──

    [Fact]
    public void StripMethod()
    {
        Assert.Equal("hello", Eval("{{ '  hello  '.strip() }}"));
    }

    // ── Arithmetic ──

    [Fact]
    public void Arithmetic_Addition()
    {
        Assert.Equal("3", Eval("{{ 1 + 2 }}"));
    }

    [Fact]
    public void Arithmetic_Subtraction()
    {
        Assert.Equal("1", Eval("{{ 3 - 2 }}"));
    }

    [Fact]
    public void Arithmetic_Modulo()
    {
        Assert.Equal("1", Eval("{{ 5 % 2 }}"));
    }

    // ── If/elif/else ──

    [Fact]
    public void IfElifElse_ElifBranch()
    {
        var result = Eval(
            "{% if x == 1 %}one{% elif x == 2 %}two{% else %}other{% endif %}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = 2 });
        Assert.Equal("two", result);
    }

    [Fact]
    public void IfElifElse_ElseBranch()
    {
        var result = Eval(
            "{% if x == 1 %}one{% elif x == 2 %}two{% else %}other{% endif %}",
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = 3 });
        Assert.Equal("other", result);
    }

    // ── Set ──

    [Fact]
    public void Set_AssignVariable()
    {
        Assert.Equal("42", Eval("{% set x = 42 %}{{ x }}"));
    }

    // ── List literal ──

    [Fact]
    public void ListLiteral_Creation()
    {
        Assert.Equal("3", Eval("{{ [1, 2, 3] | length }}"));
    }

    // ── Nested loops ──

    [Fact]
    public void NestedLoop()
    {
        var result = Eval(
            "{% for i in outer %}{% for j in inner %}{{ i }}{{ j }}{% endfor %}{% endfor %}",
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["outer"] = new List<object?> { "a", "b" },
                ["inner"] = new List<object?> { "1", "2" }
            });
        Assert.Equal("a1a2b1b2", result);
    }

    // ── WhitespaceControl in evaluator context ──

    [Fact]
    public void WhitespaceControl_StripBothSides()
    {
        var result = Eval("  {%- if true -%}  yes  {%- endif -%}  ");
        Assert.Equal("yes", result);
    }

    // ── Comments ──

    [Fact]
    public void Comment_NoOutput()
    {
        var result = Eval("before{# ignored #}after");
        Assert.Equal("beforeafter", result);
    }
}
