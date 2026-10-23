using DotLLM.Server;
using DotLLM.Tokenizers.Reasoning;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#767: <c>--reasoning-format none|auto|deepseek</c>.</summary>
public sealed class ReasoningFormatOptionTests
{
    [Fact]
    public void DefaultIsAuto()
        => Assert.Equal(ReasoningFormat.Auto, ServerOptions.Parse(["--model", "m.gguf"]).ReasoningFormat);

    [Theory]
    [InlineData("none", ReasoningFormat.None)]
    [InlineData("auto", ReasoningFormat.Auto)]
    [InlineData("deepseek", ReasoningFormat.Deepseek)]
    public void Flag_IsParsed(string value, ReasoningFormat expected)
        => Assert.Equal(expected, ServerOptions.Parse(["--model", "m.gguf", "--reasoning-format", value]).ReasoningFormat);

    [Fact]
    public void Flag_UnknownValue_IsRejected()
        => Assert.Throws<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", "--reasoning-format", "bogus"]));
}
