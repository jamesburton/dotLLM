using DotLLM.Cli.Commands;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>Issue #1720: the Quant column of <c>dotllm list</c> is parsed from the GGUF file name.</summary>
public sealed class ModelListQuantTests
{
    [Theory]
    [InlineData("SmolLM-135M.Q8_0.gguf", "Q8_0")]
    [InlineData("Llama-3.2-3B-Instruct-Q4_K_M.gguf", "Q4_K_M")]
    [InlineData("TwIL-8B-Q4_K_M-balanced.gguf", "Q4_K_M")]
    [InlineData("Ternary-Bonsai-2-2B-PQ2_0.gguf", "PQ2_0")]
    [InlineData("model-IQ4_XS-00001-of-00003.gguf", "IQ4_XS")]
    [InlineData("embed-F16.gguf", "F16")]
    [InlineData("qwen3-bf16.gguf", "BF16")]
    [InlineData("mystery.gguf", "-")]
    public void InferQuant_ParsesTheFileName(string filename, string expected) =>
        Assert.Equal(expected, ModelListCommand.InferQuant(filename));
}
