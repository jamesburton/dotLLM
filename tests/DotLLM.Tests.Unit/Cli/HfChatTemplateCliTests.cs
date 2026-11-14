using System.Globalization;
using System.Text.Json;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Spectre.Console;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

[Collection(CliConsoleCollection.Name)]
public sealed class HfChatTemplateCliTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"dotllm-hf-cli-template-{Guid.NewGuid():N}");

    [Theory]
    [InlineData("run")]
    [InlineData("chat")]
    public void MalformedTemplate_FailsBeforeOpeningWeights(string command)
    {
        Directory.CreateDirectory(_directory);
        File.WriteAllText(Path.Combine(_directory, "config.json"), "{}");
        File.WriteAllBytes(Path.Combine(_directory, "model.safetensors"), []);
        File.WriteAllText(Path.Combine(_directory, "tokenizer.json"),
            """{"model":{"type":"BPE","vocab":{"B":0},"merges":[]},"pre_tokenizer":{"type":"ByteLevel"}}""");
        File.WriteAllText(Path.Combine(_directory, "tokenizer_config.json"), "{\"chat_template\":17}");
        string[] args = string.Equals(command, "run", StringComparison.Ordinal)
            ? [command, _directory, "--device", "cpu", "--prompt", "B"]
            : [command, _directory, "--device", "cpu"];

        var result = Run(args);
        Assert.Equal(1, result.Code);
        Assert.Contains("Invalid HuggingFace chat template configuration", result.Output, StringComparison.Ordinal);
        Assert.Contains("tokenizer_config.json", result.Output, StringComparison.Ordinal);
        // The invalid weight file must never be opened; template validation owns this failure.
    }

    [Theory]
    [InlineData(false, false, 1)]
    [InlineData(true, false, 5)]
    [InlineData(true, true, 6)]
    [InlineData(false, true, 1)]
    public void Run_PrefillsTheSelectedTemplateAndToolDefinitions(bool hasTemplate, bool useTools, int promptTokens)
    {
        WriteTinyModel();
        if (hasTemplate)
            File.WriteAllText(Path.Combine(_directory, "tokenizer_config.json"), JsonSerializer.Serialize(new
            {
                chat_template = "{{ bos_token }}{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% for message in messages %}A{{ message.content }}{% endfor %}{{ eos_token }}{% if add_generation_prompt %}A{% endif %}",
            }));
        var args = new List<string> { "run", _directory, "--device", "cpu", "--prompt", "B", "--max-tokens", "1", "--threads", "1", "--json" };
        if (useTools)
            args.AddRange(["--tools", """[{"type":"function","function":{"name":"A","description":"A","parameters":{"type":"object"}}}]"""]);
        var result = Run(args.ToArray());
        Assert.Equal(0, result.Code);
        using var json = JsonDocument.Parse(result.Output);
        Assert.Equal(promptTokens, json.RootElement.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(useTools && !hasTemplate, result.Error.Contains("--tools is ignored", StringComparison.Ordinal));
    }

    private void WriteTinyModel()
    {
        Directory.CreateDirectory(_directory);
        File.WriteAllText(Path.Combine(_directory, "config.json"), """
            {"model_type":"llama","architectures":["LlamaForCausalLM"],"vocab_size":4,"hidden_size":2,
             "intermediate_size":3,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
             "max_position_embeddings":64,"rms_norm_eps":0.00001,"rope_theta":10000,"tie_word_embeddings":false}
            """);
        File.WriteAllText(Path.Combine(_directory, "tokenizer.json"), """
            {"model":{"type":"BPE","vocab":{"A":0,"B":1,"<|begin_of_text|>":2,"<|im_end|>":3},"merges":[]},
             "pre_tokenizer":{"type":"ByteLevel","use_regex":false},
             "added_tokens":[{"id":2,"content":"<|begin_of_text|>","special":true},{"id":3,"content":"<|im_end|>","special":true}]}
            """);
        var builder = new SafetensorsFixtureBuilder();
        builder.AddFloat32("model.embed_tokens.weight", [4, 2], new float[8]);
        builder.AddFloat32("lm_head.weight", [4, 2], new float[8]);
        builder.AddFloat32("model.norm.weight", [2], new float[] { 1, 1 });
        const string prefix = "model.layers.0";
        builder.AddFloat32($"{prefix}.input_layernorm.weight", [2], new float[] { 1, 1 });
        builder.AddFloat32($"{prefix}.post_attention_layernorm.weight", [2], new float[] { 1, 1 });
        foreach (string name in new[] { "q_proj", "k_proj", "v_proj", "o_proj" })
            builder.AddFloat32($"{prefix}.self_attn.{name}.weight", [2, 2], new float[4]);
        builder.AddFloat32($"{prefix}.mlp.gate_proj.weight", [3, 2], new float[6]);
        builder.AddFloat32($"{prefix}.mlp.up_proj.weight", [3, 2], new float[6]);
        builder.AddFloat32($"{prefix}.mlp.down_proj.weight", [2, 3], new float[6]);
        builder.WriteTo(Path.Combine(_directory, "model.safetensors"));
    }

    private static (int Code, string Output, string Error) Run(string[] args)
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var error = new StringWriter(CultureInfo.InvariantCulture);
        var previousConsole = AnsiConsole.Console;
        var previousOut = Console.Out;
        var previousError = Console.Error;
        var console = AnsiConsole.Create(new AnsiConsoleSettings
        {
            Out = new AnsiConsoleOutput(output), Ansi = AnsiSupport.No,
            ColorSystem = ColorSystemSupport.NoColors, Interactive = InteractionSupport.No,
        });
        console.Profile.Width = 1_000_000;
        try
        {
            AnsiConsole.Console = console;
            Console.SetOut(output);
            Console.SetError(error);
            var result = CliCommandModel.Run(args);
            return (result.ExitCode, output.ToString() + result.Output, error.ToString());
        }
        finally
        {
            AnsiConsole.Console = previousConsole;
            Console.SetOut(previousOut);
            Console.SetError(previousError);
        }
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }
}
