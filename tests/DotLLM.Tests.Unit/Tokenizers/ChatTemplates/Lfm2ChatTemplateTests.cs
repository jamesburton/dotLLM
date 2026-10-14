using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Renders LFM2.5's real chat template (read from the GGUF; skipped when no LFM2.5 GGUF is available) through dotLLM's
/// Jinja subset: <c>{% generation %}</c>, <c>namespace</c>, <c>split</c>, <c>tojson</c>, <c>replace</c>, macros, <c>raise_exception</c>.
/// Resolve the file via <c>DOTLLM_LFM2_GGUF</c> or the HF hub cache (issue #1015).
/// </summary>
public sealed class Lfm2ChatTemplateTests
{
    private readonly ITestOutputHelper _out;
    public Lfm2ChatTemplateTests(ITestOutputHelper output) => _out = output;

    private static string? FindGguf()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_LFM2_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string hub = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub", "models--LiquidAI--LFM2.5-24B-A2B-GGUF", "snapshots");
        if (!Directory.Exists(hub)) return null;
        return Directory.EnumerateFiles(hub, "*.gguf", SearchOption.AllDirectories).FirstOrDefault();
    }

    private static JinjaChatTemplateHolder? Load()
    {
        string? path = FindGguf();
        if (path is null) return null;
        using var gguf = GgufFile.Open(path);
        string? tpl = gguf.Metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        return tpl is null ? null : new JinjaChatTemplateHolder(new DotLLM.Tokenizers.ChatTemplates.JinjaChatTemplate(tpl, "<|startoftext|>", "<|im_end|>"));
    }

    private sealed record JinjaChatTemplateHolder(IChatTemplate Template);

    [SkippableFact]
    public void SimpleChat_EndsWithOpenThinkTag()
    {
        var h = Load();
        Skip.If(h is null, "LFM2.5 GGUF not found: set DOTLLM_LFM2_GGUF or fetch LiquidAI/LFM2.5-24B-A2B-GGUF into the HF hub cache.");
        Assert.NotNull(h);
        string s = h.Template.Apply(new[] { new ChatMessage { Role = "user", Content = "What is C. elegans?" } },
            new ChatTemplateOptions { AddGenerationPrompt = true });
        _out.WriteLine(s);
        Assert.Equal("<|startoftext|><|im_start|>user\nWhat is C. elegans?<|im_end|>\n<|im_start|>assistant\n<think>", s);
    }

    [SkippableFact]
    public void ToolConversation_RendersToolListAndPythonicCall()
    {
        var h = Load();
        Skip.If(h is null, "LFM2.5 GGUF not found: set DOTLLM_LFM2_GGUF or fetch LiquidAI/LFM2.5-24B-A2B-GGUF into the HF hub cache.");
        Assert.NotNull(h);
        var tools = new[]
        {
            new ToolDefinition("get_candidate_status", "Retrieves the current status of a candidate",
                "{\"type\":\"object\",\"properties\":{\"candidate_id\":{\"type\":\"string\"}},\"required\":[\"candidate_id\"]}"),
        };
        var msgs = new[]
        {
            new ChatMessage { Role = "system", Content = "You are helpful." },
            new ChatMessage { Role = "user", Content = "Status of 12345?" },
            new ChatMessage { Role = "assistant", Content = "", ToolCalls = new[] { new ToolCall("call_0", "get_candidate_status", "{\"candidate_id\":\"12345\"}") } },
            new ChatMessage { Role = "tool", Content = "{\"status\": \"Interview Scheduled\"}", ToolCallId = "call_0" },
        };
        string s = h.Template.Apply(msgs, new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools });
        _out.WriteLine(s);
        Assert.StartsWith("<|startoftext|><|im_start|>system\nYou are helpful.\nList of tools: [", s);
        Assert.Contains("get_candidate_status", s);
        Assert.Contains("<|tool_call_start|>[get_candidate_status(candidate_id='12345')]<|tool_call_end|><|im_end|>\n", s);
        Assert.Contains("<|im_start|>tool\n{\"status\": \"Interview Scheduled\"}<|im_end|>\n", s);
        Assert.EndsWith("<|im_start|>assistant\n<think>", s);
    }
}
