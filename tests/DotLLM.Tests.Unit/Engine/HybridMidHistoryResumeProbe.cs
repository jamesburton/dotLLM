using System.Diagnostics;
using System.Globalization;
using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1286 on a real hybrid model (Bonsai-2 2B, qwen35 GDN hybrid) on CPU: an agent-style trace sends a long prompt, then
/// the same prompt with an earlier tool result trimmed. With step checkpoints the second request resumes from the nearest
/// checkpoint before the trimmed result; without, it re-prefills everything. Prints TTFT (prefill) cold vs resumed and the
/// bytes per checkpoint, and checks the greedy output equals a cold prefill. Opt-in: <c>DOTLLM_HYBRID_RESUME_PROBE=1</c>.
/// </summary>
public sealed class HybridMidHistoryResumeProbe
{
    private const int Step = 512;

    private readonly ITestOutputHelper _out;

    public HybridMidHistoryResumeProbe(ITestOutputHelper output) => _out = output;

    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile(
            ["DOTLLM_BONSAI2_2B_GGUF"], "prism-ml", "Ternary-Bonsai-2-2B-gguf", ["Ternary-Bonsai-2-2B-PQ2_0.gguf"],
            [Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "models", "prism-ml", "Ternary-Bonsai-2-2B-gguf")]).Path;

    /// <summary>A deterministic tool result of about <paramref name="lines"/> lines of file listing.</summary>
    private static string ToolResult(string name, int lines)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < lines; i++)
            sb.Append(CultureInfo.InvariantCulture, $"{name}.cs:{i + 1}: var value{i} = Compute({i * 7 % 13}, \"{name}\");\n");
        return sb.ToString();
    }

    private static string Turn(string role, string content) => $"<|im_start|>{role}\n{content}<|im_end|>\n";

    /// <summary>System prompt, a task, three tool calls with results; <paramref name="trimSecond"/> replaces the second result.</summary>
    private static string Trace(bool trimSecond) =>
        Turn("system", "You are a coding agent. Use the tools to inspect the repository, then answer briefly.") +
        Turn("user", "Find where Compute is called with 5 and explain what it returns.") +
        Turn("assistant", "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"Alpha.cs\"}}</tool_call>") +
        Turn("tool", ToolResult("Alpha", 60)) +
        Turn("assistant", "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"Beta.cs\"}}</tool_call>") +
        Turn("tool", trimSecond ? "[result trimmed]" : ToolResult("Beta", 60)) +
        Turn("assistant", "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"Gamma.cs\"}}</tool_call>") +
        Turn("tool", ToolResult("Gamma", 60)) +
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";

    [SkippableFact]
    public void Bonsai2_TrimmedToolResult_ResumesFromTheNearestStep()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_HYBRID_RESUME_PROBE"), "1", StringComparison.Ordinal), "opt-in probe");
        string? path = FindGguf();
        Skip.If(path is null, "Bonsai-2 2B GGUF not found (set DOTLLM_BONSAI2_2B_GGUF).");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.Auto);

        var opts = new InferenceOptions { Temperature = 0, MaxTokens = 16 };
        string full = Trace(trimSecond: false), trimmed = Trace(trimSecond: true);
        int[] fullIds = tokenizer.Encode(full), trimmedIds = tokenizer.Encode(trimmed);
        int shared = fullIds.AsSpan().CommonPrefixLength(trimmedIds);
        _out.WriteLine($"full={fullIds.Length} tok, trimmed={trimmedIds.Length} tok, shared prefix={shared} tok, step={Step}");

        var cold = new TextGenerator(model, tokenizer, mtpEnabled: false);
        var stepped = new TextGenerator(model, tokenizer, mtpEnabled: false, recurrentPrefixCache: true,
            recurrentPrefixCacheOptions: new() { StepTokens = Step });
        var single = new TextGenerator(model, tokenizer, mtpEnabled: false, recurrentPrefixCache: true);

        // Warm the JIT on a short prompt so the first timed request is not paying for it.
        cold.Generate(Turn("user", "hi"), opts with { MaxTokens = 2 });

        var r1Cold = Timed(cold, full, opts, "turn 1, no cache          ");
        Timed(single, full, opts, "turn 1, single slot       ");
        var r1Step = Timed(stepped, full, opts, "turn 1, steps             ");
        Assert.Equal(r1Cold.GeneratedTokenIds, r1Step.GeneratedTokenIds);
        _out.WriteLine($"checkpoint: {stepped.RecurrentStepCheckpointBytes / 1048576.0:F2} MiB each, " +
                       $"{stepped.RecurrentPrefixStatistics.StepCheckpoints} steps held = " +
                       $"{stepped.RecurrentPrefixStatistics.StepCheckpointBytes / 1048576.0:F1} MiB");

        var r2Cold = Timed(cold, trimmed, opts, "turn 2 (trimmed), cold    ");
        var r2Single = Timed(single, trimmed, opts, "turn 2 (trimmed), 1 slot  ");
        var r2Step = Timed(stepped, trimmed, opts, "turn 2 (trimmed), resumed ");

        Assert.Equal(r2Cold.GeneratedTokenIds, r2Step.GeneratedTokenIds);
        Assert.Equal(r2Cold.GeneratedTokenIds, r2Single.GeneratedTokenIds);
        Assert.Equal(shared / Step * Step, r2Step.Timings.CachedTokenCount);
    }

    private InferenceResponse Timed(TextGenerator generator, string prompt, InferenceOptions opts, string label)
    {
        var sw = Stopwatch.StartNew();
        var r = generator.Generate(prompt, opts);
        sw.Stop();
        var t = r.Timings;
        _out.WriteLine($"{label}: TTFT(prefill)={t.PrefillTimeMs,8:F0} ms  cached={t.CachedTokenCount,5}  " +
                       $"prefilled={t.PrefillTokenCount,5}  wall={sw.Elapsed.TotalMilliseconds,8:F0} ms");
        return r;
    }
}
