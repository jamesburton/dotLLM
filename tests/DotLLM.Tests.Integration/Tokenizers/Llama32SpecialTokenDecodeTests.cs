using System;
using System.IO;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Integration.Tokenizers;

/// <summary>
/// #459, criterion 4. The scheduler fix makes a stop string matchable by handing stop conditions
/// the decoded tail. That closes the <c>&lt;|eom_id|&gt;</c> symptom <b>only if</b> the real
/// tokenizer renders the special token as its literal text on decode — if it skipped specials, the
/// tail would never contain <c>&lt;|eom_id|&gt;</c> and that symptom would be a <i>second</i> root
/// cause needing a token-id stop instead.
/// </summary>
/// <remarks>
/// Deliberately runs against the real Llama-3.2 GGUF vocabulary rather than a mock: a fake
/// tokenizer answers whatever it was built to answer, which is exactly the question being asked
/// here. Loads GGUF <i>metadata</i> only — no weights are touched and no forward runs, so this is
/// a metadata read, not a model run.
/// </remarks>
public sealed class Llama32SpecialTokenDecodeTests
{
    [SkippableFact]
    public void EomAndEotDecodeToTheirLiteralText_SoAStopStringCanMatchThem()
    {
        string? path = ResolveModelPath();
        Skip.If(path is null,
            "Llama-3.2-1B-Instruct-Q8_0.gguf not found. Set DOTLLM_LLAMA32_1B_GGUF or place it in "
            + "~/.dotllm/models/ or ~/.dotllm/test-cache/bartowski/Llama-3.2-1B-Instruct-GGUF/.");

        using var gguf = GgufFile.Open(path!);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        foreach (string special in new[] { "<|eom_id|>", "<|eot_id|>" })
        {
            int[] ids = tokenizer.Encode(special);
            Assert.True(ids.Length > 0, $"'{special}' did not encode to any token.");

            // The decoded text must contain the sequence verbatim — that is what makes
            // StopStringCondition's EndsWith over the decoded tail able to fire on it.
            string decoded = tokenizer.Decode(ids, stripBosSpace: false);
            Assert.Contains(special, decoded, StringComparison.Ordinal);
        }
    }

    private static string? ResolveModelPath()
    {
        string? envPath = Environment.GetEnvironmentVariable("DOTLLM_LLAMA32_1B_GGUF");
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "bartowski", "Llama-3.2-1B-Instruct-GGUF",
                "Llama-3.2-1B-Instruct-Q8_0.gguf"),
            Path.Combine(home, ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF",
                "Llama-3.2-1B-Instruct-Q8_0.gguf"),
        ];
        foreach (string candidate in candidates)
            if (File.Exists(candidate))
                return candidate;

        return null;
    }
}
