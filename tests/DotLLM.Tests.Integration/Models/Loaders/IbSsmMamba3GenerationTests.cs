using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end text generation on the real <c>ib-ssm/mamba3-370M-10BT</c>
/// checkpoint: tokenize prompt → iterative argmax decode via growing-context
/// one-shot prefill → decode back to text. Exercises the full pipeline
/// (<see cref="ModelLoader.LoadFromSafetensors"/> +
/// <see cref="ModelLoader.LoadTokenizerFromHfDirectory"/> +
/// <see cref="Mamba3TransformerModel"/> forward) on a real 1.55 GB checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why growing-context prefill (and not decode-state).</b> The
/// <see cref="Mamba3State"/> decode overload has a documented chunk-edge
/// drift (the canonical <c>shifted_γ</c> 1-token lookahead drops to 0 at
/// chunk boundaries), so multi-call decode is qualitatively correct but
/// not bitwise-equivalent to one-shot. For a correctness demonstration
/// today, each iteration runs one-shot prefill over the full growing prompt:
/// O(N²) total CPU time, but every forward is canonical. At 5 tokens on
/// 370M this is ~5-15 s total — acceptable for an integration test.
/// </para>
/// <para>
/// <b>Gating.</b> Same pattern as
/// <see cref="IbSsmMamba3RealWeightsLoadTests"/>:
/// <c>DOTLLM_IBSSM_CHECKPOINT_PATH</c>, the conventional
/// <c>C:/temp/dotllm-ibssm/</c>, or <c>%USERPROFILE%/dotllm-ibssm-370m/</c>.
/// Tests skip (do not fail) when none resolve.
/// </para>
/// </remarks>
public sealed class IbSsmMamba3GenerationTests
{
    private const string CheckpointPathEnvVar = "DOTLLM_IBSSM_CHECKPOINT_PATH";
    private const string SafetensorsName = "model.safetensors";
    private const string ConventionalDir = "C:/temp/dotllm-ibssm";
    private const string UserProfileFallbackDir = "dotllm-ibssm-370m";

    private readonly ITestOutputHelper _output;

    public IbSsmMamba3GenerationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static string? ResolveCheckpointDir()
    {
        string? env = Environment.GetEnvironmentVariable(CheckpointPathEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (File.Exists(env))
            {
                string? dir = Path.GetDirectoryName(env);
                if (dir is not null && Directory.Exists(dir)) return dir;
            }
            if (Directory.Exists(env)) return env;
        }

        if (Directory.Exists(ConventionalDir) &&
            File.Exists(Path.Combine(ConventionalDir, SafetensorsName)))
        {
            return ConventionalDir;
        }

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (!string.IsNullOrWhiteSpace(home))
        {
            string fallback = Path.Combine(home, UserProfileFallbackDir);
            if (Directory.Exists(fallback) &&
                File.Exists(Path.Combine(fallback, SafetensorsName)))
            {
                return fallback;
            }
        }

        return null;
    }

    [Fact]
    public void Mamba3_GeneratesText_FromTokenizedPrompt()
    {
        string? dir = ResolveCheckpointDir();
        if (dir is null)
        {
            _output.WriteLine(
                $"[SKIP] ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar} "
                + $"to the safetensors file or its directory, or place it at {ConventionalDir}/{SafetensorsName}.");
            return;
        }

        string weightsPath = Path.Combine(dir, SafetensorsName);
        _output.WriteLine($"Checkpoint dir: {dir}");

        // Load tokenizer + model.
        ITokenizer? tok = ModelLoader.LoadTokenizerFromHfDirectory(dir);
        Assert.NotNull(tok);

        var loadWatch = Stopwatch.StartNew();
        var (model, file, config) = ModelLoader.LoadFromSafetensors(weightsPath);
        loadWatch.Stop();
        _output.WriteLine(
            $"Loaded: vocab={config.VocabSize}  layers={config.NumLayers}  "
            + $"load={loadWatch.Elapsed.TotalMilliseconds:F0} ms");

        try
        {
            Assert.Equal(config.VocabSize, tok!.VocabSize);

            const string prompt = "The capital of France is";
            const int maxNewTokens = 5;

            int[] promptIds = tok.Encode(prompt);
            Assert.NotEmpty(promptIds);
            foreach (int id in promptIds) Assert.InRange(id, 0, tok.VocabSize - 1);
            _output.WriteLine($"Prompt: \"{prompt}\"");
            _output.WriteLine($"Encoded prompt ({promptIds.Length} tokens): [{string.Join(", ", promptIds)}]");

            int eosId = tok.EosTokenId;
            var tokens = new List<int>(promptIds.Length + maxNewTokens);
            tokens.AddRange(promptIds);

            var generated = new List<int>(maxNewTokens);
            var perTokenMs = new List<double>(maxNewTokens);
            bool hitEos = false;

            var totalWatch = Stopwatch.StartNew();
            for (int step = 0; step < maxNewTokens; step++)
            {
                int[] positions = new int[tokens.Count];
                for (int i = 0; i < positions.Length; i++) positions[i] = i;

                var stepWatch = Stopwatch.StartNew();
                using ITensor logits = model.Forward(
                    tokens.ToArray(), positions, deviceId: -1);
                stepWatch.Stop();
                perTokenMs.Add(stepWatch.Elapsed.TotalMilliseconds);

                Assert.Equal(2, logits.Shape.Rank);
                Assert.Equal(tokens.Count, logits.Shape[0]);
                Assert.Equal(config.VocabSize, logits.Shape[1]);

                int next = LastTokenArgMaxChecked(logits, config.VocabSize);
                Assert.InRange(next, 0, tok.VocabSize - 1);

                generated.Add(next);
                tokens.Add(next);

                if (next == eosId)
                {
                    hitEos = true;
                    _output.WriteLine($"  step {step}: argmax={next} (EOS) — stopping early");
                    break;
                }
            }
            totalWatch.Stop();

            // Timing guard — if this exceeds a generous ceiling the whole test
            // loop has regressed; the calibrated budget is ~5-15 s on 370M CPU.
            Assert.True(totalWatch.Elapsed.TotalSeconds < 60,
                $"Generation took {totalWatch.Elapsed.TotalSeconds:F1} s — far exceeds the 60 s ceiling.");

            string decoded = tok.Decode(tokens.ToArray());
            _output.WriteLine($"Generated IDs ({generated.Count}): [{string.Join(", ", generated)}]");
            for (int i = 0; i < perTokenMs.Count; i++)
                _output.WriteLine($"  step {i}: {perTokenMs[i] / 1000.0:F2} s");
            _output.WriteLine(
                $"Total: {totalWatch.Elapsed.TotalSeconds:F2} s  "
                + $"avg={totalWatch.Elapsed.TotalMilliseconds / Math.Max(1, generated.Count) / 1000.0:F2} s/token");
            _output.WriteLine($"Decoded: \"{decoded}\"");

            // Round-trip prefix check. The HF tokenizer.json decoder strips the
            // SentencePiece leading space on the first token, so we compare
            // against the tokenizer's own round-trip of the prompt — not the
            // raw prompt string.
            string decodedPromptOnly = tok.Decode(promptIds);
            Assert.StartsWith(decodedPromptOnly, decoded);

            // If the *very first* argmax was EOS it's a valid but curious
            // base-model behaviour — surface it, don't fail. If all 5 were EOS
            // that would be degenerate; the early-break above ensures at most
            // one EOS is appended, so this branch is already a single-EOS curiosity.
            if (hitEos && generated.Count == 1)
            {
                _output.WriteLine(
                    "[INFO] ib-ssm immediately argmaxed to EOS. Valid base-model "
                    + "behaviour on an undertrained (10BT) checkpoint; not a pipeline failure.");
            }
            else
            {
                Assert.Contains(generated, id => id != eosId);
            }
        }
        finally
        {
            model.Dispose();
            file.Dispose();
        }
    }

    /// <summary>
    /// Per-position argmax with a finiteness sweep in the same pass — a NaN
    /// or Inf in the last-token logits is surfaced as a clear assertion
    /// failure rather than silently producing a wrong argmax.
    /// </summary>
    private static unsafe int LastTokenArgMaxChecked(ITensor logits, int vocabSize)
    {
        int seqLen = logits.Shape[0];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocabSize);
        ReadOnlySpan<float> last = span.Slice((seqLen - 1) * vocabSize, vocabSize);
        int best = 0;
        float bestVal = last[0];
        Assert.True(float.IsFinite(bestVal),
            "Last-token logits contain NaN/Inf at vocab index 0 — forward pass broke.");
        for (int i = 1; i < last.Length; i++)
        {
            float v = last[i];
            Assert.True(float.IsFinite(v),
                $"Last-token logit at vocab index {i} is not finite (value={v}).");
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }
}
