using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Hf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Tokenizers;

/// <summary>
/// End-to-end smoke tests that glue the HuggingFace <c>tokenizer.json</c>
/// adapter onto the real <c>ib-ssm/mamba3-370M-10BT</c> safetensors weights:
/// prompt text → token IDs → Mamba-3 forward → argmax → decoded string.
/// Verifies the full inference loop works on a real checkpoint without
/// asserting semantic correctness (no fine-tuned expectations on a 370M
/// base model).
/// </summary>
/// <remarks>
/// Gating mirrors <see cref="Models.Loaders.IbSsmMamba3RealWeightsLoadTests"/>:
/// the <c>DOTLLM_IBSSM_CHECKPOINT_PATH</c> env var, the conventional
/// <c>C:/temp/dotllm-ibssm/</c> path, or the user-profile fallback. Tests
/// skip (do not fail) when none resolve.
/// </remarks>
public sealed class IbSsmMamba3TokenizerEndToEndTests
{
    private const string CheckpointPathEnvVar = "DOTLLM_IBSSM_CHECKPOINT_PATH";
    private const string SafetensorsName = "model.safetensors";
    private const string TokenizerName = "tokenizer.json";
    private const string ConventionalDir = "C:/temp/dotllm-ibssm";
    private const string UserProfileFallbackDir = "dotllm-ibssm-370m";

    private readonly ITestOutputHelper _output;

    public IbSsmMamba3TokenizerEndToEndTests(ITestOutputHelper output)
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
    public void Tokenizer_EncodesAsciiPrompt()
    {
        string? dir = ResolveCheckpointDir();
        if (dir is null)
        {
            _output.WriteLine(
                $"[SKIP] ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar} "
                + $"to the safetensors file or its directory, or place it at {ConventionalDir}/{SafetensorsName}.");
            return;
        }

        string tokenizerPath = Path.Combine(dir, TokenizerName);
        if (!File.Exists(tokenizerPath))
        {
            _output.WriteLine($"[SKIP] {tokenizerPath} not present next to weights.");
            return;
        }

        _output.WriteLine($"Tokenizer: {tokenizerPath}  ({new FileInfo(tokenizerPath).Length:N0} bytes)");

        var sw = Stopwatch.StartNew();
        ITokenizer tok = HfBpeTokenizerFactory.Create(File.ReadAllText(tokenizerPath))
            ?? throw new InvalidOperationException("Tokenizer load returned null.");
        sw.Stop();
        _output.WriteLine($"Load: {sw.Elapsed.TotalMilliseconds:F1} ms  vocab={tok.VocabSize}  bos={tok.BosTokenId}  eos={tok.EosTokenId}");

        const string prompt = "Hello world";
        int[] ids = tok.Encode(prompt);
        _output.WriteLine($"Encode: \"{prompt}\" -> [{string.Join(", ", ids)}]  ({ids.Length} tokens)");

        Assert.NotEmpty(ids);
        Assert.Equal(32000, tok.VocabSize);
        Assert.Equal(1, tok.BosTokenId);
        Assert.Equal(2, tok.EosTokenId);
        foreach (int id in ids)
            Assert.InRange(id, 0, tok.VocabSize - 1);

        string decoded = tok.Decode(ids);
        _output.WriteLine($"Decode: [{string.Join(", ", ids)}] -> \"{decoded}\"");
        Assert.Equal(prompt, decoded);

        // Non-ASCII: byte fallback path must round-trip too.
        const string nonAscii = "café";
        int[] naIds = tok.Encode(nonAscii);
        string naDecoded = tok.Decode(naIds);
        _output.WriteLine($"Non-ASCII: \"{nonAscii}\" -> [{string.Join(", ", naIds)}] -> \"{naDecoded}\"");
        Assert.Equal(nonAscii, naDecoded);
    }

    [Fact]
    public void MambaForward_OnTokenizedPrompt_ProducesValidArgmaxToken()
    {
        string? dir = ResolveCheckpointDir();
        if (dir is null)
        {
            _output.WriteLine(
                $"[SKIP] ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar} to enable.");
            return;
        }

        string tokenizerPath = Path.Combine(dir, TokenizerName);
        string weightsPath = Path.Combine(dir, SafetensorsName);
        if (!File.Exists(tokenizerPath) || !File.Exists(weightsPath))
        {
            _output.WriteLine($"[SKIP] Required files missing under {dir}.");
            return;
        }

        // 1. Tokenizer.
        var loadTokenizerWatch = Stopwatch.StartNew();
        ITokenizer tok = HfBpeTokenizerFactory.Create(File.ReadAllText(tokenizerPath));
        loadTokenizerWatch.Stop();

        // 2. Model.
        var loadModelWatch = Stopwatch.StartNew();
        var (model, file, config) = ModelLoader.LoadFromSafetensors(weightsPath);
        loadModelWatch.Stop();

        try
        {
            _output.WriteLine(
                $"Load: tokenizer={loadTokenizerWatch.Elapsed.TotalMilliseconds:F1} ms  "
                + $"model={loadModelWatch.Elapsed.TotalMilliseconds:F1} ms  "
                + $"vocab={config.VocabSize}  layers={config.NumLayers}");

            Assert.Equal(config.VocabSize, tok.VocabSize);

            // 3. Encode prompt.
            const string prompt = "The quick brown fox";
            int[] ids = tok.Encode(prompt);
            int[] positions = new int[ids.Length];
            for (int i = 0; i < positions.Length; i++) positions[i] = i;

            _output.WriteLine($"Prompt: \"{prompt}\"");
            _output.WriteLine($"Encoded IDs ({ids.Length}): [{string.Join(", ", ids)}]");
            foreach (int id in ids) Assert.InRange(id, 0, tok.VocabSize - 1);

            // 4. Forward.
            var fwdWatch = Stopwatch.StartNew();
            using ITensor logits = model.Forward(ids, positions, deviceId: -1);
            fwdWatch.Stop();
            _output.WriteLine(
                $"Forward: shape=[{logits.Shape[0]}, {logits.Shape[1]}]  "
                + $"elapsed={fwdWatch.Elapsed.TotalSeconds:F2} s  "
                + $"({fwdWatch.Elapsed.TotalSeconds / ids.Length:F2} s/token)");

            Assert.Equal(2, logits.Shape.Rank);
            Assert.Equal(ids.Length, logits.Shape[0]);
            Assert.Equal(config.VocabSize, logits.Shape[1]);

            // 5. Argmax of last-position logits.
            int argmax = LastTokenArgMax(logits, config.VocabSize);
            Assert.InRange(argmax, 0, tok.VocabSize - 1);

            string argmaxText = tok.DecodeToken(argmax);
            string sequenceDecoded = tok.Decode(new[] { argmax }, stripBosSpace: false);
            _output.WriteLine(
                $"Argmax: id={argmax}  token=\"{argmaxText}\"  decoded(seq)=\"{sequenceDecoded}\"");

            // Smoke: decoding the argmax must not crash. An empty string is
            // tolerable (it could be a byte-fallback continuation byte), but
            // the single-token decode is expected to return *something* —
            // either the raw token text or a single-byte string via ByteFallback.
            Assert.NotNull(argmaxText);
        }
        finally
        {
            model.Dispose();
            file.Dispose();
        }
    }

    [Fact]
    public void LoadTokenizerFromHfDirectory_SucceedsAlongsideWeights()
    {
        string? dir = ResolveCheckpointDir();
        if (dir is null)
        {
            _output.WriteLine($"[SKIP] {CheckpointPathEnvVar} not set / default path missing.");
            return;
        }

        ITokenizer? tok = ModelLoader.LoadTokenizerFromHfDirectory(dir);
        Assert.NotNull(tok);
        Assert.Equal(32000, tok!.VocabSize);

        // Path accepting a file inside the directory must also work.
        string weights = Path.Combine(dir, SafetensorsName);
        ITokenizer? tok2 = ModelLoader.LoadTokenizerFromHfDirectory(weights);
        Assert.NotNull(tok2);
        Assert.Equal(32000, tok2!.VocabSize);
    }

    private static unsafe int LastTokenArgMax(ITensor logits, int vocabSize)
    {
        int seqLen = logits.Shape[0];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocabSize);
        ReadOnlySpan<float> last = span.Slice((seqLen - 1) * vocabSize, vocabSize);
        int best = 0;
        float bestVal = last[0];
        for (int i = 1; i < last.Length; i++)
        {
            if (last[i] > bestVal) { bestVal = last[i]; best = i; }
        }
        return best;
    }
}
