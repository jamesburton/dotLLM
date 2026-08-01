using DotLLM.Cli.Benchmarking;
using DotLLM.Core.Attention;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Cli;

/// <summary>
/// CPU-only end-to-end test of the <c>bench</c> command's measurement core
/// (<see cref="BenchRunner"/>) against the real SmolLM-135M Q8_0 fixture:
/// asserts the output shape (rep counts, warm-up discard), sane timing fields,
/// and a well-formed results.csv row. Mirrors exactly how
/// <c>BenchCommand</c> loads and drives a CPU model.
/// </summary>
[Collection("SmallModel")]
public sealed class BenchRunnerTests
{
    private readonly SmallModelFixture _fixture;

    public BenchRunnerTests(SmallModelFixture fixture) => _fixture = fixture;

    [Fact]
    public void Bench_Cpu_SmolLM_Produces_Sane_Result_Shape()
    {
        const int promptLen = 32;
        const int decodeLen = 8;
        const int reps = 2;

        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] seed = tokenizer.Encode("The quick brown fox jumps over the lazy dog. ");
        int[] prompt = BenchStats.TilePrompt(seed, promptLen);
        Assert.Equal(promptLen, prompt.Length);

        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);

        var result = BenchRunner.Run(
            model,
            size => new SimpleKvCache(KvGeometry.FromConfig(config), size),
            prompt,
            decodeTokens: decodeLen,
            reps: reps,
            depth: 0,
            loadMs: 123.4);

        // Shape: warm-up discarded, exactly `reps` measured entries.
        Assert.Equal(reps, result.Reps.Count);
        Assert.Equal(promptLen, result.PromptTokens);
        Assert.Equal(decodeLen, result.DecodeTokens);
        Assert.Equal(0, result.Depth);
        Assert.Equal(promptLen, result.DecodeCtxDepth);
        Assert.Equal(123.4, result.LoadMs, 3);

        // Warm-up rep exists and was actually measured.
        Assert.True(result.Warmup.PrefillMs > 0);
        Assert.True(result.Warmup.DecodeMs > 0);

        // Every measured rep has positive timings and throughput.
        foreach (var rep in result.Reps)
        {
            Assert.True(rep.PrefillMs > 0);
            Assert.True(rep.DecodeMs > 0);
            Assert.Equal(promptLen, rep.PromptTokens);
            Assert.Equal(decodeLen, rep.DecodeTokens);
            Assert.True(rep.PrefillTokS > 0);
            Assert.True(rep.DecodeTokS > 0);
        }

        // Summary stats are consistent with the reps.
        Assert.True(result.PrefillMsMin <= result.PrefillMsMedian);
        Assert.True(result.DecodeMsMin <= result.DecodeMsMedian);
        Assert.True(result.DecodeTokSBest >= result.DecodeTokSMedian);
        Assert.True(result.PrefillTokSBest >= result.PrefillTokSMedian);
    }

    [Fact]
    public void Bench_Cpu_SmolLM_Depth_Extends_Decode_Context()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] prompt = BenchStats.TilePrompt(
            tokenizer.Encode("The quick brown fox jumps over the lazy dog. "), 16);

        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);

        var result = BenchRunner.Run(
            model,
            size => new SimpleKvCache(KvGeometry.FromConfig(config), size),
            prompt, decodeTokens: 4, reps: 1, depth: 16);

        Assert.Equal(16, result.Depth);
        Assert.Equal(32, result.DecodeCtxDepth);
        Assert.Single(result.Reps);
        Assert.True(result.Reps[0].DecodeMs > 0);
    }

    [Fact]
    public void Bench_Cpu_SmolLM_Csv_Row_Matches_Perf_Matrix_Header()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] prompt = BenchStats.TilePrompt(
            tokenizer.Encode("The quick brown fox jumps over the lazy dog. "), 16);

        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var result = BenchRunner.Run(
            model,
            size => new SimpleKvCache(KvGeometry.FromConfig(config), size),
            prompt, decodeTokens: 4, reps: 1);

        string quant = BenchEnvironment.InferQuantLabel(_fixture.FilePath);
        string modelName = BenchEnvironment.InferModelName(_fixture.FilePath, quant);
        Assert.Equal("Q8_0", quant);
        Assert.Equal("SmolLM-135M", modelName);

        string row = BenchCsv.FormatRow(
            DateOnly.FromDateTime(DateTime.Now), "test-host", "cpu-1t", "cpu", "test-version",
            modelName, quant, result.PrefillTokSMedian, result.DecodeTokSMedian,
            result.DecodeCtxDepth, "bench test", "notes");

        // Column count matches the results.csv header (respecting quoted fields).
        int columns = 1;
        bool inQuotes = false;
        foreach (char c in row)
        {
            if (c == '"') inQuotes = !inQuotes;
            else if (c == ',' && !inQuotes) columns++;
        }
        Assert.Equal(BenchCsv.Header.Split(',').Length, columns);
        Assert.Contains(",cpu,dotLLM,test-version,SmolLM-135M,Q8_0,", row);
    }
}
