using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1025: prefill observes cancellation. With no chunk size configured the engine prefills in the model's
/// <see cref="IModel.PreferredPrefillChunkTokens"/>, which is where the model would split internally anyway, and it
/// checks the token before every chunk. A disconnected client therefore stops a long prefill after at most one chunk.
/// Before this, the engine made one call per prompt and ran it to the end (10-60 min on a VRAM-thrashing prompt).
/// </summary>
public sealed class TextGeneratorPrefillChunkCancellationTests
{
    private const int Vocab = 32;
    private const int Prompt = 10;

    private static string PromptText => string.Join(",", Enumerable.Range(2, Prompt));
    private static InferenceOptions Opts => new() { Temperature = 0f, MaxTokens = 3 };

    [Fact]
    public void ModelPreferredChunk_SplitsThePrefill_WhenNoChunkSizeIsConfigured()
    {
        var model = new RecordingModel(preferredChunk: 4);
        var gen = new TextGenerator(model, new CsvTokenizer());

        gen.Generate(PromptText, Opts);

        Assert.Equal([4, 4, 2], model.Calls.Take(3));
        Assert.All(model.Calls.Skip(3), n => Assert.Equal(1, n));   // decode
    }

    [Fact]
    public void ConfiguredChunkSize_WinsOverTheModelPreference()
    {
        var model = new RecordingModel(preferredChunk: 4);
        var gen = new TextGenerator(model, new CsvTokenizer(), prefillChunkSize: 3);

        gen.Generate(PromptText, Opts);

        Assert.Equal([3, 3, 3, 1], model.Calls.Take(4));
    }

    [Fact]
    public void NoPreference_KeepsTheSinglePassPrefill()
    {
        var model = new RecordingModel(preferredChunk: 0);
        var gen = new TextGenerator(model, new CsvTokenizer());

        gen.Generate(PromptText, Opts);

        Assert.Equal(Prompt, model.Calls[0]);
    }

    [Fact]
    public void Generate_CancelledDuringPrefill_StopsBeforeTheNextChunk()
    {
        using var cts = new CancellationTokenSource();
        var model = new RecordingModel(preferredChunk: 4) { AfterForward = cts.Cancel };
        var gen = new TextGenerator(model, new CsvTokenizer());

        Assert.ThrowsAny<OperationCanceledException>(() =>
            gen.Generate(PromptText, Opts, cancellationToken: cts.Token));

        Assert.Equal([4], model.Calls);
    }

    [Fact]
    public async Task Streaming_CancelledDuringPrefill_StopsBeforeTheNextChunk()
    {
        using var cts = new CancellationTokenSource();
        var model = new RecordingModel(preferredChunk: 4) { AfterForward = cts.Cancel };
        var gen = new TextGenerator(model, new CsvTokenizer());

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in gen.GenerateStreamingTokensAsync(PromptText, Opts, cts.Token)) { }
        });

        Assert.Equal([4], model.Calls);
    }

    [Fact]
    public void Generate_CancelledDuringDecode_StopsTheDecodeLoop()
    {
        using var cts = new CancellationTokenSource();
        var model = new RecordingModel(preferredChunk: 0);
        model.AfterForward = () => { if (model.Calls.Count == 2) cts.Cancel(); };   // prefill, then one decode step
        var gen = new TextGenerator(model, new CsvTokenizer());

        Assert.ThrowsAny<OperationCanceledException>(() =>
            gen.Generate(PromptText, new InferenceOptions { Temperature = 0f, MaxTokens = 50 }, cancellationToken: cts.Token));

        Assert.Equal(2, model.Calls.Count);
    }

    // ── fakes ──

    private sealed class CsvTokenizer : ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 1;
        public int EosTokenId => Vocab - 1;
        public int[] Encode(string text)
            => text.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(x => int.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => Encode(text).Length;
    }

    /// <summary>Records the token count of every forward; logits are one-hot on token 5 (never EOS).</summary>
    private sealed class RecordingModel(int preferredChunk) : IModel
    {
        public List<int> Calls { get; } = [];
        public Action? AfterForward { get; set; }

        public int PreferredPrefillChunkTokens => preferredChunk;

        public ModelConfig Config => new()
        {
            VocabSize = Vocab, NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1,
            HiddenSize = 4, IntermediateSize = 16, HeadDim = 4,
            MaxSequenceLength = 256, Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;
        public void Dispose() { }

        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => Core(t, d);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv) => Core(t, d);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, bool last) => Core(t, d);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a) => Core(t, d);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m) => Core(t, d);
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv, ILoraAdapter? a, IMtpState? m, bool last) => Core(t, d);

        private unsafe ITensor Core(ReadOnlySpan<int> tokenIds, int deviceId)
        {
            Calls.Add(tokenIds.Length);
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(Vocab * sizeof(float)), 64);
            var row = new Span<float>((void*)ptr, Vocab);
            row.Fill(-10f);
            row[5] = 10f;
            AfterForward?.Invoke();
            return new UnmanagedTensor(new TensorShape(1, Vocab), DType.Float32, deviceId, ptr);
        }
    }
}
