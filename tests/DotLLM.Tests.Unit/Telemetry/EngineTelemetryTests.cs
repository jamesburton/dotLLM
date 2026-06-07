using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Telemetry;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Telemetry;

/// <summary>
/// Behavioural tests for the engine telemetry surface — verifies that running
/// <see cref="TextGenerator"/> emits the expected counters / histograms when a
/// <see cref="MeterListener"/> is attached, and the expected spans when an
/// <see cref="ActivityListener"/> is attached.
/// </summary>
public sealed class EngineTelemetryTests
{
    private const int VocabSize = 32;
    private const int NumLayers = 1;
    private const int NumKvHeads = 1;
    private const int HeadDim = 4;

    [Fact]
    public void MeterListener_CapturesPrefillAndDecodeMetrics()
    {
        var capture = new MetricCapture();
        using var listener = capture.Listen();

        RunGreedy(promptLen: 3, maxTokens: 4);

        Assert.True(capture.PrefillTokens > 0, "expected dotllm.engine.tokens.prefill to be recorded");
        Assert.True(capture.DecodeTokens > 0, "expected dotllm.engine.tokens.decode to be recorded");
        Assert.True(capture.PrefillTokensPerSecondCount > 0, "expected prefill tokens/sec histogram sample");
        Assert.True(capture.DecodeTokensPerSecondCount > 0, "expected decode tokens/sec histogram sample");
        Assert.True(capture.TimeToFirstTokenCount > 0, "expected TTFT histogram sample");
        Assert.Equal("Llama", capture.LastModelTag);
    }

    [Fact]
    public void ActivityListener_CapturesRequestAndPrefillSpans()
    {
        var spans = new List<Activity>();
        using var listener = new ActivityListener
        {
            ShouldListenTo = src => src.Name == EngineTelemetry.Name,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStopped = spans.Add,
        };
        ActivitySource.AddActivityListener(listener);

        RunGreedy(promptLen: 3, maxTokens: 3);

        Assert.Contains(spans, s => s.OperationName == EngineActivities.Request);
        Assert.Contains(spans, s => s.OperationName == EngineActivities.Prefill);
        Assert.Contains(spans, s => s.OperationName == EngineActivities.Sample);

        var request = spans.First(s => s.OperationName == EngineActivities.Request);
        Assert.Equal("Llama", request.GetTagItem(TelemetryTags.Model));
        Assert.NotNull(request.GetTagItem(TelemetryTags.PromptTokens));
        Assert.NotNull(request.GetTagItem(TelemetryTags.GeneratedTokens));
        Assert.NotNull(request.GetTagItem(TelemetryTags.FinishReason));

        var prefill = spans.First(s => s.OperationName == EngineActivities.Prefill);
        Assert.NotNull(prefill.GetTagItem(TelemetryTags.PrefillTokenCount));
        Assert.NotNull(prefill.GetTagItem(TelemetryTags.PrefillDurationMs));
    }

    [Fact]
    public void NoListener_NoActivityCreated()
    {
        // No MeterListener / ActivityListener attached — the API surface must stay null.
        Activity? captured = EngineTelemetry.ActivitySource.StartActivity(EngineActivities.Request);
        try
        {
            Assert.Null(captured);
        }
        finally
        {
            captured?.Dispose();
        }
    }

    [Fact]
    public void DecodeStep_SamplesAtOnePercent()
    {
        // With listeners attached, decode_step spans should be emitted at the configured rate.
        var stepSpans = new List<Activity>();
        using var listener = new ActivityListener
        {
            ShouldListenTo = src => src.Name == EngineTelemetry.Name,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStopped = a => { if (a.OperationName == EngineActivities.DecodeStep) stepSpans.Add(a); },
        };
        ActivitySource.AddActivityListener(listener);

        // 200 decode steps -> at 10 permille -> ~2 spans (step % 100 == 0 -> steps 100, 200).
        // step starts at 1 in the decode loop so steps 100 and 200 fall inside the sample window.
        RunGreedy(promptLen: 1, maxTokens: 201);
        Assert.InRange(stepSpans.Count, 1, 5);
    }

    private static void RunGreedy(int promptLen, int maxTokens)
    {
        var tokenizer = BuildVocab();
        float[] logits = MakeLogits(VocabSize, argmaxToken: 2);
        using var model = new MockModel(logits);
        var generator = new TextGenerator(model, tokenizer);

        string prompt = string.Concat(Enumerable.Repeat("a", promptLen));
        var options = new InferenceOptions
        {
            Temperature = 0f,
            MaxTokens = maxTokens,
            StopConditions = Array.Empty<DotLLM.Core.Sampling.IStopCondition>(),
        };
        // Override with a max-tokens stop so generation actually terminates without EOS.
        options = options with
        {
            StopConditions = new DotLLM.Core.Sampling.IStopCondition[]
            {
                new MaxTokensStopCondition(maxTokens),
            },
        };
        var response = generator.Generate(prompt, options);
        Assert.Equal(maxTokens, response.GeneratedTokenCount);
    }

    private static BpeTokenizer BuildVocab()
    {
        // Simple SentencePiece-style vocab: index N maps to character 'a' + (N-2)
        var tokens = new string[VocabSize];
        tokens[0] = "<unk>";
        tokens[1] = "<s>";
        for (int i = 2; i < VocabSize; i++)
            tokens[i] = ((char)('a' + (i - 2))).ToString();
        float[] scores = new float[VocabSize];
        return BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes: null,
            bosId: 1, eosId: 1, addBosSpace: false);
    }

    private static float[] MakeLogits(int vocab, int argmaxToken)
    {
        var arr = new float[vocab];
        for (int i = 0; i < vocab; i++)
            arr[i] = i == argmaxToken ? 10f : 0f;
        return arr;
    }

    private sealed class MetricCapture
    {
        public long PrefillTokens;
        public long DecodeTokens;
        public int PrefillTokensPerSecondCount;
        public int DecodeTokensPerSecondCount;
        public int TimeToFirstTokenCount;
        public string? LastModelTag;

        public MeterListener Listen()
        {
            var listener = new MeterListener
            {
                InstrumentPublished = (instrument, l) =>
                {
                    if (instrument.Meter.Name == EngineTelemetry.Name)
                        l.EnableMeasurementEvents(instrument);
                },
            };
            listener.SetMeasurementEventCallback<long>((instrument, value, tags, _) =>
            {
                CaptureModelTag(tags);
                switch (instrument.Name)
                {
                    case "dotllm.engine.tokens.prefill": Interlocked.Add(ref PrefillTokens, value); break;
                    case "dotllm.engine.tokens.decode": Interlocked.Add(ref DecodeTokens, value); break;
                }
            });
            listener.SetMeasurementEventCallback<double>((instrument, value, tags, _) =>
            {
                CaptureModelTag(tags);
                switch (instrument.Name)
                {
                    case "dotllm.engine.tokens_per_second.prefill":
                        Interlocked.Increment(ref PrefillTokensPerSecondCount); break;
                    case "dotllm.engine.tokens_per_second.decode":
                        Interlocked.Increment(ref DecodeTokensPerSecondCount); break;
                    case "dotllm.engine.time_to_first_token_ms":
                        Interlocked.Increment(ref TimeToFirstTokenCount); break;
                }
            });
            listener.Start();
            return listener;
        }

        private void CaptureModelTag(ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            foreach (var tag in tags)
                if (tag.Key == TelemetryTags.Model && tag.Value is string s)
                    LastModelTag = s;
        }
    }

    private sealed class MockModel : IModel
    {
        private readonly float[] _logits;

        public MockModel(float[] logits) { _logits = logits; }

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HeadDim * NumKvHeads,
            IntermediateSize = HeadDim * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = 1024,
            Architecture = Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
        {
            int batchSize = tokenIds.Length;
            long totalFloats = (long)batchSize * VocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);

            float* dst = (float*)ptr;
            for (int b = 0; b < batchSize; b++)
                _logits.AsSpan().CopyTo(new Span<float>(dst + b * VocabSize, VocabSize));

            if (kvCache != null)
            {
                int kvStride = NumKvHeads * HeadDim;
                for (int layer = 0; layer < NumLayers; layer++)
                {
                    nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchSize * kvStride * sizeof(float)), 64);
                    nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchSize * kvStride * sizeof(float)), 64);
                    NativeMemory.Clear((void*)kPtr, (nuint)(batchSize * kvStride * sizeof(float)));
                    NativeMemory.Clear((void*)vPtr, (nuint)(batchSize * kvStride * sizeof(float)));

                    var kRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, kPtr);
                    var vRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, vPtr);
                    kvCache.Update(kRef, vRef, positions, layer);

                    NativeMemory.AlignedFree((void*)kPtr);
                    NativeMemory.AlignedFree((void*)vPtr);
                }
            }

            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, ptr);
        }

        public void Dispose() { }
    }
}
