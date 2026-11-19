using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Hip;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Hip;

[Collection("HipGpu")]
public sealed partial class HipTransformerModelTests(ITestOutputHelper output)
{
    private const int Hidden = 160, Heads = 4, KvHeads = 2, HeadDim = 40, Intermediate = 224, Layers = 2, Vocab = 97;
    private const int LogicalContext = 65536, PageTokens = 4;

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "GPU")]
    public void PrefillDecodeRollbackAndReset_MatchCpuWithTiedOrUntiedHead(bool tied)
    {
        RequireGpu();
        Fixture fixture = CreateFixture(tied);
        using var model = LoadHipAndCloseMapping(fixture);
        using var oracle = new CpuOracle(fixture);
        using var cache = model.CreateKvCache(20, PageTokens);
        int[] tokens = Tokens(17, 3);
        Assert.Equal(1, model.MaxAllRowLogitsLength);
        Assert.Equal(0, cache.AllocatedBytes);
        Assert.Equal(fixture.StoredBytes, model.StoredWeightBytes);
        Assert.True(model.WeightMemoryBytes >= model.StoredWeightBytes);
        long compute = model.ComputeMemoryBytes;

        using (var logits = model.Forward(tokens, Positions(0, tokens.Length), 0, cache, true))
        {
            Assert.Equal(new[] { 1, Vocab }, logits.Shape.Dimensions);
            Assert.Equal(DType.Float32, logits.DType);
            Close(oracle.Last(tokens), ReadLast(logits), $"tied={tied}, prefill 17");
        }
        Assert.Equal(17, cache.CurrentLength);
        AssertCacheBytes(cache, 5, 8);

        int[] history = tokens;
        for (int i = 0; i < 3; i++)
        {
            int token = 47 + i * 7;
            int position = history.Length;
            history = [.. history, token];
            using var logits = model.Forward([token], [position], 0, cache);
            Close(oracle.Last(history), ReadLast(logits), $"tied={tied}, decode {position}");
        }
        Assert.Equal(20, cache.CurrentLength);
        AssertCacheBytes(cache, 5, 8);
        Assert.True(cache.TryGrowTo(128));
        Assert.Equal(128, cache.MaxLength);
        Assert.False(cache.TryGrowTo(LogicalContext + 1));

        // Roll back across a page boundary, then overwrite the retained partial page.
        cache.Rollback(6);
        AssertCacheBytes(cache, 2, 8);
        int[] replacement = Tokens(12, 53);
        history = [.. tokens[..6], .. replacement];
        using (var logits = model.Forward(replacement, Positions(6, replacement.Length), 0, cache, true))
            Close(oracle.Last(history), ReadLast(logits), $"tied={tied}, rollback and overwrite");
        Assert.Equal(18, cache.CurrentLength);
        AssertCacheBytes(cache, 5, 8);
        Assert.Equal(compute, model.ComputeMemoryBytes);
        Assert.Equal(fixture.StoredBytes, model.StoredWeightBytes);

        cache.Rollback(0);
        Assert.Equal(0, cache.AllocatedBytes);
        Assert.Equal(0, cache.AllocatedPages);
        using (var logits = model.Forward([29], [0], 0, cache))
            Close(oracle.Last([29]), ReadLast(logits), $"tied={tied}, reset");
        AssertCacheBytes(cache, 1, 1);
        output.WriteLine($"Fixture: {fixture.Path}; stored weights={model.StoredWeightBytes}; allocated weights={model.WeightMemoryBytes}; compute={compute} bytes.");
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void InterleavedAndBatchedRequests_PreserveIndependentHistoryAndOwnedResults()
    {
        RequireGpu();
        Fixture fixture = CreateFixture(false);
        using var model = LoadHipAndCloseMapping(fixture);
        using var oracle = new CpuOracle(fixture);
        using var a = model.CreateKvCache(LogicalContext, PageTokens);
        using var b = model.CreateKvCache(LogicalContext, PageTokens);
        int[] historyA = Tokens(5, 7), historyB = Tokens(9, 41);
        using var firstA = model.Forward(historyA, Positions(0, historyA.Length), 0, a, true);
        float[] savedA = ReadLast(firstA);
        Close(oracle.Last(historyA), savedA, "A prefill");
        using (var firstB = model.Forward(historyB, Positions(0, historyB.Length), 0, b, true))
            Close(oracle.Last(historyB), ReadLast(firstB), "B prefill");
        Assert.Equal(savedA, ReadLast(firstA));

        for (int step = 0; step < 4; step++)
        {
            int tokenA = 61 + step, tokenB = 71 + step;
            int positionA = historyA.Length, positionB = historyB.Length;
            historyA = [.. historyA, tokenA]; historyB = [.. historyB, tokenB];
            using var logitsB = model.Forward([tokenB], [positionB], 0, b);
            using var logitsA = model.Forward([tokenA], [positionA], 0, a);
            Close(oracle.Last(historyA), ReadLast(logitsA), $"interleaved A {step}");
            Close(oracle.Last(historyB), ReadLast(logitsB), $"interleaved B {step}");
        }

        int[] nextA = [82, 13, 37], nextB = [28];
        var requests = new[]
        {
            new SequenceForwardRequest { TokenIds = nextA, Positions = Positions(historyA.Length, nextA.Length), KvCache = a, LastTokenLogitsOnly = true },
            new SequenceForwardRequest { TokenIds = nextB, Positions = Positions(historyB.Length, nextB.Length), KvCache = b, LastTokenLogitsOnly = true },
        };
        IReadOnlyList<ITensor> batch = model.ForwardBatch(requests, 0);
        try
        {
            Assert.Equal(2, batch.Count);
            Assert.NotEqual(batch[0].DataPointer, batch[1].DataPointer);
            Close(oracle.Last([.. historyA, .. nextA]), ReadLast(batch[0]), "batch A");
            Close(oracle.Last([.. historyB, .. nextB]), ReadLast(batch[1]), "batch B");
            float[] savedBatchB = ReadLast(batch[1]);
            batch[0].Dispose();
            a.Rollback(0);
            using var reused = model.Forward([4], [0], 0, a);
            Assert.Equal(savedBatchB, ReadLast(batch[1]));
            Assert.Equal(savedA, ReadLast(firstA));
            Close(oracle.Last([4]), ReadLast(reused), "released request reuse");
        }
        finally { foreach (ITensor result in batch) result.Dispose(); }
        Assert.Equal(historyB.Length + nextB.Length, b.CurrentLength);
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void LogicalContextsAllocateOnlyLiveF16Pages_AndModelInstancesAccountSeparately()
    {
        RequireGpu();
        Fixture fixture = CreateFixture(false);
        using var model = LoadHipAndCloseMapping(fixture);
        using var secondModel = LoadHipAndCloseMapping(fixture);
        using var oracle = new CpuOracle(fixture);
        var requests = Enumerable.Range(0, 32).Select(_ => model.CreateKvCache(LogicalContext, PageTokens)).ToArray();
        using var secondCache = secondModel.CreateKvCache(LogicalContext, PageTokens);
        try
        {
            Assert.All(requests, cache => { Assert.Equal(LogicalContext, cache.MaxLength); Assert.Equal(0, cache.AllocatedBytes); });
            Assert.Equal(0, secondCache.AllocatedBytes);
            Assert.Equal(fixture.StoredBytes, model.StoredWeightBytes);
            Assert.Equal(fixture.StoredBytes, secondModel.StoredWeightBytes);
            Assert.Equal(model.WeightMemoryBytes, secondModel.WeightMemoryBytes);
            long weights = model.WeightMemoryBytes + secondModel.WeightMemoryBytes;
            using (var first = model.Forward([5], [0], 0, requests[0]))
                Close(oracle.Last([5]), ReadLast(first), "first model");
            int[] history = Tokens(9, 19);
            using (var second = secondModel.Forward(history, Positions(0, history.Length), 0, secondCache, true))
                Close(oracle.Last(history), ReadLast(second), "second model");
            AssertCacheBytes(requests[0], 1, 1);
            AssertCacheBytes(secondCache, 3, 4);
            Assert.All(requests.Skip(1), cache => Assert.Equal(0, cache.AllocatedBytes));
            Assert.Equal(weights, model.WeightMemoryBytes + secondModel.WeightMemoryBytes);
            output.WriteLine($"32 request slots each reserve {LogicalContext} logical tokens with 0 initial KV bytes. Two model instances: weights={weights} bytes; live KV={requests[0].AllocatedBytes + secondCache.AllocatedBytes} bytes; F16 KV={2L * Layers * KvHeads * HeadDim * sizeof(ushort)} bytes/token before page/table overhead.");
        }
        finally { foreach (var cache in requests) cache.Dispose(); }
        Assert.All(requests, cache => Assert.Equal(0, cache.AllocatedBytes));
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void InvalidRequestsAndForeignCaches_DoNotAdvanceHistory()
    {
        RequireGpu();
        Fixture fixture = CreateFixture(false);
        using var model = LoadHipAndCloseMapping(fixture);
        using var other = LoadHipAndCloseMapping(fixture);
        using var oracle = new CpuOracle(fixture);
        using var cache = model.CreateKvCache(4, PageTokens);
        using var foreign = other.CreateKvCache(4, PageTokens);
        using var host = new SimpleKvCache(Layers, KvHeads, HeadDim, 4);
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([1], [0], 0, foreign); });
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([1], [0], 0, host); });
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([1], [0], 1, cache); });
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([], [], 0, cache); });
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([1, 2], [0], 0, cache, true); });
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([-1], [0], 0, cache); });
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([Vocab], [0], 0, cache); });
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([1], [1], 0, cache); });
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([1, Vocab], [0, 1], 0, cache, true); });
        Assert.Throws<NotSupportedException>(() => { using var rejected = model.Forward([1, 2], [0, 1], 0, cache); });
        Assert.Equal(0, cache.AllocatedBytes);
        Assert.Equal(0, cache.CurrentLength);
        using (var valid = model.Forward([1, 2, 3, 4], [0, 1, 2, 3], 0, cache, true))
            Close(oracle.Last([1, 2, 3, 4]), ReadLast(valid), "recovery after rejected requests");
        long bytes = cache.AllocatedBytes;
        Assert.Throws<ArgumentException>(() => { using var rejected = model.Forward([5], [4], 0, cache); });
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Rollback(5));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Rollback(-1));
        Assert.Equal(4, cache.CurrentLength);
        Assert.Equal(bytes, cache.AllocatedBytes);
        Assert.Throws<NotSupportedException>(() => { using var rejected = cache.GetKeys(0); });
        Assert.Throws<NotSupportedException>(() => cache.GetValuesRef(0));
#pragma warning disable IDISP016, IDISP017 // Deliberately exercise disposed-object guards and idempotent cleanup.
        cache.Dispose();
        Assert.Throws<ObjectDisposedException>(() => { using var rejected = model.Forward([1], [0], 0, cache); });
        Assert.Equal(0, cache.AllocatedBytes);
        model.Dispose();
        Assert.Equal(0, model.ComputeMemoryBytes);
        Assert.Equal(0, model.WeightMemoryBytes);
        Assert.Throws<ObjectDisposedException>(() => { using var rejected = model.Forward([1], [0], 0); });
        Assert.Throws<ObjectDisposedException>(() => { using var rejected = model.CreateKvCache(4); });
#pragma warning restore IDISP016, IDISP017
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void OverBudgetPageFailsBeforeAllocation_AndModelRemainsUsable()
    {
        RequireGpu();
        Fixture fixture = CreateFixture(false);
        using var model = LoadHipAndCloseMapping(fixture);
        using var oracle = new CpuOracle(fixture);
        using var impossible = model.CreateKvCache(8, int.MaxValue);
        Assert.Throws<OutOfMemoryException>(() => { using var rejected = model.Forward([7], [0], 0, impossible); });
        Assert.Equal(0, impossible.CurrentLength);
        Assert.Equal(0, impossible.AllocatedBytes);
        using var valid = model.CreateKvCache(8, PageTokens);
        using var logits = model.Forward([7], [0], 0, valid);
        Close(oracle.Last([7]), ReadLast(logits), "recovery after allocation admission failure");
    }

    [Theory]
    [InlineData("shape")]
    [InlineData("missing")]
    [InlineData("matrix-f32")]
    [InlineData("matrix-bf16")]
    [InlineData("matrix-q5")]
    [InlineData("extra")]
    public void UnsupportedWeightsFailBeforeDeviceInitialization(string invalid)
    {
        Fixture fixture = CreateFixture(false, invalid);
        using var file = GgufFile.Open(fixture.Path);
        // A negative device proves validation runs before creating a native context.
        Exception error = Assert.ThrowsAny<Exception>(() => { using var rejected = HipTransformerModel.LoadFromGguf(file, fixture.Config, deviceId: -1); });
        if (invalid is "shape" or "missing") Assert.IsType<InvalidDataException>(error);
        else Assert.IsType<NotSupportedException>(error);
        Assert.Contains(invalid is "extra" ? "bias" : "token_embd.weight", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void UnsupportedConfigurationsFailBeforeDeviceInitialization()
    {
        Fixture fixture = CreateFixture(false);
        using var file = GgufFile.Open(fixture.Path);
        var invalid = new[]
        {
            fixture.Config with { HiddenSize = 0 },
            fixture.Config with { NumKvHeads = 3 },
            fixture.Config with { HeadDim = 320, HiddenSize = 1280 },
            fixture.Config with { Architecture = Architecture.Mistral },
            fixture.Config with { NormEpsilon = float.NaN },
            fixture.Config with { SlidingWindowSize = 4 },
            fixture.Config with { FinalLogitSoftcap = 3 },
            fixture.Config with { RoPEConfig = new RoPEConfig(DimensionCount: 39) },
            fixture.Config with { RoPEConfig = new RoPEConfig(ScalingType: RoPEScalingType.YaRN, ScalingFactor: 2) },
        };
        foreach (ModelConfig config in invalid)
        {
            Exception error = Assert.ThrowsAny<Exception>(() => { using var rejected = HipTransformerModel.LoadFromGguf(file, config, deviceId: -1); });
            Assert.True(error is InvalidDataException or NotSupportedException, error.ToString());
        }
    }

    private static void RequireGpu()
    {
        bool available = OperatingSystem.IsWindows() && HipDevice.IsAvailable();
        if (Environment.GetEnvironmentVariable("DOTLLM_TEST_HIP_REQUIRED") is "1") Assert.True(available, "Required Windows HIP GPU is unavailable.");
        Skip.IfNot(available, "Windows HIP/DirectStorage is unavailable.");
    }

    private static HipTransformerModel LoadHipAndCloseMapping(Fixture fixture)
    {
        using var file = GgufFile.Open(fixture.Path);
        return HipTransformerModel.LoadFromGguf(file, fixture.Config);
    }

    private static int[] Tokens(int count, int seed) => Enumerable.Range(0, count).Select(i => (seed + i * 17 + i * i * 3) % Vocab).ToArray();
    private static int[] Positions(int start, int count) => Enumerable.Range(start, count).ToArray();
    private static unsafe float[] ReadLast(ITensor tensor)
        => new ReadOnlySpan<float>((float*)tensor.DataPointer + (tensor.Shape[0] - 1) * Vocab, Vocab).ToArray();

    private static void AssertCacheBytes(HipPagedKvCache cache, int pages, int tableCapacity)
    {
        long payload = (long)pages * Layers * 2 * PageTokens * KvHeads * HeadDim * sizeof(ushort);
        long tables = (long)tableCapacity * Layers * 2 * IntPtr.Size;
        Assert.Equal(pages, cache.AllocatedPages);
        Assert.Equal(payload + tables, cache.AllocatedBytes);
    }

    private void Close(float[] expected, float[] actual, string context, double absoluteTolerance = 0.003)
    {
        Assert.Equal(expected.Length, actual.Length);
        double squaredError = 0, squaredReference = 0, maxError = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{context}: nonfinite logit {i}");
            double error = Math.Abs(expected[i] - actual[i]);
            maxError = Math.Max(maxError, error);
            squaredError += error * error;
            squaredReference += expected[i] * expected[i];
            Assert.True(error <= absoluteTolerance + 0.003 * Math.Abs(expected[i]), $"{context}: logit {i}, expected {expected[i]:R}, actual {actual[i]:R}, error {error:G6}");
        }
        double relative = Math.Sqrt(squaredError / squaredReference);
        Assert.True(relative <= 0.003, $"{context}: relative L2 error {relative:G6}");
        Assert.Equal(Array.IndexOf(expected, expected.Max()), Array.IndexOf(actual, actual.Max()));
        output.WriteLine($"{context}: max abs={maxError:G6}, relative L2={relative:G6}, top1={Array.IndexOf(actual, actual.Max())}");
    }

    private sealed class CpuOracle : IDisposable
    {
        private readonly GgufFile _file;
        private readonly TransformerModel _model;
        private readonly ModelConfig _config;
        public CpuOracle(Fixture fixture)
        {
            _config = fixture.Config;
            _file = GgufFile.Open(fixture.Path);
            try { _model = TransformerModel.LoadFromGguf(_file, fixture.Config with { MaxSequenceLength = 128 }, new ThreadingConfig(1, EnableCallerPinning: false)); }
            catch { _file.Dispose(); throw; }
        }
        public float[] Last(int[] history, KvCacheDType keyType = KvCacheDType.F32,
            KvCacheDType valueType = KvCacheDType.F32, bool referenceKv = false)
        {
            using IKvCache cache = referenceKv ? new ReferenceKvCache(_config, keyType, valueType)
                : keyType == KvCacheDType.F32 && valueType == KvCacheDType.F32
                    ? new SimpleKvCache(_config.NumLayers, _config.NumKvHeads, _config.HeadDim, 128)
                    : new QuantizedKvCache(_config.NumLayers, _config.NumKvHeads, _config.HeadDim, 128, keyType, valueType, windowSize: 0);
            using ITensor logits = _model.Forward(history, Positions(0, history.Length), -1, cache);
            Assert.Equal(history.Length, logits.Shape[0]);
            return ReadLast(logits);
        }
        public void Dispose() { _model.Dispose(); _file.Dispose(); }
    }

    private sealed record Fixture(string Path, ModelConfig Config, long StoredBytes);

    private static Fixture CreateFixture(bool tied, string? invalid = null,
        QuantizationType matrixType = QuantizationType.F16, int? headDimension = null, int intermediate = Intermediate)
    {
        int headDim = headDimension ?? HeadDim, hidden = Heads * headDim;
        var config = new ModelConfig
        {
            Architecture = Architecture.Llama, HiddenSize = hidden, IntermediateSize = intermediate,
            NumLayers = Layers, NumAttentionHeads = Heads, NumKvHeads = KvHeads, HeadDim = headDim,
            VocabSize = Vocab, MaxSequenceLength = LogicalContext, NormEpsilon = 1e-5f,
            TiedEmbeddings = tied, RoPEConfig = new RoPEConfig(Theta: 10000, DimensionCount: headDim),
        };
        string[] vocabulary = ["<unk>", "<s>", "</s>", .. Enumerable.Range(33, 94).Select(c => ((char)c).ToString())];
        var writer = new GgufWriter()
            .AddString("general.architecture", "llama").AddString("general.name", "dotLLM HIP synthetic F16 oracle")
            .AddUInt32("llama.context_length", LogicalContext).AddUInt32("llama.embedding_length", (uint)hidden)
            .AddUInt32("llama.block_count", Layers).AddUInt32("llama.feed_forward_length", (uint)intermediate)
            .AddUInt32("llama.attention.head_count", Heads).AddUInt32("llama.attention.head_count_kv", KvHeads)
            .AddFloat32("llama.attention.layer_norm_rms_epsilon", config.NormEpsilon)
            .AddFloat32("llama.rope.freq_base", 10000).AddUInt32("llama.rope.dimension_count", (uint)headDim)
            .AddString("tokenizer.ggml.model", "gpt2").AddStringArray("tokenizer.ggml.tokens", vocabulary)
            .AddStringArray("tokenizer.ggml.merges", []).AddUInt32("tokenizer.ggml.bos_token_id", 1)
            .AddUInt32("tokenizer.ggml.eos_token_id", 2).AddUInt32("tokenizer.ggml.unknown_token_id", 0);
        var random = new Random(1784);
        long storedBytes = 0;
        void Add(string name, int[] shape, bool norm = false, float amplitude = 0.045f)
        {
            int count = shape.Aggregate(1, (product, value) => checked(product * value));
            Half[] values = Enumerable.Range(0, count).Select(_ => (Half)(norm ? 0.9f + 0.2f * random.NextSingle() : amplitude * (2 * random.NextSingle() - 1))).ToArray();
            if (name is "token_embd.weight" && invalid is "missing") return;
            if (name is "token_embd.weight" && invalid is "shape") shape = [hidden / 2, Vocab * 2];
            bool f32 = norm || (name is "token_embd.weight" && invalid is "matrix-f32");
            byte[] data = f32 ? MemoryMarshal.AsBytes(values.Select(value => (float)value).ToArray().AsSpan()).ToArray()
                : MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
            uint type = f32 ? (uint)QuantizationType.F32 : (uint)matrixType;
            if (!f32 && matrixType != QuantizationType.F16) data = PackFixtureWeights(values, matrixType);
            if (name is "token_embd.weight" && invalid is "matrix-bf16") type = (uint)QuantizationType.BF16;
            if (name is "token_embd.weight" && invalid is "matrix-q5")
            {
                type = (uint)QuantizationType.Q5_0;
                // A real packed-size payload, padded only to the next GGUF tensor boundary.
                data = new byte[count / 32 * 22];
            }
            storedBytes += norm ? count * sizeof(ushort) : data.Length;
            if (data.Length % 32 != 0) Array.Resize(ref data, (data.Length + 31) / 32 * 32);
            Assert.Equal(0, data.Length % 32);
            writer.AddTensor(name, shape, type, data);
        }
        Add("token_embd.weight", [hidden, Vocab], amplitude: 0.4f);
        Add("output_norm.weight", [hidden], norm: true);
        if (!tied) Add("output.weight", [hidden, Vocab]);
        for (int layer = 0; layer < Layers; layer++)
        {
            string p = $"blk.{layer}.";
            Add(p + "attn_norm.weight", [hidden], norm: true);
            Add(p + "attn_q.weight", [hidden, hidden]);
            Add(p + "attn_k.weight", [hidden, KvHeads * headDim]);
            Add(p + "attn_v.weight", [hidden, KvHeads * headDim]);
            Add(p + "attn_output.weight", [hidden, hidden]);
            Add(p + "ffn_norm.weight", [hidden], norm: true);
            Add(p + "ffn_gate.weight", [hidden, intermediate]);
            Add(p + "ffn_up.weight", [hidden, intermediate]);
            Add(p + "ffn_down.weight", [intermediate, hidden]);
        }
        if (invalid is "extra") Add("blk.0.attn_q.bias", [hidden]);
        string directory = Path.Combine(Path.GetTempPath(), "dotllm-hip-dense-fixtures");
        Directory.CreateDirectory(directory);
        string geometry = headDimension is null && matrixType == QuantizationType.F16 && intermediate == Intermediate
            ? "f16" : $"{matrixType}-h{hidden}-f{intermediate}";
        string path = Path.Combine(directory, $"dense-{geometry}-seed1784-{(tied ? "tied" : "untied")}-{invalid ?? "valid"}.gguf");
        byte[] bytes = writer.Build();
        if (!File.Exists(path) || !File.ReadAllBytes(path).AsSpan().SequenceEqual(bytes)) File.WriteAllBytes(path, bytes);
        return new Fixture(path, config, storedBytes);
    }
}
