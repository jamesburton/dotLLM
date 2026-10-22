using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Server;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

public sealed class RecurrentPrefixSnapshotLifecycleTests
{
    [Fact]
    public void ActiveShutdown_ReleasesHostSnapshotsBeforeModelContext()
    {
        var (model, generator) = CaptureHostSnapshot();
        var state = new ServerState { Options = new() { Model = "test" }, Model = model, Generator = generator };
        try
        {
            state.Dispose();
            Assert.Equal(0, model.HostBytesAtModelDispose);
            Assert.Equal(0, model.HostBytes);
            Assert.Equal(new[] { "snapshot", "model" }, model.Disposals);
        }
        finally { generator.ClearRecurrentPrefixCache(); }
    }

    [Fact]
    public void ResidentEviction_ReleasesHostSnapshotsBeforeModelContext()
    {
        var (model, generator) = CaptureHostSnapshot();
        using var residency = new ModelResidencyManager { MaxResidentModels = 1 };
        residency.Stash(new ResidentModelSnapshot
        {
            Key = "old", Options = new() { Model = "test" }, Generator = generator, Model = model,
            LastUsedUtc = DateTimeOffset.UtcNow,
        });
        try
        {
            Assert.Single(residency.EnforceBudget(0));
            Assert.Equal(0, model.HostBytesAtModelDispose);
            Assert.Equal(0, model.HostBytes);
            Assert.Equal(new[] { "snapshot", "model" }, model.Disposals);
        }
        finally { generator.ClearRecurrentPrefixCache(); }
    }

    private static (LifecycleModel Model, TextGenerator Generator) CaptureHostSnapshot()
    {
        var model = new LifecycleModel();
        var generator = new TextGenerator(model, new Tokenizer(),
            (config, length) => new SimpleKvCache(KvGeometry.FromConfig(config), length),
            recurrentPrefixCache: true, recurrentPrefixCacheOptions: new() { MaxEntries = 2, HostBudgetBytes = 64 });
        generator.Generate("prompt", new() { MaxTokens = 1, Temperature = 0 });
        Assert.Equal(1, generator.RecurrentPrefixStatistics.Entries);
        Assert.Equal(64, model.HostBytes);
        return (model, generator);
    }

    private sealed class Tokenizer : ITokenizer
    {
        public int VocabSize => 8;
        public int BosTokenId => 0;
        public int EosTokenId => 7;
        public int[] Encode(string text) => Enumerable.Repeat(2, 20).ToArray();
        public string Decode(ReadOnlySpan<int> tokens) => "x";
        public string DecodeToken(int tokenId) => "x";
        public int CountTokens(string text) => 20;
    }

    // The host allocation has real deterministic ownership, without requiring CUDA in a lifecycle unit test.
    private sealed unsafe class LifecycleModel : IModel, IRecurrentPrefixSnapshotProvider
    {
        public long HostBytes;
        public long HostBytesAtModelDispose = -1;
        public List<string> Disposals { get; } = [];
        public ModelConfig Config { get; } = new()
        {
            Architecture = DotLLM.Core.Configuration.Architecture.Llama, VocabSize = 8, HiddenSize = 4,
            IntermediateSize = 8, NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1,
            HeadDim = 4, MaxSequenceLength = 64,
        };
        public long ComputeMemoryBytes => 0;
        public bool RequiresPerSequenceState => true;
        public bool SupportsRecurrentStateCheckpoint => true;
        public void ResetSequenceState() { }
        public string PrefixSnapshotIdentity => "lifecycle-model";
        public bool SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage storage) => storage == RecurrentPrefixSnapshotStorage.Host;
        public long EstimatePrefixSnapshotBytes(IKvCache cache, int prefixLength) => 64;
        public IRecurrentPrefixSnapshot CapturePrefixSnapshot(IKvCache cache, int prefixLength, RecurrentPrefixSnapshotStorage storage)
        {
            var snapshot = new HostSnapshot(this, prefixLength);
            HostBytes += 64;
            return snapshot;
        }
        public IRecurrentPrefixSnapshot TransferPrefixSnapshot(IRecurrentPrefixSnapshot source, RecurrentPrefixSnapshotStorage storage) => throw new NotSupportedException();
        public void RestorePrefixSnapshot(IRecurrentPrefixSnapshot snapshot, IKvCache cache) => throw new NotSupportedException();
        public ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions, int deviceId) => Forward(tokens, positions, deviceId, null);
        public ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions, int deviceId, IKvCache? cache)
        {
            var logits = UnmanagedTensor.Allocate(new TensorShape(tokens.Length, 8), DType.Float32, -1);
            var values = new Span<float>((void*)logits.DataPointer, tokens.Length * 8);
            values.Fill(-100);
            for (int row = 0; row < tokens.Length; row++) values[row * 8 + 1] = 100;
            var kv = new TensorRef(tokens.Length, 4, DType.Float32, -1, logits.DataPointer);
            cache?.Update(kv, kv, positions, 0);
            return logits;
        }
        public void Dispose()
        {
            HostBytesAtModelDispose = HostBytes;
            Disposals.Add("model");
        }

        private sealed class HostSnapshot(LifecycleModel owner, int prefixLength) : IRecurrentPrefixSnapshot
        {
            private nint _pointer = (nint)NativeMemory.AlignedAlloc(64, 64);
            public long ByteCount => 64;
            public string Identity => owner.PrefixSnapshotIdentity;
            public int PrefixLength => prefixLength;
            public RecurrentPrefixSnapshotStorage Storage => RecurrentPrefixSnapshotStorage.Host;
            public void Dispose()
            {
                nint pointer = Interlocked.Exchange(ref _pointer, 0);
                if (pointer == 0) return;
                NativeMemory.AlignedFree((void*)pointer);
                owner.HostBytes -= 64;
                owner.Disposals.Add("snapshot");
            }
        }
    }
}
