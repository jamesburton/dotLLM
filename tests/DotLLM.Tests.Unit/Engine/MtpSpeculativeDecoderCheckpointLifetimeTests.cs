using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Lifetime of the recurrent-state checkpoint <see cref="MtpSpeculativeDecoder"/> takes before a
/// verify batch (issue #435).
/// </summary>
/// <remarks>
/// <para>
/// <see cref="IModel.CheckpointRecurrentState"/> may return an object owning <b>device</b> memory:
/// <c>VulkanGdnStateCache.Clone</c> allocates one conv buffer and one state buffer per GDN layer —
/// 96 device buffers per round on Bonsai 2's 48 GDN layers — and <c>CudaGdnStateCache.Clone</c>
/// does the equivalent <c>cuMemAlloc</c>. A round that drops it on the floor leaks that much per
/// round, and a decode loop runs thousands of rounds.
/// </para>
/// <para>
/// These tests are backend-free on purpose: the property under test belongs to the decoder, not to
/// any backend, and it is observable through <see cref="IDisposable"/> alone. The throwing case is
/// the one with teeth — disposal at the normal return paths is easy to write and easy to believe,
/// while every path between the checkpoint and the return (the verify <c>Forward</c>, the logit
/// reads, <c>RollbackState</c>'s own restore) can throw, and on a GPU backend an allocator failure
/// in exactly that window is a likely reason for it to.
/// </para>
/// </remarks>
public sealed class MtpSpeculativeDecoderCheckpointLifetimeTests
{
    private const int VocabSize = 16;
    private const int MaxSeqLen = 128;
    private const int NumKvHeads = 1;
    private const int HeadDim = 4;
    private const int HiddenSize = 8;

    /// <summary>
    /// The verify-batch <c>Forward</c> throws after the checkpoint has been taken. The exception
    /// must propagate <b>and</b> the checkpoint must have been disposed.
    /// </summary>
    [Fact]
    public void DraftAndVerify_VerifyForwardThrows_StillDisposesTheRecurrentCheckpoint()
    {
        var checkpoints = new List<TrackedCheckpoint>();
        using var model = new CheckpointingMockModel(checkpoints, throwOnVerifyBatch: true);
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        using var kvCache = new SimpleKvCache(1, NumKvHeads, HeadDim, MaxSeqLen);
        using var mtpState = new CpuMtpState(HiddenSize, NumKvHeads, HeadDim, maxSteps: 32);
        var generatedIds = new List<int> { 1 };
        int[] outputBuffer = new int[8];

        Assert.Throws<SimulatedDeviceAllocationFailure>(() =>
            decoder.DraftAndVerify(model, kvCache, mtpState, pipeline, generatedIds,
                constraint: null, position: 1, vocabSize: VocabSize, numCandidates: 3, outputBuffer));

        // Setup guard: if no checkpoint was ever taken the assertion below is vacuous and would
        // pass against a decoder that never disposes anything.
        Assert.Single(checkpoints);
        Assert.True(checkpoints[0].Disposed,
            "The recurrent-state checkpoint was still alive when the verify Forward threw. On a GPU " +
            "backend that leaks one device buffer per GDN layer, per round.");
    }

    /// <summary>The ordinary all-accepted round disposes its checkpoint too.</summary>
    [Fact]
    public void DraftAndVerify_NormalRound_DisposesTheRecurrentCheckpoint()
    {
        var checkpoints = new List<TrackedCheckpoint>();
        using var model = new CheckpointingMockModel(checkpoints, throwOnVerifyBatch: false);
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        using var kvCache = new SimpleKvCache(1, NumKvHeads, HeadDim, MaxSeqLen);
        using var mtpState = new CpuMtpState(HiddenSize, NumKvHeads, HeadDim, maxSteps: 32);
        var generatedIds = new List<int> { 1 };
        int[] outputBuffer = new int[8];

        var result = decoder.DraftAndVerify(model, kvCache, mtpState, pipeline, generatedIds,
            constraint: null, position: 1, vocabSize: VocabSize, numCandidates: 3, outputBuffer);

        Assert.True(result.AcceptedCount > 0);
        Assert.Single(checkpoints);
        Assert.True(checkpoints[0].Disposed, "The recurrent-state checkpoint outlived its round.");
    }

    /// <summary>Stand-in for a device-allocation failure inside the verify batch.</summary>
    private sealed class SimulatedDeviceAllocationFailure : Exception
    {
        public SimulatedDeviceAllocationFailure()
            : base("simulated device-allocation failure in the verify batch") { }
    }

    private sealed class TrackedCheckpoint : IDisposable
    {
        public bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }

    /// <summary>
    /// Minimal MTP-capable model that also reports <see cref="IModel.SupportsRecurrentStateCheckpoint"/>,
    /// hands out a disposal-tracking checkpoint, and optionally throws from the multi-token verify
    /// batch (single-token calls — the catchup forward — always succeed, so the round reaches the
    /// checkpoint first).
    /// </summary>
    private sealed class CheckpointingMockModel : IModel
    {
        private readonly List<TrackedCheckpoint> _checkpoints;
        private readonly bool _throwOnVerifyBatch;

        public CheckpointingMockModel(List<TrackedCheckpoint> checkpoints, bool throwOnVerifyBatch)
        {
            _checkpoints = checkpoints;
            _throwOnVerifyBatch = throwOnVerifyBatch;
        }

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = 1,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HiddenSize,
            IntermediateSize = HiddenSize * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;
        public void Dispose() { }

        public bool SupportsMtp => true;
        public IMtpState? CreateMtpState() => new CpuMtpState(HiddenSize, NumKvHeads, HeadDim, maxSteps: 32);

        public bool SupportsRecurrentStateCheckpoint => true;

        public object? CheckpointRecurrentState()
        {
            var c = new TrackedCheckpoint();
            _checkpoints.Add(c);
            return c;
        }

        public void RestoreRecurrentState(object? checkpoint) { }

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null, null, null);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
            => Forward(tokenIds, positions, deviceId, kvCache, null, null);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache, DotLLM.Core.Lora.ILoraAdapter? adapter)
            => Forward(tokenIds, positions, deviceId, kvCache, adapter, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache, DotLLM.Core.Lora.ILoraAdapter? adapter, IMtpState? mtpState)
        {
            if (_throwOnVerifyBatch && tokenIds.Length > 1)
                throw new SimulatedDeviceAllocationFailure();

            int batchSize = tokenIds.Length;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)batchSize * VocabSize * sizeof(float)), 64);
            float* dst = (float*)ptr;
            for (int t = 0; t < batchSize; t++)
            {
                var row = new Span<float>(dst + (long)t * VocabSize, VocabSize);
                row.Fill(-10f);
                row[Successor(tokenIds[t])] = 10f;
            }

            if (kvCache != null)
            {
                int kvStride = NumKvHeads * HeadDim;
                nuint bytes = (nuint)((long)batchSize * kvStride * sizeof(float));
                nint kPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
                NativeMemory.Clear((void*)kPtr, bytes);
                NativeMemory.Clear((void*)vPtr, bytes);
                kvCache.Update(new TensorRef(batchSize, kvStride, DType.Float32, -1, kPtr),
                               new TensorRef(batchSize, kvStride, DType.Float32, -1, vPtr), positions, 0);
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }

            if (mtpState is CpuMtpState cap)
            {
                float[] fake = new float[batchSize * HiddenSize];
                cap.SetCapturedRows(fake, batchSize);
            }

            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, ptr);
        }

        public unsafe ITensor ForwardMtp(IMtpState state, int tokenId, int position)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(VocabSize * sizeof(float)), 64);
            var row = new Span<float>((float*)ptr, VocabSize);
            row.Fill(-10f);
            row[Successor(tokenId)] = 10f;
            if (state is CpuMtpState cap) cap.Advance();
            return new UnmanagedTensor(new TensorShape(1, VocabSize), DType.Float32, deviceId: -1, ptr);
        }

        // MTP and trunk agree everywhere, so the non-throwing round accepts every draft and takes
        // the all-accepted return path.
        private static int Successor(int t) => (t + 1) % VocabSize;
    }
}
