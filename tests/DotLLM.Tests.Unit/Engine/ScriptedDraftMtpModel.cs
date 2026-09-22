using DotLLM.Core.Attention;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Wraps a real MTP-capable model and overrides only what its draft head proposes (issue #473).
/// </summary>
/// <remarks>
/// A synthetic fixture's head is random, so it agrees with the trunk at chance and nearly every
/// round rejects draft 1 — the rollback then only ever restores row 0. This wrapper still runs the
/// real <see cref="IModel.ForwardMtp"/> (so the head's own state advances exactly as in
/// production) and then rewrites the logits so the draft for position <c>p</c> is the trunk's own
/// greedy token there — taken from <paramref name="greedy"/> — unless <paramref name="wrongAt"/>
/// says to miss it. Rounds therefore accept a scripted number of drafts, which makes rollback to
/// rows 1, 2, ... happen deterministically. Everything else, verify forwards included, is the
/// wrapped model's own code.
/// </remarks>
internal sealed class ScriptedDraftMtpModel(IModel inner, IReadOnlyList<int> greedy, Func<int, bool> wrongAt) : IModel
{
    public ModelConfig Config => inner.Config;
    public long ComputeMemoryBytes => inner.ComputeMemoryBytes;
    public void Dispose() { }

    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => inner.Forward(tokenIds, positions, deviceId);

    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => inner.Forward(tokenIds, positions, deviceId, kvCache);

    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter)
        => inner.Forward(tokenIds, positions, deviceId, kvCache, adapter);

    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
        => inner.Forward(tokenIds, positions, deviceId, kvCache, adapter, mtpState);

    public bool SupportsRecurrentStateCheckpoint => inner.SupportsRecurrentStateCheckpoint;
    public object? CheckpointRecurrentState() => inner.CheckpointRecurrentState();
    public void RestoreRecurrentState(object? checkpoint) => inner.RestoreRecurrentState(checkpoint);

    public bool SupportsRecurrentRowSnapshots => inner.SupportsRecurrentRowSnapshots;

    public ITensor ForwardWithRecurrentSnapshots(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                                                 int deviceId, IKvCache? kvCache, IMtpState? mtpState)
        => inner.ForwardWithRecurrentSnapshots(tokenIds, positions, deviceId, kvCache, mtpState);

    public void RestoreRecurrentStateToRow(int row) => inner.RestoreRecurrentStateToRow(row);

    public bool SupportsMtp => inner.SupportsMtp;
    public int MaxAllRowLogitsLength => inner.MaxAllRowLogitsLength;
    public IMtpState? CreateMtpState() => inner.CreateMtpState();
    public IMtpState? CreateMtpState(int maxSequenceLength) => inner.CreateMtpState(maxSequenceLength);

    public unsafe ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        ITensor logits = inner.ForwardMtp(state, tokenId, position);
        int target = position + 1;   // this step drafts the token at position + 1
        if (target < greedy.Count)
        {
            int vocab = Config.VocabSize;
            int token = wrongAt(target) ? (greedy[target] + 1) % vocab : greedy[target];
            var row = new Span<float>((void*)logits.DataPointer, vocab);
            row.Fill(-10f);
            row[token] = 10f;
        }
        return logits;
    }
}
