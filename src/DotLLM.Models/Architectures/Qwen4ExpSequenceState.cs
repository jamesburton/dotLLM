using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Byte accounting of one Qwen4-Exp sequence's state, split by owner so a memory planner can size it:
/// the fixed recurrent part (<see cref="Gdn"/> — ~113 MiB on the released model — and <see cref="Ple"/>) and the parts that
/// grow with context (<see cref="IndexerPooled"/> pooled keys, <see cref="Kv"/> QSA K/V rows) plus the constant
/// <see cref="IndexerTail"/> (raw keys of the incomplete pool block).
/// </summary>
/// <param name="Gdn">Gated-DeltaNet conv + associative-memory state of every GDN layer (constant).</param>
/// <param name="Ple">PLE hash window + dilated-conv history (constant).</param>
/// <param name="IndexerPooled">Pooled indexer keys of the complete blocks (grows with context).</param>
/// <param name="IndexerTail">Raw-key tail block buffers of the QSA layers (constant).</param>
/// <param name="Kv">QSA K/V rows (grows with context); lives in the engine KV cache when one is supplied.</param>
public readonly record struct Qwen4ExpStateBytes(long Gdn, long Ple, long IndexerPooled, long IndexerTail, long Kv)
{
    /// <summary>Sum of all parts.</summary>
    public long Total => Gdn + Ple + IndexerPooled + IndexerTail + Kv;

    /// <summary>The part that does not grow with context (what a checkpoint copies besides the tails).</summary>
    public long Fixed => Gdn + Ple + IndexerTail;

    /// <summary>
    /// Logical size of a sequence state that has consumed <paramref name="contextLength"/> tokens, derived from the model
    /// configuration alone (no model instance needed).
    /// </summary>
    /// <param name="config">A fully populated <see cref="DotLLM.Core.Configuration.Architecture.Qwen4Exp"/> configuration.</param>
    /// <param name="contextLength">Tokens consumed.</param>
    public static Qwen4ExpStateBytes Estimate(ModelConfig config, int contextLength)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Qwen4Exp is not { } q4 || config.GdnConfig is not { } gdn || config.HybridLayout is not { } layout)
            throw new ArgumentException("A fully populated Architecture.Qwen4Exp configuration is required.", nameof(config));
        if (contextLength < 0) throw new ArgumentOutOfRangeException(nameof(contextLength));

        int numGdn = 0, numQsa = 0;
        long kvStrideSum = 0;
        for (int il = 0; il < config.NumLayers; il++)
        {
            if (layout.LayerKind[il] == HybridLayerKind.GatedDeltaNet) numGdn++;
            else { numQsa++; kvStrideSum += (long)layout.HeadCountKv[il] * config.HeadDim; }
        }

        long gdnBytes = (long)numGdn * (gdn.ConvStateElements + gdn.StateElements) * sizeof(float);
        long ple = 0;
        if (q4.Ple is { } p)
            ple = p.Layers.Count * ((long)(p.NgramSize - 1) * sizeof(int)
                  + (long)(p.ConvKernel - 1) * p.NgramSize * q4.HyperConnectionCount * config.HiddenSize * sizeof(float));
        long pooled = (long)numQsa * (contextLength / q4.IndexerBlockSize) * q4.IndexerKeyLength * sizeof(float);
        long tail = (long)numQsa * q4.IndexerBlockSize * q4.IndexerKeyLength * sizeof(float);
        long kv = kvStrideSum * contextLength * 2 * sizeof(float);
        return new Qwen4ExpStateBytes(gdnBytes, ple, pooled, tail, kv);
    }
}

/// <summary>
/// Per-sequence state of a <see cref="Qwen4ExpTransformerModel"/>: the Gated-DeltaNet recurrent state of every GDN layer, the
/// PLE hash window + dilated-conv history, and each QSA layer's pooled-indexer-key cache (plus its K/V rows when no engine
/// <see cref="DotLLM.Core.Attention.IKvCache"/> carries them). One instance per sequence; chunked prefill, token-by-token
/// decode and the scheduler's interleaved <see cref="IModel.ForwardBatch"/> all thread it.
/// </summary>
/// <remarks>
/// It implements <see cref="IGdnState"/> so the continuous-batch scheduler (which routes a recurrent sequence state through
/// <see cref="SequenceForwardRequest.GdnState"/>) carries it without any scheduler change. All tensor storage is native
/// memory; dispose it when the sequence ends.
/// </remarks>
public sealed class Qwen4ExpSequenceState : IGdnState
{
    internal GdnStateCache Gdn { get; }
    internal Qwen4ExpPleState?[] Ple { get; }
    internal Qwen4ExpQsaState?[] Qsa { get; }

    /// <summary>Tokens consumed so far (the next position).</summary>
    public int Length { get; internal set; }

    /// <inheritdoc/>
    public int NumGdnLayers => Gdn.NumGdnLayers;

    internal Qwen4ExpSequenceState(GdnStateCache gdn, Qwen4ExpPleState?[] ple, Qwen4ExpQsaState?[] qsa)
    {
        Gdn = gdn; Ple = ple; Qsa = qsa;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        Gdn.Reset();
        foreach (var p in Ple) p?.Reset();
        foreach (var q in Qsa) q?.Reset();
        Length = 0;
    }

    /// <summary>
    /// Copies <paramref name="other"/> (same model geometry) into this state: GDN, PLE, the QSA indexers' pooled keys + tails
    /// and, when <paramref name="other"/> keeps them itself, its own K/V rows. K/V rows held by an engine KV cache are the
    /// cache's business. A full copy is what makes a checkpoint valid even when the live state has since moved to a
    /// different history (the text generator's cross-request prefix reuse), not just along one lineage.
    /// </summary>
    public void CopyFrom(Qwen4ExpSequenceState other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (other.Ple.Length != Ple.Length) throw new ArgumentException("state geometry mismatch.", nameof(other));
        other.Gdn.CopyTo(Gdn);
        for (int i = 0; i < Ple.Length; i++)
        {
            if (Ple[i] is { } p) p.CopyFrom(other.Ple[i]!);
            if (Qsa[i] is { } q) q.CopyFrom(other.Qsa[i]!);
        }
        Length = other.Length;
    }

    /// <summary>
    /// Resident bytes split by owner (allocation sizes, including growth headroom and any row-snapshot scratch).
    /// <see cref="Qwen4ExpStateBytes.Kv"/> is the state's OWN K/V store (zero when an engine KV cache carries the rows).
    /// </summary>
    public Qwen4ExpStateBytes ResidentBytes
    {
        get
        {
            long ple = 0, pooled = 0, kv = 0, tail = 0;
            foreach (var p in Ple) ple += p?.Bytes ?? 0;
            foreach (var q in Qsa)
            {
                if (q is null) continue;
                kv += q.Bytes - q.Indexer.Bytes;
                pooled += q.Indexer.PooledBytes;
                tail += q.Indexer.Bytes - q.Indexer.PooledBytes;
            }
            return new Qwen4ExpStateBytes(Gdn.AllocatedBytes, ple, pooled, tail, kv);
        }
    }

    /// <summary>Total resident bytes (see <see cref="ResidentBytes"/>).</summary>
    public long Bytes => ResidentBytes.Total;

    /// <inheritdoc/>
    public void Dispose()
    {
        Gdn.Dispose();
        foreach (var p in Ple) p?.Dispose();
        foreach (var q in Qsa) q?.Dispose();
    }
}
