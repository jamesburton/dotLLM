using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-sequence QSA indexer state on the device (issue #819): for every QSA layer the RAW (un-normed, un-rotated) indexer key of every
/// cached position and the block-pooled keys. Pooled block <c>b</c> is a pure function of raw rows <c>R*b..R*b+R-1</c>, so a rollback is
/// just a smaller sequence length - blocks that were not complete at the restored length are never read and are re-derived when their
/// last row is rewritten.
/// </summary>
public sealed class VulkanQwen4ExpIndexerState : IDisposable
{
    internal VulkanDevice.Buffer[] Raw { get; }
    internal VulkanDevice.Buffer[] Pooled { get; }

    /// <summary>Positions the buffers can hold.</summary>
    public int Capacity { get; }

    /// <summary>Device bytes held.</summary>
    public long Bytes { get; }

    internal VulkanQwen4ExpIndexerState(VulkanDevice device, int qsaLayers, int capacity, int dim, int blockSize)
    {
        Capacity = capacity;
        Raw = new VulkanDevice.Buffer[qsaLayers];
        Pooled = new VulkanDevice.Buffer[qsaLayers];
        long raw = (long)capacity * dim * 4, pooled = (long)(capacity / blockSize + 1) * dim * 4;
        for (int i = 0; i < qsaLayers; i++)
        {
            Raw[i] = device.AllocateDeviceLocal(raw);
            Pooled[i] = device.AllocateDeviceLocal(pooled);
        }
        Bytes = qsaLayers * (raw + pooled);
    }

    /// <summary>Device bytes the indexer state of <paramref name="qsaLayers"/> layers needs for <paramref name="capacity"/> positions.</summary>
    public static long BytesFor(int qsaLayers, int capacity, int dim, int blockSize)
        => qsaLayers * ((long)capacity * dim * 4 + (long)(capacity / blockSize + 1) * dim * 4);

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var b in Raw) b?.Dispose();
        foreach (var b in Pooled) b?.Dispose();
    }
}

/// <summary>
/// The QSA path of the Vulkan qwen4exp forward: keeps the indexer key cache current on EVERY forward (so a long context can cross
/// the dense limit mid-sequence) and, once a forward reaches positions whose complete-block count exceeds the budget, replaces the
/// dense attention of the QSA layers with indexer scoring, exact top-k block selection and gather attention (issue #819).
/// Installed on the trunk core only; the MTP draft head keeps attending densely (a draft only has to be a good guess).
/// </summary>
internal sealed class VulkanQwen4ExpQsa : IQ4AttentionHook, IDisposable
{
    internal sealed class LayerWeights : IDisposable
    {
        public required VulkanDevice.Buffer QProj, KProj, QGamma, KGamma;
        public required QuantizationType QQt, KQt;
        public long Bytes;
        public void Dispose() { QProj.Dispose(); KProj.Dispose(); QGamma.Dispose(); KGamma.Dispose(); }
    }

    private readonly VulkanDevice _device;
    private readonly VulkanQwen3MoeHybridTransformerModel _core;
    private readonly Qwen4ExpQsaKernels _k;
    private readonly LayerWeights[] _w;
    private readonly int[] _ordinal;
    private readonly int _heads, _dim, _blockSize, _budgetBlocks, _ropeDim, _hidden, _attnHeads, _kvHeads, _headDim, _capacity, _nbCap;
    private readonly float _theta, _eps;
    private VulkanDevice.Buffer? _iq, _ik, _scores, _sel, _partOut, _partMs;
    private int _rows, _qc, _qcAlloc;
    private long _partFloats;
    private VulkanQwen4ExpIndexerState? _cur;

    /// <summary>
    /// Diagnostic A/B switch (<c>DOTLLM_VK_QWEN4EXP_QSA=0</c> at startup): off = the QSA layers run the plain dense attention and the indexer is not
    /// maintained, which is only correct up to <see cref="DenseLimit"/> tokens (a longer forward then throws). Measures what the indexer costs at short context.
    /// </summary>
    public static bool Enabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_QWEN4EXP_QSA") != "0";

    /// <summary>Largest query sub-chunk (bounds the score / partial scratch regardless of the prefill chunk size).</summary>
    internal const int MaxQueryChunk = 1024;
    private const long ScoreScratchFloats = 12L << 20;   // 48 MiB

    /// <summary>Largest total of (queries x blocks) the score scratch holds; see <see cref="MaxQueryChunk"/>.</summary>
    internal int QueryChunk { get; }

    /// <summary>Tokens a query may reach before its complete-block count exceeds the budget (dense below, sparse above).</summary>
    public int DenseLimit { get; }

    /// <summary>Per-layer QSA ordinal (-1 for GDN layers).</summary>
    internal int[] Ordinal => _ordinal;

    /// <summary>Test hook: how many sparse (scored) layer recordings happened.</summary>
    internal long SparseLayerRecordings { get; private set; }

    /// <summary>Test hook: the block selection scratch <c>[queries, budget]</c> ints of the most recent sparse sub-chunk.</summary>
    internal VulkanDevice.Buffer? SelectionBuffer => _sel;

    /// <summary>Test hook: the block score scratch <c>[queries, NbCap]</c> floats of the most recent sparse sub-chunk.</summary>
    internal VulkanDevice.Buffer? ScoresBuffer => _scores;

    /// <summary>Test hook: row stride of <see cref="ScoresBuffer"/>.</summary>
    internal int NbCap => _nbCap;

    public VulkanQwen4ExpQsa(VulkanDevice device, VulkanQwen3MoeHybridTransformerModel core, Qwen4ExpQsaKernels kernels, LayerWeights[] weights,
        int[] ordinal, ModelConfig config, Qwen4ExpConfig q4, int capacity)
    {
        _device = device; _core = core; _k = kernels; _w = weights; _ordinal = ordinal;
        _heads = q4.IndexerHeadCount; _dim = q4.IndexerKeyLength; _blockSize = q4.IndexerBlockSize;
        _budgetBlocks = q4.IndexerTopK / q4.IndexerBlockSize;
        _ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        _theta = config.RoPEConfig?.Theta ?? 10000.0f;
        _eps = config.NormEpsilon;
        _hidden = config.HiddenSize; _attnHeads = config.NumAttentionHeads; _kvHeads = config.NumKvHeads; _headDim = config.HeadDim;
        _capacity = capacity;
        _nbCap = capacity / _blockSize + 1;
        DenseLimit = q4.IndexerTopK + q4.IndexerBlockSize - 1;
        QueryChunk = (int)Math.Clamp(ScoreScratchFloats / _nbCap, 1, MaxQueryChunk);
    }

    /// <summary>Sets the sequence whose indexer the next recordings use (null clears it).</summary>
    public void Begin(VulkanQwen4ExpIndexerState? state) => _cur = state;

    /// <summary>Grows the scratch for a forward of <paramref name="rows"/> rows. Call BEFORE recording starts (it may re-create buffers).</summary>
    public void EnsureRows(int rows)
    {
        bool changed = false;
        if (_iq is null || rows > _rows)
        {
            _iq?.Dispose(); _ik?.Dispose();
            _iq = _device.AllocateDeviceLocal((long)rows * _heads * _dim * 4);
            _ik = _device.AllocateDeviceLocal((long)rows * _dim * 4);
            _rows = rows;
            changed = true;
        }
        int qc = Math.Min(rows, QueryChunk);
        if (_scores is null || qc > _qcAlloc)
        {
            _scores?.Dispose(); _sel?.Dispose();
            _scores = _device.AllocateDeviceLocal((long)qc * _nbCap * 4);
            _sel = _device.AllocateDeviceLocal((long)qc * _budgetBlocks * 4);
            _qcAlloc = qc;
            changed = true;
        }
        long partRows = 1;
        for (int n = 1; n <= qc; n++) partRows = Math.Max(partRows, (long)n * MaxSplitsFor(n));
        partRows *= _attnHeads;
        if (_partOut is null || partRows > _partFloats)
        {
            _partOut?.Dispose(); _partMs?.Dispose();
            _partOut = _device.AllocateDeviceLocal(partRows * _headDim * 4);
            _partMs = _device.AllocateDeviceLocal(partRows * 2 * 4);
            _partFloats = partRows;
            changed = true;
        }
        if (changed) _k.InvalidateDescriptorCache();
        _qc = qc;
    }

    private int MaxSplitsFor(int queries) => Math.Clamp((256 + queries * _attnHeads - 1) / (queries * _attnHeads), 1, 16);

    /// <inheritdoc/>
    public bool RecordAttention(nint cmd, int layer, int seqLen, ReadOnlySpan<int> positions,
        VulkanDevice.Buffer kSrc, VulkanDevice.Buffer vSrc)
    {
        var ist = _cur;
        if (!Enabled || ist is null || _ordinal[layer] < 0) return false;
        int ord = _ordinal[layer];
        var w = _w[ord];
        var st = _core.Q4State;
        int first = positions[0];
        if (first + seqLen > ist.Capacity) throw new InvalidOperationException("QSA indexer capacity exceeded.");
        bool sparse = first + seqLen > DenseLimit;

        // Raw key of every row -> the position-indexed store; the (cheap) key projection runs on every forward so history exists
        // when a later forward crosses the dense limit.
        _core.Q4RecordMatmul(cmd, w.KProj, w.KQt, st.NormOutput, _ik!, outputDim: _dim, inputDim: _hidden, seqLen: seqLen);
        if (sparse)
            _core.Q4RecordMatmul(cmd, w.QProj, w.QQt, st.NormOutput, _iq!, outputDim: _heads * _dim, inputDim: _hidden, seqLen: seqLen);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _ik!, ist.Raw[ord], 0, (ulong)((long)first * _dim * 4), (ulong)((long)seqLen * _dim * 4));
        KernelSupport.ComputeTransferFullBarrier(cmd);

        int firstBlock = first / _blockSize, endBlock = (first + seqLen) / _blockSize;
        _k.RecordPool(cmd, ist.Raw[ord], w.KGamma, ist.Pooled[ord], firstBlock, endBlock - firstBlock, _dim, _ropeDim, _blockSize, _eps, _theta);
        if (!sparse)
        {
            KernelSupport.ComputeToComputeBarrier(cmd);
            return false;
        }

        // Indexer queries: per-head RMSNorm, then NeoX RoPE at the query position (the Rope kernel also rotates its K operand: the
        // already-copied scratch key, which is dead after this point).
        var kern = _core.Q4Kernels;
        kern.RmsNorm.Record(cmd, _iq!, w.QGamma, _iq!, rowCount: seqLen * _heads, n: _dim, eps: _eps);
        KernelSupport.ComputeToComputeBarrier(cmd);
        kern.Rope.Record(cmd, _iq!, _ik!, st.PositionsBuffer, seqLen: seqLen, numHeads: _heads, numKvHeads: 1, headDim: _dim,
            ropeDim: _ropeDim, theta: _theta, variant: RopeF32Kernel.Variant.NeoX);
        KernelSupport.ComputeToComputeBarrier(cmd);

        for (int q0 = 0; q0 < seqLen; q0 += _qc)
        {
            int n = Math.Min(_qc, seqLen - q0);
            int nbMax = (first + q0 + n) / _blockSize;
            if (nbMax > _budgetBlocks)
            {
                _k.RecordScore(cmd, _iq!, ist.Pooled[ord], _scores!, q0, n, first, _heads, _dim, _blockSize, _budgetBlocks, _nbCap, nbMax);
                KernelSupport.ComputeToComputeBarrier(cmd);
                _k.RecordSelect(cmd, _scores!, _sel!, q0, n, first, _blockSize, _budgetBlocks, _nbCap);
                KernelSupport.ComputeToComputeBarrier(cmd);
            }
            int splits = MaxSplitsFor(n);
            _k.RecordAttention(cmd, st.Q, kSrc, vSrc, _sel!, _partOut!, _partMs!, st.AttnOutput, q0, n, first, _attnHeads, _kvHeads,
                _headDim, _blockSize, _budgetBlocks, splits);
            KernelSupport.ComputeToComputeBarrier(cmd);
        }
        SparseLayerRecordings++;
        return true;
    }

    /// <summary>Device bytes of the projection weights.</summary>
    public long WeightBytes => _w.Sum(x => x.Bytes);

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var w in _w) w.Dispose();
        _iq?.Dispose(); _ik?.Dispose(); _scores?.Dispose(); _sel?.Dispose(); _partOut?.Dispose(); _partMs?.Dispose();
        _k.Dispose();
    }
}
