using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Per-layer QSA indexer key cache: the block-pooled, normed and rotated indexer keys of every COMPLETE block plus the
/// raw keys of the incomplete tail block. Pooling precedes the norm and the rotation (pool-then-rope): block <c>b</c> is
/// <c>rope(rmsnorm(mean(raw[b*R .. b*R+R-1])), position = b*R)</c>. Because a block is pooled the moment its last raw key
/// arrives, chunked prefill and token-by-token decode produce identical pooled keys to a single shot.
/// </summary>
/// <remarks>
/// <para>Native memory (<see cref="Qwen4ExpNativeBuffer"/>), grown geometrically. Pooled block <c>b</c> is a pure function
/// of raw keys <c>R*b .. R*b+R-1</c> and is never modified once written, so rolling back to token count <c>C</c> only needs
/// the <b>tail</b> (the raw keys of the incomplete block at <c>C</c>) restored: pooled rows past <c>C / R</c> are simply
/// unreachable and are re-derived bit-identically by a replay.</para>
/// </remarks>
public sealed class Qwen4ExpIndexerCache : IDisposable
{
    private readonly Qwen4ExpNativeBuffer _pooled = new();
    private readonly Qwen4ExpNativeBuffer _tailRaw = new();
    private int _tailCount;

    // Row-snapshot recording (speculative verify): the pre-chunk tail and the chunk's raw keys.
    private readonly Qwen4ExpNativeBuffer _recPreTail = new();
    private readonly Qwen4ExpNativeBuffer _recChunk = new();
    private int _recPreTokens, _recPreTailCount, _recRows;

    /// <summary>Indexer key width (128 on the released model).</summary>
    public int HeadDim { get; }

    /// <summary>Tokens per pooled block (compress ratio, 4).</summary>
    public int BlockSize { get; }

    /// <summary>Total raw keys appended so far.</summary>
    public int TokenCount { get; private set; }

    /// <summary>Number of fully pooled blocks, <c>TokenCount / BlockSize</c>.</summary>
    public int CompleteBlocks => TokenCount / BlockSize;

    /// <summary>Pooled keys of the complete blocks, <c>[CompleteBlocks, HeadDim]</c>.</summary>
    public unsafe ReadOnlySpan<float> Pooled
        => CompleteBlocks == 0 ? default : new ReadOnlySpan<float>(_pooled.Pointer, CompleteBlocks * HeadDim);

    /// <summary>Resident bytes (pooled keys + tail + any row-snapshot scratch).</summary>
    public long Bytes => _pooled.Bytes + _tailRaw.Bytes + _recPreTail.Bytes + _recChunk.Bytes;

    /// <summary>Bytes of the pooled-key store (grows with context).</summary>
    public long PooledBytes => _pooled.Bytes;

    /// <summary>Bytes of the row-snapshot recording scratch.</summary>
    public long SnapshotScratchBytes => _recPreTail.Bytes + _recChunk.Bytes;

    /// <summary>Bytes of the tail block buffer (the part a checkpoint copies).</summary>
    public long TailBytes => (long)BlockSize * HeadDim * sizeof(float);

    /// <summary>Rows of the last recorded chunk that <see cref="RestoreToRow"/> can restore.</summary>
    public int RecordedRows => _recRows;

    /// <summary>When &gt; 0, the next <see cref="Append"/> records per-row snapshots for that many rows (set by the model).</summary>
    internal int RecordRowCount { get; set; }

    /// <summary>Creates an empty cache.</summary>
    /// <param name="headDim">Indexer key width.</param>
    /// <param name="blockSize">Tokens per pooled block.</param>
    public Qwen4ExpIndexerCache(int headDim, int blockSize)
    {
        if (headDim <= 0 || blockSize <= 0) throw new ArgumentOutOfRangeException();
        HeadDim = headDim;
        BlockSize = blockSize;
        _tailRaw.EnsureExact((long)blockSize * headDim);
        _pooled.EnsureCapacity(16L * headDim);
    }

    /// <summary>Back to an empty sequence.</summary>
    public void Reset()
    {
        TokenCount = 0;
        _tailCount = 0;
        _recRows = 0;
    }

    /// <summary>Copies another cache of identical geometry (full copy: pooled blocks and tail).</summary>
    public void CopyFrom(Qwen4ExpIndexerCache other)
    {
        if (other.HeadDim != HeadDim || other.BlockSize != BlockSize) throw new ArgumentException("geometry mismatch.");
        int blocks = other.CompleteBlocks;
        if (blocks > 0)
        {
            _pooled.EnsureCapacity((long)blocks * HeadDim);
            other.Pooled.CopyTo(_pooled.Slice(0, blocks * HeadDim));
        }
        CopyTailFrom(other);
    }

    /// <summary>
    /// Copies only the token count and the tail block of <paramref name="other"/> (checkpoint / rollback within one
    /// sequence: the pooled blocks below the restored count are already in place and immutable).
    /// </summary>
    public void CopyTailFrom(Qwen4ExpIndexerCache other)
    {
        if (other.HeadDim != HeadDim || other.BlockSize != BlockSize) throw new ArgumentException("geometry mismatch.");
        other._tailRaw.Slice(0, BlockSize * HeadDim).CopyTo(_tailRaw.Slice(0, BlockSize * HeadDim));
        _tailCount = other._tailCount;
        TokenCount = other.TokenCount;
        _recRows = 0;
    }

    /// <summary>
    /// Appends <paramref name="count"/> raw (un-normed, un-rotated) indexer keys and pools every block they complete.
    /// </summary>
    /// <param name="rawKeys">Raw keys <c>[count, HeadDim]</c>.</param>
    /// <param name="count">Number of keys.</param>
    /// <param name="kNormGamma">Folded gamma of the indexer key norm <c>[HeadDim]</c>.</param>
    /// <param name="eps">Norm epsilon.</param>
    /// <param name="ropeCos">cos table <c>[position, ropeDim/2]</c> (see <see cref="RoPE.PrecomputeFrequencyTable"/> with head dim = <paramref name="ropeDim"/>).</param>
    /// <param name="ropeSin">sin table.</param>
    /// <param name="ropeDim">Rotated leading dims (64); the remaining <c>HeadDim - ropeDim</c> pass through.</param>
    [SkipLocalsInit]
    public void Append(ReadOnlySpan<float> rawKeys, int count, ReadOnlySpan<float> kNormGamma, float eps,
                       ReadOnlySpan<float> ropeCos, ReadOnlySpan<float> ropeSin, int ropeDim)
    {
        int d = HeadDim;
        if (rawKeys.Length < (long)count * d) throw new ArgumentException("rawKeys too small.", nameof(rawKeys));
        int half = ropeDim / 2;
        if (RecordRowCount > 0) RecordChunk(rawKeys, count);
        Span<float> mean = stackalloc float[d];
        for (int i = 0; i < count; i++)
        {
            rawKeys.Slice(i * d, d).CopyTo(_tailRaw.Slice((long)_tailCount * d, d));
            _tailCount++;
            TokenCount++;
            if (_tailCount < BlockSize) continue;

            // Complete block: mean (float accumulation, as HF's .float().mean), norm, rotate at the block's first position.
            int block = TokenCount / BlockSize - 1;
            mean.Clear();
            for (int j = 0; j < BlockSize; j++) TensorPrimitives.Add(mean, _tailRaw.Slice((long)j * d, d), mean);
            TensorPrimitives.Multiply(mean, 1.0f / BlockSize, mean);
            _pooled.EnsureCapacity((long)(block + 1) * d);
            Span<float> dst = _pooled.Slice((long)block * d, d);
            RmsNorm.Execute(mean, kNormGamma, eps, dst);
            int startPos = block * BlockSize;
            if (ropeDim > 0)
                RoPE.ApplyRotationNeoX(dst.Slice(0, ropeDim), ropeCos.Slice(startPos * half, half),
                                       ropeSin.Slice(startPos * half, half), ropeDim);
            _tailCount = 0;
        }
    }

    private void RecordChunk(ReadOnlySpan<float> rawKeys, int count)
    {
        int d = HeadDim;
        _recPreTokens = TokenCount;
        _recPreTailCount = _tailCount;
        _recPreTail.EnsureExact((long)BlockSize * d);
        _tailRaw.Slice(0, BlockSize * d).CopyTo(_recPreTail.Slice(0, BlockSize * d));
        _recChunk.EnsureExact(Math.Max(1L, (long)count * d));
        rawKeys.Slice(0, count * d).CopyTo(_recChunk.Slice(0, count * d));
        _recRows = Math.Min(RecordRowCount, Math.Max(count - 1, 0));
    }

    /// <summary>
    /// Sets the cache to exactly what it was right after row <paramref name="row"/> of the last recorded chunk: token count
    /// <c>pre + row + 1</c> and the tail rebuilt from the pre-chunk tail and the recorded raw keys. Pooled blocks are
    /// untouched (append-only), so the result is bit-identical to a forward of only rows <c>0..row</c>.
    /// </summary>
    /// <param name="row">Row index in <c>[0, RecordedRows)</c>.</param>
    public void RestoreToRow(int row)
    {
        if ((uint)row >= (uint)_recRows) throw new InvalidOperationException($"No indexer row snapshot for row {row} ({_recRows} recorded).");
        int d = HeadDim, newCount = _recPreTokens + row + 1, tail = newCount % BlockSize;
        int preTailStart = _recPreTokens - _recPreTailCount;
        // The new tail is assembled in the (separate) live tail buffer from the recorded copies, never from itself.
        for (int j = 0; j < tail; j++)
        {
            int q = newCount - tail + j;
            var src = q < _recPreTokens
                ? _recPreTail.Slice((long)(q - preTailStart) * d, d)
                : _recChunk.Slice((long)(q - _recPreTokens) * d, d);
            src.CopyTo(_tailRaw.Slice((long)j * d, d));
        }
        _tailCount = tail;
        TokenCount = newCount;
    }

    /// <summary>Drops recorded row snapshots (a later forward invalidates them).</summary>
    internal void InvalidateRows() => _recRows = 0;

    /// <inheritdoc/>
    public void Dispose()
    {
        _pooled.Dispose(); _tailRaw.Dispose(); _recPreTail.Dispose(); _recChunk.Dispose();
    }
}

/// <summary>
/// Per-sequence, per-layer state of a Qwen4-Exp QSA attention layer: the indexer key cache and, when no engine
/// <see cref="DotLLM.Core.Attention.IKvCache"/> carries the K/V rows, a native position-indexed K/V store.
/// </summary>
/// <remarks>
/// <para><see cref="Length"/> is the indexer's token count: the single source of truth for the layer's position, valid in both
/// modes. The own K/V store (<see cref="Keys"/>/<see cref="Values"/>) is allocated lazily on the first <see cref="AppendKv"/>,
/// so a state driven through an engine KV cache never pays for it. Position-indexed: rolling back is a length change.</para>
/// </remarks>
public sealed unsafe class Qwen4ExpQsaState : IDisposable
{
    private readonly Qwen4ExpNativeBuffer _k = new(), _v = new();

    /// <summary>Cached tokens (the indexer's token count).</summary>
    public int Length => Indexer.TokenCount;

    /// <summary>Indexer key cache of this layer.</summary>
    public Qwen4ExpIndexerCache Indexer { get; }

    /// <summary>KV row width <c>numKvHeads * headDim</c>.</summary>
    public int KvStride { get; }

    /// <summary>Creates an empty state.</summary>
    public Qwen4ExpQsaState(int kvStride, int indexerHeadDim, int blockSize)
    {
        KvStride = kvStride;
        Indexer = new Qwen4ExpIndexerCache(indexerHeadDim, blockSize);
    }

    /// <summary>Own-store keys <c>[Length, KvStride]</c> (empty when the K/V rows live in an engine KV cache).</summary>
    public unsafe ReadOnlySpan<float> Keys
        => _k.Pointer == null ? default : new ReadOnlySpan<float>(_k.Pointer, Length * KvStride);

    /// <summary>Own-store values <c>[Length, KvStride]</c>.</summary>
    public unsafe ReadOnlySpan<float> Values
        => _v.Pointer == null ? default : new ReadOnlySpan<float>(_v.Pointer, Length * KvStride);

    /// <summary>Resident bytes (own K/V store + indexer).</summary>
    public long Bytes => _k.Bytes + _v.Bytes + Indexer.Bytes;

    /// <summary>Back to an empty sequence.</summary>
    public void Reset() => Indexer.Reset();

    /// <summary>Copies another state of identical geometry (own K/V rows below <c>other.Length</c> and the indexer).</summary>
    public void CopyFrom(Qwen4ExpQsaState other)
    {
        long floats = (long)other.Length * KvStride;
        if (floats > 0 && other._k.Pointer != null)
        {
            _k.EnsureCapacity(floats); _v.EnsureCapacity(floats);
            other.Keys.CopyTo(_k.Slice(0, (int)floats));
            other.Values.CopyTo(_v.Slice(0, (int)floats));
        }
        Indexer.CopyFrom(other.Indexer);
    }

    /// <summary>
    /// Appends <paramref name="count"/> K/V rows to the own store at row <see cref="Length"/>. Call BEFORE
    /// <see cref="Qwen4ExpIndexerCache.Append"/> (which advances <see cref="Length"/>).
    /// </summary>
    public void AppendKv(ReadOnlySpan<float> k, ReadOnlySpan<float> v, int count)
    {
        long end = (long)(Length + count) * KvStride;
        _k.EnsureCapacity(end); _v.EnsureCapacity(end);
        k.Slice(0, count * KvStride).CopyTo(_k.Slice((long)Length * KvStride, count * KvStride));
        v.Slice(0, count * KvStride).CopyTo(_v.Slice((long)Length * KvStride, count * KvStride));
    }

    /// <inheritdoc/>
    public void Dispose() { _k.Dispose(); _v.Dispose(); Indexer.Dispose(); }
}

/// <summary>
/// Qwen4-Exp QSA (query-sparse attention) kernels: the block-pooled relu-sum indexer with top-k + incomplete tail, and
/// GQA over the selected keys. Verified against HF <c>Qwen4ExpTextQSAIndexer</c> (eager) and llama.cpp
/// <c>build_qsa_sel</c> / <c>build_attn_qsa</c>.
/// </summary>
/// <remarks>
/// <para>
/// For a query at absolute position <c>p</c> (<c>n = p+1</c> visible tokens) the visible tokens are grouped from the start
/// into blocks of <c>R</c> (4); <c>nb = floor(n / R)</c> blocks are complete and the remaining <c>n mod R</c> tokens are the
/// tail. Block score is <c>sum_h relu(q_h · k_block) / sqrt(D)</c>; the top <c>min(budget/R, nb)</c> blocks plus ALL tail
/// tokens are attended. Ties break toward the lower block index (HF's <c>torch.topk</c> tie order is unspecified).
/// </para>
/// <para>
/// <b>Dense equivalence.</b> When <c>nb &lt;= budget/R</c> every complete block is selected, so the keys attended are exactly
/// <c>0..p</c> and QSA == dense causal attention. That holds for <c>n &lt;= budget + R - 1</c> (2051 tokens on the released
/// model); the scoring/top-k is skipped in that regime without changing the result. Selected keys are visited in ascending
/// position order, so the fully-selected case is bitwise identical to the forced-dense path.
/// </para>
/// </remarks>
public static class Qwen4ExpQsa
{
    /// <summary>
    /// Scores every visible complete block for one query token and returns the selected block ids in ascending order.
    /// </summary>
    /// <param name="q">The token's indexer queries, normed and rotated, <c>[heads, headDim]</c>.</param>
    /// <param name="heads">Indexer query heads (4).</param>
    /// <param name="headDim">Indexer head width (128).</param>
    /// <param name="pooled">Pooled keys <c>[>= visibleBlocks, headDim]</c>.</param>
    /// <param name="visibleBlocks">Complete blocks visible to this query.</param>
    /// <param name="budgetBlocks">Blocks to keep (<c>indexer.top_k / compress_ratio</c>, 512).</param>
    /// <param name="selected">Output block ids, ascending; needs <c>min(budgetBlocks, visibleBlocks)</c> slots.</param>
    /// <param name="scores">Scratch array <c>[>= visibleBlocks]</c>; holds the block scores afterwards (unset when everything is selected).</param>
    /// <returns>Number of selected blocks.</returns>
    [SkipLocalsInit]
    public static int SelectBlocks(ReadOnlySpan<float> q, int heads, int headDim, ReadOnlySpan<float> pooled,
                                   int visibleBlocks, int budgetBlocks, Span<int> selected, float[] scores)
    {
        int take = Math.Min(budgetBlocks, visibleBlocks);
        if (take == visibleBlocks)
        {
            for (int b = 0; b < visibleBlocks; b++) selected[b] = b;   // everything selected: scoring cannot change the set
            return take;
        }

        float inv = 1.0f / MathF.Sqrt(headDim);
        for (int b = 0; b < visibleBlocks; b++)
        {
            ReadOnlySpan<float> kb = pooled.Slice(b * headDim, headDim);
            float s = 0;
            for (int h = 0; h < heads; h++)
            {
                float dot = TensorPrimitives.Dot(q.Slice(h * headDim, headDim), kb);
                if (dot > 0) s += dot;
            }
            scores[b] = s * inv;
        }

        int[] order = ArrayPool<int>.Shared.Rent(visibleBlocks);
        try
        {
            for (int b = 0; b < visibleBlocks; b++) order[b] = b;
            // Top `take` by (score desc, index asc). Full sort: oracle clarity over speed (visibleBlocks <= 65536).
            Array.Sort(order, 0, visibleBlocks, Comparer<int>.Create((a, b) =>
            {
                int c = scores[b].CompareTo(scores[a]);   // higher score first
                return c != 0 ? c : a.CompareTo(b);
            }));
            order.AsSpan(0, take).CopyTo(selected);
            selected.Slice(0, take).Sort();
        }
        finally
        {
            ArrayPool<int>.Shared.Return(order);
        }
        return take;
    }

    /// <summary>Scalar reference of <see cref="SelectBlocks"/> scoring (double accumulation) for tests.</summary>
    internal static void ScoreBlocksScalar(ReadOnlySpan<float> q, int heads, int headDim, ReadOnlySpan<float> pooled,
                                           int visibleBlocks, Span<float> scores)
    {
        for (int b = 0; b < visibleBlocks; b++)
        {
            double s = 0;
            for (int h = 0; h < heads; h++)
            {
                double dot = 0;
                for (int i = 0; i < headDim; i++) dot += (double)q[h * headDim + i] * pooled[b * headDim + i];
                if (dot > 0) s += dot;
            }
            scores[b] = (float)(s / Math.Sqrt(headDim));
        }
    }

    /// <summary>
    /// Fills <paramref name="keyIdx"/> with the attended key positions for a query at <paramref name="position"/>:
    /// selected blocks' tokens (ascending) then the tail. With <paramref name="selectedBlocks"/> empty and
    /// <paramref name="dense"/> true it is simply <c>0..position</c>.
    /// </summary>
    /// <returns>Number of keys.</returns>
    public static int BuildKeyList(int position, int blockSize, bool dense, ReadOnlySpan<int> selectedBlocks, Span<int> keyIdx)
    {
        int n = position + 1;
        if (dense)
        {
            for (int i = 0; i < n; i++) keyIdx[i] = i;
            return n;
        }
        int count = 0;
        foreach (int b in selectedBlocks)
            for (int j = 0; j < blockSize; j++) keyIdx[count++] = b * blockSize + j;
        for (int i = (n / blockSize) * blockSize; i < n; i++) keyIdx[count++] = i;
        return count;
    }

    /// <summary>
    /// GQA attention of one query token over an explicit key list: per head, softmax of scaled dot products then
    /// the weighted value sum. Head <c>h</c> reads KV head <c>h / (numHeads / numKvHeads)</c>.
    /// </summary>
    /// <param name="q">The token's queries <c>[numHeads, headDim]</c> (normed, rotated).</param>
    /// <param name="keys">Cached keys <c>[*, numKvHeads * headDim]</c>.</param>
    /// <param name="values">Cached values.</param>
    /// <param name="keyIdx">Attended positions.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Head width.</param>
    /// <param name="scale">Softmax scale (<c>1/sqrt(headDim)</c>).</param>
    /// <param name="output">Destination <c>[numHeads, headDim]</c>.</param>
    /// <param name="scores">Scratch <c>[keyIdx.Length]</c>.</param>
    [SkipLocalsInit]
    public static void AttendKeys(ReadOnlySpan<float> q, ReadOnlySpan<float> keys, ReadOnlySpan<float> values,
                                  ReadOnlySpan<int> keyIdx, int numHeads, int numKvHeads, int headDim, float scale,
                                  Span<float> output, Span<float> scores)
    {
        int group = numHeads / numKvHeads;
        int kvStride = numKvHeads * headDim;
        int n = keyIdx.Length;
        for (int h = 0; h < numHeads; h++)
        {
            int kvh = h / group;
            ReadOnlySpan<float> qh = q.Slice(h * headDim, headDim);
            float max = float.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                float s = TensorPrimitives.Dot(qh, keys.Slice(keyIdx[i] * kvStride + kvh * headDim, headDim)) * scale;
                scores[i] = s;
                if (s > max) max = s;
            }
            float sum = 0;
            for (int i = 0; i < n; i++)
            {
                float e = MathF.Exp(scores[i] - max);
                scores[i] = e;
                sum += e;
            }
            Span<float> o = output.Slice(h * headDim, headDim);
            o.Clear();
            for (int i = 0; i < n; i++)
                TensorPrimitives.MultiplyAdd(values.Slice(keyIdx[i] * kvStride + kvh * headDim, headDim),
                                             scores[i] / sum, o, o);
        }
    }
}

/// <summary>
/// One Qwen4-Exp QSA attention layer (CPU reference): fused <c>[q|gate]</c> projection, per-head QK RMSNorm, partial NeoX
/// RoPE, indexer (pooled-key cache + top-k + tail), GQA over the selected keys, <c>sigmoid(gate)</c>, output projection.
/// Projections are quant-aware callbacks so the model owns GEMM dispatch (F32 in tests).
/// </summary>
public sealed class Qwen4ExpQsaLayer
{
    private readonly int _hidden, _numHeads, _numKvHeads, _headDim, _ropeDim, _idxHeads, _idxDim, _blockSize, _budgetBlocks;
    private readonly float _eps;
    private readonly float[] _qNorm, _kNorm, _idxQNorm, _idxKNorm, _ropeCos, _ropeSin;
    private readonly Qwen4ExpProjection _qProj, _kProj, _vProj, _oProj, _idxQProj, _idxKProj;

    /// <summary>When true the indexer is bypassed and every query attends densely (diagnostic; differs from QSA beyond the budget).</summary>
    public bool ForceDense { get; set; }

    /// <summary>Token budget (<c>indexer.top_k</c>); contexts up to <c>budget + blockSize - 1</c> tokens are exactly dense.</summary>
    public int BudgetTokens => _budgetBlocks * _blockSize;

    /// <summary>Creates the layer.</summary>
    /// <param name="hidden">Hidden size (block input width).</param>
    /// <param name="numHeads">Query heads (24).</param>
    /// <param name="numKvHeads">KV heads (2).</param>
    /// <param name="headDim">Attention head width (256).</param>
    /// <param name="ropeDim">Rotated dims (64) for both the attention and the indexer.</param>
    /// <param name="indexerHeads">Indexer query heads (4).</param>
    /// <param name="indexerDim">Indexer head width (128).</param>
    /// <param name="blockSize">Pool block size (4).</param>
    /// <param name="budgetTokens">Token budget (2048).</param>
    /// <param name="eps">Norm epsilon.</param>
    /// <param name="qNorm">Folded per-head Q norm gamma <c>[headDim]</c>.</param>
    /// <param name="kNorm">Folded per-head K norm gamma.</param>
    /// <param name="indexerQNorm">Folded indexer query norm gamma <c>[indexerDim]</c>.</param>
    /// <param name="indexerKNorm">Folded indexer key norm gamma.</param>
    /// <param name="ropeCos">cos table <c>[position, ropeDim/2]</c>.</param>
    /// <param name="ropeSin">sin table.</param>
    /// <param name="qProj">Fused <c>[q|gate]</c> projection (<c>hidden -&gt; 2*numHeads*headDim</c>, interleaved per head).</param>
    /// <param name="kProj">K projection.</param>
    /// <param name="vProj">V projection.</param>
    /// <param name="oProj">Output projection (<c>numHeads*headDim -&gt; hidden</c>).</param>
    /// <param name="indexerQProj">Indexer query projection (<c>hidden -&gt; indexerHeads*indexerDim</c>).</param>
    /// <param name="indexerKProj">Indexer key projection (<c>hidden -&gt; indexerDim</c>).</param>
    public Qwen4ExpQsaLayer(int hidden, int numHeads, int numKvHeads, int headDim, int ropeDim,
                            int indexerHeads, int indexerDim, int blockSize, int budgetTokens, float eps,
                            float[] qNorm, float[] kNorm, float[] indexerQNorm, float[] indexerKNorm,
                            float[] ropeCos, float[] ropeSin,
                            Qwen4ExpProjection qProj, Qwen4ExpProjection kProj, Qwen4ExpProjection vProj,
                            Qwen4ExpProjection oProj, Qwen4ExpProjection indexerQProj, Qwen4ExpProjection indexerKProj)
    {
        if (budgetTokens % blockSize != 0) throw new ArgumentException("budget must be a multiple of the block size.");
        if (ropeDim > indexerDim || ropeDim > headDim) throw new ArgumentException("ropeDim must fit both heads.");
        _hidden = hidden; _numHeads = numHeads; _numKvHeads = numKvHeads; _headDim = headDim; _ropeDim = ropeDim;
        _idxHeads = indexerHeads; _idxDim = indexerDim; _blockSize = blockSize; _budgetBlocks = budgetTokens / blockSize;
        _eps = eps; _qNorm = qNorm; _kNorm = kNorm; _idxQNorm = indexerQNorm; _idxKNorm = indexerKNorm;
        _ropeCos = ropeCos; _ropeSin = ropeSin;
        _qProj = qProj; _kProj = kProj; _vProj = vProj; _oProj = oProj; _idxQProj = indexerQProj; _idxKProj = indexerKProj;
    }

    /// <summary>Allocates an empty per-sequence state for this layer.</summary>
    public Qwen4ExpQsaState CreateState() => new(_numKvHeads * _headDim, _idxDim, _blockSize);

    /// <summary>KV row width <c>numKvHeads * headDim</c> of this layer.</summary>
    public int KvStride => _numKvHeads * _headDim;

    /// <summary>
    /// Runs the layer on a chunk of <paramref name="tokens"/> block-input rows (positions continue from <c>state.Length</c>)
    /// and appends their K/V and indexer keys to <paramref name="state"/>.
    /// </summary>
    /// <param name="x">Block input <c>[tokens, hidden]</c>.</param>
    /// <param name="tokens">Chunk length.</param>
    /// <param name="state">Per-sequence state.</param>
    /// <param name="output">Destination <c>[tokens, hidden]</c>.</param>
    /// <param name="kvCache">Engine KV cache carrying this layer's K/V rows at <paramref name="kvSlot"/> (null: the state's own native store).
    /// Rows <c>state.Length .. state.Length + tokens - 1</c> are written; the cache must already hold at least <c>state.Length</c> rows.</param>
    /// <param name="kvSlot">Layer slot inside <paramref name="kvCache"/> (the QSA ordinal).</param>
    [SkipLocalsInit]
    public unsafe void Forward(ReadOnlySpan<float> x, int tokens, Qwen4ExpQsaState state, Span<float> output,
                               DotLLM.Core.Attention.IKvCache? kvCache = null, int kvSlot = 0)
    {
        int nH = _numHeads, nKv = _numKvHeads, d = _headDim;
        int qElems = nH * d, kvElems = nKv * d, idxQElems = _idxHeads * _idxDim;
        int first = state.Length;
        int half = _ropeDim / 2;
        if ((long)(first + tokens) * half > _ropeCos.Length)
            throw new ArgumentOutOfRangeException(nameof(tokens), "position exceeds the RoPE table.");

        float[] qg = ArrayPool<float>.Shared.Rent(tokens * 2 * qElems);
        float[] q = ArrayPool<float>.Shared.Rent(tokens * qElems);
        float[] gate = ArrayPool<float>.Shared.Rent(tokens * qElems);
        float[] k = ArrayPool<float>.Shared.Rent(tokens * kvElems);
        float[] v = ArrayPool<float>.Shared.Rent(tokens * kvElems);
        float[] iq = ArrayPool<float>.Shared.Rent(tokens * idxQElems);
        float[] ik = ArrayPool<float>.Shared.Rent(tokens * _idxDim);
        float[] attn = ArrayPool<float>.Shared.Rent(tokens * qElems);
        try
        {
            // ── projections ──
            _qProj(x, qg.AsSpan(0, tokens * 2 * qElems), tokens);
            for (int t = 0; t < tokens; t++)
            for (int h = 0; h < nH; h++)
            {
                qg.AsSpan(t * 2 * qElems + h * 2 * d, d).CopyTo(q.AsSpan(t * qElems + h * d, d));
                qg.AsSpan(t * 2 * qElems + h * 2 * d + d, d).CopyTo(gate.AsSpan(t * qElems + h * d, d));
            }
            _kProj(x, k.AsSpan(0, tokens * kvElems), tokens);
            _vProj(x, v.AsSpan(0, tokens * kvElems), tokens);
            _idxQProj(x, iq.AsSpan(0, tokens * idxQElems), tokens);
            _idxKProj(x, ik.AsSpan(0, tokens * _idxDim), tokens);

            // ── per-head QK norm, partial NeoX rope ──
            NormHeads(q, tokens, nH, d, _qNorm);
            NormHeads(k, tokens, nKv, d, _kNorm);
            NormHeads(iq, tokens, _idxHeads, _idxDim, _idxQNorm);
            for (int t = 0; t < tokens; t++)
            {
                int pos = first + t;
                var cos = _ropeCos.AsSpan(pos * half, half);
                var sin = _ropeSin.AsSpan(pos * half, half);
                for (int h = 0; h < nH; h++) RoPE.ApplyRotationNeoX(q.AsSpan(t * qElems + h * d, _ropeDim), cos, sin, _ropeDim);
                for (int h = 0; h < nKv; h++) RoPE.ApplyRotationNeoX(k.AsSpan(t * kvElems + h * d, _ropeDim), cos, sin, _ropeDim);
                for (int h = 0; h < _idxHeads; h++)
                    RoPE.ApplyRotationNeoX(iq.AsSpan(t * idxQElems + h * _idxDim, _ropeDim), cos, sin, _ropeDim);
            }

            // ── cache update (this chunk is visible to its own queries) ──
            nint kvKeysPtr = 0, kvValuesPtr = 0;
            if (kvCache is null)
                state.AppendKv(k, v, tokens);
            else
            {
                if (kvCache.CurrentLength < first)
                    throw new InvalidOperationException(
                        $"QSA KV cache holds {kvCache.CurrentLength} rows but the sequence state is at position {first}.");
                int[] positions = ArrayPool<int>.Shared.Rent(tokens);
                try
                {
                    for (int t = 0; t < tokens; t++) positions[t] = first + t;
                    fixed (float* kp = k) fixed (float* vp = v)
                    {
                        var kRef = new DotLLM.Core.Tensors.TensorRef(tokens, kvElems, DotLLM.Core.Tensors.DType.Float32, -1, (nint)kp);
                        var vRef = new DotLLM.Core.Tensors.TensorRef(tokens, kvElems, DotLLM.Core.Tensors.DType.Float32, -1, (nint)vp);
                        kvCache.Update(kRef, vRef, positions.AsSpan(0, tokens), kvSlot);
                    }
                }
                finally { ArrayPool<int>.Shared.Return(positions); }
                var kr = kvCache.GetKeysRef(kvSlot); var vr = kvCache.GetValuesRef(kvSlot);
                if (kr.DType != DotLLM.Core.Tensors.DType.Float32 || kr.Dim1 != kvElems || kr.Dim0 < first + tokens)
                    throw new NotSupportedException(
                        $"QSA needs a float32 KV cache slot of stride {kvElems} holding {first + tokens} rows; got {kr.DType} stride {kr.Dim1} rows {kr.Dim0}.");
                kvKeysPtr = kr.DataPointer; kvValuesPtr = vr.DataPointer;
            }
            state.Indexer.Append(ik, tokens, _idxKNorm, _eps, _ropeCos, _ropeSin, _ropeDim);

            // ── attention ──
            float scale = 1.0f / MathF.Sqrt(d);
            int maxKeys = first + tokens;
            int[] keyIdx = ArrayPool<int>.Shared.Rent(maxKeys);
            int[] sel = ArrayPool<int>.Shared.Rent(_budgetBlocks);
            float[] scores = ArrayPool<float>.Shared.Rent(maxKeys + 1);
            float[] blockScores = ArrayPool<float>.Shared.Rent(maxKeys / _blockSize + 1);
            try
            {
                ReadOnlySpan<float> keys, values;
                if (kvCache is null) { keys = state.Keys; values = state.Values; }
                else
                {
                    keys = new ReadOnlySpan<float>((void*)kvKeysPtr, (first + tokens) * kvElems);
                    values = new ReadOnlySpan<float>((void*)kvValuesPtr, (first + tokens) * kvElems);
                }
                var pooled = state.Indexer.Pooled;
                for (int t = 0; t < tokens; t++)
                {
                    int pos = first + t;
                    int nb = (pos + 1) / _blockSize;
                    bool dense = ForceDense || nb <= _budgetBlocks;   // nb <= budget: every complete block is selected == dense
                    int count;
                    if (dense)
                    {
                        count = Qwen4ExpQsa.BuildKeyList(pos, _blockSize, true, default, keyIdx);
                    }
                    else
                    {
                        int taken = Qwen4ExpQsa.SelectBlocks(iq.AsSpan(t * idxQElems, idxQElems), _idxHeads, _idxDim,
                                                             pooled, nb, _budgetBlocks, sel, blockScores);
                        count = Qwen4ExpQsa.BuildKeyList(pos, _blockSize, false, sel.AsSpan(0, taken), keyIdx);
                    }
                    Qwen4ExpQsa.AttendKeys(q.AsSpan(t * qElems, qElems), keys, values, keyIdx.AsSpan(0, count),
                                           nH, nKv, d, scale, attn.AsSpan(t * qElems, qElems), scores);
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(keyIdx); ArrayPool<int>.Shared.Return(sel);
                ArrayPool<float>.Shared.Return(scores); ArrayPool<float>.Shared.Return(blockScores);
            }

            // ── sigmoid gate, output projection ──
            Span<float> a = attn.AsSpan(0, tokens * qElems);
            Span<float> g = gate.AsSpan(0, tokens * qElems);
            TensorPrimitives.Sigmoid(g, g);
            TensorPrimitives.Multiply(a, g, a);
            _oProj(a, output, tokens);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(qg); ArrayPool<float>.Shared.Return(q); ArrayPool<float>.Shared.Return(gate);
            ArrayPool<float>.Shared.Return(k); ArrayPool<float>.Shared.Return(v); ArrayPool<float>.Shared.Return(iq);
            ArrayPool<float>.Shared.Return(ik); ArrayPool<float>.Shared.Return(attn);
        }
    }

    private void NormHeads(float[] data, int tokens, int heads, int dim, float[] gamma)
    {
        for (int i = 0; i < tokens * heads; i++)
            RmsNorm.Execute(data.AsSpan(i * dim, dim), gamma, _eps, data.AsSpan(i * dim, dim));
    }
}
