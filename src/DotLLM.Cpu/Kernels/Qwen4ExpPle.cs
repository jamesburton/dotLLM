using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Per-sequence state of one Qwen4-Exp PLE (n-gram hash embedding) module: the last <c>ngram-1</c> raw token ids (the
/// hash window — HF keeps these in <c>conv_states[2]</c>) and the last <c>(K-1)*dilation</c> rows of the normalised
/// gated value (the dilated-conv history, HF <c>conv_states[1]</c>). A fresh sequence has an EOS-filled window and a
/// zero conv history. Chunked prefill threads one instance through the chunks, so chunked == single-shot.
/// </summary>
public sealed class Qwen4ExpPleState : IDisposable
{
    private readonly Qwen4ExpNativeBuffer _conv = new();
    private readonly Qwen4ExpNativeBuffer _snapConv = new();
    private readonly int _eos, _histRows, _channels;
    private int[] _snapTokens = [];
    private int _snapRows;

    /// <summary>Last <c>ngram-1</c> raw token ids, oldest first.</summary>
    public int[] TokenHistory { get; }

    /// <summary>Conv history <c>[(K-1)*dilation, channels]</c>, oldest row first (native memory).</summary>
    public Span<float> ConvHistory => _conv.Slice(0, _histRows * _channels);

    /// <summary>Rows <see cref="Qwen4ExpPleBranch.Apply"/> should record per-row snapshots for on its next call (0 = off).</summary>
    internal int RecordRowCount { get; set; }

    /// <summary>Creates a fresh (start-of-sequence) state.</summary>
    /// <param name="ngramSize">N-gram order (window is <c>ngramSize - 1</c> tokens).</param>
    /// <param name="eosTokenId">EOS id the window is filled with.</param>
    /// <param name="convHistoryRows"><c>(kernel-1)*dilation</c>.</param>
    /// <param name="channels"><c>hcCount * hiddenSize</c>.</param>
    public Qwen4ExpPleState(int ngramSize, int eosTokenId, int convHistoryRows, int channels)
    {
        _eos = eosTokenId; _histRows = convHistoryRows; _channels = channels;
        TokenHistory = new int[ngramSize - 1];
        _conv.EnsureExact(Math.Max(1L, (long)convHistoryRows * channels));
        Reset();
    }

    /// <summary>Back to start-of-sequence.</summary>
    public void Reset()
    {
        Array.Fill(TokenHistory, _eos);
        ConvHistory.Clear();
        _snapRows = 0;
    }

    /// <summary>Copies another state (checkpoint / rollback).</summary>
    public void CopyFrom(Qwen4ExpPleState other)
    {
        other.TokenHistory.CopyTo(TokenHistory, 0);
        other.ConvHistory.CopyTo(ConvHistory);
        _snapRows = 0;
    }

    /// <summary>Resident bytes of this state (history plus any row-snapshot scratch).</summary>
    public long Bytes => TokenHistory.Length * 4L + _conv.Bytes + _snapConv.Bytes + _snapTokens.Length * 4L;

    /// <summary>Resident bytes of the sequence state proper (history only, no scratch).</summary>
    public long StateBytes => TokenHistory.Length * 4L + (long)_histRows * _channels * 4L;

    /// <summary>
    /// Records, for rows <c>0 .. rows-1</c> of a chunk, the hash window and conv history as they stand AFTER that row.
    /// Must be called before the chunk advances the state. The conv history after row <c>t</c> is rows
    /// <c>t+1 .. t+hist</c> of <c>[history ; normed]</c>.
    /// </summary>
    internal void RecordRows(ReadOnlySpan<int> tokens, ReadOnlySpan<float> normed, int rows)
    {
        int n1 = TokenHistory.Length, hist = _histRows, ch = _channels;
        _snapConv.EnsureExact(Math.Max(1L, (long)rows * hist * ch));
        if (_snapTokens.Length < rows * n1) _snapTokens = new int[rows * n1];
        var cur = ConvHistory;
        for (int t = 0; t < rows; t++)
        {
            var tokDst = _snapTokens.AsSpan(t * n1, n1);
            TokenHistory.CopyTo(tokDst);
            Qwen4ExpPle.AdvanceHistory(tokDst, tokens.Slice(0, t + 1));
            var dst = _snapConv.Slice((long)t * hist * ch, hist * ch);
            for (int r = 0; r < hist; r++)
            {
                int src = t + 1 + r;   // row index in [history ; normed]
                var from = src < hist ? cur.Slice(src * ch, ch) : normed.Slice((src - hist) * ch, ch);
                from.CopyTo(dst.Slice(r * ch, ch));
            }
        }
        _snapRows = rows;
    }

    /// <summary>Sets the state to what it was right after row <paramref name="row"/> of the last recorded chunk.</summary>
    internal void RestoreRow(int row)
    {
        if ((uint)row >= (uint)_snapRows) throw new InvalidOperationException($"No PLE row snapshot for row {row} ({_snapRows} recorded).");
        int n1 = TokenHistory.Length;
        _snapTokens.AsSpan(row * n1, n1).CopyTo(TokenHistory);
        _snapConv.Slice((long)row * _histRows * _channels, _histRows * _channels).CopyTo(ConvHistory);
    }

    /// <summary>Drops recorded row snapshots (a later forward invalidates them).</summary>
    internal void InvalidateRows() => _snapRows = 0;

    /// <inheritdoc/>
    public void Dispose() { _conv.Dispose(); _snapConv.Dispose(); }
}

/// <summary>
/// Qwen4-Exp (Qwen3.8-Flash-Next) PLE n-gram hash-embedding branch kernels. Verified against HF
/// <c>Qwen4ExpTextNGramEmbedding</c> / <c>Qwen4ExpTextPLELayer</c> and llama.cpp <c>qwen4exp.cpp</c>
/// (<c>llm_graph_input_qwen4exp_ple::set_input</c>, <c>build_ple</c>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Hash.</b> For position <c>p</c> with context <c>c[0]=t[p], c[1]=t[p-1], c[2]=t[p-2]</c> (an EOS in the window
/// resets everything at or before it; a missing predecessor reads as EOS; a token's own EOS does not cut its own
/// context): <c>mixed_n = (c[0]*m[0]) XOR (c[1]*m[1]) [XOR (c[2]*m[2])]</c> in wrapping int64, and
/// <c>row_h = floorMod(mixed_n, vocab[h]) + offset[h]</c>; n = 2 feeds heads <c>[0, heads)</c>, n = 3 the next block.
/// All arithmetic is exact <see cref="long"/> (the multipliers are ~45-bit constants — never pass them through
/// <see cref="double"/>). HF uses signed floor-mod; for in-vocabulary token ids the products never exceed
/// <c>2^63-1</c> by construction of the multipliers, so unsigned (llama.cpp) and signed (HF) agree — this code follows HF.
/// </para>
/// <para>
/// <b>Branch.</b> <c>emb = rows.flatten</c> (head-major <c>[numHeads * rowDim]</c>); <c>key = groupNorm(Wk emb)</c>,
/// <c>value = Wv emb</c>, <c>query = groupNorm(R)</c>; <c>g = sum_i key*query / sqrt(H)</c> per stream,
/// <c>gate = sigmoid(sign(g) * sqrt(max(|g|, 1e-6)))</c>; <c>gated = gate * value</c> (value broadcast over streams);
/// <c>out = gated + silu(dilatedConv(groupNorm(gated)))</c>; <c>R += out</c>.
/// </para>
/// </remarks>
public static unsafe class Qwen4ExpPle
{
    /// <summary>Signed floor modulo (Python / <c>torch.remainder</c> semantics), non-negative for a positive divisor.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long FloorMod(long a, long m)
    {
        long r = a % m;
        return (r != 0 && ((r ^ m) < 0)) ? r + m : r;
    }

    /// <summary>
    /// Builds the table row index of every (token, head): exact int64 hash with EOS-reset window semantics.
    /// </summary>
    /// <param name="tokens">Raw token ids of this chunk.</param>
    /// <param name="history">Window carried from the previous chunk (<c>ngramSize-1</c> ids, oldest first; EOS-filled at sequence start).</param>
    /// <param name="ngramSize">N-gram order (3).</param>
    /// <param name="headsPerNgram">Heads per order (8).</param>
    /// <param name="eosTokenId">EOS id that cuts the window.</param>
    /// <param name="multipliers">Per-position hash multipliers (at least <paramref name="ngramSize"/>), exact int64.</param>
    /// <param name="headOffsets">First table row of every head (<c>(ngramSize-1)*headsPerNgram</c> entries).</param>
    /// <param name="headVocabSizes">Hash modulus of every head.</param>
    /// <param name="rows">Destination <c>[tokens.Length * numHeads]</c>.</param>
    [SkipLocalsInit]
    public static void BuildRowIndices(ReadOnlySpan<int> tokens, ReadOnlySpan<int> history,
                                       int ngramSize, int headsPerNgram, int eosTokenId,
                                       ReadOnlySpan<long> multipliers, ReadOnlySpan<long> headOffsets,
                                       ReadOnlySpan<long> headVocabSizes, Span<long> rows)
    {
        int nPrev = ngramSize - 1;
        int numHeads = nPrev * headsPerNgram;
        if (history.Length < nPrev) throw new ArgumentException("history must hold ngramSize-1 ids.", nameof(history));
        if (multipliers.Length < ngramSize) throw new ArgumentException("multipliers too short.", nameof(multipliers));
        if (headOffsets.Length < numHeads || headVocabSizes.Length < numHeads)
            throw new ArgumentException("headOffsets / headVocabSizes too short.");
        if (rows.Length < (long)tokens.Length * numHeads) throw new ArgumentException("rows too small.", nameof(rows));

        Span<long> ctx = stackalloc long[ngramSize];
        for (int i = 0; i < tokens.Length; i++)
        {
            ctx[0] = tokens[i];
            bool cut = false;
            for (int s = 1; s < ngramSize; s++)
            {
                int src = i - s;
                int t = src >= 0 ? tokens[src] : history[nPrev + src];
                cut |= t == eosTokenId;
                ctx[s] = cut ? eosTokenId : t;
            }

            for (int n = 2; n <= ngramSize; n++)
            {
                long mixed = unchecked(ctx[0] * multipliers[0]);
                for (int j = 1; j < n; j++)
                    mixed ^= unchecked(ctx[j] * multipliers[j]);
                int baseHead = (n - 2) * headsPerNgram;
                for (int g = 0; g < headsPerNgram; g++)
                {
                    int h = baseHead + g;
                    rows[i * numHeads + h] = FloorMod(mixed, headVocabSizes[h]) + headOffsets[h];
                }
            }
        }
    }

    /// <summary>Slides the token window forward over this chunk (last <c>ngramSize-1</c> of <c>history ++ tokens</c>).</summary>
    public static void AdvanceHistory(Span<int> history, ReadOnlySpan<int> tokens)
    {
        int n = history.Length;
        if (tokens.Length >= n)
        {
            tokens.Slice(tokens.Length - n).CopyTo(history);
            return;
        }
        history.Slice(tokens.Length).CopyTo(history);
        tokens.CopyTo(history.Slice(n - tokens.Length));
    }

    /// <summary>
    /// Gathers and dequantises table rows into F32 (lazy-mmap friendly: touches only the requested rows, never copies the table).
    /// </summary>
    /// <param name="table">Base pointer of the table (row-major, one row = <paramref name="rowDim"/> elements).</param>
    /// <param name="quantType">Table storage type (IQ4_NL, BF16, F16, F32, Q8_0, ...).</param>
    /// <param name="tableRows">Rows in the table (bounds check).</param>
    /// <param name="rowDim">Elements per row (160 on the released model).</param>
    /// <param name="rows">Row indices.</param>
    /// <param name="dest">Destination <c>[rows.Length * rowDim]</c>.</param>
    public static void GatherRows(nint table, QuantizationType quantType, long tableRows, int rowDim,
                                  ReadOnlySpan<long> rows, Span<float> dest)
    {
        if (dest.Length < (long)rows.Length * rowDim) throw new ArgumentException("dest too small.", nameof(dest));
        long rowBytes = Dequantize.RowByteSize(rowDim, quantType);
        for (int i = 0; i < rows.Length; i++)
        {
            long r = rows[i];
            if ((ulong)r >= (ulong)tableRows)
                throw new ArgumentOutOfRangeException(nameof(rows), $"PLE row {r} outside table of {tableRows} rows.");
            Dequantize.ToFloat32(table + (nint)(r * rowBytes), rowDim, quantType, dest.Slice(i * rowDim, rowDim));
        }
    }

    /// <summary>
    /// Pre-sigmoid-free gate: <c>gate[t,s] = sigmoid(sign(g) * sqrt(max(|g|, 1e-6)))</c> with
    /// <c>g = sum_i key[t,s,i] * query[t,s,i] / sqrt(hiddenSize)</c>.
    /// </summary>
    /// <param name="keyNormed">Normed key <c>[seqLen, hcCount*hiddenSize]</c>.</param>
    /// <param name="queryNormed">Normed query (of the residual) <c>[seqLen, hcCount*hiddenSize]</c>.</param>
    /// <param name="hcCount">Streams.</param>
    /// <param name="hiddenSize">Channels per stream.</param>
    /// <param name="gate">Destination <c>[seqLen, hcCount]</c>.</param>
    /// <param name="seqLen">Tokens.</param>
    [SkipLocalsInit]
    public static void ComputeGate(ReadOnlySpan<float> keyNormed, ReadOnlySpan<float> queryNormed,
                                   int hcCount, int hiddenSize, Span<float> gate, int seqLen)
    {
        float inv = 1.0f / MathF.Sqrt(hiddenSize);
        for (int t = 0; t < seqLen; t++)
        for (int s = 0; s < hcCount; s++)
        {
            int off = (t * hcCount + s) * hiddenSize;
            float g = TensorPrimitives.Dot(keyNormed.Slice(off, hiddenSize), queryNormed.Slice(off, hiddenSize)) * inv;
            gate[t * hcCount + s] = SignedSqrtSigmoid(g);
        }
    }

    /// <summary>Scalar reference for <see cref="ComputeGate"/> (double-accumulated dot).</summary>
    internal static void ComputeGateScalar(ReadOnlySpan<float> keyNormed, ReadOnlySpan<float> queryNormed,
                                           int hcCount, int hiddenSize, Span<float> gate, int seqLen)
    {
        for (int t = 0; t < seqLen; t++)
        for (int s = 0; s < hcCount; s++)
        {
            int off = (t * hcCount + s) * hiddenSize;
            double d = 0;
            for (int i = 0; i < hiddenSize; i++) d += (double)keyNormed[off + i] * queryNormed[off + i];
            gate[t * hcCount + s] = SignedSqrtSigmoid((float)(d / Math.Sqrt(hiddenSize)));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SignedSqrtSigmoid(float g)
    {
        float m = MathF.Sqrt(MathF.Max(MathF.Abs(g), 1e-6f));
        float signed = g > 0 ? m : g < 0 ? -m : 0f;   // torch.sign(0) == 0 -> sigmoid(0) = 0.5
        return 1f / (1f + MathF.Exp(-signed));
    }

    /// <summary>
    /// <c>gated[t,s,:] = gate[t,s] * value[t,:]</c> — the shared value broadcast over streams, scaled by each stream's gate.
    /// </summary>
    /// <param name="gate"><c>[seqLen, hcCount]</c>.</param>
    /// <param name="value"><c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="hcCount">Streams.</param>
    /// <param name="hiddenSize">Channels per stream.</param>
    /// <param name="gated">Destination <c>[seqLen, hcCount*hiddenSize]</c>.</param>
    /// <param name="seqLen">Tokens.</param>
    public static void ApplyGate(ReadOnlySpan<float> gate, ReadOnlySpan<float> value,
                                 int hcCount, int hiddenSize, Span<float> gated, int seqLen)
    {
        for (int t = 0; t < seqLen; t++)
        for (int s = 0; s < hcCount; s++)
            TensorPrimitives.Multiply(value.Slice(t * hiddenSize, hiddenSize), gate[t * hcCount + s],
                                      gated.Slice((t * hcCount + s) * hiddenSize, hiddenSize));
    }

    /// <summary>
    /// Transposes the depthwise conv weight from the checkpoint layout <c>[channels, kernel]</c> (HF
    /// <c>conv1d.weight[C,1,K]</c> flattened; GGUF <c>ple_conv1d [K, C]</c> with K fastest) to tap-major <c>[kernel, channels]</c>.
    /// </summary>
    public static float[] TransposeConvWeight(ReadOnlySpan<float> channelMajor, int channels, int kernel)
    {
        var tap = new float[kernel * channels];
        for (int c = 0; c < channels; c++)
        for (int k = 0; k < kernel; k++)
            tap[k * channels + c] = channelMajor[c * kernel + k];
        return tap;
    }

    /// <summary>
    /// Dilated depthwise causal conv + SiLU with carried history:
    /// <c>y[t,c] = silu( sum_k w[k,c] * x[t - (K-1-k)*dilation, c] )</c>, positions before the chunk read the history
    /// (zeros at sequence start). Updates <paramref name="history"/> to the last <c>(K-1)*dilation</c> rows of input.
    /// </summary>
    /// <param name="x">Input <c>[seqLen, channels]</c>.</param>
    /// <param name="history">History <c>[(K-1)*dilation, channels]</c>, oldest first; updated in place.</param>
    /// <param name="weightTapMajor">Tap-major weight <c>[kernel, channels]</c> (see <see cref="TransposeConvWeight"/>).</param>
    /// <param name="kernel">Taps (4).</param>
    /// <param name="dilation">Dilation (ngram size, 3).</param>
    /// <param name="channels">Channels (hcCount*hiddenSize).</param>
    /// <param name="y">Destination <c>[seqLen, channels]</c>; may NOT alias <paramref name="x"/>.</param>
    /// <param name="seqLen">Tokens.</param>
    [SkipLocalsInit]
    public static void DilatedConvSilu(ReadOnlySpan<float> x, Span<float> history, ReadOnlySpan<float> weightTapMajor,
                                       int kernel, int dilation, int channels, Span<float> y, int seqLen)
    {
        int hist = (kernel - 1) * dilation;
        if (history.Length < (long)hist * channels) throw new ArgumentException("history too small.", nameof(history));
        if (weightTapMajor.Length < kernel * channels) throw new ArgumentException("weight too small.", nameof(weightTapMajor));
        if (x.Length < (long)seqLen * channels || y.Length < (long)seqLen * channels)
            throw new ArgumentException("x / y too small.");

        int rowsTotal = hist + seqLen;
        float[] comb = ArrayPool<float>.Shared.Rent(rowsTotal * channels);
        try
        {
            history.Slice(0, hist * channels).CopyTo(comb);
            x.Slice(0, seqLen * channels).CopyTo(comb.AsSpan(hist * channels));
            for (int t = 0; t < seqLen; t++)
            {
                Span<float> row = y.Slice(t * channels, channels);
                TensorPrimitives.Multiply(comb.AsSpan((t + 0 * dilation) * channels, channels),
                                          weightTapMajor.Slice(0, channels), row);
                for (int k = 1; k < kernel; k++)
                    TensorPrimitives.FusedMultiplyAdd(comb.AsSpan((t + k * dilation) * channels, channels),
                                                      weightTapMajor.Slice(k * channels, channels), row, row);
            }
            SiLu.Execute(y.Slice(0, seqLen * channels), y.Slice(0, seqLen * channels));
            // new history = last `hist` rows of comb
            comb.AsSpan(seqLen * channels, hist * channels).CopyTo(history);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(comb);
        }
    }

    /// <summary>Scalar reference for <see cref="DilatedConvSilu"/> (double accumulation; does not touch <paramref name="history"/>).</summary>
    internal static void DilatedConvSiluScalar(ReadOnlySpan<float> x, ReadOnlySpan<float> history,
                                               ReadOnlySpan<float> weightTapMajor, int kernel, int dilation,
                                               int channels, Span<float> y, int seqLen)
    {
        int hist = (kernel - 1) * dilation;
        for (int t = 0; t < seqLen; t++)
        for (int c = 0; c < channels; c++)
        {
            double acc = 0;
            for (int k = 0; k < kernel; k++)
            {
                int pos = t - (kernel - 1 - k) * dilation;   // relative to chunk start
                float v = pos >= 0 ? x[pos * channels + c] : history[(hist + pos) * channels + c];
                acc += (double)weightTapMajor[k * channels + c] * v;
            }
            y[t * channels + c] = (float)(acc / (1.0 + Math.Exp(-acc)));
        }
    }
}

/// <summary>Quant-aware linear projection callback: <c>output[t, :] = W · input[t, :]</c> for <c>tokens</c> rows.</summary>
/// <param name="input">Input <c>[tokens, inDim]</c>.</param>
/// <param name="output">Output <c>[tokens, outDim]</c>.</param>
/// <param name="tokens">Row count.</param>
public delegate void Qwen4ExpProjection(ReadOnlySpan<float> input, Span<float> output, int tokens);

/// <summary>
/// One Qwen4-Exp PLE module (the n-gram branch on layer index 1 of the released model): static hash parameters, the lazy
/// table view, norms/conv weights and the two projections. <see cref="Apply"/> adds the branch output to the residual.
/// </summary>
public sealed class Qwen4ExpPleBranch
{
    private readonly nint _table;
    private readonly QuantizationType _tableQt;
    private readonly long _tableRows;
    private readonly int _rowDim;
    private readonly int _ngram, _headsPerNgram, _eos, _convKernel;
    private readonly long[] _multipliers, _offsets, _vocab;
    private readonly float[] _normKey, _normQuery, _normConv, _convTap;
    private readonly Qwen4ExpProjection _keyProj, _valueProj;
    private readonly int _hc, _hidden;
    private readonly float _eps;

    /// <summary>Hash window / conv history geometry for allocating a <see cref="Qwen4ExpPleState"/>.</summary>
    public int ConvHistoryRows => (_convKernel - 1) * _ngram;

    /// <summary>Number of hash heads, <c>(ngram-1) * headsPerNgram</c>.</summary>
    public int NumHeads => (_ngram - 1) * _headsPerNgram;

    /// <summary>Creates the branch. The table pointer is borrowed (typically a lazy mmap view; never copied).</summary>
    /// <param name="table">Pointer to the hash table (row-major).</param>
    /// <param name="tableQt">Table storage type.</param>
    /// <param name="tableRows">Table row count.</param>
    /// <param name="rowDim">Elements per table row.</param>
    /// <param name="ngramSize">N-gram order.</param>
    /// <param name="headsPerNgram">Heads per order.</param>
    /// <param name="eosTokenId">EOS id.</param>
    /// <param name="convKernel">Conv taps.</param>
    /// <param name="multipliers">Exact int64 multipliers.</param>
    /// <param name="headOffsets">Head row offsets.</param>
    /// <param name="headVocabSizes">Head moduli.</param>
    /// <param name="hcCount">Residual streams.</param>
    /// <param name="hiddenSize">Channels per stream.</param>
    /// <param name="eps">Norm epsilon.</param>
    /// <param name="normKey">Folded gamma <c>[hc*hidden]</c> for the key norm.</param>
    /// <param name="normQuery">Folded gamma for the query norm.</param>
    /// <param name="normConv">Folded gamma for the conv-input norm.</param>
    /// <param name="convWeightTapMajor">Conv weight <c>[kernel, hc*hidden]</c> (see <see cref="Qwen4ExpPle.TransposeConvWeight"/>).</param>
    /// <param name="keyProj">Key projection (<c>numHeads*rowDim -&gt; hc*hidden</c>).</param>
    /// <param name="valueProj">Value projection (<c>numHeads*rowDim -&gt; hidden</c>).</param>
    public Qwen4ExpPleBranch(nint table, QuantizationType tableQt, long tableRows, int rowDim,
                             int ngramSize, int headsPerNgram, int eosTokenId, int convKernel,
                             long[] multipliers, long[] headOffsets, long[] headVocabSizes,
                             int hcCount, int hiddenSize, float eps,
                             float[] normKey, float[] normQuery, float[] normConv, float[] convWeightTapMajor,
                             Qwen4ExpProjection keyProj, Qwen4ExpProjection valueProj)
    {
        _table = table; _tableQt = tableQt; _tableRows = tableRows; _rowDim = rowDim;
        _ngram = ngramSize; _headsPerNgram = headsPerNgram; _eos = eosTokenId; _convKernel = convKernel;
        _multipliers = multipliers; _offsets = headOffsets; _vocab = headVocabSizes;
        _hc = hcCount; _hidden = hiddenSize; _eps = eps;
        _normKey = normKey; _normQuery = normQuery; _normConv = normConv; _convTap = convWeightTapMajor;
        _keyProj = keyProj; _valueProj = valueProj;
        long minRows = 0;
        for (int h = 0; h < NumHeads; h++) minRows = Math.Max(minRows, headOffsets[h] + headVocabSizes[h]);
        if (tableRows < minRows)
            throw new ArgumentException($"PLE table has {tableRows} rows, head ranges need {minRows}.");
    }

    /// <summary>Allocates a fresh start-of-sequence state for this module.</summary>
    public Qwen4ExpPleState CreateState() => new(_ngram, _eos, ConvHistoryRows, _hc * _hidden);

    /// <summary>
    /// Adds the PLE branch output to <paramref name="residual"/> (<c>R += gated + silu(conv(norm(gated)))</c>) and advances
    /// <paramref name="state"/> past <paramref name="tokens"/>.
    /// </summary>
    /// <param name="tokens">Raw token ids of this chunk (the ids the main embedding used).</param>
    /// <param name="state">Per-sequence state.</param>
    /// <param name="residual">Residual <c>[tokens, hc*hidden]</c>, read (query) then updated.</param>
    public void Apply(ReadOnlySpan<int> tokens, Qwen4ExpPleState state, Span<float> residual)
    {
        int T = tokens.Length, row = _hc * _hidden, numHeads = NumHeads;
        if (residual.Length < (long)T * row) throw new ArgumentException("residual too small.", nameof(residual));

        long[] rows = ArrayPool<long>.Shared.Rent(T * numHeads);
        float[] emb = ArrayPool<float>.Shared.Rent(T * numHeads * _rowDim);
        float[] key = ArrayPool<float>.Shared.Rent(T * row);
        float[] query = ArrayPool<float>.Shared.Rent(T * row);
        float[] value = ArrayPool<float>.Shared.Rent(T * _hidden);
        float[] gate = ArrayPool<float>.Shared.Rent(T * _hc);
        float[] gated = ArrayPool<float>.Shared.Rent(T * row);
        float[] normed = ArrayPool<float>.Shared.Rent(T * row);
        float[] conv = ArrayPool<float>.Shared.Rent(T * row);
        try
        {
            Qwen4ExpPle.BuildRowIndices(tokens, state.TokenHistory, _ngram, _headsPerNgram, _eos,
                                        _multipliers, _offsets, _vocab, rows);
            Qwen4ExpPle.GatherRows(_table, _tableQt, _tableRows, _rowDim, rows.AsSpan(0, T * numHeads), emb);

            var embSpan = emb.AsSpan(0, T * numHeads * _rowDim);
            _keyProj(embSpan, key.AsSpan(0, T * row), T);
            _valueProj(embSpan, value.AsSpan(0, T * _hidden), T);
            Qwen4ExpGatedResidual.GroupRmsNorm(key, _normKey, _hc, _hidden, _eps, key, T);
            Qwen4ExpGatedResidual.GroupRmsNorm(residual, _normQuery, _hc, _hidden, _eps, query, T);

            Qwen4ExpPle.ComputeGate(key, query, _hc, _hidden, gate, T);
            Qwen4ExpPle.ApplyGate(gate, value, _hc, _hidden, gated, T);
            Qwen4ExpGatedResidual.GroupRmsNorm(gated, _normConv, _hc, _hidden, _eps, normed, T);
            if (state.RecordRowCount > 0)
                state.RecordRows(tokens, normed.AsSpan(0, T * row), Math.Min(state.RecordRowCount, T - 1));
            Qwen4ExpPle.DilatedConvSilu(normed.AsSpan(0, T * row), state.ConvHistory, _convTap, _convKernel, _ngram,
                                        row, conv.AsSpan(0, T * row), T);

            Span<float> res = residual.Slice(0, T * row);
            TensorPrimitives.Add(res, gated.AsSpan(0, T * row), res);
            TensorPrimitives.Add(res, conv.AsSpan(0, T * row), res);

            Qwen4ExpPle.AdvanceHistory(state.TokenHistory, tokens);
        }
        finally
        {
            ArrayPool<long>.Shared.Return(rows);
            ArrayPool<float>.Shared.Return(emb); ArrayPool<float>.Shared.Return(key); ArrayPool<float>.Shared.Return(query);
            ArrayPool<float>.Shared.Return(value); ArrayPool<float>.Shared.Return(gate); ArrayPool<float>.Shared.Return(gated);
            ArrayPool<float>.Shared.Return(normed); ArrayPool<float>.Shared.Return(conv);
        }
    }
}
