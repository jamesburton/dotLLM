using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>Storage type of the carried short-convolution inputs (the recurrent state of Inkling 2).</summary>
public enum Inkling2ConvStateStorage
{
    /// <summary>IEEE single precision (llama.cpp's recurrent-state type; the A/B reference arm only).</summary>
    F32,

    /// <summary>bfloat16 (the default; user ruling 2026-11-26: F16 is ruled out, its 5-bit exponent cannot hold the range, KL 5.65e-3 vs 3.91e-3).</summary>
    BF16,
}

/// <summary>
/// One causal short convolution with carried state (llama.cpp <c>graph::convolve</c> + <c>ggml_ssm_conv</c>). The state keeps the last
/// <c>K - 1</c> inputs of every channel, oldest first, in <see cref="Inkling2ConvStateStorage"/>.
/// </summary>
internal sealed class Inkling2ConvState : IDisposable
{
    private readonly int _channels, _k1;
    private readonly Inkling2ConvStateStorage _storage;
    private readonly NativeBuffer<float>? _f32;
    private readonly NativeBuffer<ushort>? _u16;

    public Inkling2ConvState(int channels, int kernel, Inkling2ConvStateStorage storage)
    {
        _channels = channels; _k1 = kernel - 1; _storage = storage;
        if (storage == Inkling2ConvStateStorage.F32) _f32 = new NativeBuffer<float>((long)channels * _k1);
        else _u16 = new NativeBuffer<ushort>((long)channels * _k1);
    }

    public long Bytes => _f32?.Bytes ?? _u16!.Bytes;

    /// <summary>Test switch (mutant): false restores the pre-#2039 behaviour where in-batch history is read unrounded.</summary>
    internal static bool RoundInBatchHistory { get; set; } = true;

    public void Reset() { _f32?.Clear(); _u16?.Clear(); }

    private float Load(int i) => _storage switch
    {
        Inkling2ConvStateStorage.F32 => _f32!.Span[i],
        _ => BitConverter.Int32BitsToSingle(_u16!.Span[i] << 16),
    };

    private void Store(int i, float v)
    {
        switch (_storage)
        {
            case Inkling2ConvStateStorage.F32: _f32!.Span[i] = v; break;
            default: _u16!.Span[i] = ToBf16(v); break;
        }
    }

    /// <summary>Round-to-nearest-even float to bfloat16 bits (NaN stays NaN).</summary>
    internal static ushort ToBf16(float v)
    {
        uint b = BitConverter.SingleToUInt32Bits(v);
        if ((b & 0x7FFFFFFFu) > 0x7F800000u) return (ushort)((b >> 16) | 0x40);
        return (ushort)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
    }

    /// <summary>
    /// Runs the convolution over <paramref name="tokens"/> consecutive inputs and advances the state. Per channel <c>c</c> and token <c>t</c>,
    /// with <c>x</c> the window <c>[state; input]</c>: <c>acc = sum_i x[t + i] * w[c][i]</c> (tap 0 is the OLDEST input), plus, when
    /// <paramref name="dyn"/> is given, <c>sum_lag x[t + K - 1 - lag] * dyn[t][head(c)][lag]</c> (lag 0 is the current input); the output is
    /// <c>(x_t + acc) / sqrt((dyn ? 2 : 1) * K + 1)</c>.
    /// </summary>
    /// <param name="tokens">Number of tokens.</param>
    /// <param name="input">Inputs <c>[tokens][channels]</c>.</param>
    /// <param name="weight">Static taps <c>[channels][K]</c>.</param>
    /// <param name="dyn">Input-dependent taps <c>[tokens][heads][K]</c>, or empty.</param>
    /// <param name="heads">Head count of <paramref name="dyn"/> (channels are split evenly across heads).</param>
    /// <param name="output">Outputs <c>[tokens][channels]</c>; may alias <paramref name="input"/>.</param>
    /// <param name="scratch">At least <see cref="ScratchFloats"/> floats of caller-owned scratch (native; no allocation here).</param>
    public void Run(int tokens, ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> dyn, int heads, Span<float> output, Span<float> scratch)
    {
        int C = _channels, K1 = _k1, K = K1 + 1;
        bool hasDyn = !dyn.IsEmpty;
        int rows = K1 + tokens;
        if (scratch.Length < ScratchFloats(tokens)) throw new ArgumentException("scratch too small.", nameof(scratch));
        Span<float> win = scratch.Slice(0, rows * C);
        Span<float> res = scratch.Slice(rows * C, tokens * C);
        {
            for (int k = 0; k < K1; k++)
                for (int c = 0; c < C; c++) win[k * C + c] = Load(c * K1 + k);
            input.Slice(0, tokens * C).CopyTo(win.Slice(K1 * C, tokens * C));
            // Sequential decode reads every EARLIER input back out of the (BF16-rounded) state; only the current input is exact. Round the in-batch
            // history the same way so an N-row forward equals N single-row forwards bit for bit (speculative verify, #2039).
            if (_storage == Inkling2ConvStateStorage.BF16 && RoundInBatchHistory)
                for (int i = K1 * C; i < win.Length; i++) win[i] = BitConverter.Int32BitsToSingle(ToBf16(win[i]) << 16);

            if (RecordRows) RecordWindow(win, rows);

            int perHead = hasDyn ? C / heads : 1;
            float scale = 1f / MathF.Sqrt((hasDyn ? 2 : 1) * K + 1);
            for (int t = 0; t < tokens; t++)
            {
                for (int c = 0; c < C; c++)
                {
                    float acc = 0f;
                    var w = weight.Slice(c * K, K);
                    for (int i = 0; i < K; i++) acc += (i == K1 ? input[t * C + c] : win[(t + i) * C + c]) * w[i];
                    if (hasDyn)
                    {
                        var d = dyn.Slice(t * heads * K + (c / perHead) * K, K);
                        for (int lag = 0; lag < K; lag++) acc += (lag == 0 ? input[t * C + c] : win[(t + K - 1 - lag) * C + c]) * d[lag];
                    }
                    res[t * C + c] = (input[t * C + c] + acc) * scale;
                }
            }

            for (int k = 0; k < K1; k++)
                for (int c = 0; c < C; c++) Store(c * K1 + k, win[(tokens + k) * C + c]);
            res.Slice(0, tokens * C).CopyTo(output);
        }
    }

    private NativeBuffer<ushort>? _snapU;
    private NativeBuffer<float>? _snapF;
    private int _snapRows;

    /// <summary>When true, every <see cref="Run"/> keeps its window (old state + the inputs) so <see cref="RestoreToRow"/> can rewind to any row of that call (speculative verify, #2039).</summary>
    public bool RecordRows { get; set; }

    /// <summary>Bytes held by the recorded window (BF16 for the BF16 state, F32 only for the F32 A/B arm).</summary>
    public long SnapshotBytes => (_snapU?.Bytes ?? 0) + (_snapF?.Bytes ?? 0);

    private void RecordWindow(ReadOnlySpan<float> win, int rows)
    {
        long n = (long)rows * _channels;
        if (_storage == Inkling2ConvStateStorage.F32)
        {
            _snapF ??= new NativeBuffer<float>(n); _snapF.Grow(n);
            win.Slice(0, (int)n).CopyTo(_snapF.Span);
        }
        else
        {
            _snapU ??= new NativeBuffer<ushort>(n); _snapU.Grow(n);
            var d = _snapU.Span;
            for (int i = 0; i < n; i++) d[i] = ToBf16(win[i]);   // window values are already BF16-exact: this is lossless
        }
        _snapRows = rows;
    }

    /// <summary>Sets the state to what it was after row <paramref name="row"/> (0-based) of the last recorded <see cref="Run"/>.</summary>
    public void RestoreToRow(int row)
    {
        int C = _channels, K1 = _k1;
        if (_snapRows == 0 || row < 0 || row + K1 >= _snapRows) throw new InvalidOperationException($"No recorded window for row {row} ({_snapRows} rows recorded).");
        for (int k = 0; k < K1; k++)
            for (int c = 0; c < C; c++)
            {
                int i = (row + 1 + k) * C + c;
                if (_f32 is not null) _f32.Span[c * K1 + k] = _snapF!.Span[i];
                else _u16!.Span[c * K1 + k] = _snapU!.Span[i];
            }
    }

    /// <summary>Scratch floats <see cref="Run"/> needs for <paramref name="tokens"/> tokens.</summary>
    public int ScratchFloats(int tokens) => (_k1 + 2 * tokens) * _channels;

    public void Dispose() { _f32?.Dispose(); _u16?.Dispose(); _snapU?.Dispose(); _snapF?.Dispose(); }
}

/// <summary>
/// KV history of one cache slot, stored as IEEE half (llama.cpp's default KV type; K is L2-clipped to a norm of ~22.6 and V is RMS-normed,
/// so the range is safe) plus the four short-convolution states of the layer visit that owns the slot.
/// </summary>
internal sealed class Inkling2SlotState : IDisposable
{
    private readonly int _kvWidth;
    private readonly NativeBuffer<ushort> _k = new(0), _v = new(0);

    public Inkling2ConvState KConv { get; }
    public Inkling2ConvState VConv { get; }
    public Inkling2ConvState AttnPostConv { get; }
    public Inkling2ConvState FfnPostConv { get; }

    public Inkling2SlotState(int kvWidth, int hidden, int kernel, Inkling2ConvStateStorage storage)
    {
        _kvWidth = kvWidth;
        KConv = new Inkling2ConvState(kvWidth, kernel, storage);
        VConv = new Inkling2ConvState(kvWidth, kernel, storage);
        AttnPostConv = new Inkling2ConvState(hidden, kernel, storage);
        FfnPostConv = new Inkling2ConvState(hidden, kernel, storage);
    }

    public long Bytes => _k.Bytes + _v.Bytes + KConv.Bytes + VConv.Bytes + AttnPostConv.Bytes + FfnPostConv.Bytes;

    public void EnsureCapacity(int tokens) { _k.Grow((long)tokens * _kvWidth); _v.Grow((long)tokens * _kvWidth); }

    public void Reset() { _k.Clear(); _v.Clear(); KConv.Reset(); VConv.Reset(); AttnPostConv.Reset(); FfnPostConv.Reset(); }

    public bool RecordRows
    {
        set { KConv.RecordRows = value; VConv.RecordRows = value; AttnPostConv.RecordRows = value; FfnPostConv.RecordRows = value; }
    }

    public void RestoreToRow(int row) { KConv.RestoreToRow(row); VConv.RestoreToRow(row); AttnPostConv.RestoreToRow(row); FfnPostConv.RestoreToRow(row); }

    public void Store(int pos, ReadOnlySpan<float> k, ReadOnlySpan<float> v)
    {
        var ks = MemoryMarshal.Cast<ushort, Half>(_k.Span.Slice(pos * _kvWidth, _kvWidth));
        var vs = MemoryMarshal.Cast<ushort, Half>(_v.Span.Slice(pos * _kvWidth, _kvWidth));
        TensorPrimitives.ConvertToHalf(k.Slice(0, _kvWidth), ks);
        TensorPrimitives.ConvertToHalf(v.Slice(0, _kvWidth), vs);
    }

    /// <summary>K row of <paramref name="pos"/> and KV head <paramref name="kvHead"/>, converted to F32 into <paramref name="dest"/>.</summary>
    public void LoadK(int pos, int kvHead, int headDim, Span<float> dest)
        => TensorPrimitives.ConvertToSingle(MemoryMarshal.Cast<ushort, Half>(_k.Span.Slice(pos * _kvWidth + kvHead * headDim, headDim)), dest);

    /// <summary>V row of <paramref name="pos"/> and KV head <paramref name="kvHead"/>, converted to F32 into <paramref name="dest"/>.</summary>
    public void LoadV(int pos, int kvHead, int headDim, Span<float> dest)
        => TensorPrimitives.ConvertToSingle(MemoryMarshal.Cast<ushort, Half>(_v.Span.Slice(pos * _kvWidth + kvHead * headDim, headDim)), dest);

    public void Dispose() { _k.Dispose(); _v.Dispose(); KConv.Dispose(); VConv.Dispose(); AttnPostConv.Dispose(); FfnPostConv.Dispose(); }
}

/// <summary>
/// Per-sequence state of <see cref="Inkling2TransformerModel"/>: one <see cref="Inkling2SlotState"/> per cache slot (every physical layer plus one
/// extra copy of the looped core per extra pass: 24 + 2 x 12 = 48), because each pass keeps its own KV history and conv states.
/// </summary>
public sealed class Inkling2SequenceState : IDisposable
{
    internal Inkling2SlotState[] Slots { get; }

    /// <summary>Tokens consumed so far.</summary>
    public int Length { get; internal set; }

    internal Inkling2SequenceState(ModelConfig config, Inkling2ConvStateStorage storage)
    {
        var c = config.Inkling2!;
        int kvWidth = config.NumKvHeads * config.HeadDim;
        Slots = new Inkling2SlotState[c.CacheSlots(config.NumLayers)];
        for (int i = 0; i < Slots.Length; i++)
            Slots[i] = new Inkling2SlotState(kvWidth, config.HiddenSize, c.ConvKernel, storage);
    }

    /// <summary>Resident bytes (KV capacity + conv states).</summary>
    public long Bytes => Slots.Sum(s => s.Bytes);

    /// <summary>Bytes of the four conv states of all slots (constant in the sequence length).</summary>
    public long ConvBytes => Slots.Sum(s => s.KConv.Bytes + s.VConv.Bytes + s.AttnPostConv.Bytes + s.FfnPostConv.Bytes);

    /// <summary>Records every conv window of the next forwards (see <see cref="Inkling2ConvState.RecordRows"/>); off by default.</summary>
    internal bool RecordRows { set { foreach (var sl in Slots) sl.RecordRows = value; } }

    /// <summary>Rewinds the conv states to after row <paramref name="row"/> of the last recorded forward started at position <paramref name="pos0"/>; KV beyond is stale and overwritten.</summary>
    internal void RestoreToRow(int row, int pos0)
    {
        foreach (var sl in Slots) sl.RestoreToRow(row);
        Length = pos0 + row + 1;
    }

    /// <summary>Re-zeroes everything and restarts at position 0.</summary>
    public void Reset()
    {
        foreach (var s in Slots) s.Reset();
        Length = 0;
    }

    /// <inheritdoc/>
    public void Dispose() { foreach (var s in Slots) s.Dispose(); }
}
