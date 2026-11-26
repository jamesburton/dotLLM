using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>Storage type of the carried short-convolution inputs (the recurrent state of Inkling 2).</summary>
public enum Inkling2ConvStateStorage
{
    /// <summary>IEEE single precision (llama.cpp's recurrent-state type; the A/B reference arm).</summary>
    F32,

    /// <summary>IEEE half precision.</summary>
    F16,

    /// <summary>bfloat16.</summary>
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

    public void Reset() { _f32?.Clear(); _u16?.Clear(); }

    private float Load(int i) => _storage switch
    {
        Inkling2ConvStateStorage.F32 => _f32!.Span[i],
        Inkling2ConvStateStorage.F16 => (float)BitConverter.UInt16BitsToHalf(_u16!.Span[i]),
        _ => BitConverter.Int32BitsToSingle(_u16!.Span[i] << 16),
    };

    private void Store(int i, float v)
    {
        switch (_storage)
        {
            case Inkling2ConvStateStorage.F32: _f32!.Span[i] = v; break;
            case Inkling2ConvStateStorage.F16: _u16!.Span[i] = BitConverter.HalfToUInt16Bits((Half)v); break;
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
    public void Run(int tokens, ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> dyn, int heads, Span<float> output)
    {
        int C = _channels, K1 = _k1, K = K1 + 1;
        bool hasDyn = !dyn.IsEmpty;
        int rows = K1 + tokens;
        float[] win = ArrayPool<float>.Shared.Rent(rows * C);
        float[] res = ArrayPool<float>.Shared.Rent(tokens * C);
        try
        {
            for (int k = 0; k < K1; k++)
                for (int c = 0; c < C; c++) win[k * C + c] = Load(c * K1 + k);
            input.Slice(0, tokens * C).CopyTo(win.AsSpan(K1 * C, tokens * C));

            int perHead = hasDyn ? C / heads : 1;
            float scale = 1f / MathF.Sqrt((hasDyn ? 2 : 1) * K + 1);
            for (int t = 0; t < tokens; t++)
            {
                for (int c = 0; c < C; c++)
                {
                    float acc = 0f;
                    var w = weight.Slice(c * K, K);
                    for (int i = 0; i < K; i++) acc += win[(t + i) * C + c] * w[i];
                    if (hasDyn)
                    {
                        var d = dyn.Slice(t * heads * K + (c / perHead) * K, K);
                        for (int lag = 0; lag < K; lag++) acc += win[(t + K - 1 - lag) * C + c] * d[lag];
                    }
                    res[t * C + c] = (win[(t + K1) * C + c] + acc) * scale;
                }
            }

            for (int k = 0; k < K1; k++)
                for (int c = 0; c < C; c++) Store(c * K1 + k, win[(tokens + k) * C + c]);
            res.AsSpan(0, tokens * C).CopyTo(output);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(win);
            ArrayPool<float>.Shared.Return(res);
        }
    }

    public void Dispose() { _f32?.Dispose(); _u16?.Dispose(); }
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

    /// <summary>Re-zeroes everything and restarts at position 0.</summary>
    public void Reset()
    {
        foreach (var s in Slots) s.Reset();
        Length = 0;
    }

    /// <inheritdoc/>
    public void Dispose() { foreach (var s in Slots) s.Dispose(); }
}
