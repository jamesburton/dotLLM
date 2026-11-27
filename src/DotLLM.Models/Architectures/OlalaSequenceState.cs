using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>Zeroed, 64-byte-aligned native buffer of <typeparamref name="T"/> (freed deterministically).</summary>
internal sealed unsafe class NativeBuffer<T> : IDisposable where T : unmanaged
{
    private T* _ptr;
    private long _count;

    public NativeBuffer(long count) { Allocate(count); }

    private void Allocate(long count)
    {
        _count = Math.Max(count, 1);
        _ptr = (T*)NativeMemory.AlignedAlloc((nuint)(_count * sizeof(T)), 64);
        new Span<T>(_ptr, checked((int)_count)).Clear();
    }

    public T* Ptr => _ptr;
    public long Count => _count;
    public long Bytes => _count * sizeof(T);
    public Span<T> Span => new(_ptr, checked((int)_count));

    /// <summary>Grows to at least <paramref name="count"/> elements (doubling), preserving the contents; new elements are zero.</summary>
    public void Grow(long count)
    {
        if (count <= _count) return;
        long next = Math.Max(count, _count * 2);
        var old = _ptr; long oldCount = _count;
        Allocate(next);
        new Span<T>(old, checked((int)oldCount)).CopyTo(new Span<T>(_ptr, checked((int)oldCount)));
        NativeMemory.AlignedFree(old);
    }

    public void Clear() => new Span<T>(_ptr, checked((int)_count)).Clear();

    public void Dispose()
    {
        if (_ptr is not null) { NativeMemory.AlignedFree(_ptr); _ptr = null; }
    }
}

/// <summary>Recurrent state of one Mamba-3 MIMO layer; every buffer is F32 (<see cref="OlalaStatePolicy"/>).</summary>
internal sealed class OlalaSsmLayerState : IDisposable
{
    public NativeBuffer<float> Ssm { get; }       // [H][P][N]
    public NativeBuffer<float> Angle { get; }     // [H][A]
    public NativeBuffer<float> K { get; }         // [R][H][N]  previous token's rotated K
    public NativeBuffer<float> V { get; }         // [H][P]     previous token's raw x

    public OlalaSsmLayerState(OlalaConfig o)
    {
        int H = o.NumSsmHeads;
        Ssm = new NativeBuffer<float>((long)H * o.MambaHeadDim * o.StateSize);
        Angle = new NativeBuffer<float>((long)H * o.NumRopeAngles);
        K = new NativeBuffer<float>((long)o.MimoRank * H * o.StateSize);
        V = new NativeBuffer<float>((long)H * o.MambaHeadDim);
    }

    public long Bytes => Ssm.Bytes + Angle.Bytes + K.Bytes + V.Bytes;
    public void Reset() { Ssm.Clear(); Angle.Clear(); K.Clear(); V.Clear(); }
    public void Dispose() { Ssm.Dispose(); Angle.Dispose(); K.Dispose(); V.Dispose(); }
}

/// <summary>Attention state of one V layer: BF16 K/V history of the post-shift, post-norm K and post-shift V, plus the token-shift carry.</summary>
internal sealed unsafe class OlalaAttnLayerState : IDisposable
{
    private readonly int _kvWidth;
    private NativeBuffer<ushort> _k, _v;

    /// <summary>Previous token's raw (pre-shift) K <c>[kv heads * head_dim]</c>, F32 (<see cref="OlalaStatePolicy.TokenShift"/>).</summary>
    public NativeBuffer<float> LastK { get; }
    /// <summary>Previous token's raw (pre-shift) V.</summary>
    public NativeBuffer<float> LastV { get; }

    public OlalaAttnLayerState(int kvWidth)
    {
        _kvWidth = kvWidth;
        _k = new NativeBuffer<ushort>(0); _v = new NativeBuffer<ushort>(0);
        LastK = new NativeBuffer<float>(kvWidth); LastV = new NativeBuffer<float>(kvWidth);
    }

    public long Bytes => _k.Bytes + _v.Bytes + LastK.Bytes + LastV.Bytes;

    public void EnsureCapacity(int tokens) { _k.Grow((long)tokens * _kvWidth); _v.Grow((long)tokens * _kvWidth); }

    public void Reset() { _k.Clear(); _v.Clear(); LastK.Clear(); LastV.Clear(); }

    /// <summary>Stores the K and V rows of position <paramref name="pos"/> as BF16.</summary>
    public void Store(int pos, ReadOnlySpan<float> k, ReadOnlySpan<float> v, int width)
    {
        var ks = _k.Span.Slice(pos * width, width); var vs = _v.Span.Slice(pos * width, width);
        for (int i = 0; i < width; i++) { ks[i] = OlalaOps.ToBf16(k[i]); vs[i] = OlalaOps.ToBf16(v[i]); }
    }

    /// <summary><c>q . K[pos, kvHead]</c> (F32 accumulate over BF16 K).</summary>
    public float DotK(int pos, int kvHead, int headDim, int width, ReadOnlySpan<float> q)
    {
        var k = _k.Span.Slice(pos * width + kvHead * headDim, headDim);
        float s = 0f;
        for (int i = 0; i < headDim; i++) s += q[i] * OlalaOps.FromBf16(k[i]);
        return s;
    }

    /// <summary><c>o += w * V[pos, kvHead]</c>.</summary>
    public void AddV(int pos, int kvHead, int headDim, int width, float w, Span<float> o)
    {
        var v = _v.Span.Slice(pos * width + kvHead * headDim, headDim);
        for (int i = 0; i < headDim; i++) o[i] += w * OlalaOps.FromBf16(v[i]);
    }

    public void Dispose() { _k.Dispose(); _v.Dispose(); LastK.Dispose(); LastV.Dispose(); }
}

/// <summary>
/// Per-sequence state of <see cref="OlalaTransformerModel"/>: the F32 recurrent state of the M layers (policy:
/// <see cref="OlalaStatePolicy"/>, issue #1971) and the K/V history of the V layers.
/// </summary>
public sealed class OlalaSequenceState : IDisposable
{
    internal OlalaSsmLayerState[] Ssm { get; }
    internal OlalaAttnLayerState[] Attn { get; }

    /// <summary>Tokens consumed so far.</summary>
    public int Length { get; internal set; }

    internal OlalaSequenceState(OlalaConfig o, ModelConfig config)
    {
        Ssm = new OlalaSsmLayerState[o.SsmLayerCount];
        for (int i = 0; i < Ssm.Length; i++) Ssm[i] = new OlalaSsmLayerState(o);
        Attn = new OlalaAttnLayerState[o.AttentionLayerCount];
        for (int i = 0; i < Attn.Length; i++) Attn[i] = new OlalaAttnLayerState(o.NumNoiseHeads * config.HeadDim);
    }

    /// <summary>Resident bytes (recurrent state + current K/V capacity).</summary>
    public long Bytes => Ssm.Sum(s => s.Bytes) + Attn.Sum(a => a.Bytes);

    /// <summary>Bytes of the recurrent (SSM + angle + trapezoid) state of the M layers only; constant in the sequence length.</summary>
    public long RecurrentBytes => Ssm.Sum(s => s.Bytes);

    /// <summary>Bytes of the V layers' token-shift carry (last raw K and V), constant in the sequence length.</summary>
    public long TokenShiftBytes => Attn.Sum(a => a.LastK.Bytes + a.LastV.Bytes);

    /// <summary>Size in bytes of one element of the SSM matrix (the policy; asserted by tests).</summary>
    internal int SsmElementBytes => (int)(Ssm.Length == 0 ? 0 : Ssm[0].Ssm.Bytes / Ssm[0].Ssm.Count);

    /// <summary>Re-zeroes everything and restarts at position 0.</summary>
    public void Reset()
    {
        foreach (var s in Ssm) s.Reset();
        foreach (var a in Attn) a.Reset();
        Length = 0;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var s in Ssm) s.Dispose();
        foreach (var a in Attn) a.Dispose();
    }
}
