using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// A model's RoPE rotation held as its <c>ropeDim / 2</c> inverse frequencies and cos/sin multiplier only, with the cos/sin
/// rows computed per forward for the positions in flight (issue #1101). Replaces position-indexed tables sized by the
/// context window: with YaRN that window is about 1M tokens, and a <c>[window, ropeDim / 2]</c> F32 pair would be 256 MiB
/// of context-sized F32 storage (#1107).
/// </summary>
/// <remarks>
/// <para>
/// The values are bit-identical to the old tables: the plain inverse frequencies are formed exactly as
/// <see cref="RoPE.PrecomputeFrequencyTable"/> forms them, the YaRN ones come from
/// <see cref="RoPE.ResolveDenseYarn"/> (as <see cref="RoPE.PrecomputeFrequencyTableYarn"/>), and each entry is
/// <c>MathF.Cos/Sin((float)pos * invFreq[i])</c> (times mscale with YaRN).
/// </para>
/// <para>
/// The rows of the last position list are kept and reused, so a forward that rotates the same positions in every attention
/// layer computes them once. Not thread-safe: one instance per model, used by that model's (serial) forward.
/// </para>
/// </remarks>
public sealed class RoPERotation
{
    private readonly float[] _invFreq;
    private readonly bool _scaled;
    private int[] _positions = [];
    private int _count;
    private float[] _cos = [];
    private float[] _sin = [];
    private int[] _rowIndex = [];

    /// <summary>Creates the rotation for a model.</summary>
    /// <param name="rope">The model's RoPE configuration, or null (theta 10000).</param>
    /// <param name="architecture">Selects the YaRN mscale convention.</param>
    /// <param name="ropeDim">Rotated dims per head (even).</param>
    public RoPERotation(RoPEConfig? rope, Architecture architecture, int ropeDim)
    {
        if (ropeDim <= 0 || (ropeDim & 1) != 0)
            throw new ArgumentException($"ropeDim must be a positive even number, got {ropeDim}", nameof(ropeDim));
        RopeDim = ropeDim;
        var (yarnInvFreq, mscale) = RoPE.ResolveDenseYarn(rope, architecture, ropeDim);
        if (yarnInvFreq is not null)
        {
            _invFreq = yarnInvFreq;
            Mscale = mscale;
            _scaled = true;
        }
        else
        {
            float theta = rope?.Theta ?? 10000.0f;
            _invFreq = new float[ropeDim / 2];
            for (int i = 0; i < _invFreq.Length; i++)
                _invFreq[i] = 1.0f / MathF.Pow(theta, 2.0f * i / ropeDim);
            Mscale = 1.0f;
        }
    }

    /// <summary>Rotated dims per head.</summary>
    public int RopeDim { get; }

    /// <summary>The <c>ropeDim / 2</c> inverse frequencies.</summary>
    public ReadOnlySpan<float> InverseFrequencies => _invFreq;

    /// <summary>The cos/sin multiplier (YaRN attention factor; 1 without YaRN).</summary>
    public float Mscale { get; }

    /// <summary>
    /// Writes the cos/sin rows for <paramref name="positions"/>: row <c>t</c> holds <c>ropeDim / 2</c> entries for
    /// <c>positions[t]</c>.
    /// </summary>
    /// <param name="positions">Positions.</param>
    /// <param name="cos">Destination, at least <c>positions.Length * ropeDim / 2</c>.</param>
    /// <param name="sin">Destination, at least <c>positions.Length * ropeDim / 2</c>.</param>
    public void FillRows(ReadOnlySpan<int> positions, Span<float> cos, Span<float> sin)
    {
        int half = _invFreq.Length;
        for (int t = 0; t < positions.Length; t++)
        {
            float pos = positions[t];
            int row = t * half;
            for (int i = 0; i < half; i++)
            {
                float angle = pos * _invFreq[i];
                if (_scaled)
                {
                    cos[row + i] = MathF.Cos(angle) * Mscale;
                    sin[row + i] = MathF.Sin(angle) * Mscale;
                }
                else
                {
                    cos[row + i] = MathF.Cos(angle);
                    sin[row + i] = MathF.Sin(angle);
                }
            }
        }
    }

    /// <summary>Rotates Q and K in place (see <see cref="RoPE.Execute(Span{float}, Span{float}, ReadOnlySpan{int}, int, int, int, int, ReadOnlySpan{float}, ReadOnlySpan{float}, RoPEType)"/>).</summary>
    /// <param name="q">Query rows, <c>[seqLen, numHeads * headDim]</c> (may be empty with <paramref name="numHeads"/> 0).</param>
    /// <param name="k">Key rows, <c>[seqLen, numKvHeads * headDim]</c>.</param>
    /// <param name="positions">Position per row.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">Key heads.</param>
    /// <param name="headDim">Head width.</param>
    /// <param name="ropeType">Pairing convention.</param>
    public void Apply(Span<float> q, Span<float> k, ReadOnlySpan<int> positions,
        int numHeads, int numKvHeads, int headDim, RoPEType ropeType)
    {
        EnsureRows(positions);
        int half = _invFreq.Length;
        RoPE.Execute(q, k, _rowIndex.AsSpan(0, positions.Length), numHeads, numKvHeads, headDim, RopeDim,
            _cos.AsSpan(0, positions.Length * half), _sin.AsSpan(0, positions.Length * half), ropeType);
    }

    private void EnsureRows(ReadOnlySpan<int> positions)
    {
        int n = positions.Length;
        if (n == _count && positions.SequenceEqual(_positions.AsSpan(0, n)))
            return;

        int half = _invFreq.Length;
        if (_positions.Length < n)
        {
            int cap = Math.Max(n, _positions.Length * 2);
            _positions = new int[cap];
            _cos = new float[cap * half];
            _sin = new float[cap * half];
            _rowIndex = new int[cap];
            for (int i = 0; i < cap; i++) _rowIndex[i] = i;
        }

        positions.CopyTo(_positions);
        _count = n;
        FillRows(positions, _cos, _sin);
    }
}
