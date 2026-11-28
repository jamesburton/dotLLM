using System.Runtime.CompilerServices;

namespace DotLLM.Models.Architectures;

/// <summary>Activation storage format simulated by <see cref="Inkling2ActSim"/>.</summary>
internal enum Inkling2SimFormat
{
    /// <summary>No rounding (the CPU reference's native F32 activations).</summary>
    F32,

    /// <summary>Round to nearest even bfloat16 (the Vulkan default).</summary>
    Bf16,

    /// <summary>Round to nearest even IEEE half; overflow becomes infinity.</summary>
    F16,
}

/// <summary>Per-point statistics of the values that reach a simulated activation store (before rounding).</summary>
internal sealed class Inkling2ActStat
{
    /// <summary>Largest |value| seen.</summary>
    public float MaxAbs;

    /// <summary>Number of values seen.</summary>
    public long Count;

    /// <summary>Values with |v| &gt;= 65520 (they would round to +-infinity in F16).</summary>
    public long F16Overflow;

    /// <summary>Non-zero values below the F16 normal range (|v| &lt; 2^-14): reduced precision in F16.</summary>
    public long F16Subnormal;

    /// <summary>Non-zero values that round to zero in F16 (|v| &lt;= 2^-25).</summary>
    public long F16Flushed;
}

/// <summary>
/// TEST / MEASUREMENT hook of the CPU reference (activation-precision study, epic #2025): rounds the activations of <see cref="Inkling2TransformerModel"/> at the same STORAGE points the
/// Vulkan forward stores them (every kernel output that lands in a scratch buffer), in a per-point / per-layer / per-row chosen format, and records the value ranges. With no
/// simulator attached the model does one null check per point and nothing else. Accumulation and every value inside a fused kernel stay F32, as on the GPU.
/// </summary>
internal sealed class Inkling2ActSim : DotLLM.Models.Architectures.DSpark.IInkling2Tap
{
    /// <summary>Max |value| of the rows the DSpark drafter would receive per layer visit (<see cref="DotLLM.Models.Architectures.DSpark.IInkling2Tap"/>: visits 5, 23, 41 are layer inputs, 48 the final normed output).</summary>
    public readonly float[] TapMax = new float[64];

    /// <inheritdoc/>
    public void Capture(int visit, ReadOnlySpan<float> rows, int rowCount)
    {
        float m = 0; foreach (float v in rows) { float a = MathF.Abs(v); if (a > m) m = a; }
        if ((uint)visit < (uint)TapMax.Length && m > TapMax[visit]) TapMax[visit] = m;
    }

    /// <summary>Points on the residual stream (the layer outputs and what feeds the block stack).</summary>
    public static readonly HashSet<string> ResidualPoints = ["embed", "ffnInp", "blk", "layer_out"];

    /// <summary>Format of every point not overridden.</summary>
    public Inkling2SimFormat Base = Inkling2SimFormat.Bf16;

    /// <summary>When set, the residual-stream points use this format instead of <see cref="Base"/>.</summary>
    public Inkling2SimFormat? Residual;

    /// <summary>Layer visits (0..47, in execution order) below this use F32 for every point.</summary>
    public int F32UpToVisit;

    /// <summary>Rows whose absolute sequence position satisfies this are F32 (for example the image rows).</summary>
    public Func<int, bool>? RowIsF32;

    /// <summary>Format of the points that are NOT the residual stream, for layers in the visit range <see cref="NonResidualF32From"/>..<see cref="NonResidualF32To"/> (exclusive): F32.</summary>
    public int NonResidualF32From, NonResidualF32To;

    /// <summary>Statistics per point name (aggregated over every layer and call).</summary>
    public readonly Dictionary<string, Inkling2ActStat> Stats = new();

    /// <summary>Max |value| of the residual-stream points per layer visit (index visit + 1; 0 = the embedding, 49 = the output stage).</summary>
    public readonly float[] ResidualMaxPerVisit = new float[64];

    /// <summary>Layer visit being executed (set by the model; -1 = embedding, 48 = output).</summary>
    public int Visit = -1;

    /// <summary>Absolute position of the first row of the current call (set by the model).</summary>
    public int Pos0;

    /// <summary>When false the simulator records statistics but never rounds (the F32 reference run).</summary>
    public bool Round = true;

    private Inkling2SimFormat FormatOf(string point, int token)
    {
        if (!Round) return Inkling2SimFormat.F32;
        if (Visit < F32UpToVisit) return Inkling2SimFormat.F32;
        if (RowIsF32 is { } f && f(Pos0 + token)) return Inkling2SimFormat.F32;
        bool resid = ResidualPoints.Contains(point);
        if (!resid && Visit >= NonResidualF32From && Visit < NonResidualF32To) return Inkling2SimFormat.F32;
        return resid && Residual is { } r ? r : Base;
    }

    /// <summary>Applies the simulated store to <paramref name="data"/> = <c>[T][rowsPerToken][cols]</c> (token index = row / rowsPerToken).</summary>
    public void Apply(string point, Span<float> data, int cols, int rowsPerToken)
    {
        int rows = data.Length / cols;
        for (int r = 0; r < rows; r++) ApplyRow(point, data.Slice(r * cols, cols), r / rowsPerToken);
    }

    /// <summary>Applies the simulated store to one row of token <paramref name="token"/> (relative to the call).</summary>
    public void ApplyRow(string point, Span<float> row, int token)
    {
        if (!Stats.TryGetValue(point, out var st)) Stats[point] = st = new Inkling2ActStat();
        float max = 0;
        long over = 0, sub = 0, flush = 0;
        for (int i = 0; i < row.Length; i++)
        {
            float a = MathF.Abs(row[i]);
            if (a > max) max = a;
            if (a >= 65520f) over++;
            else if (a != 0f)
            {
                if (a < 6.103515625e-5f) sub++;
                if (a <= 2.98023224e-8f) flush++;
            }
        }
        st.Count += row.Length; st.F16Overflow += over; st.F16Subnormal += sub; st.F16Flushed += flush;
        if (max > st.MaxAbs) st.MaxAbs = max;
        if (ResidualPoints.Contains(point))
        {
            int ix = Math.Clamp(Visit + 1, 0, ResidualMaxPerVisit.Length - 1);
            if (max > ResidualMaxPerVisit[ix]) ResidualMaxPerVisit[ix] = max;
        }
        switch (FormatOf(point, token))
        {
            case Inkling2SimFormat.Bf16:
                for (int i = 0; i < row.Length; i++) row[i] = ToBf16(row[i]);
                break;
            case Inkling2SimFormat.F16:
                for (int i = 0; i < row.Length; i++) row[i] = (float)(Half)row[i];
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ToBf16(float f)
    {
        uint b = BitConverter.SingleToUInt32Bits(f);
        if ((b & 0x7fffffffu) > 0x7f800000u) return f;
        b += 0x7fffu + ((b >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(b & 0xFFFF0000u);
    }
}
