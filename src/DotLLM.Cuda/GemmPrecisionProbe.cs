using System.Globalization;
using System.Text;

namespace DotLLM.Cuda;

/// <summary>
/// Issue #1109 diagnostic: per-projection error of the prefill GEMM's accumulation and activation formats,
/// measured on the model's real operands. For every probed call the caller runs the same <c>X · Wᵀ</c> four ways
/// (F32 accumulate = reference, F16 accumulate, F16 accumulate promoted to F32 every <see cref="ProbeChunk"/>
/// elements of K, and F32 accumulate over Q8_1-style int8 activations) and hands the host copies here.
/// Results are grouped by projection shape <c>(K, N)</c>, so error growth with the reduction length is visible
/// directly. <c>DOTLLM_CUDA_GEMM_PROBE=1</c> enables it; off by default and costs one static read when off.
/// </summary>
internal sealed class GemmPrecisionProbe
{
    /// <summary>Activation quantization block (Q8_1 / the PQ2_0 dp4a quantizer).</summary>
    internal const int QuantBlock = 32;

    /// <summary>Whether prefill GEMMs are probed. Mutable so a harness can probe a single forward.</summary>
    internal static bool Enabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_CUDA_GEMM_PROBE") == "1";

    /// <summary>K elements per F16 partial in the promoted-chunk arm (<c>DOTLLM_CUDA_GEMM_PROBE_CHUNK</c>, default 256).</summary>
    internal static int ProbeChunk { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_CUDA_GEMM_PROBE_CHUNK"), NumberStyles.Integer,
            CultureInfo.InvariantCulture, out int c) && c > 0 ? c : 256;

    /// <summary>The process-wide accumulator the model reports into.</summary>
    internal static GemmPrecisionProbe Current { get; } = new();

    private readonly Dictionary<(int K, int N), Entry> _entries = [];

    /// <summary>Arms compared against the F32-accumulate reference.</summary>
    internal enum Arm
    {
        /// <summary>F16 accumulate (cuBLAS <c>COMPUTE_16F</c>).</summary>
        F16Accumulate,

        /// <summary>F16 accumulate promoted to F32 every <see cref="ProbeChunk"/> elements.</summary>
        F16Promoted,

        /// <summary>F32 accumulate over Q8_1 round-tripped activations (the W·A8 error, weights exact).</summary>
        Int8Activations,
    }

    /// <summary>Drops everything recorded so far.</summary>
    internal void Reset() => _entries.Clear();

    /// <summary>Number of probed GEMM calls so far.</summary>
    internal int Calls => _entries.Values.Sum(e => e.Calls);

    /// <summary>Records the activation statistics of one call's input.</summary>
    /// <param name="k">Reduction length.</param>
    /// <param name="n">Output features.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="x">Activations as the GEMM saw them, row-major <c>[rows, k]</c>.</param>
    /// <param name="xQ8">The Q8_1 round trip of <paramref name="x"/>.</param>
    internal void RecordActivations(int k, int n, int rows, ReadOnlySpan<Half> x, ReadOnlySpan<Half> xQ8)
    {
        Entry e = GetEntry(k, n);
        e.Calls++;
        e.Rows += rows;
        double sumSqDiff = 0, sumSq = 0;
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<Half> row = x.Slice(r * k, k);
            double rowSq = 0;
            float rowMax = 0;
            for (int i = 0; i < k; i++)
            {
                float v = (float)row[i];
                if (!float.IsFinite(v)) { e.NonFiniteActivations++; continue; }
                float a = MathF.Abs(v);
                rowSq += (double)v * v;
                if (a > rowMax) rowMax = a;
                float d = v - (float)xQ8[r * k + i];
                sumSqDiff += (double)d * d;
            }
            sumSq += rowSq;
            double rms = Math.Sqrt(rowSq / k);
            if (rms > 0) e.RowCrest.Add((float)(rowMax / rms));
            if (rowMax > e.ActivationAbsMax) e.ActivationAbsMax = rowMax;
        }
        e.ActivationSumSq += sumSq;
        e.ActivationElems += (long)rows * k;
        e.Q8SumSqDiff += sumSqDiff;
    }

    /// <summary>Records one arm's output against the reference for one call.</summary>
    /// <param name="k">Reduction length.</param>
    /// <param name="n">Output features.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="arm">Which arm <paramref name="y"/> is.</param>
    /// <param name="reference">F32-accumulate output, row-major <c>[rows, n]</c>.</param>
    /// <param name="y">The arm's output.</param>
    internal void RecordOutput(int k, int n, int rows, Arm arm, ReadOnlySpan<Half> reference, ReadOnlySpan<Half> y)
    {
        Entry e = GetEntry(k, n);
        ErrorStats s = e.Arms[(int)arm];
        if (arm == Arm.F16Accumulate)
        {
            for (int i = 0; i < reference.Length; i++)
            {
                float a = MathF.Abs((float)reference[i]);
                if (a > e.OutputAbsMax) e.OutputAbsMax = a;
            }
        }
        s.Add(reference, y, rows, n);
    }

    /// <summary>
    /// Q8_1-style round trip of <paramref name="x"/>: per block of <see cref="QuantBlock"/> elements,
    /// <c>d = half(absmax / 127)</c>, <c>q = clamp(round(x / d), ±127)</c>, written back as <c>q · d</c>.
    /// Matches the PQ2_0 dp4a quantizer's scale rounding (a half-rounded <c>d</c>).
    /// </summary>
    /// <param name="x">Input, length a multiple of <see cref="QuantBlock"/>.</param>
    /// <param name="dst">Output, same length.</param>
    /// <exception cref="ArgumentException">Lengths differ or are not a multiple of the block.</exception>
    internal static void QuantizeDequantizeQ8_1(ReadOnlySpan<Half> x, Span<Half> dst)
    {
        if (x.Length != dst.Length || x.Length % QuantBlock != 0)
            throw new ArgumentException("x and dst must have equal length, a multiple of 32.");
        for (int b = 0; b < x.Length; b += QuantBlock)
        {
            float amax = 0;
            for (int i = 0; i < QuantBlock; i++) amax = MathF.Max(amax, MathF.Abs((float)x[b + i]));
            float d = (float)(Half)(amax / 127f);
            for (int i = 0; i < QuantBlock; i++)
            {
                float q = d == 0 ? 0 : Math.Clamp(MathF.Round((float)x[b + i] / d), -127f, 127f);
                dst[b + i] = (Half)(q * d);
            }
        }
    }

    /// <summary>Formats the per-shape table (markdown).</summary>
    /// <param name="label">Optional name for a <c>(K, N)</c> shape.</param>
    /// <returns>The report.</returns>
    internal string Report(Func<int, int, string>? label = null)
    {
        var sb = new StringBuilder();
        sb.AppendLine("| K | N | projection | calls | rows | x absmax | x row crest p50/max | Q8_1 x rel err | y absmax "
                      + "| F16acc rel err (agg) | F16acc row rel p50/p99/max | F16acc max|Δ|/rms | promoted rel err (agg) | promoted row p99 "
                      + "| A8 rel err (agg) | A8 row p50/p99/max | non-finite (x / F16acc) |");
        sb.AppendLine("|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|");
        foreach (var ((k, n), e) in _entries.OrderBy(p => p.Key.K).ThenBy(p => p.Key.N))
        {
            ErrorStats f16 = e.Arms[(int)Arm.F16Accumulate];
            ErrorStats pr = e.Arms[(int)Arm.F16Promoted];
            ErrorStats a8 = e.Arms[(int)Arm.Int8Activations];
            sb.Append(CultureInfo.InvariantCulture,
                $"| {k} | {n} | {label?.Invoke(k, n) ?? ""} | {e.Calls} | {e.Rows} | {e.ActivationAbsMax:G4} | "
                + $"{Pct(e.RowCrest, 0.5):F1} / {Pct(e.RowCrest, 1):F1} | {Math.Sqrt(e.Q8SumSqDiff / Math.Max(e.ActivationSumSq, 1e-30)):E2} | "
                + $"{e.OutputAbsMax:G4} | {f16.Aggregate:E2} | {Pct(f16.RowRel, 0.5):E2} / {Pct(f16.RowRel, 0.99):E2} / {Pct(f16.RowRel, 1):E2} | "
                + $"{f16.MaxAbsOverRms:E2} | {pr.Aggregate:E2} | {Pct(pr.RowRel, 0.99):E2} | "
                + $"{a8.Aggregate:E2} | {Pct(a8.RowRel, 0.5):E2} / {Pct(a8.RowRel, 0.99):E2} / {Pct(a8.RowRel, 1):E2} | "
                + $"{e.NonFiniteActivations} / {f16.NonFinite} |");
            sb.AppendLine();
        }
        return sb.ToString();
    }

    /// <summary>Percentile <paramref name="q"/> (0..1) of <paramref name="values"/>; NaN when empty.</summary>
    /// <param name="values">Samples (sorted in place).</param>
    /// <param name="q">Quantile.</param>
    /// <returns>The nearest-rank percentile.</returns>
    internal static double Pct(List<float> values, double q)
    {
        if (values.Count == 0) return double.NaN;
        values.Sort();
        int i = (int)Math.Clamp(Math.Ceiling(q * values.Count) - 1, 0, values.Count - 1);
        return values[i];
    }

    private Entry GetEntry(int k, int n)
    {
        if (!_entries.TryGetValue((k, n), out Entry? e))
            _entries[(k, n)] = e = new Entry();
        return e;
    }

    private sealed class Entry
    {
        public int Calls;
        public long Rows;
        public double ActivationSumSq;
        public long ActivationElems;
        public double Q8SumSqDiff;
        public float ActivationAbsMax;
        public float OutputAbsMax;
        public long NonFiniteActivations;
        public readonly List<float> RowCrest = [];
        public readonly ErrorStats[] Arms = [new(), new(), new()];
    }

    /// <summary>Accumulated error of one arm against the reference.</summary>
    internal sealed class ErrorStats
    {
        private double _sumSqDiff;
        private double _sumSqRef;

        /// <summary>Per-row relative L2 error samples.</summary>
        public List<float> RowRel { get; } = [];

        /// <summary>Largest |Δ| over the RMS of its reference row.</summary>
        public double MaxAbsOverRms { get; private set; }

        /// <summary>Non-finite outputs of the arm.</summary>
        public long NonFinite { get; private set; }

        /// <summary>Aggregate relative L2 error over every recorded element: <c>‖Δ‖ / ‖ref‖</c>.</summary>
        public double Aggregate => _sumSqRef > 0 ? Math.Sqrt(_sumSqDiff / _sumSqRef) : double.NaN;

        /// <summary>Adds one call's rows.</summary>
        /// <param name="reference">Reference output.</param>
        /// <param name="y">Arm output.</param>
        /// <param name="rows">Rows.</param>
        /// <param name="n">Row length.</param>
        public void Add(ReadOnlySpan<Half> reference, ReadOnlySpan<Half> y, int rows, int n)
        {
            if (y.IsEmpty) return;
            for (int r = 0; r < rows; r++)
            {
                double dd = 0, rr = 0, maxAbs = 0;
                for (int i = r * n; i < (r + 1) * n; i++)
                {
                    float a = (float)reference[i], b = (float)y[i];
                    if (!float.IsFinite(b)) { NonFinite++; continue; }
                    if (!float.IsFinite(a)) continue;
                    double d = (double)b - a;
                    dd += d * d;
                    rr += (double)a * a;
                    if (Math.Abs(d) > maxAbs) maxAbs = Math.Abs(d);
                }
                _sumSqDiff += dd;
                _sumSqRef += rr;
                if (rr > 0)
                {
                    RowRel.Add((float)Math.Sqrt(dd / rr));
                    double rms = Math.Sqrt(rr / n);
                    if (maxAbs / rms > MaxAbsOverRms) MaxAbsOverRms = maxAbs / rms;
                }
            }
        }
    }
}
