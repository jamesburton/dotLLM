using System.Runtime.CompilerServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Env-controlled tensor stats dumper for layer-by-layer parity diffs against llama.cpp.
/// Default off (no env var = no output, no allocation, no per-tensor branch cost). Enable
/// by setting <c>DOTLLM_TENSOR_DUMP=1</c> to write to stderr, or <c>DOTLLM_TENSOR_DUMP_DIR=&lt;path&gt;</c>
/// to write per-tensor binary files to a directory.
/// </summary>
/// <remarks>
/// <para>
/// Output format (stderr line-per-tensor, compatible with llama-eval-callback's debug print):
/// </para>
/// <code>
/// [tensor] {name}: shape=[d0,d1,...] dtype=F32 n=N min=... max=... mean=... rms=... abs_max=...
/// </code>
/// <para>
/// When <c>DOTLLM_TENSOR_DUMP_DIR</c> is set the raw F32 bytes are also written to
/// <c>{dir}/{name}.bin</c> with a 16-byte header: <c>[i32 rank, i32 dim0, i32 dim1, i32 dim2]</c>
/// (rank ≤ 3 covered; padded with -1 for unused dims), followed by <c>n × float32</c>.
/// </para>
/// <para>
/// Tensor names should mirror llama.cpp's <c>cb(tensor, "name", il)</c> tags so the parity
/// diff is a simple key-by-key compare. Per-layer names are prefixed with <c>blk.NN.</c>.
/// </para>
/// </remarks>
internal static unsafe class TensorDump
{
    private static readonly bool s_enabled =
        Environment.GetEnvironmentVariable("DOTLLM_TENSOR_DUMP") == "1"
        || !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTLLM_TENSOR_DUMP_DIR"));
    private static readonly string? s_dir = Environment.GetEnvironmentVariable("DOTLLM_TENSOR_DUMP_DIR");
    private static int s_callCount;

    /// <summary>Hot-path branch predicate. Compile-time-constant when env var unset.</summary>
    public static bool Enabled => s_enabled;

    /// <summary>
    /// Dumps a 1D F32 tensor's stats (and optionally raw bytes) under the given name.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Dump1D(string name, float* data, int n)
    {
        if (!s_enabled || n <= 0) return;
        WriteStats(name, new[] { n }, data, n);
        WriteBin(name, new[] { n }, data, n);
    }

    /// <summary>
    /// Dumps a 2D F32 tensor [d0, d1] (d1 innermost in memory) — i.e., row-major with d1 columns.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Dump2D(string name, float* data, int d0, int d1)
    {
        if (!s_enabled) return;
        long n = (long)d0 * d1;
        if (n <= 0) return;
        WriteStats(name, new[] { d0, d1 }, data, n);
        WriteBin(name, new[] { d0, d1 }, data, n);
    }

    /// <summary>
    /// Dumps a 3D F32 tensor [d0, d1, d2] (d2 innermost in memory).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Dump3D(string name, float* data, int d0, int d1, int d2)
    {
        if (!s_enabled) return;
        long n = (long)d0 * d1 * d2;
        if (n <= 0) return;
        WriteStats(name, new[] { d0, d1, d2 }, data, n);
        WriteBin(name, new[] { d0, d1, d2 }, data, n);
    }

    /// <summary>Dump from a managed array.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Dump1D(string name, ReadOnlySpan<float> data)
    {
        if (!s_enabled || data.Length == 0) return;
        fixed (float* p = data) { Dump1D(name, p, data.Length); }
    }

    private static void WriteStats(string name, int[] shape, float* data, long n)
    {
        double sum = 0, sumSq = 0;
        float mn = float.PositiveInfinity, mx = float.NegativeInfinity;
        int nans = 0, infs = 0;
        for (long i = 0; i < n; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) { nans++; continue; }
            if (float.IsInfinity(v)) { infs++; continue; }
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
            sumSq += (double)v * v;
        }
        long valid = n - nans - infs;
        double mean = valid > 0 ? sum / valid : 0;
        double rms = valid > 0 ? Math.Sqrt(sumSq / valid) : 0;
        float absMax = MathF.Max(MathF.Abs(mn), MathF.Abs(mx));
        string shapeStr = "[" + string.Join(",", shape) + "]";
        string nanInf = (nans > 0 ? $" NaN={nans}" : "") + (infs > 0 ? $" Inf={infs}" : "");
        Console.Error.WriteLine(
            $"[tensor] {name}: shape={shapeStr} n={n} min={mn:F6} max={mx:F6} mean={mean:F6} rms={rms:F6} abs_max={absMax:F6}{nanInf}");
    }

    private static void WriteBin(string name, int[] shape, float* data, long n)
    {
        if (s_dir is null) return;
        try
        {
            Directory.CreateDirectory(s_dir);
            // Order tensors by emit order so diff tools can match by call index.
            int idx = Interlocked.Increment(ref s_callCount) - 1;
            string safe = name.Replace('/', '_').Replace('\\', '_');
            string path = Path.Combine(s_dir, $"{idx:D5}_{safe}.bin");
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);
            // Header: rank then up to 3 dims (padded with -1).
            bw.Write(shape.Length);
            bw.Write(shape.Length > 0 ? shape[0] : -1);
            bw.Write(shape.Length > 1 ? shape[1] : -1);
            bw.Write(shape.Length > 2 ? shape[2] : -1);
            // Payload: n × float32.
            byte* bytePtr = (byte*)data;
            int chunkBytes = checked((int)Math.Min(n * sizeof(float), 1 << 20));
            for (long off = 0; off < n * sizeof(float); off += chunkBytes)
            {
                int remaining = (int)Math.Min(chunkBytes, n * sizeof(float) - off);
                bw.Write(new ReadOnlySpan<byte>(bytePtr + off, remaining));
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[tensor] WriteBin failed for '{name}': {ex.Message}");
        }
    }
}
