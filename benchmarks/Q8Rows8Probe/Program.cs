using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;

// Q8Rows8Probe (#1119, #1118): in-process interleaved A/B of CPU GEMM/GEMV kernels.
//   dotnet run -c Release --project benchmarks/Q8Rows8Probe -- [--threads 16] [--reps 7] [--n 64] [--shapes 8192x3072,4096x4096] [--bf16] [--q8]
// Arms rotate every rep so clock ramp and contention hit every arm alike; min and median GFLOP/s are reported.

unsafe
{
    int threads = Arg("--threads", Math.Min(16, Environment.ProcessorCount));
    int reps = Arg("--reps", 7);
    int[] ns = ArgList("--n", "64,1");
    string[] shapes = ArgStr("--shapes", "8192x3072,4096x4096,12800x4096").Split(',');
    bool doBf16 = Has("--bf16") || !Has("--q8");
    bool doQ8 = Has("--q8") || !Has("--bf16");

    Console.WriteLine($"{CpuFeatures.Describe()} threads={threads} reps={reps} Q8Rows8Tier(default)={MatMul.Q8_0Rows8Tier} Bf16FmaTier={MatMul.Bf16FmaTier} Avx512F={CpuFeatures.Avx512F}");
    using var pool = new ComputeThreadPool(threads);

    foreach (string shape in shapes)
    {
        var p = shape.Split('x');
        int m = int.Parse(p[0], CultureInfo.InvariantCulture), k = int.Parse(p[1], CultureInfo.InvariantCulture);
        foreach (int n in ns)
        {
            var rng = new Random(42);
            float* wf = Alloc<float>((long)m * k);
            float* b = Alloc<float>((long)n * k);
            float* c = Alloc<float>((long)n * m);
            for (long i = 0; i < (long)m * k; i++) wf[i] = (rng.NextSingle() * 2f - 1f) * 0.05f;
            for (long i = 0; i < (long)n * k; i++) b[i] = (rng.NextSingle() * 2f - 1f) * 2f;
            double flop = 2.0 * m * k * n;

            var arms = new List<(string Name, Action Run)>();
            arms.Add(("F32 GemmF32", () => MatMul.GemmF32(wf, b, c, m, k, n, pool)));

            byte* wq = null; byte* xq = null; ushort* wb = null;
            if (doQ8)
            {
                int blockCount = k / 32, rowBytes = blockCount * 34;
                wq = Alloc<byte>((long)m * rowBytes);
                xq = Alloc<byte>((long)n * rowBytes);
                for (int r = 0; r < m; r++) MatMul.QuantizeF32ToQ8_0(wf + (long)r * k, wq + (long)r * rowBytes, k);
                for (int t = 0; t < n; t++) MatMul.QuantizeF32ToQ8_0(b + (long)t * k, xq + (long)t * rowBytes, k);
                var rw = WeightRepacking.RepackR4((nint)wq, QuantizationType.Q8_0, m, k);
                // Production prefill path (TransformerModel.GemmInterleaved -> GemmR4TiledQ8_0; n == 1 -> ComputeRowsQ8_0Interleaved).
                arms.Add(("Q8_0 old (R4 view, DOTLLM_CPU_Q8_ROWS8=0)", () =>
                {
                    MatMul.Q8_0Rows8Enabled = false;
                    if (n == 1) MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xq, c, rw.FullGroupCount, rw.TailRows, blockCount, pool);
                    else MatMul.GemmR4TiledQ8_0((byte*)rw.Ptr, xq, c, rw.FullGroupCount, rw.TailRows, blockCount, m, n, pool);
                }));
                arms.Add(("Q8_0 new (8-row VNNI)", () =>
                {
                    MatMul.Q8_0Rows8Enabled = true;
                    if (n == 1) MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xq, c, rw.FullGroupCount, rw.TailRows, blockCount, pool);
                    else MatMul.GemmR4TiledQ8_0((byte*)rw.Ptr, xq, c, rw.FullGroupCount, rw.TailRows, blockCount, m, n, pool);
                }));
                arms.Add(("Q8_0 old row-major GemmQ8_0", () => { MatMul.Q8_0Rows8Enabled = false; MatMul.GemmQ8_0(wq, null, c, m, k, n, pool, xq); }));
                arms.Add(("Q8_0 new row-major GemmQ8_0", () => { MatMul.Q8_0Rows8Enabled = true; MatMul.GemmQ8_0(wq, null, c, m, k, n, pool, xq); }));
            }
            if (doBf16)
            {
                wb = Alloc<ushort>((long)m * k);
                for (long i = 0; i < (long)m * k; i++) wb[i] = (ushort)(BitConverter.SingleToInt32Bits(wf[i]) >> 16);
                arms.Add(("BF16 old (dequant-rows)", () => MatMul.GemmDequantRows((byte*)wb, QuantizationType.BF16, b, c, m, k, n, pool)));
                arms.Add(("BF16 new 256 tile (AVX2/FMA)", () => { MatMul.Bf16Avx512Enabled = false; MatMul.Bf16FmaScratch = false; MatMul.GemmBf16((nint)wb, b, c, m, k, n, pool); }));
                arms.Add(("BF16 new 256 scratch (AVX2/FMA)", () => { MatMul.Bf16Avx512Enabled = false; MatMul.Bf16FmaScratch = true; MatMul.GemmBf16((nint)wb, b, c, m, k, n, pool); }));
                if (CpuFeatures.Avx512F)
                    arms.Add(("BF16 new 512 (AVX-512F opt-in)", () => { MatMul.Bf16Avx512Enabled = true; MatMul.GemmBf16((nint)wb, b, c, m, k, n, pool); }));
            }

            // Warm-up (JIT + clocks), then rotate the arm order every rep.
            foreach (var a in arms) a.Run();
            var times = new List<double>[arms.Count];
            for (int i = 0; i < arms.Count; i++) times[i] = new();
            for (int rep = 0; rep < reps; rep++)
            {
                for (int j = 0; j < arms.Count; j++)
                {
                    int i = (j + rep) % arms.Count;
                    var sw = Stopwatch.StartNew();
                    arms[i].Run();
                    sw.Stop();
                    times[i].Add(sw.Elapsed.TotalSeconds);
                }
            }
            MatMul.Q8_0Rows8Enabled = true; MatMul.Bf16Avx512Enabled = false; MatMul.Bf16FmaScratch = false;

            Console.WriteLine($"\n== {m}x{k} n={n} threads={threads} (GFLOP/s: min-time | median; ms median) ==");
            for (int i = 0; i < arms.Count; i++)
            {
                var t = times[i].OrderBy(v => v).ToList();
                double tmin = t[0], tmed = t[t.Count / 2];
                Console.WriteLine($"{arms[i].Name,-44} {flop / tmin / 1e9,8:F1} | {flop / tmed / 1e9,8:F1}   {tmed * 1e3,9:F3} ms");
            }

            NativeMemory.AlignedFree(wf); NativeMemory.AlignedFree(b); NativeMemory.AlignedFree(c);
            if (wq != null) NativeMemory.AlignedFree(wq);
            if (xq != null) NativeMemory.AlignedFree(xq);
            if (wb != null) NativeMemory.AlignedFree(wb);
        }
    }
}

static unsafe T* Alloc<T>(long count) where T : unmanaged => (T*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(T)), 64);

static bool Has(string name) => Environment.GetCommandLineArgs().Contains(name);
static string ArgStr(string name, string def)
{
    var a = Environment.GetCommandLineArgs();
    int i = Array.IndexOf(a, name);
    return i >= 0 && i + 1 < a.Length ? a[i + 1] : def;
}
static int Arg(string name, int def) => int.Parse(ArgStr(name, def.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
static int[] ArgList(string name, string def) => ArgStr(name, def).Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
