using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Engine.Evaluation;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Benchmarks.Profile;

/// <summary>
/// Issue #1109 (CPU workstream): which numeric format each CPU operation should use.
/// Stopwatch harness with same-session round-robin interleave (one sample of every arm per round),
/// so background load (other agents' GPU jobs competing for UMA bandwidth on Strix, P/E-core
/// scheduling on Meteor Lake) hits every arm alike. Report medians and min..max ranges.
///
/// <code>
/// dotnet DotLLM.Benchmarks.dll precision-cpu isa
/// dotnet DotLLM.Benchmarks.dll precision-cpu gemv [--threads 1,6] [--samples 9]
/// dotnet DotLLM.Benchmarks.dll precision-cpu gemv-prod [--threads 1,12]   (production entry points; any ISA incl. SSE4.2)
/// dotnet DotLLM.Benchmarks.dll precision-cpu gemm [--threads 6] [--n 64]
/// dotnet DotLLM.Benchmarks.dll precision-cpu attn [--threads 6] [--ctx 4096,16384]
/// dotnet DotLLM.Benchmarks.dll precision-cpu kl-act &lt;gguf&gt; [--windows 4] [--len 256] [--threads 6]
/// dotnet DotLLM.Benchmarks.dll precision-cpu kl-kv &lt;gguf&gt; [--ctx 4096] [--decode 64] [--threads 6]
/// </code>
/// ISA tiers that the JIT fixes at startup are compared across processes with the runtime knobs
/// <c>DOTNET_EnableAVXVNNI=0</c> / <c>DOTNET_EnableAVX512F=0</c>; the header prints what is live.
/// </summary>
internal static unsafe class PrecisionCpuProfile
{
    private static ComputeThreadPool? _pool;
    private static int _samples = 9;
    private static int _threads = 1;
    private static long _coldBytes = 768L << 20;
    private static string _corpus = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache", "corpora", "datasets", "wiki.test.lf.raw");

    // ───────────────────────── entry ─────────────────────────

    public static int Run(string[] args)
    {
        string mode = args.Length > 0 ? args[0] : "isa";
        string? model = args.Length > 1 && !args[1].StartsWith("--", StringComparison.Ordinal) ? args[1] : null;
        int[] threads = [1];
        int n = 64, windows = 4, len = 256, decode = 64;
        int[] ctxs = [4096, 16384];
        string[] arms = [];
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--threads": threads = args[++i].Split(',').Select(int.Parse).ToArray(); break;
                case "--samples": _samples = int.Parse(args[++i]); break;
                case "--cold-mb": _coldBytes = long.Parse(args[++i]) << 20; break;
                case "--n": n = int.Parse(args[++i]); break;
                case "--windows": windows = int.Parse(args[++i]); break;
                case "--len": len = int.Parse(args[++i]); break;
                case "--decode": decode = int.Parse(args[++i]); break;
                case "--ctx": ctxs = args[++i].Split(',').Select(int.Parse).ToArray(); break;
                case "--arms": arms = args[++i].Split(','); break;
                case "--corpus": _corpus = args[++i]; break;
            }
        }

        PrintIsa();
        if (mode == "isa") return 0;

        foreach (int t in threads)
        {
            _threads = t;
            _pool?.Dispose();
            _pool = t > 1 ? new ComputeThreadPool(t) : null;
            switch (mode)
            {
                case "gemv": GemvSuite(); break;
                case "gemv-prod": GemvProdSuite(); break;
                case "gemm": GemmSuite(n); break;
                case "attn": foreach (int c in ctxs) AttnSuite(c); break;
                case "kl-act": KlActivations(model!, windows, len, t); break;
                case "kl-kv": foreach (int c in ctxs) KlKv(model!, c, decode, t, arms); break;
                default: Console.WriteLine($"unknown mode {mode}"); return 1;
            }
        }

        _pool?.Dispose();
        return 0;
    }

    private static void PrintIsa()
    {
        Console.WriteLine($"precision-cpu  date={DateTime.Now:s}  machine={Environment.MachineName}  logical={Environment.ProcessorCount}  runtime={Environment.Version}");
        Console.WriteLine($"  Avx2={Avx2.IsSupported} Fma={Fma.IsSupported} F16C(Avx2 implies)  AvxVnni={AvxVnni.IsSupported}  Avx512F={Avx512F.IsSupported} Avx512BW={Avx512BW.IsSupported}  Avx10v1={Avx10v1.IsSupported} Avx10v1.V512={Avx10v1.V512.IsSupported}");
        Console.WriteLine($"  env: DOTNET_EnableAVXVNNI={Environment.GetEnvironmentVariable("DOTNET_EnableAVXVNNI") ?? "-"} DOTNET_EnableAVX512F={Environment.GetEnvironmentVariable("DOTNET_EnableAVX512F") ?? "-"}");
        Console.WriteLine();
    }

    // ───────────────────────── timing helpers ─────────────────────────

    private static double Median(List<double> v)
    {
        var s = v.OrderBy(x => x).ToList();
        return s.Count % 2 == 1 ? s[s.Count / 2] : 0.5 * (s[s.Count / 2 - 1] + s[s.Count / 2]);
    }

    private sealed class Arm
    {
        public string Name = "";
        public long WeightBytes;      // bytes streamed per call (one instance)
        public int Instances = 1;     // distinct copies cycled (cold = DRAM-bound)
        public Action<int> Run = null!;
        public Action Free = () => { };
        public double Flops;          // per call
    }

    /// <summary>Round-robin: every round runs each arm for ~40 ms, so drift hits all arms equally.</summary>
    private static List<double>[] Interleave(Arm[] arms)
    {
        var calls = new int[arms.Length];
        var cursor = new int[arms.Length];
        for (int a = 0; a < arms.Length; a++)
        {
            for (int w = 0; w < 4; w++) arms[a].Run(w % arms[a].Instances);
            var sw = Stopwatch.StartNew();
            int c = 0;
            while (sw.Elapsed.TotalMilliseconds < 20) arms[a].Run(c++ % arms[a].Instances);
            calls[a] = Math.Max(2, (int)(15 / (sw.Elapsed.TotalMilliseconds / c)));
        }

        var times = arms.Select(_ => new List<double>()).ToArray();
        for (int s = 0; s < _samples; s++)
        {
            for (int a = 0; a < arms.Length; a++)
            {
                var sw = Stopwatch.StartNew();
                for (int c = 0; c < calls[a]; c++) arms[a].Run(cursor[a]++ % arms[a].Instances);
                times[a].Add(sw.Elapsed.TotalMilliseconds * 1e6 / calls[a]);   // ns per call
            }
        }

        return times;
    }

    private static void Report(string title, Arm[] arms, List<double>[] times, string? baselineName)
    {
        int bi = baselineName is null ? -1 : Array.FindIndex(arms, a => a.Name == baselineName);
        double baseMed = bi >= 0 ? Median(times[bi]) : double.NaN;
        double baseMin = bi >= 0 ? times[bi].Min() : double.NaN;
        Console.WriteLine($"### {title}  threads={_threads} samples={_samples}");
        Console.WriteLine("| arm | weight MB | µs/call median | µs min..max | GB/s (weights, at min) | GFLOP/s (at min) | vs baseline (median) | vs baseline (min) |");
        Console.WriteLine("|---|---:|---:|---|---:|---:|---:|---:|");
        for (int a = 0; a < arms.Length; a++)
        {
            double med = Median(times[a]), mn = times[a].Min();
            string rel = bi >= 0 ? $"{baseMed / med:F2}x" : "";
            string relMin = bi >= 0 ? $"{baseMin / mn:F2}x" : "";
            Console.WriteLine($"| {arms[a].Name} | {arms[a].WeightBytes / 1048576.0:F1} | {med / 1000:F0} | {mn / 1000:F0}..{times[a].Max() / 1000:F0} | {arms[a].WeightBytes / mn:F1} | {arms[a].Flops / mn:F1} | {rel} | {relMin} |");
        }

        Console.WriteLine();
    }

    private static nint Alloc(long bytes) => (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);

    private static void Free(nint p) => NativeMemory.AlignedFree((void*)p);

    // ───────────────────────── weight generation ─────────────────────────

    /// <summary>Gaussian-ish weights with a few heavy rows, quantized with the repo's own quantizers.</summary>
    private static float[] MakeF32(int m, int k, int seed)
    {
        var rng = new Random(seed);
        var w = new float[(long)m * k];
        for (long i = 0; i < w.LongLength; i++)
            w[i] = (rng.NextSingle() + rng.NextSingle() + rng.NextSingle() - 1.5f) * 0.05f;
        return w;
    }

    private static nint Encode(float[] w, int m, int k, QuantizationType qt, out long bytes)
    {
        fixed (float* src = w)
        {
            switch (qt)
            {
                case QuantizationType.F32:
                    bytes = (long)m * k * 4;
                    nint f = Alloc(bytes);
                    Buffer.MemoryCopy(src, (void*)f, bytes, bytes);
                    return f;
                case QuantizationType.F16:
                    bytes = (long)m * k * 2;
                    nint h = Alloc(bytes);
                    TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(src, m * k), new Span<Half>((void*)h, m * k));
                    return h;
                case QuantizationType.BF16:
                    bytes = (long)m * k * 2;
                    nint bf = Alloc(bytes);
                    for (long i = 0; i < (long)m * k; i++)
                    {
                        uint u = BitConverter.SingleToUInt32Bits(src[i]);
                        u += 0x7FFFu + ((u >> 16) & 1u);
                        ((ushort*)bf)[i] = (ushort)(u >> 16);
                    }

                    return bf;
                case QuantizationType.Q8_0:
                    bytes = (long)m * k / 32 * 34;
                    nint q8 = Alloc(bytes);
                    MatMul.QuantizeF32ToQ8_0(src, (byte*)q8, m * k);
                    return q8;
                case QuantizationType.Q4_0:
                    bytes = (long)m * k / 32 * 18;
                    nint q4 = Alloc(bytes);
                    EncodeQ4_0(src, (byte*)q4, (long)m * k);
                    return q4;
                case QuantizationType.Q4_K:
                    bytes = (long)m * k / 256 * 144;
                    nint qk = Alloc(bytes);
                    Quantize.FromFloat32(new ReadOnlySpan<float>(src, m * k), (long)m * k, QuantizationType.Q4_K, new Span<byte>((void*)qk, checked((int)bytes)));
                    return qk;
                default: throw new NotSupportedException(qt.ToString());
            }
        }
    }

    /// <summary>GGUF Q4_0 encoder (llama.cpp <c>quantize_row_q4_0_ref</c>): <c>d = max/-8</c>, nibble = round(x/d)+8, low nibbles hold elements 0..15.</summary>
    private static void EncodeQ4_0(float* src, byte* dst, long count)
    {
        for (long b = 0; b < count / 32; b++)
        {
            float* x = src + b * 32;
            byte* o = dst + b * 18;
            float amax = 0, max = 0;
            for (int i = 0; i < 32; i++)
            {
                if (MathF.Abs(x[i]) > amax) { amax = MathF.Abs(x[i]); max = x[i]; }
            }

            float d = max / -8f, id = d != 0 ? 1f / d : 0f;
            *(Half*)o = (Half)d;
            for (int i = 0; i < 16; i++)
            {
                int lo = Math.Min(15, (int)(x[i] * id + 8.5f));
                int hi = Math.Min(15, (int)(x[i + 16] * id + 8.5f));
                o[2 + i] = (byte)(lo | (hi << 4));
            }
        }
    }

    // ───────────────────────── GEMV (decode) ─────────────────────────

    /// <summary>A 4-row kernel over prepared activation <c>x</c> (F32 or a quantized block form).</summary>
    private delegate void Rows4(byte* w, nint rowStride, byte* x, int k, float* res);

    private struct GemvCtx
    {
        public byte* W;
        public nint RowStride;
        public byte* X;
        public float* Y;
        public int M;
        public int K;
        public delegate*<byte*, nint, byte*, int, float*, void> Fn;
    }

    private static void GemvWorker(nint p, int ti, int tc)
    {
        ref GemvCtx c = ref *(GemvCtx*)p;
        ComputeThreadPool.PartitionRange(c.M / 4, ti, tc, out int s, out int e);
        for (int g = s; g < e; g++)
            c.Fn(c.W + (long)g * 4 * c.RowStride, c.RowStride, c.X, c.K, c.Y + g * 4);
    }

    private static void RunGemv(delegate*<byte*, nint, byte*, int, float*, void> fn, byte* w, nint rowStride, byte* x, float* y, int m, int k)
    {
        var ctx = new GemvCtx { W = w, RowStride = rowStride, X = x, Y = y, M = m, K = k, Fn = fn };
        if (_pool is null) GemvWorker((nint)(&ctx), 0, 1);
        else _pool.Dispatch((nint)(&ctx), &GemvWorker);
    }

    // 4-row adapters (k in elements)
    private static void KF32(byte* w, nint rs, byte* x, int k, float* r)
    {
        for (int j = 0; j < 4; j++)
            r[j] = TensorPrimitives.Dot(new ReadOnlySpan<float>(w + j * rs, k), new ReadOnlySpan<float>(x, k));
    }

    private static void KF16(byte* w, nint rs, byte* x, int k, float* r)
    {
        // Production GemvF16 row kernel is not exposed per-row; widen 4 rows through a stack tile.
        float* tile = stackalloc float[512];
        Vector4Acc(w, rs, x, k, r, tile);
    }

    private static void Vector4Acc(byte* w, nint rs, byte* x, int k, float* r, float* tile)
    {
        for (int j = 0; j < 4; j++)
        {
            float acc = 0;
            Half* h = (Half*)(w + j * rs);
            for (int c = 0; c < k; c += 512)
            {
                int len = Math.Min(512, k - c);
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(h + c, len), new Span<float>(tile, len));
                acc += TensorPrimitives.Dot(new ReadOnlySpan<float>(tile, len), new ReadOnlySpan<float>((float*)x + c, len));
            }

            r[j] = acc;
        }
    }

    private static void KBf16(byte* w, nint rs, byte* x, int k, float* r) => MatMul.VecDotBf16F32Avx2_4Rows(w, rs, (float*)x, k, r);

    private static void KQ8F32(byte* w, nint rs, byte* x, int k, float* r) => MatMul.VecDotQ8_0F32Avx2_4Rows(w, rs, (float*)x, k / 32, r);

    private static void KQ4F32(byte* w, nint rs, byte* x, int k, float* r) => MatMul.VecDotQ4_0F32Avx2_4Rows(w, rs, (float*)x, k / 32, r);

    private static void KQ8Avx2(byte* w, nint rs, byte* x, int k, float* r) => MatMul.VecDotQ8_0Avx2_4Rows(w, w + rs, w + 2 * rs, w + 3 * rs, x, k / 32, r);

    private static void KQ8Vnni(byte* w, nint rs, byte* x, int k, float* r) => MatMul.VecDotQ8_0VnniZp_4Rows(w, w + rs, w + 2 * rs, w + 3 * rs, x, k / 32, r);

    private static void KQ8Avx512(byte* w, nint rs, byte* x, int k, float* r) => MatMul.VecDotQ8_0Avx512_4Rows(w, w + rs, w + 2 * rs, w + 3 * rs, x, k / 32, r);

    private static void KQ4Q81(byte* w, nint rs, byte* x, int k, float* r) => MatMul.VecDotLegacyQuantAvx2_4Rows(QuantizationType.Q4_0, w, rs, x, k / 32, r);

    private static void KQ4K(byte* w, nint rs, byte* x, int k, float* r) => MatMul.VecDotQ4_K_Q8_K_4Rows(w, w + rs, w + 2 * rs, w + 3 * rs, x, k / 256, r);

    private enum Prep { F32, Q8_0, Q8_1, Q8_K }

    private static Arm GemvArm(string name, QuantizationType qt, Prep prep, delegate*<byte*, nint, byte*, int, float*, void> fn,
        int m, int k, bool cold, float[] master, float* x, Action? before = null, Action? after = null)
    {
        nint w0 = Encode(master, m, k, qt, out long bytes);
        int inst = cold ? (int)Math.Clamp(_coldBytes / bytes, 2, 256) : 1;
        var ws = new nint[inst];
        ws[0] = w0;
        for (int i = 1; i < inst; i++)
        {
            ws[i] = Alloc(bytes);
            Buffer.MemoryCopy((void*)w0, (void*)ws[i], bytes, bytes);
        }

        nint rs = (nint)(bytes / m);
        nint xq = Alloc((long)k * 4 + 1024);
        nint y = Alloc((long)m * 4 + 64);
        return new Arm
        {
            Name = name,
            WeightBytes = bytes,
            Instances = inst,
            Flops = 2.0 * m * k,
            Run = i =>
            {
                before?.Invoke();
                byte* xp = (byte*)x;
                switch (prep)   // activation quantization is part of the cost of an integer arm
                {
                    case Prep.Q8_0: MatMul.QuantizeF32ToQ8_0(x, (byte*)xq, k); xp = (byte*)xq; break;
                    case Prep.Q8_1: MatMul.QuantizeF32ToQ8_1(x, (byte*)xq, k); xp = (byte*)xq; break;
                    case Prep.Q8_K: MatMul.QuantizeF32ToQ8_K(x, (byte*)xq, k); xp = (byte*)xq; break;
                }

                RunGemv(fn, (byte*)ws[i], rs, xp, (float*)y, m, k);
                after?.Invoke();
            },
            Free = () =>
            {
                foreach (var p in ws) Free(p);
                Free(xq);
                Free(y);
            },
        };
    }

    private static void GemvSuite()
    {
        // Llama-3.2-3B shapes: attn q/o 3072x3072, ffn gate/up 8192x3072, ffn down 3072x8192.
        (int M, int K, string Label)[] shapes = [(3072, 3072, "attn q/o 3072x3072"), (8192, 3072, "ffn gate/up 8192x3072"), (3072, 8192, "ffn down 3072x8192")];
        foreach (bool cold in new[] { false, true })
        {
            foreach (var (m, k, label) in shapes)
            {
                float[] master = MakeF32(m, k, m + k);
                float* x = (float*)Alloc((long)k * 4);
                var rng = new Random(5);
                for (int i = 0; i < k; i++) x[i] = (rng.NextSingle() - 0.5f) * 2f;

                var arms = new List<Arm>
                {
                    GemvArm("F32 w x F32 a (reference)", QuantizationType.F32, Prep.F32, &KF32, m, k, cold, master, x),
                    GemvArm("F16 w x F32 a (convert tile)", QuantizationType.F16, Prep.F32, &KF16, m, k, cold, master, x),
                    GemvArm("BF16 w x F32 a (shift widen)", QuantizationType.BF16, Prep.F32, &KBf16, m, k, cold, master, x),
                    GemvArm("Q8_0 w x F32 a (in-reg dequant FMA)", QuantizationType.Q8_0, Prep.F32, &KQ8F32, m, k, cold, master, x),
                    GemvArm("Q8_0 w x Q8_0 a AVX2 maddubs", QuantizationType.Q8_0, Prep.Q8_0, &KQ8Avx2, m, k, cold, master, x),
                };
                if (AvxVnni.IsSupported)
                    arms.Add(GemvArm("Q8_0 w x Q8_0 a AVX-VNNI vpdpbusd", QuantizationType.Q8_0, Prep.Q8_0, &KQ8Vnni, m, k, cold, master, x));
                if (Avx512BW.IsSupported)
                    arms.Add(GemvArm("Q8_0 w x Q8_0 a AVX-512BW maddubs", QuantizationType.Q8_0, Prep.Q8_0, &KQ8Avx512, m, k, cold, master, x));
                arms.Add(GemvArm("Q4_0 w x F32 a (in-reg dequant FMA)", QuantizationType.Q4_0, Prep.F32, &KQ4F32, m, k, cold, master, x));
                arms.Add(GemvArm("Q4_0 w x Q8_1 a AVX2 maddubs", QuantizationType.Q4_0, Prep.Q8_1, &KQ4Q81, m, k, cold, master, x,
                    () => MatMul.LegacyVnniEnabled = false));
                if (AvxVnni.IsSupported)
                    arms.Add(GemvArm("Q4_0 w x Q8_1 a AVX-VNNI (proto)", QuantizationType.Q4_0, Prep.Q8_1, &KQ4Q81, m, k, cold, master, x,
                        () => MatMul.LegacyVnniEnabled = true, () => MatMul.LegacyVnniEnabled = false));
                arms.Add(GemvArm($"Q4_K w x Q8_K a (prod 4-row{(AvxVnni.IsSupported ? ", VNNI scales" : "")})", QuantizationType.Q4_K, Prep.Q8_K, &KQ4K, m, k, cold, master, x));

                var arr = arms.ToArray();
                var times = Interleave(arr);
                Report($"GEMV {label} [{(cold ? $"COLD: {arr.Max(a => a.Instances)}+ copies cycled, DRAM-bound" : "HOT: one copy")}]", arr, times, "F32 w x F32 a (reference)");
                foreach (var a in arr) a.Free();
                Free((nint)x);
            }
        }
    }

    // ───────────────────────── GEMV through production entry points (portable) ─────────────────────────

    /// <summary>BF16 × F32, 4 rows, 128-bit (SSE4.1 <c>pmovzxwd</c> + shift): the widen any x86-64 since Penryn can do.</summary>
    private static void KBf16Sse(byte* w, nint rs, byte* x, int k, float* r)
    {
        for (int j = 0; j < 4; j++)
        {
            ushort* row = (ushort*)(w + j * rs);
            var acc = System.Runtime.Intrinsics.Vector128<float>.Zero;
            for (int i = 0; i < k; i += 4)
            {
                var wv = Sse2.ShiftLeftLogical(Sse41.ConvertToVector128Int32(row + i), 16).AsSingle();
                acc = Sse.Add(acc, Sse.Multiply(wv, Sse.LoadVector128((float*)x + i)));
            }

            r[j] = System.Runtime.Intrinsics.Vector128.Sum(acc);
        }
    }

    private static Arm ProdArm(string name, QuantizationType qt, int m, int k, bool cold, float[] master, float* x, Action<nint, nint> run)
    {
        nint w0 = Encode(master, m, k, qt, out long bytes);
        int inst = cold ? (int)Math.Clamp(_coldBytes / bytes, 2, 256) : 1;
        var ws = new nint[inst];
        ws[0] = w0;
        for (int i = 1; i < inst; i++)
        {
            ws[i] = Alloc(bytes);
            Buffer.MemoryCopy((void*)w0, (void*)ws[i], bytes, bytes);
        }

        nint y = Alloc((long)m * 4 + 64);
        return new Arm
        {
            Name = name, WeightBytes = bytes, Instances = inst, Flops = 2.0 * m * k,
            Run = i => run(ws[i], y),
            Free = () => { foreach (var p in ws) Free(p); Free(y); },
        };
    }

    private static void GemvProdSuite()
    {
        (int M, int K, string Label)[] shapes = [(8192, 3072, "ffn gate/up 8192x3072"), (3072, 8192, "ffn down 3072x8192")];
        foreach (bool cold in new[] { false, true })
        {
            foreach (var (m, k, label) in shapes)
            {
                float[] master = MakeF32(m, k, m + k);
                float* x = (float*)Alloc((long)k * 4);
                var rng = new Random(5);
                for (int i = 0; i < k; i++) x[i] = (rng.NextSingle() - 0.5f) * 2f;
                var arms = new List<Arm>
                {
                    ProdArm("F32 w x F32 a GemvF32 (reference)", QuantizationType.F32, m, k, cold, master, x, (w, y) => MatMul.GemvF32((float*)w, x, (float*)y, m, k, _pool)),
                    ProdArm("F16 w x F32 a GemvF16 (prod)", QuantizationType.F16, m, k, cold, master, x, (w, y) => MatMul.GemvF16(w, x, (float*)y, m, k, _pool)),
                    ProdArm("BF16 w x F32 a GemvDequantRows (prod fallback)", QuantizationType.BF16, m, k, cold, master, x, (w, y) => MatMul.GemvDequantRows((byte*)w, QuantizationType.BF16, x, (float*)y, m, k, _pool)),
                };
                if (Sse41.IsSupported)
                    arms.Add(GemvArm("BF16 w x F32 a SSE4.1 shift widen (harness)", QuantizationType.BF16, Prep.F32, &KBf16Sse, m, k, cold, master, x));
                arms.Add(ProdArm("Q8_0 w x F32 a GemvDequantRows (float tier)", QuantizationType.Q8_0, m, k, cold, master, x, (w, y) => MatMul.GemvDequantRows((byte*)w, QuantizationType.Q8_0, x, (float*)y, m, k, _pool)));
                arms.Add(ProdArm("Q8_0 w x Q8_0 a GemvQ8_0 (prod int8)", QuantizationType.Q8_0, m, k, cold, master, x, (w, y) => MatMul.GemvQ8_0((byte*)w, x, (float*)y, m, k, _pool)));
                arms.Add(ProdArm("Q4_0 w x F32 a GemvDequantRows (float tier)", QuantizationType.Q4_0, m, k, cold, master, x, (w, y) => MatMul.GemvDequantRows((byte*)w, QuantizationType.Q4_0, x, (float*)y, m, k, _pool)));
                arms.Add(ProdArm("Q4_0 w x Q8_1 a GemvLegacyQuant (prod int8)", QuantizationType.Q4_0, m, k, cold, master, x, (w, y) => MatMul.GemvLegacyQuant((byte*)w, QuantizationType.Q4_0, x, (float*)y, m, k, _pool)));
                arms.Add(ProdArm("Q4_K w x F32 a GemvDequantRows (float tier)", QuantizationType.Q4_K, m, k, cold, master, x, (w, y) => MatMul.GemvDequantRows((byte*)w, QuantizationType.Q4_K, x, (float*)y, m, k, _pool)));
                arms.Add(ProdArm("Q4_K w x Q8_K a GemvQ4_K (prod int8)", QuantizationType.Q4_K, m, k, cold, master, x, (w, y) => MatMul.GemvQ4_K((byte*)w, x, (float*)y, m, k, _pool)));
                var arr = arms.ToArray();
                var times = Interleave(arr);
                Report($"GEMV (production entry points) {label} [{(cold ? $"COLD: {arr.Max(a => a.Instances)}+ copies cycled" : "HOT: one copy")}]", arr, times, "F32 w x F32 a GemvF32 (reference)");
                foreach (var a in arr) a.Free();
                Free((nint)x);
            }
        }
    }

    // ───────────────────────── GEMM (prefill) ─────────────────────────

    private static void GemmSuite(int n)
    {
        (int M, int K, string Label)[] shapes = [(8192, 3072, "ffn gate/up 8192x3072"), (3072, 8192, "ffn down 3072x8192")];
        foreach (var (m, k, label) in shapes)
        {
            float[] master = MakeF32(m, k, m + k);
            nint b = Alloc((long)n * k * 4), c = Alloc((long)n * m * 4);
            var rng = new Random(9);
            for (long i = 0; i < (long)n * k; i++) ((float*)b)[i] = (rng.NextSingle() - 0.5f) * 2f;
            var arms = new List<Arm>();
            Arm Make(string name, QuantizationType qt, Action<nint> run)
            {
                nint w = Encode(master, m, k, qt, out long bytes);
                return new Arm { Name = name, WeightBytes = bytes, Flops = 2.0 * m * k * n, Run = _ => run(w), Free = () => Free(w) };
            }

            arms.Add(Make("F32 w x F32 a GemmF32 (reference)", QuantizationType.F32, w => MatMul.GemmF32((float*)w, (float*)b, (float*)c, m, k, n, _pool)));
            arms.Add(Make("F16 w x F32 a GemmF16", QuantizationType.F16, w => MatMul.GemmF16(w, (float*)b, (float*)c, m, k, n, _pool)));
            arms.Add(Make("Q8_0 w x F32 a GemmDequantRows (generic float tier)", QuantizationType.Q8_0, w => MatMul.GemmDequantRows((byte*)w, QuantizationType.Q8_0, (float*)b, (float*)c, m, k, n, _pool)));
            arms.Add(Make("Q8_0 w x Q8_0 a GemmQ8_0 (prod int8)", QuantizationType.Q8_0, w => MatMul.GemmQ8_0((byte*)w, (float*)b, (float*)c, m, k, n, _pool)));
            {
                nint w = Encode(master, m, k, QuantizationType.Q8_0, out long bytes);
                var rw = WeightRepacking.RepackR4(w, QuantizationType.Q8_0, m, k);
                nint bq = Alloc((long)n * (k / 32) * 34);
                arms.Add(new Arm
                {
                    Name = "Q8_0 w x Q8_0 a R4 tiled (prod prefill path)", WeightBytes = bytes, Flops = 2.0 * m * k * n,
                    Run = _ =>
                    {
                        for (int t = 0; t < n; t++) MatMul.QuantizeF32ToQ8_0((float*)b + (long)t * k, (byte*)bq + (long)t * (k / 32) * 34, k);
                        MatMul.GemmR4TiledQ8_0((byte*)rw.Ptr, (byte*)bq, (float*)c, rw.FullGroupCount, rw.TailRows, k / 32, m, n, _pool);
                    },
                    Free = () => { Free(w); rw.Dispose(); Free(bq); },
                });
            }

            arms.Add(Make("Q4_0 w x F32 a GemmDequantRows (generic float tier)", QuantizationType.Q4_0, w => MatMul.GemmDequantRows((byte*)w, QuantizationType.Q4_0, (float*)b, (float*)c, m, k, n, _pool)));
            arms.Add(Make("Q4_0 w x Q8_1 a GemmLegacyQuant AVX2", QuantizationType.Q4_0, w => { MatMul.LegacyVnniEnabled = false; MatMul.GemmLegacyQuantOrDequant((byte*)w, QuantizationType.Q4_0, (float*)b, (float*)c, m, k, n, _pool); }));
            if (AvxVnni.IsSupported)
                arms.Add(Make("Q4_0 w x Q8_1 a GemmLegacyQuant AVX-VNNI (proto)", QuantizationType.Q4_0, w => { MatMul.LegacyVnniEnabled = true; MatMul.GemmLegacyQuantOrDequant((byte*)w, QuantizationType.Q4_0, (float*)b, (float*)c, m, k, n, _pool); MatMul.LegacyVnniEnabled = false; }));
            arms.Add(Make("Q4_K w x F32 a GemmDequantRows (generic float tier)", QuantizationType.Q4_K, w => MatMul.GemmDequantRows((byte*)w, QuantizationType.Q4_K, (float*)b, (float*)c, m, k, n, _pool)));
            arms.Add(Make("Q4_K w x Q8_K a GemmQ4_K (prod int8)", QuantizationType.Q4_K, w => MatMul.GemmQ4_K((byte*)w, (float*)b, (float*)c, m, k, n, _pool)));

            var arr = arms.ToArray();
            var times = Interleave(arr);
            Report($"GEMM {label}, n={n} tokens", arr, times, "F32 w x F32 a GemmF32 (reference)");
            foreach (var a in arr) a.Free();
            Free(b);
            Free(c);
        }
    }

    // ───────────────────────── decode attention ─────────────────────────

    private enum KvFmt { F32, F16, BF16, Q8_0, Q8_0IntQk }

    private struct AttnCtx
    {
        public float* Q;
        public byte* K;
        public byte* V;
        public float* Out;
        public int SeqKv;
        public int NumHeads;
        public int Group;
        public int HeadDim;
        public long RowBytes;      // one position's row (all kv heads)
        public long HeadBytes;     // one kv head's slice within the row
        public KvFmt Fmt;
    }

    private static void AttnWorker(nint p, int ti, int tc)
    {
        ref AttnCtx c = ref *(AttnCtx*)p;
        ComputeThreadPool.PartitionRange(c.NumHeads, ti, tc, out int s, out int e);
        const int Tile = 64;
        int hd = c.HeadDim;
        float* kt = stackalloc float[Tile * 256];
        float* vt = stackalloc float[Tile * 256];
        float* sc = stackalloc float[Tile];
        byte* qq = stackalloc byte[256 / 32 * 34];
        float scale = 1f / MathF.Sqrt(hd);
        for (int h = s; h < e; h++)
        {
            int kvh = h / c.Group;
            float* q = c.Q + h * hd;
            var qs = new ReadOnlySpan<float>(q, hd);
            var o = new Span<float>(c.Out + h * hd, hd);
            o.Clear();
            if (c.Fmt == KvFmt.Q8_0IntQk) MatMul.QuantizeF32ToQ8_0(q, qq, hd);
            float mx = float.NegativeInfinity, sum = 0;
            for (int t0 = 0; t0 < c.SeqKv; t0 += Tile)
            {
                int tl = Math.Min(Tile, c.SeqKv - t0);
                for (int j = 0; j < tl; j++)
                {
                    byte* kr = c.K + (t0 + j) * c.RowBytes + kvh * c.HeadBytes;
                    switch (c.Fmt)
                    {
                        case KvFmt.F32: sc[j] = TensorPrimitives.Dot(qs, new ReadOnlySpan<float>(kr, hd)); break;
                        case KvFmt.Q8_0IntQk: sc[j] = MatMul.VecDotQ8_0BestRow(kr, qq, hd / 32); break;
                        default:
                            Widen(kr, kt + j * hd, hd, c.Fmt);
                            sc[j] = TensorPrimitives.Dot(qs, new ReadOnlySpan<float>(kt + j * hd, hd));
                            break;
                    }

                    sc[j] *= scale;
                }

                var sspan = new Span<float>(sc, tl);
                float nm = MathF.Max(mx, TensorPrimitives.Max(sspan));
                float corr = MathF.Exp(mx - nm);
                sum *= corr;
                TensorPrimitives.Multiply(o, corr, o);
                TensorPrimitives.Subtract(sspan, nm, sspan);
                TensorPrimitives.Exp(sspan, sspan);
                sum += TensorPrimitives.Sum(sspan);
                for (int j = 0; j < tl; j++)
                {
                    byte* vr = c.V + (t0 + j) * c.RowBytes + kvh * c.HeadBytes;
                    if (c.Fmt == KvFmt.F32)
                    {
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(vr, hd), sc[j], o, o);
                    }
                    else
                    {
                        Widen(vr, vt, hd, c.Fmt == KvFmt.Q8_0IntQk ? KvFmt.Q8_0 : c.Fmt);
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(vt, hd), sc[j], o, o);
                    }
                }

                mx = nm;
            }

            TensorPrimitives.Multiply(o, 1f / sum, o);
        }
    }

    private static void Widen(byte* src, float* dst, int n, KvFmt fmt)
    {
        switch (fmt)
        {
            case KvFmt.F16: TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(src, n), new Span<float>(dst, n)); break;
            case KvFmt.BF16:
                for (int i = 0; i < n; i++) ((uint*)dst)[i] = (uint)((ushort*)src)[i] << 16;
                break;
            case KvFmt.Q8_0: KvQuantize.Q8_0ToF32(src, dst, n); break;
        }
    }

    private static void AttnSuite(int ctx)
    {
        // Llama-3.2-3B attention geometry: 24 query heads, 8 KV heads, head_dim 128.
        const int nh = 24, nkv = 8, hd = 128;
        int stride = nkv * hd;
        var rng = new Random(3);
        float[] kf = new float[(long)ctx * stride], vf = new float[(long)ctx * stride];
        for (long i = 0; i < kf.LongLength; i++)
        {
            kf[i] = (rng.NextSingle() - 0.5f) * 4f;
            vf[i] = (rng.NextSingle() - 0.5f) * 2f;
        }

        float* q = (float*)Alloc(nh * hd * 4);
        for (int i = 0; i < nh * hd; i++) q[i] = (rng.NextSingle() - 0.5f) * 2f;
        float* outRef = (float*)Alloc(nh * hd * 4), outT = (float*)Alloc(nh * hd * 4);

        (nint K, nint V, long RowBytes, long HeadBytes) Store(KvFmt f)
        {
            long rowBytes = f switch { KvFmt.F32 => stride * 4L, KvFmt.F16 or KvFmt.BF16 => stride * 2L, _ => stride / 32 * 34L };
            nint kb = Alloc(rowBytes * ctx), vb = Alloc(rowBytes * ctx);
            fixed (float* ks = kf, vs = vf)
            {
                for (int p = 0; p < ctx; p++)
                {
                    float* kr = ks + (long)p * stride, vr = vs + (long)p * stride;
                    byte* kd = (byte*)kb + p * rowBytes, vd = (byte*)vb + p * rowBytes;
                    switch (f)
                    {
                        case KvFmt.F32: Buffer.MemoryCopy(kr, kd, rowBytes, rowBytes); Buffer.MemoryCopy(vr, vd, rowBytes, rowBytes); break;
                        case KvFmt.F16:
                            TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(kr, stride), new Span<Half>(kd, stride));
                            TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(vr, stride), new Span<Half>(vd, stride));
                            break;
                        case KvFmt.BF16:
                            for (int i = 0; i < stride; i++)
                            {
                                ((ushort*)kd)[i] = (ushort)(BitConverter.SingleToUInt32Bits(kr[i]) >> 16);
                                ((ushort*)vd)[i] = (ushort)(BitConverter.SingleToUInt32Bits(vr[i]) >> 16);
                            }

                            break;
                        default: KvQuantize.F32ToQ8_0(kr, kd, stride); KvQuantize.F32ToQ8_0(vr, vd, stride); break;
                    }
                }
            }

            long headBytes = rowBytes / nkv;
            return (kb, vb, rowBytes, headBytes);
        }

        var fmts = new[] { KvFmt.F32, KvFmt.F16, KvFmt.BF16, KvFmt.Q8_0, KvFmt.Q8_0IntQk };
        var stores = fmts.Select(f => Store(f == KvFmt.Q8_0IntQk ? KvFmt.Q8_0 : f)).ToArray();
        var arms = new List<Arm>();
        for (int i = 0; i < fmts.Length; i++)
        {
            var (kb, vb, rb, hb) = stores[i];
            KvFmt f = fmts[i];
            float* dst = i == 0 ? outRef : outT;
            arms.Add(new Arm
            {
                Name = $"harness tile-attn K/V={f}", WeightBytes = rb * ctx * 2, Flops = 4.0 * nh * hd * ctx,
                Run = _ =>
                {
                    var c = new AttnCtx { Q = q, K = (byte*)kb, V = (byte*)vb, Out = dst, SeqKv = ctx, NumHeads = nh, Group = nh / nkv, HeadDim = hd, RowBytes = rb, HeadBytes = hb, Fmt = f };
                    if (_pool is null) AttnWorker((nint)(&c), 0, 1);
                    else _pool.Dispatch((nint)(&c), &AttnWorker);
                },
            });
        }

        // Production kernels for calibration: F32 Attention.Execute and the quantized-KV path (+ int8 QK prototype).
        nint pk = stores[0].K, pv = stores[0].V;
        arms.Add(new Arm
        {
            Name = "prod Attention.Execute F32 KV", WeightBytes = stride * 8L * ctx, Flops = 4.0 * nh * hd * ctx,
            Run = _ => Attention.Execute(q, (float*)pk, (float*)pv, outT, 1, ctx, nh, nkv, hd, ctx - 1, 1f / MathF.Sqrt(hd), _pool),
        });
        var qkv = new QuantizedKvCache(1, nkv, hd, ctx, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0);
        var pos = Enumerable.Range(0, ctx).ToArray();
        fixed (float* ks = kf, vs = vf)
            qkv.Update(new TensorRef(ctx, stride, DType.Float32, -1, (nint)ks), new TensorRef(ctx, stride, DType.Float32, -1, (nint)vs), pos, 0);
        long q8Bytes = stride / 32 * 34L * ctx * 2;
        arms.Add(new Arm
        {
            Name = "prod quantized-KV Execute q8_0 (dequant tile)", WeightBytes = q8Bytes, Flops = 4.0 * nh * hd * ctx,
            Run = _ => { Attention.KvInt8QkEnabled = false; Attention.Execute(q, qkv, 0, outT, 1, ctx, nh, nkv, hd, ctx - 1, _pool); },
        });
        arms.Add(new Arm
        {
            Name = "prod quantized-KV Execute q8_0 + int8 QK (proto)", WeightBytes = q8Bytes, Flops = 4.0 * nh * hd * ctx,
            Run = _ => { Attention.KvInt8QkEnabled = true; Attention.Execute(q, qkv, 0, outT, 1, ctx, nh, nkv, hd, ctx - 1, _pool); Attention.KvInt8QkEnabled = false; },
        });

        var arr = arms.ToArray();
        var times = Interleave(arr);
        Report($"decode attention, 1 layer, ctx={ctx}, 24q/8kv heads, hd=128 (WeightBytes column = K+V bytes read)", arr, times, "harness tile-attn K/V=F32");

        // Accuracy of each harness format vs the F32 arm on this (synthetic, outlier-free) data.
        arms[0].Run(0);
        float[] reference = new Span<float>(outRef, nh * hd).ToArray();
        Console.WriteLine("| arm | rel L2 err of attention output vs F32 KV |");
        Console.WriteLine("|---|---:|");
        for (int i = 1; i < fmts.Length; i++)
        {
            arms[i].Run(0);
            var t = new ReadOnlySpan<float>(outT, nh * hd);
            float[] diff = new float[nh * hd];
            TensorPrimitives.Subtract(t, reference, diff);
            Console.WriteLine($"| {arms[i].Name} | {TensorPrimitives.Norm(diff) / TensorPrimitives.Norm(reference):E2} |");
        }

        Console.WriteLine();
        qkv.Dispose();
        foreach (var (k, v, _, _) in stores) { Free(k); Free(v); }
        Free((nint)q);
        Free((nint)outRef);
        Free((nint)outT);
    }

    // ───────────────────────── accuracy: KL vs reference ─────────────────────────

    private sealed class KlStats
    {
        public readonly List<double> Kl = [];
        public int Top1Agree;
        public double NllRef;
        public double NllTest;

        public string Row(string name)
        {
            var s = Kl.OrderBy(x => x).ToArray();
            int n = s.Length;
            double Pct(double p) => s[Math.Min(n - 1, (int)(p * n))];
            double trimmed = s.Take(Math.Max(1, (int)(n * 0.99))).Average();
            double over1e2 = s.Count(x => x > 1e-2) / (double)n;
            double over1e1 = s.Count(x => x > 1e-1) / (double)n;
            return $"| {name} | {n} | {Pct(0.5):E2} | {trimmed:E2} | {s.Average():E2} | {Pct(0.99):E2} | {s[^1]:E2} | {over1e2:P2} | {over1e1:P2} | {Top1Agree / (double)n:P2} | {(NllTest - NllRef) / n:+0.00000;-0.00000} |";
        }

        public static string Header =>
            "| arm | positions | median KL | trimmed-mean KL (drop top 1%) | mean KL | p99 KL | max KL | KL>0.01 | KL>0.1 | top-1 agree | ΔNLL/token (nats) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|";
    }

    /// <summary>Accumulates KL(ref‖test), top-1 agreement and target NLL for one position.</summary>
    private static void Score(ReadOnlySpan<float> refLogits, ReadOnlySpan<float> testLogits, int target, KlStats st, double[] pRef, double[] pTest)
    {
        double lr = LogSoftmax(refLogits, pRef), lt = LogSoftmax(testLogits, pTest);
        _ = lr;
        _ = lt;
        double kl = 0;
        for (int i = 0; i < pRef.Length; i++)
        {
            double p = Math.Exp(pRef[i]);
            if (p > 0) kl += p * (pRef[i] - pTest[i]);
        }

        st.Kl.Add(Math.Max(0, kl));
        if (TensorPrimitives.IndexOfMax(refLogits) == TensorPrimitives.IndexOfMax(testLogits)) st.Top1Agree++;
        if (target >= 0)
        {
            st.NllRef -= pRef[target];
            st.NllTest -= pTest[target];
        }
    }

    private static double LogSoftmax(ReadOnlySpan<float> x, double[] dst)
    {
        double mx = TensorPrimitives.Max(x);
        double s = 0;
        for (int i = 0; i < x.Length; i++) s += Math.Exp(x[i] - mx);
        double lse = mx + Math.Log(s);
        for (int i = 0; i < x.Length; i++) dst[i] = x[i] - lse;
        return lse;
    }

    private static (IModel Model, ModelConfig Config, int[] Tokens) LoadModel(string path, int threads, int tokenCount)
    {
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        bool bos = GgufAddBosResolver.Resolve(gguf.Metadata);
        var model = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(threads));
        using var reader = new StreamReader(_corpus);
        int[] tokens = CorpusReader.StreamTokens(reader, tokenizer, tokenCount, bosTokenId: bos ? tokenizer.BosTokenId : -1).ToArray();
        Console.WriteLine($"model={Path.GetFileName(path)} arch={config.Architecture} layers={config.NumLayers} hidden={config.HiddenSize} heads={config.NumAttentionHeads}/{config.NumKvHeads} vocab={config.VocabSize} threads={threads} corpus={Path.GetFileName(_corpus)} tokens={tokens.Length}");
        return (model, config, tokens);
    }

    /// <summary>
    /// Activation-format KL: the same weights with (a) F32 activations × dequantized weights
    /// (<see cref="MatMul.ReferenceF32Activations"/>) as reference and (b) the production integer
    /// activation path (Q8_0 / Q8_1 / Q8_K). Cacheless prefill windows; every position scored.
    /// </summary>
    private static void KlActivations(string path, int windows, int len, int threads)
    {
        var (model, config, tokens) = LoadModel(path, threads, windows * len + 1);
        int vocab = config.VocabSize;
        var refLog = new float[len * vocab];
        double[] pr = new double[vocab], pt = new double[vocab];
        var arms = new (string Name, Action Set)[]
        {
            ("production integer activations (default dispatch)", () => { MatMul.ReferenceF32Activations = false; MatMul.LegacyVnniEnabled = false; }),
            ("production + legacy VNNI proto", () => { MatMul.ReferenceF32Activations = false; MatMul.LegacyVnniEnabled = true; }),
        };
        var stats = arms.Select(_ => new KlStats()).ToArray();
        var times = new double[arms.Length + 1];
        int[] positions = Enumerable.Range(0, len).ToArray();
        for (int w = 0; w < windows; w++)
        {
            var win = tokens.AsSpan(w * len, len);
            MatMul.ReferenceF32Activations = true;
            var sw = Stopwatch.StartNew();
            using (var lg = model.Forward(win, positions, -1))
                new ReadOnlySpan<float>((void*)lg.DataPointer, len * vocab).CopyTo(refLog);
            times[0] += sw.Elapsed.TotalSeconds;
            MatMul.ReferenceF32Activations = false;
            for (int a = 0; a < arms.Length; a++)
            {
                arms[a].Set();
                sw.Restart();
                using var lg = model.Forward(win, positions, -1);
                times[a + 1] += sw.Elapsed.TotalSeconds;
                var tl = new ReadOnlySpan<float>((void*)lg.DataPointer, len * vocab);
                for (int p = 0; p < len; p++)
                    Score(refLog.AsSpan(p * vocab, vocab), tl.Slice(p * vocab, vocab), tokens[w * len + p + 1], stats[a], pr, pt);
            }

            MatMul.LegacyVnniEnabled = false;
            Console.WriteLine($"  window {w + 1}/{windows} done");
        }

        Console.WriteLine($"### activation-format KL vs F32-activation reference (same weights): {Path.GetFileName(path)}, {windows}x{len} positions, threads={threads}");
        Console.WriteLine(KlStats.Header);
        for (int a = 0; a < arms.Length; a++) Console.WriteLine(stats[a].Row(arms[a].Name));
        Console.WriteLine($"prefill tok/s: reference(F32-act dequant-rows)={windows * len / times[0]:F1}  " + string.Join("  ", arms.Select((a, i) => $"{a.Name}={windows * len / times[i + 1]:F1}")));
        Console.WriteLine();
        model.Dispose();
    }

    /// <summary>
    /// KV-format KL at long context: prefill <paramref name="ctx"/> tokens into each cache format,
    /// then teacher-forced decode of <paramref name="decode"/> tokens; KL of each decode step's logits
    /// against the F32-KV run. Also reports decode tok/s at that context per format.
    /// </summary>
    private static void KlKv(string path, int ctx, int decode, int threads, string[] armFilter)
    {
        var (model, config, tokens) = LoadModel(path, threads, ctx + decode + 1);
        int vocab = config.VocabSize;
        var geom = KvGeometry.FromConfig(config);
        int total = ctx + decode;
        var arms = new List<(string Name, Func<IKvCache> Make, Action Set)>
        {
            ("f32 (reference)", () => new SimpleKvCache(geom, total), () => SimpleKvCache.SimulatedRounding = SimpleKvCache.SimulatedKvRounding.None),
            ("f16 (simulated: rounded on write)", () => new SimpleKvCache(geom, total), () => SimpleKvCache.SimulatedRounding = SimpleKvCache.SimulatedKvRounding.F16),
            ("bf16 (simulated: rounded on write)", () => new SimpleKvCache(geom, total), () => SimpleKvCache.SimulatedRounding = SimpleKvCache.SimulatedKvRounding.BF16),
            ("q8_0 (QuantizedKvCache, dequant tile)", () => new QuantizedKvCache(geom, total, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0), () => Attention.KvInt8QkEnabled = false),
            ("q8_0 + int8 QK^T (proto)", () => new QuantizedKvCache(geom, total, KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0), () => Attention.KvInt8QkEnabled = true),
            ("q4_0 (QuantizedKvCache)", () => new QuantizedKvCache(geom, total, KvCacheDType.Q4_0, KvCacheDType.Q4_0, 0), () => Attention.KvInt8QkEnabled = false),
            ("k q8_0 / v q4_0", () => new QuantizedKvCache(geom, total, KvCacheDType.Q8_0, KvCacheDType.Q4_0, 0), () => Attention.KvInt8QkEnabled = false),
        };
        if (armFilter.Length > 0) arms = arms.Where((a, i) => i == 0 || armFilter.Any(f => a.Name.StartsWith(f, StringComparison.Ordinal))).ToList();

        float[] refLog = new float[decode * vocab];
        double[] pr = new double[vocab], pt = new double[vocab];
        var results = new List<string>();
        var speed = new List<string>();
        const int Chunk = 512;
        for (int a = 0; a < arms.Count; a++)
        {
            arms[a].Set();
            using IKvCache cache = arms[a].Make();
            var sw = Stopwatch.StartNew();
            for (int s0 = 0; s0 < ctx; s0 += Chunk)
            {
                int l = Math.Min(Chunk, ctx - s0);
                int[] pos = Enumerable.Range(s0, l).ToArray();
                using var _ = model.Forward(tokens.AsSpan(s0, l), pos, -1, cache);
            }

            double prefillS = sw.Elapsed.TotalSeconds;
            var st = new KlStats();
            sw.Restart();
            for (int d = 0; d < decode; d++)
            {
                int p = ctx + d;
                using var lg = model.Forward(tokens.AsSpan(p, 1), [p], -1, cache);
                var row = new ReadOnlySpan<float>((void*)lg.DataPointer, vocab);
                if (a == 0) row.CopyTo(refLog.AsSpan(d * vocab, vocab));
                else Score(refLog.AsSpan(d * vocab, vocab), row, tokens[p + 1], st, pr, pt);
            }

            double decodeS = sw.Elapsed.TotalSeconds;
            long kvBytesPerTok = cache switch
            {
                QuantizedKvCache q => Enumerable.Range(0, config.NumLayers).Sum(l => (long)q.KeyQuantizedRowBytesOf(l) + q.ValueQuantizedRowBytesOf(l)),
                _ => Enumerable.Range(0, config.NumLayers).Sum(l => (long)geom.KvStrideOf(l) * 2 * (arms[a].Name.StartsWith("f32", StringComparison.Ordinal) ? 4 : 2)),
            };
            speed.Add($"| {arms[a].Name} | {kvBytesPerTok:N0} | {kvBytesPerTok * (long)ctx / 1048576.0:F0} | {ctx / prefillS:F1} | {decode / decodeS:F2} |");
            if (a > 0) results.Add(st.Row(arms[a].Name));
            Console.WriteLine($"  arm {arms[a].Name}: prefill {prefillS:F1}s decode {decodeS:F1}s");
        }

        SimpleKvCache.SimulatedRounding = SimpleKvCache.SimulatedKvRounding.None;
        Attention.KvInt8QkEnabled = false;
        Console.WriteLine($"### KV-format KL vs F32 KV: {Path.GetFileName(path)}, ctx={ctx}, {decode} teacher-forced decode steps, threads={threads}");
        Console.WriteLine(KlStats.Header);
        foreach (var r in results) Console.WriteLine(r);
        Console.WriteLine();
        Console.WriteLine("| KV format | KV bytes/token (all layers; simulated rows show the narrow format's size) | KV MiB at ctx | prefill tok/s | decode tok/s at ctx |");
        Console.WriteLine("|---|---:|---:|---:|---:|");
        foreach (var s in speed) Console.WriteLine(s);
        Console.WriteLine();
        model.Dispose();
    }
}
