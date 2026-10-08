using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Kernels.Experimental;
using DotLLM.Cpu.Threading;

namespace DotLLM.Benchmarks.Profile;

/// <summary>
/// Issue #832 (exploratory): LittleBit factorized-linear GEMV vs the production Q4_K / I2_S / PQ2_0 CPU GEMV.
/// Stopwatch harness (BenchmarkDotNet cannot express "N distinct weight copies cycled, all formats interleaved
/// per sample, thread-pool dispatch" without fighting its process model). Same-session round-robin interleave
/// per sample, median over samples.
///
/// Usage: dotnet run -c Release -- littlebit-bench [micro|stack|body] [--threads 1,16] [--samples 7]
///        [--shapes 4096x4096,8192x28672] [--cold-mb 512]
/// </summary>
internal static unsafe class LittleBitBenchProfile
{
    private static ComputeThreadPool? _pool;
    private static LittleBitScratch _scratch = null!;
    private static int _samples = 7;
    private static int _decode = 1;
    private static long _coldBytes = 512L << 20;

    private sealed class Op
    {
        public string Name = "";
        public long Bytes;                 // weight bytes streamed per GEMV
        public int DOut, DIn, NInst;
        public Action<int, nint, nint> Run = null!;
        public Action Free = () => { };
        public double BitsPerWeight => Bytes * 8.0 / ((double)DOut * DIn);
    }

    private sealed class Fmt
    {
        public string Name = "";
        public Func<int, int, int, Op> Make = null!;   // dOut, dIn, nInst
    }

    // ───────────────────────── format factories ─────────────────────────

    private static Fmt LittleBit(string label, int? rank, double? bpw, LittleBitKernel kernel, bool residual = true) => new()
    {
        Name = label,
        Make = (dOut, dIn, nInst) =>
        {
            int r = rank ?? RankForBpw(dOut, dIn, bpw!.Value, residual ? 2 : 1);
            var rng = new Random(1234 + dOut + dIn + r);
            var tmpl = new LittleBitLayer(Enumerable.Range(0, residual ? 2 : 1).Select(_ => LittleBitPath.Random(dOut, dIn, r, rng)).ToArray());
            var layers = new LittleBitLayer[nInst];
            layers[0] = tmpl;
            for (int i = 1; i < nInst; i++) layers[i] = tmpl.Clone();
            return new Op
            {
                Name = $"{label} (r={r}, {tmpl.BitsPerWeight:F3} bpw)", Bytes = tmpl.WeightBytes, DOut = dOut, DIn = dIn, NInst = nInst,
                Run = (i, x, y) => layers[i].Gemv((float*)x, (float*)y, _scratch, _pool, kernel),
                Free = () => { foreach (var l in layers) l.Dispose(); },
            };
        },
    };

    /// <summary>
    /// Rank so that the bytes actually stored (1 bit per sign, FP16 scales: paths x [r*a + 16(a+r)] bits, a=dOut+dIn)
    /// hit the target bpw. NOTE: the "2r(a+1)" paper formula (issue text) gives roughly HALF this rank
    /// (r=270 for 0.55 bpw at 4096x4096) - with 1-bit signs that rank stores only ~0.29 bpw. See the review note.
    /// </summary>
    private static int RankForBpw(int dOut, int dIn, double bpw, int paths)
    {
        double a = dOut + dIn;
        double r = (bpw * dOut * dIn / paths - 16 * a) / (a + 16);
        return Math.Max(1, (int)Math.Round(r));
    }

    private static nint AllocFilled(long bytes, nint template)
    {
        nint p = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        Buffer.MemoryCopy((void*)template, (void*)p, bytes, bytes);
        return p;
    }

    private static Fmt Raw(string name, QuantizationType qt, bool repack = false) => new()
    {
        Name = name,
        Make = (m, k, nInst) =>
        {
            var rng = new Random(99 + m + k);
            long rowBytes = qt switch
            {
                QuantizationType.Q4_K => (long)k / 256 * 144,
                QuantizationType.PQ2_0 => (long)k / 128 * QuantFormat.PQ2_0BlockBytes,
                _ => (long)k / 4,   // I2_S
            };
            long bytes = rowBytes * m + (qt == QuantizationType.I2_S ? 4 : 0);
            nint tmpl = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
            rng.NextBytes(new Span<byte>((void*)tmpl, checked((int)bytes)));
            // sane fp16 scales so no NaN/denormal slow paths
            if (qt == QuantizationType.Q4_K)
                for (long b = 0; b < (long)m * k / 256; b++) { var p = (Half*)((byte*)tmpl + b * 144); p[0] = (Half)0.01f; p[1] = (Half)0.01f; }
            else if (qt == QuantizationType.PQ2_0)
                for (long b = 0; b < (long)m * k / 128; b++) *(Half*)((byte*)tmpl + b * QuantFormat.PQ2_0BlockBytes) = (Half)0.02f;
            else
                *(float*)((byte*)tmpl + (long)m * k / 4) = 0.02f;

            if (repack)
            {
                var rw0 = WeightRepacking.RepackR4(tmpl, qt, m, k);
                NativeMemory.AlignedFree((void*)tmpl);
                int sb = k / 256;
                var ptrs = new nint[nInst];
                ptrs[0] = rw0.Ptr;
                for (int i = 1; i < nInst; i++) ptrs[i] = AllocFilled(rw0.AllocatedBytes, rw0.Ptr);
                nint xq = (nint)NativeMemory.AlignedAlloc((nuint)(sb * MatMul.Q8_K_BlockBytes), 64);
                return new Op
                {
                    Name = name, Bytes = rw0.AllocatedBytes, DOut = m, DIn = k, NInst = nInst,
                    Run = (i, x, y) =>
                    {
                        MatMul.QuantizeF32ToQ8_K((float*)x, (byte*)xq, k);
                        MatMul.ComputeRowsQ4_KInterleaved((byte*)ptrs[i], (byte*)xq, (float*)y, rw0.FullGroupCount, rw0.TailRows, sb, _pool);
                    },
                    Free = () => { foreach (var p in ptrs) NativeMemory.AlignedFree((void*)p); NativeMemory.AlignedFree((void*)xq); },
                };
            }

            var w = new nint[nInst];
            w[0] = tmpl;
            for (int i = 1; i < nInst; i++) w[i] = AllocFilled(bytes, tmpl);
            return new Op
            {
                Name = name, Bytes = bytes, DOut = m, DIn = k, NInst = nInst,
                Run = qt switch
                {
                    QuantizationType.Q4_K => (i, x, y) => MatMul.GemvQ4_K((byte*)w[i], (float*)x, (float*)y, m, k, _pool),
                    QuantizationType.PQ2_0 => (i, x, y) => MatMul.GemvPQ2_0((byte*)w[i], (float*)x, (float*)y, m, k, _pool),
                    _ => (i, x, y) => MatMul.GemvI2_S((byte*)w[i], (float*)x, (float*)y, m, k, _pool),
                },
                Free = () => { foreach (var p in w) NativeMemory.AlignedFree((void*)p); },
            };
        },
    };

    // ───────────────────────── timing ─────────────────────────

    private static double Median(List<double> v) { v.Sort(); return v.Count % 2 == 1 ? v[v.Count / 2] : 0.5 * (v[v.Count / 2 - 1] + v[v.Count / 2]); }

    private static void FillX(nint x, int n, int seed)
    {
        var rng = new Random(seed);
        for (int i = 0; i < n; i++) ((float*)x)[i] = rng.NextSingle() * 2 - 1;
    }

    private static void MicroRun(string title, int dOut, int dIn, Fmt[] fmts, bool cold)
    {
        var ops = fmts.Select(f =>
        {
            long perInst = f.Make(dOut, dIn, 1) is { } probe ? Probe(probe) : 0;
            int n = cold ? (int)Math.Clamp(_coldBytes / Math.Max(perInst, 1), 1, 8192) : 1;
            return f.Make(dOut, dIn, n);
        }).ToArray();
        nint x = (nint)NativeMemory.AlignedAlloc((nuint)(dIn * 4 + 64), 64), y = (nint)NativeMemory.AlignedAlloc((nuint)(dOut * 4 + 64), 64);
        FillX(x, dIn, 7);
        var calls = new int[ops.Length];
        var cursor = new int[ops.Length];
        for (int o = 0; o < ops.Length; o++)
        {
            for (int w = 0; w < 6; w++) ops[o].Run(w % ops[o].NInst, x, y);
            var sw = Stopwatch.StartNew();
            int c = 0;
            while (sw.Elapsed.TotalMilliseconds < 25) { ops[o].Run(c++ % ops[o].NInst, x, y); }
            double per = sw.Elapsed.TotalMilliseconds / c;
            calls[o] = Math.Max(8, (int)(60 / per));
        }
        var times = ops.Select(_ => new List<double>()).ToArray();
        for (int s = 0; s < _samples; s++)
            for (int o = 0; o < ops.Length; o++)
            {
                var sw = Stopwatch.StartNew();
                for (int c = 0; c < calls[o]; c++) ops[o].Run(cursor[o]++ % ops[o].NInst, x, y);
                times[o].Add(sw.Elapsed.TotalMilliseconds * 1e6 / calls[o]);   // ns per call
            }
        Console.WriteLine($"### {title}  [{(cold ? "COLD: cycling " + ops.Max(o => o.NInst) + "+ distinct copies" : "HOT: one copy, repeated")}]  threads={_decode} (SpinWait pool)");
        Console.WriteLine("| format | weight B/token | bpw | ns/call (median) | min..max | ns/row | GB/s of weight bytes | MB footprint |");
        Console.WriteLine("|---|---:|---:|---:|---|---:|---:|---:|");
        for (int o = 0; o < ops.Length; o++)
        {
            double med = Median(times[o]);
            Console.WriteLine($"| {ops[o].Name} | {ops[o].Bytes:N0} | {ops[o].BitsPerWeight:F3} | {med:F0} | {times[o].Min():F0}..{times[o].Max():F0} | {med / dOut:F2} | {ops[o].Bytes / med:F1} | {ops[o].Bytes * ops[o].NInst / 1048576.0:F0} |");
        }
        Console.WriteLine();
        foreach (var op in ops) op.Free();
        NativeMemory.AlignedFree((void*)x); NativeMemory.AlignedFree((void*)y);
    }

    private static long Probe(Op op) { long b = op.Bytes; op.Free(); return b; }

    // ───────────────────────── end-to-end stacks ─────────────────────────

    private static void Norm(nint buf, int n)
    {
        var s = new Span<float>((void*)buf, n);
        float ss = TensorPrimitives.SumOfSquares(s);
        float inv = ss > 0 ? 1f / MathF.Sqrt(ss / n) : 1f;
        TensorPrimitives.Multiply(s, inv, s);
    }

    /// <summary>Layer = list of (shape, input buffer id, output buffer id); buffers are filled with the chained activations.</summary>
    private sealed record Lin(int DOut, int DIn, int In, int Out);

    private static void StackRun(string title, Lin[] layerPlan, int layers, Fmt[] fmts)
    {
        // buffers: 0 h(4096-ish max) 1 a 2 b 3 c 4 h1 5 gate 6 up (sized to max)
        int maxDim = layerPlan.Max(l => Math.Max(l.DOut, l.DIn));
        var bufs = Enumerable.Range(0, 8).Select(_ => (nint)NativeMemory.AlignedAlloc((nuint)(maxDim * 4 + 64), 64)).ToArray();
        foreach (var b in bufs) FillX(b, maxDim, 5);
        var stacks = new List<Op[]>();
        foreach (var f in fmts)
        {
            var ops = new Op[layers * layerPlan.Length];
            for (int l = 0; l < layers; l++)
                for (int j = 0; j < layerPlan.Length; j++)
                    ops[l * layerPlan.Length + j] = f.Make(layerPlan[j].DOut, layerPlan[j].DIn, 1);
            stacks.Add(ops);
        }
        void Token(Op[] ops)
        {
            for (int l = 0; l < layers; l++)
                for (int j = 0; j < layerPlan.Length; j++)
                {
                    var p = layerPlan[j];
                    ops[l * layerPlan.Length + j].Run(0, bufs[p.In], bufs[p.Out]);
                    Norm(bufs[p.Out], p.DOut);
                }
        }
        foreach (var ops in stacks) { Token(ops); Token(ops); }
        var times = stacks.Select(_ => new List<double>()).ToArray();
        const int tokensPerSample = 3;
        for (int s = 0; s < _samples; s++)
            for (int i = 0; i < stacks.Count; i++)
            {
                var sw = Stopwatch.StartNew();
                for (int t = 0; t < tokensPerSample; t++) Token(stacks[i]);
                times[i].Add(sw.Elapsed.TotalMilliseconds / tokensPerSample);
            }
        // reference: normalisation cost alone (identical across formats)
        var swN = Stopwatch.StartNew();
        for (int t = 0; t < 200; t++) for (int l = 0; l < layers; l++) foreach (var p in layerPlan) Norm(bufs[p.Out], p.DOut);
        double normMs = swN.Elapsed.TotalMilliseconds / 200;
        Console.WriteLine($"### {title}  threads={_decode} (SpinWait pool); {layers} layers x {layerPlan.Length} linears = {layers * layerPlan.Length} GEMVs/token; normalise-between-linears overhead (same for all) = {normMs:F2} ms/token");
        Console.WriteLine("| format | weight GB/token | ms/token (median) | min..max | tok/s (projection-only) | effective GB/s |");
        Console.WriteLine("|---|---:|---:|---|---:|---:|");
        for (int i = 0; i < stacks.Count; i++)
        {
            double gb = stacks[i].Sum(o => o.Bytes) / 1e9;
            double med = Median(times[i]);
            Console.WriteLine($"| {stacks[i][0].Name.Split(" (r=")[0]}{(stacks[i][0].Name.Contains("(r=") ? " (r by bpw)" : "")} | {gb:F3} | {med:F2} | {times[i].Min():F2}..{times[i].Max():F2} | {1000 / med:F1} | {gb * 1000 / med:F1} |");
        }
        Console.WriteLine();
        foreach (var ops in stacks) foreach (var o in ops) o.Free();
        foreach (var b in bufs) NativeMemory.AlignedFree((void*)b);
    }

    // ───────────────────────── entry ─────────────────────────

    public static int Run(string[] args)
    {
        string mode = args.Length > 0 && !args[0].StartsWith("--") ? args[0] : "micro";
        int[] threads = [1, 16];
        (int, int)[] shapes = [(4096, 4096), (8192, 28672)];
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--threads") threads = args[++i].Split(',').Select(int.Parse).ToArray();
            else if (args[i] == "--samples") _samples = int.Parse(args[++i]);
            else if (args[i] == "--cold-mb") _coldBytes = long.Parse(args[++i]) << 20;
            else if (args[i] == "--shapes") shapes = args[++i].Split(',').Select(s => s.Split('x')).Select(p => (int.Parse(p[0]), int.Parse(p[1]))).ToArray();
        }
        Console.WriteLine($"littlebit-bench mode={mode} samples={_samples} cpu_logical={Environment.ProcessorCount} avx2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} avxvnni={LittleBitLayer.VnniSupported} date={DateTime.Now:s}");
        _scratch = new LittleBitScratch(28672, 1200);

        var micro = new[]
        {
            LittleBit("LB f32 r43", 43, null, LittleBitKernel.Avx2Float), LittleBit("LB f32 r270", 270, null, LittleBitKernel.Avx2Float), LittleBit("LB f32 r500", 500, null, LittleBitKernel.Avx2Float),
            LittleBit("LB vnni r43", 43, null, LittleBitKernel.VnniInt8), LittleBit("LB vnni r270", 270, null, LittleBitKernel.VnniInt8), LittleBit("LB vnni r500", 500, null, LittleBitKernel.VnniInt8),
            LittleBit("LB f32 r546(~0.55 true bpw @4096^2)", 546, null, LittleBitKernel.Avx2Float), LittleBit("LB vnni r546", 546, null, LittleBitKernel.VnniInt8),
            Raw("Q4_K plain GemvQ4_K", QuantizationType.Q4_K), Raw("Q4_K R4 (engine decode path)", QuantizationType.Q4_K, repack: true),
            Raw("I2_S GemvI2_S", QuantizationType.I2_S), Raw("PQ2_0 GemvPQ2_0", QuantizationType.PQ2_0),
        };
        var bodyFmts = new[]
        {
            LittleBit("LB f32 0.55bpw", null, 0.55, LittleBitKernel.Avx2Float), LittleBit("LB f32 0.1bpw", null, 0.10, LittleBitKernel.Avx2Float),
            LittleBit("LB vnni 0.55bpw", null, 0.55, LittleBitKernel.VnniInt8), LittleBit("LB vnni 0.1bpw", null, 0.10, LittleBitKernel.VnniInt8),
            Raw("Q4_K R4 (engine decode path)", QuantizationType.Q4_K, repack: true), Raw("I2_S GemvI2_S", QuantizationType.I2_S), Raw("PQ2_0 GemvPQ2_0", QuantizationType.PQ2_0),
        };

        foreach (int t in threads)
        {
            _pool?.Dispose();
            _decode = t;
            // Production decode config: pool sized to all logical CPUs, SetDispatchMode(SpinWait) at seqLen==1, which caps the
            // active threads at the decode thread count (here t; engine default without topology = 8).
            _pool = t > 1 ? new ComputeThreadPool(Environment.ProcessorCount, topology: null, new ThreadingConfig(Environment.ProcessorCount, DecodeThreadCount: t)) : null;
            _pool?.SetDispatchMode(DispatchMode.SpinWait);
            if (mode == "micro")
                foreach (var (m, k) in shapes)
                    foreach (bool cold in new[] { false, true })
                        MicroRun($"GEMV {m}x{k} (d_out x d_in), LittleBit = primary+residual", m, k, micro, cold);
            else if (mode == "stack")
                StackRun("stack: 32 x (4096x4096) linears, activations chained", [new Lin(4096, 4096, 0, 1)], 32, bodyFmts);
            else if (mode == "body")
                StackRun("7B-body: 32 layers x [q,k,v,o 4096x4096 + gate,up 11008x4096 + down 4096x11008]; excludes embed/lm_head/attention/norm",
                    [new Lin(4096, 4096, 0, 1), new Lin(4096, 4096, 0, 2), new Lin(4096, 4096, 0, 3), new Lin(4096, 4096, 1, 4),
                     new Lin(11008, 4096, 4, 5), new Lin(11008, 4096, 4, 6), new Lin(4096, 11008, 6, 0)], 32, bodyFmts);
        }
        _pool?.Dispose();
        return 0;
    }
}
