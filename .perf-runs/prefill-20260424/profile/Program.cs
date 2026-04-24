using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Core.Configuration;

const int Q8_0BlockBytes = 34;
const int Q8_0GroupSize = 32;

// SmolLM-135M shapes: hidden=576, ffn=1536, heads=9, kv=3, headDim=64, vocab=49152, layers=30
const int Hidden = 576;
const int QDim = 576;       // 9 * 64
const int KvDim = 192;      // 3 * 64
const int FfnDim = 1536;
const int Vocab = 49152;
const int Layers = 30;

int N = args.Length > 0 && int.TryParse(args[0], out var n) ? n : 512;
int Threads = args.Length > 1 && int.TryParse(args[1], out var t) ? t : Environment.ProcessorCount;

var pool = new ComputeThreadPool(Threads, topology: null, ThreadingConfig.Auto);
pool.SetDispatchMode(DispatchMode.EventBased);

// Warm up the pool — spin up workers and let the OS schedule them before any measurement.
unsafe
{
    nint wwarm = (nint)NativeMemory.AlignedAlloc(1024 * 1024, 64);
    var inw = new float[Hidden * N];
    var outw = new float[Hidden * N];
    fixed (float* bp = inw)
    fixed (float* cp = outw)
        for (int i = 0; i < 5; i++)
            MatMul.GemmQ8_0((byte*)wwarm, bp, cp, Hidden, Hidden, N, pool);
    NativeMemory.AlignedFree((void*)wwarm);
}

Console.WriteLine($"Prefill kernel breakdown: SmolLM-135M Q8_0, N={N}, Threads={Threads}");
Console.WriteLine(new string('-', 80));
Console.WriteLine($"  {"name",-12} {"m",6} {"k",6} {"n",4}  {"ms/call",8}  {"GFLOPS",7}  {"x30",10}");
Console.WriteLine(new string('-', 80));

double totalMsPerLayer = 0;
void Bench(string name, int m, int k, int nTok, int reps, bool inPerLayer)
{
    unsafe
    {
        nint w = AllocQ8(m, k);
        var input = AllocF32(nTok * k);
        var output = new float[(long)nTok * m];

        fixed (float* bp = input)
        fixed (float* cp = output)
        {
            // Warm up (3 calls) to let JIT tier-up complete and cache settle.
            for (int i = 0; i < 3; i++)
                MatMul.GemmQ8_0((byte*)w, bp, cp, m, k, nTok, pool);
            var sw = Stopwatch.StartNew();
            for (int r = 0; r < reps; r++)
                MatMul.GemmQ8_0((byte*)w, bp, cp, m, k, nTok, pool);
            sw.Stop();
            double msPerCall = sw.Elapsed.TotalMilliseconds / reps;
            double gflops = (2.0 * m * k * nTok) / (msPerCall / 1000.0) / 1e9;
            double xLayers = inPerLayer ? msPerCall * Layers : msPerCall;
            Console.WriteLine($"  {name,-12} {m,6} {k,6} {nTok,4}  {msPerCall,8:F2}  {gflops,7:F1}  {xLayers,10:F1} ms");
            if (inPerLayer) totalMsPerLayer += msPerCall;
        }
        NativeMemory.AlignedFree((void*)w);
    }
}

// Warmup the parallel path on a realistic shape — the *first* kernel measured
// after pool setup consistently runs ~5× slower than subsequent calls on this
// host, so we run a throwaway first before timing anything.
Bench("warmup",  FfnDim, Hidden, N, 10, false);
Bench("Q",       QDim,   Hidden, N, 30, true);
Bench("K",       KvDim,  Hidden, N, 30, true);
Bench("V",       KvDim,  Hidden, N, 30, true);
Bench("O",       Hidden, QDim,   N, 30, true);
Bench("Gate",    FfnDim, Hidden, N, 30, true);
Bench("Up",      FfnDim, Hidden, N, 30, true);
Bench("Down",    Hidden, FfnDim, N, 30, true);

Console.WriteLine();
double totalGemmTime = totalMsPerLayer * Layers;
Console.WriteLine($"Sum of MatMul time per layer: {totalMsPerLayer,7:F2} ms  (x {Layers} = {totalGemmTime,7:F1} ms for all layers)");

Console.WriteLine();
Bench("lm_head", Vocab, Hidden, N, 2, false);

// Single-threaded comparisons for the same shapes — pool=null takes the single-thread path.
Console.WriteLine();
Console.WriteLine("Single-threaded (pool=null) same shapes — compare GFLOPS to parallel above:");
void BenchST(string name, int m, int k, int nTok, int reps)
{
    unsafe
    {
        nint w = AllocQ8(m, k);
        var input = AllocF32(nTok * k);
        var output = new float[(long)nTok * m];
        fixed (float* bp = input)
        fixed (float* cp = output)
        {
            for (int i = 0; i < 3; i++)
                MatMul.GemmQ8_0((byte*)w, bp, cp, m, k, nTok, null);
            var sw = Stopwatch.StartNew();
            for (int r = 0; r < reps; r++)
                MatMul.GemmQ8_0((byte*)w, bp, cp, m, k, nTok, null);
            sw.Stop();
            double msPerCall = sw.Elapsed.TotalMilliseconds / reps;
            double gflops = (2.0 * m * k * nTok) / (msPerCall / 1000.0) / 1e9;
            Console.WriteLine($"  {name,-12} {m,6} {k,6} {nTok,4}  {msPerCall,8:F2}  {gflops,7:F1}");
        }
        NativeMemory.AlignedFree((void*)w);
    }
}
BenchST("Q-1T",     QDim,   Hidden, N, 3);
BenchST("K-1T",     KvDim,  Hidden, N, 3);
BenchST("Gate-1T",  FfnDim, Hidden, N, 3);
BenchST("Down-1T",  Hidden, FfnDim, N, 3);

Console.WriteLine();
Console.WriteLine($"Expected pp{N} wall from MatMul alone (ignoring attention, RMSNorm, RoPE, SiLU):");
Console.WriteLine($"   {totalGemmTime:F0} ms → {(N / (totalGemmTime / 1000.0)):F1} tok/s");

pool.Dispose();

static unsafe nint AllocQ8(int m, int k)
{
    int blockCount = k / Q8_0GroupSize;
    int rowBytes = blockCount * Q8_0BlockBytes;
    nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)m * rowBytes), 64);
    var rng = new Random(42);
    byte* p = (byte*)ptr;
    for (long r = 0; r < m; r++)
    {
        for (int b = 0; b < blockCount; b++)
        {
            *(Half*)p = (Half)(rng.NextSingle() * 0.1f);
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(p + 2))[i] = (sbyte)rng.Next(-127, 128);
            p += Q8_0BlockBytes;
        }
    }
    return ptr;
}

static float[] AllocF32(int elems)
{
    var a = new float[elems];
    var rng = new Random(7);
    for (int i = 0; i < elems; i++) a[i] = rng.NextSingle() * 2 - 1;
    return a;
}
