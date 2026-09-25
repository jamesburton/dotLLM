using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Cpu.Threading;

namespace DotLLM.Benchmarks;

/// <summary>
/// Microbenchmarks for <see cref="ComputeThreadPool.Dispatch"/> coordination cost.
/// Motivated by a PerfView kernel CPU profile showing ~76% of dotLLM CPU burned in
/// the worker spin loop during decode. These benchmarks isolate the three cost
/// components so we can tell whether the loss is per-dispatch barrier cost,
/// inter-dispatch idle spin, or worker wake latency.
///
/// <para>Benchmark design:</para>
/// <list type="number">
/// <item>
///   <description>
///     <see cref="Dispatch_NoWork"/> — <c>Dispatch</c> with a no-op kernel.
///     Measures pure round-trip: wake N workers, execute empty fn, barrier.
///     Decode gap between dispatches in real workloads is dominated by this.
///   </description>
/// </item>
/// <item>
///   <description>
///     <see cref="Dispatch_SmallWork"/> — <c>Dispatch</c> with ~1 µs of actual
///     SIMD-shape work per thread (sum a fixed-size buffer). Mirrors one row
///     of a SmolLM-135M-size decode matmul. The ratio of this vs
///     <see cref="SingleThreaded_SmallWork"/> tells us whether the pool is
///     helping at this problem size or only hurting.
///   </description>
/// </item>
/// <item>
///   <description>
///     <see cref="SingleThreaded_SmallWork"/> — the same work on the caller
///     thread only, no dispatch. Fixed-cost baseline.
///   </description>
/// </item>
/// <item>
///   <description>
///     <see cref="DispatchBurst_DecodePattern"/> — 30 back-to-back small-work
///     dispatches. Mirrors one SmolLM decode step (30 layers × ~1 matmul
///     dispatch per layer). Lets us see whether batching amortises the
///     spin-fallback cost or whether each dispatch pays it again.
///   </description>
/// </item>
/// </list>
///
/// <para>All timings reported by BenchmarkDotNet include barrier overhead —
/// the point of the microbench is to make that overhead visible, not to hide
/// it. Comparing 1-thread runs to N-thread runs at the same work size shows
/// the dispatch floor.</para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class ThreadPoolDispatchBenchmarks : IDisposable
{
    /// <summary>
    /// Per-thread work buffer length. 576 × 4 bytes matches a SmolLM-135M hidden
    /// row (F32 view for a scalar reduction — we're measuring coordination cost,
    /// not kernel quality, so a plain sum keeps the worker work simple and
    /// predictable.)
    /// </summary>
    private const int WorkBufferLen = 576;

    /// <summary>
    /// Back-to-back dispatch count in <see cref="DispatchBurst_DecodePattern"/>.
    /// 30 matches SmolLM-135M layer count; other Llama-family decode-layer loops
    /// sit in a similar 20–40 range.
    /// </summary>
    private const int BurstCount = 30;

    private ComputeThreadPool? _pool;
    private float* _work;

    [Params(2, 4, 8, 16, 32)]
    public int Threads { get; set; }

    [Params(DispatchMode.EventBased, DispatchMode.SpinWait)]
    public DispatchMode Mode { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _pool = new ComputeThreadPool(Threads);
        _pool.SetDispatchMode(Mode);

        _work = (float*)NativeMemory.AlignedAlloc((nuint)(WorkBufferLen * sizeof(float)), 64);
        var rng = new Random(42);
        for (int i = 0; i < WorkBufferLen; i++)
            _work[i] = rng.NextSingle();

        // Warm the pool so the first measured iteration isn't paying for
        // first-touch pin + event initialisation.
        for (int i = 0; i < 8; i++)
            _pool.Dispatch((nint)_work, &NoOpWorker);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree(_work);
        _pool?.Dispose();
    }

    public void Dispose() => Cleanup();

    /// <summary>Dispatch a no-op to every thread — pure barrier cost.</summary>
    [Benchmark(Description = "Dispatch(no-op)")]
    public void Dispatch_NoWork()
    {
        _pool!.Dispatch((nint)_work, &NoOpWorker);
    }

    /// <summary>Dispatch ~576-element sum per thread — barrier plus realistic row work.</summary>
    [Benchmark(Description = "Dispatch(~1µs work)")]
    public void Dispatch_SmallWork()
    {
        _pool!.Dispatch((nint)_work, &SmallWorkWorker);
    }

    /// <summary>Same small work, executed on the caller thread only (no dispatch).</summary>
    [Benchmark(Description = "Single-thread same work")]
    public float SingleThreaded_SmallWork()
    {
        return SumBuffer(_work, WorkBufferLen);
    }

    /// <summary>30 back-to-back dispatches — one SmolLM-135M decode step's layer loop.</summary>
    [Benchmark(Description = "30× Dispatch(small work)")]
    public void DispatchBurst_DecodePattern()
    {
        var pool = _pool!;
        for (int i = 0; i < BurstCount; i++)
            pool.Dispatch((nint)_work, &SmallWorkWorker);
    }

    // ---- kernel functions (static, pointer-dispatched) ----

    private static void NoOpWorker(nint context, int threadIdx, int threadCount)
    {
        // Intentionally empty — measures pure dispatch barrier.
        _ = context; _ = threadIdx; _ = threadCount;
    }

    private static void SmallWorkWorker(nint context, int threadIdx, int threadCount)
    {
        float* buf = (float*)context;
        int chunk = WorkBufferLen / threadCount;
        int start = threadIdx * chunk;
        int end = threadIdx == threadCount - 1 ? WorkBufferLen : start + chunk;
        float sum = SumBuffer(buf + start, end - start);
        // Prevent the JIT from eliding the sum. Writing back into the buffer
        // is cheap and keeps the partition realistic.
        if (sum == float.NegativeInfinity) buf[start] = sum;
    }

    private static float SumBuffer(float* buf, int len)
    {
        float s = 0f;
        for (int i = 0; i < len; i++)
            s += buf[i];
        return s;
    }
}
