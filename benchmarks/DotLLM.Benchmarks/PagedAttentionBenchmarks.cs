using BenchmarkDotNet.Attributes;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Engine.KvCache;

namespace DotLLM.Benchmarks;

/// <summary>
/// One decode token of attention over a fragmented paged cache (#1268): the legacy gather of the history into F32 staging
/// followed by contiguous attention, versus the paged kernel reading the blocks in place, per block row type.
/// 16 query heads / 4 KV heads, head dim 64. Adapted from Cheongroach's continuous-batching prototype.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class PagedAttentionBenchmarks
{
    private const int Heads = 16;
    private const int KvHeads = 4;
    private const int Dim = 64;

    private PagedKvCacheFactory _factory = null!;
    private PagedKvCache _cache = null!;
    private PagedKvCache _blocker = null!;
    private UnmanagedTensor _query = null!;
    private UnmanagedTensor _output = null!;
    private ComputeThreadPool? _pool;
    private nint _k;
    private nint _v;

    /// <summary>Cached context length.</summary>
    [Params(512, 4096, 16384)]
    public int ContextLength { get; set; }

    /// <summary>Block row type: f32 (the pre-#1268 pool), f16 (new default) or q8_0.</summary>
    [Params("f32", "f16", "q8_0")]
    public string Rows { get; set; } = "f16";

    /// <summary>Worker threads (1 = caller thread only).</summary>
    [Params(1, 8)]
    public int Threads { get; set; }

    /// <summary>Builds a deterministic fragmented cache before measuring.</summary>
    [GlobalSetup]
    public void Setup()
    {
        DType type = Rows switch { "f32" => DType.Float32, "q8_0" => DType.Q8_0, _ => DType.Float16 };
        _pool = Threads > 1 ? new ComputeThreadPool(Threads) : null;
        _factory = new PagedKvCacheFactory(1, KvHeads, Dim, 16, 2 * ContextLength + 16, type);
        _cache = _factory.Create(ContextLength);
        _blocker = _factory.Create(ContextLength);
        _query = UnmanagedTensor.Allocate(new TensorShape(1, Heads * Dim), DType.Float32);
        _output = UnmanagedTensor.Allocate(_query.Shape, DType.Float32);
        using var data = UnmanagedTensor.Allocate(new TensorShape(ContextLength, KvHeads * Dim), DType.Float32);
        var rng = new Random(123);
        var values = new Span<float>((void*)data.DataPointer, (int)data.ElementCount);
        var q = new Span<float>((void*)_query.DataPointer, (int)_query.ElementCount);
        for (int i = 0; i < values.Length; i++) values[i] = rng.NextSingle() * 2 - 1;
        for (int i = 0; i < q.Length; i++) q[i] = rng.NextSingle() * 2 - 1;

        // Alternate blocks with another sequence so logical and physical order differ.
        int[] positions = Enumerable.Range(0, ContextLength).ToArray();
        for (int pos = 0; pos < ContextLength; pos += 16)
        {
            var rows = new TensorRef(16, KvHeads * Dim, DType.Float32, -1, data.DataPointer + (nint)pos * KvHeads * Dim * sizeof(float));
            _cache.Update(rows, rows, positions.AsSpan(pos, 16), 0);
            _blocker.Update(rows, rows, positions.AsSpan(pos, 16), 0);
        }

        // Allocate (and fill) the legacy staging in setup, not the timed region.
        _k = _cache.GetKeysRef(0).DataPointer;
        _v = _cache.GetValuesRef(0).DataPointer;
    }

    /// <summary>Pre-#1268 path: gather (decode) both KV tensors into F32 staging, then contiguous attention.</summary>
    [Benchmark(Baseline = true)]
    public void GatherAndAttend()
    {
        var k = _cache.GetKeysRef(0);
        var v = _cache.GetValuesRef(0);
        Attention.Execute((float*)_query.DataPointer, (float*)k.DataPointer, (float*)v.DataPointer,
            (float*)_output.DataPointer, 1, ContextLength, Heads, KvHeads, Dim, ContextLength - 1, _pool);
    }

    /// <summary>Contiguous attention over the already-gathered staging (the gather-free lower bound for F32).</summary>
    [Benchmark]
    public void AttendContiguous()
        => Attention.Execute((float*)_query.DataPointer, (float*)_k, (float*)_v,
            (float*)_output.DataPointer, 1, ContextLength, Heads, KvHeads, Dim, ContextLength - 1, _pool);

    /// <summary>Paged kernel reading the blocks in place (#1268).</summary>
    [Benchmark]
    public void AttendPaged()
        => Attention.Execute((float*)_query.DataPointer, _cache, 0, (float*)_output.DataPointer,
            1, Heads, KvHeads, Dim, ContextLength - 1, 1f / MathF.Sqrt(Dim), _pool);

    /// <summary>Releases all host storage.</summary>
    [GlobalCleanup]
    public void Cleanup()
    {
        _cache.Dispose();
        _blocker.Dispose();
        _factory.Dispose();
        _query.Dispose();
        _output.Dispose();
        _pool?.Dispose();
    }
}
