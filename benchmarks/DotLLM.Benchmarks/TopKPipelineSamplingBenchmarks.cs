using BenchmarkDotNet.Attributes;
using DotLLM.Core.Configuration;
using DotLLM.Engine.Samplers;

namespace DotLLM.Benchmarks;

/// <summary>
/// Benchmarks end-to-end sampled decoding overhead for common top-K requests.
/// The legacy path composes temperature + top-K masking + full-vocab categorical sampling;
/// the auto path uses the bounded top-K categorical fast path.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class TopKPipelineSamplingBenchmarks
{
    private float[] _srcLogits = null!;
    private float[] _scratch = null!;
    private SamplerPipeline _legacy = null!;
    private SamplerPipeline _auto = null!;

    [Params(32_000, 128_000)]
    public int VocabSize { get; set; }

    [Params(40, 100)]
    public int TopK { get; set; }

    [Params(0.8f)]
    public float Temperature { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _srcLogits = new float[VocabSize];
        _scratch = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++)
            _srcLogits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);

        _legacy = new SamplerPipeline(
            processors: null,
            steps: [new TemperatureSampler(Temperature), new TopKSampler(TopK)],
            seed: 123);

        _auto = new SamplerPipeline(new InferenceOptions
        {
            Temperature = Temperature,
            TopK = TopK,
            Seed = 123
        });
    }

    [Benchmark(Baseline = true)]
    public int Legacy_TopKMask_FullCategorical()
    {
        _srcLogits.AsSpan().CopyTo(_scratch);
        return _legacy.Sample(_scratch, []);
    }

    [Benchmark]
    public int Auto_BoundedTopKCategorical()
    {
        _srcLogits.AsSpan().CopyTo(_scratch);
        return _auto.Sample(_scratch, []);
    }
}
