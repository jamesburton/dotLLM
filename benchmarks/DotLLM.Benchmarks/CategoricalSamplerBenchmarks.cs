using BenchmarkDotNet.Attributes;
using DotLLM.Engine.Samplers;

namespace DotLLM.Benchmarks;

/// <summary>
/// Benchmarks categorical sampling over a full vocabulary-sized logit vector.
/// Each invocation copies source logits because sampling mutates the input path.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class CategoricalSamplerBenchmarks
{
    private float[] _srcLogits = null!;
    private float[] _scratch = null!;
    private Random _rng = null!;

    [Params(32_000, 128_000)]
    public int VocabSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _srcLogits = new float[VocabSize];
        _scratch = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++)
            _srcLogits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);

        _rng = new Random(123);
    }

    [Benchmark]
    public int Categorical_Sample()
    {
        _srcLogits.AsSpan().CopyTo(_scratch);
        return CategoricalSampler.Sample(_scratch, _rng);
    }
}
