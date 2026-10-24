using BenchmarkDotNet.Attributes;
using System.Buffers;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;

namespace DotLLM.Benchmarks;

/// <summary>
/// Benchmarks repetition penalty over full-vocabulary logits with realistic recent-token windows.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class RepetitionPenaltyBenchmarks
{
    private float[] _srcLogits = null!;
    private float[] _scratch = null!;
    private int[] _previousTokens = null!;
    private RepetitionPenaltyProcessor _processor = null!;
    private ProcessorContext _context;

    [Params(32_000, 128_000)]
    public int VocabSize { get; set; }

    [Params(64, 256)]
    public int Window { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _srcLogits = new float[VocabSize];
        _scratch = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++)
            _srcLogits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);

        _previousTokens = new int[Window];
        for (int i = 0; i < _previousTokens.Length; i++)
            _previousTokens[i] = rng.Next(VocabSize);

        _processor = new RepetitionPenaltyProcessor();
        _context = new ProcessorContext(
            RepetitionPenalty: 1.15f,
            RepetitionPenaltyWindow: Window,
            SequenceId: 0);
    }

    [Benchmark(Baseline = true)]
    public void Current_StackWindow()
    {
        _srcLogits.AsSpan().CopyTo(_scratch);
        _processor.Process(_scratch, _previousTokens, _context);
    }

    [Benchmark]
    public void Legacy_RentedWindow()
    {
        _srcLogits.AsSpan().CopyTo(_scratch);
        ApplyLegacyRented(_scratch, _previousTokens, _context);
    }

    private static void ApplyLegacyRented(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context)
    {
        float penalty = context.RepetitionPenalty;
        if (penalty == 1.0f || previousTokens.Count == 0)
            return;

        int window = context.RepetitionPenaltyWindow;
        int startIndex = window > 0 ? Math.Max(0, previousTokens.Count - window) : 0;
        int windowLength = previousTokens.Count - startIndex;

        int[] rented = ArrayPool<int>.Shared.Rent(windowLength);
        try
        {
            for (int i = 0; i < windowLength; i++)
                rented[i] = previousTokens[startIndex + i];

            Array.Sort(rented, 0, windowLength);

            int prev = -1;
            for (int i = 0; i < windowLength; i++)
            {
                int tokenId = rented[i];
                if (tokenId == prev)
                    continue;
                prev = tokenId;

                if ((uint)tokenId >= (uint)logits.Length)
                    continue;

                if (logits[tokenId] > 0f)
                    logits[tokenId] /= penalty;
                else
                    logits[tokenId] *= penalty;
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }
}
