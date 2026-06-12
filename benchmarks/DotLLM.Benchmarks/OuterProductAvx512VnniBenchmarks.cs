using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Benchmarks;

/// <summary>
/// A/B comparison of the AVX-512 Q8_0 4×6 outer-product microkernel: the baseline
/// <c>maddubs</c>-on-256-bit-halves reduction (<see cref="MatMul.OuterProductQ8_0Avx512_4x6"/>)
/// versus the AVX-512 VNNI variant (<see cref="MatMul.OuterProductQ8_0Avx512Vnni_4x6"/>,
/// VPDPBUSD-512 via <c>AvxVnni.V512</c>), over a realistic deep-K tile (4 rows × 6 tokens × K=4096).
/// </summary>
/// <remarks>
/// The VNNI method is compiled only on net11.0 (<c>AvxVnni.V512</c> is a .NET 11 API). On net10.0
/// the VNNI benchmark is excluded; only the baseline is present (BenchmarkDotNet emits a runtime
/// warning about a lone baseline, which is harmless when the suite is not actually run here).
/// Run on AVX512-VNNI hardware (Zen5 / Strix Halo) with:
/// <c>dotnet run -c Release -f net11.0 -- --filter '*OuterProductAvx512Vnni*'</c>.
/// </remarks>
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class OuterProductAvx512VnniBenchmarks : IDisposable
{
    // Deep, realistic K: 128 blocks × 32 = 4096.
    private const int K = 4096;
    private const int Q8_0GroupSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int BlockCount = K / Q8_0GroupSize; // 128
    private const int M = 4;                           // exactly one R4 group (the 4×6 tile width)
    private const int N = 6;                           // the 4×6 tile token count

    // R4-repacked Q8_0 weights (4 rows).
    private nint _repackedWeights;
    private nint _originalWeights;

    // 6 pre-quantized Q8_0 token streams, contiguous.
    private nint _inputQ8;
    private int _rowBytes;

    // Output buffer: N tokens × M rows.
    private float* _output;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _rowBytes = BlockCount * Q8_0BlockBytes;

        // Row-major weights, then repack to R4.
        _originalWeights = (nint)NativeMemory.AlignedAlloc((nuint)((long)M * _rowBytes), 64);
        byte* p = (byte*)_originalWeights;
        for (int row = 0; row < M; row++)
            FillRandomQ8(ref p, rng);

        var rw = WeightRepacking.RepackR4(_originalWeights, QuantizationType.Q8_0, M, K);
        _repackedWeights = rw.Ptr;

        // 6 contiguous Q8_0 token streams.
        _inputQ8 = (nint)NativeMemory.AlignedAlloc((nuint)((long)N * _rowBytes), 64);
        byte* q = (byte*)_inputQ8;
        for (int t = 0; t < N; t++)
            FillRandomQ8(ref q, rng);

        _output = (float*)NativeMemory.AlignedAlloc((nuint)(N * M * sizeof(float)), 64);
    }

    private void FillRandomQ8(ref byte* ptr, Random rng)
    {
        for (int b = 0; b < BlockCount; b++)
        {
            *(Half*)ptr = (Half)(rng.NextSingle() * 0.1f);
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(ptr + 2))[i] = (sbyte)rng.Next(-127, 128);
            ptr += Q8_0BlockBytes;
        }
    }

    /// <summary>
    /// Baseline: AVX-512 4×6 tile with the <c>maddubs</c>+<c>madd(ones)</c> reduction on 256-bit halves.
    /// </summary>
    [Benchmark(Baseline = true)]
    public void Avx512_4x6_Maddubs()
    {
        byte* groupBase = (byte*)_repackedWeights;
        byte* x = (byte*)_inputQ8;
        MatMul.OuterProductQ8_0Avx512_4x6(
            groupBase,
            x, x + _rowBytes, x + 2 * _rowBytes,
            x + 3 * _rowBytes, x + 4 * _rowBytes, x + 5 * _rowBytes,
            _output, BlockCount, M);
    }

#if NET11_0_OR_GREATER
    /// <summary>
    /// AVX-512 VNNI: same 4×6 tile, single VPDPBUSD-512 per cell (<c>AvxVnni.V512</c>) replacing the
    /// <c>maddubs</c> pair. Drops the <c>ones256</c> register and the per-half <c>prod</c> temporaries.
    /// </summary>
    [Benchmark]
    public void Avx512_4x6_Vnni()
    {
        byte* groupBase = (byte*)_repackedWeights;
        byte* x = (byte*)_inputQ8;
        MatMul.OuterProductQ8_0Avx512Vnni_4x6(
            groupBase,
            x, x + _rowBytes, x + 2 * _rowBytes,
            x + 3 * _rowBytes, x + 4 * _rowBytes, x + 5 * _rowBytes,
            _output, BlockCount, M);
    }

    /// <summary>
    /// AVX-512 VNNI <b>zero-point</b>: same 4×6 tile, single VPDPBUSD-512 per cell, but using the
    /// <c>+128</c> compensation method (<c>u = x + 128</c> unsigned) instead of the per-cell sign trick.
    /// The <c>u512</c>/<c>w512</c> packs and signed weight sums are hoisted out of the per-cell loop
    /// (<c>u512</c> per token, <c>w512</c>+<c>sw</c> per row), with no <c>Avx2.Sign</c> in the hot path.
    /// </summary>
    [Benchmark]
    public void Avx512_4x6_VnniZp()
    {
        byte* groupBase = (byte*)_repackedWeights;
        byte* x = (byte*)_inputQ8;
        MatMul.OuterProductQ8_0Avx512VnniZp_4x6(
            groupBase,
            x, x + _rowBytes, x + 2 * _rowBytes,
            x + 3 * _rowBytes, x + 4 * _rowBytes, x + 5 * _rowBytes,
            _output, BlockCount, M);
    }
#endif

    public void Dispose()
    {
        NativeMemory.AlignedFree((void*)_repackedWeights);
        NativeMemory.AlignedFree((void*)_originalWeights);
        NativeMemory.AlignedFree((void*)_inputQ8);
        NativeMemory.AlignedFree(_output);
    }
}
