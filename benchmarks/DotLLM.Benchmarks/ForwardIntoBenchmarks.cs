using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Core.Tensors;

namespace DotLLM.Benchmarks;

/// <summary>
/// Microbenchmarks for the LM-head buffer routing change.
/// <para>
/// The legacy <c>TransformerModel.Forward</c> return path:
/// (1) allocates a fresh <see cref="UnmanagedTensor"/> via
///     <see cref="System.Runtime.InteropServices.NativeMemory.AlignedAlloc(nuint, nuint)"/>,
/// (2) memcpys <c>vocabSize × sizeof(float)</c> bytes from <c>_state.Logits</c> into
///     the new tensor, and
/// (3) frees the tensor on <see cref="IDisposable.Dispose"/>.
/// </para>
/// <para>
/// <c>TransformerModel.ForwardInto</c> writes the LM-head matmul output directly
/// into the caller's pre-pinned span — so none of (1) (2) (3) happen per call.
/// The matmul itself runs in both paths and is identical, so this microbench
/// isolates only the eliminated overhead: alloc + copy + free.
/// </para>
/// <para>
/// At per-token decode rates (TextGenerator hot path), this overhead runs once per
/// generated token. <see cref="MemoryDiagnoserAttribute"/> tracks the native-alloc
/// elimination too.
/// </para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class ForwardIntoBenchmarks
{
    private float[] _stateLogits = null!;
    private float[] _callerBuffer = null!;

    /// <summary>Vocab size — matches typical Llama (128K) / SmolLM (49K) / TinyLlama (32K).</summary>
    [Params(32_000, 49_152, 128_000)]
    public int VocabSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _stateLogits = new float[VocabSize];
        _callerBuffer = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++)
            _stateLogits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);
    }

    /// <summary>
    /// Legacy return-path overhead: NativeMemory.AlignedAlloc + memcpy from
    /// _state.Logits + Dispose (which calls NativeMemory.AlignedFree).
    /// Touches the destination to prevent dead-code elimination.
    /// </summary>
    [Benchmark(Baseline = true)]
    public void Legacy_AllocAndCopy()
    {
        var shape = new TensorShape(1, VocabSize);
        using var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        _stateLogits.AsSpan().CopyTo(new Span<float>((void*)result.DataPointer, VocabSize));

        // Touch the result to keep the copy live.
        ((float*)result.DataPointer)[0] += 0f;
    }

    /// <summary>
    /// ForwardInto return path: no allocation, no copy — the LM-head matmul writes
    /// directly into the caller's buffer. The only operation in this benchmark body
    /// is the equivalent "touch" to keep the comparison fair (Legacy's final touch
    /// is matched here).
    /// </summary>
    [Benchmark]
    public void ForwardInto_DirectWrite()
    {
        fixed (float* dst = _callerBuffer)
        {
            // No alloc, no copy. The matmul (not part of this microbench) writes
            // directly into `dst` in the real ForwardInto path.
            dst[0] += 0f;
        }
    }
}
