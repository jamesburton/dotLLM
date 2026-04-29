using System.Numerics;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Pre-allocated GPU scratch buffers for the CUDA forward pass. Activation buffers
/// are FP16, logits output is FP32. Mirrors <c>TransformerForwardState</c>
/// but on GPU memory allocated via <c>cuMemAlloc_v2</c>.
/// </summary>
internal sealed class CudaForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;

    private int _currentSeqLen;

    /// <summary>Total bytes currently allocated on GPU.</summary>
    public long AllocatedBytes { get; private set; }

    // All activation pointers are cuMemAlloc'd device memory, FP16 (sizeof(ushort) = 2 bytes)
    public nint HiddenState;    // [seqLen, hiddenSize]
    public nint Residual;       // [seqLen, hiddenSize]
    public nint NormOutput;     // [seqLen, hiddenSize]
    public nint Q;              // [seqLen, numHeads * headDim]
    public nint K;              // [seqLen, numKvHeads * headDim]
    public nint V;              // [seqLen, numKvHeads * headDim]
    public nint AttnOutput;     // [seqLen, numHeads * headDim]
    public nint FfnGate;        // [seqLen, intermediateSize]
    public nint FfnUp;          // [seqLen, intermediateSize]
    public nint SiluOutput;     // [seqLen, intermediateSize]

    // Optional high-precision activation buffers. Used by correctness paths
    // where cumulative FP16 activation truncation dominates real-model parity.
    public nint HiddenStateF32;  // [seqLen, hiddenSize]
    public nint ResidualF32;     // [seqLen, hiddenSize]
    public nint NormOutputF32;   // [seqLen, hiddenSize]
    public nint QF32;            // [seqLen, numHeads * headDim]
    public nint KF32;            // [seqLen, numKvHeads * headDim]
    public nint VF32;            // [seqLen, numKvHeads * headDim]
    public nint AttnOutputF32;   // [seqLen, numHeads * headDim]
    public nint FfnGateF32;      // [seqLen, intermediateSize]
    public nint FfnUpF32;        // [seqLen, intermediateSize]
    public nint SiluOutputF32;   // [seqLen, intermediateSize]

    // Fused-projection scratches — written by a single packed quantized GEMV
    // when the layer has CudaLayerWeights.QkvPacked / GateUpPacked set.
    // Decode-only: only safe when seqLen==1 because consumers downstream
    // (RoPE, KV update, attention, SwiGLU) read each slice with the per-tensor
    // stride, not the packed (n_q+2*n_kv) / (2*intermediate) stride.
    public nint QkvPacked;      // [n_q + 2 * n_kv] (seqLen=1 decode only)
    public nint GateUpPacked;   // [2 * intermediateSize] (seqLen=1 decode only)

    // Logits — FP16 on device, then converted to FP32
    public nint LogitsF16;      // [vocabSize] FP16
    public nint LogitsF32;      // [vocabSize] FP32

    // General-purpose FP16 scratch buffer
    public nint GemmOutputF16;

    // On-the-fly dequantization scratch: holds one projection's FP16 weights
    // for cuBLAS GEMM. Sized for the largest projection (max of Gate/Up/Down/Q/O).
    // Reused across all cuBLAS calls — safe because all ops are on the same stream.
    public nint DequantScratch;
    public nint DequantScratchF32;

    // Pre-Q8_1 input quantization scratch (single buffer per forward state, sized
    // for the largest GEMV input vector across all call sites in the model).
    // Layout: int8_t xq[K] | half dx[K/32] | half sx2[K/16]. Consumed by the MMQ
    // `_preq` GEMV kernel variants which skip Stage 1. Populated once per fused
    // projection group via LaunchQuantizeXToQ8_1, then reused across all rows
    // (eliminates the redundant Stage 1 work the on-the-fly kernels run per block).
    public nint PreQ8_1Scratch;
    public int  PreQ8_1ScratchK;        // capacity in elements (must be a multiple of 32)

    // Small device buffers for H2D copy of token IDs and positions
    public nint TokenIdsDevice; // [maxSeqLen] int32
    public nint PositionsDevice;// [maxSeqLen] int32

    public CudaForwardState(int hiddenSize, int numHeads, int numKvHeads, int headDim,
                              int intermediateSize, int vocabSize)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;
        _currentSeqLen = 0;

        // Logits are fixed-size (only last token)
        LogitsF16 = AllocDevice((long)vocabSize * sizeof(ushort));
        LogitsF32 = AllocDevice((long)vocabSize * sizeof(float));

        // Dequant scratch: sized for the largest per-layer projection in FP16.
        // Used for on-the-fly dequantization of quantized weights before cuBLAS GEMM.
        long maxProjectionElements = Math.Max(
            (long)Math.Max(numHeads * headDim, numKvHeads * headDim) * hiddenSize,
            (long)intermediateSize * hiddenSize);
        DequantScratch = AllocDevice(maxProjectionElements * sizeof(ushort));
        DequantScratchF32 = AllocDevice(maxProjectionElements * sizeof(float));

        // Fused decode scratches (seqLen=1 only). Sized once — tiny.
        long qkvPackedFp16Bytes = (long)(numHeads * headDim + 2 * numKvHeads * headDim) * sizeof(ushort);
        long gateUpPackedFp16Bytes = (long)(2 * intermediateSize) * sizeof(ushort);
        QkvPacked = AllocDevice(qkvPackedFp16Bytes);
        GateUpPacked = AllocDevice(gateUpPackedFp16Bytes);

        // Pre-Q8_1 scratch: sized for the largest GEMV input vector.
        // Inputs are: hidden (Q/K/V/Gate/Up/LmHead read NormOutput), intermediate (Down reads SiluOutput),
        // num_q*head_dim (O reads AttnOutput). Take the max and round up to a 32-element multiple.
        int preQ8K = Math.Max(hiddenSize, Math.Max(intermediateSize, numHeads * headDim));
        preQ8K = ((preQ8K + 31) / 32) * 32;
        PreQ8_1ScratchK = preQ8K;
        PreQ8_1Scratch = AllocDevice(PreQ8_1ScratchBytes(preQ8K));

        // Initial allocation for decode (seqLen=1)
        EnsureCapacity(1);
    }

    /// <summary>
    /// Ensures all scratch buffers are large enough for <paramref name="seqLen"/> tokens.
    /// Uses power-of-2 growth to amortize reallocation cost.
    /// </summary>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen)
            return;

        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeSequenceBuffers();

        int half = sizeof(ushort); // FP16 = 2 bytes

        // All activation buffers are FP16 — per GPU.md spec for memory-bandwidth-optimal inference.
        // Only LogitsF32 (output to host) stays FP32.
        HiddenState = AllocDevice((long)newCapacity * _hiddenSize * half);
        Residual = AllocDevice((long)newCapacity * _hiddenSize * half);
        NormOutput = AllocDevice((long)newCapacity * _hiddenSize * half);
        Q = AllocDevice((long)newCapacity * _numHeads * _headDim * half);
        K = AllocDevice((long)newCapacity * _numKvHeads * _headDim * half);
        V = AllocDevice((long)newCapacity * _numKvHeads * _headDim * half);
        AttnOutput = AllocDevice((long)newCapacity * _numHeads * _headDim * half);
        FfnGate = AllocDevice((long)newCapacity * _intermediateSize * half);
        FfnUp = AllocDevice((long)newCapacity * _intermediateSize * half);
        SiluOutput = AllocDevice((long)newCapacity * _intermediateSize * half);
        HiddenStateF32 = AllocDevice((long)newCapacity * _hiddenSize * sizeof(float));
        ResidualF32 = AllocDevice((long)newCapacity * _hiddenSize * sizeof(float));
        NormOutputF32 = AllocDevice((long)newCapacity * _hiddenSize * sizeof(float));
        QF32 = AllocDevice((long)newCapacity * _numHeads * _headDim * sizeof(float));
        KF32 = AllocDevice((long)newCapacity * _numKvHeads * _headDim * sizeof(float));
        VF32 = AllocDevice((long)newCapacity * _numKvHeads * _headDim * sizeof(float));
        AttnOutputF32 = AllocDevice((long)newCapacity * _numHeads * _headDim * sizeof(float));
        FfnGateF32 = AllocDevice((long)newCapacity * _intermediateSize * sizeof(float));
        FfnUpF32 = AllocDevice((long)newCapacity * _intermediateSize * sizeof(float));
        SiluOutputF32 = AllocDevice((long)newCapacity * _intermediateSize * sizeof(float));
        // General scratch: must fit largest projection output or LM head logits
        long maxPerLayer = (long)newCapacity * Math.Max(Math.Max(_numHeads * _headDim, _intermediateSize), _hiddenSize);
        long maxLmHead = _vocabSize;
        GemmOutputF16 = AllocDevice(Math.Max(maxPerLayer, maxLmHead) * half);
        TokenIdsDevice = AllocDevice((long)newCapacity * sizeof(int));
        PositionsDevice = AllocDevice((long)newCapacity * sizeof(int));

        _currentSeqLen = newCapacity;
    }

    private nint AllocDevice(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        AllocatedBytes += bytes;
        return ptr;
    }

    private void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            CudaDriverApi.cuMemFree_v2(ptr);
            ptr = 0;
        }
    }

    private void FreeSequenceBuffers()
    {
        FreeIfNonZero(ref HiddenState);
        FreeIfNonZero(ref Residual);
        FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref Q);
        FreeIfNonZero(ref K);
        FreeIfNonZero(ref V);
        FreeIfNonZero(ref AttnOutput);
        FreeIfNonZero(ref FfnGate);
        FreeIfNonZero(ref FfnUp);
        FreeIfNonZero(ref SiluOutput);
        FreeIfNonZero(ref HiddenStateF32);
        FreeIfNonZero(ref ResidualF32);
        FreeIfNonZero(ref NormOutputF32);
        FreeIfNonZero(ref QF32);
        FreeIfNonZero(ref KF32);
        FreeIfNonZero(ref VF32);
        FreeIfNonZero(ref AttnOutputF32);
        FreeIfNonZero(ref FfnGateF32);
        FreeIfNonZero(ref FfnUpF32);
        FreeIfNonZero(ref SiluOutputF32);
        FreeIfNonZero(ref GemmOutputF16);
        FreeIfNonZero(ref TokenIdsDevice);
        FreeIfNonZero(ref PositionsDevice);
    }

    public void Dispose()
    {
        FreeSequenceBuffers();
        FreeIfNonZero(ref LogitsF16);
        FreeIfNonZero(ref LogitsF32);
        FreeIfNonZero(ref DequantScratch);
        FreeIfNonZero(ref DequantScratchF32);
        FreeIfNonZero(ref QkvPacked);
        FreeIfNonZero(ref GateUpPacked);
        FreeIfNonZero(ref PreQ8_1Scratch);
        PreQ8_1ScratchK = 0;
        _currentSeqLen = 0;
    }

    /// <summary>
    /// Total bytes required for the pre-Q8_1 scratch buffer holding a vector of <paramref name="k"/> elements.
    /// Layout: int8_t xq[k] | half dx[k/32] | half sx2[k/16].
    /// </summary>
    /// <remarks>k must be a multiple of 32 (pre-quant kernel chunks 32 elements at a time).</remarks>
    public static long PreQ8_1ScratchBytes(int k)
    {
        // xq: k bytes, dx: (k/32) halves = k/16 bytes, sx2: (k/16) halves = k/8 bytes.
        // Total: k + k/16 + k/8 = k * (1 + 0.0625 + 0.125) = k * 1.1875.
        return (long)k + ((long)k / 16) + ((long)k / 8);
    }
}
