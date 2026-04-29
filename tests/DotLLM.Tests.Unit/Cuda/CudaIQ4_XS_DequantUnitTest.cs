using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Isolated CUDA IQ4_XS dequant kernel verification.
/// Tests the low-level dequant kernel in isolation to separate dequant bugs
/// from direct-GEMV bugs.
/// </summary>
[Trait("Category", "GPU")]
[Trait("Category", "Diagnostics")]
public sealed unsafe class CudaIQ4_XS_DequantUnitTest
{
    private readonly ITestOutputHelper _output;

    public CudaIQ4_XS_DequantUnitTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableFact]
    public void IQ4_XS_Dequant_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        // Minimal IQ4_XS tensor: 256 values = 1 super-block.
        // Block layout: d(2B) + scales_h(2B) + scales_l(4B) + qs(128B) = 136 bytes
        const int ElementCount = 256;
        const int BlockBytes = 136;

        // Allocate host memory for IQ4_XS packed data
        byte[] packedHost = new byte[BlockBytes];
        float[] cpuOutput = new float[ElementCount];
        float[] cudaOutputHost = new float[ElementCount];

        // Create test data: simple pattern
        CreateTestIQ4_XSData(packedHost);

        // CPU oracle dequant
        fixed (byte* pSrc = packedHost)
        {
            Dequantize.DequantizeIQ4_XS((nint)pSrc, ElementCount, cpuOutput);
        }

        _output.WriteLine($"CPU oracle dequantized: {ElementCount} elements");
        _output.WriteLine($"  min={cpuOutput.Min():F6}, max={cpuOutput.Max():F6}, mean={cpuOutput.Average():F6}");

        // CUDA dequant
        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();

        string ptxDir = ResolvePtxDir();
        var kernels = new CudaKernels(ptxDir);

        CudaDriverApi.cuMemAlloc_v2(out nint dSrc, (nuint)BlockBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dDest, (nuint)(ElementCount * sizeof(ushort))).ThrowOnError();

        try
        {
            // Copy test data to GPU
            fixed (byte* pSrc = packedHost)
            {
                CudaDriverApi.cuMemcpyHtoDAsync_v2(dSrc, (nint)pSrc, (nuint)BlockBytes, stream.Handle).ThrowOnError();
            }

            // Launch dequant kernel (outputs FP16)
            kernels.LaunchDequantToF16(dSrc, QuantizationType.IQ4_XS, dDest, ElementCount, stream.Handle);
            stream.Synchronize();

            // Copy result back as FP16
            ushort[] cudaOutputF16 = new ushort[ElementCount];
            fixed (ushort* pOut = cudaOutputF16)
            {
                CudaDriverApi.cuMemcpyDtoHAsync_v2((nint)pOut, dDest, (nuint)(ElementCount * sizeof(ushort)), stream.Handle)
                    .ThrowOnError();
            }
            stream.Synchronize();

            // Convert raw FP16 bit patterns to float for comparison
            for (int i = 0; i < ElementCount; i++)
                cudaOutputHost[i] = (float)BitConverter.UInt16BitsToHalf(cudaOutputF16[i]);

            _output.WriteLine($"CUDA dequant produced: {ElementCount} elements (FP16)");
            _output.WriteLine($"  min={cudaOutputHost.Min():F6}, max={cudaOutputHost.Max():F6}, mean={cudaOutputHost.Average():F6}");

            // Compare: absolute and relative tolerance
            const float AbsTol = 0.01f;
            const float RelTol = 0.001f;

            float maxAbsDiff = 0;
            int maxAbsDiffIdx = -1;
            float meanAbsDiff = 0;
            int divergeCount = 0;

            for (int i = 0; i < ElementCount; i++)
            {
                float diff = MathF.Abs(cudaOutputHost[i] - cpuOutput[i]);
                float relDiff = cpuOutput[i] != 0 ? diff / MathF.Abs(cpuOutput[i]) : diff;

                if (diff > AbsTol || relDiff > RelTol)
                    divergeCount++;

                if (diff > maxAbsDiff)
                {
                    maxAbsDiff = diff;
                    maxAbsDiffIdx = i;
                }

                meanAbsDiff += diff;
            }

            meanAbsDiff /= ElementCount;

            _output.WriteLine($"Comparison (CPU vs CUDA):");
            _output.WriteLine($"  max_abs_diff={maxAbsDiff:F6} at index {maxAbsDiffIdx}");
            _output.WriteLine($"  mean_abs_diff={meanAbsDiff:F6}");
            _output.WriteLine($"  diverge_count={divergeCount}/{ElementCount}");

            if (maxAbsDiffIdx >= 0)
            {
                _output.WriteLine($"  largest divergence: cuda[{maxAbsDiffIdx}]={cudaOutputHost[maxAbsDiffIdx]:F6} vs cpu={cpuOutput[maxAbsDiffIdx]:F6}");
            }

            // Assert: CUDA dequant should match CPU oracle within tolerance
            Assert.True(maxAbsDiff < 0.01f, $"IQ4_XS dequant divergence too large: max_abs={maxAbsDiff:F6}");
            Assert.True(meanAbsDiff < 0.001f, $"Mean IQ4_XS dequant error too large: mean_abs={meanAbsDiff:F6}");
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dDest != 0) CudaDriverApi.cuMemFree_v2(dDest);
            kernels.Dispose();
        }
    }

    [SkippableFact]
    public void IQ4_XS_Dequant_RealLayer0Q_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf");
        Skip.IfNot(File.Exists(modelPath), "Meta-Llama-3.1-8B IQ4_XS GGUF not found");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var weights = TransformerWeights.LoadFromGguf(gguf, config, skipF32MoeDequant: true);

        ref readonly var lw = ref weights.Layers[0];
        Skip.If(lw.QQuantType != QuantizationType.IQ4_XS, $"Layer 0 Q is {lw.QQuantType}, not IQ4_XS");

        int elementCount = lw.QOutputDim * lw.QInputDim;
        long weightBytes = Dequantize.RowByteSize(lw.QInputDim, lw.QQuantType) * (long)lw.QOutputDim;
        _output.WriteLine($"Layer0.Q {lw.QOutputDim}x{lw.QInputDim}, elements={elementCount}, bytes={weightBytes}");

        float[] cpuF32 = new float[elementCount];
        Dequantize.ToFloat32(lw.QWeight, elementCount, QuantizationType.IQ4_XS, cpuF32);

        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());

        nint dSrc = 0;
        nint dDest = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)weightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDest, (nuint)(elementCount * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dSrc, lw.QWeight, (nuint)weightBytes).ThrowOnError();

            kernels.LaunchDequantToF16(dSrc, QuantizationType.IQ4_XS, dDest, elementCount, stream.Handle);
            stream.Synchronize();

            ushort[] cudaF16 = new ushort[elementCount];
            fixed (ushort* p = cudaF16)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dDest, (nuint)(elementCount * sizeof(ushort))).ThrowOnError();

            float maxAbs = 0;
            float meanAbs = 0;
            int maxIdx = -1;
            for (int i = 0; i < elementCount; i++)
            {
                float cpuRounded = (float)(Half)cpuF32[i];
                float cuda = (float)BitConverter.UInt16BitsToHalf(cudaF16[i]);
                float diff = MathF.Abs(cuda - cpuRounded);
                meanAbs += diff;
                if (diff > maxAbs)
                {
                    maxAbs = diff;
                    maxIdx = i;
                }
            }

            meanAbs /= elementCount;
            _output.WriteLine($"max_abs_diff={maxAbs:F6} at {maxIdx}, mean_abs_diff={meanAbs:F8}");
            if (maxIdx >= 0)
            {
                float cpuRounded = (float)(Half)cpuF32[maxIdx];
                float cuda = (float)BitConverter.UInt16BitsToHalf(cudaF16[maxIdx]);
                _output.WriteLine($"largest divergence: cuda={cuda:F6}, cpu_f16={cpuRounded:F6}, cpu_f32={cpuF32[maxIdx]:F6}");
            }

            Assert.True(maxAbs < 0.001f, $"Real IQ4_XS dequant max diff too large: {maxAbs}");
            Assert.True(meanAbs < 0.00001f, $"Real IQ4_XS dequant mean diff too large: {meanAbs}");
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dDest != 0) CudaDriverApi.cuMemFree_v2(dDest);
        }
    }

    [SkippableFact]
    public void IQ4_XS_Layer0QPrefillProjection_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf");
        Skip.IfNot(File.Exists(modelPath), "Meta-Llama-3.1-8B IQ4_XS GGUF not found");

        int[] promptTokens = [128000, 791, 6864, 315, 9822, 374];

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var weights = TransformerWeights.LoadFromGguf(gguf, config, skipF32MoeDequant: true);
        ref readonly var lw = ref weights.Layers[0];
        Skip.If(lw.QQuantType != QuantizationType.IQ4_XS, $"Layer 0 Q is {lw.QQuantType}, not IQ4_XS");

        int seqLen = promptTokens.Length;
        int hidden = config.HiddenSize;
        int qDim = lw.QOutputDim;

        float[] hiddenF32 = new float[seqLen * hidden];
        Half[] hiddenF16 = new Half[hiddenF32.Length];
        LoadEmbeddingRows(weights.TokenEmbedWeight, weights.TokenEmbedQuantType, promptTokens, hidden, hiddenF32);
        for (int i = 0; i < hiddenF32.Length; i++)
            hiddenF16[i] = (Half)hiddenF32[i];

        float[] normWeightF16 = new float[lw.AttnNormWeight.Length];
        for (int i = 0; i < normWeightF16.Length; i++)
            normWeightF16[i] = (float)(Half)lw.AttnNormWeight[i];

        float[] cpuNorm = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++)
        {
            float[] hiddenRow = new float[hidden];
            for (int i = 0; i < hidden; i++)
                hiddenRow[i] = (float)hiddenF16[t * hidden + i];

            RmsNorm.Execute(hiddenRow, normWeightF16, config.NormEpsilon,
                cpuNorm.AsSpan(t * hidden, hidden));
        }

        float[] cpuQ = ComputeProjectionReference(lw.QWeight, lw.QQuantType, qDim, hidden, cpuNorm, seqLen);

        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);
        using var kernels = new CudaKernels(ResolvePtxDir());

        nint dHidden = 0, dNormWeightF32 = 0, dNormWeightF16 = 0, dNorm = 0;
        nint dQRaw = 0, dQF16 = 0, dQOut = 0;
        try
        {
            long hiddenBytes = (long)hiddenF16.Length * sizeof(ushort);
            long normWeightBytes = (long)hidden * sizeof(float);
            long normBytes = hiddenBytes;
            long qWeightBytes = Dequantize.RowByteSize(hidden, lw.QQuantType) * (long)qDim;
            long qF16Bytes = (long)qDim * hidden * sizeof(ushort);
            long qOutBytes = (long)seqLen * qDim * sizeof(ushort);

            CudaDriverApi.cuMemAlloc_v2(out dHidden, (nuint)hiddenBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dNormWeightF32, (nuint)normWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dNormWeightF16, (nuint)(hidden * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dNorm, (nuint)normBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQRaw, (nuint)qWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQF16, (nuint)qF16Bytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQOut, (nuint)qOutBytes).ThrowOnError();

            fixed (Half* p = hiddenF16)
                CudaDriverApi.cuMemcpyHtoD_v2(dHidden, (nint)p, (nuint)hiddenBytes).ThrowOnError();
            fixed (float* p = normWeightF16)
                CudaDriverApi.cuMemcpyHtoD_v2(dNormWeightF32, (nint)p, (nuint)normWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dQRaw, lw.QWeight, (nuint)qWeightBytes).ThrowOnError();

            kernels.LaunchConvertF32ToF16(dNormWeightF32, dNormWeightF16, hidden, stream.Handle);
            kernels.LaunchRmsNorm(dHidden, dNormWeightF16, dNorm, hidden, config.NormEpsilon, seqLen, stream.Handle);
            kernels.LaunchDequantToF16(dQRaw, QuantizationType.IQ4_XS, dQF16, qDim * hidden, stream.Handle);
            CudaGemm.LinearF16(cublas.Handle, dNorm, dQF16, dQOut, seqLen, hidden, qDim, stream.Handle);
            stream.Synchronize();

            Half[] gpuNormF16 = new Half[seqLen * hidden];
            Half[] gpuQF16 = new Half[seqLen * qDim];
            fixed (Half* p = gpuNormF16)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dNorm, (nuint)normBytes).ThrowOnError();
            fixed (Half* p = gpuQF16)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dQOut, (nuint)qOutBytes).ThrowOnError();

            var normDiff = CompareHalfToFloat(gpuNormF16, cpuNorm);
            var qDiff = CompareHalfToFloat(gpuQF16, cpuQ);
            _output.WriteLine($"norm: max={normDiff.maxAbs:F6} mean={normDiff.meanAbs:F8} idx={normDiff.maxIdx}");
            _output.WriteLine($"qproj: max={qDiff.maxAbs:F6} mean={qDiff.meanAbs:F8} idx={qDiff.maxIdx}");

            Assert.True(normDiff.maxAbs < 0.01f, $"Layer0 norm diverged: {normDiff.maxAbs}");
            Assert.True(qDiff.meanAbs < 0.25f, $"Layer0 Q projection mean diff too large: {qDiff.meanAbs}");
        }
        finally
        {
            if (dHidden != 0) CudaDriverApi.cuMemFree_v2(dHidden);
            if (dNormWeightF32 != 0) CudaDriverApi.cuMemFree_v2(dNormWeightF32);
            if (dNormWeightF16 != 0) CudaDriverApi.cuMemFree_v2(dNormWeightF16);
            if (dNorm != 0) CudaDriverApi.cuMemFree_v2(dNorm);
            if (dQRaw != 0) CudaDriverApi.cuMemFree_v2(dQRaw);
            if (dQF16 != 0) CudaDriverApi.cuMemFree_v2(dQF16);
            if (dQOut != 0) CudaDriverApi.cuMemFree_v2(dQOut);
        }
    }

    [SkippableFact]
    public void IQ4_XS_Layer0AttentionPrefill_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf");
        Skip.IfNot(File.Exists(modelPath), "Meta-Llama-3.1-8B IQ4_XS GGUF not found");

        int[] promptTokens = [128000, 791, 6864, 315, 9822, 374];
        int[] positions = [0, 1, 2, 3, 4, 5];

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var weights = TransformerWeights.LoadFromGguf(gguf, config, skipF32MoeDequant: true);
        ref readonly var lw = ref weights.Layers[0];

        int seqLen = promptTokens.Length;
        int hidden = config.HiddenSize;
        int qDim = lw.QOutputDim;
        int kvDim = lw.KOutputDim;
        int headDim = config.HeadDim;
        int ropeDim = config.RoPEConfig?.DimensionCount ?? headDim;
        if (ropeDim == 0) ropeDim = headDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int ropeType = (int)(config.RoPEConfig?.Type ?? RoPEType.Norm);

        float[] hiddenF32 = new float[seqLen * hidden];
        Half[] hiddenF16 = new Half[hiddenF32.Length];
        LoadEmbeddingRows(weights.TokenEmbedWeight, weights.TokenEmbedQuantType, promptTokens, hidden, hiddenF32);
        for (int i = 0; i < hiddenF32.Length; i++)
            hiddenF16[i] = (Half)hiddenF32[i];

        float[] normWeightF16 = new float[lw.AttnNormWeight.Length];
        for (int i = 0; i < normWeightF16.Length; i++)
            normWeightF16[i] = (float)(Half)lw.AttnNormWeight[i];

        float[] cpuNorm = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++)
        {
            float[] hiddenRow = new float[hidden];
            for (int i = 0; i < hidden; i++)
                hiddenRow[i] = (float)hiddenF16[t * hidden + i];

            RmsNorm.Execute(hiddenRow, normWeightF16, config.NormEpsilon,
                cpuNorm.AsSpan(t * hidden, hidden));
        }

        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);
        using var kernels = new CudaKernels(ResolvePtxDir());

        nint dHidden = 0, dNormWeightF32 = 0, dNormWeightF16 = 0, dFfnNormWeightF32 = 0, dFfnNormWeightF16 = 0;
        nint dNorm = 0, dResidual = 0, dPositions = 0;
        nint dQRaw = 0, dKRaw = 0, dVRaw = 0, dORaw = 0, dGateRaw = 0, dUpRaw = 0, dDownRaw = 0;
        nint dWF16 = 0, dQ = 0, dK = 0, dV = 0, dAttn = 0, dGate = 0, dUp = 0, dSilu = 0, dHiddenOut = 0;
        try
        {
            long hiddenBytes = (long)hiddenF16.Length * sizeof(ushort);
            long normWeightBytes = (long)hidden * sizeof(float);
            long qWeightBytes = Dequantize.RowByteSize(hidden, lw.QQuantType) * (long)qDim;
            long kWeightBytes = Dequantize.RowByteSize(hidden, lw.KQuantType) * (long)kvDim;
            long vWeightBytes = Dequantize.RowByteSize(hidden, lw.VQuantType) * (long)kvDim;
            long oWeightBytes = Dequantize.RowByteSize(qDim, lw.OQuantType) * (long)hidden;
            long gateWeightBytes = Dequantize.RowByteSize(hidden, lw.GateQuantType) * (long)config.IntermediateSize;
            long upWeightBytes = Dequantize.RowByteSize(hidden, lw.UpQuantType) * (long)config.IntermediateSize;
            long downWeightBytes = Dequantize.RowByteSize(config.IntermediateSize, lw.DownQuantType) * (long)hidden;
            long maxWeightF16Bytes = (long)Math.Max(config.IntermediateSize, hidden) * hidden * sizeof(ushort);
            long qBytes = (long)seqLen * qDim * sizeof(ushort);
            long kvBytes = (long)seqLen * kvDim * sizeof(ushort);
            long intermediateBytes = (long)seqLen * config.IntermediateSize * sizeof(ushort);

            CudaDriverApi.cuMemAlloc_v2(out dHidden, (nuint)hiddenBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dNormWeightF32, (nuint)normWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dNormWeightF16, (nuint)(hidden * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dFfnNormWeightF32, (nuint)normWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dFfnNormWeightF16, (nuint)(hidden * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dNorm, (nuint)hiddenBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dResidual, (nuint)hiddenBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPositions, (nuint)(positions.Length * sizeof(int))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQRaw, (nuint)qWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dKRaw, (nuint)kWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dVRaw, (nuint)vWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dORaw, (nuint)oWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dGateRaw, (nuint)gateWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dUpRaw, (nuint)upWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDownRaw, (nuint)downWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dWF16, (nuint)maxWeightF16Bytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dAttn, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dGate, (nuint)intermediateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dUp, (nuint)intermediateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dSilu, (nuint)intermediateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dHiddenOut, (nuint)hiddenBytes).ThrowOnError();

            fixed (Half* p = hiddenF16)
            {
                CudaDriverApi.cuMemcpyHtoD_v2(dHidden, (nint)p, (nuint)hiddenBytes).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoD_v2(dResidual, (nint)p, (nuint)hiddenBytes).ThrowOnError();
            }
            fixed (float* p = normWeightF16)
                CudaDriverApi.cuMemcpyHtoD_v2(dNormWeightF32, (nint)p, (nuint)normWeightBytes).ThrowOnError();
            float[] ffnNormWeightF16 = new float[lw.FfnNormWeight.Length];
            for (int i = 0; i < ffnNormWeightF16.Length; i++)
                ffnNormWeightF16[i] = (float)(Half)lw.FfnNormWeight[i];
            fixed (float* p = ffnNormWeightF16)
                CudaDriverApi.cuMemcpyHtoD_v2(dFfnNormWeightF32, (nint)p, (nuint)normWeightBytes).ThrowOnError();
            fixed (int* p = positions)
                CudaDriverApi.cuMemcpyHtoD_v2(dPositions, (nint)p, (nuint)(positions.Length * sizeof(int))).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dQRaw, lw.QWeight, (nuint)qWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dKRaw, lw.KWeight, (nuint)kWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dVRaw, lw.VWeight, (nuint)vWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dORaw, lw.OWeight, (nuint)oWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dGateRaw, lw.GateWeight, (nuint)gateWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dUpRaw, lw.UpWeight, (nuint)upWeightBytes).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dDownRaw, lw.DownWeight, (nuint)downWeightBytes).ThrowOnError();

            kernels.LaunchConvertF32ToF16(dNormWeightF32, dNormWeightF16, hidden, stream.Handle);
            kernels.LaunchConvertF32ToF16(dFfnNormWeightF32, dFfnNormWeightF16, hidden, stream.Handle);
            kernels.LaunchRmsNorm(dHidden, dNormWeightF16, dNorm, hidden, config.NormEpsilon, seqLen, stream.Handle);
            ProjectF16(kernels, cublas, stream, dQRaw, lw.QQuantType, dWF16, dNorm, dQ, qDim, hidden, seqLen);
            ProjectF16(kernels, cublas, stream, dKRaw, lw.KQuantType, dWF16, dNorm, dK, kvDim, hidden, seqLen);
            ProjectF16(kernels, cublas, stream, dVRaw, lw.VQuantType, dWF16, dNorm, dV, kvDim, hidden, seqLen);
            stream.Synchronize();

            Half[] qBeforeRope = CopyHalfDevice(dQ, seqLen * qDim);
            Half[] kBeforeRope = CopyHalfDevice(dK, seqLen * kvDim);
            Half[] vGpu = CopyHalfDevice(dV, seqLen * kvDim);

            kernels.LaunchRoPE(dQ, dK, dPositions, seqLen, config.NumAttentionHeads, config.NumKvHeads,
                headDim, ropeDim, ropeTheta, ropeType, stream.Handle);
            stream.Synchronize();

            Half[] qAfterRopeGpu = CopyHalfDevice(dQ, seqLen * qDim);
            Half[] kAfterRopeGpu = CopyHalfDevice(dK, seqLen * kvDim);

            float[] qCpu = HalfArrayToFloat(qBeforeRope);
            float[] kCpu = HalfArrayToFloat(kBeforeRope);
            float[] vCpu = HalfArrayToFloat(vGpu);
            float[] cos = new float[config.MaxSequenceLength * (ropeDim / 2)];
            float[] sin = new float[cos.Length];
            RoPE.PrecomputeFrequencyTable(config.MaxSequenceLength, ropeDim, ropeTheta, cos, sin);
            RoPE.Execute(qCpu, kCpu, positions, config.NumAttentionHeads, config.NumKvHeads,
                headDim, ropeDim, cos, sin, config.RoPEConfig?.Type ?? RoPEType.Norm);

            var ropeQDiff = CompareHalfToFloat(qAfterRopeGpu, qCpu);
            var ropeKDiff = CompareHalfToFloat(kAfterRopeGpu, kCpu);
            _output.WriteLine($"rope q: max={ropeQDiff.maxAbs:F6} mean={ropeQDiff.meanAbs:F8} idx={ropeQDiff.maxIdx}");
            _output.WriteLine($"rope k: max={ropeKDiff.maxAbs:F6} mean={ropeKDiff.meanAbs:F8} idx={ropeKDiff.maxIdx}");

            float[] attnCpu = new float[seqLen * qDim];
            fixed (float* pAttnCpu = attnCpu)
            fixed (float* pQCpu = qCpu)
            fixed (float* pKCpu = kCpu)
            fixed (float* pVCpu = vCpu)
            {
                Attention.Execute(pQCpu, pKCpu, pVCpu, pAttnCpu, seqLen, seqLen,
                    config.NumAttentionHeads, config.NumKvHeads, headDim, 0, null,
                    config.SlidingWindowSize);
            }

            kernels.LaunchAttention(dQ, dK, dV, dAttn, seqLen, seqLen, config.NumAttentionHeads,
                config.NumKvHeads, headDim, 0, config.SlidingWindowSize ?? 0, stream.Handle);
            stream.Synchronize();
            Half[] attnGpu = CopyHalfDevice(dAttn, seqLen * qDim);
            var attnDiff = CompareHalfToFloat(attnGpu, attnCpu);
            _output.WriteLine($"attention: max={attnDiff.maxAbs:F6} mean={attnDiff.meanAbs:F8} idx={attnDiff.maxIdx}");

            float[] attnForO = HalfArrayToFloat(attnGpu);
            float[] oCpu = ComputeProjectionReference(lw.OWeight, lw.OQuantType, hidden, qDim, attnForO, seqLen);
            ProjectF16(kernels, cublas, stream, dORaw, lw.OQuantType, dWF16, dAttn, dNorm, hidden, qDim, seqLen);
            stream.Synchronize();
            Half[] oGpu = CopyHalfDevice(dNorm, seqLen * hidden);
            var oDiff = CompareHalfToFloat(oGpu, oCpu);
            _output.WriteLine($"o proj: max={oDiff.maxAbs:F6} mean={oDiff.meanAbs:F8} idx={oDiff.maxIdx}");

            float[] ffnNormCpu = new float[seqLen * hidden];
            Half[] residualAfterAttnCpu = FusedAddRmsNormReference(hiddenF16, oGpu, ffnNormWeightF16,
                config.NormEpsilon, seqLen, hidden, ffnNormCpu);
            kernels.LaunchFusedAddRmsNorm(dResidual, dNorm, dFfnNormWeightF16, dNorm,
                hidden, config.NormEpsilon, seqLen, stream.Handle);
            stream.Synchronize();
            Half[] residualAfterAttnGpu = CopyHalfDevice(dResidual, seqLen * hidden);
            Half[] ffnNormGpu = CopyHalfDevice(dNorm, seqLen * hidden);
            var residualDiff = CompareHalfArrays(residualAfterAttnGpu, residualAfterAttnCpu);
            var ffnNormDiff = CompareHalfToFloat(ffnNormGpu, ffnNormCpu);
            _output.WriteLine($"attn residual: max={residualDiff.maxAbs:F6} mean={residualDiff.meanAbs:F8} idx={residualDiff.maxIdx}");
            _output.WriteLine($"ffn norm: max={ffnNormDiff.maxAbs:F6} mean={ffnNormDiff.meanAbs:F8} idx={ffnNormDiff.maxIdx}");

            float[] gateCpu = ComputeProjectionReference(lw.GateWeight, lw.GateQuantType,
                config.IntermediateSize, hidden, HalfArrayToFloat(ffnNormGpu), seqLen);
            float[] upCpu = ComputeProjectionReference(lw.UpWeight, lw.UpQuantType,
                config.IntermediateSize, hidden, HalfArrayToFloat(ffnNormGpu), seqLen);
            ProjectF16(kernels, cublas, stream, dGateRaw, lw.GateQuantType, dWF16, dNorm, dGate,
                config.IntermediateSize, hidden, seqLen);
            ProjectF16(kernels, cublas, stream, dUpRaw, lw.UpQuantType, dWF16, dNorm, dUp,
                config.IntermediateSize, hidden, seqLen);
            stream.Synchronize();
            Half[] gateGpu = CopyHalfDevice(dGate, seqLen * config.IntermediateSize);
            Half[] upGpu = CopyHalfDevice(dUp, seqLen * config.IntermediateSize);
            var gateDiff = CompareHalfToFloat(gateGpu, gateCpu);
            var upDiff = CompareHalfToFloat(upGpu, upCpu);
            _output.WriteLine($"gate proj: max={gateDiff.maxAbs:F6} mean={gateDiff.meanAbs:F8} idx={gateDiff.maxIdx}");
            _output.WriteLine($"up proj: max={upDiff.maxAbs:F6} mean={upDiff.meanAbs:F8} idx={upDiff.maxIdx}");

            float[] siluCpu = SwiGluReference(gateGpu, upGpu);
            kernels.LaunchSwiGLU(dGate, dUp, dSilu, config.IntermediateSize, seqLen, stream.Handle);
            stream.Synchronize();
            Half[] siluGpu = CopyHalfDevice(dSilu, seqLen * config.IntermediateSize);
            var siluDiff = CompareHalfToFloat(siluGpu, siluCpu);
            _output.WriteLine($"swiglu: max={siluDiff.maxAbs:F6} mean={siluDiff.meanAbs:F8} idx={siluDiff.maxIdx}");

            float[] downCpu = ComputeProjectionReference(lw.DownWeight, lw.DownQuantType,
                hidden, config.IntermediateSize, HalfArrayToFloat(siluGpu), seqLen);
            ProjectF16(kernels, cublas, stream, dDownRaw, lw.DownQuantType, dWF16, dSilu, dNorm,
                hidden, config.IntermediateSize, seqLen);
            stream.Synchronize();
            Half[] downGpu = CopyHalfDevice(dNorm, seqLen * hidden);
            var downDiff = CompareHalfToFloat(downGpu, downCpu);
            _output.WriteLine($"down proj: max={downDiff.maxAbs:F6} mean={downDiff.meanAbs:F8} idx={downDiff.maxIdx}");

            Half[] hiddenOutCpu = AddHalfReference(residualAfterAttnGpu, downGpu);
            kernels.LaunchAdd(dResidual, dNorm, dHiddenOut, seqLen * hidden, stream.Handle);
            stream.Synchronize();
            Half[] hiddenOutGpu = CopyHalfDevice(dHiddenOut, seqLen * hidden);
            var hiddenOutDiff = CompareHalfArrays(hiddenOutGpu, hiddenOutCpu);
            _output.WriteLine($"layer output: max={hiddenOutDiff.maxAbs:F6} mean={hiddenOutDiff.meanAbs:F8} idx={hiddenOutDiff.maxIdx}");

            Assert.True(ropeQDiff.meanAbs < 0.001f, $"Layer0 RoPE Q mean diff too large: {ropeQDiff.meanAbs}");
            Assert.True(ropeKDiff.meanAbs < 0.001f, $"Layer0 RoPE K mean diff too large: {ropeKDiff.meanAbs}");
            Assert.True(attnDiff.meanAbs < 0.01f, $"Layer0 attention mean diff too large: {attnDiff.meanAbs}");
            Assert.True(hiddenOutDiff.meanAbs < 0.01f, $"Layer0 output mean diff too large: {hiddenOutDiff.meanAbs}");
        }
        finally
        {
            if (dHidden != 0) CudaDriverApi.cuMemFree_v2(dHidden);
            if (dNormWeightF32 != 0) CudaDriverApi.cuMemFree_v2(dNormWeightF32);
            if (dNormWeightF16 != 0) CudaDriverApi.cuMemFree_v2(dNormWeightF16);
            if (dFfnNormWeightF32 != 0) CudaDriverApi.cuMemFree_v2(dFfnNormWeightF32);
            if (dFfnNormWeightF16 != 0) CudaDriverApi.cuMemFree_v2(dFfnNormWeightF16);
            if (dNorm != 0) CudaDriverApi.cuMemFree_v2(dNorm);
            if (dResidual != 0) CudaDriverApi.cuMemFree_v2(dResidual);
            if (dPositions != 0) CudaDriverApi.cuMemFree_v2(dPositions);
            if (dQRaw != 0) CudaDriverApi.cuMemFree_v2(dQRaw);
            if (dKRaw != 0) CudaDriverApi.cuMemFree_v2(dKRaw);
            if (dVRaw != 0) CudaDriverApi.cuMemFree_v2(dVRaw);
            if (dORaw != 0) CudaDriverApi.cuMemFree_v2(dORaw);
            if (dGateRaw != 0) CudaDriverApi.cuMemFree_v2(dGateRaw);
            if (dUpRaw != 0) CudaDriverApi.cuMemFree_v2(dUpRaw);
            if (dDownRaw != 0) CudaDriverApi.cuMemFree_v2(dDownRaw);
            if (dWF16 != 0) CudaDriverApi.cuMemFree_v2(dWF16);
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dAttn != 0) CudaDriverApi.cuMemFree_v2(dAttn);
            if (dGate != 0) CudaDriverApi.cuMemFree_v2(dGate);
            if (dUp != 0) CudaDriverApi.cuMemFree_v2(dUp);
            if (dSilu != 0) CudaDriverApi.cuMemFree_v2(dSilu);
            if (dHiddenOut != 0) CudaDriverApi.cuMemFree_v2(dHiddenOut);
        }
    }

    [SkippableFact]
    public void IQ4_XS_Layer0PrefillF32_MatchesCpuSemantics()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf");
        Skip.IfNot(File.Exists(modelPath), "Meta-Llama-3.1-8B IQ4_XS GGUF not found");

        int[] promptTokens = [128000, 791, 6864, 315, 9822, 374];
        int[] positions = [0, 1, 2, 3, 4, 5];

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var weights = TransformerWeights.LoadFromGguf(gguf, config, skipF32MoeDequant: true);
        ref readonly var lw = ref weights.Layers[0];

        int seqLen = promptTokens.Length;
        int hidden = config.HiddenSize;
        int qDim = lw.QOutputDim;
        int kvDim = lw.KOutputDim;
        int headDim = config.HeadDim;
        int ropeDim = config.RoPEConfig?.DimensionCount ?? headDim;
        if (ropeDim == 0) ropeDim = headDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int ropeType = (int)(config.RoPEConfig?.Type ?? RoPEType.Norm);

        float[] hiddenCpu = new float[seqLen * hidden];
        LoadEmbeddingRows(weights.TokenEmbedWeight, weights.TokenEmbedQuantType, promptTokens, hidden, hiddenCpu);

        float[] normCpu = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++)
        {
            RmsNorm.Execute(hiddenCpu.AsSpan(t * hidden, hidden), lw.AttnNormWeight,
                config.NormEpsilon, normCpu.AsSpan(t * hidden, hidden));
        }

        float[] qCpu = ComputeProjectionReference(lw.QWeight, lw.QQuantType, qDim, hidden, normCpu, seqLen);
        float[] kCpu = ComputeProjectionReference(lw.KWeight, lw.KQuantType, kvDim, hidden, normCpu, seqLen);
        float[] vCpu = ComputeProjectionReference(lw.VWeight, lw.VQuantType, kvDim, hidden, normCpu, seqLen);

        float[] cos = new float[config.MaxSequenceLength * (ropeDim / 2)];
        float[] sin = new float[cos.Length];
        RoPE.PrecomputeFrequencyTable(config.MaxSequenceLength, ropeDim, ropeTheta, cos, sin);
        RoPE.Execute(qCpu, kCpu, positions, config.NumAttentionHeads, config.NumKvHeads,
            headDim, ropeDim, cos, sin, config.RoPEConfig?.Type ?? RoPEType.Norm);

        float[] attnCpu = new float[seqLen * qDim];
        unsafe
        {
            fixed (float* pAttn = attnCpu)
            fixed (float* pQ = qCpu)
            fixed (float* pK = kCpu)
            fixed (float* pV = vCpu)
            {
                Attention.Execute(pQ, pK, pV, pAttn, seqLen, seqLen,
                    config.NumAttentionHeads, config.NumKvHeads, headDim, 0, null,
                    config.SlidingWindowSize);
            }
        }

        float[] oCpu = ComputeProjectionReference(lw.OWeight, lw.OQuantType, hidden, qDim, attnCpu, seqLen);
        float[] postAttnCpu = AddFloatReference(hiddenCpu, oCpu);

        float[] ffnNormCpu = new float[seqLen * hidden];
        for (int t = 0; t < seqLen; t++)
        {
            RmsNorm.Execute(postAttnCpu.AsSpan(t * hidden, hidden), lw.FfnNormWeight,
                config.NormEpsilon, ffnNormCpu.AsSpan(t * hidden, hidden));
        }

        float[] gateCpu = ComputeProjectionReference(lw.GateWeight, lw.GateQuantType,
            config.IntermediateSize, hidden, ffnNormCpu, seqLen);
        float[] upCpu = ComputeProjectionReference(lw.UpWeight, lw.UpQuantType,
            config.IntermediateSize, hidden, ffnNormCpu, seqLen);
        float[] siluCpu = SwiGluReference(gateCpu, upCpu);
        float[] downCpu = ComputeProjectionReference(lw.DownWeight, lw.DownQuantType,
            hidden, config.IntermediateSize, siluCpu, seqLen);
        float[] hiddenOutCpu = AddFloatReference(postAttnCpu, downCpu);

        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);
        using var kernels = new CudaKernels(ResolvePtxDir());

        nint dHidden = 0, dNormWeight = 0, dFfnNormWeight = 0, dNorm = 0, dResidual = 0, dPositions = 0;
        nint dQRaw = 0, dKRaw = 0, dVRaw = 0, dORaw = 0, dGateRaw = 0, dUpRaw = 0, dDownRaw = 0;
        nint dWF16 = 0, dWF32 = 0, dQ = 0, dK = 0, dV = 0, dAttn = 0, dGate = 0, dUp = 0, dSilu = 0, dDown = 0;
        try
        {
            long hiddenBytes = (long)seqLen * hidden * sizeof(float);
            long qBytes = (long)seqLen * qDim * sizeof(float);
            long kvBytes = (long)seqLen * kvDim * sizeof(float);
            long intermediateBytes = (long)seqLen * config.IntermediateSize * sizeof(float);
            long maxWeightF32Bytes = (long)Math.Max(config.IntermediateSize, hidden) * hidden * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dHidden, (nuint)hiddenBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dResidual, (nuint)hiddenBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dNormWeight, (nuint)(hidden * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dFfnNormWeight, (nuint)(hidden * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dNorm, (nuint)hiddenBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPositions, (nuint)(positions.Length * sizeof(int))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQRaw, (nuint)(Dequantize.RowByteSize(hidden, lw.QQuantType) * (long)qDim)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dKRaw, (nuint)(Dequantize.RowByteSize(hidden, lw.KQuantType) * (long)kvDim)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dVRaw, (nuint)(Dequantize.RowByteSize(hidden, lw.VQuantType) * (long)kvDim)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dORaw, (nuint)(Dequantize.RowByteSize(qDim, lw.OQuantType) * (long)hidden)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dGateRaw, (nuint)(Dequantize.RowByteSize(hidden, lw.GateQuantType) * (long)config.IntermediateSize)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dUpRaw, (nuint)(Dequantize.RowByteSize(hidden, lw.UpQuantType) * (long)config.IntermediateSize)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDownRaw, (nuint)(Dequantize.RowByteSize(config.IntermediateSize, lw.DownQuantType) * (long)hidden)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dWF16, (nuint)(maxWeightF32Bytes / 2)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dWF32, (nuint)maxWeightF32Bytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dAttn, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dGate, (nuint)intermediateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dUp, (nuint)intermediateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dSilu, (nuint)intermediateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDown, (nuint)hiddenBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = hiddenCpu)
                {
                    CudaDriverApi.cuMemcpyHtoD_v2(dHidden, (nint)p, (nuint)hiddenBytes).ThrowOnError();
                    CudaDriverApi.cuMemcpyHtoD_v2(dResidual, (nint)p, (nuint)hiddenBytes).ThrowOnError();
                }
                fixed (float* p = lw.AttnNormWeight)
                    CudaDriverApi.cuMemcpyHtoD_v2(dNormWeight, (nint)p, (nuint)(hidden * sizeof(float))).ThrowOnError();
                fixed (float* p = lw.FfnNormWeight)
                    CudaDriverApi.cuMemcpyHtoD_v2(dFfnNormWeight, (nint)p, (nuint)(hidden * sizeof(float))).ThrowOnError();
                fixed (int* p = positions)
                    CudaDriverApi.cuMemcpyHtoD_v2(dPositions, (nint)p, (nuint)(positions.Length * sizeof(int))).ThrowOnError();
            }

            CudaDriverApi.cuMemcpyHtoD_v2(dQRaw, lw.QWeight, (nuint)(Dequantize.RowByteSize(hidden, lw.QQuantType) * (long)qDim)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dKRaw, lw.KWeight, (nuint)(Dequantize.RowByteSize(hidden, lw.KQuantType) * (long)kvDim)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dVRaw, lw.VWeight, (nuint)(Dequantize.RowByteSize(hidden, lw.VQuantType) * (long)kvDim)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dORaw, lw.OWeight, (nuint)(Dequantize.RowByteSize(qDim, lw.OQuantType) * (long)hidden)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dGateRaw, lw.GateWeight, (nuint)(Dequantize.RowByteSize(hidden, lw.GateQuantType) * (long)config.IntermediateSize)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dUpRaw, lw.UpWeight, (nuint)(Dequantize.RowByteSize(hidden, lw.UpQuantType) * (long)config.IntermediateSize)).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoD_v2(dDownRaw, lw.DownWeight, (nuint)(Dequantize.RowByteSize(config.IntermediateSize, lw.DownQuantType) * (long)hidden)).ThrowOnError();

            kernels.LaunchRmsNormF32(dHidden, dNormWeight, dNorm, hidden, config.NormEpsilon, seqLen, stream.Handle);
            ProjectF32(kernels, cublas, stream, dQRaw, lw.QQuantType, dWF16, dWF32, dNorm, dQ, qDim, hidden, seqLen);
            ProjectF32(kernels, cublas, stream, dKRaw, lw.KQuantType, dWF16, dWF32, dNorm, dK, kvDim, hidden, seqLen);
            ProjectF32(kernels, cublas, stream, dVRaw, lw.VQuantType, dWF16, dWF32, dNorm, dV, kvDim, hidden, seqLen);
            stream.Synchronize();
            ReportDiff("norm f32", CopyFloatDevice(dNorm, seqLen * hidden), normCpu);
            ReportDiff("q f32", CopyFloatDevice(dQ, seqLen * qDim), ComputeProjectionReference(lw.QWeight, lw.QQuantType, qDim, hidden, normCpu, seqLen));
            ReportDiff("k f32", CopyFloatDevice(dK, seqLen * kvDim), ComputeProjectionReference(lw.KWeight, lw.KQuantType, kvDim, hidden, normCpu, seqLen));
            ReportDiff("v f32", CopyFloatDevice(dV, seqLen * kvDim), ComputeProjectionReference(lw.VWeight, lw.VQuantType, kvDim, hidden, normCpu, seqLen));

            kernels.LaunchRoPEF32(dQ, dK, dPositions, seqLen, config.NumAttentionHeads, config.NumKvHeads,
                headDim, ropeDim, ropeTheta, ropeType, stream.Handle);
            stream.Synchronize();
            ReportDiff("q rope f32", CopyFloatDevice(dQ, seqLen * qDim), qCpu);
            ReportDiff("k rope f32", CopyFloatDevice(dK, seqLen * kvDim), kCpu);
            kernels.LaunchAttentionF32(dQ, dK, dV, dAttn, seqLen, seqLen, config.NumAttentionHeads,
                config.NumKvHeads, headDim, 0, config.SlidingWindowSize ?? 0, stream.Handle);
            ProjectF32(kernels, cublas, stream, dORaw, lw.OQuantType, dWF16, dWF32, dAttn, dNorm, hidden, qDim, seqLen);
            kernels.LaunchAddF32(dResidual, dNorm, dResidual, seqLen * hidden, stream.Handle);
            kernels.LaunchRmsNormF32(dResidual, dFfnNormWeight, dNorm, hidden, config.NormEpsilon, seqLen, stream.Handle);
            ProjectF32(kernels, cublas, stream, dGateRaw, lw.GateQuantType, dWF16, dWF32, dNorm, dGate, config.IntermediateSize, hidden, seqLen);
            ProjectF32(kernels, cublas, stream, dUpRaw, lw.UpQuantType, dWF16, dWF32, dNorm, dUp, config.IntermediateSize, hidden, seqLen);
            kernels.LaunchSwiGLUF32(dGate, dUp, dSilu, config.IntermediateSize, seqLen, stream.Handle);
            ProjectF32(kernels, cublas, stream, dDownRaw, lw.DownQuantType, dWF16, dWF32, dSilu, dDown, hidden, config.IntermediateSize, seqLen);
            kernels.LaunchAddF32(dResidual, dDown, dResidual, seqLen * hidden, stream.Handle);
            stream.Synchronize();

            ReportDiff("attn f32", CopyFloatDevice(dAttn, seqLen * qDim), attnCpu);
            ReportDiff("o f32", CopyFloatDevice(dNorm, seqLen * hidden), ffnNormCpu);
            var hiddenOutDiff = ReportDiff("hidden out f32", CopyFloatDevice(dResidual, seqLen * hidden), hiddenOutCpu);
            Assert.True(hiddenOutDiff.meanAbs < 0.001f,
                $"F32 layer output mean diff too large: {hiddenOutDiff.meanAbs}");
        }
        finally
        {
            foreach (ref nint ptr in new Span<nint>([
                dHidden, dNormWeight, dFfnNormWeight, dNorm, dResidual, dPositions,
                dQRaw, dKRaw, dVRaw, dORaw, dGateRaw, dUpRaw, dDownRaw,
                dWF16, dWF32, dQ, dK, dV, dAttn, dGate, dUp, dSilu, dDown]))
            {
                if (ptr != 0) CudaDriverApi.cuMemFree_v2(ptr);
            }
        }
    }

    private static string ResolvePtxDir()
    {
        string? envDir = Environment.GetEnvironmentVariable("DOTLLM_PTX_DIR");
        if (envDir is not null && Directory.Exists(envDir))
            return envDir;

        string repoRoot = Path.GetDirectoryName(typeof(CudaIQ4_XS_DequantUnitTest).Assembly.Location)!;
        while (repoRoot.Length > 3 && !File.Exists(Path.Combine(repoRoot, "dotLLM.slnx")))
            repoRoot = Path.GetDirectoryName(repoRoot)!;

        string ptxDir = Path.Combine(repoRoot, "native", "ptx");
        return ptxDir;
    }

    private static unsafe void LoadEmbeddingRows(nint embedding, QuantizationType qt, int[] tokenIds, int hidden, float[] dest)
    {
        long rowBytes = Dequantize.RowByteSize(hidden, qt);
        for (int t = 0; t < tokenIds.Length; t++)
        {
            Dequantize.ToFloat32(embedding + tokenIds[t] * (nint)rowBytes, hidden, qt,
                dest.AsSpan(t * hidden, hidden));
        }
    }

    private static unsafe float[] ComputeProjectionReference(
        nint weight, QuantizationType qt, int outputDim, int inputDim, float[] input, int seqLen)
    {
        long rowBytes = Dequantize.RowByteSize(inputDim, qt);
        float[] result = new float[seqLen * outputDim];
        float[] row = new float[inputDim];

        for (int o = 0; o < outputDim; o++)
        {
            Dequantize.ToFloat32(weight + o * (nint)rowBytes, inputDim, qt, row);
            for (int t = 0; t < seqLen; t++)
            {
                float acc = 0;
                int inputOffset = t * inputDim;
                for (int i = 0; i < inputDim; i++)
                    acc += row[i] * input[inputOffset + i];
                result[t * outputDim + o] = acc;
            }
        }

        return result;
    }

    private static void ProjectF16(CudaKernels kernels, CudaCublasHandle cublas, CudaStream stream,
        nint rawWeight, QuantizationType qt, nint weightF16Scratch, nint inputF16, nint outputF16,
        int outputDim, int inputDim, int seqLen)
    {
        kernels.LaunchDequantToF16(rawWeight, qt, weightF16Scratch, outputDim * inputDim, stream.Handle);
        CudaGemm.LinearF16(cublas.Handle, inputF16, weightF16Scratch, outputF16,
            seqLen, inputDim, outputDim, stream.Handle);
    }

    private static void ProjectF32(CudaKernels kernels, CudaCublasHandle cublas, CudaStream stream,
        nint rawWeight, QuantizationType qt, nint weightF16Scratch, nint weightF32Scratch, nint inputF32, nint outputF32,
        int outputDim, int inputDim, int seqLen)
    {
        if (qt is QuantizationType.IQ4_NL or QuantizationType.IQ4_XS or QuantizationType.Q5_K)
        {
            kernels.LaunchDequantToF32(rawWeight, qt, weightF32Scratch, outputDim * inputDim, stream.Handle);
        }
        else
        {
            kernels.LaunchDequantToF16(rawWeight, qt, weightF16Scratch, outputDim * inputDim, stream.Handle);
            kernels.LaunchConvertF16ToF32(weightF16Scratch, weightF32Scratch, outputDim * inputDim, stream.Handle);
        }
        CudaGemm.LinearF32(cublas.Handle, inputF32, weightF32Scratch, outputF32,
            seqLen, inputDim, outputDim, stream.Handle);
    }

    private static unsafe Half[] CopyHalfDevice(nint devicePtr, int elementCount)
    {
        Half[] result = new Half[elementCount];
        fixed (Half* p = result)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(elementCount * sizeof(ushort))).ThrowOnError();
        return result;
    }

    private static unsafe float[] CopyFloatDevice(nint devicePtr, int elementCount)
    {
        float[] result = new float[elementCount];
        fixed (float* p = result)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(elementCount * sizeof(float))).ThrowOnError();
        return result;
    }

    private static float[] HalfArrayToFloat(Half[] values)
    {
        float[] result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
            result[i] = (float)values[i];
        return result;
    }

    private static Half[] FusedAddRmsNormReference(
        Half[] residual, Half[] x, float[] weight, float eps, int rows, int n, float[] output)
    {
        Half[] residualOut = new Half[residual.Length];
        for (int row = 0; row < rows; row++)
        {
            int offset = row * n;
            float sumSq = 0;
            float[] sum = new float[n];
            for (int i = 0; i < n; i++)
            {
                float s = (float)residual[offset + i] + (float)x[offset + i];
                sum[i] = s;
                residualOut[offset + i] = (Half)s;
                sumSq += s * s;
            }

            float rmsInv = 1.0f / MathF.Sqrt(sumSq / n + eps);
            for (int i = 0; i < n; i++)
                output[offset + i] = sum[i] * rmsInv * weight[i];
        }

        return residualOut;
    }

    private static float[] SwiGluReference(Half[] gate, Half[] up)
    {
        float[] result = new float[gate.Length];
        for (int i = 0; i < gate.Length; i++)
        {
            float g = (float)gate[i];
            float u = (float)up[i];
            float silu = g / (1.0f + MathF.Exp(-g));
            result[i] = silu * u;
        }

        return result;
    }

    private static float[] SwiGluReference(float[] gate, float[] up)
    {
        float[] result = new float[gate.Length];
        for (int i = 0; i < gate.Length; i++)
        {
            float g = gate[i];
            float silu = g / (1.0f + MathF.Exp(-g));
            result[i] = silu * up[i];
        }

        return result;
    }

    private static float[] AddFloatReference(float[] a, float[] b)
    {
        float[] result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] + b[i];
        return result;
    }

    private (float maxAbs, float meanAbs, int maxIdx) ReportDiff(string label, float[] actual, float[] expected)
    {
        var diff = CompareFloatArrays(actual, expected);
        _output.WriteLine($"{label}: max={diff.maxAbs:F6} mean={diff.meanAbs:F8} idx={diff.maxIdx}");
        return diff;
    }

    private static Half[] AddHalfReference(Half[] a, Half[] b)
    {
        Half[] result = new Half[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = (Half)((float)a[i] + (float)b[i]);
        return result;
    }

    private static (float maxAbs, float meanAbs, int maxIdx) CompareHalfToFloat(Half[] actual, float[] expected)
    {
        float maxAbs = 0;
        float sum = 0;
        int maxIdx = -1;
        for (int i = 0; i < actual.Length; i++)
        {
            float diff = MathF.Abs((float)actual[i] - expected[i]);
            sum += diff;
            if (diff > maxAbs)
            {
                maxAbs = diff;
                maxIdx = i;
            }
        }

        return (maxAbs, sum / actual.Length, maxIdx);
    }

    private static (float maxAbs, float meanAbs, int maxIdx) CompareHalfArrays(Half[] actual, Half[] expected)
    {
        float maxAbs = 0;
        float sum = 0;
        int maxIdx = -1;
        for (int i = 0; i < actual.Length; i++)
        {
            float diff = MathF.Abs((float)actual[i] - (float)expected[i]);
            sum += diff;
            if (diff > maxAbs)
            {
                maxAbs = diff;
                maxIdx = i;
            }
        }

        return (maxAbs, sum / actual.Length, maxIdx);
    }

    private static (float maxAbs, float meanAbs, int maxIdx) CompareFloatArrays(float[] actual, float[] expected)
    {
        float maxAbs = 0;
        float sum = 0;
        int maxIdx = -1;
        for (int i = 0; i < actual.Length; i++)
        {
            float diff = MathF.Abs(actual[i] - expected[i]);
            sum += diff;
            if (diff > maxAbs)
            {
                maxAbs = diff;
                maxIdx = i;
            }
        }

        return (maxAbs, sum / actual.Length, maxIdx);
    }

    private static void CreateTestIQ4_XSData(byte[] block)
    {
        if (block.Length != 136)
            throw new ArgumentException("IQ4_XS block must be 136 bytes");

        // d = 1.0 (as Half in little-endian)
        Half d = (Half)1.0f;
        MemoryMarshal.Write(new Span<byte>(block, 0, 2), ref d);

        // scales_h = 0x0000 (all sub-blocks use scale 32, bias cancels)
        MemoryMarshal.Write(new Span<byte>(block, 2, 2), (ushort)0);

        // scales_l[4] = 0x00: each nibble = 0 (low 4 bits of 6-bit scale)
        block[4] = 0;
        block[5] = 0;
        block[6] = 0;
        block[7] = 0;

        // qs[128]: packed nibbles. Use pattern of 0x00, 0x11, 0x22, ..., 0xFF
        byte[] qsData = new byte[128];
        for (int i = 0; i < 128; i++)
            qsData[i] = (byte)((i * 0x11) & 0xFF); // Simple ascending pattern

        Array.Copy(qsData, 0, block, 8, 128);
    }
}
