using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Q3_K CPU↔CUDA dequant parity on <b>real GGUF bytes</b>.
/// </summary>
/// <remarks>
/// <para>
/// Every Q3_K kernel test that existed before #311 decoded bytes produced by a packer we
/// wrote ourselves, which is exactly how the transposed Q3_K layout shipped: the fixture
/// encoded with the same wrong layout the kernels decoded with, so the family passed green
/// while real weights decoded to noise (correlation 0.006 against the true weights). A
/// fixture only ever exercises the byte patterns its own quantiser happens to emit.
/// </para>
/// <para>
/// This test takes its input from a real <c>llama-quantize</c>-produced GGUF, so the bytes
/// come from the authoritative quantiser rather than from us, and compares the CUDA
/// <c>dequant_q3_k_f16</c> kernel against <see cref="Dequantize.DequantizeQ3_K"/>. The CPU
/// path is a legitimate oracle: it is independently anchored to a literal transcription of
/// llama.cpp's <c>dequantize_row_q3_K</c> by
/// <c>DequantizeKQuantTests.Q3_K_DenseRandomBlocks_MatchLlamaCppReference</c>.
/// </para>
/// <para>
/// Issue #318 motivation: the committed <c>native/ptx/dequant.ptx</c> is a build artifact, so
/// the #311 CUDA source fix is only real if the artifact the loader reads was rebuilt from it.
/// This is the on-hardware check of that, on bytes nobody in this repository encoded. It is the
/// CUDA counterpart of <c>RealGgufQ3KDequantParityTests</c> (Vulkan, #320).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class RealGgufQ3KCudaDequantParityTests
{
    private const int Q3KGroupSize = 256;
    private const int Q3KBlockBytes = 110;

    /// <summary>Cap per tensor so the test stays seconds, not minutes.</summary>
    private const int MaxBlocksPerTensor = 4096;

    private readonly ITestOutputHelper _output;

    public RealGgufQ3KCudaDequantParityTests(ITestOutputHelper output) => _output = output;

    // NOTE: a filename claiming Q3_K is not evidence that any Q3_K tensor is present —
    // mradermacher's SmolLM-135M "i1-Q3_K_M" carries none at all (IQ4_NL/Q5_0/Q4_K only).
    // The Assert on tensorsChecked below is what actually establishes that bytes were compared.
    [SkippableFact]
    public void Q3K_RealGgufBytes_CudaDequant_MatchesCpuOracle()
    {
        string? path = ResolveQ3KFixture();
        Skip.If(path is null,
            "No Q3_K GGUF fixture. Set DOTLLM_QUANT_FIXTURE_Q3_K, or generate the quant ladder "
            + "per docs/QUANT_FIXTURES.md into ~/.dotllm/quant-ladder/Llama-3.2-1B-pure/.");
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using GgufFile gguf = GgufFile.Open(path!);
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        // Size the device buffers to the largest case up front and reuse them across
        // tensors. A per-tensor allocate/free loop recycles device pointers, which has
        // already produced a convincing but entirely spurious "correct prefix, then
        // zeros" failure signature elsewhere in this codebase.
        int maxBlocks = 0;
        foreach (GgufTensorDescriptor t in gguf.Tensors)
        {
            if (t.QuantizationType != QuantizationType.Q3_K) continue;
            long elements = t.Shape.ElementCount;
            if (elements % Q3KGroupSize != 0) continue;
            maxBlocks = Math.Max(maxBlocks, (int)Math.Min(elements / Q3KGroupSize, MaxBlocksPerTensor));
        }

        Assert.True(maxBlocks > 0,
            $"'{Path.GetFileName(path)}' contains no Q3_K tensor with a whole number of "
            + "256-element super-blocks — this fixture cannot establish anything about the Q3_K path.");

        nint dSrc = 0, dDst = 0;
        int tensorsChecked = 0;
        long elementsChecked = 0;
        long mismatchTotal = 0;
        double worstAbs = 0;
        string worstWhere = "(none)";
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;

        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)((long)maxBlocks * Q3KBlockBytes)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDst, (nuint)((long)maxBlocks * Q3KGroupSize * sizeof(ushort))).ThrowOnError();

            foreach (GgufTensorDescriptor tensor in gguf.Tensors)
            {
                if (tensor.QuantizationType != QuantizationType.Q3_K) continue;

                long totalElements = tensor.Shape.ElementCount;
                if (totalElements % Q3KGroupSize != 0) continue;

                int blocks = (int)Math.Min(totalElements / Q3KGroupSize, MaxBlocksPerTensor);
                int elementCount = blocks * Q3KGroupSize;
                int byteCount = blocks * Q3KBlockBytes;

                byte[] raw = new byte[byteCount];
                unsafe
                {
                    new ReadOnlySpan<byte>((void*)(gguf.DataBasePointer + (nint)tensor.DataOffset), byteCount)
                        .CopyTo(raw);
                }

                float[] cpu = new float[elementCount];
                unsafe
                {
                    fixed (byte* rawPtr = raw)
                        Dequantize.DequantizeQ3_K((nint)rawPtr, elementCount, cpu);
                }

                ushort[] gpuBits = new ushort[elementCount];
                unsafe
                {
                    fixed (byte* rawPtr = raw)
                        CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)rawPtr, (nuint)byteCount).ThrowOnError();

                    kernels.LaunchDequantToF16(dSrc, QuantizationType.Q3_K, dDst, elementCount, stream.Handle);
                    stream.Synchronize();

                    fixed (ushort* p = gpuBits)
                        CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dDst, (nuint)((long)elementCount * sizeof(ushort))).ThrowOnError();
                }

                for (int i = 0; i < elementCount; i++)
                {
                    float gpu = (float)BitConverter.UInt16BitsToHalf(gpuBits[i]);
                    if (BitConverter.HalfToUInt16Bits((Half)cpu[i]) != gpuBits[i])
                    {
                        mismatchTotal++;
                        double abs = Math.Abs(cpu[i] - gpu);
                        if (abs > worstAbs)
                        {
                            worstAbs = abs;
                            worstWhere = $"{tensor.Name}[{i}] (cpu={cpu[i]:R} cuda={gpu:R})";
                        }
                    }

                    double x = cpu[i], y = gpu;
                    sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
                }

                elementsChecked += elementCount;
                tensorsChecked++;
            }
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dDst != 0) CudaDriverApi.cuMemFree_v2(dDst);
        }

        Assert.True(tensorsChecked > 0, "No Q3_K tensor was actually compared.");

        double n = elementsChecked;
        double cov = (sxy / n) - ((sx / n) * (sy / n));
        double varX = (sxx / n) - ((sx / n) * (sx / n));
        double varY = (syy / n) - ((sy / n) * (sy / n));
        double corr = (varX > 0 && varY > 0) ? cov / Math.Sqrt(varX * varY) : double.NaN;

        _output.WriteLine(
            $"{Path.GetFileName(path)}: {tensorsChecked} Q3_K tensors, {elementsChecked} elements, "
            + $"mismatches={mismatchTotal}, worst-abs={worstAbs:E3}, corr={corr:F6}");

        // Both sides evaluate (d * signedScale) * signed3 — pure FP32 multiplication in the
        // same order, then round-to-nearest-even into FP16 — so agreement is exact, not
        // approximate. Correlation is reported because it is the statistic that exposed #311
        // (0.006 broken vs 0.988 fixed): a layout disagreement collapses it, rounding drift
        // would not.
        Assert.True(mismatchTotal == 0,
            $"{mismatchTotal} of {elementsChecked} elements across {tensorsChecked} real Q3_K tensors "
            + $"differ between the CUDA kernel and the llama.cpp-anchored CPU oracle. "
            + $"Worst: {worstWhere}, abs={worstAbs:E3}, correlation={corr:F6}. "
            + "A correlation far below 1 means the CUDA Q3_K bit layout disagrees with the CPU one "
            + "(see #311) — most likely native/ptx/dequant.ptx is a stale build of "
            + "native/kernels/dequant.cu (see #318).");
    }

    /// <summary>
    /// Env override <c>DOTLLM_QUANT_FIXTURE_Q3_K</c> first, else the conventional quant-ladder
    /// path documented in <c>docs/QUANT_FIXTURES.md</c>. Returns null (never throws) so the
    /// caller self-skips.
    /// </summary>
    private static string? ResolveQ3KFixture()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_QUANT_FIXTURE_Q3_K");
        if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string conventional = Path.Combine(
            home, ".dotllm", "quant-ladder", "Llama-3.2-1B-pure", "Llama-3.2-1B-pure-Q3_K.gguf");
        return File.Exists(conventional) ? conventional : null;
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static string? FindPtxDir()
    {
        string[] candidates =
        [
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        ];
        foreach (string dir in candidates)
        {
            string full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
