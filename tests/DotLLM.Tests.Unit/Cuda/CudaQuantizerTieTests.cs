using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1499: rounding contract of the CUDA quantizers - the pre-Q8_1 activation quantizers (<c>quantize_x_to_q8_1</c>,
/// <c>quantize_x_to_q8_1_batched</c>) and the KV-cache quantizers (<c>quant_f16_to_q8_0</c>, <c>quant_f16_to_q4_0</c>).
/// Contract (== CPU reference, == Vulkan after #1340/#1500): <c>d = amax / qmax</c> (IEEE), <c>id = 1 / d</c> (IEEE),
/// <c>q = roundHalfToEven(x * id)</c>, stored scale <c>half(d)</c>. The pre-fix kernels multiplied by <c>127 / amax</c>
/// (a differently rounded reciprocal) while storing <c>amax / 127</c>; <c>QuantizerConventionSensitivityTests</c> shows
/// that convention flips lanes on this very corpus, so this test fails against the old PTX. Compares int8 lanes and the
/// FP16 scale bits against the CPU scalar reference run on the same Half-rounded payload.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaQuantizerTieTests
{
    private const int Group = 32;
    private readonly ITestOutputHelper _out;
    public CudaQuantizerTieTests(ITestOutputHelper output) => _out = output;

    private void Compare(string kernel, float[] x, string[] labels, int[] cpuQ, short[] cpuD, int[] gpuQ, short[] gpuD)
    {
        int badQ = 0, badD = 0, firstBad = -1;
        for (int b = 0; b < labels.Length; b++)
        {
            bool dBad = cpuD[b] != gpuD[b];
            int laneBad = 0;
            for (int i = 0; i < Group; i++) if (cpuQ[b * Group + i] != gpuQ[b * Group + i]) laneBad++;
            badQ += laneBad;
            if (dBad) badD++;
            if ((dBad || laneBad > 0) && firstBad < 0) firstBad = b;
        }
        if (firstBad >= 0)
        {
            int b = firstBad;
            _out.WriteLine($"[{kernel}] first mismatching block {b} '{labels[b]}': d cpu=0x{cpuD[b]:x4} gpu=0x{gpuD[b]:x4}");
            for (int i = 0; i < Group; i++)
                if (cpuQ[b * Group + i] != gpuQ[b * Group + i])
                    _out.WriteLine($"   lane {i}: x={x[b * Group + i]:R} cpu q={cpuQ[b * Group + i]} gpu q={gpuQ[b * Group + i]}");
        }
        _out.WriteLine($"[{kernel}] blocks={labels.Length} mismatching lanes={badQ} mismatching scales={badD}");
        Assert.True(badQ == 0 && badD == 0, $"[{kernel}] {badQ} lanes / {badD} scales differ from the CPU reference (first block {firstBad}).");
    }

    private static Half[] ToHalf(float[] x)
    {
        var h = new Half[x.Length];
        for (int i = 0; i < x.Length; i++) h[i] = (Half)x[i];
        return h;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(d, 0xA5, (nuint)bytes).ThrowOnError();   // poison: an unwritten block cannot pass
        return d;
    }

    private static void Upload<T>(nint dst, T[] src) where T : unmanaged
    {
        fixed (T* p = src) CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(src.Length * sizeof(T))).ThrowOnError();
    }

    private static byte[] Download(nint src, long bytes)
    {
        var b = new byte[bytes];
        fixed (byte* p = b) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)bytes).ThrowOnError();
        return b;
    }

    /// <summary>CPU Q8_1 reference: int8 lanes and the Half scale bits per block.</summary>
    private static (int[] Q, short[] D) CpuQ8_1(float[] x)
    {
        int blocks = x.Length / Group;
        var dest = new byte[blocks * 36];
        fixed (float* src = x) fixed (byte* dst = dest) MatMul.QuantizeF32ToQ8_1Scalar(src, dst, x.Length);
        var q = new int[x.Length];
        var d = new short[blocks];
        for (int b = 0; b < blocks; b++)
        {
            d[b] = BitConverter.ToInt16(dest, b * 36);
            for (int i = 0; i < Group; i++) q[b * Group + i] = (sbyte)dest[b * 36 + 4 + i];
        }
        return (q, d);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuantizeXToQ8_1_MatchesCpuReference_OnTieCorpus(bool batched)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        float[] x = ActivationQuantizerCorpus.BuildHalfCorpus(127, out string[] labels);
        var cpu = CpuQ8_1(x);
        int blocks = labels.Length;

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        Skip.IfNot(kernels.HasPreQ8_1, "quantize_x.ptx not loaded");

        // Single-row: k = whole corpus (one row of `blocks` chunks). Batched: m rows of one chunk each (row stride 32),
        // which exercises the per-token section layout (xq[m,k] | dx[m,k/32] | sx2[m,k/16]).
        int k = batched ? Group : x.Length;
        int m = batched ? blocks : 1;
        long scratchBytes = (long)m * CudaForwardState.PreQ8_1ScratchBytes(k);
        nint dX = Alloc((long)x.Length * 2), dS = Alloc(scratchBytes);
        try
        {
            Upload(dX, ToHalf(x));
            if (batched) kernels.LaunchQuantizeXToQ8_1Batched(dX, dS, k, m, stream.Handle);
            else kernels.LaunchQuantizeXToQ8_1(dX, dS, k, stream.Handle);
            stream.Synchronize();
            byte[] s = Download(dS, scratchBytes);
            // Both layouts put all xq bytes first (x.Length of them) and all dx halves next (blocks of them).
            var q = new int[x.Length];
            for (int i = 0; i < x.Length; i++) q[i] = (sbyte)s[i];
            var d = new short[blocks];
            for (int b = 0; b < blocks; b++) d[b] = BitConverter.ToInt16(s, x.Length + 2 * b);
            Compare(batched ? "quantize_x_to_q8_1_batched" : "quantize_x_to_q8_1", x, labels, cpu.Q, cpu.D, q, d);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dX); CudaDriverApi.cuMemFree_v2(dS);
        }
    }

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void QuantKv_MatchesCpuReference_OnTieCorpus(KvCacheDType dtype)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        int qmax = dtype == KvCacheDType.Q8_0 ? 127 : 7;
        int blockBytes = dtype == KvCacheDType.Q8_0 ? 34 : 18;
        float[] x = ActivationQuantizerCorpus.BuildHalfCorpus(qmax, out string[] labels);
        int blocks = labels.Length;

        // CPU reference bytes in the same GGUF block layout (half d + payload).
        var expected = new byte[blocks * blockBytes];
        fixed (float* src = x) fixed (byte* dst = expected)
        {
            if (dtype == KvCacheDType.Q8_0) MatMul.QuantizeF32ToQ8_0Scalar(src, dst, x.Length);
            else KvQuantize.F32ToQ4_0Scalar(src, dst, x.Length);
        }

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        nint dX = Alloc((long)x.Length * 2), dOut = Alloc(expected.Length);
        try
        {
            Upload(dX, ToHalf(x));
            kernels.LaunchQuantKv(dX, dOut, x.Length, dtype, stream.Handle);
            stream.Synchronize();
            byte[] actual = Download(dOut, expected.Length);
            Unpack(dtype, expected, out int[] cq, out short[] cd);
            Unpack(dtype, actual, out int[] gq, out short[] gd);
            Compare(dtype == KvCacheDType.Q8_0 ? "quant_f16_to_q8_0" : "quant_f16_to_q4_0", x, labels, cq, cd, gq, gd);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dX); CudaDriverApi.cuMemFree_v2(dOut);
        }
    }

    private static void Unpack(KvCacheDType dtype, byte[] blocksBytes, out int[] q, out short[] d)
    {
        int bb = dtype == KvCacheDType.Q8_0 ? 34 : 18;
        int blocks = blocksBytes.Length / bb;
        q = new int[blocks * Group];
        d = new short[blocks];
        for (int b = 0; b < blocks; b++)
        {
            d[b] = BitConverter.ToInt16(blocksBytes, b * bb);
            if (dtype == KvCacheDType.Q8_0)
                for (int i = 0; i < Group; i++) q[b * Group + i] = (sbyte)blocksBytes[b * bb + 2 + i];
            else
                for (int j = 0; j < 16; j++)
                {
                    byte p = blocksBytes[b * bb + 2 + j];
                    q[b * Group + 2 * j] = p & 0xF;
                    q[b * Group + 2 * j + 1] = p >> 4;
                }
        }
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }
}
