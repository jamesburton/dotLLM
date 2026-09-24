using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// PROBE for issue #532, CUDA half — does CUDA attention's output for a given
/// query position depend on the KV-cache length beyond that query's causally
/// visible prefix? Mirror of
/// <c>DotLLM.Tests.Unit.Vulkan.Probe532VulkanAttentionKvLengthTests</c>.
/// </summary>
/// <remarks>
/// <para>
/// Comparison is BITWISE. The CPU defect (#525) is 1.19E-07 — inside every
/// tolerance in this suite — so a tolerance comparison cannot answer this.
/// </para>
/// <para>
/// <b>UNRUN as of the #532 audit</b>: the audit box is an AMD Strix Halo with no
/// NVIDIA GPU, so both facts below skip locally. Run on the CUDA box (T5500):
/// <c>dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~Probe532Cuda"
/// --logger "console;verbosity=detailed"</c>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class Probe532CudaAttentionKvLengthTests
{
    private readonly ITestOutputHelper _out;
    public Probe532CudaAttentionKvLengthTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    private static float[] Rand(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    private static (long differing, float maxAbs) CompareBitwise(float[] a, float[] b)
    {
        long d = 0; float maxAbs = 0;
        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            { d++; maxAbs = MathF.Max(maxAbs, MathF.Abs(a[i] - b[i])); }
        }
        return (d, maxAbs);
    }

    /// <summary>
    /// CUDA DENSE <c>attention_f32</c>. Same K/V buffer, same query rows; only the
    /// declared <c>seqKv</c> changes. Prediction from the Vulkan twin
    /// (<c>attention_f32.comp</c>, measured bitwise-invariant across 22 shapes):
    /// invariant.
    /// </summary>
    [SkippableFact]
    public void Cuda_Dense_KvPadding()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        const int numHeads = 9, numKvHeads = 3, headDim = 64;
        var sb = new StringBuilder();
        sb.AppendLine("=== CUDA DENSE attention_f32 — KV padding ===");
        long worst = 0;

        foreach (var (v, kvPad, seqQ) in new[]
                 {
                     (3, 4, 3), (3, 5, 3), (3, 6, 3), (3, 7, 3),
                     (5, 6, 5), (5, 7, 5), (5, 8, 5), (5, 9, 5),
                     (17, 18, 17), (17, 19, 17), (17, 20, 17), (17, 21, 17),
                     (250, 251, 250), (250, 260, 250), (3, 300, 3), (250, 600, 250),
                     // decode shapes
                     (3, 5, 1), (5, 9, 1), (16, 260, 1),
                 })
        {
            int posOff = seqQ == 1 ? v - 1 : 0;
            var rng = new Random(0x532 + v * 31 + kvPad * 7 + seqQ);
            float[] q = Rand(rng, seqQ * numHeads * headDim);
            float[] k = Rand(rng, kvPad * numKvHeads * headDim);
            float[] vv = Rand(rng, kvPad * numKvHeads * headDim);
            int outLen = seqQ * numHeads * headDim;

            float[] a = new float[outLen], b = new float[outLen];
            RunDensePair(kernels, stream, q, k, vv, a, b,
                seqQ, v, kvPad, numHeads, numKvHeads, headDim, posOff);

            var (d, maxAbs) = CompareBitwise(a, b);
            worst = Math.Max(worst, d);
            sb.AppendLine($"  seqQ={seqQ,4} V={v,4} seqKv={kvPad,4}: differing={d,6}/{outLen,-7} maxAbs={maxAbs:E3}");
        }

        sb.AppendLine();
        sb.AppendLine(worst == 0
            ? "VERDICT: CUDA dense attention is BITWISE INVARIANT to KV padding."
            : $"VERDICT: CUDA dense attention IS KV-length dependent (worst {worst} elements).");
        _out.WriteLine(sb.ToString());
    }

    /// <summary>
    /// CUDA <c>attention_f16</c> (<c>LaunchAttention</c>) — the ACTUAL production dense
    /// path in <c>CudaTransformerModel</c> (the F32 kernel above is the F32-state variant).
    /// <c>attention.cu</c> is structurally the same kernel as <c>attention_f32.cu</c>:
    /// tiles from 0 in <c>TILE_KV</c> steps, masks with <c>-FLT_MAX</c>, fixed-width warp
    /// reduction over <c>nw = ceil(blockDim.x/warpSize)</c>, <c>score_tile[t] &gt; 0.0f</c>
    /// skip in the weighted-V loop. Predicted invariant.
    /// </summary>
    [SkippableFact]
    public void Cuda_DenseF16_KvPadding()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        const int numHeads = 9, numKvHeads = 3, headDim = 64;
        var sb = new StringBuilder();
        sb.AppendLine("=== CUDA attention_f16 (production dense path) — KV padding ===");
        long worst = 0;

        foreach (var (v, kvPad, seqQ, posOff) in new[]
                 {
                     (3, 4, 3, 0), (3, 5, 3, 0), (3, 7, 3, 0),
                     (5, 7, 5, 0), (5, 9, 5, 0),
                     (17, 19, 17, 0), (17, 21, 17, 0),
                     (250, 260, 250, 0), (3, 300, 3, 0),
                     // decode
                     (3, 5, 1, 2), (16, 260, 1, 15),
                     // chunked-prefill: later chunk at positionOffset > 0
                     (5, 7, 2, 3), (103, 105, 3, 100),
                 })
        {
            var rng = new Random(0x532 + v * 31 + kvPad * 7 + seqQ * 3 + posOff);
            ushort[] q = RandHalf(rng, seqQ * numHeads * headDim);
            ushort[] k = RandHalf(rng, kvPad * numKvHeads * headDim);
            ushort[] vv = RandHalf(rng, kvPad * numKvHeads * headDim);
            int outLen = seqQ * numHeads * headDim;
            ushort[] a = new ushort[outLen], b = new ushort[outLen];

            RunF16Pair(kernels, stream, q, k, vv, a, b,
                seqQ, v, kvPad, numHeads, numKvHeads, headDim, posOff);

            long d = 0; float maxAbs = 0;
            for (int i = 0; i < outLen; i++)
            {
                if (a[i] != b[i])
                { d++; maxAbs = MathF.Max(maxAbs, MathF.Abs((float)BitConverter.UInt16BitsToHalf(a[i]) - (float)BitConverter.UInt16BitsToHalf(b[i]))); }
            }
            worst = Math.Max(worst, d);
            sb.AppendLine($"  seqQ={seqQ,4} posOff={posOff,4} V={v,4} seqKv={kvPad,4}: differing={d,6}/{outLen,-7} maxAbs={maxAbs:E3}");
        }

        sb.AppendLine();
        sb.AppendLine(worst == 0
            ? "VERDICT: CUDA attention_f16 is BITWISE INVARIANT to KV padding."
            : $"VERDICT: CUDA attention_f16 IS KV-length dependent (worst {worst} elements).");
        _out.WriteLine(sb.ToString());
    }

    private static ushort[] RandHalf(Random rng, int n)
    {
        var a = new ushort[n];
        for (int i = 0; i < n; i++)
            a[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 2.0 - 1.0));
        return a;
    }

    private static unsafe void RunF16Pair(
        CudaKernels kernels, CudaStream stream,
        ushort[] q, ushort[] k, ushort[] v, ushort[] outA, ushort[] outB,
        int seqQ, int seqKvA, int seqKvB, int numHeads, int numKvHeads, int headDim, int posOff)
    {
        long qBytes = (long)q.Length * sizeof(ushort);
        long kvBytes = (long)k.Length * sizeof(ushort);
        long oBytes = (long)outA.Length * sizeof(ushort);
        nint dQ = 0, dK = 0, dV = 0, dA = 0, dB = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)oBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)oBytes).ThrowOnError();
            fixed (ushort* p = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
            fixed (ushort* p = k) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kvBytes).ThrowOnError();
            fixed (ushort* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)kvBytes).ThrowOnError();

            nint s = stream.Handle;
            kernels.LaunchAttention(dQ, dK, dV, dA, seqQ, seqKvA, numHeads, numKvHeads, headDim, posOff, 0, s);
            kernels.LaunchAttention(dQ, dK, dV, dB, seqQ, seqKvB, numHeads, numKvHeads, headDim, posOff, 0, s);
            stream.Synchronize();

            fixed (ushort* p = outA) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dA, (nuint)oBytes).ThrowOnError();
            fixed (ushort* p = outB) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dB, (nuint)oBytes).ThrowOnError();
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dA != 0) CudaDriverApi.cuMemFree_v2(dA);
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
        }
    }

    /// <summary>
    /// CUDA SPLIT-KV <c>attention_f32_split_kv</c> (opt-in, #183). Structurally
    /// identical exposure to the Vulkan split-KV kernel that MEASURED affected:
    /// <c>chunk = ceil(seq_kv / ATTN_KV_SPLIT)</c> moves the reduction boundaries
    /// through the visible region as <c>seq_kv</c> grows. ATTN_KV_SPLIT is a
    /// compile-time 4 here (Vulkan's S is dynamic), so every seqKv that changes
    /// <c>ceil(seqKv/4)</c> moves a boundary. Prediction: AFFECTED.
    /// </summary>
    [SkippableFact]
    public void Cuda_SplitKv_KvPaddingAndBoundaryMove()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasAttentionF32SplitKv, "attention_f32_split_kv not present in PTX (stale build)");

        const int numHeads = 24, numKvHeads = 4, headDim = 256;
        Skip.IfNot(kernels.IsAttentionSplitKvSafe(numHeads, headDim),
            $"split-KV cooperative launch not safe for numHeads={numHeads}, headDim={headDim}");

        int split = CudaKernels.AttentionKvSplit;
        var sb = new StringBuilder();
        sb.AppendLine($"=== CUDA SPLIT-KV attention_f32_split_kv — ATTN_KV_SPLIT={split} ===");
        long worst = 0;

        foreach (var (baseKv, padKv) in new[]
                 {
                     (1024, 1025), (1024, 1028), (1024, 1052), (1024, 1100),
                     (600, 601), (600, 604), (600, 628),
                     (1300, 1304), (1300, 1400),
                 })
        {
            int c0 = (baseKv + split - 1) / split;
            int c1 = (padKv + split - 1) / split;
            bool moves = c0 != c1;

            int posOff = baseKv - 1;
            var rng = new Random(0x532 + baseKv * 31 + padKv * 7 + 2000);
            float[] q = Rand(rng, numHeads * headDim);
            float[] k = Rand(rng, padKv * numKvHeads * headDim);
            float[] v = Rand(rng, padKv * numKvHeads * headDim);
            int outLen = numHeads * headDim;
            float[] a = new float[outLen], b = new float[outLen];

            RunSplitPair(kernels, stream, q, k, v, a, b,
                baseKv, padKv, numHeads, numKvHeads, headDim, posOff);

            var (d, maxAbs) = CompareBitwise(a, b);
            worst = Math.Max(worst, d);
            sb.AppendLine($"  posQ={baseKv - 1,5} seqKv {baseKv,5}->{padKv,5} (chunk {c0}->{c1}, " +
                          $"boundariesMove={moves,-5}): differing={d,5}/{outLen,-5} maxAbs={maxAbs:E3}");
        }

        sb.AppendLine();
        sb.AppendLine(worst == 0
            ? "VERDICT: CUDA split-KV is BITWISE INVARIANT to KV padding."
            : $"VERDICT: CUDA split-KV IS KV-length dependent (worst {worst} elements).");
        _out.WriteLine(sb.ToString());
    }

    // ── device plumbing ────────────────────────────────────────────────────

    private static unsafe void RunDensePair(
        CudaKernels kernels, CudaStream stream,
        float[] q, float[] k, float[] v, float[] outA, float[] outB,
        int seqQ, int seqKvA, int seqKvB, int numHeads, int numKvHeads, int headDim, int posOff)
    {
        long qBytes = (long)q.Length * sizeof(float);
        long kvBytes = (long)k.Length * sizeof(float);
        long oBytes = (long)outA.Length * sizeof(float);
        nint dQ = 0, dK = 0, dV = 0, dA = 0, dB = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)oBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)oBytes).ThrowOnError();
            fixed (float* p = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
            fixed (float* p = k) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kvBytes).ThrowOnError();
            fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)kvBytes).ThrowOnError();

            nint s = stream.Handle;
            kernels.LaunchAttentionF32(dQ, dK, dV, dA, seqQ, seqKvA, numHeads, numKvHeads, headDim, posOff, 0, s);
            kernels.LaunchAttentionF32(dQ, dK, dV, dB, seqQ, seqKvB, numHeads, numKvHeads, headDim, posOff, 0, s);
            stream.Synchronize();

            fixed (float* p = outA) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dA, (nuint)oBytes).ThrowOnError();
            fixed (float* p = outB) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dB, (nuint)oBytes).ThrowOnError();
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dA != 0) CudaDriverApi.cuMemFree_v2(dA);
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
        }
    }

    private static unsafe void RunSplitPair(
        CudaKernels kernels, CudaStream stream,
        float[] q, float[] k, float[] v, float[] outA, float[] outB,
        int seqKvA, int seqKvB, int numHeads, int numKvHeads, int headDim, int posOff)
    {
        long qBytes = (long)q.Length * sizeof(float);
        long kvBytes = (long)k.Length * sizeof(float);
        long oBytes = (long)outA.Length * sizeof(float);
        long scalarBytes = (long)numHeads * CudaKernels.AttentionKvSplit * sizeof(float);
        long partOutBytes = scalarBytes * headDim;

        nint dQ = 0, dK = 0, dV = 0, dA = 0, dB = 0, dPM = 0, dPS = 0, dPO = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)oBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)oBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPM, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPS, (nuint)scalarBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dPO, (nuint)partOutBytes).ThrowOnError();

            fixed (float* p = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
            fixed (float* p = k) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kvBytes).ThrowOnError();
            fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)kvBytes).ThrowOnError();

            nint s = stream.Handle;
            kernels.LaunchAttentionF32SplitKv(dQ, dK, dV, dA, seqKvA, numHeads, numKvHeads, headDim,
                posOff, 0, dPM, dPS, dPO, s);
            stream.Synchronize();
            kernels.LaunchAttentionF32SplitKv(dQ, dK, dV, dB, seqKvB, numHeads, numKvHeads, headDim,
                posOff, 0, dPM, dPS, dPO, s);
            stream.Synchronize();

            fixed (float* p = outA) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dA, (nuint)oBytes).ThrowOnError();
            fixed (float* p = outB) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dB, (nuint)oBytes).ThrowOnError();
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dA != 0) CudaDriverApi.cuMemFree_v2(dA);
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
            if (dPM != 0) CudaDriverApi.cuMemFree_v2(dPM);
            if (dPS != 0) CudaDriverApi.cuMemFree_v2(dPS);
            if (dPO != 0) CudaDriverApi.cuMemFree_v2(dPO);
        }
    }
}
