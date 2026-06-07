using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity test for the Q6_K-resident MoE upload path
/// (<see cref="VulkanQwen3MoeMoeUpload.UploadLayer"/> with
/// <c>residentQuant: true</c>) plus the matching Vulkan dispatch
/// (<see cref="MoeIndexedMatmulQ6_KF32Kernel"/>).
/// </summary>
/// <remarks>
/// <para>
/// This test exercises the full resident-Q6_K seam:
/// </para>
/// <list type="number">
///   <item>Build a synthetic <see cref="MoeLayerWeights"/> with a populated
///         raw-quant view: per-routed-projection (W1/W2/W3) bytes are
///         <see cref="Q6KFixture.QuantizeRows"/>-quantised from random F32
///         weights, the F32 array slots hold the matching dequant for the CPU
///         oracle.</item>
///   <item>Upload the layer twice — once with <c>residentQuant: false</c>
///         (existing F32 dequant + streaming) and once with
///         <c>residentQuant: true</c> (Q6_K-resident raw-block copy). Verify
///         <see cref="VulkanQwen3MoeMoeUpload.LayerBundle.BankQuantType"/>
///         reflects the chosen path.</item>
///   <item>Run the indexed matmul over each bundle (F32 path uses
///         <see cref="MoeIndexedMatmulF32Kernel"/>, Q6_K path uses
///         <see cref="MoeIndexedMatmulQ6_KF32Kernel"/>) on the same expert
///         indices and input rows.</item>
///   <item>Both outputs must agree to within Q6_K round-trip tolerance — the
///         Q6_K path's dequant inside the shader produces the same floats as
///         the F32-resident path's pre-dequant on the host (modulo
///         reduction-order noise).</item>
/// </list>
/// <para>
/// This is the "resident-MoE end-to-end" coverage in the Phase-10 follow-up
/// acceptance criteria: it validates that resident-Q6_K banks survive the
/// full upload+dispatch round trip with finite, non-degenerate, parity-bound
/// outputs vs the F32 streaming default. A real-GGUF smoke test on
/// Qwen3.6-A3B-UD-Q6_K_XL is the model-level extension; lives in
/// <see cref="RealQwen3MoeHybridQ6KResidentSmokeTests"/> and is gated on the
/// presence of the cached GGUF file.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanQwen3MoeMoeUploadQ6KResidentTests : IDisposable
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly List<nint> _allocs = new();

    public void Dispose()
    {
        foreach (var p in _allocs) NativeMemory.AlignedFree((void*)p);
        _allocs.Clear();
    }

    [SkippableTheory]
    // Both `interm` and `hidden` must be multiples of 256 (Q6_K group size) —
    // the W2 down-projection's contraction dim is `interm`, so the parity
    // matmul of W2 needs `interm % 256 == 0` too. The smallest viable shape
    // is therefore (interm=256, hidden=256). At qwen35moe-A3B production scale
    // both dims are multiples of 256 by design.
    [InlineData(4, 4, 256, 256, 2)]   // smallest: 1 super-block per row both axes, 2 active experts
    [InlineData(8, 8, 256, 512, 4)]   // 2 super-blocks per W1/W3 row (row stride 420 — 4-aligned)
    [InlineData(6, 4, 256, 768, 3)]   // 3 super-blocks per W1/W3 row (row stride 630 — NOT 4-aligned)
    [InlineData(3, 4, 512, 256, 2)]   // wider intermediate — exercises W2's larger contraction
    public void ResidentQ6K_MatchesStreamingF32(
        int n, int numExperts, int interm, int hidden, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        // ── Build the synthetic MoE bank with a Q6_K raw quant view. ──
        // Each routed projection (W1/W3 = [interm, hidden], W2 = [hidden, interm])
        // is built by quantising random F32 weights to Q6_K, then dequantising
        // to F32 so MoeLayerWeights' "F32 fallback" arrays match the bytes the
        // raw view points at — the streaming path consumes the F32 arrays via
        // Dequantize.ToFloat32 (which round-trips through the same bytes), so
        // the comparison is fair.
        var rng = new Random(0x6CB6D + n * 31 + numExperts * 17 + interm * 11 + hidden * 7);
        int w1Elems = interm * hidden;
        int w2Elems = hidden * interm;
        int w1Rows = interm, w1Cols = hidden;
        int w2Rows = hidden, w2Cols = interm;
        int w1RowBytes = (w1Cols / Q6KFixture.Q6KGroupSize) * Q6KFixture.Q6KBlockBytes;
        int w2RowBytes = (w2Cols / Q6KFixture.Q6KGroupSize) * Q6KFixture.Q6KBlockBytes;
        int w1PerExpertBytes = w1Rows * w1RowBytes;
        int w2PerExpertBytes = w2Rows * w2RowBytes;

        // GGUF fused-experts layout: experts contiguous on the slowest axis.
        nint gateExpsRaw = AllocBytes(numExperts * w1PerExpertBytes);
        nint upExpsRaw = AllocBytes(numExperts * w1PerExpertBytes);
        nint downExpsRaw = AllocBytes(numExperts * w2PerExpertBytes);

        var w1Ptrs = new nint[numExperts];
        var w2Ptrs = new nint[numExperts];
        var w3Ptrs = new nint[numExperts];

        for (int e = 0; e < numExperts; e++)
        {
            // Gate (W1)
            float[] w1F32 = Q6KFixture.RandomFloats(rng, w1Elems, range: 0.1f);
            byte[] w1Q6K = Q6KFixture.QuantizeRows(w1F32, w1Rows, w1Cols);
            FillRawSlice(gateExpsRaw, e * w1PerExpertBytes, w1Q6K);
            w1Ptrs[e] = AllocAndDequant(w1Q6K, w1Rows, w1Cols);

            // Up (W3)
            float[] w3F32 = Q6KFixture.RandomFloats(rng, w1Elems, range: 0.1f);
            byte[] w3Q6K = Q6KFixture.QuantizeRows(w3F32, w1Rows, w1Cols);
            FillRawSlice(upExpsRaw, e * w1PerExpertBytes, w3Q6K);
            w3Ptrs[e] = AllocAndDequant(w3Q6K, w1Rows, w1Cols);

            // Down (W2)
            float[] w2F32 = Q6KFixture.RandomFloats(rng, w2Elems, range: 0.1f);
            byte[] w2Q6K = Q6KFixture.QuantizeRows(w2F32, w2Rows, w2Cols);
            FillRawSlice(downExpsRaw, e * w2PerExpertBytes, w2Q6K);
            w2Ptrs[e] = AllocAndDequant(w2Q6K, w2Rows, w2Cols);
        }

        float[] routerGate = Q6KFixture.RandomFloats(rng, numExperts * hidden, range: 0.05f);

        // No shared expert in this fixture — simplifies the upload comparison.
        var moe = new MoeLayerWeights(
            gate: routerGate,
            w1: w1Ptrs, w2: w2Ptrs, w3: w3Ptrs,
            numExperts: numExperts, numExpertsPerTok: activeExperts,
            hiddenSize: hidden, intermediateSize: interm,
            normTopKProb: true,
            sharedGateProj: Array.Empty<nint>(),
            sharedUpProj: Array.Empty<nint>(),
            sharedDownProj: Array.Empty<nint>(),
            sharedIntermediateSize: 0,
            sharedExpertGate: null,
            gateExpsRaw: gateExpsRaw, gateExpsRawQt: QuantizationType.Q6_K,
            gateExpsMDim: w1Rows, gateExpsKDim: w1Cols,
            upExpsRaw: upExpsRaw, upExpsRawQt: QuantizationType.Q6_K,
            upExpsMDim: w1Rows, upExpsKDim: w1Cols,
            downExpsRaw: downExpsRaw, downExpsRawQt: QuantizationType.Q6_K,
            downExpsMDim: w2Rows, downExpsKDim: w2Cols,
            sharedGateRaw: Array.Empty<nint>(), sharedGateRawQt: QuantizationType.F32,
            sharedUpRaw: Array.Empty<nint>(), sharedUpRawQt: QuantizationType.F32,
            sharedDownRaw: Array.Empty<nint>(), sharedDownRawQt: QuantizationType.F32);

        Assert.True(moe.HasRawQuantView);

        // ── Two random input rows + indices. Use the W1-shape (m=interm, k=hidden)
        //    for the parity matmul — same as the gate/up projections. ─────────
        float[] x = Q6KFixture.RandomFloats(rng, n * hidden, range: 1.0f);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        using var device = VulkanDevice.Create();
        using var f32Kernel = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
        using var q6kKernel = MoeIndexedMatmulQ6_KF32Kernel.Create(device, spvDir);

        // ── Streaming F32 upload (current default). ──
        using var streamingBundle = VulkanQwen3MoeMoeUpload.UploadLayer(
            device, moe, hiddenSize: hidden, residentQuant: false);
        Assert.Equal(QuantizationType.F32, streamingBundle.BankQuantType);

        // ── Resident Q6_K upload. ──
        using var residentBundle = VulkanQwen3MoeMoeUpload.UploadLayer(
            device, moe, hiddenSize: hidden, residentQuant: true);
        Assert.Equal(QuantizationType.Q6_K, residentBundle.BankQuantType);

        // ── Compare W1 dispatch outputs. ──
        AssertBankParity(device, f32Kernel, q6kKernel,
            streamingBundle.W1Bank, residentBundle.W1Bank,
            x, indices, m: interm, k: hidden, n: n, numExperts: numExperts);

        // ── W3 dispatch outputs. ──
        AssertBankParity(device, f32Kernel, q6kKernel,
            streamingBundle.W3Bank, residentBundle.W3Bank,
            x, indices, m: interm, k: hidden, n: n, numExperts: numExperts);

        // ── W2 dispatch outputs (down projection, m=hidden, k=interm). ──
        // Build a separate input matrix sized for the down-projection contraction.
        float[] xDown = Q6KFixture.RandomFloats(rng, n * interm, range: 1.0f);
        AssertBankParity(device, f32Kernel, q6kKernel,
            streamingBundle.W2Bank, residentBundle.W2Bank,
            xDown, indices, m: hidden, k: interm, n: n, numExperts: numExperts);
    }

    [SkippableFact]
    public void ResidentQ6K_FallsBackToF32_WhenSourceNotQ6K()
    {
        // When the source MoE banks aren't Q6_K, the resident upload must
        // gracefully fall back to F32 (the existing streaming path) — the
        // Q6_K-resident path is only safe when ALL three routed projections
        // are uniformly Q6_K. Use the F32-only synthetic-fixture pattern (no
        // raw quant view at all) to validate the gate.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int numExperts = 4, interm = 16, hidden = 32;
        var rng = new Random(0xF8CA);

        var w1Ptrs = new nint[numExperts];
        var w2Ptrs = new nint[numExperts];
        var w3Ptrs = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            w1Ptrs[e] = AllocFloats(Q6KFixture.RandomFloats(rng, interm * hidden, 0.05f));
            w2Ptrs[e] = AllocFloats(Q6KFixture.RandomFloats(rng, hidden * interm, 0.05f));
            w3Ptrs[e] = AllocFloats(Q6KFixture.RandomFloats(rng, interm * hidden, 0.05f));
        }
        var moe = new MoeLayerWeights(
            gate: Q6KFixture.RandomFloats(rng, numExperts * hidden, 0.05f),
            w1: w1Ptrs, w2: w2Ptrs, w3: w3Ptrs,
            numExperts: numExperts, numExpertsPerTok: 2,
            hiddenSize: hidden, intermediateSize: interm);

        Assert.False(moe.HasRawQuantView);

        using var device = VulkanDevice.Create();
        using var bundle = VulkanQwen3MoeMoeUpload.UploadLayer(
            device, moe, hiddenSize: hidden, residentQuant: true);

        Assert.Equal(QuantizationType.F32, bundle.BankQuantType);
    }

    private static void AssertBankParity(
        VulkanDevice device,
        MoeIndexedMatmulF32Kernel f32Kernel, MoeIndexedMatmulQ6_KF32Kernel q6kKernel,
        VulkanDevice.Buffer f32Bank, VulkanDevice.Buffer q6kBank,
        float[] x, int[] indices, int m, int k, int n, int numExperts)
    {
        long yElems = (long)n * m;
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var f32YBuf = device.Allocate(yElems * sizeof(float));
        using var q6kYBuf = device.Allocate(yElems * sizeof(float));

        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);

        f32Kernel.Launch(f32Bank, xBuf, idxBuf, f32YBuf, m, k, n, numExperts);
        q6kKernel.Launch(q6kBank, xBuf, idxBuf, q6kYBuf, m, k, n, numExperts);

        var f32Out = new float[yElems];
        var q6kOut = new float[yElems];
        device.Download(f32YBuf, f32Out);
        device.Download(q6kYBuf, q6kOut);

        for (int i = 0; i < yElems; i++)
        {
            float diff = MathF.Abs(f32Out[i] - q6kOut[i]);
            float bar = AbsTol + RelTol * MathF.Abs(f32Out[i]);
            Assert.True(diff <= bar,
                $"i={i}, row={i / m}, col={i % m}: f32-streaming={f32Out[i]:F6} vs q6k-resident={q6kOut[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        // Sanity: outputs should be non-degenerate (not all zeros / NaN).
        bool anyNonZero = false;
        for (int i = 0; i < yElems; i++)
        {
            Assert.True(float.IsFinite(f32Out[i]), $"non-finite F32 logit at {i}: {f32Out[i]}");
            Assert.True(float.IsFinite(q6kOut[i]), $"non-finite Q6_K logit at {i}: {q6kOut[i]}");
            if (f32Out[i] != 0f) anyNonZero = true;
        }
        Assert.True(anyNonZero, "Output is degenerate (all zero) — fixture is too sparse to be a meaningful parity check.");
    }

    private nint AllocBytes(int count)
    {
        nint p = (nint)NativeMemory.AlignedAlloc((nuint)count, 64);
        _allocs.Add(p);
        new Span<byte>((void*)p, count).Clear();
        return p;
    }

    private static void FillRawSlice(nint dst, int byteOffset, byte[] src)
    {
        var dstSpan = new Span<byte>((byte*)dst + byteOffset, src.Length);
        src.AsSpan().CopyTo(dstSpan);
    }

    private nint AllocAndDequant(byte[] q6kBytes, int rows, int cols)
    {
        // Dequantise once via the CPU oracle; the resulting F32 matches what
        // the streaming-F32 upload path would produce when consuming the same
        // raw bytes via Dequantize.ToFloat32. Stored in unmanaged memory so
        // MoeLayerWeights can hold it as a nint.
        long elems = (long)rows * cols;
        nint p = (nint)NativeMemory.AlignedAlloc((nuint)(elems * sizeof(float)), 64);
        _allocs.Add(p);
        var dst = new Span<float>((void*)p, checked((int)elems));
        fixed (byte* src = q6kBytes)
        {
            DotLLM.Cpu.Kernels.Dequantize.ToFloat32((nint)src, elems, QuantizationType.Q6_K, dst);
        }
        return p;
    }

    private nint AllocFloats(float[] data)
    {
        nint p = (nint)NativeMemory.AlignedAlloc((nuint)((long)data.Length * sizeof(float)), 64);
        _allocs.Add(p);
        var dst = new Span<float>((void*)p, data.Length);
        data.AsSpan().CopyTo(dst);
        return p;
    }

    private static int[] RandomIndices(Random rng, int count, int numExperts, int activePool)
    {
        int pool = Math.Min(activePool, numExperts);
        var unique = new HashSet<int>();
        while (unique.Count < pool) unique.Add(rng.Next(numExperts));
        var poolArr = unique.ToArray();
        var indices = new int[count];
        for (int i = 0; i < count; i++) indices[i] = poolArr[rng.Next(poolArr.Length)];
        return indices;
    }
}
