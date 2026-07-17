using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity test for the Q5_K-resident MoE upload path
/// (<see cref="VulkanQwen3MoeMoeUpload.UploadLayer"/> with
/// <c>residentQuant: true</c>) plus the matching Vulkan dispatch
/// (<see cref="MoeIndexedMatmulQ5_KF32Kernel"/>). Mirrors
/// <see cref="VulkanQwen3MoeMoeUploadQ4KResidentTests"/> — this is the Q5_K
/// sibling (#372), plus the headline per-bank-mixed-quant scenario that
/// motivated the issue: llama.cpp "UD" mixed-quant GGUFs where a layer's
/// gate/up projections quantize to Q4_K but down quantizes to Q5_K (exactly
/// the real cached <c>unsloth/Qwen3.6-35B-A3B-GGUF</c> UD-Q4_K_XL checkpoint
/// on this box — see #371's ledger for how that was discovered).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanQwen3MoeMoeUploadQ5KResidentTests : IDisposable
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
    // Both `interm` and `hidden` must be multiples of 256 (Q5_K group size).
    [InlineData(4, 4, 256, 256, 2)]   // smallest: 1 super-block per row both axes, 2 active experts
    [InlineData(8, 8, 256, 512, 4)]   // 2 super-blocks per W1/W3 row (row stride 352)
    [InlineData(6, 4, 256, 768, 3)]   // 3 super-blocks per W1/W3 row (row stride 528)
    [InlineData(3, 4, 512, 256, 2)]   // wider intermediate — exercises W2's larger contraction
    public void ResidentQ5K_MatchesStreamingF32(
        int n, int numExperts, int interm, int hidden, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x5D9F1 + n * 31 + numExperts * 17 + interm * 11 + hidden * 7);
        int w1Elems = interm * hidden;
        int w2Elems = hidden * interm;
        int w1Rows = interm, w1Cols = hidden;
        int w2Rows = hidden, w2Cols = interm;
        int w1RowBytes = (w1Cols / Q5KFixture.Q5KGroupSize) * Q5KFixture.Q5KBlockBytes;
        int w2RowBytes = (w2Cols / Q5KFixture.Q5KGroupSize) * Q5KFixture.Q5KBlockBytes;
        int w1PerExpertBytes = w1Rows * w1RowBytes;
        int w2PerExpertBytes = w2Rows * w2RowBytes;

        nint gateExpsRaw = AllocBytes(numExperts * w1PerExpertBytes);
        nint upExpsRaw = AllocBytes(numExperts * w1PerExpertBytes);
        nint downExpsRaw = AllocBytes(numExperts * w2PerExpertBytes);

        var w1Ptrs = new nint[numExperts];
        var w2Ptrs = new nint[numExperts];
        var w3Ptrs = new nint[numExperts];

        for (int e = 0; e < numExperts; e++)
        {
            float[] w1F32 = Q5KFixture.RandomFloats(rng, w1Elems, range: 0.1f);
            byte[] w1Q5K = Q5KFixture.QuantizeRows(w1F32, w1Rows, w1Cols);
            FillRawSlice(gateExpsRaw, e * w1PerExpertBytes, w1Q5K);
            w1Ptrs[e] = AllocAndDequant(w1Q5K, w1Rows, w1Cols);

            float[] w3F32 = Q5KFixture.RandomFloats(rng, w1Elems, range: 0.1f);
            byte[] w3Q5K = Q5KFixture.QuantizeRows(w3F32, w1Rows, w1Cols);
            FillRawSlice(upExpsRaw, e * w1PerExpertBytes, w3Q5K);
            w3Ptrs[e] = AllocAndDequant(w3Q5K, w1Rows, w1Cols);

            float[] w2F32 = Q5KFixture.RandomFloats(rng, w2Elems, range: 0.1f);
            byte[] w2Q5K = Q5KFixture.QuantizeRows(w2F32, w2Rows, w2Cols);
            FillRawSlice(downExpsRaw, e * w2PerExpertBytes, w2Q5K);
            w2Ptrs[e] = AllocAndDequant(w2Q5K, w2Rows, w2Cols);
        }

        float[] routerGate = Q5KFixture.RandomFloats(rng, numExperts * hidden, range: 0.05f);

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
            gateExpsRaw: gateExpsRaw, gateExpsRawQt: QuantizationType.Q5_K,
            gateExpsMDim: w1Rows, gateExpsKDim: w1Cols,
            upExpsRaw: upExpsRaw, upExpsRawQt: QuantizationType.Q5_K,
            upExpsMDim: w1Rows, upExpsKDim: w1Cols,
            downExpsRaw: downExpsRaw, downExpsRawQt: QuantizationType.Q5_K,
            downExpsMDim: w2Rows, downExpsKDim: w2Cols,
            sharedGateRaw: Array.Empty<nint>(), sharedGateRawQt: QuantizationType.F32,
            sharedUpRaw: Array.Empty<nint>(), sharedUpRawQt: QuantizationType.F32,
            sharedDownRaw: Array.Empty<nint>(), sharedDownRawQt: QuantizationType.F32);

        Assert.True(moe.HasRawQuantView);

        float[] x = Q5KFixture.RandomFloats(rng, n * hidden, range: 1.0f);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        using var device = VulkanDevice.Create();
        using var f32Kernel = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
        using var q5kKernel = MoeIndexedMatmulQ5_KF32Kernel.Create(device, spvDir);

        // ── Streaming F32 upload (current default). ──
        using var streamingBundle = VulkanQwen3MoeMoeUpload.UploadLayer(
            device, moe, hiddenSize: hidden, residentQuant: false);
        Assert.Equal(QuantizationType.F32, streamingBundle.W1QuantType);
        Assert.Equal(QuantizationType.F32, streamingBundle.W2QuantType);
        Assert.Equal(QuantizationType.F32, streamingBundle.W3QuantType);

        // ── Resident Q5_K upload. ──
        using var residentBundle = VulkanQwen3MoeMoeUpload.UploadLayer(
            device, moe, hiddenSize: hidden, residentQuant: true);
        Assert.Equal(QuantizationType.Q5_K, residentBundle.W1QuantType);
        Assert.Equal(QuantizationType.Q5_K, residentBundle.W2QuantType);
        Assert.Equal(QuantizationType.Q5_K, residentBundle.W3QuantType);

        // ── Compare W1 dispatch outputs. ──
        AssertBankParity(device, f32Kernel, q5kKernel,
            streamingBundle.W1Bank, residentBundle.W1Bank,
            x, indices, m: interm, k: hidden, n: n, numExperts: numExperts);

        // ── W3 dispatch outputs. ──
        AssertBankParity(device, f32Kernel, q5kKernel,
            streamingBundle.W3Bank, residentBundle.W3Bank,
            x, indices, m: interm, k: hidden, n: n, numExperts: numExperts);

        // ── W2 dispatch outputs (down projection, m=hidden, k=interm). ──
        float[] xDown = Q5KFixture.RandomFloats(rng, n * interm, range: 1.0f);
        AssertBankParity(device, f32Kernel, q5kKernel,
            streamingBundle.W2Bank, residentBundle.W2Bank,
            xDown, indices, m: hidden, k: interm, n: n, numExperts: numExperts);
    }

    [SkippableFact]
    public void ResidentQ5K_FallsBackToF32_WhenSourceNotQ5K()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int numExperts = 4, interm = 16, hidden = 32;
        var rng = new Random(0xF8CB5);

        var w1Ptrs = new nint[numExperts];
        var w2Ptrs = new nint[numExperts];
        var w3Ptrs = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            w1Ptrs[e] = AllocFloats(Q5KFixture.RandomFloats(rng, interm * hidden, 0.05f));
            w2Ptrs[e] = AllocFloats(Q5KFixture.RandomFloats(rng, hidden * interm, 0.05f));
            w3Ptrs[e] = AllocFloats(Q5KFixture.RandomFloats(rng, interm * hidden, 0.05f));
        }
        var moe = new MoeLayerWeights(
            gate: Q5KFixture.RandomFloats(rng, numExperts * hidden, 0.05f),
            w1: w1Ptrs, w2: w2Ptrs, w3: w3Ptrs,
            numExperts: numExperts, numExpertsPerTok: 2,
            hiddenSize: hidden, intermediateSize: interm);

        Assert.False(moe.HasRawQuantView);

        using var device = VulkanDevice.Create();
        using var bundle = VulkanQwen3MoeMoeUpload.UploadLayer(
            device, moe, hiddenSize: hidden, residentQuant: true);

        Assert.Equal(QuantizationType.F32, bundle.W1QuantType);
        Assert.Equal(QuantizationType.F32, bundle.W2QuantType);
        Assert.Equal(QuantizationType.F32, bundle.W3QuantType);
    }

    /// <summary>
    /// The headline scenario for #372: a layer whose gate/up (W1/W3)
    /// quantize to Q4_K but whose down (W2) quantizes to Q5_K — exactly the
    /// real <c>unsloth/Qwen3.6-35B-A3B-GGUF</c> UD-Q4_K_XL shape discovered
    /// while validating #371. Each bank must independently resolve to its
    /// OWN resident quant type (gate/up Q4_K-resident, down Q5_K-resident)
    /// rather than any bank forcing the others to F32.
    /// </summary>
    [SkippableFact]
    public void ResidentQuant_MixedQ4KGateUpQ5KDown_MatchesStreamingF32()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int numExperts = 2, interm = 256, hidden = 256, n = 4, activeExperts = 2;
        var rng = new Random(0x4C55);

        int w1Elems = interm * hidden;
        int w2Elems = hidden * interm;
        int w1RowBytes = (hidden / Q4KFixture.Q4KGroupSize) * Q4KFixture.Q4KBlockBytes;
        int w2RowBytes = (interm / Q5KFixture.Q5KGroupSize) * Q5KFixture.Q5KBlockBytes;
        int w1PerExpertBytes = interm * w1RowBytes;
        int w2PerExpertBytes = hidden * w2RowBytes;

        nint gateExpsRaw = AllocBytes(numExperts * w1PerExpertBytes);
        nint upExpsRaw = AllocBytes(numExperts * w1PerExpertBytes);
        nint downExpsRaw = AllocBytes(numExperts * w2PerExpertBytes);

        var w1Ptrs = new nint[numExperts];
        var w2Ptrs = new nint[numExperts];
        var w3Ptrs = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            byte[] w1Q4K = Q4KFixture.QuantizeRows(
                Q4KFixture.RandomFloats(rng, w1Elems, 0.1f), interm, hidden);
            FillRawSlice(gateExpsRaw, e * w1PerExpertBytes, w1Q4K);
            w1Ptrs[e] = AllocAndDequant(w1Q4K, interm, hidden);

            byte[] w3Q4K = Q4KFixture.QuantizeRows(
                Q4KFixture.RandomFloats(rng, w1Elems, 0.1f), interm, hidden);
            FillRawSlice(upExpsRaw, e * w1PerExpertBytes, w3Q4K);
            w3Ptrs[e] = AllocAndDequant(w3Q4K, interm, hidden);

            // W2 is Q5_K — the mismatched down-projection quant this issue targets.
            byte[] w2Q5K = Q5KFixture.QuantizeRows(
                Q5KFixture.RandomFloats(rng, w2Elems, 0.1f), hidden, interm);
            FillRawSlice(downExpsRaw, e * w2PerExpertBytes, w2Q5K);
            w2Ptrs[e] = AllocAndDequantQ5K(w2Q5K, hidden, interm);
        }

        float[] routerGate = Q4KFixture.RandomFloats(rng, numExperts * hidden, range: 0.05f);
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
            gateExpsRaw: gateExpsRaw, gateExpsRawQt: QuantizationType.Q4_K,
            gateExpsMDim: interm, gateExpsKDim: hidden,
            upExpsRaw: upExpsRaw, upExpsRawQt: QuantizationType.Q4_K,
            upExpsMDim: interm, upExpsKDim: hidden,
            downExpsRaw: downExpsRaw, downExpsRawQt: QuantizationType.Q5_K,
            downExpsMDim: hidden, downExpsKDim: interm,
            sharedGateRaw: Array.Empty<nint>(), sharedGateRawQt: QuantizationType.F32,
            sharedUpRaw: Array.Empty<nint>(), sharedUpRawQt: QuantizationType.F32,
            sharedDownRaw: Array.Empty<nint>(), sharedDownRawQt: QuantizationType.F32);

        Assert.True(moe.HasRawQuantView);

        float[] x = Q4KFixture.RandomFloats(rng, n * hidden, range: 1.0f);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        using var device = VulkanDevice.Create();
        using var f32Kernel = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
        using var q4kKernel = MoeIndexedMatmulQ4_KF32Kernel.Create(device, spvDir);
        using var q5kKernel = MoeIndexedMatmulQ5_KF32Kernel.Create(device, spvDir);

        using var streamingBundle = VulkanQwen3MoeMoeUpload.UploadLayer(
            device, moe, hiddenSize: hidden, residentQuant: false);
        using var residentBundle = VulkanQwen3MoeMoeUpload.UploadLayer(
            device, moe, hiddenSize: hidden, residentQuant: true);

        // The core assertion of this issue: gate/up resolve to Q4_K-resident,
        // down resolves to Q5_K-resident — NOT a uniform F32 fallback.
        Assert.Equal(QuantizationType.Q4_K, residentBundle.W1QuantType);
        Assert.Equal(QuantizationType.Q4_K, residentBundle.W3QuantType);
        Assert.Equal(QuantizationType.Q5_K, residentBundle.W2QuantType);

        AssertBankParity(device, f32Kernel, q4kKernel,
            streamingBundle.W1Bank, residentBundle.W1Bank,
            x, indices, m: interm, k: hidden, n: n, numExperts: numExperts);
        AssertBankParity(device, f32Kernel, q4kKernel,
            streamingBundle.W3Bank, residentBundle.W3Bank,
            x, indices, m: interm, k: hidden, n: n, numExperts: numExperts);

        float[] xDown = Q4KFixture.RandomFloats(rng, n * interm, range: 1.0f);
        AssertBankParity(device, f32Kernel, q5kKernel,
            streamingBundle.W2Bank, residentBundle.W2Bank,
            xDown, indices, m: hidden, k: interm, n: n, numExperts: numExperts);
    }

    private static void AssertBankParity(
        VulkanDevice device,
        MoeIndexedMatmulF32Kernel f32Kernel, MoeIndexedMatmulQ5_KF32Kernel q5kKernel,
        VulkanDevice.Buffer f32Bank, VulkanDevice.Buffer q5kBank,
        float[] x, int[] indices, int m, int k, int n, int numExperts)
    {
        long yElems = (long)n * m;
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var f32YBuf = device.Allocate(yElems * sizeof(float));
        using var q5kYBuf = device.Allocate(yElems * sizeof(float));

        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);

        f32Kernel.Launch(f32Bank, xBuf, idxBuf, f32YBuf, m, k, n, numExperts);
        q5kKernel.Launch(q5kBank, xBuf, idxBuf, q5kYBuf, m, k, n, numExperts);

        var f32Out = new float[yElems];
        var q5kOut = new float[yElems];
        device.Download(f32YBuf, f32Out);
        device.Download(q5kYBuf, q5kOut);

        for (int i = 0; i < yElems; i++)
        {
            float diff = MathF.Abs(f32Out[i] - q5kOut[i]);
            float bar = AbsTol + RelTol * MathF.Abs(f32Out[i]);
            Assert.True(diff <= bar,
                $"i={i}, row={i / m}, col={i % m}: f32-streaming={f32Out[i]:F6} vs q5k-resident={q5kOut[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        bool anyNonZero = false;
        for (int i = 0; i < yElems; i++)
        {
            Assert.True(float.IsFinite(f32Out[i]), $"non-finite F32 logit at {i}: {f32Out[i]}");
            Assert.True(float.IsFinite(q5kOut[i]), $"non-finite Q5_K logit at {i}: {q5kOut[i]}");
            if (f32Out[i] != 0f) anyNonZero = true;
        }
        Assert.True(anyNonZero, "Output is degenerate (all zero) — fixture is too sparse to be a meaningful parity check.");
    }

    private static void AssertBankParity(
        VulkanDevice device,
        MoeIndexedMatmulF32Kernel f32Kernel, MoeIndexedMatmulQ4_KF32Kernel q4kKernel,
        VulkanDevice.Buffer f32Bank, VulkanDevice.Buffer q4kBank,
        float[] x, int[] indices, int m, int k, int n, int numExperts)
    {
        long yElems = (long)n * m;
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var f32YBuf = device.Allocate(yElems * sizeof(float));
        using var q4kYBuf = device.Allocate(yElems * sizeof(float));

        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);

        f32Kernel.Launch(f32Bank, xBuf, idxBuf, f32YBuf, m, k, n, numExperts);
        q4kKernel.Launch(q4kBank, xBuf, idxBuf, q4kYBuf, m, k, n, numExperts);

        var f32Out = new float[yElems];
        var q4kOut = new float[yElems];
        device.Download(f32YBuf, f32Out);
        device.Download(q4kYBuf, q4kOut);

        for (int i = 0; i < yElems; i++)
        {
            float diff = MathF.Abs(f32Out[i] - q4kOut[i]);
            float bar = AbsTol + RelTol * MathF.Abs(f32Out[i]);
            Assert.True(diff <= bar,
                $"i={i}, row={i / m}, col={i % m}: f32-streaming={f32Out[i]:F6} vs q4k-resident={q4kOut[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        bool anyNonZero = false;
        for (int i = 0; i < yElems; i++)
        {
            Assert.True(float.IsFinite(f32Out[i]), $"non-finite F32 logit at {i}: {f32Out[i]}");
            Assert.True(float.IsFinite(q4kOut[i]), $"non-finite Q4_K logit at {i}: {q4kOut[i]}");
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

    private nint AllocAndDequant(byte[] q5kBytes, int rows, int cols)
    {
        long elems = (long)rows * cols;
        nint p = (nint)NativeMemory.AlignedAlloc((nuint)(elems * sizeof(float)), 64);
        _allocs.Add(p);
        var dst = new Span<float>((void*)p, checked((int)elems));
        fixed (byte* src = q5kBytes)
        {
            DotLLM.Cpu.Kernels.Dequantize.ToFloat32((nint)src, elems, QuantizationType.Q5_K, dst);
        }
        return p;
    }

    private nint AllocAndDequantQ5K(byte[] q5kBytes, int rows, int cols) => AllocAndDequant(q5kBytes, rows, cols);

    private nint AllocFloats(float[] data)
    {
        nint p = (nint)NativeMemory.AlignedAlloc((nuint)(data.Length * sizeof(float)), 64);
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
