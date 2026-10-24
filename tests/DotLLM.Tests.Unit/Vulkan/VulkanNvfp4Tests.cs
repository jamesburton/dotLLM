using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Variant = DotLLM.Tests.Unit.Models.Architectures.NemotronHNvfp4SsmTests.Variant;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Vulkan NVFP4 (issue #1038): the packed GEMV/GEMM against the CPU dequantizer (itself pinned
/// bit-exact to llama.cpp b11739 in <c>Nvfp4Tests</c>), and the Nemotron-H GGUF path end to end
/// against the CPU model on the F32 <c>scale · dequant(W)</c> twin of the same fixture.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNvfp4Tests : IDisposable
{
    private readonly string _scratch;

    public VulkanNvfp4Tests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-nvfp4-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableTheory]
    [InlineData(1, 64, 1, 1.0f)]       // GEMV, one block
    [InlineData(70, 192, 1, 0.37f)]    // GEMV, 3 blocks, scale
    [InlineData(300, 4096, 1, 1.3e-4f)] // GEMV, Bolt-like K and scale
    [InlineData(70, 192, 5, 2.5f)]     // GEMM, partial M and N tiles
    [InlineData(33, 128, 17, 1.0f)]    // GEMM, N crosses one tile
    [InlineData(64, 4096, 32, 1.3e-4f)] // GEMM, Bolt-like K
    public void MatMul_MatchesCpuDequantReference(int m, int k, int n, float scale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var (w, x, expected) = MakeCase(m, k, n, scale);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulNvfp4F32Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(w.Length);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        kernel.Launch(bufW, bufX, bufY, m, k, n, scale);
        float[] actual = new float[n * m];
        device.Download(bufY, actual);

        float maxAbs = expected.Max(MathF.Abs);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 1e-4f * maxAbs + 1e-6f * MathF.Abs(expected[i]),
                $"m={m} k={k} n={n} [{i}]: expected {expected[i]}, vulkan {actual[i]} (max |y| {maxAbs})");
    }

    /// <summary>
    /// #1066: the three decode GEMVs against the CPU dequant reference. The FP32 ones (coalesced multi-row, and #1038's
    /// one-block-per-lane) match the double-precision product to F32 rounding. The MMVQ quantises the activations to Q8_1 first, so it
    /// is held tightly to the product with the activations quantised exactly as <c>quantize_q8_1.comp</c> does (ties aside), and
    /// loosely to the unquantised one. Odd M covers the row guard of multi-row builds; K covers one block, a window tail
    /// (3 blocks), and Bolt's ssm_in / ssm_out widths with their per-tensor scales.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 64, 1.0f, GemvVariant.Coalesced)]
    [InlineData(71, 192, 0.37f, GemvVariant.Coalesced)]
    [InlineData(300, 4096, 1.3e-4f, GemvVariant.Coalesced)]
    [InlineData(77, 8192, 3.3e-4f, GemvVariant.Coalesced)]
    [InlineData(71, 192, 0.37f, GemvVariant.Legacy)]
    [InlineData(77, 8192, 3.3e-4f, GemvVariant.Legacy)]
    [InlineData(1, 64, 1.0f, GemvVariant.Mmvq)]
    [InlineData(71, 192, 0.37f, GemvVariant.Mmvq)]
    [InlineData(300, 4096, 1.3e-4f, GemvVariant.Mmvq)]
    [InlineData(77, 8192, 3.3e-4f, GemvVariant.Mmvq)]
    public void Gemv_Variants_MatchCpuDequantReference(int m, int k, float scale, GemvVariant variant)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (w, x, expected) = MakeCase(m, k, 1, scale);

        using var device = VulkanDevice.Create();
        if (variant != GemvVariant.Legacy) Skip.IfNot(device.HasSubgroupArithmetic, "no subgroup arithmetic");
        if (variant == GemvVariant.Mmvq) Skip.IfNot(device.HasIntegerDotProduct, "no integer dot product");
        using var kernel = MatMulNvfp4F32Kernel.Create(device, spvDir,
            variant == GemvVariant.Legacy ? null : MatMulNvfp4F32Kernel.CoalescedGemvSpvFileName, MatMulNvfp4F32Kernel.CoalescedGemvRows,
            variant == GemvVariant.Mmvq ? MatMulNvfp4F32Kernel.MmvqSpvFileName : null, mmvqMinWeightBytes: 0);
        Assert.Equal(variant != GemvVariant.Legacy, kernel.UsesCoalescedGemv);
        Assert.Equal(variant == GemvVariant.Mmvq, kernel.HasMmvq);
        using var bufW = device.Allocate(w.Length);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        long coalescedBefore = Interlocked.Read(ref MatMulNvfp4F32Kernel.CoalescedGemvDispatches);
        long mmvqBefore = Interlocked.Read(ref MatMulNvfp4F32Kernel.MmvqDispatches);
        kernel.Launch(bufW, bufX, bufY, m, k, 1, scale);
        Assert.Equal(variant == GemvVariant.Coalesced ? 1 : 0,
            Interlocked.Read(ref MatMulNvfp4F32Kernel.CoalescedGemvDispatches) - coalescedBefore);
        Assert.Equal(variant == GemvVariant.Mmvq ? 1 : 0, Interlocked.Read(ref MatMulNvfp4F32Kernel.MmvqDispatches) - mmvqBefore);
        float[] actual = new float[m];
        device.Download(bufY, actual);

        float maxAbs = expected.Max(MathF.Abs);
        if (variant == GemvVariant.Mmvq)
        {
            float[] expectedQ8 = ReferenceWithQ8_1Activations(w, x, m, k, scale);
            for (int i = 0; i < m; i++)
            {
                Assert.True(MathF.Abs(actual[i] - expectedQ8[i]) <= 2e-3f * maxAbs,
                    $"m={m} k={k} mmvq [{i}]: Q8_1-activation reference {expectedQ8[i]}, vulkan {actual[i]} (max |y| {maxAbs})");
                Assert.True(MathF.Abs(actual[i] - expected[i]) <= 3e-2f * maxAbs,
                    $"m={m} k={k} mmvq [{i}]: F32 reference {expected[i]}, vulkan {actual[i]} (max |y| {maxAbs})");
            }
            return;
        }
        for (int i = 0; i < m; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 1e-4f * maxAbs + 1e-6f * MathF.Abs(expected[i]),
                $"m={m} k={k} {variant} [{i}]: expected {expected[i]}, vulkan {actual[i]} (max |y| {maxAbs})");
    }

    /// <summary>The decode GEMV a test forces.</summary>
    public enum GemvVariant
    {
        /// <summary>#1038's one-block-per-lane FP32 GEMV.</summary>
        Legacy,

        /// <summary>The coalesced multi-row FP32 GEMV (#1066).</summary>
        Coalesced,

        /// <summary>The Q8_1-activation integer-dot MMVQ (#1066).</summary>
        Mmvq,
    }

    /// <summary>
    /// <c>scale * dequant(W) . x</c> with x quantised as <c>quantize_q8_1.comp</c> does: per 32 elements d = amax / 127 (F32),
    /// q = round(x / d) (computed as x * (1 / d)), then dequantised back to q * d.
    /// </summary>
    private static float[] ReferenceWithQ8_1Activations(byte[] w, float[] x, int m, int k, float scale)
    {
        float[] xd = new float[k];
        for (int b = 0; b < k; b += 32)
        {
            float amax = 0f;
            for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(x[b + i]));
            float d = amax / 127f;
            float id = d != 0f ? 1f / d : 0f;
            for (int i = 0; i < 32; i++) xd[b + i] = MathF.Round(x[b + i] * id, MidpointRounding.AwayFromZero) * d;
        }
        float[] wf = Dequant(w, m * k);
        float[] y = new float[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * xd[c];
            y[r] = (float)(acc * scale);
        }
        return y;
    }

    /// <summary>
    /// #1066: the Bolt-shaped NVFP4 Nemotron-H fixture, prefilled for 4 tokens and then DECODED for the 5th (so the NVFP4 SSM
    /// projections run as GEMVs), against the CPU F32 twin's 5-token prefill. Covers both GEMVs; the decode path's counter proves
    /// which ran.
    /// </summary>
    /// <param name="variant">The GEMV that must run: the MMVQ (opted in, its size gate lowered to 0 for the small fixture), the
    /// coalesced FP32 GEMV (the default), or #1038's (forced by its legacy variable).</param>
    [SkippableTheory]
    [InlineData(GemvVariant.Mmvq)]
    [InlineData(GemvVariant.Coalesced)]
    [InlineData(GemvVariant.Legacy)]
    public void NemotronH_Nvfp4Ssm_DecodeMatchesCpuF32Twin(GemvVariant variant)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using (var probe = VulkanDevice.Create())
        {
            if (variant != GemvVariant.Legacy) Skip.IfNot(probe.HasSubgroupArithmetic, "no subgroup arithmetic");
            if (variant == GemvVariant.Mmvq) Skip.IfNot(probe.HasIntegerDotProduct, "no integer dot product");
        }
        using var legacy = variant == GemvVariant.Legacy ? new EnvVarScope(MatMulNvfp4F32Kernel.LegacyGemvEnvVar, "1") : null;
        using var mmvqOn = variant == GemvVariant.Mmvq ? new EnvVarScope(MatMulNvfp4F32Kernel.EnableMmvqEnvVar, "1") : null;
        using var mmvqGate = variant == GemvVariant.Mmvq ? new EnvVarScope(MatMulNvfp4F32Kernel.MmvqMinBytesEnvVar, "0") : null;
        int[] ids = NemotronHNvfp4SsmTests.Ids, pos = NemotronHNvfp4SsmTests.Pos;
        int last = ids.Length - 1;

        float[] cpu;
        using (var gguf = GgufFile.Open(NemotronHNvfp4SsmTests.WriteFixture(_scratch, Variant.F32Reference)))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using ITensor logits = model.Forward(ids, pos, deviceId: -1);
            cpu = LastRow(logits);
        }

        float[] vk;
        long before = Interlocked.Read(ref MatMulNvfp4F32Kernel.CoalescedGemvDispatches);
        long mmvqBefore = Interlocked.Read(ref MatMulNvfp4F32Kernel.MmvqDispatches);
        using (var gguf = GgufFile.Open(NemotronHNvfp4SsmTests.WriteFixture(_scratch, Variant.Nvfp4Scaled)))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
            using (model)
            {
                using var kv = kvFactory(16);
                using (model.Forward(ids[..last], pos[..last], deviceId: -1, kvCache: kv)) { }
                using ITensor logits = model.Forward(ids[last..], pos[last..], deviceId: -1, kvCache: kv);
                vk = LastRow(logits);
            }
        }
        long viaCoalesced = Interlocked.Read(ref MatMulNvfp4F32Kernel.CoalescedGemvDispatches) - before;
        long viaMmvq = Interlocked.Read(ref MatMulNvfp4F32Kernel.MmvqDispatches) - mmvqBefore;
        Assert.True(variant == GemvVariant.Coalesced ? viaCoalesced > 0 : viaCoalesced == 0, $"coalesced GEMV dispatches: {viaCoalesced}");
        Assert.True(variant == GemvVariant.Mmvq ? viaMmvq > 0 : viaMmvq == 0, $"MMVQ dispatches: {viaMmvq}");

        float maxAbs = cpu.Max(MathF.Abs);
        Assert.True(maxAbs > 0.5f, $"degenerate fixture: max |logit| {maxAbs}");
        float tol = variant == GemvVariant.Mmvq ? 3e-2f : 1e-2f;   // MMVQ: activations rounded to 8 bits per 32
        Assert.Equal(Array.IndexOf(cpu, cpu.Max()), Array.IndexOf(vk, vk.Max()));
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(MathF.Abs(cpu[i] - vk[i]) <= tol * maxAbs,
                $"logit {i}: cpu(F32 twin) {cpu[i]}, vulkan(NVFP4 decode) {vk[i]} (max |logit| {maxAbs})");
    }

    /// <summary>
    /// #1060: the NVFP4 to F16 dequant is exact (E2M1 x UE4M3 needs at most 6 mantissa bits, inside F16's normal range), so it must
    /// equal the CPU dequantizer bit for bit, UE4M3 edge bytes included.
    /// </summary>
    [SkippableFact]
    public void DequantF16_IsBitExactVsCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int m = 37, k = 320;
        var (w, _, _) = MakeCase(m, k, n: 1, scale: 1f);
        float[] cpu = Dequant(w, m * k);

        using var device = VulkanDevice.Create();
        using var kernel = Nvfp4DequantF16Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(w.Length);
        using var bufY = device.Allocate((long)m * k * sizeof(ushort));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        kernel.Launch(bufW, bufY, m * k / 64);
        float[] raw = new float[m * k / 2];   // two F16 per float word; Download reads float spans only
        device.Download(bufY, raw);
        ushort[] bits = System.Runtime.InteropServices.MemoryMarshal.Cast<float, ushort>(raw).ToArray();

        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.True((float)(Half)cpu[i] == cpu[i], $"[{i}]: CPU value {cpu[i]} is not exact in F16, so the premise is wrong");
            Assert.True(BitConverter.HalfToUInt16Bits((Half)cpu[i]) == bits[i] || (cpu[i] == 0 && (bits[i] & 0x7FFF) == 0),
                $"[{i}]: cpu {cpu[i]}, vulkan {BitConverter.UInt16BitsToHalf(bits[i])}");
        }
    }

    /// <summary>
    /// #1060: <see cref="IqF16PrefillMatmul"/>'s NVFP4 arm (exact F16 dequant, F16 coopmat GEMM with F32 accumulation, then the
    /// per-tensor scale on the F32 output) against the double-precision reference. The activations round to F16, so the bound is
    /// looser than the F32 GEMM's.
    /// </summary>
    [SkippableTheory]
    [InlineData(70, 192, 32, 0.37f)]     // partial tiles, scale
    [InlineData(64, 4096, 40, 1.3e-4f)]  // Bolt-like K and scale
    [InlineData(33, 128, 64, 1.0f)]      // no scale
    public void F16Prefill_MatchesCpuDequantReference(int m, int k, int n, float scale)
    {
        var (w, x, expected) = MakeCase(m, k, n, scale);
        float[] actual = RunF16Prefill(w, x, m, k, n, scale);

        float maxAbs = expected.Max(MathF.Abs);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 4e-3f * maxAbs,
                $"m={m} k={k} n={n} [{i}]: expected {expected[i]}, vulkan {actual[i]} (max |y| {maxAbs})");
    }

    /// <summary>
    /// #1060 regression guard for the #1059 trap (a tiny per-tensor scale applied after an F16 intermediate): at Bolt's scale (~1e-4) the
    /// UNSCALED products are ~1e4 x the real outputs, so here every unscaled dot product lies far beyond F16's 65504 (large sub-block
    /// scales, same-sign weights and activations) while the scaled result is ordinary. Passes only while the GEMM accumulates and writes
    /// F32 and the scale multiplies that F32 output; an F16 accumulator, an F16 output or an F16 copy before the scale is inf/NaN here.
    /// </summary>
    [SkippableFact]
    public void F16Prefill_BoltScale_UnscaledSumsBeyondF16Range_StayFinite()
    {
        const int m = 48, k = 4096, n = IqF16PrefillMatmul.DefaultMinTokens + 8; // at or above the F16-prefill threshold
        const float scale = 1.0e-4f;
        var rng = new Random(1060);
        int blocks = m * (k / 64);
        byte[] w = new byte[blocks * 36];
        for (int b = 0; b < blocks; b++)
        {
            for (int s = 0; s < 4; s++) w[b * 36 + s] = (byte)rng.Next(0x70, 0x7F);   // UE4M3 64..224 (0x7F would decode to 0)
            for (int q = 4; q < 36; q++) w[b * 36 + q] = (byte)(rng.Next(5, 8) | (rng.Next(5, 8) << 4)); // positive E2M1 6, 8, 12
        }
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(0.25 + 0.75 * rng.NextDouble());
        float[] expected = Reference(w, x, m, k, n, scale);

        float minUnscaled = expected.Min(MathF.Abs) / scale;
        Assert.True(minUnscaled > 10 * 65504f, $"premise: the smallest unscaled dot product {minUnscaled:E3} is not far beyond F16's range");

        float[] actual = RunF16Prefill(w, x, m, k, n, scale);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && MathF.Abs(actual[i] - expected[i]) <= 4e-3f * MathF.Abs(expected[i]),
                $"[{i}]: expected {expected[i]}, vulkan {actual[i]}");
    }

    /// <summary>Runs <see cref="IqF16PrefillMatmul"/>'s NVFP4 arm once and checks the path counter.</summary>
    /// <param name="w">Packed NVFP4 weights, <c>[m, k]</c>.</param>
    /// <param name="x">F32 activations, <c>[n, k]</c>.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Input width.</param>
    /// <param name="n">Tokens.</param>
    /// <param name="scale">Per-tensor scale.</param>
    /// <returns>The F32 output, <c>[n, m]</c>.</returns>
    private static float[] RunF16Prefill(byte[] w, float[] x, int m, int k, int n, float scale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "no cooperative-matrix F16 GEMM on this device");
        using var gemm = MatMulF16GemmCoopmatKernel.Create(device, spvDir);
        using var prefill = IqF16PrefillMatmul.TryCreate(device, spvDir, gemm, (long)m * k)
            ?? throw new InvalidOperationException("F16 prefill helper unavailable");
        using var bufW = device.Allocate(w.Length);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        IqF16PrefillMatmul.Counters.Reset();
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            Assert.True(prefill.TryRecord(ctx.CommandBuffer, QuantizationType.NVFP4, bufW, bufX, bufY, m, k, n, scale),
                "the F16 prefill declined an NVFP4 matmul");
            ctx.SubmitAndWait();
        }
        Assert.Equal(1, Interlocked.Read(ref IqF16PrefillMatmul.Counters.Nvfp4));
        float[] actual = new float[n * m];
        device.Download(bufY, actual);
        return actual;
    }

    /// <summary>
    /// #1064: the native NVFP4 coopmat GEMM (decode into shared memory, F32 accumulation, X.scale on the F32 accumulator) against the
    /// double-precision reference. Activations round to F16 as in every coopmat GEMM, hence the F16-prefill bound. Shapes cover partial
    /// M and N tiles on both sides of the 128 tile, a single-block K, an odd number of 32-wide K chunks per block pair, and Bolt's K.
    /// </summary>
    [SkippableTheory]
    [InlineData(70, 192, 32, 0.37f)]      // partial M and N tiles, 3 blocks, scale
    [InlineData(64, 4096, 40, 1.3e-4f)]   // Bolt-like K and scale
    [InlineData(33, 128, 64, 1.0f)]       // no scale
    [InlineData(130, 64, 129, 1.0f)]      // one block of K; M and N each cross a 128 tile by one
    [InlineData(256, 4096, 256, 2.2e-4f)] // whole tiles only (the direct-store path)
    [InlineData(70, 192, 2, 0.5f)]        // the smallest batch the Nemotron-H host routes here
    public void NativeGemm_MatchesCpuDequantReference(int m, int k, int n, float scale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (w, x, expected) = MakeCase(m, k, n, scale);

        float[] actual = RunNativeGemm(spvDir, w, x, m, k, n, scale);

        float maxAbs = expected.Max(MathF.Abs);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 4e-3f * maxAbs,
                $"m={m} k={k} n={n} [{i}]: expected {expected[i]}, vulkan {actual[i]} (max |y| {maxAbs})");
    }

    /// <summary>
    /// #1064 regression fixture for #1059's overflow class, at Bolt magnitudes: every weight is the largest NVFP4 value under a canonical E4M3 scale
    /// (code 7 = 12, times UE4M3 0x7E = 224, i.e. 2688) and every activation 16, over Bolt's K = 4096, so each unscaled dot product
    /// is 4096 x 2688 x 16 = 176 160 768, far past F16's 65 504. Scaled by ~1e-4 it is ~2.3e4. The products and their partial sums
    /// are multiples of 2^11 below 2^28, so F32 accumulation is exact and the only rounding is the one F32 multiply by the scale. An
    /// F16 accumulator or an F16 write before the scale gives Inf/NaN; an F16 write after it is off by up to 2^-11; a scale applied
    /// twice or not at all is off by ~1e4x. Odd rows use the negative code so the sign path is covered too. Tails on both M and N.
    /// </summary>
    [SkippableTheory]
    [InlineData(1.3e-4f)]
    [InlineData(3.3e-4f)]
    public void NativeGemm_BoltScaleMagnitudes_AccumulatesAndScalesInF32(float scale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int m = 130, k = 4096, n = 129, blocksPerRow = k / 64;
        byte[] w = new byte[m * blocksPerRow * 36];
        for (int r = 0; r < m; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int o = (r * blocksPerRow + b) * 36;
                w.AsSpan(o, 4).Fill(0x7E);                                          // UE4M3 0x7E -> 224 (the x0.5 folded in)
                w.AsSpan(o + 4, 32).Fill((r & 1) == 0 ? (byte)0x77 : (byte)0xFF); // code 7 (+12) or 15 (-12) in both nibbles
            }
        float[] x = new float[n * k];
        x.AsSpan().Fill(16f);

        float[] actual = RunNativeGemm(spvDir, w, x, m, k, n, scale);

        const float unscaled = 4096f * 2688f * 16f;
        Assert.True(unscaled > 65504f * 1000f, "the fixture must overflow any F16 intermediate");
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                float expected = ((r & 1) == 0 ? unscaled : -unscaled) * scale;   // one F32 multiply, as the epilogue does
                float got = actual[t * m + r];
                Assert.True(float.IsFinite(got), $"[t={t}, r={r}] is {got}: an F16 intermediate overflowed");
                Assert.True(MathF.Abs(got - expected) <= 1e-6f * MathF.Abs(expected),
                    $"[t={t}, r={r}]: expected {expected}, vulkan {got}");
            }
    }

    private static float[] RunNativeGemm(string spvDir, byte[] w, float[] x, int m, int k, int n, float scale)
    {
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulNvfp4GemmCoopmatKernel.IsSupportedOn(device, spvDir), "no wave64 cooperative-matrix device");
        using var kernel = MatMulNvfp4GemmCoopmatKernel.Create(device, spvDir);
        using var bufW = device.Allocate(w.Length);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        kernel.Launch(bufW, bufX, bufY, m, k, n, scale);
        float[] actual = new float[n * m];
        device.Download(bufY, actual);
        return actual;
    }

    /// <summary>Random NVFP4 blocks (sub-block scales sprinkled with the UE4M3 edge bytes), activations and the double-precision product.</summary>
    private static (byte[] W, float[] X, float[] Expected) MakeCase(int m, int k, int n, float scale)
    {
        var rng = new Random(m * 131 + k * 7 + n);
        int blocks = m * (k / 64);
        byte[] w = new byte[blocks * 36];
        rng.NextBytes(w);
        byte[] edge = [0x00, 0x7F, 0xFF, 0x01, 0x80, 0x7E];
        for (int b = 0; b < blocks; b++)
            for (int s = 0; s < 4; s++)
                w[b * 36 + s] = rng.Next(8) == 0 ? edge[rng.Next(edge.Length)] : (byte)rng.Next(0x20, 0x50);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        return (w, x, Reference(w, x, m, k, n, scale));
    }

    /// <summary>Double-precision <c>scale * (x . dequant(w)^T)</c>, <c>[n, m]</c>.</summary>
    private static float[] Reference(byte[] w, float[] x, int m, int k, int n, float scale)
    {
        float[] wf = Dequant(w, m * k);
        float[] expected = new float[n * m];
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * x[t * k + c];
                expected[t * m + r] = (float)(acc * scale);
            }
        return expected;
    }

    private static unsafe float[] Dequant(byte[] w, int elements)
    {
        float[] wf = new float[elements];
        fixed (byte* pw = w)
            Dequantize.ToFloat32((nint)pw, wf.Length, QuantizationType.NVFP4, wf);
        return wf;
    }

    /// <summary>
    /// The Bolt-shaped fixture (NVFP4 ssm_in/ssm_out with per-tensor scales 0.05..3) through
    /// <see cref="VulkanModelLoader.CreateFromGguf"/>, against the CPU model on the F32 twin whose SSM weights are
    /// <c>scale · dequant(W)</c>. Vulkan uses FP32 activations, so this is tight; a dropped scale, a wrong
    /// nibble order or a widened-and-unscaled upload is off by a multiple of the logits.
    /// </summary>
    /// <param name="path">Which prefill GEMM the 5-token prompt's NVFP4 projections must take: the native coopmat GEMM (#1064, the
    /// default; activations rounded to F16: a looser bound), the F16 dequant + coopmat GEMM (#1060, native disabled and its threshold
    /// lowered), or the packed scalar NVFP4 GEMM (#1038, both disabled).</param>
    [SkippableTheory]
    [InlineData(Nvfp4PrefillPath.Scalar)]
    [InlineData(Nvfp4PrefillPath.F16Dequant)]
    [InlineData(Nvfp4PrefillPath.Native)]
    public void NemotronH_Nvfp4Ssm_ForwardMatchesCpuF32Twin(Nvfp4PrefillPath path)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var prefillKernels = new RowExactOffScope();   // a short batch would otherwise take the decode GEMVs (#1237)
        if (path != Nvfp4PrefillPath.Scalar)
        {
            using var probe = VulkanDevice.Create();
            Skip.IfNot(probe.HasCooperativeMatrix, "no cooperative-matrix F16 GEMM on this device");
            if (path == Nvfp4PrefillPath.Native)
                Skip.IfNot(MatMulNvfp4GemmCoopmatKernel.IsSupportedOn(probe, spvDir), "no wave64 cooperative-matrix device");
        }
        using var f16Scope = path == Nvfp4PrefillPath.F16Dequant ? new F16PrefillTestScope() : null;
        using var nativeOff = path != Nvfp4PrefillPath.Native ? new EnvVarScope(MatMulNvfp4GemmCoopmatKernel.LegacyEnvVar, "1") : null;
        IqF16PrefillMatmul.Counters.Reset();
        Interlocked.Exchange(ref MatMulNvfp4GemmCoopmatKernel.Dispatches, 0);
        long scratchBytes = -1, scratchCapacity = -1;

        float[] cpu;
        using (var gguf = GgufFile.Open(NemotronHNvfp4SsmTests.WriteFixture(_scratch, Variant.F32Reference)))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using ITensor logits = model.Forward(NemotronHNvfp4SsmTests.Ids, NemotronHNvfp4SsmTests.Pos, deviceId: -1);
            cpu = LastRow(logits);
        }

        float[] vk;
        using (var gguf = GgufFile.Open(NemotronHNvfp4SsmTests.WriteFixture(_scratch, Variant.Nvfp4Scaled)))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            var (model, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
            using (model)
            {
                Assert.IsType<VulkanNemotronHTransformerModel>(model);
                using ITensor logits = model.Forward(NemotronHNvfp4SsmTests.Ids, NemotronHNvfp4SsmTests.Pos, deviceId: -1);
                vk = LastRow(logits);
                scratchBytes = ((VulkanNemotronHTransformerModel)model).F16PrefillScratchBytes;
                scratchCapacity = ((VulkanNemotronHTransformerModel)model).F16PrefillScratchCapacityBytes;
            }
        }

        long viaF16 = Interlocked.Read(ref IqF16PrefillMatmul.Counters.Nvfp4);
        long viaNative = Interlocked.Read(ref MatMulNvfp4GemmCoopmatKernel.Dispatches);
        switch (path)
        {
            case Nvfp4PrefillPath.Native:
                Assert.True(viaNative > 0, "the NVFP4 projections did not take the native coopmat GEMM");
                Assert.Equal(0, viaF16);
                Assert.Equal(0, scratchBytes);   // #1064: no F16 staging buffer exists ...
                Assert.True(scratchCapacity <= sizeof(ushort),   // ... nor is one sized for NVFP4 (the fixture has no IQ projection)
                    $"F16 scratch sized to {scratchCapacity} bytes for an NVFP4-only model with the native GEMM");
                break;
            case Nvfp4PrefillPath.F16Dequant:
                Assert.True(viaF16 > 0, "the NVFP4 projections did not take the F16 prefill path");
                Assert.Equal(0, viaNative);
                Assert.True(scratchBytes > 0, "the F16 path ran without its scratch");
                Assert.Equal(scratchCapacity, scratchBytes);
                break;
            default:
                Assert.Equal(0, viaF16);
                Assert.Equal(0, viaNative);
                break;
        }

        float maxAbs = cpu.Max(MathF.Abs);
        float tol = path == Nvfp4PrefillPath.Scalar ? 2e-3f : 1e-2f;
        Assert.True(maxAbs > 0.5f, $"degenerate fixture: max |logit| {maxAbs}");
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(MathF.Abs(cpu[i] - vk[i]) <= tol * maxAbs,
                $"logit {i}: cpu(F32 twin) {cpu[i]}, vulkan(NVFP4) {vk[i]} (max |logit| {maxAbs})");
    }

    /// <summary>The NVFP4 prefill GEMM a test forces on the Nemotron-H host.</summary>
    public enum Nvfp4PrefillPath
    {
        /// <summary>The packed scalar 16x16 NVFP4 GEMM (#1038).</summary>
        Scalar,

        /// <summary>Whole-matrix NVFP4 to F16 dequant, then the F16 coopmat GEMM (#1060).</summary>
        F16Dequant,

        /// <summary>The native NVFP4 coopmat GEMM (#1064).</summary>
        Native,
    }

    [Fact]
    public void ScaleOnNonNvfp4Projection_IsRefused()
    {
        VulkanNemotronHWeights.ThrowIfScaleUnsupported("ssm_in", 1.0f, QuantizationType.F32);
        VulkanNemotronHWeights.ThrowIfScaleUnsupported("ssm_in", 0.5f, QuantizationType.NVFP4);
        Assert.Throws<NotSupportedException>(() =>
            VulkanNemotronHWeights.ThrowIfScaleUnsupported("ssm_in", 0.5f, QuantizationType.F32));
    }

    private static unsafe float[] LastRow(ITensor logits)
    {
        int vocab = NemotronHNvfp4SsmTests.Vocab;
        int rows = (int)(logits.Shape.ElementCount / vocab);
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (rows - 1) * vocab, vocab).ToArray();
    }
}
