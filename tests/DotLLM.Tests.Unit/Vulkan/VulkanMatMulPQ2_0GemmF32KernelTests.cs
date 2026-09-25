using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Correctness gates for the Vulkan PQ2_0 (PrismML Bonsai ternary) prefill GEMM — every variant
/// <see cref="PQ2_0GemmVariant.AvailableOn"/> reports for the device, so the register-blocked F32
/// kernel of issue #233 and the coopmat kernels of issue #236 are held to the same gates. Three
/// layers, in increasing order of authority:
/// <list type="number">
/// <item><see cref="Launch_MatchesScalarReference"/> — synthetic parity against a
/// double-precision ground truth across tile-ragged shapes.</item>
/// <item><see cref="RealBonsaiTensor_DequantIsBitExactVsCpuOracle"/> — <b>bit-exact</b>
/// agreement with the CPU oracle on a real Bonsai-27B tensor (1 ULP for the coopmat variants;
/// see that method's remarks for why the gap is the device's), via a one-hot activation matrix
/// that makes every output cell a single exact product.</item>
/// <item><see cref="RealBonsaiTensor_MatchesCpuGemmReference"/> — full GEMM against
/// <c>MatMul.GemmPQ2_0</c> on real Bonsai-27B weights.</item>
/// </list>
/// </summary>
/// <remarks>
/// <para><b>Why "bit-exact" is layered rather than asserted on the full GEMM.</b> Float addition
/// is not associative, and the GPU (per-thread sequential accumulation over 128-element K-chunks)
/// and the CPU reference (<c>TensorPrimitives.Dot</c>, multi-lane pairwise) sum in different
/// orders. No GPU GEMM can be bit-identical to a vectorised CPU dot on a general input, so
/// demanding it of layer 3 would be demanding an impossibility rather than testing anything.
/// What <i>can</i> be pinned exactly is the part where a bug would actually live: the
/// dequantisation — the 34-byte group stride, the fp16 group-scale read, the {6,4,2,0} bit
/// unpack, and the code−1 mapping. Layer 2 isolates that: with a one-hot B, each output cell is
/// <c>w · 1.0</c> plus a sum of exact zeros, so it must equal the CPU-dequantised weight
/// bit-for-bit, and it is checked with <see cref="Assert.Equal(float, float)"/> on the raw bit
/// patterns. Layers 1 and 3 then cover the tiling and accumulation at fp32 rounding scale.</para>
/// <para>Tolerance for the tolerance-based layers — PQ2_0 codes are exact ternary, so divergence
/// is fp16-group-scale rounding plus reduction order. Same 5e-3 / 1e-3 tolerances as the PQ2_0
/// GEMV and I2_S GEMM parity tests.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulPQ2_0GemmF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;
    private const int GroupSize = 128;
    private const int GroupBytes = 34;

    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-27B-Q2_0.gguf";

    /// <summary>
    /// Synthetic parity against a double-precision ground truth
    /// (<c>C[t·M + m] = Σ_g scale(m,g) · Σ_{c∈g} ternary[m,c] · B[t,c]</c>).
    /// </summary>
    /// <remarks>
    /// The shapes deliberately straddle the 32x32 output tile and the SUB=16 micro-tile stride in
    /// both dimensions: the register-blocked mapping guards its four micro-tile corners
    /// separately, and only a shape that is ragged in both dims can catch a corner whose bound
    /// check or store offset is wrong.
    /// </remarks>
    /// <param name="m">Weight rows (output columns of C).</param>
    /// <param name="k">Shared dimension; must be a multiple of 128.</param>
    /// <param name="n">Token rows (batch).</param>
    [SkippableTheory]
    [InlineData(16, 128, 4)]      // m and n both below TILE — every micro-tile corner guarded
    [InlineData(32, 256, 8)]      // exactly one tile, 2 groups per row (distinct group scales)
    [InlineData(64, 128, 16)]     // 2 tiles in m, partial in n
    [InlineData(48, 768, 12)]     // partial tile in both dims, 6 groups
    [InlineData(33, 128, 33)]     // one element past a full tile in both dims
    [InlineData(17, 256, 47)]     // ragged in both dims, straddling the SUB=16 stride
    [InlineData(15, 128, 3)]      // below the micro-tile stride entirely
    [InlineData(2560, 2560, 5)]   // hidden × hidden, small prefill batch
    [InlineData(576, 1024, 64)]   // wide prefill batch (2 full tiles in n)
    public void Launch_MatchesScalarReference(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x2A_51 ^ (m * 7 + k * 11 + n * 13));
        int groups = k / GroupSize;

        sbyte[] ternary = new sbyte[(long)m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);

        Half[] scales = new Half[(long)m * groups];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);

        float[] inputB = new float[(long)n * k];
        for (int i = 0; i < inputB.Length; i++) inputB[i] = rng.NextSingle() * 2f - 1f;

        byte[] weights = PackPQ2_0(ternary, scales, m, k);

        float[] expected = new float[(long)n * m];
        for (int t = 0; t < n; t++)
        {
            long bRowBase = (long)t * k;
            for (int r = 0; r < m; r++)
            {
                long wRowBase = (long)r * k;
                double acc = 0;
                for (int g = 0; g < groups; g++)
                {
                    double groupScale = (float)scales[(long)r * groups + g];
                    double groupAcc = 0;
                    int groupBase = g * GroupSize;
                    for (int c = 0; c < GroupSize; c++)
                        groupAcc += ternary[wRowBase + groupBase + c] * (double)inputB[bRowBase + groupBase + c];
                    acc += groupScale * groupAcc;
                }
                expected[(long)t * m + r] = (float)acc;
            }
        }

        foreach (var (variant, actual) in RunAllVariants(spvDir, weights, inputB, m, k, n))
        {
            for (int t = 0; t < n; t++)
            {
                for (int r = 0; r < m; r++)
                {
                    long idx = (long)t * m + r;
                    float diff = MathF.Abs(expected[idx] - actual[idx]);
                    float tol = AbsTol + RelTol * MathF.Abs(expected[idx]);
                    Assert.True(diff <= tol,
                        $"[{variant.SpvFileName}] cell t={t} m={r} (m={m}, k={k}, n={n}): expected {expected[idx]:G9}, got {actual[idx]:G9}, |Δ|={diff:G9} > tol {tol:G9}");
                }
            }
        }
    }

    /// <summary>
    /// <b>Dequantisation gate.</b> Feeds a one-hot activation matrix
    /// (<c>B[t, c] = 1.0 when c == k0 + t, else 0</c>) against a slice of a <i>real</i>
    /// Bonsai-27B PQ2_0 tensor, so every output cell reduces to one exact product
    /// <c>dequant(W[m, k0+t]) · 1.0</c> plus exact zeros. This pins the 34-byte group stride, the
    /// unaligned little-endian fp16 group-scale read, the {6,4,2,0} bit unpack and the code−1
    /// mapping against real packed bytes, with no reduction-order escape hatch.
    /// </summary>
    /// <remarks>
    /// <para><b>The F32 kernels are held to bit-exactness; the coopmat kernels are held to 1 ULP,
    /// and that gap is the device's, not ours (issue #236).</b> On gfx1151 (Radeon 8060S, RDNA3.5)
    /// <c>coopMatMulAdd</c> is <i>not</i> IEEE-exact even when the exact result is representable:
    /// with an F16 A operand of exactly ±1.0, an F16 B operand of exactly 1.0 and an F32
    /// accumulator seeded to zero, roughly a third of cells come back one F32 ULP toward −∞
    /// (1.0 → 0x3F7FFFFF, −3.0 → 0xC0400001). Measured properties: independent of K
    /// (identical counts at K = 128, 256 and 1024, so it is not accumulation order), independent of
    /// the scale value, unchanged by pinning the pipeline to wave32 via
    /// <c>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</c>, and — decisively — reproduced
    /// byte-for-byte by the pre-existing, untouched <c>matmul_i2_s_f32_gemm_coopmat.spv</c>, which
    /// stages raw ternary and applies its scale afterwards. So it is a property of the driver's
    /// coopmat path, not of PQ2_0 decode or of any scale folding.</para>
    /// <para>A 1-ULP envelope still gates the decode as hard as bit-exactness would: no wrong group
    /// stride, wrong scale byte offset, wrong bit position or wrong code mapping can land within one
    /// F32 ULP of the right answer on 6144 consecutive real-weight cells. What the envelope gives up
    /// is only the ability to detect a device that rounds — which is exactly what it is
    /// documenting.</para>
    /// </remarks>
    [SkippableFact]
    public unsafe void RealBonsaiTensor_DequantIsBitExactVsCpuOracle()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? path = ResolveFixturePath();
        Skip.If(path is null, FixtureSkipMessage);

        using var gguf = GgufFile.Open(path!);
        var (weights, m, k) = LoadRealSlice(gguf, maxRows: 64, maxGroups: 8);

        // CPU oracle: dequantise the same slice via the shipped CPU path.
        float[] cpuDequant = new float[(long)m * k];
        fixed (byte* wPtr = weights)
            Dequantize.ToFloat32((nint)wPtr, (long)m * k, QuantizationType.PQ2_0, cpuDequant);

        // One-hot B selecting the first GroupSize columns — n = 128 tokens, each a basis vector.
        int n = GroupSize;
        float[] inputB = new float[(long)n * k];
        for (int t = 0; t < n; t++) inputB[(long)t * k + t] = 1.0f;

        foreach (var (variant, actual) in RunAllVariants(spvDir, weights, inputB, m, k, n))
        {
            // F32 kernels must be bit-identical. Coopmat kernels get 1 ULP, and only 1 — see
            // the remarks: the device's coopMatMulAdd rounds, our decode does not.
            long ulpBudget = variant.RequiresCooperativeMatrix ? 1 : 0;

            for (int t = 0; t < n; t++)
            {
                for (int r = 0; r < m; r++)
                {
                    float expected = cpuDequant[(long)r * k + t];
                    float got = actual[(long)t * m + r];
                    int e = BitConverter.SingleToInt32Bits(expected);
                    int g = BitConverter.SingleToInt32Bits(got);
                    // Both operands share a sign here (the coopmat drift is sub-ULP, never a sign
                    // flip), so the raw bit-pattern distance is a valid ULP distance.
                    long ulps = Math.Abs((long)e - g);
                    Assert.True(ulps <= ulpBudget,
                        $"[{variant.SpvFileName}] {ulps} ULP > budget {ulpBudget} at row {r}, col {t}: "
                        + $"CPU oracle {expected:G9} (0x{BitConverter.SingleToUInt32Bits(expected):X8}), "
                        + $"GPU {got:G9} (0x{BitConverter.SingleToUInt32Bits(got):X8}).");
                }
            }
        }
    }

    /// <summary>
    /// Full-GEMM parity against the CPU reference <c>MatMul.GemmPQ2_0Scalar</c> on real Bonsai-27B
    /// packed weights with a random activation batch. Complements the bit-exactness gate: that
    /// one pins the decode, this one pins the tiling and accumulation over a real weight
    /// distribution (real per-group fp16 scales, real ternary sparsity) at fp32 rounding scale.
    ///
    /// <para><b>Reference tier matters (issue #229).</b> Uses <c>GemmPQ2_0Scalar</c>, the float
    /// tier, NOT the dispatching <c>MatMul.GemmPQ2_0</c> — that one takes the W2A8 int8-activation
    /// path on any AVX2 host, while this Vulkan kernel is F32-in. Comparing them would measure
    /// per-token activation-quantization error rather than fp32 rounding, and would silently change
    /// with the host's ISA. Same defect as the CUDA parity tests fixed under #229.</para>
    /// </summary>
    [SkippableFact]
    public unsafe void RealBonsaiTensor_MatchesCpuGemmReference()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? path = ResolveFixturePath();
        Skip.If(path is null, FixtureSkipMessage);

        using var gguf = GgufFile.Open(path!);
        var (weights, m, k) = LoadRealSlice(gguf, maxRows: 128, maxGroups: 16);

        const int n = 40;   // ragged against the 32-token tile
        var rng = new Random(0x2A_52);
        float[] inputB = new float[(long)n * k];
        for (int i = 0; i < inputB.Length; i++) inputB[i] = rng.NextSingle() * 2f - 1f;

        float[] expected = new float[(long)n * m];
        fixed (byte* wPtr = weights)
        fixed (float* bPtr = inputB)
        fixed (float* cPtr = expected)
            MatMul.GemmPQ2_0Scalar(wPtr, bPtr, cPtr, m, k, n);

        foreach (var (variant, actual) in RunAllVariants(spvDir, weights, inputB, m, k, n))
        {
            double maxAbs = 0, maxRel = 0;
            for (long i = 0; i < expected.LongLength; i++)
            {
                float diff = MathF.Abs(expected[i] - actual[i]);
                float tol = AbsTol + RelTol * MathF.Abs(expected[i]);
                maxAbs = Math.Max(maxAbs, diff);
                if (expected[i] != 0f) maxRel = Math.Max(maxRel, diff / MathF.Abs(expected[i]));
                Assert.True(diff <= tol,
                    $"[{variant.SpvFileName}] cell {i} (m={m}, k={k}, n={n}): CPU {expected[i]:G9}, GPU {actual[i]:G9}, |Δ|={diff:G9} > tol {tol:G9}");
            }

            Assert.True(maxAbs < AbsTol, $"[{variant.SpvFileName}] max |Δ| {maxAbs:G6}, max relative {maxRel:G6}");
        }
    }

    /// <summary>
    /// Every GEMM variant this device can actually create: always the register-blocked F32
    /// kernel, plus the coopmat variants when <c>VK_KHR_cooperative_matrix</c> is advertised
    /// and the declared workgroup size maps to one subgroup here.
    /// </summary>
    /// <remarks>
    /// Each correctness gate runs over ALL of them rather than only
    /// <see cref="PQ2_0GemmVariant.SelectFor"/>'s pick, so a variant cannot rot unmeasured while
    /// a benchmark still selects it.
    /// </remarks>
    private static IEnumerable<PQ2_0GemmVariant> AvailableVariants(VulkanDevice device)
        => PQ2_0GemmVariant.AvailableOn(device);

    /// <summary>Uploads, dispatches and downloads one GEMM per available variant.</summary>
    /// <returns>One (variant, result) pair per variant the device can run.</returns>
    private static List<(PQ2_0GemmVariant Variant, float[] Result)> RunAllVariants(
        string spvDir, byte[] weights, float[] inputB, int m, int k, int n)
    {
        using var device = VulkanDevice.Create();

        long weightsBufBytes = ((long)weights.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weights), bufW);
        device.Upload(inputB, bufB);

        var results = new List<(PQ2_0GemmVariant, float[])>();
        foreach (var variant in AvailableVariants(device))
        {
            using var kernel = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, variant);
            kernel.Launch(bufW, bufB, bufC, m, k, n);

            float[] actual = new float[(long)n * m];
            device.Download(bufC, actual);
            results.Add((variant, actual));
        }
        return results;
    }

    /// <summary>
    /// Copies a <c>[rows, groups·128]</c> corner out of the smallest PQ2_0 tensor in the file,
    /// re-packed to a contiguous <c>(k/128)·34</c> row stride. Keeps the GPU-side working set
    /// small while using genuinely real packed bytes (real fp16 group scales, real code
    /// distribution) rather than synthetic ones.
    /// </summary>
    private static unsafe (byte[] Weights, int M, int K) LoadRealSlice(
        GgufFile gguf, int maxRows, int maxGroups)
    {
        var tensor = gguf.Tensors
            .Where(t => t.QuantizationType == QuantizationType.PQ2_0 && t.Shape.Rank == 2)
            .OrderBy(t => t.Shape.ElementCount)
            .First();

        // GGUF is row-major with dim[0] the contraction axis for a 2-D weight.
        int srcK = (int)tensor.Shape[0];
        int srcM = (int)tensor.Shape[1];
        Assert.True(srcK % GroupSize == 0, $"'{tensor.Name}' K={srcK} is not a multiple of {GroupSize}.");

        int groups = Math.Min(maxGroups, srcK / GroupSize);
        int m = Math.Min(maxRows, srcM);
        int k = groups * GroupSize;

        int srcRowBytes = (srcK / GroupSize) * GroupBytes;
        int dstRowBytes = groups * GroupBytes;
        byte[] buf = new byte[(long)m * dstRowBytes];

        byte* basePtr = (byte*)(gguf.DataBasePointer + (nint)tensor.DataOffset);
        for (int r = 0; r < m; r++)
            new ReadOnlySpan<byte>(basePtr + (long)r * srcRowBytes, dstRowBytes)
                .CopyTo(buf.AsSpan(r * dstRowBytes, dstRowBytes));

        return (buf, m, k);
    }

    private const string FixtureSkipMessage =
        "Bonsai PQ2_0 GGUF fixture not found. Set " + ModelPathEnvVar + ", or place " + FileName
        + " under ~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ or "
        + "~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.";

    /// <summary>Mirrors <c>PQ2_0RealGgufSmokeTests.ResolveFixturePath</c>.</summary>
    private static string? ResolveFixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
            Path.Combine(home, ".dotllm", "test-cache", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
        ];
        foreach (string candidate in candidates)
            if (File.Exists(candidate))
                return candidate;

        return null;
    }

    /// <summary>
    /// Packs a row-major <c>[m, k]</c> ternary matrix + per-(row,group) fp16 scales into the
    /// PQ2_0 byte layout the kernel decodes: each 128-element group is 34 bytes — a leading
    /// little-endian fp16 scale, then 32 bytes where byte <c>b</c> holds 4 CONSECUTIVE
    /// group-relative positions {4b, 4b+1, 4b+2, 4b+3} at ASCENDING bit offsets {0, 2, 4, 6}
    /// (stored code = value + 1 ∈ {0,1,2}). Row stride = (k/128)·34 bytes. See issue #271 —
    /// this is PrismML's real format, verified byte-for-byte against their reference
    /// dequantize_row_q2_0, not I2_S's strided {gp,gp+32,gp+64,gp+96} interleave this helper
    /// used to (wrongly) assume. Mirrors <see cref="VulkanMatMulPQ2_0GemvF32KernelTests"/>'s
    /// packer.
    /// </summary>
    private static byte[] PackPQ2_0(sbyte[] ternary, Half[] scales, int m, int k)
    {
        int groups = k / GroupSize;
        int rowBytes = groups * GroupBytes;
        byte[] buf = new byte[(long)m * rowBytes];
        for (int r = 0; r < m; r++)
        {
            long rowBase = (long)r * k;
            long rowByteBase = (long)r * rowBytes;
            for (int g = 0; g < groups; g++)
            {
                long groupByteBase = rowByteBase + g * GroupBytes;
                BitConverter.GetBytes(scales[(long)r * groups + g]).CopyTo(buf, groupByteBase);
                long codeBase = groupByteBase + 2;
                int groupElemBase = g * GroupSize;
                for (int p = 0; p < GroupSize; p++)
                {
                    int code = ternary[rowBase + groupElemBase + p] + 1;   // {-1,0,1} -> {0,1,2}
                    int byteInGroup = p / 4;
                    int shift = 2 * (p % 4);                               // 4b+0->0, 4b+1->2, 4b+2->4, 4b+3->6
                    buf[codeBase + byteInGroup] |= (byte)(code << shift);
                }
            }
        }
        return buf;
    }
}
