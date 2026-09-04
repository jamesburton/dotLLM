using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Real-GGUF-bytes parity for the routed-expert (<c>moe_indexed</c>) matmul kernels
/// added by #407 — <see cref="MoeIndexedMatmulQ5_0F32Kernel"/> and
/// <see cref="MoeIndexedMatmulIq4NlF32Kernel"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why real bytes (#344 rules 1 and 2).</b> Q3_K shipped broken for months because
/// its self-authored fixture encoded with the same transposed layout its kernels
/// decoded with (#311) — the encoder and the decoder agreed, and the test passed. Every
/// weight byte consumed here comes from a llama.cpp-quantised GGUF; the only thing this
/// test assembles is the <i>arrangement</i> of already-real rows into an
/// <c>[numExperts, M, K]</c> expert bank, which is a pure memcpy-free reinterpretation:
/// GGUF stores a <c>[K, M]</c> tensor as M contiguous packed rows, so the first
/// <c>E · Me</c> rows ARE a valid E-expert bank with per-expert stride
/// <c>Me · rowBytes</c>, byte-for-byte.
/// </para>
/// <para>
/// <b>Q5_0's qh hazard (#344's named highest-risk surface).</b> Q5_0 keeps the 5th bit
/// of every weight in a separate 32-bit <c>qh</c> field, and the element → (nibble,
/// <c>qh</c> bit) mapping is the same indexing class that Q3_K got transposed
/// self-consistently across every backend. The reference here is
/// <c>DotLLM.Cpu.Kernels.Dequantize</c> — the CPU scalar path, which is the only valid
/// cross-backend reference for a packed layout (CUDA decodes once at load into FP16 and
/// cannot serve as an oracle, #330).
/// </para>
/// <para>
/// <b>Negative control (#344 rule 3).</b> Each parity test is paired with a
/// <c>Discriminates</c> sibling that recomputes the reference with the exact bug class
/// under suspicion — a transposed <c>qh</c> bit index for Q5_0, swapped nibble planes
/// for IQ4_NL — and asserts the kernel output does NOT match it. A test that passes
/// against both the correct and the broken reference proves nothing; these assert the
/// gap is real on this hardware, in this session.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class RealGgufMoeIndexedRoutedBankParityTests
{
    /// <summary>Experts synthesised from consecutive real tensor rows.</summary>
    private const int NumExperts = 4;

    /// <summary>Rows per expert (bank M). Keeps the double-precision CPU reference bounded.</summary>
    private const int RowsPerExpert = 8;

    /// <summary>Expanded MoE rows (seqLen × topK) driven through the kernel.</summary>
    private const int Rows = 6;

    /// <summary>Tensors checked per run.</summary>
    private const int MaxTensors = 6;

    /// <summary>
    /// Relative-error bound, applied against an RMS-of-expected denominator. Same shape
    /// and threshold as <c>RealGgufQ5_0ParityTests.RelTol</c> (#344 Task 5/6), which was
    /// itself validated against a real transposed-bit-layout bug on real hardware (#311).
    /// </summary>
    private const double RelTol = 2e-2;

    private readonly ITestOutputHelper _output;

    /// <summary>Creates the fixture.</summary>
    /// <param name="output">xUnit output sink.</param>
    public RealGgufMoeIndexedRoutedBankParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Q5_0 routed-bank matmul vs the CPU scalar dequant oracle, on real GGUF bytes.
    /// </summary>
    [SkippableFact]
    public void Q5_0_RealGgufBytes_MoeIndexedMatmul_MatchesCpuReference()
        => RunParity(QuantizationType.Q5_0, breakReference: false);

    /// <summary>
    /// Negative control for <see cref="Q5_0_RealGgufBytes_MoeIndexedMatmul_MatchesCpuReference"/>:
    /// the reference is recomputed with the <c>qh</c> bit for element <c>j</c> read from
    /// bit <c>2j (mod 32)</c> instead of bit <c>j</c> — the #311 failure class named by
    /// #344 — and the kernel must NOT match it.
    /// </summary>
    [SkippableFact]
    public void Q5_0_ParityAssertion_Discriminates_AgainstTransposedQhMapping()
        => RunParity(QuantizationType.Q5_0, breakReference: true);

    /// <summary>
    /// IQ4_NL routed-bank matmul vs the CPU scalar dequant oracle, on real GGUF bytes.
    /// </summary>
    [SkippableFact]
    public void Iq4Nl_RealGgufBytes_MoeIndexedMatmul_MatchesCpuReference()
        => RunParity(QuantizationType.IQ4_NL, breakReference: false);

    /// <summary>
    /// Negative control for <see cref="Iq4Nl_RealGgufBytes_MoeIndexedMatmul_MatchesCpuReference"/>:
    /// the reference is recomputed with the low/high nibble planes of every <c>qs</c>
    /// byte swapped (elements <c>[0,16)</c> and <c>[16,32)</c> exchanged), and the kernel
    /// must NOT match it.
    /// </summary>
    [SkippableFact]
    public void Iq4Nl_ParityAssertion_Discriminates_AgainstSwappedNibblePlanes()
        => RunParity(QuantizationType.IQ4_NL, breakReference: true);

    private static FixtureLocation ResolveFixture(QuantizationType qt)
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string ladder = Path.Combine(home, ".dotllm", "quant-ladder", "SmolLM2-135M");
        return qt == QuantizationType.Q5_0
            ? TestFixtureResolver.ResolveFile(
                ["DOTLLM_SMOLLM135M_Q5_0_GGUF", "DOTLLM_QUANT_FIXTURE_Q5_0"],
                "QuantFactory", "SmolLM-135M-GGUF",
                ["SmolLM2-135M-pure-Q5_0.gguf", "SmolLM-135M.Q5_0.gguf"],
                [ladder])
            : TestFixtureResolver.ResolveFile(
                ["DOTLLM_SMOLLM135M_IQ4_NL_GGUF", "DOTLLM_QUANT_FIXTURE_IQ4_NL"],
                "QuantFactory", "SmolLM-135M-GGUF",
                ["SmolLM2-135M-pure-IQ4_NL.gguf", "SmolLM-135M.IQ4_NL.gguf"],
                [ladder])
            ;
    }

    private static string SpvName(QuantizationType qt)
        => qt == QuantizationType.Q5_0
            ? "moe_indexed_matmul_q5_0_f32.spv"
            : "moe_indexed_matmul_iq4_nl_f32.spv";

    private static int BlockBytes(QuantizationType qt)
        => qt == QuantizationType.Q5_0
            ? MoeIndexedMatmulQ5_0F32Kernel.Q5_0BlockBytes
            : MoeIndexedMatmulIq4NlF32Kernel.Iq4NlBlockBytes;

    private unsafe void RunParity(QuantizationType qt, bool breakReference)
    {
        FixtureLocation fixture = ResolveFixture(qt);
        Skip.If(!fixture.Found, fixture.SkipMessage($"SmolLM2-135M {qt} GGUF"));

        SkipIfVulkanUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, SpvName(qt))),
            $"{SpvName(qt)} not compiled (glslc / Vulkan SDK required).");

        _output.WriteLine($"gguf: {fixture.Path}  type={qt}  breakReference={breakReference}");

        using var gguf = GgufFile.Open(fixture.Path!);

        // #344 rule 4: the fixture must actually contain the type. A filename is not
        // evidence — SmolLM-135M "i1-Q3_K_M" carries no Q3_K tensor at all.
        var candidates = gguf.Tensors
            .Where(t => t.QuantizationType == qt
                        && t.Shape.Rank == 2
                        && (t.Shape[0] % 32) == 0
                        && t.Shape[1] >= NumExperts * RowsPerExpert)
            .Take(MaxTensors)
            .ToList();

        if (candidates.Count == 0)
        {
            var typesPresent = gguf.Tensors
                .Select(t => t.QuantizationType)
                .Distinct()
                .OrderBy(t => t.ToString(), StringComparer.Ordinal)
                .ToList();
            Assert.Fail(
                $"Fixture at {fixture.Path} contains no usable 2D {qt} tensors. Types present: "
                + string.Join(", ", typesPresent));
        }

        int blockBytes = BlockBytes(qt);
        int maxK = candidates.Max(t => t.Shape[0]);
        long maxRowBytes = (long)(maxK / 32) * blockBytes;
        long maxBankBytes = ((long)NumExperts * RowsPerExpert * maxRowBytes + 3) & ~3L;
        long maxXBytes = (long)Rows * maxK * sizeof(float);
        long idxBytes = Rows * sizeof(int);
        long yBytes = (long)Rows * RowsPerExpert * sizeof(float);

        using var device = VulkanDevice.Create();
        using var q5Kernel = qt == QuantizationType.Q5_0
            ? MoeIndexedMatmulQ5_0F32Kernel.Create(device, spvDir) : null;
        using var iqKernel = qt == QuantizationType.IQ4_NL
            ? MoeIndexedMatmulIq4NlF32Kernel.Create(device, spvDir) : null;

        // Allocate once at the max size and reuse: DescriptorSetCache is keyed on raw
        // Vulkan buffer handles, which the driver recycles — per-iteration alloc/free
        // can silently bind a dead, smaller buffer.
        using var bufBank = device.Allocate(maxBankBytes);
        using var bufX = device.Allocate(maxXBytes);
        using var bufIdx = device.Allocate(idxBytes);
        using var bufY = device.Allocate(yBytes);

        var rng = new Random(407);

        // Every expert is hit, and one expert is hit twice, so a bank-stride bug
        // (expert offset dropped or scaled wrong) cannot pass by coincidence.
        int[] indices = new int[Rows];
        for (int r = 0; r < Rows; r++) indices[r] = r % NumExperts;
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(indices.AsSpan()), bufIdx);

        int checkedTensors = 0;
        double worstRelSeen = 0;

        foreach (var tensor in candidates)
        {
            int k = tensor.Shape[0];
            const int m = RowsPerExpert;
            long rowBytes = (long)(k / 32) * blockBytes;
            long bankBytes = (long)NumExperts * m * rowBytes;

            nint tensorPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

            // CPU oracle over exactly the bytes the bank covers.
            long bankElems = (long)NumExperts * m * k;
            float[] wDequant;
            if (breakReference)
            {
                wDequant = BreakDequant(tensorPtr, bankElems, qt);
            }
            else
            {
                wDequant = new float[bankElems];
                Dequantize.ToFloat32(tensorPtr, bankElems, qt, wDequant);
            }

            float[] x = new float[(long)Rows * k];
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            var wSpan = new ReadOnlySpan<byte>((void*)tensorPtr, checked((int)bankBytes));
            device.Upload(wSpan, bufBank);
            device.Upload(x.AsSpan(), bufX);

            if (q5Kernel is not null)
                q5Kernel.Launch(bufBank, bufX, bufIdx, bufY, m: m, k: k, n: Rows, numExperts: NumExperts);
            else
                iqKernel!.Launch(bufBank, bufX, bufIdx, bufY, m: m, k: k, n: Rows, numExperts: NumExperts);

            float[] actual = new float[Rows * m];
            device.Download(bufY, actual);

            var expected = new double[Rows * m];
            for (int r = 0; r < Rows; r++)
            {
                int e = indices[r];
                long xBase = (long)r * k;
                for (int row = 0; row < m; row++)
                {
                    long wBase = ((long)e * m + row) * k;
                    double acc = 0.0;
                    for (int col = 0; col < k; col++)
                        acc += (double)wDequant[wBase + col] * x[xBase + col];
                    expected[r * m + row] = acc;
                }
            }

            double worstRel = CompareRelative(expected, actual, out int badCells, out double absTol);
            worstRelSeen = Math.Max(worstRelSeen, worstRel);
            _output.WriteLine(
                $"[{tensor.Name}] k={k} m={m} experts={NumExperts} n={Rows}: "
                + $"absTol={absTol:G6} worstRel={worstRel:E3} badCells={badCells}/{expected.Length}");

            if (!breakReference)
            {
                Assert.True(badCells == 0,
                    $"[{tensor.Name}] {badCells}/{expected.Length} cells exceed tolerance "
                    + $"(absTol={absTol:G6}, RelTol={RelTol:G6}); worst rel={worstRel:E3}.");
            }

            checkedTensors++;
        }

        Assert.True(checkedTensors > 0);

        if (breakReference)
        {
            // The point of the negative control: the SAME assertion the positive test
            // makes must FAIL against the deliberately-broken reference. If this passes,
            // the positive test is not discriminating and proves nothing.
            Assert.True(worstRelSeen > RelTol,
                $"NEGATIVE CONTROL DID NOT DISCRIMINATE for {qt}: the deliberately broken "
                + $"reference still agreed with the kernel to within {worstRelSeen:E3} "
                + $"(tolerance {RelTol:G6}). The positive parity test is not proving the "
                + "packed layout is decoded correctly.");
            _output.WriteLine(
                $"negative control OK: broken {qt} reference diverges, worst rel={worstRelSeen:E3}.");
        }
    }

    /// <summary>
    /// Dequantizes with a deliberately wrong element mapping — the negative control's
    /// "broken implementation". Q5_0 reads the <c>qh</c> bit for element <c>j</c> from
    /// bit <c>2j (mod 32)</c>; IQ4_NL swaps the low/high nibble planes.
    /// </summary>
    private static unsafe float[] BreakDequant(nint src, long elementCount, QuantizationType qt)
    {
        var dest = new float[elementCount];
        long blocks = elementCount / 32;
        byte* p = (byte*)src;

        if (qt == QuantizationType.Q5_0)
        {
            for (long b = 0; b < blocks; b++)
            {
                byte* blk = p + b * QuantFormat.Q5_0BlockBytes;
                float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(blk);
                uint qh = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<uint>(blk + 2);
                byte* qs = blk + 6;
                for (int j = 0; j < 16; j++)
                {
                    int lo = qs[j] & 0xF;
                    int hi = (qs[j] >> 4) & 0xF;
                    int bLo = (int)((qh >> ((2 * j) & 31)) & 1);
                    int bHi = (int)((qh >> ((2 * (j + 16)) & 31)) & 1);
                    dest[b * 32 + j] = d * ((lo | (bLo << 4)) - 16);
                    dest[b * 32 + j + 16] = d * ((hi | (bHi << 4)) - 16);
                }
            }
            return dest;
        }

        ReadOnlySpan<float> kv =
        [
            -127f, -104f, -83f, -65f, -49f, -35f, -22f, -10f,
            1f, 13f, 25f, 38f, 53f, 69f, 89f, 113f,
        ];
        for (long b = 0; b < blocks; b++)
        {
            byte* blk = p + b * QuantFormat.IQ4_NLBlockBytes;
            float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(blk);
            byte* qs = blk + 2;
            for (int j = 0; j < 16; j++)
            {
                // Swapped planes: high nibble drives the first half, low the second.
                dest[b * 32 + j] = d * kv[(qs[j] >> 4) & 0xF];
                dest[b * 32 + j + 16] = d * kv[qs[j] & 0xF];
            }
        }
        return dest;
    }

    private static double CompareRelative(double[] expected, float[] actual, out int badCells, out double absTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        double ss = 0;
        for (int i = 0; i < expected.Length; i++) ss += expected[i] * expected[i];
        double rms = Math.Sqrt(ss / expected.Length);
        absTol = Math.Max(rms, 1e-6) * RelTol;

        badCells = 0;
        double worstRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(actual[i] - expected[i]);
            double rel = diff / Math.Max(Math.Abs(expected[i]), 1e-7);
            if (rel > worstRel) worstRel = rel;
            if (diff > absTol && rel > RelTol) badCells++;
        }
        return worstRel;
    }

    private static void SkipIfVulkanUnavailable(out string spvDir)
    {
        Skip.IfNot(IsVulkanRuntimeAvailable(),
            "Vulkan runtime not available on this host (vulkan-1.dll missing or no compatible device).");
        spvDir = ResolveSpvDir() ?? string.Empty;
        Skip.If(spvDir.Length == 0 || !Directory.Exists(spvDir),
            $"Vulkan SPV directory not found (resolved: {(spvDir.Length == 0 ? "null" : spvDir)}).");
    }

    private static bool IsVulkanRuntimeAvailable()
    {
        try
        {
            using var d = VulkanDevice.Create();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static string? ResolveSpvDir()
    {
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return null;
    }
}
