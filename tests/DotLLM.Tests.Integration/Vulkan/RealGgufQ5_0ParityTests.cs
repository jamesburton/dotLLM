using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Real-bytes parity test for the Vulkan Q5_0 → FP32 dequant kernel
/// (<see cref="Q5_0DequantF32Kernel"/>) against the CPU oracle anchored in
/// <c>DequantizeQ5_0AnchorTests</c> (#344). Sister test to the Q3_K real-GGUF
/// parity gate in <see cref="RealGgufVulkanParityTests"/>: that test caught
/// #311 (Q3_K's bit layout was transposed) only because it ran against real
/// tensor bytes rather than a self-authored fixture, since the fixture
/// generator shared the same bug as the kernel. This test reads every Q5_0
/// tensor out of a real GGUF and dequantizes it both ways, comparing
/// element-for-element with 0 ULP tolerance (both paths do one multiply per
/// element, no reduction, so any difference is a real divergence).
/// </summary>
[Trait("Category", "GPU")]
// VulkanWeights.LastResidencyReport is a mutable static published at the end of
// VulkanWeights.Upload, so two classes that load a model and then read it MUST NOT
// run concurrently — xUnit would otherwise let one load overwrite the other's
// report, and the failure mode is a FALSE PASS (after #344 both this fixture and
// RealGgufVulkanParityTests' Q8_0 fixture report exactly 1 expanded tensor, so a
// swapped report still satisfies that class's Assert.Equal(1, ExpandedTensorCount)).
// Sharing a collection name is what serializes them; verified empirically, see
// task-7-report.md §Fix round 1.
[Collection("VulkanResidencyReport")]
public sealed class RealGgufQ5_0ParityTests
{
    private readonly ITestOutputHelper _output;

    public RealGgufQ5_0ParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Relative-error bound for the Q5_0 F32 matmul paths (GEMV and GEMM),
    /// checked against an RMS-of-expected-output denominator rather than a
    /// per-row L1 sum of |w*x| terms.
    ///
    /// The L1-sum form (the first version of the GEMV test below) dilutes
    /// sensitivity by roughly sqrt(K): with K up to 1536 and random +/-1 x,
    /// sumAbs ~ K*term while a genuinely cancelling dot product is only
    /// ~sqrt(K)*term, so dividing by sumAbs makes the bound ~sqrt(K) times
    /// looser than it looks. RMS(expected) — the RMS of the actual output
    /// values across the tensor — is itself ~sqrt(K)*term (the same
    /// cancellation scale the true dot product lives at), so normalising by
    /// it does not carry that dilution.
    ///
    /// Same numeric threshold (2e-2) and same dual
    /// "diff > absTol &amp;&amp; rel > RelTol" pass condition as the Q3_K
    /// real-bytes GEMM/MMQ/MMVQ precedent
    /// (RealGgufQ3KDequantParityTests.RunRealGgufMatMulParity, #320), which
    /// already validated this exact shape of bound against a real
    /// transposed-bit-layout bug (#311) on real hardware — reused rather
    /// than re-derived.
    /// </summary>
    private const double RelTol = 2e-2;

    /// <summary>
    /// Asserts every cell of <paramref name="actual"/> is within tolerance of
    /// <paramref name="expected"/>, using a global RMS(expected)-scaled
    /// absolute tolerance combined with a per-cell relative check — a cell
    /// only fails if BOTH the absolute and the relative bound are violated,
    /// so tiny outputs near zero (where the relative bound is impossible to
    /// satisfy honestly) are covered by the absolute floor instead.
    /// </summary>
    private void AssertMatMulWithinTolerance(string label, double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double ss = 0;
        for (int i = 0; i < expected.Length; i++) ss += expected[i] * expected[i];
        double rms = Math.Sqrt(ss / expected.Length);
        double absTol = Math.Max(rms, 1e-6) * RelTol;

        int badCells = 0;
        double worstRel = 0;
        double worstDiff = 0;
        int worstIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(actual[i] - expected[i]);
            double rel = diff / Math.Max(Math.Abs(expected[i]), 1e-7);
            if (rel > worstRel) { worstRel = rel; worstDiff = diff; worstIdx = i; }
            if (diff > absTol && rel > RelTol) badCells++;
        }

        _output.WriteLine(
            $"[{label}] cells={expected.Length} rms={rms:G6} absTol={absTol:G6} "
            + $"worstRel={worstRel:E3} worstDiff={worstDiff:E3} worstIdx={worstIdx} badCells={badCells}");
        Assert.True(badCells == 0,
            $"[{label}] {badCells}/{expected.Length} cells exceed tolerance "
            + $"(rms={rms:G6}, absTol={absTol:G6}, RelTol={RelTol:G6}); worst rel={worstRel:E3} "
            + $"diff={worstDiff:E3} at index {worstIdx} "
            + $"(cpu={(worstIdx >= 0 ? expected[worstIdx] : 0):G9} gpu={(worstIdx >= 0 && worstIdx < actual.Length ? actual[worstIdx] : 0):G9}).");
    }

    [SkippableFact]
    public unsafe void SmolLM135M_Q5_0_VulkanDequant_MatchesCpuOracle()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM135M_Q5_0_GGUF", "QuantFactory", "SmolLM-135M-GGUF",
            "SmolLM-135M.Q5_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q5_0 GGUF"));
        string path = fixture.Path!;

        SkipIfVulkanUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "q5_0_dequant_f32.spv")),
            "q5_0_dequant_f32.spv not compiled (glslc / Vulkan SDK required).");

        _output.WriteLine($"gguf: {path}");

        using var gguf = GgufFile.Open(path);
        var q5_0Tensors = gguf.Tensors
            .Where(t => t.QuantizationType == QuantizationType.Q5_0)
            .ToList();

        if (q5_0Tensors.Count == 0)
        {
            var typesPresent = gguf.Tensors
                .Select(t => t.QuantizationType)
                .Distinct()
                .OrderBy(t => t.ToString(), StringComparer.Ordinal)
                .ToList();
            Assert.Fail(
                $"Fixture at {path} contains no Q5_0 tensors. Types present: "
                + string.Join(", ", typesPresent));
            return;
        }

        // Allocate device buffers ONCE at the maximum tensor size and reuse
        // them across the loop below. DescriptorSetCache is keyed on raw
        // Vulkan buffer handles, which Vulkan recycles — per-iteration
        // alloc/free can bind a dead smaller buffer and silently read/write
        // zeros, masquerading as a kernel truncation bug.
        long maxElements = q5_0Tensors.Max(t => t.Shape.ElementCount);
        long maxBlocks = maxElements / Q5_0DequantF32Kernel.Q5_0GroupSize;
        long maxSrcBytes = maxBlocks * Q5_0DequantF32Kernel.Q5_0BlockBytes;
        long maxSrcBytesAligned = (maxSrcBytes + 3) & ~3L;
        long maxDstBytes = maxElements * sizeof(float);

        using var device = VulkanDevice.Create();
        using var kernel = Q5_0DequantF32Kernel.Create(device, spvDir);
        using var bufSrc = device.Allocate(maxSrcBytesAligned);
        using var bufDst = device.Allocate(maxDstBytes);

        long totalBlocksChecked = 0;
        int tensorsChecked = 0;

        foreach (var tensor in q5_0Tensors)
        {
            long elements = tensor.Shape.ElementCount;
            Assert.Equal(0, elements % Q5_0DequantF32Kernel.Q5_0GroupSize);
            int totalBlocks = checked((int)(elements / Q5_0DequantF32Kernel.Q5_0GroupSize));
            long srcBytes = (long)totalBlocks * Q5_0DequantF32Kernel.Q5_0BlockBytes;

            nint tensorPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

            float[] expected = new float[elements];
            Dequantize.ToFloat32(tensorPtr, elements, QuantizationType.Q5_0, expected);

            var srcSpan = new ReadOnlySpan<byte>((void*)tensorPtr, checked((int)srcBytes));
            device.Upload(srcSpan, bufSrc);
            kernel.Launch(bufSrc, bufDst, totalBlocks);

            float[] actual = new float[elements];
            device.Download(bufDst, actual);

            for (int i = 0; i < elements; i++)
            {
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                    Assert.Fail(
                        $"[{tensor.Name}] Q5_0 dequant mismatch at element {i} "
                        + $"(block {i / Q5_0DequantF32Kernel.Q5_0GroupSize}): "
                        + $"cpu={expected[i]:G9} gpu={actual[i]:G9}");
            }

            totalBlocksChecked += totalBlocks;
            tensorsChecked++;
            _output.WriteLine($"[{tensor.Name}] {totalBlocks} blocks, {elements} elements — match.");
        }

        _output.WriteLine($"Checked {tensorsChecked} Q5_0 tensors, {totalBlocksChecked} blocks total.");
        Assert.True(tensorsChecked > 0);
    }

    /// <summary>
    /// Real-bytes parity test for the Vulkan Q5_0 decode-path GEMV
    /// (<see cref="MatMulQ5_0GemvF32Kernel"/>) against a CPU reference computed
    /// from the trusted CPU dequant oracle (#344 Task 5). Mirrors the dequant
    /// parity test above but exercises the matmul kernel: for every 2D Q5_0
    /// weight tensor in a real GGUF, dequantize the row with the CPU oracle,
    /// dot it against a random x in double precision, and compare against the
    /// Vulkan GEMV output relative to the row's magnitude. Sister test to the
    /// Q3_K real-GGUF GEMV parity gate that caught #311 (Q3_K's bit layout was
    /// transposed) only because it ran against real tensor bytes rather than a
    /// self-authored fixture, since the fixture generator shared the same bug
    /// as the kernel — real bytes are the point, not a synthetic block.
    /// </summary>
    [SkippableFact]
    public unsafe void SmolLM135M_Q5_0_RealGgufBytes_VulkanGemv_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM135M_Q5_0_GGUF", "QuantFactory", "SmolLM-135M-GGUF",
            "SmolLM-135M.Q5_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q5_0 GGUF"));
        string path = fixture.Path!;

        SkipIfVulkanUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "matmul_q5_0_f32_gemv.spv")),
            "matmul_q5_0_f32_gemv.spv not compiled (glslc / Vulkan SDK required).");

        _output.WriteLine($"gguf: {path}");

        using var gguf = GgufFile.Open(path);
        var q5_0MatmulTensors = gguf.Tensors
            .Where(t => t.QuantizationType == QuantizationType.Q5_0
                     && t.Shape.Rank == 2
                     && t.Shape[0] % MatMulQ5_0GemvF32Kernel.Q5_0GroupSize == 0)
            .ToList();

        if (q5_0MatmulTensors.Count == 0)
        {
            var typesPresent = gguf.Tensors
                .Select(t => t.QuantizationType)
                .Distinct()
                .OrderBy(t => t.ToString(), StringComparer.Ordinal)
                .ToList();
            Assert.Fail(
                $"Fixture at {path} contains no 2D Q5_0 matmul tensors. Types present: "
                + string.Join(", ", typesPresent));
            return;
        }

        // Allocate device buffers ONCE at the maximum tensor size and reuse
        // them across the loop below — see the note on the dequant test above
        // for why per-iteration alloc/free is unsafe with DescriptorSetCache.
        int maxK = q5_0MatmulTensors.Max(t => t.Shape[0]);
        int maxM = q5_0MatmulTensors.Max(t => t.Shape[1]);
        long maxRowBytes = (long)(maxK / MatMulQ5_0GemvF32Kernel.Q5_0GroupSize) * MatMulQ5_0GemvF32Kernel.Q5_0BlockBytes;
        long maxWeightBytes = maxM * maxRowBytes;
        long maxWeightBytesAligned = (maxWeightBytes + 3) & ~3L;
        long maxXBytes = ((long)maxK * sizeof(float) + 3) & ~3L;
        long maxYBytes = ((long)maxM * sizeof(float) + 3) & ~3L;

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ5_0GemvF32Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(maxWeightBytesAligned);
        using var bufX = device.Allocate(maxXBytes);
        using var bufY = device.Allocate(maxYBytes);

        var rng = new Random(344);
        int tensorsChecked = 0;
        long rowsChecked = 0;

        foreach (var tensor in q5_0MatmulTensors)
        {
            int k = tensor.Shape[0];
            int m = tensor.Shape[1];
            long elements = (long)k * m;

            nint tensorPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

            float[] wDequant = new float[elements];
            Dequantize.ToFloat32(tensorPtr, elements, QuantizationType.Q5_0, wDequant);

            float[] x = new float[k];
            for (int i = 0; i < k; i++)
                x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            long rowBytes = (long)(k / MatMulQ5_0GemvF32Kernel.Q5_0GroupSize) * MatMulQ5_0GemvF32Kernel.Q5_0BlockBytes;
            long weightBytes = (long)m * rowBytes;
            var wSpan = new ReadOnlySpan<byte>((void*)tensorPtr, checked((int)weightBytes));
            device.Upload(wSpan, bufW);
            device.Upload(x.AsSpan(), bufX);

            kernel.Launch(bufW, bufX, bufY, m, k);

            float[] actual = new float[m];
            device.Download(bufY, actual);

            var expected = new double[m];
            for (int row = 0; row < m; row++)
            {
                double refDot = 0.0;
                long rowBase = (long)row * k;
                for (int col = 0; col < k; col++)
                    refDot += (double)wDequant[rowBase + col] * x[col];
                expected[row] = refDot;
            }

            AssertMatMulWithinTolerance($"{tensor.Name} gemv m={m} k={k}", expected, actual);

            rowsChecked += m;
            tensorsChecked++;
            _output.WriteLine($"[{tensor.Name}] m={m} k={k} — {m} rows match within RMS-relative tolerance.");
        }

        _output.WriteLine($"Checked {tensorsChecked} Q5_0 matmul tensors, {rowsChecked} rows total.");
        Assert.True(tensorsChecked > 0);
    }

    /// <summary>Rows per tensor for the GEMM test. Keeps the O(n·m·k) double-precision CPU
    /// reference bounded — mirrors <c>RealGgufQ3KDequantParityTests.MatMulMaxRows</c> (#320).</summary>
    private const int GemmMatMulMaxRows = 64;

    /// <summary>Tensors checked per GEMM run — mirrors <c>RealGgufQ3KDequantParityTests.MaxTensors</c> (#320).</summary>
    private const int GemmMaxTensors = 16;

    /// <summary>
    /// Real-bytes parity test for the Vulkan Q5_0 prefill-path GEMM
    /// (<see cref="MatMulQ5_0GemmF32Kernel"/>) against a CPU reference computed
    /// from the trusted CPU dequant oracle (#344 Task 6). Same shape as the
    /// Q3_K real-GGUF GEMM precedent (#320): <c>n = 6</c> is the shape
    /// <c>RealGgufVulkanParityTests</c> actually prefills (a partial tile —
    /// TILE_N = 16), <c>n = 68</c> exceeds one output tile so the tile loop is
    /// exercised rather than being entirely masked away.
    /// </summary>
    [SkippableTheory]
    [InlineData(6)]
    [InlineData(68)]
    public unsafe void SmolLM135M_Q5_0_RealGgufBytes_VulkanGemm_MatchesCpuReference(int n)
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM135M_Q5_0_GGUF", "QuantFactory", "SmolLM-135M-GGUF",
            "SmolLM-135M.Q5_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q5_0 GGUF"));
        string path = fixture.Path!;

        SkipIfVulkanUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "matmul_q5_0_f32_gemm.spv")),
            "matmul_q5_0_f32_gemm.spv not compiled (glslc / Vulkan SDK required).");

        _output.WriteLine($"gguf: {path}, n={n}");

        using var gguf = GgufFile.Open(path);

        // Cap rows per tensor and the number of tensors checked so the
        // O(n*m*k) double-precision CPU reference stays bounded — SmolLM's
        // lm_head/embedding tensors can have M in the tens of thousands.
        var tensors = new List<(string Name, int K, int M, long DataOffset)>();
        foreach (var t in gguf.Tensors)
        {
            if (t.QuantizationType != QuantizationType.Q5_0 || t.Shape.Rank != 2) continue;
            int tk = t.Shape[0];
            if (tk % MatMulQ5_0GemmF32Kernel.Q5_0GroupSize != 0 || t.Shape[1] <= 0) continue;
            tensors.Add((t.Name, tk, Math.Min(t.Shape[1], GemmMatMulMaxRows), (long)t.DataOffset));
            if (tensors.Count >= GemmMaxTensors) break;
        }

        if (tensors.Count == 0)
        {
            var typesPresent = gguf.Tensors
                .Select(t => t.QuantizationType)
                .Distinct()
                .OrderBy(t => t.ToString(), StringComparer.Ordinal)
                .ToList();
            Assert.Fail(
                $"Fixture at {path} contains no 2D Q5_0 matmul tensors. Types present: "
                + string.Join(", ", typesPresent));
            return;
        }

        // Allocate device buffers ONCE at the maximum size and reuse them
        // across the loop below — see the note on the dequant/GEMV tests
        // above for why per-iteration alloc/free is unsafe with
        // DescriptorSetCache.
        int maxK = tensors.Max(t => t.K);
        int maxM = tensors.Max(t => t.M);
        long maxRowBytes = (long)(maxK / MatMulQ5_0GemmF32Kernel.Q5_0GroupSize) * MatMulQ5_0GemmF32Kernel.Q5_0BlockBytes;
        long maxWeightBytes = (long)maxM * maxRowBytes;
        long maxWeightBytesAligned = (maxWeightBytes + 3) & ~3L;
        long maxBBytes = (long)n * maxK * sizeof(float);
        long maxCBytes = (long)n * maxM * sizeof(float);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ5_0GemmF32Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(maxWeightBytesAligned);
        using var bufB = device.Allocate(maxBBytes);
        using var bufC = device.Allocate(maxCBytes);

        var rng = new Random(0x344 ^ (n * 31));
        int tensorsChecked = 0;

        foreach (var (name, k, m, dataOffset) in tensors)
        {
            long elements = (long)k * m;
            nint tensorPtr = gguf.DataBasePointer + (nint)dataOffset;

            float[] wDequant = new float[elements];
            Dequantize.ToFloat32(tensorPtr, elements, QuantizationType.Q5_0, wDequant);

            var b = new float[(long)n * k];
            for (int i = 0; i < b.Length; i++)
                b[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            // Reference: dequantise with the CPU oracle (independent of the
            // packed weight decode under test), then a plain double-precision
            // F32 matmul.
            var expected = new double[(long)n * m];
            for (int t = 0; t < n; t++)
            {
                long bBase = (long)t * k;
                for (int row = 0; row < m; row++)
                {
                    double acc = 0.0;
                    long wBase = (long)row * k;
                    for (int col = 0; col < k; col++)
                        acc += (double)wDequant[wBase + col] * b[bBase + col];
                    expected[(long)t * m + row] = acc;
                }
            }

            long rowBytes = (long)(k / MatMulQ5_0GemmF32Kernel.Q5_0GroupSize) * MatMulQ5_0GemmF32Kernel.Q5_0BlockBytes;
            long weightBytes = (long)m * rowBytes;
            var wSpan = new ReadOnlySpan<byte>((void*)tensorPtr, checked((int)weightBytes));
            device.Upload(wSpan, bufW);
            device.Upload(b.AsSpan(), bufB);

            kernel.Launch(bufW, bufB, bufC, m, k, n);

            var actual = new float[(long)n * m];
            device.Download(bufC, actual);

            AssertMatMulWithinTolerance($"{name} gemm n={n} m={m} k={k}", expected, actual);

            tensorsChecked++;
            _output.WriteLine($"[{name}] n={n} m={m} k={k} — {n * m} cells match within RMS-relative tolerance.");
        }

        _output.WriteLine($"Checked {tensorsChecked} Q5_0 GEMM tensors at n={n}.");
        Assert.True(tensorsChecked > 0);
    }

    /// <summary>
    /// Loads a real Q5_0 GGUF through the full Vulkan upload path and asserts that
    /// NO tensor whose source format is Q5_0 was widened to F32 on upload (#344).
    /// <para>
    /// This is the reachability gate for the Task 4/5/6 Q5_0 kernels: they can be
    /// bit-exact against real GGUF bytes and still never run, because
    /// <c>VulkanWeights.DeviceQuantTypeFor</c> ends in an unconditional
    /// <c>return QuantizationType.F32</c>. Only the load-time residency report can
    /// tell "a kernel exists" apart from "a kernel is routed to".
    /// </para>
    /// <para>
    /// <b>Scoped to <c>Source == Q5_0</c> on purpose.</b> Two other expansions are
    /// legitimate and out of scope here: (a) <c>token_embd.weight</c>, which
    /// <c>VulkanWeights.UploadTokenEmbedding</c> uploads with
    /// <c>dequantToFp32: true</c> for every source type except Q4_K/Q5_K/Q6_K
    /// (there is no GPU gather+dequant kernel for the rest) — an orthogonal design
    /// choice, not a missing matmul kernel; and (b) any non-Q5_0 tensor the file
    /// happens to carry (llama.cpp's Q5_0 recipe commonly emits a Q6_K or Q8_0
    /// <c>output.weight</c>), whose residency belongs to its own unit. Pinning
    /// <c>ExpandedTensorCount</c> here would couple this test to both.
    /// </para>
    /// </summary>
    [SkippableFact]
    public void SmolLM135M_Q5_0_KeepsWeightsPacked_NoF32Expansion()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM135M_Q5_0_GGUF", "QuantFactory", "SmolLM-135M-GGUF",
            "SmolLM-135M.Q5_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q5_0 GGUF"));
        SkipIfVulkanUnavailable(out string spvDir);

        _output.WriteLine($"gguf: {fixture.Path!}");

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var vkModel = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);

        // Static side-channel — read it immediately after the load returns.
        VulkanResidencyReport? report = VulkanWeights.LastResidencyReport;
        Assert.NotNull(report);
        _output.WriteLine(report!.Describe());

        foreach (var group in report.Entries.GroupBy(e => (e.Source, e.Device)))
        {
            _output.WriteLine(
                $"  {group.Key.Source} -> {group.Key.Device}: {group.Count()} tensor(s), "
                + $"{group.Sum(e => e.PackedBytes):N0} packed -> {group.Sum(e => e.UploadedBytes):N0} uploaded.");
        }

        var q5Expanded = report.Entries
            .Where(e => e.Source == QuantizationType.Q5_0 && e.Expanded
                        && e.Name != "token_embd.weight")
            .ToArray();

        Assert.True(q5Expanded.Length == 0,
            $"Q5_0 tensors still widened on upload ({q5Expanded.Length} of "
            + $"{report.Entries.Count(e => e.Source == QuantizationType.Q5_0)} Q5_0 tensors):\n"
            + string.Join("\n", q5Expanded.Take(10).Select(
                e => $"  {e.Name}: {e.Source}->{e.Device} {e.PackedBytes:N0}->{e.UploadedBytes:N0} bytes"))
            + $"\n{report.Describe()}");

        // A model with no Q5_0 tensors at all would satisfy the assertion above
        // vacuously — pin that the fixture actually exercised the path.
        int q5Resident = report.Entries.Count(
            e => e.Source == QuantizationType.Q5_0 && e.Device == QuantizationType.Q5_0);
        Assert.True(q5Resident > 0,
            $"No Q5_0 tensor was kept device-resident — the fixture may not be Q5_0.\n{report.Describe()}");
        _output.WriteLine($"Q5_0 tensors kept device-resident: {q5Resident}.");
    }

    /// <summary>
    /// End-to-end CPU-vs-Vulkan argmax parity on the same Q5_0 fixture (#344).
    /// <para>
    /// The residency test above proves the packed bytes reach the device; this one
    /// proves the CONSUMERS can read them. Prefill (seqLen&gt;1, step 0) exercises
    /// <see cref="MatMulQ5_0GemmF32Kernel"/> and every decode step exercises
    /// <see cref="MatMulQ5_0GemvF32Kernel"/>. Before the residency wiring it passes
    /// via the F32-widened path; after it, a dispatch gap (packed Q5_0 bytes
    /// reinterpreted as floats by the generic F32 matmul) turns the logits into
    /// noise and this test goes red — which is the whole point.
    /// </para>
    /// Tolerances mirror the Q8_0 SmolLM-135M parity precedent in
    /// <c>RealGgufVulkanParityTests</c>: the CPU path runs Q-format arithmetic
    /// against Q8_1-quantised activations while Vulkan runs F32 activations
    /// against packed weights, so per-projection arithmetic differs slightly and
    /// the argmax floor is the load-bearing assertion.
    /// </summary>
    [SkippableFact]
    public void SmolLM135M_Q5_0_VulkanForward_MatchesCpuArgmax()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM135M_Q5_0_GGUF", "QuantFactory", "SmolLM-135M-GGUF",
            "SmolLM-135M.Q5_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q5_0 GGUF"));
        SkipIfVulkanUnavailable(out string spvDir);

        const int decodeSteps = 4;
        const int argmaxFloor = 3; // out of prefill + 4 decode = 5 steps

        using var cpuGguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(cpuGguf.Metadata);
        using var cpuModel = TransformerModel.LoadFromGguf(cpuGguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);

        using var vkGguf = GgufFile.Open(fixture.Path!);
        using var vkModel = VulkanTransformerModel.LoadFromGguf(vkGguf, config, spvDir);

        int vocab = config.VocabSize;
        var tokens = new List<int>(tokenizer.Encode("The capital of France is").ToArray());
        Assert.NotEmpty(tokens);

        int matches = 0;
        for (int step = 0; step <= decodeSteps; step++)
        {
            int[] tokenIds = tokens.ToArray();
            int[] positions = new int[tokenIds.Length];
            for (int i = 0; i < positions.Length; i++) positions[i] = i;

            float[] cpuLogits = LastRowLogits(cpuModel, tokenIds, positions, vocab);
            float[] vkLogits = LastRowLogits(vkModel, tokenIds, positions, vocab);

            int cpuArgmax = ArgmaxOf(cpuLogits);
            int vkArgmax = ArgmaxOf(vkLogits);
            if (cpuArgmax == vkArgmax) matches++;

            double corr = Correlation(cpuLogits, vkLogits);
            _output.WriteLine(
                $"step {step} (seqLen={tokenIds.Length}): cpu_argmax={cpuArgmax} vk_argmax={vkArgmax} "
                + $"corr={corr:F5}{(cpuArgmax == vkArgmax ? " [match]" : " [diff]")}");
            Assert.True(corr > 0.99,
                $"step {step}: logit correlation {corr:F5} — Vulkan Q5_0 path is not "
                + "reading the same weights as the CPU reference.");

            tokens.Add(cpuArgmax);
        }

        Assert.True(matches >= argmaxFloor,
            $"strict argmax matches {matches}/{decodeSteps + 1} below floor {argmaxFloor}.");
        _output.WriteLine($"strict argmax matches: {matches}/{decodeSteps + 1}");

        // The growing-context reprefill loop above is seqLen > 1 at every step, so
        // it only ever exercises MatMulQ5_0GemmF32Kernel. Run one explicit
        // single-token forward to route the seqLen == 1 branch through
        // MatMulQ5_0GemvF32Kernel — otherwise the decode kernel would be as
        // unreachable-but-untested after this change as it was before it.
        int[] one = [tokens[0]];
        int[] onePos = [0];
        float[] cpuOne = LastRowLogits(cpuModel, one, onePos, vocab);
        float[] vkOne = LastRowLogits(vkModel, one, onePos, vocab);
        double oneCorr = Correlation(cpuOne, vkOne);
        int cpuOneArg = ArgmaxOf(cpuOne);
        int vkOneArg = ArgmaxOf(vkOne);
        _output.WriteLine(
            $"decode shape (seqLen=1): cpu_argmax={cpuOneArg} vk_argmax={vkOneArg} corr={oneCorr:F5}");
        Assert.True(oneCorr > 0.99,
            $"seqLen=1 (GEMV) logit correlation {oneCorr:F5} — the Q5_0 decode kernel "
            + "is not reading the packed weights correctly.");
        Assert.Equal(cpuOneArg, vkOneArg);
    }

    private static unsafe float[] LastRowLogits(IModel model, int[] tokenIds, int[] positions, int vocab)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(2, logits.Shape.Rank);
        int seqLen = logits.Shape[0];
        Assert.Equal(vocab, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private static int ArgmaxOf(float[] xs)
    {
        int best = 0;
        float bestV = xs[0];
        for (int i = 1; i < xs.Length; i++) if (xs[i] > bestV) { bestV = xs[i]; best = i; }
        return best;
    }

    /// <summary>
    /// Pearson correlation between two logit vectors. A dispatch gap that feeds
    /// packed quant bytes to an F32 kernel produces near-zero correlation (the
    /// Q3_K #311 diagnosis measured 0.006), which no per-element tolerance on
    /// wildly-scaled logits would reliably catch.
    /// </summary>
    private static double Correlation(float[] a, float[] b)
    {
        int n = a.Length;
        double ma = 0, mb = 0;
        for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;
        double sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < n; i++)
        {
            double da = a[i] - ma, db = b[i] - mb;
            sab += da * db; saa += da * da; sbb += db * db;
        }
        return sab / Math.Sqrt(Math.Max(saa * sbb, 1e-30));
    }

    private static void SkipIfVulkanUnavailable(out string spvDir)
    {
        Skip.IfNot(IsVulkanRuntimeAvailable(),
            "Vulkan runtime not available on this host (vulkan-1.dll missing or no compatible device).");
        spvDir = ResolveSpvDir();
        Skip.If(spvDir is null || !Directory.Exists(spvDir),
            $"Vulkan SPV directory not found (resolved: {spvDir ?? "null"}).");
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

    private static string ResolveSpvDir()
    {
        // The repo ships SPV blobs at native/vulkan/spv/ relative to the
        // repo root. Tests run from bin/Debug/net10.0/, so walk up to the
        // repo root.
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return null!;
    }
}
