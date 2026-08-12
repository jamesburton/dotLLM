using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit.Abstractions;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Q3_K CPU↔Vulkan dequant parity on <b>real GGUF bytes</b>.
/// </summary>
/// <remarks>
/// <para>
/// Every other Q3_K kernel test decodes bytes produced by <c>Q3KFixture</c> — a packer
/// we wrote ourselves. That is how the transposed Q3_K layout (#311) shipped: the fixture
/// encoded with the same wrong layout the kernels decoded with, so the whole family passed
/// while real weights decoded to noise. Fixing the packer closes *that* loop, but it does
/// not close the general one: a fixture only ever exercises the byte patterns its own
/// quantiser happens to emit.
/// </para>
/// <para>
/// This test takes its input from a real llama.cpp-quantised GGUF instead, so the bytes come
/// from the authoritative quantiser rather than from us. It is the discriminating counterpart
/// the #311 post-mortem called for, and it is deliberately in the integration suite (it needs
/// a model fixture) rather than alongside the fixture-based kernel tests.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class RealGgufQ3KDequantParityTests
{
    private const int Q3KGroupSize = 256;
    private const int Q3KBlockBytes = 110;

    /// <summary>Cap per tensor so the test stays seconds, not minutes.</summary>
    private const int MaxBlocksPerTensor = 4096;

    private readonly ITestOutputHelper _output;

    public RealGgufQ3KDequantParityTests(ITestOutputHelper output) => _output = output;

    // NOTE: do not be tempted to point this at mradermacher's SmolLM-135M "i1-Q3_K_M" — despite
    // the name it carries no Q3_K tensor at all (IQ4_NL×120, Q5_0×58, Q4_K×29, …). A filename is
    // not evidence of a quantisation type; the Assert below is what actually establishes it.
    [SkippableFact]
    public void Bielik15B_Q3_K_RealGgufBytes_VulkanDequant_MatchesCpuOracle()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_BIELIK_15B_Q3_K_M_GGUF", "second-state", "Bielik-1.5B-v3.0-Instruct-GGUF",
            "Bielik-1.5B-v3.0-Instruct-Q3_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Bielik-1.5B Q3_K_M GGUF"));
        AssertRealGgufQ3KDequantParity(fixture.Path!);
    }

    /// <summary>
    /// The dequant kernel agreeing on real bytes does not certify the <i>matmul</i> kernels:
    /// #311 rewrote the packed Q3_K GEMV/GEMM/MMVQ/MMQ paths separately ("a sub-block is now
    /// 16 consecutive bytes at a fixed bit-pair, not one funnelled word"), and those decode the
    /// weight bytes themselves rather than consuming dequantised F32. This covers the GEMV
    /// (decode) path on real GGUF bytes.
    /// </summary>
    [SkippableFact]
    public void Bielik15B_Q3_K_RealGgufBytes_VulkanGemv_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_BIELIK_15B_Q3_K_M_GGUF", "second-state", "Bielik-1.5B-v3.0-Instruct-GGUF",
            "Bielik-1.5B-v3.0-Instruct-Q3_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Bielik-1.5B Q3_K_M GGUF"));

        string? spvDir = ResolveSpvDir();
        Skip.If(spvDir is null || !Directory.Exists(spvDir),
            $"Vulkan SPV directory not found (resolved: {spvDir ?? "null"}).");

        using GgufFile gguf = GgufFile.Open(fixture.Path!);
        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ3KGemvF32Kernel.Create(device, spvDir!);

        const int MaxRows = 512;

        // Size the buffers to the largest case up front and reuse them. Per-tensor
        // allocate/free would recycle Vulkan buffer handles into the kernel's
        // handle-keyed DescriptorSetCache and silently bind a dead, smaller buffer.
        int maxK = 0, maxWeightBytes = 0;
        foreach (GgufTensorDescriptor t in gguf.Tensors)
        {
            if (t.QuantizationType != QuantizationType.Q3_K || t.Shape.Rank != 2) continue;
            int tk = t.Shape[0];
            if (tk % Q3KGroupSize != 0) continue;
            maxK = Math.Max(maxK, tk);
            maxWeightBytes = Math.Max(
                maxWeightBytes, Math.Min(t.Shape[1], MaxRows) * (tk / Q3KGroupSize) * Q3KBlockBytes);
        }
        Assert.True(maxK > 0, "No 2-D Q3_K tensor found to exercise the GEMV path.");

        using var bufW = device.Allocate(((long)maxWeightBytes + 3) & ~3L);
        using var bufX = device.Allocate((long)maxK * sizeof(float));
        using var bufY = device.Allocate((long)MaxRows * sizeof(float));

        int tensorsChecked = 0;
        double worstRelErr = 0;
        string worstWhere = "(none)";

        foreach (GgufTensorDescriptor tensor in gguf.Tensors)
        {
            if (tensor.QuantizationType != QuantizationType.Q3_K) continue;
            if (tensor.Shape.Rank != 2) continue;

            int k = tensor.Shape[0];
            int m = Math.Min(tensor.Shape[1], MaxRows);
            if (k % Q3KGroupSize != 0 || m <= 0) continue;

            int blocksPerRow = k / Q3KGroupSize;
            int byteCount = m * blocksPerRow * Q3KBlockBytes;

            var raw = new byte[byteCount];
            unsafe
            {
                new ReadOnlySpan<byte>((void*)(gguf.DataBasePointer + (nint)tensor.DataOffset), byteCount)
                    .CopyTo(raw);
            }

            var rng = new Random(0x3CAFE0 ^ tensorsChecked);
            var x = new float[k];
            for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            // Reference: dequantise with the llama.cpp-anchored CPU oracle, then a plain
            // F32 dot per row. Independent of the packed GEMV weight decode under test.
            var w = new float[(long)m * k];
            unsafe
            {
                fixed (byte* rawPtr = raw) Dequantize.DequantizeQ3_K((nint)rawPtr, (long)m * k, w);
            }
            var expected = new float[m];
            for (int row = 0; row < m; row++)
            {
                double acc = 0;
                long baseIdx = (long)row * k;
                for (int i = 0; i < k; i++) acc += (double)w[baseIdx + i] * x[i];
                expected[row] = (float)acc;
            }

            device.Upload(new ReadOnlySpan<byte>(raw), bufW);
            device.Upload(x, bufX);
            kernel.Launch(bufW, bufX, bufY, m, k);
            var actual = new float[m];
            device.Download(bufY, actual);

            // Different accumulation order, so compare relative to the row magnitude rather
            // than bit-exactly. A layout disagreement is orders of magnitude larger than this.
            for (int row = 0; row < m; row++)
            {
                float scale = Math.Max(Math.Abs(expected[row]), 1e-3f);
                double relErr = Math.Abs(expected[row] - actual[row]) / scale;
                if (relErr > worstRelErr)
                {
                    worstRelErr = relErr;
                    worstWhere = $"{tensor.Name} row {row} (cpu={expected[row]:R} vulkan={actual[row]:R})";
                }
            }

            tensorsChecked++;
        }

        Assert.True(tensorsChecked > 0, "No 2-D Q3_K tensor found to exercise the GEMV path.");
        _output.WriteLine(
            $"[Q3_K real-bytes GEMV] tensors={tensorsChecked} worst relative error={worstRelErr:E3} at {worstWhere}");
        Assert.True(worstRelErr < 2e-2,
            $"Vulkan Q3_K GEMV disagrees with the CPU oracle on real GGUF bytes: "
            + $"worst relative error {worstRelErr:E3} at {worstWhere}. "
            + "The fixture-based GEMV test passes, so this would mean the packed GEMV weight "
            + "decode diverges from real llama.cpp-quantised bytes.");
    }

    /// <summary>
    /// F32-in prefill GEMM (<c>matmul_q3_k_f32_gemm</c>) on real GGUF bytes. Same weight
    /// decode as the GEMV but a different traversal (16×16 output tile, one thread per
    /// output cell, K-chunk = one 16-element sub-block), so a layout error can live in one
    /// and not the other.
    /// </summary>
    [SkippableTheory]
    [InlineData(6)]   // the shape RealGgufVulkanParityTests actually prefills
    [InlineData(68)]  // > one TILE_N, so the tile loop is exercised rather than masked away
    public void Bielik15B_Q3_K_RealGgufBytes_VulkanGemm_MatchesCpuReference(int n)
    {
        RunRealGgufMatMulParity(n, MatMulFlavour.GemmF32);
    }

    /// <summary>
    /// dp4a prefill GEMM (<c>matmul_q3_k_mmq</c>) on real GGUF bytes. This is the kernel the
    /// failing end-to-end Bielik parity test actually dispatches for its 6-token prefill
    /// (MMQ is enabled by default wherever the device advertises integer dot product).
    /// </summary>
    [SkippableTheory]
    [InlineData(6)]
    [InlineData(68)]
    public void Bielik15B_Q3_K_RealGgufBytes_VulkanMmq_MatchesCpuReference(int n)
    {
        RunRealGgufMatMulParity(n, MatMulFlavour.Mmq);
    }

    /// <summary>
    /// dp4a decode GEMV (<c>matmul_q3_k_mmvq</c>) on real GGUF bytes — the kernel that
    /// actually serves decode when integer dot product is available (the F32
    /// <c>matmul_q3_k_f32_gemv</c> already covered above is only the fallback).
    /// </summary>
    [SkippableFact]
    public void Bielik15B_Q3_K_RealGgufBytes_VulkanMmvq_MatchesCpuReference()
    {
        RunRealGgufMatMulParity(1, MatMulFlavour.Mmvq);
    }

    private enum MatMulFlavour { GemmF32, Mmq, Mmvq }

    /// <summary>Rows per tensor. Keeps the O(n·m·k) double-precision CPU reference bounded.</summary>
    private const int MatMulMaxRows = 64;

    /// <summary>Tensors per run. 129 × the reference cost would dominate the suite.</summary>
    private const int MaxTensors = 24;

    /// <summary>
    /// Int8-activation drift bound, matching the fixture-based MMQ/MMVQ tests
    /// (<c>VulkanMatMulQ3KMmqKernelTests.RelTol</c> = 3e-2,
    /// <c>VulkanMatMulQ3KMmvqKernelTests.RelTol</c> = 6e-2). A layout disagreement moves
    /// outputs by O(1) relative, orders of magnitude above this.
    /// </summary>
    private static float RelTolFor(MatMulFlavour flavour) => flavour switch
    {
        MatMulFlavour.GemmF32 => 2e-2f, // F32 activations: only reduction-order drift
        MatMulFlavour.Mmq     => 3e-2f,
        _                     => 6e-2f,
    };

    private void RunRealGgufMatMulParity(int n, MatMulFlavour flavour)
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_BIELIK_15B_Q3_K_M_GGUF", "second-state", "Bielik-1.5B-v3.0-Instruct-GGUF",
            "Bielik-1.5B-v3.0-Instruct-Q3_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Bielik-1.5B Q3_K_M GGUF"));

        string? spvDir = ResolveSpvDir();
        Skip.If(spvDir is null || !Directory.Exists(spvDir),
            $"Vulkan SPV directory not found (resolved: {spvDir ?? "null"}).");

        using GgufFile gguf = GgufFile.Open(fixture.Path!);
        using var device = VulkanDevice.Create();

        bool needsDp4a = flavour != MatMulFlavour.GemmF32;
        Skip.If(needsDp4a && !device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMQ/MMVQ unavailable.");

        // Enumerate the tensors first so every buffer can be sized to the largest case and
        // then REUSED. Allocating per tensor would recycle Vulkan buffer handles into each
        // kernel's handle-keyed DescriptorSetCache and silently bind a dead, smaller buffer.
        var tensors = new List<(string Name, int K, int M, long DataOffset)>();
        foreach (GgufTensorDescriptor t in gguf.Tensors)
        {
            if (t.QuantizationType != QuantizationType.Q3_K || t.Shape.Rank != 2) continue;
            int tk = t.Shape[0];
            if (tk % Q3KGroupSize != 0 || t.Shape[1] <= 0) continue;
            tensors.Add((t.Name, tk, Math.Min(t.Shape[1], MatMulMaxRows), (long)t.DataOffset));
            if (tensors.Count >= MaxTensors) break;
        }
        Assert.True(tensors.Count > 0,
            "No 2-D Q3_K tensor found to exercise the matmul path. Types present: "
            + string.Join(", ", gguf.Tensors.GroupBy(t => t.QuantizationType)
                .OrderByDescending(g => g.Count()).Select(g => $"{g.Key}×{g.Count()}")));

        int maxK = tensors.Max(t => t.K);
        int maxRows = tensors.Max(t => t.M);
        long maxWeightBytes = tensors.Max(t => (long)t.M * (t.K / Q3KGroupSize) * Q3KBlockBytes);

        using var bufW = device.Allocate((maxWeightBytes + 3) & ~3L);
        using var bufB = device.Allocate((long)n * maxK * sizeof(float));
        using var bufC = device.Allocate((long)n * maxRows * sizeof(float));

        MatMulQ3KGemmF32Kernel? gemm = null;
        MatMulQ3KMmqKernel? mmq = null;
        MatMulQ3KMmvqKernel? mmvq = null;
        QuantizeQ8_1RowsKernel? quantRows = null;
        QuantizeQ8_1Kernel? quantOne = null;
        VulkanDevice.Buffer? bufXq = null, bufXds = null;

        try
        {
            switch (flavour)
            {
                case MatMulFlavour.GemmF32:
                    gemm = MatMulQ3KGemmF32Kernel.Create(device, spvDir!);
                    break;
                case MatMulFlavour.Mmq:
                    quantRows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir!)
                        ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
                    mmq = MatMulQ3KMmqKernel.TryCreate(device, spvDir!)
                        ?? throw new Xunit.Sdk.XunitException("matmul_q3_k_mmq.spv missing.");
                    bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, maxK));
                    bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, maxK));
                    break;
                default:
                    quantOne = QuantizeQ8_1Kernel.TryCreate(device, spvDir!)
                        ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
                    mmvq = MatMulQ3KMmvqKernel.TryCreate(device, spvDir!)
                        ?? throw new Xunit.Sdk.XunitException("matmul_q3_k_mmvq.spv missing.");
                    bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(maxK));
                    bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(maxK));
                    break;
            }

            float relTol = RelTolFor(flavour);
            int tensorsChecked = 0, worstErrors = 0;
            double worstRel = 0;
            string worstWhere = "(none)";

            for (int ti = 0; ti < tensors.Count; ti++)
            {
                (string name, int k, int m, long dataOffset) = tensors[ti];
                int blocksPerRow = k / Q3KGroupSize;
                int byteCount = m * blocksPerRow * Q3KBlockBytes;

                var raw = new byte[byteCount];
                unsafe
                {
                    new ReadOnlySpan<byte>((void*)(gguf.DataBasePointer + (nint)dataOffset), byteCount)
                        .CopyTo(raw);
                }

                var rng = new Random(0x3CAFE1 ^ (ti * 7919) ^ (n * 31));
                var b = new float[(long)n * k];
                for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

                // Reference: dequantise with the llama.cpp-anchored CPU oracle, then a plain
                // F32 matmul. Independent of the packed weight decode under test.
                var w = new float[(long)m * k];
                unsafe
                {
                    fixed (byte* rawPtr = raw) Dequantize.DequantizeQ3_K((nint)rawPtr, (long)m * k, w);
                }
                var expected = new float[(long)n * m];
                for (int t = 0; t < n; t++)
                {
                    long bBase = (long)t * k;
                    for (int row = 0; row < m; row++)
                    {
                        double acc = 0;
                        long wBase = (long)row * k;
                        for (int i = 0; i < k; i++) acc += (double)w[wBase + i] * b[bBase + i];
                        expected[(long)t * m + row] = (float)acc;
                    }
                }

                device.Upload(new ReadOnlySpan<byte>(raw), bufW);
                device.Upload(b, bufB);

                var actual = new float[(long)n * m];
                if (flavour == MatMulFlavour.GemmF32)
                {
                    gemm!.Launch(bufW, bufB, bufC, m, k, n);
                }
                else
                {
                    using var ctx = device.CreateSubmitContext();
                    ctx.Begin();
                    if (flavour == MatMulFlavour.Mmq)
                    {
                        quantRows!.Record(ctx.CommandBuffer, bufB, bufXq!, bufXds!, n, k);
                        KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                        mmq!.Record(ctx.CommandBuffer, bufW, bufXq!, bufXds!, bufC, m, k, n);
                    }
                    else
                    {
                        quantOne!.Record(ctx.CommandBuffer, bufB, bufXq!, bufXds!, k);
                        KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                        mmvq!.Record(ctx.CommandBuffer, bufW, bufXq!, bufXds!, bufC, m, k);
                    }
                    ctx.SubmitAndWait();
                }
                device.Download(bufC, actual);

                // rms-relative bound, same shape as the fixture-based MMQ/MMVQ assertions.
                double ss = 0;
                for (int i = 0; i < expected.Length; i++) ss += (double)expected[i] * expected[i];
                float absTol = MathF.Max((float)Math.Sqrt(ss / expected.Length), 1e-6f) * relTol;

                for (int i = 0; i < expected.Length; i++)
                {
                    float e = expected[i], a = actual[i];
                    float diff = MathF.Abs(e - a);
                    float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
                    // Track the worst observed relative error unconditionally — a green run
                    // that reports "0" says nothing about how much headroom the bound has.
                    if (rel > worstRel && MathF.Abs(e) > absTol)
                    {
                        worstRel = rel;
                        worstWhere = $"{name} [n={n},m={m},k={k}] token {i / m} row {i % m} "
                            + $"(cpu={e:R} vulkan={a:R}, absTol={absTol:G6})";
                    }
                    if (diff > absTol && rel > relTol) worstErrors++;
                }

                tensorsChecked++;
            }

            _output.WriteLine(
                $"[Q3_K real-bytes {flavour} n={n}] tensors={tensorsChecked} "
                + $"out-of-tolerance cells={worstErrors} worst rel={worstRel:E3} at {worstWhere}");
            Assert.True(worstErrors == 0,
                $"Vulkan Q3_K {flavour} disagrees with the CPU oracle on real GGUF bytes: "
                + $"{worstErrors} out-of-tolerance cells, worst relative error {worstRel:E3} at {worstWhere}. "
                + "The fixture-based test for this kernel passes, so the packed weight decode "
                + "diverges on real llama.cpp-quantised bytes.");
        }
        finally
        {
            bufXq?.Dispose();
            bufXds?.Dispose();
            gemm?.Dispose();
            mmq?.Dispose();
            mmvq?.Dispose();
            quantRows?.Dispose();
            quantOne?.Dispose();
        }
    }

    private void AssertRealGgufQ3KDequantParity(string ggufPath)
    {
        string? spvDir = ResolveSpvDir();
        Skip.If(spvDir is null || !Directory.Exists(spvDir),
            $"Vulkan SPV directory not found (resolved: {spvDir ?? "null"}).");

        using GgufFile gguf = GgufFile.Open(ggufPath);

        var q3kTensors = gguf.Tensors
            .Where(t => t.QuantizationType == QuantizationType.Q3_K)
            .ToList();

        // A Q3_K_M file is a *mixed* quant, and some "Q3_K_M" builds (imatrix/i1 repacks in
        // particular) carry no Q3_K tensor at all. Without this check the test would report
        // green having compared nothing — the exact failure mode #307/#311 were about.
        Assert.True(q3kTensors.Count > 0,
            "No Q3_K tensors in this GGUF, so it cannot exercise the Q3_K path. Types present: "
            + string.Join(", ", gguf.Tensors.GroupBy(t => t.QuantizationType)
                .OrderByDescending(g => g.Count())
                .Select(g => $"{g.Key}×{g.Count()}")));
        _output.WriteLine($"[Q3_K real-bytes] {q3kTensors.Count} Q3_K tensors in {Path.GetFileName(ggufPath)}");

        using var device = VulkanDevice.Create();
        using var kernel = Q3KDequantF32Kernel.Create(device, spvDir!);

        // Allocate ONCE at the per-tensor cap and reuse. Allocating and freeing a buffer per
        // tensor is not safe here: the kernel's DescriptorSetCache is keyed on raw buffer
        // handles, and Vulkan recycles handles, so a freed buffer's handle can collide with a
        // later allocation and hand back a descriptor set still bound to the dead buffer.
        // (Observed exactly that: writes past the older, smaller buffer's extent dropped to
        // zero, which reads convincingly like a kernel truncation bug.)
        using var bufSrc = device.Allocate(((long)MaxBlocksPerTensor * Q3KBlockBytes + 3) & ~3L);
        using var bufDst = device.Allocate((long)MaxBlocksPerTensor * Q3KGroupSize * sizeof(float));

        int tensorsChecked = 0;
        long blocksChecked = 0;
        long negativeDBlocks = 0;
        double worstAbsDiff = 0;

        foreach (GgufTensorDescriptor tensor in q3kTensors)
        {
            long elementCount = tensor.Shape.ElementCount;
            if (elementCount % Q3KGroupSize != 0) continue;

            long totalBlocks = Math.Min(elementCount / Q3KGroupSize, MaxBlocksPerTensor);
            int elements = checked((int)(totalBlocks * Q3KGroupSize));
            int byteCount = checked((int)(totalBlocks * Q3KBlockBytes));

            nint srcPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

            var raw = new byte[byteCount];
            unsafe
            {
                new ReadOnlySpan<byte>((void*)srcPtr, byteCount).CopyTo(raw);
            }

            negativeDBlocks += CountNegativeBlockScales(raw, totalBlocks);

            // CPU oracle: DequantizeQ3_K is anchored to a literal transcription of
            // llama.cpp's dequantize_row_q3_K by
            // DequantizeKQuantTests.Q3_K_DenseRandomBlocks_MatchLlamaCppReference.
            var expected = new float[elements];
            unsafe
            {
                fixed (byte* rawPtr = raw)
                {
                    Dequantize.DequantizeQ3_K((nint)rawPtr, elements, expected);
                }
            }

            device.Upload(new ReadOnlySpan<byte>(raw), bufSrc);
            kernel.Launch(bufSrc, bufDst, checked((int)totalBlocks));

            var actual = new float[elements];
            device.Download(bufDst, actual);

            // Both paths read the same bytes and do one multiply per element — no
            // reduction, so this is exact. Any drift is a layout disagreement.
            long mismatches = 0;
            int firstMismatch = -1;
            long vulkanZeroWhereCpuNot = 0;
            for (int i = 0; i < elements; i++)
            {
                if (expected[i] == actual[i]) continue;
                if (firstMismatch < 0) firstMismatch = i;
                mismatches++;
                if (actual[i] == 0f && expected[i] != 0f) vulkanZeroWhereCpuNot++;
                double diff = Math.Abs((double)expected[i] - actual[i]);
                if (diff > worstAbsDiff) worstAbsDiff = diff;
            }

            if (mismatches > 0)
            {
                int b0 = firstMismatch / Q3KGroupSize;
                var sb = new System.Text.StringBuilder();
                sb.AppendLine(
                    $"Q3_K real-GGUF dequant mismatch in '{tensor.Name}' "
                    + $"({elements} elements = {totalBlocks} blocks checked):");
                sb.AppendLine(
                    $"  mismatching elements : {mismatches} ({100.0 * mismatches / elements:F2}%)");
                sb.AppendLine(
                    $"  first mismatch       : element {firstMismatch} (block {b0}, offset {firstMismatch % Q3KGroupSize})");
                sb.AppendLine(
                    $"  vulkan==0 & cpu!=0   : {vulkanZeroWhereCpuNot} of {mismatches}");
                sb.AppendLine($"  worst |diff|         : {worstAbsDiff:R}");
                sb.Append("  sample (cpu vs vulkan):");
                for (int i = firstMismatch; i < Math.Min(firstMismatch + 6, elements); i++)
                    sb.Append($" [{i}] {expected[i]:R}/{actual[i]:R}");
                sb.AppendLine();
                sb.Append(
                    "CPU is the llama.cpp-anchored oracle "
                    + "(DequantizeKQuantTests.Q3_K_DenseRandomBlocks_MatchLlamaCppReference), so the "
                    + "Vulkan Q3_K path disagrees with real GGUF bytes even though every "
                    + "fixture-based Q3_K kernel test passes.");
                Assert.Fail(sb.ToString());
            }

            tensorsChecked++;
            blocksChecked += totalBlocks;
        }

        Assert.True(tensorsChecked > 0, "No Q3_K tensor had a 256-aligned element count.");

        // Coverage note, not an assertion: a negative super-block scale is a pattern
        // llama.cpp's quantiser emits (d = max_scale / -32) but Q3KFixture never does
        // (its dF is always >= 0). Recording it makes the added coverage visible.
        _output.WriteLine(
            $"[Q3_K real-bytes] tensors={tensorsChecked} blocks={blocksChecked} "
            + $"negative-d blocks={negativeDBlocks} "
            + $"({(blocksChecked == 0 ? 0 : 100.0 * negativeDBlocks / blocksChecked):F1}%)");
    }

    /// <summary>
    /// Counts super-blocks whose fp16 <c>d</c> (at byte 108) is negative — a byte pattern
    /// the self-authored fixture never produces.
    /// </summary>
    private static long CountNegativeBlockScales(byte[] raw, long totalBlocks)
    {
        long negative = 0;
        for (long b = 0; b < totalBlocks; b++)
        {
            int off = checked((int)(b * Q3KBlockBytes)) + 108;
            ushort bits = (ushort)(raw[off] | (raw[off + 1] << 8));
            if ((bits & 0x8000) != 0 && (bits & 0x7FFF) != 0) negative++;
        }
        return negative;
    }

    private static string ResolveSpvDir()
    {
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
