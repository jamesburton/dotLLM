using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity for the dp4a IQ2_XXS MMQ prefill GEMM (issue #344):
/// <c>QuantizeQ8_1RowsKernel</c> then <c>MatMulIq2XxsMmqKernel</c> (codebook-grid weights
/// sign·grid → int8 → dp4a, no min term), vs the IQ2_XXS-byte-identical CPU oracle.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq2XxsMmqKernelTests
{
    private const int GroupSize = 256;
    private const int BlockBytes = 66;
    private const float RelTol = 6e-2f;

    private readonly ITestOutputHelper _out;

    public VulkanMatMulIq2XxsMmqKernelTests(ITestOutputHelper output) => _out = output;

    // Small/medium shapes only. The kernel is bit-correct at every shape, but on the
    // gfx1151 iGPU the AMD driver miscompiles this grid-codebook shader into a GPU fault
    // once a single submit's total work is large (per-submit-cumulative — neither
    // robustBufferAccess nor per-dispatch chunking removes it). Production-scale shapes
    // (e.g. 8×2048×2048) crash the device, so the kernel is opt-in only
    // (DOTLLM_VULKAN_ENABLE_IQ2XXS_MMQ=1) and IQ2_XXS prefill defaults to the F32 GEMM.
    // See VulkanTransformerModel.IsIq2XxsMmqEnabled (issue #344).
    [SkippableTheory]
    [InlineData(2, 4, 256)]
    [InlineData(1, 1, 256)]
    [InlineData(17, 33, 512)]
    [InlineData(7, 4, 768)]
    public void Mmq_MatchesF32Oracle_ArgmaxAndTolerance(int n, int m, int k) => RunShape(n, m, k);

    /// <summary>
    /// Opt-in cross-check at production scale (8×2048×2048) — the shape that GPU-faults on the
    /// AMD gfx1151 iGPU (driver LLPC miscompile, #344). NEVER run by default (it would device-lost the
    /// AMD box). Set <c>DOTLLM_IQ2XXS_LARGE_CROSSCHECK=1</c> on a NON-AMD GPU to confirm the fault is
    /// AMD-specific: e.g. on the Framework box run with <c>DOTLLM_VULKAN_DEVICE_INDEX=0</c> (RTX 3060)
    /// and <c>=1</c> (Intel Arc). A pass on both pins the miscompile to AMD's shader compiler and is a
    /// strong artifact for the driver bug report (see <c>[[vulkan-iq2xxs-mmq-driver-fault]]</c>).
    /// </summary>
    [SkippableFact]
    public void Mmq_LargeShape_NonAmdCrossCheck()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_IQ2XXS_LARGE_CROSSCHECK") == "1",
            "Opt-in (DOTLLM_IQ2XXS_LARGE_CROSSCHECK=1) — 8×2048×2048 faults AMD gfx1151 by design; "
            + "run only on a non-AMD GPU to confirm the fault is AMD-specific.");
        RunShape(8, 2048, 2048);
    }

    private void RunShape(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        _out.WriteLine($"device: {device.DeviceName} (type={device.DeviceType}, vendor=0x{device.VendorId:X4}); shape n={n} m={m} k={k}");
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var codebooks = Iq2Codebooks.Create(device);
        using var mmq = MatMulIq2XxsMmqKernel.TryCreate(device, spvDir, codebooks)
            ?? throw new Xunit.Sdk.XunitException("matmul_iq2_xxs_mmq.spv missing or unsupported.");

        var rng = new Random(0x22 + n * 13 + m * 7 + k * 3);
        float[] weightsF32 = Iq2Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Iq2Fixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] wq = Iq2Fixture.QuantizeRowsIq2Xxs(weightsF32, m, k);
        Assert.Equal(m * (k / GroupSize) * BlockBytes, wq.Length);
        float[] expected = Iq2Fixture.CpuGemmIq2Xxs(wq, inputB, m, k, n);

        long weightsBufBytes = ((long)wq.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(wq), bufW);
        device.Upload(inputB, bufB);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n, k);
            DotLLM.Vulkan.Kernels.KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m, k, n);
            ctx.SubmitAndWait();
        }

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        AssertParity(expected, actual, m, k, n);
    }

    private static void AssertParity(float[] expected, float[] actual, int m, int k, int n)
    {
        Assert.Equal(expected.Length, actual.Length);

        double ss = 0;
        for (int i = 0; i < expected.Length; i++) ss += (double)expected[i] * expected[i];
        float rms = (float)Math.Sqrt(ss / Math.Max(1, expected.Length));
        float absTol = MathF.Max(rms, 1e-6f) * RelTol;

        for (int t = 0; t < n; t++)
        {
            int argE = 0, argA = 0;
            for (int i = 1; i < m; i++)
            {
                if (expected[t * m + i] > expected[t * m + argE]) argE = i;
                if (actual[t * m + i] > actual[t * m + argA]) argA = i;
            }
            float oracleMax = expected[t * m + argE];
            float oracleAtMmqArg = expected[t * m + argA];
            Assert.True(argE == argA || (oracleMax - oracleAtMmqArg) <= absTol,
                $"Argmax mismatch (n={n},m={m},k={k}) row {t}: oracle={argE} " +
                $"({oracleMax:G6}), mmq={argA} (oracle@{argA}={oracleAtMmqArg:G6}, " +
                $"gap={oracleMax - oracleAtMmqArg:G6} > absTol={absTol:G6}).");
        }

        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > absTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"MMQ drift exceeded tolerance (n={n},m={m},k={k}): errors={errors}/{expected.Length}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");
    }
}
