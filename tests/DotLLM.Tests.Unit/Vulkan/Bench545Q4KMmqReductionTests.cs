using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using System.Diagnostics;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Cost side of issue #545 — what the two-level accumulation in
/// <c>matmul_q4_k_mmq.comp</c> costs the prefill GEMM.
/// </summary>
/// <remarks>
/// <para>
/// The accuracy side lives in <see cref="Probe538Q8SplitTests"/>. This measures
/// the shipping kernel against <c>matmul_q4_k_mmq_pre545</c> — a retained copy of
/// the shader as it stood before the change, which compiles byte-identical to the
/// SPIR-V on <c>dev</c> — so the comparison is against what users actually had.
/// </para>
/// <para>
/// Both modules are held open in ONE process and the arms are run
/// <b>order-reversed within each pass</b>. Process-level A/B on this box has
/// produced 2-3x phantom deltas purely from GPU clock ramp, and UMA memory
/// contention swings absolute figures by ~40%, so only a same-session,
/// order-reversed ratio is trustworthy. Medians and min-max are reported rather
/// than single numbers.
/// </para>
/// <para>Opt-in: <c>DOTLLM_545_BENCH=1</c>. It is a benchmark, not a gate.</para>
/// <para>
/// Skips when <c>matmul_q4_k_mmq_pre545.spv</c> is absent, which is the normal state once the
/// issue has landed — the baseline is retired rather than carried forever. To
/// re-derive the numbers, restore it from history and recompile:
/// <code>
/// git show &lt;pre-545-rev&gt;:native/vulkan/shaders/matmul_q4_k_mmq.comp \n///   &gt; native/vulkan/shaders/matmul_q4_k_mmq_pre545.comp
/// glslc --target-env=vulkan1.2 -o native/vulkan/spv/matmul_q4_k_mmq_pre545.spv \n///   native/vulkan/shaders/matmul_q4_k_mmq_pre545.comp
/// </code>
/// Verify the result is byte-identical to the shipped spv at that revision
/// before trusting it as a baseline.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Bench545Q4KMmqReductionTests
{
    private const int Passes = 9;
    private const int Batch = 4;
    private const int WarmupPasses = 2;

    private readonly ITestOutputHelper _out;
    public Bench545Q4KMmqReductionTests(ITestOutputHelper output) => _out = output;

    private static bool Enabled =>
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_545_BENCH"), "1", StringComparison.Ordinal);

    /// <summary>(n, m, k) — real prefill shapes, and the K sweep that shows the depth effect.</summary>
    private static readonly (int n, int m, int k)[] Shapes =
    [
        (512, 2048, 2048),   // Llama-3.2-1B q/k/v/o at pp512
        (512, 8192, 2048),   // gate / up
        (512, 2048, 8192),   // down — 256 K-blocks, where the accuracy gain was largest
        (128, 2048, 2048),   // short prefill
    ];

    [SkippableFact]
    public void Mmq_TwoLevelAccumulation_PrefillCost()
    {
        Skip.IfNot(Enabled, "DOTLLM_545_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No VK_KHR_shader_integer_dot_product.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var shipped = MatMulQ4KMmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q4_k_mmq.spv missing.");
        MatMulQ4KMmqKernel? baseline =
            MatMulQ4KMmqKernel.TryCreate(device, spvDir, "matmul_q4_k_mmq_pre545");
        Skip.If(baseline is null, "matmul_q4_k_mmq_pre545.spv missing — nothing to compare against.");

        var sb = new StringBuilder();
        sb.AppendLine($"Device: {device.DeviceName} (VendorId 0x{device.VendorId:X4})");
        sb.AppendLine($"Passes={Passes} (median, min-max), Batch={Batch} dispatches/pass, order reversed per pass.");
        sb.AppendLine("baseline = matmul_q4_k_mmq_pre545 (byte-identical to dev's shipped spv).");
        sb.AppendLine();
        sb.AppendLine("| shape (n,m,k) | baseline us (min-max) | #545 us (min-max) | speedup (median) |");
        sb.AppendLine("|---|---:|---:|---:|");

        using (baseline)
        {
            foreach ((int n, int m, int k) in Shapes)
            {
                var rng = new Random(0x545 + n + m * 7 + k * 13);
                byte[] weightsQ8 = Q4KFixture.QuantizeRows(Q4KFixture.RandomFloats(rng, m * k, 0.1f), m, k);
                float[] b = Q4KFixture.RandomFloats(rng, n * k, 1.0f);

                using var bufW = device.Allocate(((long)weightsQ8.Length + 3) & ~3L);
                using var bufB = device.Allocate((long)n * k * sizeof(float));
                using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
                using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
                using var bufC = device.Allocate((long)n * m * sizeof(float));

                device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
                device.Upload(b, bufB);

                // Quantize once — it is common to both arms and must not be timed.
                using (var ctx = device.CreateSubmitContext())
                {
                    ctx.Begin();
                    quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n, k);
                    ctx.SubmitAndWait();
                }

                double Time(MatMulQ4KMmqKernel kernel)
                {
                    var sw = Stopwatch.StartNew();
                    using var ctx = device.CreateSubmitContext();
                    ctx.Begin();
                    for (int i = 0; i < Batch; i++)
                        kernel.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m, k, n);
                    ctx.SubmitAndWait();
                    sw.Stop();
                    return sw.Elapsed.TotalMilliseconds * 1000.0 / Batch;
                }

                for (int w = 0; w < WarmupPasses; w++) { Time(baseline!); Time(shipped); }

                var baseUs = new List<double>();
                var newUs = new List<double>();
                for (int p = 0; p < Passes; p++)
                {
                    // Reverse the order every other pass so a warm-up or clock-ramp
                    // advantage cannot accrue to whichever arm runs first.
                    if ((p & 1) == 0) { baseUs.Add(Time(baseline!)); newUs.Add(Time(shipped)); }
                    else { newUs.Add(Time(shipped)); baseUs.Add(Time(baseline!)); }
                }

                baseUs.Sort(); newUs.Sort();
                double bMed = baseUs[Passes / 2], nMed = newUs[Passes / 2];
                sb.AppendLine($"| ({n},{m},{k}) | {bMed:F2} ({baseUs[0]:F2}-{baseUs[^1]:F2}) "
                            + $"| {nMed:F2} ({newUs[0]:F2}-{newUs[^1]:F2}) | {bMed / nMed:F3}x |");
            }
        }

        _out.WriteLine(sb.ToString());
    }

    // ─────────────────────────────────────────────────────────────

}
