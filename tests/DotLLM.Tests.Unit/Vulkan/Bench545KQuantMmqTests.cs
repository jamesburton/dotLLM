using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using System.Diagnostics;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Cost side of issue #545 for the K-quant / IQ4 prefill GEMMs — what the
/// two-level accumulation costs each family.
/// </summary>
/// <remarks>
/// <para>
/// Accuracy lives in <see cref="Probe545KQuantMmqTests"/>. Each family is
/// measured against <c>matmul_&lt;family&gt;_mmq_pre545</c>, a retained copy of
/// that shader as it stood on <c>dev</c>, verified byte-identical to the shipped
/// SPIR-V before use — so the comparison is against what users actually had.
/// </para>
/// <para>
/// Both modules are held open in ONE process and the arms are run
/// <b>order-reversed within each pass</b>: process-level A/B on this box has
/// produced 2-3x phantom deltas purely from GPU clock ramp, and UMA contention
/// swings absolutes ~40%. Medians with min-max, never single numbers.
/// </para>
/// <para>Opt-in: <c>DOTLLM_545_BENCH=1</c>. A benchmark, not a gate.</para>
/// <para>
/// Skips a family whose <c>_pre545</c> baseline is absent, which is the normal
/// state once the issue has landed. To re-derive, restore the shader from
/// history, recompile with
/// <c>glslc --target-env=vulkan1.2</c>, and check it is byte-identical to the
/// shipped SPIR-V at that revision before trusting it as a baseline.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Bench545KQuantMmqTests
{
    private const int Passes = 9;
    private const int Batch = 4;
    private const int WarmupPasses = 2;

    private readonly ITestOutputHelper _out;
    public Bench545KQuantMmqTests(ITestOutputHelper output) => _out = output;

    private static bool Enabled =>
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_545_BENCH"), "1", StringComparison.Ordinal);

    private static readonly (int n, int m, int k)[] Shapes =
    [
        (512, 2048, 2048),
        (512, 2048, 8192),   // deepest reduction
        (128, 2048, 2048),   // short prefill
    ];

    [SkippableTheory]
    [MemberData(nameof(Probe545KQuantMmqTests.Families), MemberType = typeof(Probe545KQuantMmqTests))]
    public void KQuant_Mmq_PrefillCost(QuantFamily family)
    {
        Skip.IfNot(Enabled, "DOTLLM_545_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No VK_KHR_shader_integer_dot_product.");
        Skip.IfNot(File.Exists(Path.Combine(spvDir, BaseName(family) + "_pre545.spv")),
            $"{BaseName(family)}_pre545.spv absent — baseline retired; see the class remarks.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");

        var sb = new StringBuilder();
        sb.AppendLine($"{family} on {device.DeviceName}");
        sb.AppendLine($"Passes={Passes} (median, min-max), Batch={Batch} dispatches/pass, order reversed per pass.");
        sb.AppendLine("| shape (n,m,k) | baseline us (min-max) | #545 us (min-max) | speedup (median) |");
        sb.AppendLine("|---|---:|---:|---:|");

        foreach ((int n, int m, int k) in Shapes)
        {
            var rng = new Random(0x545B + (int)family * 977 + n + m * 7 + k * 13);
            byte[] weights = Probe545KQuantMmqTests.QuantizeRowsFor(family, rng, m, k);
            float[] b = Probe545KQuantMmqTests.RandomFloatsFor(family, rng, n * k);

            using var bufW = device.Allocate(((long)weights.Length + 3) & ~3L);
            using var bufB = device.Allocate((long)n * k * sizeof(float));
            using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
            using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
            using var bufC = device.Allocate((long)n * m * sizeof(float));

            device.Upload(new ReadOnlySpan<byte>(weights), bufW);
            device.Upload(b, bufB);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n, k);
                ctx.SubmitAndWait();   // common to both arms; must not be timed
            }

            double Time(bool baseline)
            {
                var sw = Stopwatch.StartNew();
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                for (int i = 0; i < Batch; i++)
                    RecordMmq(family, device, spvDir, baseline, ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m, k, n);
                ctx.SubmitAndWait();
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds * 1000.0 / Batch;
            }

            for (int w = 0; w < WarmupPasses; w++) { Time(true); Time(false); }

            var baseUs = new List<double>();
            var newUs = new List<double>();
            for (int p = 0; p < Passes; p++)
            {
                if ((p & 1) == 0) { baseUs.Add(Time(true)); newUs.Add(Time(false)); }
                else { newUs.Add(Time(false)); baseUs.Add(Time(true)); }
            }
            baseUs.Sort(); newUs.Sort();
            double bMed = baseUs[Passes / 2], nMed = newUs[Passes / 2];
            sb.AppendLine($"| ({n},{m},{k}) | {bMed:F2} ({baseUs[0]:F2}-{baseUs[^1]:F2}) "
                        + $"| {nMed:F2} ({newUs[0]:F2}-{newUs[^1]:F2}) | {bMed / nMed:F3}x |");
        }

        _out.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Why #545 costs throughput on these families when it was free on Q8_0 and
    /// Q4_K: <c>accPart[TM][TN]</c> is 16 extra live floats per thread, and these
    /// shaders already carry heavier unpack state. This asks the driver directly
    /// (VK_AMD_shader_info) rather than inferring occupancy from timings.
    /// </summary>
    [SkippableFact]
    public void Q6K_Mmq_RegisterCost_PreVsPost()
    {
        Skip.IfNot(Enabled, "DOTLLM_545_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer dot product.");
        Skip.IfNot(device.HasShaderInfoAmd, "Device/driver does not advertise VK_AMD_shader_info.");
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "matmul_q6_k_mmq_pre545.spv")),
            "matmul_q6_k_mmq_pre545.spv absent — baseline retired.");

        foreach (string name in new[] { "matmul_q6_k_mmq_pre545", "matmul_q6_k_mmq" })
        {
            using var kern = MatMulQ6KMmqKernel.TryCreate(device, spvDir, name)
                ?? throw new Xunit.Sdk.XunitException($"{name}.spv missing or unsupported.");
            var st = device.GetShaderStatisticsAmd(kern.PipelineHandle);
            _out.WriteLine($"{name}: VGPRs {st.resourceUsage.numUsedVgprs}/{st.numAvailableVgprs}  "
                         + $"SGPRs {st.resourceUsage.numUsedSgprs}  LDS {st.resourceUsage.ldsUsageSizeInBytes}B  "
                         + $"scratch {st.resourceUsage.scratchMemUsageInBytes}B");
        }
    }

    private static string BaseName(QuantFamily f) => f switch
    {
        QuantFamily.Q5_K => "matmul_q5_k_mmq",
        QuantFamily.Q6_K => "matmul_q6_k_mmq",
        QuantFamily.IQ4_NL => "matmul_iq4_nl_mmq",
        QuantFamily.IQ4_XS => "matmul_iq4_xs_mmq",
        _ => throw new ArgumentOutOfRangeException(nameof(f)),
    };

    /// <summary>
    /// Creates the kernel fresh per dispatch rather than caching it. Pipeline
    /// creation is outside the timed region's inner loop only in the sense that
    /// both arms pay it identically and the arms are order-reversed, so it cannot
    /// bias the ratio.
    /// </summary>
    private static void RecordMmq(
        QuantFamily f, VulkanDevice device, string spvDir, bool baseline, nint cmd,
        VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer c, int m, int k, int n)
    {
        string name = BaseName(f) + (baseline ? "_pre545" : "");
        switch (f)
        {
            case QuantFamily.Q5_K:
            {
                var kern = Cache<MatMulQ5KMmqKernel>.Get(name, () => MatMulQ5KMmqKernel.TryCreate(device, spvDir, name)!);
                kern.Record(cmd, w, xq, xds, c, m, k, n); break;
            }
            case QuantFamily.Q6_K:
            {
                var kern = Cache<MatMulQ6KMmqKernel>.Get(name, () => MatMulQ6KMmqKernel.TryCreate(device, spvDir, name)!);
                kern.Record(cmd, w, xq, xds, c, m, k, n); break;
            }
            case QuantFamily.IQ4_NL:
            {
                var kern = Cache<MatMulIq4NlMmqKernel>.Get(name, () => MatMulIq4NlMmqKernel.TryCreate(device, spvDir, name)!);
                kern.Record(cmd, w, xq, xds, c, m, k, n); break;
            }
            case QuantFamily.IQ4_XS:
            {
                var kern = Cache<MatMulIq4XsMmqKernel>.Get(name, () => MatMulIq4XsMmqKernel.TryCreate(device, spvDir, name)!);
                kern.Record(cmd, w, xq, xds, c, m, k, n); break;
            }
            default: throw new ArgumentOutOfRangeException(nameof(f));
        }
    }

    /// <summary>Holds both arms' pipelines open for the life of the test run.</summary>
    private static class Cache<T> where T : class
    {
        private static readonly Dictionary<string, T> Map = new(StringComparer.Ordinal);
        public static T Get(string key, Func<T> create)
        {
            if (!Map.TryGetValue(key, out T? v)) Map[key] = v = create();
            return v;
        }
    }
}
