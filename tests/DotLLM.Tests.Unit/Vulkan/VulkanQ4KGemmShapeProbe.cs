using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in probe (DOTLLM_GEMM_SHAPE_PROBE=1; compare with DOTLLM_VK_GEMM_SPLITK=0): achieved TFLOPS of the Q4_K 128x128 coopmat GEMM at the Tev1-4B prefill shapes
/// (n = 128 tokens). Each shape runs <c>Reps</c> dispatches back to back in ONE command buffer, so launch/submit cost
/// amortises and the figure is the kernel's steady-state rate (weights device-local, as in a real run).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQ4KGemmShapeProbe
{
    private readonly ITestOutputHelper _out;
    public VulkanQ4KGemmShapeProbe(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void Probe_Tev1PrefillShapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GEMM_SHAPE_PROBE") == "1", "opt-in probe");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulQ4KGemmCoopmatKernel.IsSupportedOn(device, spvDir), "Q4_K coopmat kernel unsupported here.");
        using var kernel = MatMulQ4KGemmCoopmatKernel.Create(device, spvDir);

        (string Name, int M, int K)[] shapes =
        [
            ("ffn gate/up   ", 9216, 2560),
            ("gdn qkv       ", 8192, 2560),
            ("ffn down      ", 2560, 9216),
            ("gdn out/attn-o", 2560, 4096),
            ("gdn z/attn-q  ", 4096, 2560),
        ];
        foreach (int n in new[] { 128, 512 })
        foreach (var (name, m, k) in shapes)
        {
            long wBytes = (long)m * (k / 256) * 144;
            using var w = device.AllocateDeviceLocal((wBytes + 3) & ~3L);
            using var b = device.AllocateDeviceLocal((long)n * k * sizeof(float));
            using var c = device.AllocateDeviceLocal((long)n * m * sizeof(float));
            const int Reps = 40;
            double best = double.MaxValue;
            for (int trial = 0; trial < 5; trial++)
            {
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                for (int r = 0; r < Reps; r++)
                {
                    kernel.Record(ctx.CommandBuffer, w, b, c, m, k, n);
                    KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                }
                var sw = Stopwatch.StartNew();
                ctx.SubmitAndWait();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds / Reps);
            }
            double tflops = 2.0 * m * k * n / (best * 1e-3) / 1e12;
            int wgs = ((m + 127) / 128) * ((n + 127) / 128);
            _out.WriteLine($"n={n,4} {name} m={m,5} k={k,5}  WGs={wgs,4}  {best,7:F3} ms  {tflops,5:F1} TFLOPS");
        }
    }
}
