using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #240 audit, attention half — real, same-session, order-reversed A/B for the
/// production coopmat Flash-Attention kernel's SubgroupSize pin, following the exact
/// methodology <see cref="VulkanCoopmat32SubgroupPinBench"/> established for the GEMM
/// family.
/// </summary>
/// <remarks>
/// <para>
/// Unlike the GEMM kernels, <c>attention_flash_f32_coopmat.comp</c>'s own header already
/// documents its intended wave width as 64: <c>WG_SIZE=256</c>, <c>NUM_SLICES=4</c>, one
/// subgroup per KV-column slice via <c>gl_SubgroupID % 4</c> — that mapping is only
/// 1-subgroup-per-slice when the driver compiles the stage at subgroup size 64.
/// <see cref="VulkanDevice.HasSubgroupSizeControl"/>'s own doc comment records that
/// gfx1151's confirmed UNPINNED compute default already IS wave64 (the K-quant decode GEMV
/// kernels pin DOWN to 32 specifically to escape that default) — so, unlike the GEMM
/// pipelines, this kernel's workgroup size should already match its intended wave width
/// without any pin on the one hardware family this codebase has measured. This bench exists
/// to turn that structural prediction into a measured fact rather than ship (or skip) it on
/// theory alone, per #299's precedent: a pin is not assumed to help just because a prior
/// pin helped a structurally different kernel family.
/// </para>
/// <para>
/// Both variants load the exact same <c>attention_flash_f32_coopmat.spv</c> /
/// <c>attention_flash_f32_coopmat_hd64.spv</c> — only the
/// <c>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</c> passed at pipeline-creation
/// time differs (see <see cref="VulkanFlashAttentionCoopmatKernel.Create(VulkanDevice, string, FlashAttentionCoopmatVariant)"/>).
/// Shapes cover both the base (headDim 128, and headDim&lt;=64 below the hd64 seqKv
/// threshold) and hd64 (seqKv&gt;=640) dispatch paths so a pin effect specific to either
/// shader would show up.
/// </para>
/// <para>
/// <b>Result (Strix Halo, gfx1151, driver-reported <c>SubgroupSize=64</c> unpinned,
/// <c>VK_EXT_subgroup_size_control</c> min=32/max=64):</b> the pin measured as a confirmed
/// no-op across every shape tried, matching the structural prediction exactly —
/// </para>
/// <code>
/// | shape                                    | default us | pinned64 us | speedup |
/// |-------------------------------------------|-----------:|------------:|--------:|
/// | smollm_p128   (hd64,  base, seqKv&lt;640)    |      34.10 |       34.12 |   0.96x |
/// | llama3_p512   (hd64,  base, seqKv&lt;640)    |     811.25 |      825.17 |   0.99x |
/// | longctx_p2048 (hd64,  hd64 path)          |    2408.90 |     2419.57 |   0.99x |
/// | mha_p512_hd128 (base, headDim128)         |     384.95 |      389.73 |   1.00x |
/// </code>
/// <para>
/// See <see cref="FlashAttentionCoopmatVariant.SelectFor"/> for the default-selection
/// decision this result drives: stay unpinned. Reported per the issue's own "negative
/// results documented, not discarded" requirement — this is exactly the audit outcome
/// where "intended wave width == actual workgroup size already" means no fix is needed,
/// turned into a measured fact instead of an assumption.
/// </para>
/// <para>Enable with <c>DOTLLM_FA_COOPMAT64_BENCH=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanFlashAttentionCoopmatSubgroupPinBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 9;
    private const int Batch = 4;

    private readonly ITestOutputHelper _output;
    public VulkanFlashAttentionCoopmatSubgroupPinBench(ITestOutputHelper output) => _output = output;

    private static readonly (string Tag, int SeqQ, int SeqKv, int NumHeads, int NumKvHeads, int HeadDim)[] Shapes =
    [
        ("smollm_p128   (hd64,  base, seqKv<640)", 128,  128,  9,  3, 64),
        ("llama3_p512   (hd64,  base, seqKv<640)", 512,  512, 32,  4, 64),
        ("longctx_p2048 (hd64,  hd64 path)",       2048, 2048,  8,  2, 64),
        ("mha_p512_hd128 (base, headDim128)",       512,  512,  8,  8, 128),
    ];

    [SkippableFact]
    public void Bench_FlashAttentionCoopmatPinned64()
    {
        Skip.IfNot(Enabled, EnableMsg);
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        ReportDevice(device);
        Skip.IfNot(VulkanFlashAttentionCoopmatKernel.SupportsDevice(device),
            "Device does not expose a subgroup-scope 16x16x16 F16xF16->F32 cooperative-matrix tile.");
        Skip.IfNot(FlashAttentionCoopmatVariant.Pinned64.IsSupportedOn(device),
            "Device cannot pin requiredSubgroupSize=64.");

        using var refKernel = VulkanFlashAttentionCoopmatKernel.Create(device, spvDir, FlashAttentionCoopmatVariant.Default);
        using var challKernel = VulkanFlashAttentionCoopmatKernel.Create(device, spvDir, FlashAttentionCoopmatVariant.Pinned64);

        _output.WriteLine("");
        _output.WriteLine("### attention_flash_f32_coopmat: unpinned (Default) -> pinned requiredSubgroupSize=64");
        _output.WriteLine("| shape | default us | pinned64 us | speedup |");
        _output.WriteLine("|---|---:|---:|---:|");

        var rng = new Random(0x2A_53);
        foreach (var (tag, seqQ, seqKv, numHeads, numKvHeads, headDim) in Shapes)
        {
            long qBytes = (long)seqQ * numHeads * headDim * sizeof(float);
            long kvBytes = (long)seqKv * numKvHeads * headDim * sizeof(float);
            using var bufQ = device.Allocate(qBytes);
            using var bufK = device.Allocate(kvBytes);
            using var bufV = device.Allocate(kvBytes);
            using var bufOut = device.Allocate(qBytes);

            device.Upload(RandomFloats(rng, (int)(qBytes / sizeof(float))), bufQ);
            device.Upload(RandomFloats(rng, (int)(kvBytes / sizeof(float))), bufK);
            device.Upload(RandomFloats(rng, (int)(kvBytes / sizeof(float))), bufV);

            (double refUs, double challUs, double ratio) = MeasurePaired(
                (cb, batch) => { for (int i = 0; i < batch; i++) refKernel.Record(cb, bufQ, bufK, bufV, bufOut, seqQ, seqKv, numHeads, numKvHeads, headDim); },
                (cb, batch) => { for (int i = 0; i < batch; i++) challKernel.Record(cb, bufQ, bufK, bufV, bufOut, seqQ, seqKv, numHeads, numKvHeads, headDim); },
                device);

            _output.WriteLine($"| {tag} | {refUs:F2} | {challUs:F2} | {ratio:F2}x |");
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Shared A/B harness — same schedule as VulkanCoopmat32SubgroupPinBench.
    // ─────────────────────────────────────────────────────────────────────

    private static (double reference, double challenger, double ratio) MeasurePaired(
        Action<nint, int> recordReference, Action<nint, int> recordChallenger, VulkanDevice device)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunPass(device, recordReference, Batch);
            RunPass(device, recordChallenger, Batch);
        }

        var refUs = new double[Passes];
        var challUs = new double[Passes];
        var ratios = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            double tr, tc;
            if ((p & 1) == 0)
            {
                tr = RunPass(device, recordReference, Batch);
                tc = RunPass(device, recordChallenger, Batch);
            }
            else
            {
                tc = RunPass(device, recordChallenger, Batch);
                tr = RunPass(device, recordReference, Batch);
            }
            refUs[p] = tr;
            challUs[p] = tc;
            ratios[p] = tc > 0 ? tr / tc : 0;
        }

        Array.Sort(refUs); Array.Sort(challUs); Array.Sort(ratios);
        return (refUs[Passes / 2], challUs[Passes / 2], ratios[Passes / 2]);
    }

    private static double RunPass(VulkanDevice device, Action<nint, int> record, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        record(ctx.CommandBuffer, batch);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private void ReportDevice(VulkanDevice device)
    {
        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}  " +
            $"subgroup-size-control: supported={device.HasSubgroupSizeControl} min={device.MinSubgroupSize} max={device.MaxSubgroupSize}");
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    private static bool Enabled =>
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_FA_COOPMAT64_BENCH"), "1", StringComparison.Ordinal);

    private const string EnableMsg = "DOTLLM_FA_COOPMAT64_BENCH=1 to enable this benchmark.";
}
