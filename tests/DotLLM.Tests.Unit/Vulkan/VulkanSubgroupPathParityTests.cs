using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Runs both the shared-memory tree-reduce and the subgroup-arithmetic
/// variants of <c>rmsnorm_f32</c> and <c>attention_f32</c> against the same
/// inputs on a single device instance and asserts the outputs agree within
/// the existing abs 1e-4 / rel 1e-3 tolerance.
/// </summary>
/// <remarks>
/// This is the belt-and-braces cross-check that the two shader paths are
/// numerically interchangeable. Reduction order differs (warp-local vs.
/// stride-halving tree), so some drift is expected on the last float bit —
/// but it must stay inside the project's tolerance budget.
///
/// Reflection is used only to toggle the kernel's internal path flag: the
/// public API is intentionally unchanged (other agents are wiring these
/// kernels into the end-to-end forward path in parallel).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSubgroupPathParityTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableFact]
    public void RmsNorm_SubgroupAndSharedPathsAgree()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasSubgroupArithmetic,
            "Device does not advertise VK_SUBGROUP_FEATURE_ARITHMETIC_BIT; parity test is a no-op.");

        const int rowCount = 4;
        const int n = 1536;
        const float eps = 1e-5f;

        var rng = new Random(0xC0FFEE);
        float[] input = new float[rowCount * n];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        float[] weight = new float[n];
        for (int i = 0; i < n; i++) weight[i] = 0.5f + (float)rng.NextDouble();

        float[] subgroupOut = RunRmsNorm(device, spvDir, input, weight, rowCount, n, eps, forceShared: false, out bool subUses);
        float[] sharedOut   = RunRmsNorm(device, spvDir, input, weight, rowCount, n, eps, forceShared: true,  out bool sharedUses);

        Assert.True(subUses, "Subgroup path was requested but kernel chose shared-memory variant.");
        Assert.False(sharedUses, "Shared-memory path was forced but kernel chose subgroup variant.");
        AssertClose(sharedOut, subgroupOut);
    }

    [SkippableFact]
    public void Attention_SubgroupAndSharedPathsAgree()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasSubgroupArithmetic,
            "Device does not advertise VK_SUBGROUP_FEATURE_ARITHMETIC_BIT; parity test is a no-op.");

        // SmolLM-ish decode shape — single-tile but non-trivial reduction width.
        const int seqQ = 1, seqKv = 128, numHeads = 9, numKvHeads = 3, headDim = 64;
        const int positionOffset = 127;

        var rng = new Random(0xBEEF);
        float[] qh = new float[seqQ * numHeads * headDim];
        float[] kh = new float[seqKv * numKvHeads * headDim];
        float[] vh = new float[seqKv * numKvHeads * headDim];
        for (int i = 0; i < qh.Length; i++) qh[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < kh.Length; i++) kh[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < vh.Length; i++) vh[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] sub = RunAttention(device, spvDir, qh, kh, vh, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
                                   forceShared: false, out bool subUses);
        float[] shared = RunAttention(device, spvDir, qh, kh, vh, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
                                      forceShared: true, out bool sharedUses);

        Assert.True(subUses, "Subgroup path was requested but kernel chose shared-memory variant.");
        Assert.False(sharedUses, "Shared-memory path was forced but kernel chose subgroup variant.");
        AssertClose(shared, sub);
    }

    [SkippableFact]
    public void Attention_MultiTile_SubgroupAndSharedPathsAgree()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasSubgroupArithmetic,
            "Device does not advertise VK_SUBGROUP_FEATURE_ARITHMETIC_BIT; parity test is a no-op.");

        // Exercises the online-softmax tile loop: seqKv > TILE_KV = 256.
        const int seqQ = 1, seqKv = 400, numHeads = 4, numKvHeads = 2, headDim = 64;
        const int positionOffset = 399;

        var rng = new Random(0x1234);
        float[] qh = new float[seqQ * numHeads * headDim];
        float[] kh = new float[seqKv * numKvHeads * headDim];
        float[] vh = new float[seqKv * numKvHeads * headDim];
        for (int i = 0; i < qh.Length; i++) qh[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < kh.Length; i++) kh[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < vh.Length; i++) vh[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] sub = RunAttention(device, spvDir, qh, kh, vh, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
                                   forceShared: false, out _);
        float[] shared = RunAttention(device, spvDir, qh, kh, vh, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
                                      forceShared: true, out _);

        AssertClose(shared, sub);
    }

    // ─────────────────────────────────────────────────────────────

    private static float[] RunRmsNorm(
        VulkanDevice device, string spvDir, float[] input, float[] weight,
        int rowCount, int n, float eps, bool forceShared, out bool usesSubgroup)
    {
        string? original = Environment.GetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar, forceShared ? "1" : null);
            using var kernel = RmsNormF32Kernel.Create(device, spvDir);
            usesSubgroup = kernel.UsesSubgroupReduce;

            using var bufIn  = device.Allocate((long)rowCount * n * sizeof(float));
            using var bufW   = device.Allocate((long)n * sizeof(float));
            using var bufOut = device.Allocate((long)rowCount * n * sizeof(float));
            device.Upload(input, bufIn);
            device.Upload(weight, bufW);
            kernel.Launch(bufIn, bufW, bufOut, rowCount, n, eps);
            var actual = new float[rowCount * n];
            device.Download(bufOut, actual);
            return actual;
        }
        finally
        {
            Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar, original);
        }
    }

    private static float[] RunAttention(
        VulkanDevice device, string spvDir,
        float[] qh, float[] kh, float[] vh,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset,
        bool forceShared, out bool usesSubgroup)
    {
        string? original = Environment.GetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar, forceShared ? "1" : null);
            using var kernel = AttentionF32Kernel.Create(device, spvDir);
            usesSubgroup = kernel.UsesSubgroupReduce;

            using var bufQ   = device.Allocate((long)qh.Length * sizeof(float));
            using var bufK   = device.Allocate((long)kh.Length * sizeof(float));
            using var bufV   = device.Allocate((long)vh.Length * sizeof(float));
            using var bufOut = device.Allocate((long)seqQ * numHeads * headDim * sizeof(float));
            device.Upload(qh, bufQ);
            device.Upload(kh, bufK);
            device.Upload(vh, bufV);
            kernel.Launch(bufQ, bufK, bufV, bufOut, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);
            var actual = new float[seqQ * numHeads * headDim];
            device.Download(bufOut, actual);
            return actual;
        }
        finally
        {
            Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar, original);
        }
    }

    private static void AssertClose(float[] baseline, float[] candidate)
    {
        Assert.Equal(baseline.Length, candidate.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < baseline.Length; i++)
        {
            float e = baseline[i];
            float a = candidate[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"Cross-path drift exceeded tolerance: errors={errors}/{baseline.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
