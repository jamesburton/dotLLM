using System;
using DotLLM.Core.Attention;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Pins the scratch-sizing invariant that <see cref="VulkanSplitKvAttentionKernel"/> depends on
/// for command-buffer safety (#465).
/// </summary>
/// <remarks>
/// <para>
/// <b>The defect.</b> <c>EnsureScratch</c> grows the partial scratch and then calls
/// <c>InvalidateDescriptorCache()</c> → <c>vkResetDescriptorPool</c>, which <b>frees descriptor
/// sets that dispatches already recorded into the open command buffer still reference.</b> It
/// justified this with an invariant stated in its own comment — <i>"within one forward seqKv
/// (hence numSplits) is constant"</i>. That holds for single-sequence <c>Forward</c> and is
/// <b>false for <c>ForwardBatch</c></b>, which reads <c>seqKv</c> from each sequence's own cache
/// inside the per-sequence loop. A batch with ascending KV lengths therefore reallocated
/// mid-recording.
/// </para>
/// <para>
/// <b>Why this test is shaped as a reallocation count.</b> The use-after-free itself is
/// timing- and driver-dependent: it may corrupt, may fault, may silently work. Asserting on
/// output would be a flaky test of a deterministic defect. The <i>reallocation</i> is the
/// deterministic precondition — no growth, no pool reset, no freed descriptors — so that is
/// what is pinned. This also means the test stays meaningful on drivers that happen to tolerate
/// the dangling sets.
/// </para>
/// <para>
/// <b>Why not a batched-decode model test.</b> Existing batch coverage
/// (<c>VulkanPipelineParityTests.PipelinedForwardBatch_MatchesPerSequenceForward</c>) batches
/// unequal lengths 4/3/5 and <b>cannot</b> reach this: <c>ComputeSplits(5, …) =
/// ceil(5/16) = 1</c>, so no split occurs, and the split path is gated on <c>seqLen == 1</c>
/// besides. Reaching the defect needs decode-shaped calls whose KV lengths straddle a
/// <c>numSplits</c> boundary — which is exactly what the shapes below do, at the kernel level
/// where it is deterministic.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class SplitKvScratchInvarianceTests
{
    /// <summary>
    /// The bound must not move with the cache length — this is the whole fix, and it is pure
    /// arithmetic, so it runs without a GPU.
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(9)]   // SmolLM: TargetWorkgroups/numHeads does not divide evenly
    [InlineData(32)]
    [InlineData(512)] // more heads than the workgroup target -> bound clamps to 1
    public void MaxSplitsFor_BoundsComputeSplits_AtEveryContextLength(int numHeads)
    {
        int bound = VulkanSplitKvAttentionKernel.MaxSplitsFor(numHeads);

        foreach (int seqKv in new[] { 1, 15, 16, 17, 63, 256, 999, 4096, 32768, 131072 })
        {
            int actual = VulkanSplitKvAttentionKernel.ComputeSplits(seqKv, numHeads);
            Assert.True(actual <= bound,
                $"ComputeSplits(seqKv={seqKv}, numHeads={numHeads}) = {actual} exceeds the " +
                $"sizing bound {bound} — the scratch would grow mid-command-buffer.");
        }

        // The bound must also be REACHABLE, or it is merely a large number and this test would
        // pass for a bound of int.MaxValue while the scratch wasted memory.
        Assert.Equal(bound, VulkanSplitKvAttentionKernel.ComputeSplits(1 << 20, numHeads));
    }

    /// <summary>
    /// The behavioural half: driving the real <c>Record</c> path with ascending KV lengths must
    /// allocate the scratch exactly once. Before the fix this incremented on every shape whose
    /// split count grew — the mid-command-buffer reset.
    /// </summary>
    [SkippableFact]
    public void AscendingKvLengths_AllocateScratchOnce()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int numHeads = 4;
        const int numKvHeads = 2;
        const int headDim = 64;

        // Ascending so each step would grow the old sizing: S = min(64, ceil(seqKv/16)),
        // giving 1 -> 4 -> 25 -> 64 -> 64. The first four strictly increase, so the pre-fix
        // code reallocated (and reset the descriptor pool) four times.
        int[] kvLengths = [16, 64, 400, 4096, 8192];

        // Guard: the shapes must actually differ in split count, or this proves nothing.
        var splits = Array.ConvertAll(kvLengths, n => VulkanSplitKvAttentionKernel.ComputeSplits(n, numHeads));
        Assert.True(splits[^1] > splits[0],
            $"Shapes do not vary the split count ({string.Join(",", splits)}) — the test cannot discriminate.");

        using var device = VulkanDevice.Create();
        using var kernel = VulkanSplitKvAttentionKernel.Create(device, spvDir);

        foreach (int seqKv in kvLengths)
            LaunchOne(device, kernel, seqKv, numHeads, numKvHeads, headDim);

        Assert.Equal(1, kernel.ScratchAllocationCount);
    }

    /// <summary>
    /// DESCENDING order is the control. The pre-fix code sized on the call's own split count and
    /// only grew, so descending lengths never reallocated and would have passed even unfixed.
    /// Keeping both directions makes it explicit that the ascending case is the discriminating
    /// one — and catches a "fix" that merely reallocated in the other direction.
    /// </summary>
    [SkippableFact]
    public void DescendingKvLengths_AlsoAllocateScratchOnce()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int numHeads = 4;
        const int numKvHeads = 2;
        const int headDim = 64;

        using var device = VulkanDevice.Create();
        using var kernel = VulkanSplitKvAttentionKernel.Create(device, spvDir);

        foreach (int seqKv in new[] { 8192, 4096, 400, 64, 16 })
            LaunchOne(device, kernel, seqKv, numHeads, numKvHeads, headDim);

        Assert.Equal(1, kernel.ScratchAllocationCount);
    }

    /// <summary>
    /// Per-layer headDim variation must not reallocate either — the pre-existing half of the
    /// invariant (Gemma alternates global/sliding layers). Regression cover for the sizing
    /// rewrite: it would be easy to fix the numSplits axis and break this one.
    /// </summary>
    [SkippableFact]
    public void VaryingHeadDim_AllocatesScratchOnce()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        using var kernel = VulkanSplitKvAttentionKernel.Create(device, spvDir);

        foreach (int headDim in new[] { 64, 128, 64 })
            LaunchOne(device, kernel, seqKv: 4096, numHeads: 8, numKvHeads: 8, headDim: headDim);

        Assert.Equal(1, kernel.ScratchAllocationCount);
    }

    private static void LaunchOne(
        VulkanDevice device, VulkanSplitKvAttentionKernel kernel,
        int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        var rng = new Random(0x465 + seqKv + headDim);
        float[] qh = RandomFloats(rng, numHeads * headDim);          // seqQ == 1
        float[] kh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] vh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] outh = new float[numHeads * headDim];

        using var bufQ = device.Allocate((long)qh.Length * sizeof(float));
        using var bufK = device.Allocate((long)kh.Length * sizeof(float));
        using var bufV = device.Allocate((long)vh.Length * sizeof(float));
        using var bufO = device.Allocate((long)outh.Length * sizeof(float));

        device.Upload(qh.AsSpan(), bufQ);
        device.Upload(kh.AsSpan(), bufK);
        device.Upload(vh.AsSpan(), bufV);

        kernel.Launch(bufQ, bufK, bufV, bufO,
            seqQ: 1, seqKv: seqKv, numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            positionOffset: seqKv - 1, maskMode: AttentionMaskMode.Causal);

        device.Download(bufO, outh);

        // Not a parity assertion (that is VulkanSplitKvAttentionKernelTests' job) — but the
        // output must at least be finite, so a run that silently produced garbage after a
        // descriptor reset does not read as a pass.
        foreach (float f in outh)
            Assert.True(float.IsFinite(f), "Split-KV produced a non-finite value.");
    }

    private static float[] RandomFloats(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
