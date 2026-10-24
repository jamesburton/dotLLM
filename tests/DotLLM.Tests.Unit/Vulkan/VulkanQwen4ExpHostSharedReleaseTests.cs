using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #890: the Qwen4-Exp Vulkan loader frees the host F32 shared-expert copies once they are uploaded (they were ~0.9 GiB of private memory
/// on the real file, the margin between loading and the OS per-process residency cap). No GPU needed.
/// </summary>
public sealed unsafe class VulkanQwen4ExpHostSharedReleaseTests
{
    private static nint Alloc(long bytes) => (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);

    private static MoeLayerWeights Layer(nint gate, nint up, nint down, int sharedInter) => new(
        gate: [], w1: [], w2: [], w3: [], numExperts: 0, numExpertsPerTok: 0, hiddenSize: 8, intermediateSize: 0, normTopKProb: false,
        sharedGateProj: [gate], sharedUpProj: [up], sharedDownProj: [down], sharedIntermediateSize: sharedInter, sharedExpertGate: null,
        gateExpsRaw: 0, gateExpsRawQt: QuantizationType.F32, gateExpsMDim: 0, gateExpsKDim: 0,
        upExpsRaw: 0, upExpsRawQt: QuantizationType.F32, upExpsMDim: 0, upExpsKDim: 0,
        downExpsRaw: 0, downExpsRawQt: QuantizationType.F32, downExpsMDim: 0, downExpsKDim: 0,
        sharedGateRaw: [], sharedGateRawQt: QuantizationType.F32, sharedUpRaw: [], sharedUpRawQt: QuantizationType.F32,
        sharedDownRaw: [], sharedDownRawQt: QuantizationType.F32);

    [Fact]
    public void Release_FreesOwnedPointers_ZeroesThem_AndLeavesForeignOnesAlone()
    {
        const int hidden = 8, inter = 4;
        long bytes = (long)hidden * inter * sizeof(float);
        nint g = Alloc(bytes), u = Alloc(bytes), foreign = Alloc(bytes);
        nint d = Alloc(bytes);
        var owned = new List<nint> { g, u, d };   // `foreign` is not tracked by the loader: it must survive (e.g. an mmap view)
        var moe = Layer(g, u, foreign, inter);
        try
        {
            long freed = VulkanQwen4ExpTransformerModel.ReleaseHostSharedExperts(moe, owned, hidden);

            Assert.Equal(2 * bytes, freed);
            Assert.Equal(0, moe.SharedGateProj[0]);
            Assert.Equal(0, moe.SharedUpProj[0]);
            Assert.Equal(foreign, moe.SharedDownProj[0]);
            Assert.Equal([d], owned);   // only the pointers the call freed left the ledger
        }
        finally
        {
            NativeMemory.AlignedFree((void*)foreign);
            NativeMemory.AlignedFree((void*)d);
        }
    }

    [Fact]
    public void Release_IsIdempotent()
    {
        const int hidden = 8, inter = 4;
        long bytes = (long)hidden * inter * sizeof(float);
        nint g = Alloc(bytes), u = Alloc(bytes), d = Alloc(bytes);
        var owned = new List<nint> { g, u, d };
        var moe = Layer(g, u, d, inter);

        Assert.Equal(3 * bytes, VulkanQwen4ExpTransformerModel.ReleaseHostSharedExperts(moe, owned, hidden));
        Assert.Equal(0, VulkanQwen4ExpTransformerModel.ReleaseHostSharedExperts(moe, owned, hidden));   // second call: nothing left, no double free
        Assert.Empty(owned);
    }
}
