using DotLLM.Core.Models;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

public sealed partial class VulkanQwen3MoeHybridTransformerModel
{
    /// <summary>
    /// Builds a layer-less "host" of this class's recording machinery (kernel registry, scratch state, matmul / routed-MoE / embedding
    /// recorders) for <see cref="VulkanLfm2MoeTransformerModel"/>, the same composition pattern <c>VulkanQwen4ExpTransformerModel</c> uses.
    /// The host owns no per-layer weights and no Gated-DeltaNet state (a one-tap dummy GDN config sizes the unused scratch).
    /// </summary>
    internal static VulkanQwen3MoeHybridTransformerModel CreateLayerlessHost(
        VulkanDevice device, ModelConfig config, string spvDir, VulkanQwen3MoeHybridWeights headWeights,
        int[] kvSlotForLayer, int attentionLayerCount, int ropeDim, float ropeTheta)
    {
        var gdn = new GatedDeltaNetConfig(FullAttnInterval: 4, NVHead: 1, NKHead: 1, DState: 1, DInner: 1, DConv: 2);
        var state = new VulkanQwen3MoeHybridForwardState(device, config, gdn, initialSeqLen: 1);
        var gdnCache = new VulkanGdnStateCache(device, gdn, 0);
        var kernels = VulkanQwen3MoeHybridKernels.Create(device, spvDir, config.HeadDim);
        return new VulkanQwen3MoeHybridTransformerModel(
            device, ownsDevice: false, config, gguf: null, cpuModel: null,
            Array.Empty<DotLLM.Models.Architectures.Qwen3MoeLayerWeights>(), headWeights, state, gdnCache, kernels,
            kvSlotForLayer, attentionLayerCount, new int[config.NumLayers], ropeDim, ropeTheta, nCpuMoeLayers: 0);
    }

    /// <summary>Routed-MoE FFN for one layer with the pre-FFN norm already applied to <c>NormOutput</c>; result lands in <c>NormOutput</c>.</summary>
    internal void LfmRecordMoe(nint cmdBuf, VulkanQwen3MoeMoeUpload.LayerBundle bundle, VulkanDevice.Buffer ffnNorm, int seqLen)
        => RecordMoeLayer(cmdBuf, bundle, ffnNorm, seqLen, Config.HiddenSize, Config.NormEpsilon);

    internal int LfmGroupedMinTokens { get => GroupedMinTokens; set => GroupedMinTokens = value; }

    internal long[] LfmMoePathCounts => MoePathCounts;
}
