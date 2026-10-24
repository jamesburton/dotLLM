using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

/// <summary>
/// Separate-file MTP head (issue #1185): the CPU model validates and maps the <c>mtp-*.gguf</c> (see
/// <c>Qwen3HybridDenseTransformerModel.AttachMtpHead</c>); only the head block's own tensors are uploaded, in their stored quantisation,
/// exactly as for an embedded head. The file's <c>token_embd</c> / <c>output</c> copies never reach the device.
/// </summary>
public sealed partial class VulkanQwen3HybridDenseTransformerModel : IMtpHeadAttachable
{
    /// <inheritdoc/>
    public bool HasMtpHead => _mtpHead is not null;

    /// <inheritdoc/>
    /// <remarks>False for a model built from prebuilt weights: there is no CPU model to map the head file.</remarks>
    public bool CanAttachMtpHead => _cpuModel is not null;

    /// <inheritdoc/>
    public void AttachMtpHead(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_cpuModel is null) throw new NotSupportedException("An MTP head can only be attached to a model built from a GGUF.");
        if (_mtpHead is not null) throw new InvalidOperationException("An MTP head is already attached.");
        RejectFoldedMtpHead(Config);
        _cpuModel.AttachMtpHead(path);
        _mtpHead = VulkanQwen3HybridDenseMtpWeights.Upload(_device, Config, ExtractMtpHead(_cpuModel)!);
    }

    /// <summary>
    /// The MTP block's OWN projections are assumed unfolded — that is what Bonsai 2's MTP checkpoint ships (<c>blk.{NumLayers}.*</c> is absent
    /// from <c>prism.hadamard.weight_names</c>, and its matmuls are Q8_0 rather than PQ2_0) and it is what the CPU reference computes. Refuse
    /// loudly rather than silently mis-compute if a checkpoint folds them.
    /// </summary>
    private static void RejectFoldedMtpHead(ModelConfig config)
    {
        if (config.HadamardFold is not { } fold) return;
        string mtpPrefix = $"blk.{config.NumLayers}";
        foreach (string suffix in new[] { "attn_qkv.weight", "attn_gate.weight", "attn_q.weight",
                                          "attn_output.weight", "ffn_gate.weight", "ffn_up.weight",
                                          "ffn_down.weight", "nextn.eh_proj.weight" })
        {
            if (fold.IsFolded($"{mtpPrefix}.{suffix}"))
                throw new NotSupportedException(
                    $"Checkpoint folds the MTP block's own weight '{mtpPrefix}.{suffix}' " +
                    "(prism.hadamard.weight_names). Neither the CPU reference nor this Vulkan host " +
                    "rotates the MTP block's own projections — see issue #435.");
        }
    }
}
