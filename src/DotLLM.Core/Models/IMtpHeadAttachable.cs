namespace DotLLM.Core.Models;

/// <summary>
/// A model whose MTP draft head ships as a SEPARATE GGUF (qwen4exp: <c>mtp-*.gguf</c> beside the trunk) and is attached after the
/// trunk is loaded (issue #820). Implemented by the CPU and the Vulkan qwen4exp models; the resolver finds the file, the model attaches it.
/// </summary>
public interface IMtpHeadAttachable
{
    /// <summary>True once a head is attached (<see cref="IModel.SupportsMtp"/> is then true).</summary>
    bool HasMtpHead { get; }

    /// <summary>
    /// False when the model cannot use a head in its current configuration (expert tiering, #939: a 5-row verify touches 31-33 experts per layer, so under an SSD-bound
    /// slot cache speculation costs more than it saves). The resolver then attaches nothing (and refuses an explicit <c>--mtp-head</c>).
    /// </summary>
    bool CanAttachMtpHead => true;

    /// <summary>Opens <paramref name="path"/> as the head GGUF, attaches it and takes ownership of the file.</summary>
    /// <param name="path">Path of the <c>mtp-*.gguf</c> file.</param>
    void AttachMtpHead(string path);
}
