namespace DotLLM.Core.Models;

/// <summary>
/// A model whose MTP draft head ships as a SEPARATE GGUF (qwen4exp: <c>mtp-*.gguf</c> beside the trunk) and is attached after the
/// trunk is loaded (issue #820). Implemented by the CPU and the Vulkan qwen4exp models; the resolver finds the file, the model attaches it.
/// </summary>
public interface IMtpHeadAttachable
{
    /// <summary>True once a head is attached (<see cref="IModel.SupportsMtp"/> is then true).</summary>
    bool HasMtpHead { get; }

    /// <summary>Opens <paramref name="path"/> as the head GGUF, attaches it and takes ownership of the file.</summary>
    /// <param name="path">Path of the <c>mtp-*.gguf</c> file.</param>
    void AttachMtpHead(string path);
}
