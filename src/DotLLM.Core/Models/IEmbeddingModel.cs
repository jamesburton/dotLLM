using DotLLM.Core.Tensors;

namespace DotLLM.Core.Models;

/// <summary>
/// A model that can expose its final hidden state (the input to the LM head) instead of logits.
/// Implemented by backends that support embedding extraction.
/// </summary>
/// <remarks>
/// <para>This is deliberately a <b>separate</b> interface from <see cref="IModel"/>: only backends
/// that have actually been validated against an external reference implement it. As of issue #451
/// that is the CPU <c>TransformerModel</c> only — the Vulkan and CUDA models do not implement this
/// interface, and the server reports that plainly rather than silently producing an unvalidated
/// vector.</para>
/// <para>The returned hidden state is taken at the same graph point llama.cpp calls
/// <c>result_norm</c> (<c>src/models/llama.cpp</c>: <c>res-&gt;t_embd = cur;</c> immediately after
/// the final <c>output_norm</c> and before <c>build_lora_mm(model.output, cur)</c>), which is the
/// tensor llama.cpp's pooling operates on.</para>
/// </remarks>
public interface IEmbeddingModel
{
    /// <summary>
    /// Runs a forward pass and returns the post-final-norm hidden state for every input position,
    /// stopping before the LM head.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this sequence.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <returns>A freshly allocated tensor of shape <c>[seqLen, hiddenSize]</c>, owned by the caller.</returns>
    ITensor ForwardHidden(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId);

    /// <summary>
    /// The pooling strategy declared by the model checkpoint (GGUF <c>{arch}.pooling_type</c>),
    /// or <c>null</c> when the checkpoint does not declare one.
    /// </summary>
    PoolingType? DeclaredPoolingType => null;
}
