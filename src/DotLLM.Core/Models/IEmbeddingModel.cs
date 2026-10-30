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
    /// Receives the post-norm hidden state of one sequence of a <see cref="ForwardHiddenBatch"/> call.
    /// May be invoked <b>concurrently</b> from several threads (one call per sequence), so it must only
    /// touch state owned by <paramref name="index"/>. <paramref name="hidden"/> is valid only for the call.
    /// </summary>
    /// <param name="index">Position of the sequence in the batch input list.</param>
    /// <param name="hidden"><c>[tokenCount, hiddenSize]</c> row-major F32 hidden state.</param>
    delegate void HiddenStateConsumer(int index, ReadOnlySpan<float> hidden);

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
    /// Runs <see cref="ForwardHidden"/> for every sequence and hands each result to <paramref name="consume"/>.
    /// Results are identical to calling <see cref="ForwardHidden"/> one sequence at a time (position <c>0..n-1</c>).
    /// The default implementation is that sequential loop; backends with inter-sequence parallelism (#1405)
    /// override it, length-sort internally, and may call <paramref name="consume"/> concurrently and out of order.
    /// </summary>
    /// <param name="sequences">Token id sequences; each must be non-empty.</param>
    /// <param name="consume">Per-sequence result sink (see <see cref="HiddenStateConsumer"/>).</param>
    /// <param name="deviceId">Target device for computation.</param>
    unsafe void ForwardHiddenBatch(IReadOnlyList<int[]> sequences, HiddenStateConsumer consume, int deviceId)
    {
        int h = 0;
        for (int i = 0; i < sequences.Count; i++)
        {
            int[] tokens = sequences[i];
            var positions = new int[tokens.Length];
            for (int p = 0; p < positions.Length; p++) positions[p] = p;
            using var hidden = ForwardHidden(tokens, positions, deviceId);
            h = hidden.Shape[1];
            consume(i, new ReadOnlySpan<float>((void*)hidden.DataPointer, tokens.Length * h));
        }
    }

    /// <summary>
    /// The pooling strategy declared by the model checkpoint (GGUF <c>{arch}.pooling_type</c>),
    /// or <c>null</c> when the checkpoint does not declare one.
    /// </summary>
    PoolingType? DeclaredPoolingType => null;

    /// <summary>
    /// Whether text inputs need the tokenizer's EOS token appended before <see cref="PoolingType.Last"/>
    /// pooling (decoder embedders such as Qwen3-Embedding, #1398). <c>false</c> for encoders, whose
    /// tokenizer already frames the sequence (<c>[CLS] ... [SEP]</c>).
    /// </summary>
    bool AppendsEosForLastPooling => true;
}
