using DotLLM.Core.Tensors;

namespace DotLLM.Core.Vision;

/// <summary>
/// Turns an image into rows of text-model embeddings (a vision tower plus its projector). Backend-pluggable: the CPU implementation is
/// <c>Inkling2VisionEncoder</c>; a Vulkan one produces the same <see cref="ImageEmbeddings"/> (a device-side encoder can copy its output into
/// the host F16 buffer, or a later device-resident variant can extend this interface).
/// </summary>
public interface IImageEncoder : IDisposable
{
    /// <summary>The text model hidden size the embeddings are projected to.</summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Number of prompt tokens (embedding rows) the image will occupy after preprocessing (resize/pad to the patch grid), without encoding it.
    /// The caller needs this to expand the chat template's single image placeholder into the right number of positions.
    /// </summary>
    int CountTokens(int width, int height);

    /// <summary>Preprocesses and encodes <paramref name="image"/>. The result is owned by the caller.</summary>
    ImageEmbeddings Encode(DecodedImage image, CancellationToken cancellationToken = default);
}

/// <summary>An image occupying <see cref="Embeddings"/>.Tokens consecutive prompt positions starting at <see cref="Start"/>.</summary>
/// <param name="Start">Absolute position (in the sequence) of the first image row.</param>
/// <param name="Embeddings">The encoded image; the model reads it during the forward call only.</param>
public readonly record struct ImageSpan(int Start, ImageEmbeddings Embeddings)
{
    /// <summary>One past the last position the image covers.</summary>
    public int End => Start + Embeddings.Tokens;
}

/// <summary>
/// A model whose prompt can carry image embeddings in place of token embeddings. The token ids at the covered positions are ignored (they hold
/// the template placeholder); everything else is an ordinary forward.
/// </summary>
public interface IImageInputModel
{
    /// <summary>
    /// Forward over <paramref name="tokenIds"/> at <paramref name="positions"/>; for every position covered by an entry of
    /// <paramref name="images"/> the corresponding embedding row is used instead of the token embedding.
    /// </summary>
    /// <param name="tokenIds">Token ids of this chunk (the ids at image positions are ignored).</param>
    /// <param name="positions">Absolute sequence positions of <paramref name="tokenIds"/>.</param>
    /// <param name="deviceId">Device id (-1 = host).</param>
    /// <param name="images">Image spans in absolute sequence positions (the same list for every chunk of a chunked prefill).</param>
    /// <param name="kvCache">
    /// The KV cache (or external KV handle) the engine allocated for this request, the same object it passes to <c>IModel.Forward</c>. A model that
    /// keeps its own sequence state may ignore it (the CPU Inkling 2 model does); one that stores K/V through the handle (the Vulkan Inkling 2 model's
    /// <c>CreateKvCacheHandle</c> / <c>Inkling2KvHandle</c>) writes the image rows' K/V into it. Added in review of #2035 so the interface is not CPU-only.
    /// There is no LoRA adapter or MTP state parameter: the callers refuse an adapter together with images (server, CLI and engine), because a prefill
    /// that silently dropped it would answer from the base model.
    /// </param>
    /// <param name="lastTokenLogitsOnly">Return only the last row's logits.</param>
    ITensor ForwardWithImages(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IReadOnlyList<ImageSpan> images,
        DotLLM.Core.Attention.IKvCache? kvCache, bool lastTokenLogitsOnly);
}
