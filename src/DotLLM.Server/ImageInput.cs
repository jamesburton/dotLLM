using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;

namespace DotLLM.Server;

/// <summary>A request image that cannot be served (no vision support, no projector, unsupported or malformed image): HTTP 400.</summary>
public sealed class ImageInputException(string message, string code = "invalid_image") : Exception(message)
{
    /// <summary>The OpenAI error code (<c>invalid_image</c>, or <c>context_length_exceeded</c> when the images do not fit).</summary>
    public string Code { get; } = code;
}

/// <summary>
/// Image input of the OpenAI chat endpoint (issue #2035): decodes <c>data:</c> URIs and encodes them with the loaded model's vision projector, which
/// is found next to the model GGUF (or named by <c>DOTLLM_MMPROJ</c>) and loaded on the first image request.
/// </summary>
/// <remarks>
/// The images are untrusted. Before anything expensive runs, the request is bounded by <see cref="ImageLimits"/> (configurable through
/// <see cref="ServerOptions.ImageLimits"/>: image count, decoded pixels per request and per image, side length, and whether an image over the token
/// limit is refused or downscaled), and the widened prompt (each image's token count, from
/// <see cref="IImageEncoder.CountTokens"/>, which costs nothing) must fit the context. Only then are the images encoded.
/// </remarks>
public static class ImageInput
{
    private sealed class Holder
    {
        public readonly object Lock = new();
        public IImageEncoder? Encoder;
        public string? EncoderFor;
        public ImageOverLimit OverLimit;
    }

    private static readonly ConditionalWeakTable<ServerState, Holder> s_holders = new();

    /// <summary>
    /// Encodes <paramref name="urls"/> (in order). The caller owns and disposes the embeddings. Runs on a worker thread so the request thread is
    /// not blocked by the encoder.
    /// </summary>
    /// <param name="state">The server state.</param>
    /// <param name="urls">The request's image URLs, in prompt order.</param>
    /// <param name="extraTokenBudget">
    /// Positions the prompt can still grow by: each image replaces its one placeholder by <c>Tokens</c> positions, so the request is refused
    /// (<c>context_length_exceeded</c>) when the images' <c>Tokens - 1</c> sum exceeds this. Checked before encoding.
    /// </param>
    /// <param name="ct">Cancellation.</param>
    /// <exception cref="ImageInputException">Image input is not available for this model, an image is unusable, or the images do not fit.</exception>
    public static Task<List<ImageEmbeddings>> EncodeAsync(ServerState state, IReadOnlyList<string> urls, int extraTokenBudget, CancellationToken ct)
        => Task.Run(() => Encode(state, urls, extraTokenBudget, ct), ct);

    /// <summary>Decodes the request's images under the per-request bounds (no model needed).</summary>
    /// <exception cref="ImageInputException">Too many images, too many pixels, or an image is not a usable <c>data:</c> URI.</exception>
    internal static List<DecodedImage> DecodeAll(IReadOnlyList<string> urls, ImageLimits? limits = null)
    {
        limits ??= ImageLimits.Default;
        if (urls.Count > limits.MaxImagesPerRequest)
            throw new ImageInputException($"Too many images: {urls.Count} (at most {limits.MaxImagesPerRequest} per request; raise it with --max-images or {ImageLimits.MaxImagesEnvVar}).");
        var decoded = new List<DecodedImage>(urls.Count);
        long pixels = 0;
        foreach (string url in urls)
        {
            if (!url.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                throw new ImageInputException("Only base64 data: URIs are accepted as image_url (remote URLs are not fetched by the server).");
            DecodedImage img;
            try { img = ImageDecoder.DecodeDataUri(url, limits); }
            catch (Exception ex) when (ex is not OperationCanceledException and not OutOfMemoryException)
            {
                // The decoder is pure: whatever it throws on attacker-supplied bytes is the client's error, never a 500.
                throw new ImageInputException("Unusable image: " + ex.Message);
            }
            pixels += (long)img.Width * img.Height;
            if (pixels > limits.MaxDecodedPixelsPerRequest)
                throw new ImageInputException($"The request's images hold more than {limits.MaxDecodedPixelsPerRequest} pixels in total (raise it with --max-request-image-pixels or {ImageLimits.MaxRequestPixelsEnvVar}).");
            decoded.Add(img);
        }
        return decoded;
    }

    /// <summary>
    /// The prompt positions the images add (<c>sum(Tokens - 1)</c>, one placeholder each), refusing an image the encoder cannot plan and a total
    /// above <paramref name="extraTokenBudget"/>.
    /// </summary>
    internal static int CheckTokenBudget(IImageEncoder encoder, IReadOnlyList<DecodedImage> images, int extraTokenBudget)
    {
        long extra = 0;
        foreach (var img in images)
        {
            int n;
            try { n = encoder.CountTokens(img.Width, img.Height); }
            catch (NotSupportedException ex) { throw new ImageInputException("Unusable image: " + ex.Message); }
            extra += n - 1;
        }
        if (extra > extraTokenBudget)
            throw new ImageInputException(
                $"The images add {extra} prompt tokens, more than the {Math.Max(0, extraTokenBudget)} the context has left.", "context_length_exceeded");
        return (int)extra;
    }

    /// <summary>
    /// The inference options of an image request: the encoded images and the placeholder token, with <c>MaxTokens</c> clamped to what the context has left
    /// after each placeholder widens to its image's token count. It runs AFTER the thinking-budget answer reserve has raised <c>MaxTokens</c>
    /// (<c>ReasoningSupport.WithAnswerReserve</c>) and after <c>ThinkingBudget</c> was attached, so the clamp bounds the reserve-raised value exactly as the context
    /// check bounds a text request, and the budget is carried through untouched (the context window caps the answer, never the budget).
    /// </summary>
    /// <param name="options">The options built for the text of the request.</param>
    /// <param name="images">The encoded images, in prompt order.</param>
    /// <param name="placeholderTokenId">The chat template's image placeholder token.</param>
    /// <param name="contextLength">The model's context length.</param>
    /// <param name="promptTokenCount">The prompt's token count before the placeholders widen.</param>
    internal static DotLLM.Core.Configuration.InferenceOptions WithImages(DotLLM.Core.Configuration.InferenceOptions options,
        IReadOnlyList<ImageEmbeddings> images, int placeholderTokenId, int contextLength, int promptTokenCount)
    {
        long extra = 0;
        foreach (var e in images) extra += e.Tokens - 1;
        int remaining = (int)Math.Clamp(contextLength - promptTokenCount - extra, 1, int.MaxValue);
        return options with { Images = images, ImagePlaceholderTokenId = placeholderTokenId, MaxTokens = Math.Min(options.MaxTokens, remaining) };
    }

    private static List<ImageEmbeddings> Encode(ServerState state, IReadOnlyList<string> urls, int extraTokenBudget, CancellationToken ct)
    {
        if (state.Model is not IImageInputModel)
            throw new ImageInputException($"The loaded model ({state.Config?.Architecture}) does not accept image input.");
        var limits = state.Options.ImageLimits;
        var decoded = DecodeAll(urls, limits);

        var holder = s_holders.GetOrCreateValue(state);
        lock (holder.Lock)
        {
            if (holder.Encoder is null || holder.EncoderFor != state.LoadedModelPath || holder.OverLimit != limits.OverLimit)
            {
                holder.Encoder?.Dispose();
                holder.Encoder = null;
                // DOTLLM_MMPROJ: path of the vision projector GGUF for image input (default: the mmproj*.gguf next to the model file).
                string? mmproj = Environment.GetEnvironmentVariable("DOTLLM_MMPROJ");
                if (string.IsNullOrEmpty(mmproj)) mmproj = VisionSupport.FindMmproj(state.LoadedModelPath);
                if (mmproj is null || !File.Exists(mmproj))
                    throw new ImageInputException("This model has no vision projector: place its mmproj*.gguf next to the model file (or set DOTLLM_MMPROJ).");
                // A malformed DOTLLM_VISION_DEVICE is reported as such; any other ArgumentException from the loader is a bug and must not be dressed up as a client error.
                try { VisionSupport.ForceCpuFromEnvironment(); }
                catch (ArgumentException ex) { throw new ImageInputException(ex.Message); }
                try { holder.Encoder = VisionSupport.LoadEncoderFor(state.Model, mmproj, new ThreadingConfig(state.Options.Threads, state.Options.DecodeThreads), limits.OverLimit); }
                catch (NotSupportedException ex) { throw new ImageInputException(ex.Message); }
                holder.EncoderFor = state.LoadedModelPath;
                holder.OverLimit = limits.OverLimit;
            }
            CheckTokenBudget(holder.Encoder, decoded, extraTokenBudget);
            var result = new List<ImageEmbeddings>(decoded.Count);
            try
            {
                foreach (var img in decoded) result.Add(holder.Encoder.Encode(img, ct));
                return result;
            }
            catch
            {
                foreach (var e in result) e.Dispose();
                throw;
            }
        }
    }
}
