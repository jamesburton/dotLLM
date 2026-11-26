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
/// The images are untrusted. Before anything expensive runs, the request is bounded: at most <see cref="MaxImagesPerRequest"/> images and
/// <see cref="MaxDecodedPixelsPerRequest"/> decoded pixels, and the widened prompt (each image's token count, from
/// <see cref="IImageEncoder.CountTokens"/>, which costs nothing) must fit the context. Only then are the images encoded.
/// </remarks>
public static class ImageInput
{
    /// <summary>Most images one request may carry.</summary>
    public const int MaxImagesPerRequest = 16;

    /// <summary>Most decoded pixels one request may hold at once (2^26: two 8K frames, or many ordinary photos).</summary>
    public const long MaxDecodedPixelsPerRequest = 1L << 26;

    private sealed class Holder
    {
        public readonly object Lock = new();
        public IImageEncoder? Encoder;
        public string? EncoderFor;
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
    internal static List<DecodedImage> DecodeAll(IReadOnlyList<string> urls)
    {
        if (urls.Count > MaxImagesPerRequest)
            throw new ImageInputException($"Too many images: {urls.Count} (at most {MaxImagesPerRequest} per request).");
        var decoded = new List<DecodedImage>(urls.Count);
        long pixels = 0;
        foreach (string url in urls)
        {
            if (!url.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                throw new ImageInputException("Only base64 data: URIs are accepted as image_url (remote URLs are not fetched by the server).");
            DecodedImage img;
            try { img = ImageDecoder.DecodeDataUri(url); }
            catch (Exception ex) when (ex is not OperationCanceledException and not OutOfMemoryException)
            {
                // The decoder is pure: whatever it throws on attacker-supplied bytes is the client's error, never a 500.
                throw new ImageInputException("Unusable image: " + ex.Message);
            }
            pixels += (long)img.Width * img.Height;
            if (pixels > MaxDecodedPixelsPerRequest)
                throw new ImageInputException($"The request's images hold more than {MaxDecodedPixelsPerRequest} pixels in total.");
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

    private static List<ImageEmbeddings> Encode(ServerState state, IReadOnlyList<string> urls, int extraTokenBudget, CancellationToken ct)
    {
        if (state.Model is not IImageInputModel)
            throw new ImageInputException($"The loaded model ({state.Config?.Architecture}) does not accept image input.");
        var decoded = DecodeAll(urls);

        var holder = s_holders.GetOrCreateValue(state);
        lock (holder.Lock)
        {
            if (holder.Encoder is null || holder.EncoderFor != state.LoadedModelPath)
            {
                holder.Encoder?.Dispose();
                holder.Encoder = null;
                // DOTLLM_MMPROJ: path of the vision projector GGUF for image input (default: the mmproj*.gguf next to the model file).
                string? mmproj = Environment.GetEnvironmentVariable("DOTLLM_MMPROJ");
                if (string.IsNullOrEmpty(mmproj)) mmproj = VisionSupport.FindMmproj(state.LoadedModelPath);
                if (mmproj is null || !File.Exists(mmproj))
                    throw new ImageInputException("This model has no vision projector: place its mmproj*.gguf next to the model file (or set DOTLLM_MMPROJ).");
                try { holder.Encoder = VisionSupport.LoadEncoder(mmproj, new ThreadingConfig(state.Options.Threads, state.Options.DecodeThreads)); }
                catch (NotSupportedException ex) { throw new ImageInputException(ex.Message); }
                holder.EncoderFor = state.LoadedModelPath;
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
