using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;

namespace DotLLM.Server;

/// <summary>A request image that cannot be served (no vision support, no projector, unsupported or malformed image): HTTP 400.</summary>
public sealed class ImageInputException(string message) : Exception(message);

/// <summary>
/// Image input of the OpenAI chat endpoint (issue #2035): decodes <c>data:</c> URIs and encodes them with the loaded model's vision projector, which
/// is found next to the model GGUF (or named by <c>DOTLLM_MMPROJ</c>) and loaded on the first image request.
/// </summary>
public static class ImageInput
{
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
    /// <exception cref="ImageInputException">Image input is not available for this model, or an image is unusable.</exception>
    public static Task<List<ImageEmbeddings>> EncodeAsync(ServerState state, IReadOnlyList<string> urls, CancellationToken ct)
        => Task.Run(() => Encode(state, urls, ct), ct);

    private static List<ImageEmbeddings> Encode(ServerState state, IReadOnlyList<string> urls, CancellationToken ct)
    {
        if (state.Model is not IImageInputModel)
            throw new ImageInputException($"The loaded model ({state.Config?.Architecture}) does not accept image input.");
        var decoded = new List<DecodedImage>(urls.Count);
        foreach (string url in urls)
        {
            try
            {
                if (!url.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                    throw new ImageInputException("Only base64 data: URIs are accepted as image_url (remote URLs are not fetched by the server).");
                decoded.Add(ImageDecoder.DecodeDataUri(url));
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidDataException or FormatException or ArgumentException or EndOfStreamException)
            {
                throw new ImageInputException("Unusable image: " + ex.Message);
            }
        }

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
