using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Vision;

/// <summary>Resolution of the vision projector (<c>mmproj</c>) that belongs to a model, and construction of its <see cref="IImageEncoder"/>.</summary>
public static class VisionSupport
{
    /// <summary>The chat template's single image placeholder token of Inkling 2 (<c>processor_config.json</c> <c>image_token</c>).</summary>
    public const string Inkling2ImageToken = "<|unused_200054|>";

    /// <summary>
    /// The mmproj GGUF next to <paramref name="modelPath"/> (the layout of a Hugging Face GGUF repo snapshot and of a local folder): the first
    /// <c>mmproj*.gguf</c> in the same directory, or null.
    /// </summary>
    public static string? FindMmproj(string modelPath)
    {
        string? dir = Path.GetDirectoryName(Path.GetFullPath(modelPath));
        if (dir is null || !Directory.Exists(dir)) return null;
        return Directory.EnumerateFiles(dir, "mmproj*.gguf").OrderBy(p => p, StringComparer.OrdinalIgnoreCase).FirstOrDefault();
    }

    /// <summary>Environment variable that forces the CPU vision encoder even when the model can build a device encoder (<c>cpu</c>).</summary>
    public const string DeviceEnvVar = "DOTLLM_VISION_DEVICE";

    /// <summary>
    /// The encoder for <paramref name="model"/>: the model's own device encoder when it provides one (<see cref="IImageEncoderProvider"/>: the Vulkan Inkling 2 model's Vulkan ViT, issue #2062),
    /// unless <c>DOTLLM_VISION_DEVICE=cpu</c>; otherwise the CPU encoder (<see cref="LoadEncoder"/>).
    /// </summary>
    public static IImageEncoder LoadEncoderFor(object? model, string mmprojPath, ThreadingConfig threading, ImageOverLimit overLimit = ImageOverLimit.Refuse)
    {
        bool cpu = ForceCpuFromEnvironment();
        if (!cpu && model is IImageEncoderProvider provider)
        {
            // A device encoder covers a narrower contract than the CPU one (the Vulkan ViT reads Q8_0 matrices only and needs subgroup arithmetic / 16-bit storage);
            // an mmproj or device it declines still gets the CPU encoder, as before the device encoder existed, rather than losing image input.
            try { return provider.CreateImageEncoder(mmprojPath, overLimit); }
            catch (NotSupportedException ex) { Console.Error.WriteLine($"[vision] device encoder declined ({ex.Message}); using the CPU encoder."); }
        }
        return LoadEncoder(mmprojPath, threading, overLimit);
    }

    /// <summary>
    /// Reads <see cref="DeviceEnvVar"/>: <c>cpu</c> forces the CPU encoder; unset, empty, <c>auto</c>, <c>gpu</c> or <c>device</c> use the model's own device encoder when it has one.
    /// Anything else is an error (a typo must not silently run the other encoder, as for <c>DOTLLM_VISION_EMBD_PRECISION</c>).
    /// </summary>
    /// <exception cref="ArgumentException">An unrecognised value.</exception>
    public static bool ForceCpuFromEnvironment(Func<string, string?>? env = null)
    {
        string? text = (env ?? Environment.GetEnvironmentVariable)(DeviceEnvVar)?.Trim();
        if (string.IsNullOrEmpty(text)) return false;
        return text.ToLowerInvariant() switch
        {
            "cpu" => true,
            "auto" or "gpu" or "device" => false,
            _ => throw new ArgumentException($"{DeviceEnvVar} must be cpu, auto, gpu or device, got '{text}'."),
        };
    }

    /// <summary>Opens <paramref name="mmprojPath"/> and builds the CPU encoder its projector type needs.</summary>
    /// <exception cref="NotSupportedException">The projector type has no encoder yet.</exception>
    /// <param name="mmprojPath">The projector GGUF.</param>
    /// <param name="threading">Thread configuration.</param>
    /// <param name="overLimit">What happens to an image over the token limit (default: refuse).</param>
    /// <param name="embedPrecision">Storage format of the patch / position embeddings; null reads <c>DOTLLM_VISION_EMBD_PRECISION</c> (default BF16, <see cref="Inkling2VisionEncoder.DefaultEmbedPrecision"/>).</param>
    public static IImageEncoder LoadEncoder(string mmprojPath, ThreadingConfig threading,
        ImageOverLimit overLimit = ImageOverLimit.Refuse, VisionEmbedPrecision? embedPrecision = null)
    {
        var gguf = GgufFile.Open(mmprojPath);
        try
        {
            string projector = gguf.Metadata.GetStringOrDefault("clip.vision.projector_type");
            if (projector == "inkling2v") return Inkling2VisionEncoder.Load(gguf, threading, ownsGguf: true, overLimit, embedPrecision);
            throw new NotSupportedException($"mmproj projector type '{projector}' is not supported (supported: inkling2v).");
        }
        catch
        {
            gguf.Dispose();
            throw;
        }
    }
}
