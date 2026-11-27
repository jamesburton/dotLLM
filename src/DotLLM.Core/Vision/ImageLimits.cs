using System.Globalization;

namespace DotLLM.Core.Vision;

/// <summary>What to do with an image whose planned size needs more image tokens than the model's limit (Inkling 2: 4,096).</summary>
public enum ImageOverLimit
{
    /// <summary>Refuse the image (HTTP 400 on the server, an error on the CLI). The default: nothing is silently altered.</summary>
    Refuse,

    /// <summary>
    /// Downscale it to fit, as llama.cpp's mtmd does (<c>calc_size_preserved_ratio</c>, then a Pillow-compatible bicubic resize). The model then
    /// sees a smaller image than the one sent, so this is opt-in.
    /// </summary>
    Downscale,
}

/// <summary>
/// The bounds on image input (user ruling 2026-11-27, #2035: the reviewer's limits are the defaults, all configurable). Images are untrusted input;
/// these bound the memory and time one request can cost. Server: <c>dotllm serve --max-images ...</c> / <c>DOTLLM_IMAGE_*</c>; CLI <c>run</c> reads
/// the environment variables and <c>--image-over-limit</c>.
/// </summary>
public sealed record ImageLimits
{
    /// <summary>Environment variable: <see cref="MaxImagesPerRequest"/>.</summary>
    public const string MaxImagesEnvVar = "DOTLLM_IMAGE_MAX_IMAGES";

    /// <summary>Environment variable: <see cref="MaxDecodedPixelsPerRequest"/>.</summary>
    public const string MaxRequestPixelsEnvVar = "DOTLLM_IMAGE_MAX_REQUEST_PIXELS";

    /// <summary>Environment variable: <see cref="MaxPixelsPerImage"/>.</summary>
    public const string MaxPixelsEnvVar = "DOTLLM_IMAGE_MAX_PIXELS";

    /// <summary>Environment variable: <see cref="MaxSidePerImage"/>.</summary>
    public const string MaxSideEnvVar = "DOTLLM_IMAGE_MAX_SIDE";

    /// <summary>Environment variable: <see cref="OverLimit"/> (<c>refuse</c> or <c>downscale</c>).</summary>
    public const string OverLimitEnvVar = "DOTLLM_IMAGE_OVER_LIMIT";

    /// <summary>Most images one request may carry (default 16).</summary>
    public int MaxImagesPerRequest { get; init; } = 16;

    /// <summary>Most decoded pixels one request may hold at once (default 2^26: two 8K frames, or many ordinary photos).</summary>
    public long MaxDecodedPixelsPerRequest { get; init; } = 1L << 26;

    /// <summary>
    /// Largest decoded image, in pixels (default 2^25, about 33.5 MP: 8K UHD fits). Bounds the decode buffers: at most 8 bytes per pixel of
    /// filtered PNG rows plus 3 of RGB output.
    /// </summary>
    public long MaxPixelsPerImage { get; init; } = 1L << 25;

    /// <summary>Largest width or height of an image in pixels (default 65,536).</summary>
    public int MaxSidePerImage { get; init; } = 1 << 16;

    /// <summary>What happens to an image that needs more image tokens than the model's limit (default <see cref="ImageOverLimit.Refuse"/>).</summary>
    public ImageOverLimit OverLimit { get; init; } = ImageOverLimit.Refuse;

    /// <summary>The defaults.</summary>
    public static ImageLimits Default { get; } = new();

    /// <summary>Throws <see cref="ArgumentOutOfRangeException"/> when a bound is not positive.</summary>
    public ImageLimits Validated()
    {
        if (MaxImagesPerRequest < 1) throw new ArgumentOutOfRangeException(nameof(MaxImagesPerRequest), "must be >= 1");
        if (MaxDecodedPixelsPerRequest < 1) throw new ArgumentOutOfRangeException(nameof(MaxDecodedPixelsPerRequest), "must be >= 1");
        if (MaxPixelsPerImage < 1) throw new ArgumentOutOfRangeException(nameof(MaxPixelsPerImage), "must be >= 1");
        if (MaxSidePerImage < 1) throw new ArgumentOutOfRangeException(nameof(MaxSidePerImage), "must be >= 1");
        return this;
    }

    /// <summary>
    /// Resolves the limits from explicit values (a CLI flag) and the <c>DOTLLM_IMAGE_*</c> environment variables; an explicit value wins, an unset one
    /// falls back to the variable, then to the default. A malformed value is an <see cref="ArgumentException"/> (a typo must not silently keep the
    /// default and so leave a cap in force the operator thought they had changed).
    /// </summary>
    /// <param name="maxImages">Explicit <see cref="MaxImagesPerRequest"/>.</param>
    /// <param name="maxRequestPixels">Explicit <see cref="MaxDecodedPixelsPerRequest"/>.</param>
    /// <param name="maxPixels">Explicit <see cref="MaxPixelsPerImage"/>.</param>
    /// <param name="maxSide">Explicit <see cref="MaxSidePerImage"/>.</param>
    /// <param name="overLimit">Explicit <see cref="OverLimit"/>: <c>refuse</c> or <c>downscale</c>.</param>
    /// <param name="env">Reads an environment variable (null = <see cref="Environment.GetEnvironmentVariable(string)"/>); a seam for tests.</param>
    public static ImageLimits Resolve(int? maxImages = null, long? maxRequestPixels = null, long? maxPixels = null, int? maxSide = null,
        string? overLimit = null, Func<string, string?>? env = null)
    {
        env ??= Environment.GetEnvironmentVariable;
        return new ImageLimits
        {
            MaxImagesPerRequest = (int)Pick(maxImages, MaxImagesEnvVar, "--max-images", int.MaxValue, env),
            MaxDecodedPixelsPerRequest = Pick(maxRequestPixels, MaxRequestPixelsEnvVar, "--max-request-image-pixels", long.MaxValue, env, Default.MaxDecodedPixelsPerRequest),
            MaxPixelsPerImage = Pick(maxPixels, MaxPixelsEnvVar, "--max-image-pixels", long.MaxValue, env, Default.MaxPixelsPerImage),
            MaxSidePerImage = (int)Pick(maxSide, MaxSideEnvVar, "--max-image-side", int.MaxValue, env, Default.MaxSidePerImage),
            OverLimit = ParseOverLimit(overLimit ?? env(OverLimitEnvVar), overLimit is null ? OverLimitEnvVar : "--image-over-limit"),
        }.Validated();
    }

    private static long Pick(long? explicitValue, string envVar, string flag, long max, Func<string, string?> env, long? fallback = null)
    {
        fallback ??= Default.MaxImagesPerRequest;
        if (explicitValue is long v)
            return v >= 1 && v <= max ? v : throw new ArgumentException($"{flag} must be an integer from 1 to {max}, got {v}.");
        string? text = env(envVar);
        if (string.IsNullOrWhiteSpace(text)) return fallback.Value;
        if (!long.TryParse(text.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out long e) || e < 1 || e > max)
            throw new ArgumentException($"{envVar} must be an integer from 1 to {max}, got '{text}'.");
        return e;
    }

    private static ImageOverLimit ParseOverLimit(string? text, string source)
    {
        if (string.IsNullOrWhiteSpace(text)) return ImageOverLimit.Refuse;
        return text.Trim().ToLowerInvariant() switch
        {
            "refuse" => ImageOverLimit.Refuse,
            "downscale" => ImageOverLimit.Downscale,
            _ => throw new ArgumentException($"{source} must be 'refuse' or 'downscale', got '{text}'."),
        };
    }
}
