using DotLLM.Core.Vision;

namespace DotLLM.Models.Vision;

/// <summary>
/// Image preprocessing of the Qwen3-VL projector (<c>qwen3vl_merger</c>; llama.cpp b12780 <c>mtmd_image_preprocessor_qwen3vl</c> =
/// <c>dyn_size</c>: <c>img_tool::calc_size_preserved_ratio</c> with half-even rounding, then a Pillow-compatible bicubic resize with no padding,
/// then <c>(v / 255 - 0.5) / 0.5</c>), issue #2073.
/// </summary>
/// <remarks>
/// The image is always resized to a multiple of <c>patch * merge = 32</c> on each side, with the pixel budget
/// <c>[8, 4096] tokens x 1024 px</c> (<c>set_limit_image_tokens(8, 4096)</c>): an image under the minimum is enlarged, one over the maximum is shrunk.
/// The size arithmetic is single-precision exactly as the reference. Shrinking an image that needs more than 4096 tokens changes what the model sees,
/// so, as for Inkling 2 (user ruling 2026-11-27), it is refused unless the policy is <see cref="ImageOverLimit.Downscale"/>; the grid rounding and the
/// enlargement of a tiny image are part of the model's preprocessing and always apply.
/// </remarks>
public static class QwenVlImagePreprocessor
{
    /// <summary>Patch size of the vision tower in pixels.</summary>
    public const int PatchSize = 16;

    /// <summary>Spatial merge factor of the projector.</summary>
    public const int Merge = 2;

    /// <summary>Largest number of image tokens accepted before the image is downscaled (or refused).</summary>
    public const int MaxTokens = 4096;

    /// <summary>Smallest number of image tokens: smaller images are enlarged to it.</summary>
    public const int MinTokens = 8;

    private const int Align = PatchSize * Merge;
    private const int PatchArea = Align * Align;
    private const long MinPixels = (long)MinTokens * PatchArea;
    private const long MaxPixels = (long)MaxTokens * PatchArea;

    /// <summary>The size the image is resized to (a multiple of 32 on each side), as <c>calc_size_preserved_ratio</c>.</summary>
    /// <exception cref="NotSupportedException">The image needs more than <see cref="MaxTokens"/> tokens at its own size and the policy is Refuse.</exception>
    public static (int Width, int Height) PlanSize(int width, int height, ImageOverLimit overLimit = ImageOverLimit.Refuse)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width), "image dimensions must be positive");
        static int RoundBy(float x) => (int)MathF.Round(x / Align, MidpointRounding.ToEven) * Align;
        int wBar = Math.Max(Align, RoundBy(width));
        int hBar = Math.Max(Align, RoundBy(height));
        if ((long)hBar * wBar > MaxPixels)
        {
            if (overLimit == ImageOverLimit.Refuse)
                throw new NotSupportedException(
                    $"image {width}x{height} needs {(long)(wBar / Align) * (hBar / Align)} image tokens, over the limit of {MaxTokens}; " +
                    "send a smaller image or enable downscaling (server --image-over-limit downscale / DOTLLM_IMAGE_OVER_LIMIT=downscale).");
            float beta = MathF.Sqrt((float)height * width / MaxPixels);
            hBar = Math.Max(Align, (int)MathF.Floor(height / beta / Align) * Align);
            wBar = Math.Max(Align, (int)MathF.Floor(width / beta / Align) * Align);
        }
        else if ((long)hBar * wBar < MinPixels)
        {
            float beta = MathF.Sqrt((float)MinPixels / ((float)height * width));
            hBar = (int)MathF.Ceiling(height * beta / Align) * Align;
            wBar = (int)MathF.Ceiling(width * beta / Align) * Align;
        }
        long tokens = (long)(wBar / Align) * (hBar / Align);
        if (tokens > MaxTokens)
            throw new NotSupportedException($"image aspect ratio too extreme: {width}x{height} would need {tokens} image tokens (limit {MaxTokens}).");
        return (wBar, hBar);
    }

    /// <summary>Tokens (merged patches) the image occupies in the prompt.</summary>
    public static int CountTokens(int width, int height, ImageOverLimit overLimit = ImageOverLimit.Refuse)
    {
        var (w, h) = PlanSize(width, height, overLimit);
        return w / Align * (h / Align);
    }

    /// <summary>
    /// Writes the resized, normalised image as interleaved float RGB (<c>HWC</c>) of the planned size into <paramref name="dest"/>
    /// (<c>W * H * 3</c> floats). Returns the size.
    /// </summary>
    public static (int Width, int Height) Normalize(DecodedImage image, Span<float> dest, bool swapRgb = false, ImageOverLimit overLimit = ImageOverLimit.Refuse)
    {
        var (w, h) = PlanSize(image.Width, image.Height, overLimit);
        if (dest.Length < (long)w * h * 3) throw new ArgumentException("dest too small", nameof(dest));
        byte[] pixels = w == image.Width && h == image.Height ? image.Rgb : ImageResampler.ResizeBicubic(image.Rgb, image.Width, image.Height, w, h);
        const float mean = 0.5f * 255.0f, std = 0.5f * 255.0f;
        for (int i = 0; i < w * h * 3; i++) dest[i] = (pixels[i] - mean) / std;
        if (swapRgb)
            for (int i = 0; i < w * h; i++) (dest[i * 3], dest[i * 3 + 2]) = (dest[i * 3 + 2], dest[i * 3]);
        return (w, h);
    }
}
