using DotLLM.Core.Vision;

namespace DotLLM.Models.Vision;

/// <summary>
/// Image preprocessing of Inkling-2 (llama.cpp b12780 <c>mtmd_image_preprocessor_inkling2v</c>, <c>tools/mtmd/mtmd-image.cpp</c>).
/// </summary>
/// <remarks>
/// Pixels are NOT resized to a fixed square: the image keeps its size and is padded at the right/bottom up to a multiple of
/// <c>patch_size * merge</c> (40). The padded area is initialised to <b>-1.0 in raw pixel units</b> before the fused normalisation
/// <c>(v - mean*255) / (std*255)</c>, so it comes out at -1.0078, not -1.0 (the reference's own quirk, kept for parity). Only an image whose padded
/// area exceeds <c>image_max_pixels = 4096 tokens * 1600</c> is resized (to bound the token count), by a Pillow-compatible bicubic as in the
/// reference (<see cref="ImageResampler"/>). Whether such an image is resized or refused is the caller's choice (<see cref="ImageOverLimit"/>; user
/// ruling 2026-11-27: refuse is the default, so nothing is silently altered).
/// </remarks>
public static class Inkling2ImagePreprocessor
{
    /// <summary>Patch size of the vision tower in pixels.</summary>
    public const int PatchSize = 20;

    /// <summary>Spatial merge factor of the adapter.</summary>
    public const int Merge = 2;

    /// <summary>Largest number of image tokens (embedding rows) accepted before the image is downscaled.</summary>
    public const int MaxTokens = 4096;

    /// <summary>Pad value in raw pixel units (before normalisation).</summary>
    public const float PadRaw = -1.0f;

    private const int Align = PatchSize * Merge;
    private const long MaxPixels = (long)MaxTokens * Align * Align;

    private static int AlignUp(int v) => (int)Math.Min(int.MaxValue / 2, ((long)v + Align - 1) / Align * Align);

    /// <summary>The padded size the image is encoded at, after the over-limit downscale if the policy allows it.</summary>
    /// <exception cref="NotSupportedException">The image needs more than <see cref="MaxTokens"/> tokens and <paramref name="overLimit"/> is <see cref="ImageOverLimit.Refuse"/>
    /// (or its aspect ratio is so extreme that no downscale fits).</exception>
    public static (int Width, int Height, int SrcWidth, int SrcHeight) PlanSize(int width, int height, ImageOverLimit overLimit = ImageOverLimit.Refuse)
    {
        bool tooLarge = (long)AlignUp(width) * AlignUp(height) > MaxPixels;
        int sw = width, sh = height;
        if (tooLarge && overLimit == ImageOverLimit.Refuse)
            throw new NotSupportedException(
                $"image {width}x{height} needs {(long)(AlignUp(width) / Align) * (AlignUp(height) / Align)} image tokens, over the limit of {MaxTokens}; " +
                "send a smaller image or enable downscaling (server --image-over-limit downscale / DOTLLM_IMAGE_OVER_LIMIT=downscale).");
        if (tooLarge)
        {
            // calc_size_preserved_ratio: round to the grid, then bound by max_pixels preserving the aspect ratio.
            // std::round (half away from zero), not MathF.Round's default half-to-even: the reference rounds 4020 / 40 = 100.5 up.
            int wBar = Math.Max(Align, (int)MathF.Round(width / (float)Align, MidpointRounding.AwayFromZero) * Align);
            int hBar = Math.Max(Align, (int)MathF.Round(height / (float)Align, MidpointRounding.AwayFromZero) * Align);
            if ((long)hBar * wBar > MaxPixels)
            {
                float beta = MathF.Sqrt((float)height * width / MaxPixels);
                hBar = Math.Max(Align, (int)MathF.Floor(height / beta / Align) * Align);
                wBar = Math.Max(Align, (int)MathF.Floor(width / beta / Align) * Align);
            }
            sw = wBar; sh = hBar;
        }
        // An extreme aspect ratio defeats the bound: the short side clamps to one cell and the long side keeps growing (a 2,000,000 x 1 image
        // plans 90,000 tokens). The reference would encode it; this refuses it, because the ViT cost is quadratic in the token count.
        long tokens = (long)(AlignUp(sw) / Align) * (AlignUp(sh) / Align);
        if (tokens > MaxTokens)
            throw new NotSupportedException($"image aspect ratio too extreme: {width}x{height} would need {tokens} image tokens (limit {MaxTokens}).");
        return (AlignUp(sw), AlignUp(sh), sw, sh);
    }

    /// <summary>Tokens (merged patches) the image occupies in the prompt.</summary>
    public static int CountTokens(int width, int height, ImageOverLimit overLimit = ImageOverLimit.Refuse)
    {
        var (w, h, _, _) = PlanSize(width, height, overLimit);
        return w / Align * (h / Align);
    }

    /// <summary>
    /// Writes the normalised image as interleaved float RGB (<c>HWC</c>, the layout of <c>mtmd_image_f32</c>) of the padded size into
    /// <paramref name="dest"/> (<c>W * H * 3</c> floats, caller-provided native or stack scratch). Returns the padded size.
    /// </summary>
    public static (int Width, int Height) Normalize(DecodedImage image, Span<float> dest, bool swapRgb = false, float padRaw = PadRaw, ImageOverLimit overLimit = ImageOverLimit.Refuse)
    {
        var (w, h, sw, sh) = PlanSize(image.Width, image.Height, overLimit);
        if (dest.Length < (long)w * h * 3) throw new ArgumentException("dest too small", nameof(dest));
        const float mean = 0.5f * 255.0f, std = 0.5f * 255.0f;
        float pad = (padRaw - mean) / std;
        dest.Slice(0, w * h * 3).Fill(pad);
        // Over the token limit (downscale policy only): the reference's resize, in 8-bit, before the fused normalisation.
        byte[] pixels = sw == image.Width && sh == image.Height ? image.Rgb : ImageResampler.ResizeBicubic(image.Rgb, image.Width, image.Height, sw, sh);
        ReadOnlySpan<byte> src = pixels;
        for (int y = 0; y < sh; y++)
        {
            var row = dest.Slice(y * w * 3, sw * 3);
            var s = src.Slice(y * sw * 3, sw * 3);
            for (int i = 0; i < row.Length; i++) row[i] = (s[i] - mean) / std;
        }
        if (swapRgb)
            for (int i = 0; i < w * h; i++) (dest[i * 3], dest[i * 3 + 2]) = (dest[i * 3 + 2], dest[i * 3]);
        return (w, h);
    }
}
