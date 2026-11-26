using DotLLM.Core.Vision;

namespace DotLLM.Models.Vision;

/// <summary>
/// Image preprocessing of Inkling-2 (llama.cpp b12780 <c>mtmd_image_preprocessor_inkling2v</c>, <c>tools/mtmd/mtmd-image.cpp</c>).
/// </summary>
/// <remarks>
/// Pixels are NOT resized to a fixed square: the image keeps its size and is padded at the right/bottom up to a multiple of
/// <c>patch_size * merge</c> (40). The padded area is initialised to <b>-1.0 in raw pixel units</b> before the fused normalisation
/// <c>(v - mean*255) / (std*255)</c>, so it comes out at -1.0078, not -1.0 (the reference's own quirk, kept for parity). Only an image whose padded
/// area exceeds <c>image_max_pixels = 4096 tokens * 1600</c> is resized (to bound the token count); the reference uses bicubic there, this
/// implementation an area-average, so over-limit images are NOT parity-gated (documented in <c>docs/INKLING2.md</c>).
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

    private static int AlignUp(int v) => (v + Align - 1) / Align * Align;

    /// <summary>The padded size the image is encoded at, after the over-limit downscale if any.</summary>
    public static (int Width, int Height, int SrcWidth, int SrcHeight) PlanSize(int width, int height)
    {
        bool tooLarge = (long)AlignUp(width) * AlignUp(height) > MaxPixels;
        int sw = width, sh = height;
        if (tooLarge)
        {
            // calc_size_preserved_ratio: round to the grid, then bound by max_pixels preserving the aspect ratio.
            int wBar = Math.Max(Align, (int)MathF.Round(width / (float)Align) * Align);
            int hBar = Math.Max(Align, (int)MathF.Round(height / (float)Align) * Align);
            if ((long)hBar * wBar > MaxPixels)
            {
                float beta = MathF.Sqrt((float)height * width / MaxPixels);
                hBar = Math.Max(Align, (int)MathF.Floor(height / beta / Align) * Align);
                wBar = Math.Max(Align, (int)MathF.Floor(width / beta / Align) * Align);
            }
            sw = wBar; sh = hBar;
        }
        return (AlignUp(sw), AlignUp(sh), sw, sh);
    }

    /// <summary>Tokens (merged patches) the image occupies in the prompt.</summary>
    public static int CountTokens(int width, int height)
    {
        var (w, h, _, _) = PlanSize(width, height);
        return w / Align * (h / Align);
    }

    /// <summary>
    /// Writes the normalised image as interleaved float RGB (<c>HWC</c>, the layout of <c>mtmd_image_f32</c>) of the padded size into
    /// <paramref name="dest"/> (<c>W * H * 3</c> floats, caller-provided native or stack scratch). Returns the padded size.
    /// </summary>
    public static (int Width, int Height) Normalize(DecodedImage image, Span<float> dest, bool swapRgb = false, float padRaw = PadRaw)
    {
        var (w, h, sw, sh) = PlanSize(image.Width, image.Height);
        if (dest.Length < (long)w * h * 3) throw new ArgumentException("dest too small", nameof(dest));
        const float mean = 0.5f * 255.0f, std = 0.5f * 255.0f;
        float pad = (padRaw - mean) / std;
        dest.Slice(0, w * h * 3).Fill(pad);
        ReadOnlySpan<byte> src = image.Rgb;
        if (sw == image.Width && sh == image.Height)
        {
            for (int y = 0; y < sh; y++)
            {
                var row = dest.Slice(y * w * 3, sw * 3);
                var s = src.Slice(y * sw * 3, sw * 3);
                for (int i = 0; i < row.Length; i++) row[i] = (s[i] - mean) / std;
            }
        }
        else
        {
            // Area-average downscale (over-limit images only).
            for (int y = 0; y < sh; y++)
            {
                int y0 = (int)((long)y * image.Height / sh), y1 = Math.Max(y0 + 1, (int)((long)(y + 1) * image.Height / sh));
                for (int x = 0; x < sw; x++)
                {
                    int x0 = (int)((long)x * image.Width / sw), x1 = Math.Max(x0 + 1, (int)((long)(x + 1) * image.Width / sw));
                    for (int c = 0; c < 3; c++)
                    {
                        float sum = 0; int n = 0;
                        for (int yy = y0; yy < y1; yy++)
                            for (int xx = x0; xx < x1; xx++) { sum += src[(yy * image.Width + xx) * 3 + c]; n++; }
                        dest[(y * w + x) * 3 + c] = (sum / n - mean) / std;
                    }
                }
            }
        }
        if (swapRgb)
            for (int i = 0; i < w * h; i++) (dest[i * 3], dest[i * 3 + 2]) = (dest[i * 3 + 2], dest[i * 3]);
        return (w, h);
    }
}
