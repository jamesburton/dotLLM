using System.Security.Cryptography;
using DotLLM.Cli.Commands;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Image limits are configurable and the over-limit policy is refuse (default) or downscale (user ruling 2026-11-27, #2035). Each test names
/// the mutant it kills; the ones run for the PR body are listed in docs/INKLING2.md.
/// </summary>
public sealed class Inkling2VisionLimitsTests
{
    private static byte[] Ppm(int w, int h) => [.. System.Text.Encoding.ASCII.GetBytes($"P6\n{w} {h}\n255\n"), .. new byte[checked(w * h * 3)]];

    private static string Uri(byte[] ppm) => "data:image/x-portable-pixmap;base64," + Convert.ToBase64String(ppm);

    // ───────────── defaults and resolution ─────────────

    /// <summary>The reviewer's defaults, accepted by the user. Kills: a changed default constant.</summary>
    [Fact]
    public void Defaults_AreTheAcceptedValues()
    {
        var d = ImageLimits.Default;
        Assert.Equal(16, d.MaxImagesPerRequest);
        Assert.Equal(1L << 26, d.MaxDecodedPixelsPerRequest);
        Assert.Equal(1L << 25, d.MaxPixelsPerImage);
        Assert.Equal(65536, d.MaxSidePerImage);
        Assert.Equal(ImageOverLimit.Refuse, d.OverLimit);
        Assert.Equal(d, ImageLimits.Resolve(env: _ => null));
    }

    /// <summary>The environment variables set each bound; an explicit (flag) value beats the variable; a typo is an error, never a silent default.</summary>
    [Fact]
    public void Resolve_EnvironmentVariables_FlagWins_AndTyposAreErrors()
    {
        var env = new Dictionary<string, string>
        {
            [ImageLimits.MaxImagesEnvVar] = "3",
            [ImageLimits.MaxRequestPixelsEnvVar] = "1000",
            [ImageLimits.MaxPixelsEnvVar] = "500",
            [ImageLimits.MaxSideEnvVar] = "77",
            [ImageLimits.OverLimitEnvVar] = "Downscale",
        };
        string? Get(string k) => env.GetValueOrDefault(k);
        var r = ImageLimits.Resolve(env: Get);
        Assert.Equal((3, 1000L, 500L, 77, ImageOverLimit.Downscale), (r.MaxImagesPerRequest, r.MaxDecodedPixelsPerRequest, r.MaxPixelsPerImage, r.MaxSidePerImage, r.OverLimit));

        var flag = ImageLimits.Resolve(maxImages: 9, overLimit: "refuse", env: Get);
        Assert.Equal(9, flag.MaxImagesPerRequest);
        Assert.Equal(ImageOverLimit.Refuse, flag.OverLimit);
        Assert.Equal(77, flag.MaxSidePerImage);   // the unset flags still fall back to the variable

        env[ImageLimits.MaxImagesEnvVar] = "lots";
        Assert.Throws<ArgumentException>(() => ImageLimits.Resolve(env: Get));
        env[ImageLimits.MaxImagesEnvVar] = "0";
        Assert.Throws<ArgumentException>(() => ImageLimits.Resolve(env: Get));
        env[ImageLimits.MaxImagesEnvVar] = "3";
        env[ImageLimits.OverLimitEnvVar] = "shrink";
        Assert.Throws<ArgumentException>(() => ImageLimits.Resolve(env: Get));
        Assert.Throws<ArgumentException>(() => ImageLimits.Resolve(maxSide: -1, env: _ => null));
    }

    /// <summary>The serve flags reach <see cref="ServerOptions.ImageLimits"/> (and the env fallback does too).</summary>
    [Fact]
    public void Serve_FlagsAndEnvironment_ReachTheServerOptions()
    {
        var settings = new ServeCommand.Settings { MaxImages = 4, MaxRequestImagePixels = 123456, MaxImagePixels = 7777, MaxImageSide = 999, ImageOverLimit = "downscale" };
        var limits = ServeCommand.ResolveImageLimits(settings, _ => null);
        Assert.Equal((4, 123456L, 7777L, 999, ImageOverLimit.Downscale),
            (limits.MaxImagesPerRequest, limits.MaxDecodedPixelsPerRequest, limits.MaxPixelsPerImage, limits.MaxSidePerImage, limits.OverLimit));
        Assert.Equal(limits, ServeCommand.BuildServerOptions(settings, null).ImageLimits);
        Assert.Equal(ImageOverLimit.Downscale, ServeCommand.ResolveImageLimits(new ServeCommand.Settings(), k => k == ImageLimits.OverLimitEnvVar ? "downscale" : null).OverLimit);
        Assert.Equal(ImageLimits.Default, ServerOptions.Parse(["--model", "x.gguf"]).ImageLimits);
        Assert.Equal(5, ServerOptions.Parse(["--model", "x.gguf", "--max-images", "5"]).ImageLimits.MaxImagesPerRequest);
        Assert.Equal(ImageOverLimit.Downscale, ServerOptions.Parse(["--model", "x.gguf", "--image-over-limit", "downscale"]).ImageLimits.OverLimit);
    }

    // ───────────── the decoder honours the limits ─────────────

    /// <summary>
    /// Lowering a limit refuses what the default accepts, raising one accepts what the default refuses. Kills: a decoder that still reads the
    /// constants (<c>ImageDecoder.MaxPixels</c> / <c>MaxSide</c>) instead of the passed limits.
    /// </summary>
    [Fact]
    public void Decoder_HonoursPerImagePixelAndSideLimits_BothDirections()
    {
        var ok = Ppm(4, 4);
        Assert.Equal(16, ImageDecoder.Decode(ok).Width * 4);
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(ok, new ImageLimits { MaxPixelsPerImage = 15 }));
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(ok, new ImageLimits { MaxSidePerImage = 3 }));
        Assert.NotNull(ImageDecoder.Decode(ok, new ImageLimits { MaxPixelsPerImage = 16, MaxSidePerImage = 4 }));

        var wide = Ppm(70000, 1);                    // wider than the default 65,536 side limit
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(wide));
        Assert.Equal(70000, ImageDecoder.Decode(wide, new ImageLimits { MaxSidePerImage = 70000 }).Width);
    }

    /// <summary>The per-request bounds: image count and total decoded pixels, both configurable and both with the option named in the error.</summary>
    [Fact]
    public void ServerDecode_HonoursConfiguredCountAndRequestPixelLimits()
    {
        string a = Uri(Ppm(10, 10));
        Assert.Equal(3, ImageInput.DecodeAll([a, a, a], new ImageLimits { MaxImagesPerRequest = 3 }).Count);
        var tooMany = Assert.Throws<ImageInputException>(() => ImageInput.DecodeAll([a, a, a], new ImageLimits { MaxImagesPerRequest = 2 }));
        Assert.Contains("--max-images", tooMany.Message, StringComparison.Ordinal);
        Assert.Equal(2, ImageInput.DecodeAll([a, a], new ImageLimits { MaxDecodedPixelsPerRequest = 200 }).Count);
        Assert.Throws<ImageInputException>(() => ImageInput.DecodeAll([a, a], new ImageLimits { MaxDecodedPixelsPerRequest = 199 }));
        // the default is 16 images: raising it admits a 17th
        Assert.Throws<ImageInputException>(() => ImageInput.DecodeAll(Enumerable.Repeat(a, 17).ToList()));
        Assert.Equal(17, ImageInput.DecodeAll(Enumerable.Repeat(a, 17).ToList(), new ImageLimits { MaxImagesPerRequest = 17 }).Count);
        // the per-image limit reaches the decoder through the request path too
        Assert.Throws<ImageInputException>(() => ImageInput.DecodeAll([a], new ImageLimits { MaxPixelsPerImage = 99 }));
    }

    // ───────────── over-limit: refuse or downscale ─────────────

    /// <summary>
    /// The default refuses an image over 4,096 tokens and says how to allow it; downscale fits it (and nothing at or under the limit is touched by
    /// either policy). Kills: a refuse that downscales anyway (the pre-ruling behaviour), and a downscale that refuses.
    /// </summary>
    [Fact]
    public void OverLimit_RefuseIsTheDefault_DownscaleFitsTheImage()
    {
        foreach (var (w, h) in new[] { (4000, 4000), (8000, 900), (600, 12000) })
        {
            var ex = Assert.Throws<NotSupportedException>(() => Inkling2ImagePreprocessor.CountTokens(w, h));
            Assert.Contains("downscale", ex.Message, StringComparison.Ordinal);
            Assert.Throws<NotSupportedException>(() => Inkling2ImagePreprocessor.PlanSize(w, h, ImageOverLimit.Refuse));
            Assert.InRange(Inkling2ImagePreprocessor.CountTokens(w, h, ImageOverLimit.Downscale), 1, Inkling2ImagePreprocessor.MaxTokens);
        }
        // exactly at the limit (64 x 64 cells = 4096 tokens): accepted by both, unchanged
        Assert.Equal(4096, Inkling2ImagePreprocessor.CountTokens(2560, 2560));
        Assert.Equal(4096, Inkling2ImagePreprocessor.CountTokens(2560, 2560, ImageOverLimit.Downscale));
        Assert.Equal((2560, 2560, 2560, 2560), Inkling2ImagePreprocessor.PlanSize(2560, 2560, ImageOverLimit.Downscale));
        // an extreme aspect ratio cannot be downscaled to fit either: refused under both policies
        Assert.Throws<NotSupportedException>(() => Inkling2ImagePreprocessor.CountTokens(1_000_000, 1, ImageOverLimit.Downscale));
    }

    /// <summary>
    /// The downscaled pixels are the reference's: <c>Normalize</c> resizes 8-bit RGB with the Pillow-compatible bicubic and then applies the
    /// normal normalisation and padding. Kills: an area-average or nearest resize (the pre-review implementation was an area-average).
    /// </summary>
    [Fact]
    public void Downscale_ResizesWithBicubic_ThenNormalisesAndPads()
    {
        const int sw = 2600, sh = 2600;                     // 65 x 65 cells = 4225 tokens: over the limit
        var rgb = new byte[sw * sh * 3];
        var rng = new Random(7);
        rng.NextBytes(rgb);   // noise: a linear gradient would make the area average equal the bicubic
        var img = new DecodedImage(sw, sh, rgb);
        var (w, h, tw, th) = Inkling2ImagePreprocessor.PlanSize(sw, sh, ImageOverLimit.Downscale);
        Assert.True(tw * th < sw * sh);
        var buf = new float[(long)w * h * 3];
        Inkling2ImagePreprocessor.Normalize(img, buf, overLimit: ImageOverLimit.Downscale);

        var expected = ImageResampler.ResizeBicubic(rgb, sw, sh, tw, th);
        for (int y = 0; y < th; y += 37)
            for (int x = 0; x < tw; x += 41)
                for (int c = 0; c < 3; c++)
                    Assert.Equal((expected[(y * tw + x) * 3 + c] - 127.5f) / 127.5f, buf[(y * w + x) * 3 + c]);
        // and it is not an area average of the source
        int cx = tw / 2, cy = th / 2;
        float area = 0; int n = 0;
        int x0 = cx * sw / tw, x1 = (cx + 1) * sw / tw, y0 = cy * sh / th, y1 = (cy + 1) * sh / th;
        for (int yy = y0; yy < y1; yy++) for (int xx = x0; xx < x1; xx++) { area += rgb[(yy * sw + xx) * 3]; n++; }
        float areaNorm = (area / n - 127.5f) / 127.5f;
        float bicubic = (expected[(cy * tw + cx) * 3] - 127.5f) / 127.5f;
        Assert.Equal(bicubic, buf[(cy * w + cx) * 3]);
        Assert.NotEqual(areaNorm, buf[(cy * w + cx) * 3]);
    }

    /// <summary>
    /// The resampler is Pillow's, byte for byte: the expected digests are <c>PIL.Image.resize(size, Image.BICUBIC)</c> (Pillow 12.2) of a
    /// deterministic LCG image, generated by <c>tools/inkling2-vision-oracle/pillow_bicubic_golden.py</c>. Covers down-, up- and mixed scaling. Kills: a=-0.75 (GGML's
    /// cubic), a missing filter-support widening, vertical-before-horizontal fixed-point rounding, and any change of the weight rounding.
    /// </summary>
    [Theory]
    [InlineData(97, 61, 40, 24, "f2d3cfc3318dda084c626e1e1691430aa5a56c588f3db7c09e93d3e58b618fad")]
    [InlineData(61, 97, 80, 120, "f6b456f329745dc6de042efff797b6c504995c5634f37cda27227ef7d5fbba3f")]
    [InlineData(130, 50, 40, 90, "d6cfeb6c5f2a3c1299f0cf051d889c4827df309326e3d197524c1887ec34fe5a")]
    [InlineData(2500, 2500, 1240, 1240, "c5e80dabaf0329227eaa88966fea0ed80c180cce88917985d27cf78823cf53b8")]
    [InlineData(300, 41, 300, 20, "15d0627f12eb0cc75c2e876ccefea18800c427c041c05d00650576ec2dfc4fb9")]
    public void Bicubic_IsByteIdenticalToPillow(int sw, int sh, int tw, int th, string sha256)
    {
        var src = new byte[sw * sh * 3];
        int x = 12345;
        for (int i = 0; i < src.Length; i++) { x = (int)(((long)x * 1103515245 + 12345) & 0x7fffffff); src[i] = (byte)((x >> 16) & 255); }
        var dst = ImageResampler.ResizeBicubic(src, sw, sh, tw, th);
        Assert.Equal(sha256, Convert.ToHexStringLower(SHA256.HashData(dst)));
    }

    /// <summary>The preprocessor is the only place the policy is decided: the encoder's token count follows it (see also the real-model test).</summary>
    [Fact]
    public void ImageLimits_OverLimitEnum_HasExactlyTheTwoPolicies()
    {
        Assert.Equal(["Refuse", "Downscale"], Enum.GetNames<ImageOverLimit>());
    }
}
