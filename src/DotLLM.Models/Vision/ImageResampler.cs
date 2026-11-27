namespace DotLLM.Models.Vision;

/// <summary>
/// Pillow-compatible separable bicubic resampling of 8-bit RGB, a port of llama.cpp's <c>img_tool::resize_pillow</c>
/// (<c>tools/mtmd/mtmd-image.cpp</c>, b12780 / <c>09c141064</c>), which is what <c>mtmd_image_preprocessor_inkling2v</c> calls for an image over the
/// token limit (<c>hparams.image_resize_algo</c> defaults to bicubic and Inkling 2 does not change it). Cubic convolution with <c>a = -0.5</c> (Pillow's,
/// not GGML's -0.75), a horizontal pass then a vertical pass, filter support widened by the scale when downsampling, weights in 22-bit fixed point.
/// Output equals <c>PIL.Image.resize(size, Image.BICUBIC)</c> byte for byte (checked against Pillow in the tests).
/// </summary>
internal static class ImageResampler
{
    private const int PrecisionBits = 32 - 8 - 2;   // 8-bit pixels, 2 bits of accumulation headroom
    // Largest target side. A downscale plan is bounded by its token count, not by the source side: one side clamps to a single 40-px cell and the
    // other may reach MaxTokens * 40 = 163,840 px (e.g. 300000x80 under a raised --max-image-side plans 156760x40). The bound must admit every
    // plan, or the encode throws ArgumentOutOfRangeException after the token check passed (a 500 on the server; review of #2037).
    internal const int MaxTarget = 1 << 18;

    /// <summary>Resizes interleaved RGB <paramref name="src"/> (<c>sw x sh</c>) to <c>tw x th</c>.</summary>
    public static byte[] ResizeBicubic(byte[] src, int sw, int sh, int tw, int th)
    {
        if (tw <= 0 || tw > MaxTarget || th <= 0 || th > MaxTarget) throw new ArgumentOutOfRangeException(nameof(tw), $"resize target {tw}x{th} is out of range (max {MaxTarget})");
        if (src.LongLength != (long)sw * sh * 3) throw new ArgumentException("src must hold sw * sh * 3 bytes", nameof(src));
        bool horizontal = tw != sw, vertical = th != sh;
        if (!horizontal && !vertical) return (byte[])src.Clone();

        byte[] cur = src;
        int curW = sw;
        if (horizontal)
        {
            int ksize = Weights(sw, tw, out int[] bounds, out int[] weights);
            cur = Horizontal(cur, curW, sh, tw, ksize, bounds, weights);
            curW = tw;
        }
        if (vertical)
        {
            int ksize = Weights(sh, th, out int[] bounds, out int[] weights);
            cur = Vertical(cur, curW, th, ksize, bounds, weights);
        }
        return cur;
    }

    private static double Filter(double x)
    {
        const double a = -0.5;
        if (x < 0.0) x = -x;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    }

    /// <summary>Per output pixel the first input pixel and the count (<paramref name="bounds"/>) and the fixed-point weights; returns the kernel size.</summary>
    private static int Weights(int inSize, int outSize, out int[] bounds, out int[] weights)
    {
        double scale = (double)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);   // widen the filter when downsampling, keep it sharp when upsampling
        double support = 2.0 * filterScale;
        int ksize = (int)Math.Ceiling(support) * 2 + 1;
        var pre = new double[(long)outSize * ksize];
        bounds = new int[outSize * 2];
        double ss = 1.0 / filterScale;
        for (int xx = 0; xx < outSize; xx++)
        {
            double center = (xx + 0.5) * scale;
            int xmin = Math.Max(0, (int)(center - support + 0.5));
            int xmax = Math.Min(inSize, (int)(center + support + 0.5));
            xmax -= xmin;
            double ww = 0.0;
            long row = (long)xx * ksize;
            for (int x = 0; x < xmax; x++)
            {
                double w = Filter((x + xmin - center + 0.5) * ss);
                pre[row + x] = w;
                ww += w;
            }
            for (int x = 0; x < xmax; x++)
                if (ww != 0.0) pre[row + x] /= ww;
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        weights = new int[pre.Length];
        double fxp = Math.ScaleB(1.0, PrecisionBits);
        for (int i = 0; i < pre.Length; i++)
        {
            // Pillow adds +/- 0.5 then truncates toward zero (a single rounding).
            double rounded = pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5);
            weights[i] = (int)rounded;
        }
        return ksize;
    }

    private static byte Clip8(int v) => v < 0 ? (byte)0 : v > 255 ? (byte)255 : (byte)v;

    private static byte[] Horizontal(byte[] src, int inW, int h, int outW, int ksize, int[] bounds, int[] weights)
    {
        var dst = new byte[(long)outW * h * 3];
        for (int y = 0; y < h; y++)
        {
            long srcRow = (long)y * inW * 3, dstRow = (long)y * outW * 3;
            for (int xx = 0; xx < outW; xx++)
            {
                int xmin = bounds[xx * 2], cnt = bounds[xx * 2 + 1];
                int k0 = xx * ksize;
                int s0 = 1 << (PrecisionBits - 1), s1 = s0, s2 = s0;
                long p = srcRow + (long)xmin * 3;
                for (int x = 0; x < cnt; x++, p += 3)
                {
                    int w = weights[k0 + x];
                    s0 += src[p] * w; s1 += src[p + 1] * w; s2 += src[p + 2] * w;
                }
                dst[dstRow + xx * 3] = Clip8(s0 >> PrecisionBits);
                dst[dstRow + xx * 3 + 1] = Clip8(s1 >> PrecisionBits);
                dst[dstRow + xx * 3 + 2] = Clip8(s2 >> PrecisionBits);
            }
        }
        return dst;
    }

    private static byte[] Vertical(byte[] src, int w, int outH, int ksize, int[] bounds, int[] weights)
    {
        long rowElems = (long)w * 3;
        var dst = new byte[rowElems * outH];
        var acc = new int[rowElems];
        for (int yy = 0; yy < outH; yy++)
        {
            int ymin = bounds[yy * 2], cnt = bounds[yy * 2 + 1];
            Array.Fill(acc, 1 << (PrecisionBits - 1));
            for (int y = 0; y < cnt; y++)
            {
                long srcRow = (long)(ymin + y) * rowElems;
                int wt = weights[yy * ksize + y];
                for (long i = 0; i < rowElems; i++) acc[i] += src[srcRow + i] * wt;
            }
            long dstRow = yy * rowElems;
            for (long i = 0; i < rowElems; i++) dst[dstRow + i] = Clip8(acc[i] >> PrecisionBits);
        }
        return dst;
    }
}
