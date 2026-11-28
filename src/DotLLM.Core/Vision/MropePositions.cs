namespace DotLLM.Core.Vision;

/// <summary>
/// M-RoPE (multimodal rotary) position bookkeeping for Qwen-VL-style prompts (issue #2073), after llama.cpp mtmd
/// <c>mtmd_image_tokens_get_decoder_pos</c> and <c>mtmd_image_tokens_get_n_pos</c> (type <c>MTMD_POS_TYPE_MROPE</c>).
/// </summary>
/// <remarks>
/// <para>The KV cache is indexed by <i>slot</i> (the engine's position: one per token), but the rotary angle uses a <i>logical</i>
/// position on three axes (t, y, x) that does not advance one per token inside an image. For an image of merged grid
/// <c>gw x gh</c> that starts at slot <c>s</c> with logical base <c>b</c>, token <c>i</c> of the image has
/// <c>t = b</c>, <c>y = b + i / gw</c>, <c>x = b + i % gw</c>, and the text after the image resumes at <c>b + max(gw, gh)</c>.
/// A text token's three axes are equal. So after an image the logical position trails the slot by
/// <c>tokens - max(gw, gh)</c> (<see cref="Delta"/>), which the model keeps as the sequence's "rope delta" for the decode steps.</para>
/// <para>The image spans are in absolute slot positions and sorted by <see cref="ImageSpan.Start"/> (what
/// <see cref="ImagePlaceholders.Expand"/> returns), so any prefill chunk can be positioned without seeing the others.</para>
/// </remarks>
public static class MropePositions
{
    /// <summary>How far one image makes the logical position trail the slot index: <c>tokens - max(gridWidth, gridHeight)</c>.</summary>
    public static int Delta(ImageEmbeddings e) => e.Tokens - Math.Max(e.GridWidth, e.GridHeight);

    /// <summary>The total rope delta once every image of the request has been consumed (what the decode steps subtract from the slot).</summary>
    public static int TotalDelta(IReadOnlyList<ImageSpan> images)
    {
        int d = 0;
        for (int i = 0; i < images.Count; i++) d += Delta(images[i].Embeddings);
        return d;
    }

    /// <summary>
    /// Fills the three logical position axes of <paramref name="slots"/> (absolute slot positions, ascending or not).
    /// </summary>
    /// <param name="slots">Absolute slot positions of the rows being processed.</param>
    /// <param name="images">Image spans in absolute slots, sorted by start.</param>
    /// <param name="t">Temporal axis per row.</param>
    /// <param name="y">Height axis per row.</param>
    /// <param name="x">Width axis per row.</param>
    public static void Fill(ReadOnlySpan<int> slots, IReadOnlyList<ImageSpan> images, Span<int> t, Span<int> y, Span<int> x)
    {
        if (t.Length < slots.Length || y.Length < slots.Length || x.Length < slots.Length)
            throw new ArgumentException("axis buffers must hold one entry per slot.");
        for (int r = 0; r < slots.Length; r++)
        {
            int p = slots[r];
            int trailing = 0;           // delta of the images that end at or before p
            int tt = p, yy = p, xx = p;
            for (int i = 0; i < images.Count; i++)
            {
                var sp = images[i];
                if (p < sp.Start) break;
                var e = sp.Embeddings;
                if (p < sp.End)
                {
                    int b = sp.Start - trailing;
                    int idx = p - sp.Start;
                    tt = b;
                    yy = b + idx / e.GridWidth;
                    xx = b + idx % e.GridWidth;
                    trailing = int.MinValue;   // inside an image: handled
                    break;
                }
                trailing += Delta(e);
            }
            if (trailing != int.MinValue) { tt = yy = xx = p - trailing; }
            t[r] = tt; y[r] = yy; x[r] = xx;
        }
    }

    /// <summary>
    /// The default section split of the Qwen3.5 / Qwen3-VL text models (<c>{arch}.rope.dimension_sections</c> = [11, 11, 10, 0]) when the
    /// GGUF carries none.
    /// </summary>
    public static ReadOnlySpan<int> DefaultSections => [11, 11, 10, 0];

    /// <summary>
    /// Which position axis rotates each rotary pair under the <b>interleaved</b> M-RoPE (ggml <c>IMROPE</c>, <c>GGML_ROPE_TYPE_IMROPE</c>):
    /// pair <c>j</c> has <c>sector = j % sum(sections)</c>, then <c>sector % 3 == 1 &amp;&amp; sector &lt; 3*s1</c> uses y (axis 1),
    /// <c>sector % 3 == 2 &amp;&amp; sector &lt; 3*s2</c> uses x (axis 2), <c>sector % 3 == 0 &amp;&amp; sector &lt; 3*s0</c> uses t (axis 0),
    /// and anything else the fourth axis (axis 3, always 0 here).
    /// </summary>
    /// <param name="sections">The four section sizes.</param>
    /// <param name="pairs">Rotary pairs per head (<c>ropeDim / 2</c>).</param>
    public static byte[] AxisPerPair(ReadOnlySpan<int> sections, int pairs)
    {
        if (sections.Length != 4) throw new ArgumentException("M-RoPE needs 4 sections.", nameof(sections));
        int sum = sections[0] + sections[1] + sections[2] + sections[3];
        if (sum <= 0) throw new ArgumentException("M-RoPE sections must not all be zero.", nameof(sections));
        var axis = new byte[pairs];
        for (int j = 0; j < pairs; j++)
        {
            int sector = j % sum;
            axis[j] = sector % 3 == 1 && sector < 3 * sections[1] ? (byte)1
                    : sector % 3 == 2 && sector < 3 * sections[2] ? (byte)2
                    : sector % 3 == 0 && sector < 3 * sections[0] ? (byte)0
                    : (byte)3;
        }
        return axis;
    }
}
