namespace DotLLM.Core.Vision;

/// <summary>
/// Prompt-side image bookkeeping shared by every backend. The chat template renders ONE placeholder token per image (Inkling 2:
/// <c>&lt;|content_image|&gt;&lt;|unused_200054|&gt;</c>); the processor widens it to as many placeholder positions as the image has embedding rows
/// (HF <c>Inkling2Processor</c>), and the embeddings replace those positions' token embeddings.
/// </summary>
public static class ImagePlaceholders
{
    /// <summary>
    /// Replaces the first <paramref name="tokensPerImage"/>.Length occurrences of <paramref name="placeholderId"/> in <paramref name="ids"/> by
    /// <c>tokensPerImage[i]</c> copies. Returns the new ids and each image's start position in them.
    /// </summary>
    /// <exception cref="ArgumentException">The prompt has fewer placeholders than images, or more (an unmatched placeholder would be fed as a real token).</exception>
    public static (int[] Ids, int[] Starts) Expand(ReadOnlySpan<int> ids, int placeholderId, IReadOnlyList<int> tokensPerImage)
    {
        int found = 0;
        foreach (int id in ids) if (id == placeholderId) found++;
        if (found != tokensPerImage.Count)
            throw new ArgumentException($"the prompt has {found} image placeholder(s) but {tokensPerImage.Count} image(s) were supplied.");
        long extra = 0;
        foreach (int n in tokensPerImage) extra += n - 1;
        var result = new int[ids.Length + extra];
        var starts = new int[tokensPerImage.Count];
        int w = 0, img = 0;
        foreach (int id in ids)
        {
            if (id == placeholderId)
            {
                starts[img] = w;
                for (int k = 0; k < tokensPerImage[img]; k++) result[w++] = placeholderId;
                img++;
            }
            else result[w++] = id;
        }
        return (result, starts);
    }
}
