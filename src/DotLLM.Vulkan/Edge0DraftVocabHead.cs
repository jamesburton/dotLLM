namespace DotLLM.Vulkan;

/// <summary>
/// The Edge0 draft head's vocabulary subset on the device (#2013): the pack's <c>draft_vocab.bin</c> ids (unsorted, base subset order then appended ids) as an
/// int32 table, plus a compact logits buffer. The LM head's Q8_0 rows are read in place at those ids (<c>matmul_q8_0_mmvq_rows.comp</c>); no weight is copied or
/// widened. Output row <c>i</c> scores token <c>Ids[i]</c>, so the draft argmax is mapped back to the full vocabulary with <see cref="ArgMax"/>.
/// </summary>
internal sealed class Edge0DraftVocabHead : IDisposable
{
    /// <summary>Token id of each compact output row (file order).</summary>
    public int[] Ids { get; }

    /// <summary>int32 copy of <see cref="Ids"/> on the device (4 bytes per id, 425 KB for the 106,299-id pack).</summary>
    public VulkanDevice.Buffer IdsBuffer { get; }

    /// <summary>Compact F32 logits, one per id.</summary>
    public VulkanDevice.Buffer Logits { get; }

    /// <summary>Host landing area for <see cref="Logits"/>.</summary>
    public float[] Host { get; }

    private Edge0DraftVocabHead(int[] ids, VulkanDevice.Buffer idsBuffer, VulkanDevice.Buffer logits)
    {
        Ids = ids; IdsBuffer = idsBuffer; Logits = logits; Host = new float[ids.Length];
    }

    /// <summary>Uploads the id table and allocates the compact logits.</summary>
    public static Edge0DraftVocabHead Create(VulkanDevice device, VulkanStagingBuffer staging, int[] ids)
    {
        var bits = new float[ids.Length];
        for (int i = 0; i < ids.Length; i++) bits[i] = BitConverter.Int32BitsToSingle(ids[i]);
        var idsBuf = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, bits);
        VulkanDevice.Buffer? logits = null;
        try
        {
            logits = device.AllocateDeviceLocal((long)ids.Length * sizeof(float));
            return new Edge0DraftVocabHead(ids, idsBuf, logits);
        }
        catch { logits?.Dispose(); idsBuf.Dispose(); throw; }
    }

    /// <summary>
    /// Full-vocabulary token id of the highest compact logit. The first maximum in file order wins a tie (the full-vocab argmax keeps the lowest id; the two can
    /// differ only on an exact tie).
    /// </summary>
    /// <param name="compactLogits">One logit per entry of <paramref name="ids"/>.</param>
    /// <param name="ids">Token id of each logit.</param>
    public static int ArgMax(ReadOnlySpan<float> compactLogits, ReadOnlySpan<int> ids)
    {
        if (compactLogits.Length != ids.Length || ids.Length == 0) throw new ArgumentException("One logit per draft-vocab id is required.");
        int best = 0;
        for (int i = 1; i < compactLogits.Length; i++) if (compactLogits[i] > compactLogits[best]) best = i;
        return ids[best];
    }

    /// <inheritdoc/>
    public void Dispose() { IdsBuffer.Dispose(); Logits.Dispose(); }
}
