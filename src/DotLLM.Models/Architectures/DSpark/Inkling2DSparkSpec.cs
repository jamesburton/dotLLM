using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures.DSpark;

/// <summary>
/// The validated shape of an Inkling 2 DSpark drafter GGUF (issue #2046): everything the GPU backends need to size buffers and bind tensors, with the same
/// refusals as the CPU drafter (<see cref="Inkling2DSparkDrafter"/>) so no backend accepts a file another rejects. No tensor data is read.
/// </summary>
internal sealed record Inkling2DSparkSpec(
    int Hidden, int Layers, int Heads, int KvHeads, int HeadDim, int Ffn, float Eps, int BlockSize, int Window, float Softcap, float RopeTheta,
    int[] TargetVisits, bool[] Local, int Rank, int Vocab, int MaskRows, bool HasW2Bias)
{
    /// <summary>Drafts per round (cap): <c>DOTLLM_DSPARK_DRAFT</c> or 3 (the value llama.cpp's docs use for Inkling 2), never above <c>BlockSize - 1</c>.</summary>
    public int DraftTokensPerRound
        => Math.Min(BlockSize - 1, int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_DSPARK_DRAFT"), out int n) && n >= 1 ? n : 3);

    /// <summary>Reads and validates <paramref name="file"/> against a target of the given hidden size, layer-visit count and vocabulary.</summary>
    public static Inkling2DSparkSpec Read(GgufFile file, int targetHidden, int targetVisits, int targetVocab)
    {
        var md = file.Metadata;
        var t = file.TensorsByName;
        if (md.GetStringOrDefault("general.architecture") != "dflash") throw new NotSupportedException("Not a DFlash-family drafter GGUF.");
        if (!t.ContainsKey("markov_w1.weight") || !t.ContainsKey("joint_proj.weight") || !t.ContainsKey("mask_embd.weight") || !t.ContainsKey("blk.0.ctx_mix.weight"))
            throw new NotSupportedException("Not an Inkling 2 DSpark drafter (needs markov_w1, joint_proj, mask_embd and per-layer ctx_mix).");
        if (!md.GetBool("dflash.has_markov_context")) throw new NotSupportedException("Only the rnn_hidden (has_markov_context) DSpark head is supported.");
        if (md.ContainsKey("dflash.sample_from_anchor") && md.GetBool("dflash.sample_from_anchor")) throw new NotSupportedException("sample_from_anchor DSpark drafters are not supported.");
        if (md.ContainsKey("dflash.has_confidence_head") && md.GetBool("dflash.has_confidence_head")) throw new NotSupportedException("Confidence-head DSpark drafters are not supported.");
        if (t.ContainsKey("d2t")) throw new NotSupportedException("Reduced-vocabulary DSpark drafters are not supported.");

        int hidden = (int)md.GetUInt32("dflash.embedding_length");
        if (hidden != targetHidden) throw new InvalidOperationException($"Drafter hidden size {hidden} does not match the target's {targetHidden}.");
        int nLayer = (int)md.GetUInt32("dflash.block_count");
        int heads = (int)md.GetUInt32("dflash.attention.head_count");
        int kvHeads = (int)md.GetUInt32("dflash.attention.head_count_kv");
        int headDim = (int)md.GetUInt32("dflash.attention.key_length");
        int ffn = (int)md.GetUInt32("dflash.feed_forward_length");
        float eps = md.GetFloat32("dflash.attention.layer_norm_rms_epsilon");
        int block = (int)md.GetUInt32("dflash.block_size");
        int window = (int)md.GetUInt32("dflash.attention.sliding_window");
        float softcap = md.GetFloat32OrDefault("dflash.final_logit_softcapping", 0f);
        if (md.GetFloat32OrDefault("dflash.logit_scale", 0f) != 0f || md.GetFloat32OrDefault("dflash.embedding_scale", 0f) != 0f)
            throw new NotSupportedException("Drafter logit/embedding scales are not supported (Inkling 2 uses none).");
        float theta = md.GetFloat32OrDefault("dflash.rope.freq_base", 10000f);
        if ((int)md.GetUInt32OrDefault("dflash.rope.dimension_count", (uint)headDim) != headDim) throw new NotSupportedException("Partial-rotary drafters are not supported.");
        if (md.ContainsKey("dflash.attention.causal") && md.GetBool("dflash.attention.causal")) throw new NotSupportedException("Causal drafters are not supported.");
        int[] visits = ReadInts(md, "dflash.target_layers");
        if (visits.Length < 2 || visits[^1] != targetVisits)
            throw new InvalidOperationException($"Drafter target_layers {string.Join(",", visits)} must end at the target's {targetVisits} layer visits (the final normed output).");
        bool[] swa = ReadBools(md, "dflash.attention.sliding_window_pattern", nLayer);
        for (int i = 0; i < nLayer; i++)
            foreach (string bad in new[] { "attn_post_norm", "ffn_post_norm", "layer_output_scale", "rope_freqs", "attn_sinks" })
                if (t.ContainsKey($"blk.{i}.{bad}.weight")) throw new NotSupportedException($"Drafter tensor blk.{i}.{bad} is not supported.");
        if (t.ContainsKey("markov_w2.scale")) throw new NotSupportedException("markov_w2.scale is not supported.");
        int rank = t["markov_w1.weight"].Shape[0];
        int maskRows = t["mask_embd.weight"].Shape[1];
        if (maskRows < block - 1) throw new InvalidOperationException($"mask_embd has {maskRows} rows, the block needs {block - 1}.");
        return new Inkling2DSparkSpec(hidden, nLayer, heads, kvHeads, headDim, ffn, eps, block, window, softcap, theta, visits, swa, rank, targetVocab, maskRows, t.ContainsKey("markov_w2.bias"));
    }

    private static int[] ReadInts(GgufMetadata md, string key)
    {
        if (!md.TryGetValue(key, out var v)) throw new InvalidOperationException($"Drafter GGUF has no '{key}'.");
        return v.Value switch
        {
            int[] a => a,
            uint[] a => Array.ConvertAll(a, x => checked((int)x)),
            long[] a => Array.ConvertAll(a, x => checked((int)x)),
            ulong[] a => Array.ConvertAll(a, x => checked((int)x)),
            _ => throw new InvalidOperationException($"'{key}' is not an integer array."),
        };
    }

    private static bool[] ReadBools(GgufMetadata md, string key, int n)
    {
        // A missing pattern must not default to "every layer windowed" (the key was misspelled once, #2046: the global layer 1 was then windowed on both backends).
        if (!md.TryGetValue(key, out var v)) throw new InvalidOperationException($"Drafter GGUF has no '{key}'.");
        bool[] r = v.Value switch
        {
            bool[] a => a,
            object[] a => Array.ConvertAll(a, x => Convert.ToBoolean(x, System.Globalization.CultureInfo.InvariantCulture)),
            _ => throw new InvalidOperationException($"'{key}' is not a bool array."),
        };
        if (r.Length != n) throw new InvalidOperationException($"'{key}' has {r.Length} entries, expected {n}.");
        return r;
    }
}
