using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Deterministic synthetic <c>qwen4exp</c> (Qwen3.8-Flash-Next) GGUF fixture builder: a TINY but architecturally complete
/// checkpoint carrying every tensor family of the real model under the real names and (scaled) shapes — 4 hyper-connection
/// streams, four blocks (GDN, GDN + n-gram module, GDN, QSA), a 8-expert top-2 softmax MoE with a sigmoid-gated shared
/// expert in every block, the final head mixer, the n-gram table, and (optionally) a trailing MTP block.
/// </summary>
/// <remarks>
/// <para><b>Why it exists.</b> The CPU reference forward (issue #814 / I2) needs a model it can load in milliseconds, and the
/// config / tensor-contract tests need bytes that round-trip through the real <see cref="GgufFile"/> +
/// <see cref="GgufModelConfigExtractor"/> path. The tensor names and per-layer-kind sets are asserted against the real
/// header dump by <c>Qwen4ExpRealHeaderTests</c>; the shapes follow <see cref="Qwen4ExpTensors"/>.</para>
/// <para><b>Split output.</b> <see cref="WriteSplit"/> emits the same model as a <c>-0000N-of-0000M</c> set laid out like the
/// real Unsloth files: shard 1 is metadata only, the tensors are spread over shards 2..M with shard-local offsets.</para>
/// <para><b>Determinism.</b> Weights come from the seeded xorshift PRNG shared with the other synthetic fixtures; all tensors
/// are F32. The n-gram hash multipliers deliberately include a value above 2^63 so any float / signed round-trip is caught.</para>
/// </remarks>
public static class SyntheticQwen4ExpGguf
{
    /// <summary>Hidden size.</summary>
    public const int HiddenSize = 32;
    /// <summary>Vocabulary size.</summary>
    public const int VocabSize = 16;
    /// <summary>Trunk block count (GDN, GDN+PLE, GDN, QSA).</summary>
    public const int TrunkLayers = 4;
    /// <summary>Hyper-connection stream count.</summary>
    public const int HcCount = 4;
    /// <summary>Hyper-connection low rank.</summary>
    public const int HcLowRank = 8;
    /// <summary>Attention heads.</summary>
    public const int NumHeads = 2;
    /// <summary>KV heads.</summary>
    public const int NumKvHeads = 1;
    /// <summary>Per-head width (query/key/value).</summary>
    public const int HeadDim = 16;
    /// <summary>Rotary dimension count.</summary>
    public const int RopeDim = 8;
    /// <summary>Routed experts per MoE.</summary>
    public const int NumExperts = 8;
    /// <summary>Experts used per token.</summary>
    public const int NumExpertsUsed = 2;
    /// <summary>Routed-expert intermediate size.</summary>
    public const int MoeIntermediate = 16;
    /// <summary>Shared-expert intermediate size.</summary>
    public const int SharedIntermediate = 8;
    /// <summary>GDN key/value head width (<c>ssm.state_size</c>).</summary>
    public const int DState = 8;
    /// <summary>GDN key heads (<c>ssm.group_count</c>).</summary>
    public const int NKHead = 1;
    /// <summary>GDN value heads (<c>ssm.time_step_rank</c>).</summary>
    public const int NVHead = 2;
    /// <summary>GDN conv kernel.</summary>
    public const int DConv = 4;
    /// <summary>Full-attention interval (layer 3 is the QSA layer).</summary>
    public const int FullAttnInterval = 4;
    /// <summary>QSA indexer query heads.</summary>
    public const int IndexerHeads = 2;
    /// <summary>QSA indexer head width.</summary>
    public const int IndexerKeyLength = 8;
    /// <summary>QSA token budget.</summary>
    public const int IndexerTopK = 8;
    /// <summary>QSA pooling ratio.</summary>
    public const int CompressRatio = 4;
    /// <summary>PLE module layer (zero-based).</summary>
    public const int PleLayer = 1;
    /// <summary>PLE n-gram order.</summary>
    public const int PleNgram = 3;
    /// <summary>PLE heads per n-gram order.</summary>
    public const int PleHeadsPerNgram = 2;
    /// <summary>PLE conv kernel.</summary>
    public const int PleConvKernel = 4;
    /// <summary>PLE row width.</summary>
    public const int PleRowDim = 8;
    /// <summary>PLE EOS token id.</summary>
    public const int PleEosTokenId = 2;
    /// <summary>PLE image stand-in token id.</summary>
    public const int PleImageTokenId = 3;
    /// <summary>Context length.</summary>
    public const int ContextLength = 64;

    /// <summary>Exact uint64 hash multipliers (the last one is above 2^63).</summary>
    public static readonly ulong[] PleMultipliers = [23703573157769UL, 20109073645365UL, 18446744073709551557UL];

    /// <summary>Per-head hash moduli (distinct small primes, mirroring the real "distinct prime per head" scheme).</summary>
    public static readonly ulong[] PleHeadVocabSizes = [11UL, 13UL, 17UL, 19UL];

    /// <summary>Per-head first rows: the running sum of <see cref="PleHeadVocabSizes"/>, exactly as the real file chains its ranges.</summary>
    public static readonly ulong[] PleHeadOffsets = [0UL, 11UL, 24UL, 41UL];

    /// <summary>Rows in the (padded) n-gram table: the head ranges end at row 60, padded to 64 like the real table is padded.</summary>
    public const int PleTableRows = 64;

    private static int NumHeadsPle => (PleNgram - 1) * PleHeadsPerNgram;

    private sealed record Tensor(string Name, int[] Dims, byte[] Data);

    /// <summary>Builds the synthetic fixture to a single-file byte array.</summary>
    /// <param name="seed">PRNG seed.</param>
    /// <param name="includeMtp">Append a QSA + nextn MTP block (<c>block_count</c> 5, <c>nextn_predict_layers</c> 1).</param>
    public static byte[] Build(uint seed = 0xC0FFEEu, bool includeMtp = false)
    {
        var w = new GgufWriter();
        WriteMetadata(w, includeMtp);
        foreach (var t in BuildTensors(seed, includeMtp))
            w.AddTensor(t.Name, t.Dims, (uint)QuantizationType.F32, t.Data);
        return w.Build();
    }

    /// <summary>Writes the single-file fixture to <paramref name="path"/>.</summary>
    public static string Write(string path, uint seed = 0xC0FFEEu, bool includeMtp = false)
    {
        File.WriteAllBytes(path, Build(seed, includeMtp));
        return path;
    }

    /// <summary>
    /// Writes the fixture as a <c>{stem}-0000N-of-0000M.gguf</c> set into <paramref name="directory"/>, laid out like the real
    /// Unsloth files (shard 1 = metadata only, tensors spread over shards 2..M).
    /// </summary>
    /// <param name="directory">Existing output directory.</param>
    /// <param name="stem">File-name stem.</param>
    /// <param name="shardCount">Total shards (at least 2).</param>
    /// <param name="seed">PRNG seed.</param>
    /// <param name="includeMtp">Append the MTP block, as in <see cref="Build"/>.</param>
    /// <returns>Path of the FIRST shard — the one to hand to <see cref="GgufFile.Open"/>.</returns>
    public static string WriteSplit(string directory, string stem, int shardCount, uint seed = 0xC0FFEEu, bool includeMtp = false)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(shardCount, 2);
        var tensors = BuildTensors(seed, includeMtp);
        int dataShards = shardCount - 1;

        for (int s = 1; s <= shardCount; s++)
        {
            var w = new GgufWriter();
            if (s == 1)
                WriteMetadata(w, includeMtp);
            w.AddUInt16("split.no", (ushort)(s - 1)).AddUInt16("split.count", (ushort)shardCount).AddInt32("split.tensors.count", tensors.Count);
            if (s > 1)
            {
                // Contiguous slice of the tensor list for data shard (s - 1).
                int lo = (int)((long)tensors.Count * (s - 2) / dataShards);
                int hi = (int)((long)tensors.Count * (s - 1) / dataShards);
                for (int i = lo; i < hi; i++)
                    w.AddTensor(tensors[i].Name, tensors[i].Dims, (uint)QuantizationType.F32, tensors[i].Data);
            }
            File.WriteAllBytes(Path.Combine(directory, $"{stem}-{s:D5}-of-{shardCount:D5}.gguf"), w.Build());
        }
        return Path.Combine(directory, $"{stem}-{1:D5}-of-{shardCount:D5}.gguf");
    }

    // ───────────────────────────── metadata ─────────────────────────────

    private static void WriteMetadata(GgufWriter w, bool includeMtp)
    {
        const string arch = "qwen4exp";
        int blocks = TrunkLayers + (includeMtp ? 1 : 0);

        w.AddString("general.architecture", arch);
        w.AddString("general.name", "synthetic-qwen4exp-tiny");
        w.AddUInt32("general.alignment", 32);

        w.AddUInt32($"{arch}.block_count", (uint)blocks);
        w.AddUInt32($"{arch}.context_length", ContextLength);
        w.AddUInt32($"{arch}.embedding_length", HiddenSize);
        w.AddUInt32($"{arch}.attention.head_count", NumHeads);
        w.AddUInt32($"{arch}.attention.head_count_kv", NumKvHeads);
        w.AddInt32Array($"{arch}.rope.dimension_sections", [3, 3, 2, 0]);
        w.AddFloat32($"{arch}.rope.freq_base", 10000000.0f);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
        w.AddUInt32($"{arch}.expert_count", NumExperts);
        w.AddUInt32($"{arch}.expert_used_count", NumExpertsUsed);
        w.AddUInt32($"{arch}.attention.key_length", HeadDim);
        w.AddUInt32($"{arch}.attention.value_length", HeadDim);
        w.AddUInt32($"{arch}.expert_feed_forward_length", MoeIntermediate);
        w.AddUInt32($"{arch}.expert_shared_feed_forward_length", SharedIntermediate);
        if (includeMtp)
            w.AddUInt32($"{arch}.nextn_predict_layers", 1);
        w.AddUInt32($"{arch}.ssm.conv_kernel", DConv);
        w.AddUInt32($"{arch}.ssm.state_size", DState);
        w.AddUInt32($"{arch}.ssm.group_count", NKHead);
        w.AddUInt32($"{arch}.ssm.time_step_rank", NVHead);
        w.AddUInt32($"{arch}.ssm.inner_size", DState * NVHead);
        w.AddUInt32($"{arch}.full_attention_interval", FullAttnInterval);
        w.AddUInt32($"{arch}.rope.dimension_count", RopeDim);

        w.AddUInt32($"{arch}.hyper_connection.count", HcCount);
        w.AddUInt32($"{arch}.hyper_connection.low_rank", HcLowRank);

        w.AddUInt32($"{arch}.attention.indexer.head_count", IndexerHeads);
        w.AddUInt32($"{arch}.attention.indexer.key_length", IndexerKeyLength);
        w.AddUInt32($"{arch}.attention.indexer.top_k", IndexerTopK);
        var ratios = new int[blocks];
        for (int i = 0; i < blocks; i++)
            ratios[i] = i >= TrunkLayers || (i + 1) % FullAttnInterval == 0 ? CompressRatio : 0;
        w.AddInt32Array($"{arch}.attention.compress_ratios", ratios);

        w.AddInt32Array($"{arch}.ple.layers", [PleLayer]);
        w.AddUInt32($"{arch}.ple.ngram_size", PleNgram);
        w.AddUInt32($"{arch}.ple.heads_per_ngram", PleHeadsPerNgram);
        w.AddUInt32($"{arch}.ple.conv_kernel", PleConvKernel);
        w.AddUInt32($"{arch}.ple.eos_token_id", PleEosTokenId);
        w.AddUInt32($"{arch}.embedding_length_per_layer_input", PleRowDim);
        w.AddUInt64Array($"{arch}.ple.layer_multipliers", PleMultipliers);
        w.AddUInt64Array($"{arch}.ple.head_offsets", PleHeadOffsets);
        w.AddUInt64Array($"{arch}.ple.head_vocab_sizes", PleHeadVocabSizes);
        w.AddUInt32($"{arch}.ple.image_token_id", PleImageTokenId);

        AddTokenizer(w);
    }

    private static void AddTokenizer(GgufWriter w)
    {
        w.AddString("tokenizer.ggml.model", "llama");
        var tokens = new string[VocabSize];
        var scores = new float[VocabSize];
        var types = new int[VocabSize];
        for (int i = 0; i < VocabSize; i++)
        {
            tokens[i] = i switch { 0 => "<unk>", 1 => "<bos>", 2 => "<eos>", 3 => "<image>", _ => $"tok{i}" };
            types[i] = i switch { 0 => 2, 1 or 2 => 3, _ => 1 };
        }
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddFloat32Array("tokenizer.ggml.scores", scores);
        w.AddInt32Array("tokenizer.ggml.token_type", types);
        w.AddUInt32("tokenizer.ggml.bos_token_id", 1);
        w.AddUInt32("tokenizer.ggml.eos_token_id", 2);
        w.AddUInt32("tokenizer.ggml.unknown_token_id", 0);
    }

    // ───────────────────────────── tensors ─────────────────────────────

    private static List<Tensor> BuildTensors(uint seed, bool includeMtp)
    {
        var rng = new SyntheticGemma4Gguf.Xorshift(seed);
        var list = new List<Tensor>();
        const int hcDim = HcCount * HiddenSize;

        Matrix(list, rng, Qwen4ExpTensors.TokenEmbd, HiddenSize, VocabSize);
        Matrix(list, rng, Qwen4ExpTensors.Output, HiddenSize, VocabSize);
        Norm(list, rng, Qwen4ExpTensors.OutputHcNorm, hcDim);
        Matrix(list, rng, Qwen4ExpTensors.OutputHcDown, hcDim, HcLowRank);
        Matrix(list, rng, Qwen4ExpTensors.OutputHcUp, HcLowRank, hcDim);
        Matrix(list, rng, Qwen4ExpTensors.PerLayerTokenEmbd, PleRowDim, PleTableRows);

        for (int il = 0; il < TrunkLayers; il++)
        {
            bool attention = (il + 1) % FullAttnInterval == 0;
            AddBlock(list, rng, il, attention, hasPle: il == PleLayer);
        }

        if (includeMtp)
        {
            int il = TrunkLayers;
            AddBlock(list, rng, il, attention: true, hasPle: false);
            Matrix(list, rng, Qwen4ExpTensors.Block(il, Qwen4ExpTensors.Suffix.NextnEhProj), 2 * HiddenSize, HiddenSize);
            Norm(list, rng, Qwen4ExpTensors.Block(il, Qwen4ExpTensors.Suffix.NextnEnorm), HiddenSize);
            Norm(list, rng, Qwen4ExpTensors.Block(il, Qwen4ExpTensors.Suffix.NextnHnorm), hcDim);
            Norm(list, rng, Qwen4ExpTensors.Block(il, Qwen4ExpTensors.Suffix.NextnHcHeadNorm), hcDim);
            Matrix(list, rng, Qwen4ExpTensors.Block(il, Qwen4ExpTensors.Suffix.NextnHcHeadDown), hcDim, HcLowRank);
            Matrix(list, rng, Qwen4ExpTensors.Block(il, Qwen4ExpTensors.Suffix.NextnHcHeadUp), HcLowRank, hcDim);
        }

        return list;
    }

    private static void AddBlock(List<Tensor> t, SyntheticGemma4Gguf.Xorshift rng, int il, bool attention, bool hasPle)
    {
        const int hcDim = HcCount * HiddenSize;
        string B(string suffix) => Qwen4ExpTensors.Block(il, suffix);

        foreach (var (norm, down, up, inject) in new[]
        {
            (Qwen4ExpTensors.Suffix.HcAttnNorm, Qwen4ExpTensors.Suffix.HcAttnDown, Qwen4ExpTensors.Suffix.HcAttnUp, Qwen4ExpTensors.Suffix.HcAttnInject),
            (Qwen4ExpTensors.Suffix.HcFfnNorm, Qwen4ExpTensors.Suffix.HcFfnDown, Qwen4ExpTensors.Suffix.HcFfnUp, Qwen4ExpTensors.Suffix.HcFfnInject),
        })
        {
            Norm(t, rng, B(norm), hcDim);
            Matrix(t, rng, B(down), hcDim, HcLowRank);
            Matrix(t, rng, B(up), HcLowRank, hcDim);
            Matrix(t, rng, B(inject), hcDim, HcCount);
        }

        if (attention)
        {
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.AttnQ), HiddenSize, NumHeads * HeadDim * 2);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.AttnK), HiddenSize, NumKvHeads * HeadDim);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.AttnV), HiddenSize, NumKvHeads * HeadDim);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.AttnOutput), NumHeads * HeadDim, HiddenSize);
            Norm(t, rng, B(Qwen4ExpTensors.Suffix.AttnQNorm), HeadDim);
            Norm(t, rng, B(Qwen4ExpTensors.Suffix.AttnKNorm), HeadDim);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.IndexerQProj), HiddenSize, IndexerHeads * IndexerKeyLength);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.IndexerKProj), HiddenSize, IndexerKeyLength);
            Norm(t, rng, B(Qwen4ExpTensors.Suffix.IndexerQNorm), IndexerKeyLength);
            Norm(t, rng, B(Qwen4ExpTensors.Suffix.IndexerKNorm), IndexerKeyLength);
        }
        else
        {
            int valueDim = DState * NVHead;
            int convDim = 2 * DState * NKHead + valueDim;
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.AttnQkv), HiddenSize, convDim);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.AttnGate), HiddenSize, valueDim);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.SsmConv1d), DConv, convDim);
            Vector(t, rng, B(Qwen4ExpTensors.Suffix.SsmDtBias), NVHead, 0.1f, offset: 0f);
            Vector(t, rng, B(Qwen4ExpTensors.Suffix.SsmA), NVHead, 0.25f, offset: -0.5f); // decay base must be negative
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.SsmBeta), HiddenSize, NVHead);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.SsmAlpha), HiddenSize, NVHead);
            Norm(t, rng, B(Qwen4ExpTensors.Suffix.SsmNorm), DState);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.SsmOut), valueDim, HiddenSize);
        }

        if (hasPle)
        {
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.PleKey), HiddenSize, hcDim);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.PleValue), HiddenSize, HiddenSize);
            Norm(t, rng, B(Qwen4ExpTensors.Suffix.PleNormKey), hcDim);
            Norm(t, rng, B(Qwen4ExpTensors.Suffix.PleNormQuery), hcDim);
            Norm(t, rng, B(Qwen4ExpTensors.Suffix.PleNormConv), hcDim);
            Matrix(t, rng, B(Qwen4ExpTensors.Suffix.PleConv1d), PleConvKernel, hcDim);
        }

        Matrix(t, rng, B(Qwen4ExpTensors.Suffix.FfnGateInp), HiddenSize, NumExperts);
        Bank(t, rng, B(Qwen4ExpTensors.Suffix.FfnDownExps), MoeIntermediate, HiddenSize);
        Bank(t, rng, B(Qwen4ExpTensors.Suffix.FfnGateExps), HiddenSize, MoeIntermediate);
        Bank(t, rng, B(Qwen4ExpTensors.Suffix.FfnUpExps), HiddenSize, MoeIntermediate);
        Vector(t, rng, B(Qwen4ExpTensors.Suffix.FfnGateInpShexp), HiddenSize, 0.05f, offset: 0f);
        Matrix(t, rng, B(Qwen4ExpTensors.Suffix.FfnGateShexp), HiddenSize, SharedIntermediate);
        Matrix(t, rng, B(Qwen4ExpTensors.Suffix.FfnUpShexp), HiddenSize, SharedIntermediate);
        Matrix(t, rng, B(Qwen4ExpTensors.Suffix.FfnDownShexp), SharedIntermediate, HiddenSize);
    }

    private static void Matrix(List<Tensor> t, SyntheticGemma4Gguf.Xorshift rng, string name, int ne0, int ne1)
        => Add(t, name, [ne0, ne1], Fill(rng, (long)ne0 * ne1, 0.05f, 0f));

    private static void Bank(List<Tensor> t, SyntheticGemma4Gguf.Xorshift rng, string name, int ne0, int ne1)
        => Add(t, name, [ne0, ne1, NumExperts], Fill(rng, (long)ne0 * ne1 * NumExperts, 0.05f, 0f));

    private static void Vector(List<Tensor> t, SyntheticGemma4Gguf.Xorshift rng, string name, int n, float scale, float offset)
        => Add(t, name, [n], Fill(rng, n, scale, offset));

    /// <summary>Norm gammas: small values (values are irrelevant to the config / contract tests).</summary>
    private static void Norm(List<Tensor> t, SyntheticGemma4Gguf.Xorshift rng, string name, int n)
        => Add(t, name, [n], Fill(rng, n, 0.05f, 0f));

    private static float[] Fill(SyntheticGemma4Gguf.Xorshift rng, long count, float scale, float offset)
    {
        var f = new float[count];
        for (long i = 0; i < f.LongLength; i++) f[i] = offset + rng.NextSigned(scale);
        return f;
    }

    private static void Add(List<Tensor> t, string name, int[] dims, float[] values)
        => t.Add(new Tensor(name, dims, MemoryMarshal.AsBytes(values.AsSpan()).ToArray()));
}
