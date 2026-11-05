using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>How <see cref="SyntheticRicursiveGguf"/> writes <c>ricursive.swiglu_clamp</c>.</summary>
public enum SyntheticClampKey
{
    /// <summary>No key (the "no clamp" variant the converters emit for a null HF threshold).</summary>
    Absent,
    /// <summary>A Float32 scalar (AtomicChat / prometheusAIR files).</summary>
    Scalar,
    /// <summary>A per-layer Float32 array (llama.cpp PR #31692's current converter).</summary>
    PerLayerArray,
}

/// <summary>Knobs for <see cref="SyntheticRicursiveGguf"/> (defaults = an AtomicChat-shaped file).</summary>
public sealed record SyntheticRicursiveOptions
{
    /// <summary>Value-residual tensor stem (<c>attn_vres_lambda</c> or <c>attn_value_res</c>); <see langword="null"/> = no value residual.</summary>
    public string? ValueResidualName { get; init; } = RicursiveNames.VresLambda;
    /// <summary>A layer that uses the OTHER value-residual spelling (a mixed, invalid file).</summary>
    public int? MixedValueResidualLayer { get; init; }
    /// <summary>Emit <c>attn_gate</c> on every layer.</summary>
    public bool WithGate { get; init; } = true;
    /// <summary>Emit an untied <c>output.weight</c>.</summary>
    public bool WithOutputHead { get; init; } = true;
    /// <summary>How the clamp key is written.</summary>
    public SyntheticClampKey Clamp { get; init; } = SyntheticClampKey.Scalar;
    /// <summary><c>embedding_scale</c> value; <see langword="null"/> = key absent (folded).</summary>
    public float? EmbeddingScale { get; init; } = SyntheticRicursiveGguf.EmbedScale;
    /// <summary>Write <c>attention.sliding_window_pattern</c> (false = rely on the every-5th-layer default).</summary>
    public bool WritePattern { get; init; } = true;
    /// <summary>Explicit pattern to write instead of the default (true = sliding).</summary>
    public bool[]? Pattern { get; init; }
    /// <summary>Also write PR #31692's <c>attention.rope_pattern</c> (= the sliding pattern unless <see cref="RopePatternOverride"/>).</summary>
    public bool WriteRopePattern { get; init; }
    /// <summary>A rope_pattern different from the sliding pattern (invalid file).</summary>
    public bool[]? RopePatternOverride { get; init; }
    /// <summary><c>tokenizer.ggml.add_bos_token</c>; <see langword="null"/> = key absent (AtomicChat).</summary>
    public bool? AddBosToken { get; init; }
    /// <summary>Write <c>tokenizer.ggml.eot_token_id</c> = <c>&lt;|im_end|&gt;</c> (AtomicChat).</summary>
    public bool WriteEot { get; init; } = true;
    /// <summary>Tensor to leave out.</summary>
    public string? OmitTensor { get; init; }
    /// <summary>Tensor to emit with one extra element on its first dim.</summary>
    public string? WrongShapeTensor { get; init; }
    /// <summary>An extra, unknown tensor name to add (shape [1]).</summary>
    public string? ExtraTensor { get; init; }
    /// <summary>Embedded chat template (null = none).</summary>
    public string? ChatTemplate { get; init; }
    /// <summary>Xorshift seed.</summary>
    public uint Seed { get; init; } = 0x51D3u;
}

/// <summary>GGUF / HF tensor-name stems shared by the ricursive fixtures.</summary>
public static class RicursiveNames
{
    /// <summary>AtomicChat / llama.cpp PR #31692 spelling.</summary>
    public const string VresLambda = "attn_vres_lambda";
    /// <summary>prometheusAIR spelling.</summary>
    public const string VresValueRes = "attn_value_res";
}

/// <summary>
/// Deterministic synthetic <c>ricursive</c> (Ripple) fixtures (issue #1643): a tiny GGUF with the exact key and tensor naming of the
/// real AtomicChat / prometheusAIR conversions, and a matching tiny HF directory (<c>config.json</c> + <c>model.safetensors</c>) with the
/// vendor names, for the S1 forward pass and its <c>modeling_ricursive.py</c> F32 oracle. All tensors are F32.
/// </summary>
/// <remarks>
/// The shapes are non-degenerate on purpose: GQA ratio 3 (6 q / 2 kv heads), head_dim 16 != hidden / heads (= 8, as the real 128 vs 64),
/// q/gate/o width 96 = 2 x hidden (as the real 10240 = 2 x 5120), window 8 (a sequence of 10+ tokens crosses it), two full-attention
/// periods (layers 4 and 9 are full / NoPE), a binding clamp (0.5), a non-sqrt(1) embedding scale, and value-residual logits that differ
/// per layer and from 0 (so sigmoid != 0.5).
/// </remarks>
public static class SyntheticRicursiveGguf
{
    /// <summary>Vocabulary size.</summary>
    public const int VocabSize = 32;
    /// <summary>Hidden size.</summary>
    public const int HiddenSize = 48;
    /// <summary>Query heads.</summary>
    public const int NumHeads = 6;
    /// <summary>KV heads (GQA 3).</summary>
    public const int NumKvHeads = 2;
    /// <summary>Head dim (!= hidden / heads).</summary>
    public const int HeadDim = 16;
    /// <summary>FFN width.</summary>
    public const int Intermediate = 40;
    /// <summary>Layers (two 4:1 periods).</summary>
    public const int NumLayers = 10;
    /// <summary>Sliding window (keys incl. current).</summary>
    public const int Window = 8;
    /// <summary>SwiGLU clamp (small enough to bind on random weights).</summary>
    public const float Clamp = 0.5f;
    /// <summary>Embedding scale (sqrt(hidden)).</summary>
    public const float EmbedScale = 6.928203f;
    /// <summary>Context length.</summary>
    public const int ContextLength = 64;
    /// <summary>Id of <c>&lt;|endoftext|&gt;</c> (bos / unk; an EOG).</summary>
    public const int EndOfTextId = 29;
    /// <summary>Id of <c>&lt;|im_start|&gt;</c>.</summary>
    public const int ImStartId = 30;
    /// <summary>Id of <c>&lt;|im_end|&gt;</c> (eos).</summary>
    public const int ImEndId = 31;

    /// <summary>The default layer pattern (true = sliding): full at il % 5 == 4.</summary>
    public static bool[] DefaultPattern()
    {
        var p = new bool[NumLayers];
        for (int l = 0; l < NumLayers; l++) p[l] = l % 5 != 4;
        return p;
    }

    /// <summary>Value-residual logit of layer <paramref name="l"/> (non-zero, distinct per layer).</summary>
    public static float VresLogit(int l) => 0.75f - 0.3f * l;

    /// <summary>Builds the GGUF bytes.</summary>
    /// <param name="o">Options (null = defaults).</param>
    /// <returns>The file bytes.</returns>
    public static byte[] Build(SyntheticRicursiveOptions? o = null)
    {
        o ??= new SyntheticRicursiveOptions();
        var w = new GgufWriter();
        var rng = new SyntheticGemma4Gguf.Xorshift(o.Seed);
        const string arch = "ricursive";
        bool[] pattern = o.Pattern ?? DefaultPattern();

        w.AddString("general.architecture", arch);
        w.AddString("general.type", "model");
        w.AddString("general.name", "synthetic-ricursive-tiny");
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.block_count", NumLayers);
        w.AddUInt32($"{arch}.context_length", ContextLength);
        w.AddUInt32($"{arch}.embedding_length", HiddenSize);
        w.AddUInt32($"{arch}.feed_forward_length", Intermediate);
        w.AddUInt32($"{arch}.attention.head_count", NumHeads);
        w.AddUInt32($"{arch}.attention.head_count_kv", NumKvHeads);
        w.AddFloat32($"{arch}.rope.freq_base", 10000.0f);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
        w.AddUInt32($"{arch}.attention.key_length", HeadDim);
        w.AddUInt32($"{arch}.attention.value_length", HeadDim);
        w.AddUInt32($"{arch}.vocab_size", VocabSize);
        w.AddUInt32($"{arch}.rope.dimension_count", HeadDim);
        w.AddUInt32($"{arch}.attention.sliding_window", Window);
        if (o.WritePattern) w.AddBoolArray($"{arch}.attention.sliding_window_pattern", pattern);
        if (o.WriteRopePattern) w.AddBoolArray($"{arch}.attention.rope_pattern", o.RopePatternOverride ?? pattern);
        if (o.EmbeddingScale is { } es) w.AddFloat32($"{arch}.embedding_scale", es);
        switch (o.Clamp)
        {
            case SyntheticClampKey.Scalar: w.AddFloat32($"{arch}.swiglu_clamp", Clamp); break;
            case SyntheticClampKey.PerLayerArray:
                var arr = new float[NumLayers];
                for (int l = 0; l < NumLayers; l++) arr[l] = l == 3 ? 0f : Clamp + 0.25f * (l % 2);
                w.AddFloat32Array($"{arch}.swiglu_clamp", arr);
                break;
        }
        AddTokenizer(w, o);

        void Tensor(string name, int[] dims, Func<int, float> value)
        {
            if (name == o.OmitTensor) return;
            if (name == o.WrongShapeTensor) dims = [dims[0] + 1, .. dims[1..]];
            long n = 1;
            foreach (int d in dims) n *= d;
            var f = new float[n];
            for (int i = 0; i < f.Length; i++) f[i] = value(i);
            w.AddTensor(name, dims, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(f.AsSpan()).ToArray());
        }
        void Mat(string name, int inK, int outM) => Tensor(name, [inK, outM], _ => rng.NextSigned(0.08f));
        void Norm(string name, int n) => Tensor(name, [n], _ => 1f + rng.NextSigned(0.1f));

        int nq = NumHeads * HeadDim, nkv = NumKvHeads * HeadDim;
        Mat("token_embd.weight", HiddenSize, VocabSize);
        Norm("output_norm.weight", HiddenSize);
        if (o.WithOutputHead) Mat("output.weight", HiddenSize, VocabSize);
        for (int l = 0; l < NumLayers; l++)
        {
            string p = $"blk.{l}";
            Norm($"{p}.attn_norm.weight", HiddenSize);
            Mat($"{p}.attn_q.weight", HiddenSize, nq);
            Mat($"{p}.attn_k.weight", HiddenSize, nkv);
            Mat($"{p}.attn_v.weight", HiddenSize, nkv);
            Mat($"{p}.attn_output.weight", nq, HiddenSize);
            Norm($"{p}.attn_q_norm.weight", HeadDim);
            Norm($"{p}.attn_k_norm.weight", HeadDim);
            if (o.WithGate) Mat($"{p}.attn_gate.weight", HiddenSize, nq);
            if (o.ValueResidualName is { } vres)
            {
                string stem = l == o.MixedValueResidualLayer
                    ? (vres == RicursiveNames.VresLambda ? RicursiveNames.VresValueRes : RicursiveNames.VresLambda)
                    : vres;
                float logit = VresLogit(l);
                Tensor($"{p}.{stem}.weight", [1], _ => logit);
            }
            Norm($"{p}.post_attention_norm.weight", HiddenSize);
            Norm($"{p}.ffn_norm.weight", HiddenSize);
            Mat($"{p}.ffn_gate.weight", HiddenSize, Intermediate);
            Mat($"{p}.ffn_up.weight", HiddenSize, Intermediate);
            Mat($"{p}.ffn_down.weight", Intermediate, HiddenSize);
            Norm($"{p}.post_ffw_norm.weight", HiddenSize);
        }
        if (o.ExtraTensor is { } extra) Tensor(extra, [1], _ => 0f);
        return w.Build();
    }

    /// <summary>Writes the GGUF to <paramref name="path"/>.</summary>
    /// <param name="path">Destination.</param>
    /// <param name="o">Options.</param>
    /// <returns><paramref name="path"/>.</returns>
    public static string Write(string path, SyntheticRicursiveOptions? o = null)
    {
        File.WriteAllBytes(path, Build(o));
        return path;
    }

    private static void AddTokenizer(GgufWriter w, SyntheticRicursiveOptions o)
    {
        w.AddString("tokenizer.ggml.model", "gpt2");
        w.AddString("tokenizer.ggml.pre", "dbrx");
        var tokens = new string[VocabSize];
        var types = new int[VocabSize];
        for (int i = 0; i < VocabSize; i++) { tokens[i] = ((char)('a' + i % 26)).ToString() + (i >= 26 ? i.ToString() : ""); types[i] = 1; }
        tokens[EndOfTextId] = "<|endoftext|>";
        tokens[ImStartId] = "<|im_start|>";
        tokens[ImEndId] = "<|im_end|>";
        types[EndOfTextId] = types[ImStartId] = types[ImEndId] = 3;
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddInt32Array("tokenizer.ggml.token_type", types);
        w.AddStringArray("tokenizer.ggml.merges", []);
        w.AddUInt32("tokenizer.ggml.bos_token_id", EndOfTextId);
        w.AddUInt32("tokenizer.ggml.eos_token_id", ImEndId);
        w.AddUInt32("tokenizer.ggml.unknown_token_id", EndOfTextId);
        if (o.WriteEot) w.AddUInt32("tokenizer.ggml.eot_token_id", ImEndId);
        if (o.AddBosToken is { } addBos) w.AddBool("tokenizer.ggml.add_bos_token", addBos);
        if (o.ChatTemplate is { } t) w.AddString("tokenizer.chat_template", t);
    }

    /// <summary>
    /// Writes a tiny HF checkpoint directory (<c>config.json</c> with the vendor field names + one F32 <c>model.safetensors</c> with the
    /// vendor tensor names, <c>value_res_lambda</c> without a <c>.weight</c> suffix) of the same shape as <see cref="Build"/>. The JSON
    /// mirrors <c>ricursiveAI/Ripple-65B-Preview/config.json</c> (including the derived <c>hybrid_layer_pattern</c>), so it is also a
    /// ready-made input for the vendor <c>modeling_ricursive.py</c> oracle (S1, #1644).
    /// </summary>
    /// <param name="dir">Directory to create/overwrite.</param>
    /// <param name="useValres">HF <c>use_valres</c> (and whether the logits are written).</param>
    /// <param name="clampThreshold">HF <c>ffn_clamp_threshold</c> (null = JSON null).</param>
    /// <param name="embedScale">HF <c>embed_scale</c> (null = JSON null, folded).</param>
    /// <param name="seed">Xorshift seed.</param>
    /// <returns><paramref name="dir"/>.</returns>
    public static string WriteHfDirectory(string dir, bool useValres = true, float? clampThreshold = Clamp,
                                          float? embedScale = EmbedScale, uint seed = 0x51D3u)
    {
        Directory.CreateDirectory(dir);
        bool[] pattern = DefaultPattern();
        using (var fs = File.Create(Path.Combine(dir, "config.json")))
        using (var j = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true }))
        {
            j.WriteStartObject();
            j.WriteStartArray("architectures"); j.WriteStringValue("RicursiveForCausalLM"); j.WriteEndArray();
            j.WriteString("model_type", "ricursive");
            j.WriteNumber("bos_token_id", EndOfTextId);
            j.WriteStartArray("eos_token_id"); j.WriteNumberValue(ImEndId); j.WriteNumberValue(EndOfTextId); j.WriteEndArray();
            if (embedScale is { } es) j.WriteNumber("embed_scale", es); else j.WriteNull("embed_scale");
            if (clampThreshold is { } ct) j.WriteNumber("ffn_clamp_threshold", ct); else j.WriteNull("ffn_clamp_threshold");
            j.WriteBoolean("gate_full_precision", true);
            j.WriteNumber("head_dim", HeadDim);
            j.WriteString("hidden_act", "silu");
            j.WriteNumber("hidden_size", HiddenSize);
            j.WriteStartArray("hybrid_layer_pattern");
            foreach (bool sl in pattern) j.WriteNumberValue(sl ? 1 : 0);
            j.WriteEndArray();
            j.WriteNumber("intermediate_size", Intermediate);
            j.WriteBoolean("is_hybrid_swa", true);
            j.WriteStartArray("layer_types");
            foreach (bool sl in pattern) j.WriteStringValue(sl ? "sliding_attention" : "full_attention");
            j.WriteEndArray();
            j.WriteNumber("max_position_embeddings", ContextLength);
            j.WriteNumber("num_attention_heads", NumHeads);
            j.WriteNumber("num_hidden_layers", NumLayers);
            j.WriteNumber("num_key_value_heads", NumKvHeads);
            j.WriteNumber("pad_token_id", EndOfTextId);
            j.WriteNumber("rms_norm_eps", 1e-6);
            j.WriteNumber("rope_theta", 10000.0);
            j.WriteNumber("sliding_window", Window);
            j.WriteBoolean("tie_word_embeddings", false);
            j.WriteString("torch_dtype", "float32");
            j.WriteBoolean("use_cache", true);
            j.WriteBoolean("use_gated_attention", true);
            j.WriteBoolean("use_valres", useValres);
            j.WriteNumber("vocab_size", VocabSize);
            j.WriteEndObject();
        }

        var rng = new SyntheticGemma4Gguf.Xorshift(seed);
        var tensors = new List<(string Name, int[] Shape, float[] Data)>();
        void T(string name, int[] shape, Func<float> v)
        {
            long n = 1;
            foreach (int d in shape) n *= d;
            var f = new float[n];
            for (int i = 0; i < f.Length; i++) f[i] = v();
            tensors.Add((name, shape, f));
        }
        void Lin(string name, int outM, int inK) => T(name, [outM, inK], () => rng.NextSigned(0.08f));
        void Norm(string name, int n) => T(name, [n], () => 1f + rng.NextSigned(0.1f));
        int nq = NumHeads * HeadDim, nkv = NumKvHeads * HeadDim;
        Lin("model.embed_tokens.weight", VocabSize, HiddenSize);
        Norm("model.norm.weight", HiddenSize);
        Lin("lm_head.weight", VocabSize, HiddenSize);
        for (int l = 0; l < NumLayers; l++)
        {
            string p = $"model.layers.{l}";
            Norm($"{p}.input_layernorm.weight", HiddenSize);
            Lin($"{p}.self_attn.q_proj.weight", nq, HiddenSize);
            Lin($"{p}.self_attn.k_proj.weight", nkv, HiddenSize);
            Lin($"{p}.self_attn.v_proj.weight", nkv, HiddenSize);
            Lin($"{p}.self_attn.o_proj.weight", HiddenSize, nq);
            Norm($"{p}.self_attn.q_norm.weight", HeadDim);
            Norm($"{p}.self_attn.k_norm.weight", HeadDim);
            Lin($"{p}.self_attn.g_proj.weight", nq, HiddenSize);
            if (useValres)
            {
                float logit = VresLogit(l);
                T($"{p}.self_attn.value_res_lambda", [1], () => logit);
            }
            Norm($"{p}.post_attention_layernorm.weight", HiddenSize);
            Norm($"{p}.pre_feedforward_layernorm.weight", HiddenSize);
            Lin($"{p}.mlp.gate_proj.weight", Intermediate, HiddenSize);
            Lin($"{p}.mlp.up_proj.weight", Intermediate, HiddenSize);
            Lin($"{p}.mlp.down_proj.weight", HiddenSize, Intermediate);
            Norm($"{p}.post_feedforward_layernorm.weight", HiddenSize);
        }
        WriteSafetensors(Path.Combine(dir, "model.safetensors"), tensors);
        return dir;
    }

    private static void WriteSafetensors(string path, List<(string Name, int[] Shape, float[] Data)> tensors)
    {
        var ms = new MemoryStream();
        using (var j = new Utf8JsonWriter(ms))
        {
            j.WriteStartObject();
            long offset = 0;
            foreach (var (name, shape, data) in tensors)
            {
                long bytes = (long)data.Length * sizeof(float);
                j.WriteStartObject(name);
                j.WriteString("dtype", "F32");
                j.WriteStartArray("shape");
                foreach (int d in shape) j.WriteNumberValue(d);
                j.WriteEndArray();
                j.WriteStartArray("data_offsets"); j.WriteNumberValue(offset); j.WriteNumberValue(offset + bytes); j.WriteEndArray();
                j.WriteEndObject();
                offset += bytes;
            }
            j.WriteEndObject();
        }
        byte[] json = ms.ToArray();
        int pad = (8 - json.Length % 8) % 8;
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);
        bw.Write((ulong)(json.Length + pad));
        bw.Write(json);
        for (int i = 0; i < pad; i++) bw.Write((byte)' ');
        foreach (var (_, _, data) in tensors)
            bw.Write(MemoryMarshal.AsBytes(data.AsSpan()));
    }
}
