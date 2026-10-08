using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>Geometry of a random-weight <c>qwen4exp</c> test checkpoint. Defaults are the degenerate-shape-proof HF-tiny geometry.</summary>
/// <remarks>
/// Every ratio the head-broadcast conventions hide behind is deliberately unequal: 2 GDN key heads vs 4 value heads, 4 query heads over
/// 2 KV heads, 3 indexer heads. <see cref="Experts"/> = 512 with a tiny hidden size exercises the 512-expert router (top-10) at test cost.
/// </remarks>
internal sealed record Q4eGeometry
{
    public int Hidden { get; init; } = 64;
    public int Layers { get; init; } = 4;
    public int Heads { get; init; } = 4;
    public int KvHeads { get; init; } = 2;
    public int HeadDim { get; init; } = 32;
    public int RopeDim { get; init; } = 8;
    public int Experts { get; init; } = 8;
    public int TopK { get; init; } = 3;
    public int MoeInter { get; init; } = 32;
    public int SharedInter { get; init; } = 32;
    public int Nk { get; init; } = 2;
    public int Nv { get; init; } = 4;
    public int DState { get; init; } = 16;
    public int HcLowRank { get; init; } = 20;
    public int IdxHeads { get; init; } = 3;
    public int IdxDim { get; init; } = 16;
    public int Block { get; init; } = 4;
    public int Budget { get; init; } = 256;
    public int Vocab { get; init; } = 100;
    public int Context { get; init; } = 512;
    /// <summary>Rows of the n-gram table; the four head ranges are carved out of it.</summary>
    public int TableRows { get; init; } = 64;
    public int Streams => 4;
    public int PleHeadsPerNgram => 2;
    public int PleRowDim => Hidden / (2 * PleHeadsPerNgram);
}

/// <summary>Quantization choices per tensor family (<c>F32</c> = the family stays unquantised).</summary>
internal sealed record Q4eQuant
{
    public QuantizationType ExpertGateUp { get; init; } = QuantizationType.F32;
    public QuantizationType ExpertDown { get; init; } = QuantizationType.F32;
    public QuantizationType Proj { get; init; } = QuantizationType.F32;
    public QuantizationType HcDown { get; init; } = QuantizationType.F32;
    public QuantizationType Embed { get; init; } = QuantizationType.F32;
    public QuantizationType Table { get; init; } = QuantizationType.F32;
    public QuantizationType Indexer { get; init; } = QuantizationType.F32;

    public static Q4eQuant F32 => new();
    /// <summary>Q8_0 projections / embedding, Q8_0 gate+up and Q5_1 down experts (the real file's fallback-banks), Q4_K hc-down where K allows.</summary>
    public static Q4eQuant Q8Q51 => new()
    {
        ExpertGateUp = QuantizationType.Q8_0, ExpertDown = QuantizationType.Q5_1, Proj = QuantizationType.Q8_0,
        HcDown = QuantizationType.Q4_K, Embed = QuantizationType.Q8_0, Table = QuantizationType.F16, Indexer = QuantizationType.BF16,
    };
    /// <summary>The K-quant resident path: Q4_K gate/up and down (the test quantizer only writes Q4_K; needs hidden and moe_inter multiples of 256).</summary>
    public static Q4eQuant KQuant => new()
    {
        ExpertGateUp = QuantizationType.Q4_K, ExpertDown = QuantizationType.Q4_K, Proj = QuantizationType.Q8_0,
        HcDown = QuantizationType.Q4_K, Embed = QuantizationType.Q8_0, Table = QuantizationType.F16, Indexer = QuantizationType.BF16,
    };
}

/// <summary>
/// Random-weight <c>qwen4exp</c> GGUF writer for GPU-vs-CPU-oracle parity (issue #818). The CPU oracle (<c>Qwen4ExpTransformerModel</c>) is
/// the reference, so no HF dump is needed and the geometry/quant can be anything the contract allows. Weights are scaled so activations
/// stay O(1) through the gated residual and MoE (a degenerate near-constant model would pass any parity test).
/// </summary>
internal static class Qwen4ExpRandomGguf
{
    public static byte[] Build(Q4eGeometry g, Q4eQuant q, uint seed = 0xA11CEu)
    {
        const string arch = "qwen4exp";
        var rng = new Random(unchecked((int)seed));
        double Gauss() { double u = 1 - rng.NextDouble(), v = rng.NextDouble(); return Math.Sqrt(-2 * Math.Log(u)) * Math.Cos(2 * Math.PI * v); }
        float[] Randn(long n, float scale, float offset = 0f)
        {
            var a = new float[n];
            for (long i = 0; i < n; i++) a[i] = offset + scale * (float)Gauss();
            return a;
        }

        int H = g.Hidden, S = g.Streams, hcDim = S * H;
        int gdnV = g.Nv * g.DState, gdnK = g.Nk * g.DState, convDim = 2 * gdnK + gdnV;
        int numPleHeads = 2 * g.PleHeadsPerNgram;
        var w = new GgufWriter();
        w.AddString("general.architecture", arch);
        w.AddString("general.name", "random-qwen4exp");
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.block_count", (uint)g.Layers);
        w.AddUInt32($"{arch}.context_length", (uint)g.Context);
        w.AddUInt32($"{arch}.embedding_length", (uint)H);
        w.AddUInt32($"{arch}.attention.head_count", (uint)g.Heads);
        w.AddUInt32($"{arch}.attention.head_count_kv", (uint)g.KvHeads);
        w.AddInt32Array($"{arch}.rope.dimension_sections", [1, 1, 2, 0]);
        w.AddFloat32($"{arch}.rope.freq_base", 1.0e7f);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
        w.AddUInt32($"{arch}.expert_count", (uint)g.Experts);
        w.AddUInt32($"{arch}.expert_used_count", (uint)g.TopK);
        w.AddUInt32($"{arch}.attention.key_length", (uint)g.HeadDim);
        w.AddUInt32($"{arch}.attention.value_length", (uint)g.HeadDim);
        w.AddUInt32($"{arch}.expert_feed_forward_length", (uint)g.MoeInter);
        w.AddUInt32($"{arch}.expert_shared_feed_forward_length", (uint)g.SharedInter);
        w.AddUInt32($"{arch}.ssm.conv_kernel", 4);
        w.AddUInt32($"{arch}.ssm.state_size", (uint)g.DState);
        w.AddUInt32($"{arch}.ssm.group_count", (uint)g.Nk);
        w.AddUInt32($"{arch}.ssm.time_step_rank", (uint)g.Nv);
        w.AddUInt32($"{arch}.ssm.inner_size", (uint)gdnV);
        w.AddUInt32($"{arch}.full_attention_interval", 4);
        w.AddUInt32($"{arch}.rope.dimension_count", (uint)g.RopeDim);
        w.AddUInt32($"{arch}.hyper_connection.count", (uint)S);
        w.AddUInt32($"{arch}.hyper_connection.low_rank", (uint)g.HcLowRank);
        w.AddUInt32($"{arch}.attention.indexer.head_count", (uint)g.IdxHeads);
        w.AddUInt32($"{arch}.attention.indexer.key_length", (uint)g.IdxDim);
        w.AddUInt32($"{arch}.attention.indexer.top_k", (uint)g.Budget);
        var ratios = new int[g.Layers];
        for (int i = 0; i < g.Layers; i++) ratios[i] = (i + 1) % 4 == 0 ? g.Block : 0;
        w.AddInt32Array($"{arch}.attention.compress_ratios", ratios);

        // PLE on block 1 (zero-based), four head ranges carved out of TableRows.
        ulong per = (ulong)(g.TableRows / numPleHeads);
        var vocabs = Enumerable.Repeat(per, numPleHeads).ToArray();
        var offsets = Enumerable.Range(0, numPleHeads).Select(i => (ulong)i * per).ToArray();
        w.AddInt32Array($"{arch}.ple.layers", [1]);
        w.AddUInt32($"{arch}.ple.ngram_size", 3);
        w.AddUInt32($"{arch}.ple.heads_per_ngram", (uint)g.PleHeadsPerNgram);
        w.AddUInt32($"{arch}.ple.conv_kernel", 4);
        w.AddUInt32($"{arch}.ple.eos_token_id", 5);
        w.AddUInt32($"{arch}.embedding_length_per_layer_input", (uint)g.PleRowDim);
        w.AddUInt64Array($"{arch}.ple.layer_multipliers", [23703573157769UL, 20109073645365UL, 18446744073709551557UL]);   // the last is above 2^63
        w.AddUInt64Array($"{arch}.ple.head_offsets", offsets);
        w.AddUInt64Array($"{arch}.ple.head_vocab_sizes", vocabs);

        w.AddString("tokenizer.ggml.model", "llama");
        w.AddStringArray("tokenizer.ggml.tokens", Enumerable.Range(0, g.Vocab).Select(i => i == 5 ? "<eos>" : $"tok{i}").ToArray());
        w.AddFloat32Array("tokenizer.ggml.scores", new float[g.Vocab]);
        w.AddInt32Array("tokenizer.ggml.token_type", Enumerable.Repeat(1, g.Vocab).ToArray());
        w.AddUInt32("tokenizer.ggml.bos_token_id", 1);
        w.AddUInt32("tokenizer.ggml.eos_token_id", 5);
        w.AddUInt32("tokenizer.ggml.unknown_token_id", 0);

        void Add(string name, int[] dims, float[] data, QuantizationType qt)
        {
            // Quantised rows need ne0 to be a block multiple; otherwise the family silently stays F32 (the contract allows any type per tensor).
            int block = qt switch { QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K => 256, QuantizationType.F32 or QuantizationType.F16 or QuantizationType.BF16 => 1, _ => 32 };
            if (dims[0] % block != 0) qt = QuantizationType.F32;
            byte[] bytes;
            if (qt == QuantizationType.F32)
            {
                bytes = new byte[data.Length * 4];
                Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            }
            else if (qt == QuantizationType.BF16)
            {
                bytes = new byte[data.Length * 2];
                for (int i = 0; i < data.Length; i++)
                {
                    uint bits = BitConverter.SingleToUInt32Bits(data[i]);
                    ushort bf = (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
                    bytes[2 * i] = (byte)bf; bytes[2 * i + 1] = (byte)(bf >> 8);
                }
            }
            else if (qt == QuantizationType.F16)
            {
                bytes = new byte[data.Length * 2];
                for (int i = 0; i < data.Length; i++)
                    BitConverter.TryWriteBytes(bytes.AsSpan(2 * i), (Half)data[i]);
            }
            else bytes = Quantize.FromFloat32(data, data.Length, qt);
            w.AddTensor(name, dims, (uint)qt, bytes);
        }
        void Mat(string name, int ne0, int ne1, QuantizationType qt, float gain = 1f)
            => Add(name, [ne0, ne1], Randn((long)ne0 * ne1, gain / MathF.Sqrt(ne0)), qt);
        void Norm(string name, int n) => Add(name, [n], Randn(n, 0.05f, 1f), QuantizationType.F32);

        Mat("token_embd.weight", H, g.Vocab, q.Embed, gain: 1.5f * MathF.Sqrt(H));
        Mat("output.weight", H, g.Vocab, q.Embed, gain: 3f);
        Norm("output_hc_norm.weight", hcDim);
        Mat("output_hc_down.weight", hcDim, g.HcLowRank, q.HcDown);
        Mat("output_hc_up.weight", g.HcLowRank, hcDim, QuantizationType.F32, gain: 2f);
        Add("per_layer_token_embd.weight", [g.PleRowDim, g.TableRows], Randn((long)g.PleRowDim * g.TableRows, 0.5f), q.Table);

        for (int il = 0; il < g.Layers; il++)
        {
            string b = $"blk.{il}.";
            bool attention = (il + 1) % 4 == 0;
            foreach (string m in new[] { "hc_attn_", "hc_ffn_" })
            {
                Norm(b + m + "norm.weight", hcDim);
                Mat(b + m + "down.weight", hcDim, g.HcLowRank, q.HcDown);
                Mat(b + m + "up.weight", g.HcLowRank, hcDim, QuantizationType.F32, gain: 2f);
                Add(b + m + "inject.weight", [hcDim, S], Randn((long)hcDim * S, 1f / MathF.Sqrt(hcDim)), QuantizationType.F32);
            }

            if (attention)
            {
                Mat(b + "attn_q.weight", H, 2 * g.Heads * g.HeadDim, q.Proj, gain: 2f);
                Mat(b + "attn_k.weight", H, g.KvHeads * g.HeadDim, q.Proj, gain: 2f);
                Mat(b + "attn_v.weight", H, g.KvHeads * g.HeadDim, q.Proj);
                Mat(b + "attn_output.weight", g.Heads * g.HeadDim, H, q.Proj);
                Norm(b + "attn_q_norm.weight", g.HeadDim);
                Norm(b + "attn_k_norm.weight", g.HeadDim);
                Mat(b + "indexer.q_proj.weight", H, g.IdxHeads * g.IdxDim, q.Indexer);
                Mat(b + "indexer.k_proj.weight", H, g.IdxDim, q.Indexer);
                Norm(b + "indexer.q_norm.weight", g.IdxDim);
                Norm(b + "indexer.k_norm.weight", g.IdxDim);
            }
            else
            {
                Mat(b + "attn_qkv.weight", H, convDim, q.Proj);
                Mat(b + "attn_gate.weight", H, gdnV, q.Proj);
                Add(b + "ssm_conv1d.weight", [4, convDim], Randn(4L * convDim, 0.4f), QuantizationType.F32);
                Add(b + "ssm_dt.bias", [g.Nv], Randn(g.Nv, 0.1f), QuantizationType.F32);
                var a = Randn(g.Nv, 0.25f, -0.7f);
                for (int i = 0; i < a.Length; i++) a[i] = -MathF.Abs(a[i]) - 0.1f;   // decay base must be negative
                Add(b + "ssm_a", [g.Nv], a, QuantizationType.F32);
                Mat(b + "ssm_beta.weight", H, g.Nv, QuantizationType.F32);
                Mat(b + "ssm_alpha.weight", H, g.Nv, QuantizationType.F32);
                Norm(b + "ssm_norm.weight", g.DState);
                Mat(b + "ssm_out.weight", gdnV, H, q.Proj);
            }

            if (il == 1)
            {
                Mat(b + "ple_key.weight", H, hcDim, q.Proj);
                Mat(b + "ple_value.weight", H, H, q.Proj);
                Norm(b + "ple_norm_key.weight", hcDim);
                Norm(b + "ple_norm_query.weight", hcDim);
                Norm(b + "ple_norm_conv.weight", hcDim);
                Add(b + "ple_conv1d.weight", [4, hcDim], Randn(4L * hcDim, 0.4f), QuantizationType.F32);
            }

            // Router weights scaled up so the top-k boundary has margin (discrete routing flips are the usual parity-noise source).
            Mat(b + "ffn_gate_inp.weight", H, g.Experts, QuantizationType.F32, gain: 4f);
            Add(b + "ffn_down_exps.weight", [g.MoeInter, H, g.Experts], Randn((long)g.MoeInter * H * g.Experts, 1f / MathF.Sqrt(g.MoeInter)), q.ExpertDown);
            Add(b + "ffn_gate_exps.weight", [H, g.MoeInter, g.Experts], Randn((long)H * g.MoeInter * g.Experts, 1f / MathF.Sqrt(H)), q.ExpertGateUp);
            Add(b + "ffn_up_exps.weight", [H, g.MoeInter, g.Experts], Randn((long)H * g.MoeInter * g.Experts, 1f / MathF.Sqrt(H)), q.ExpertGateUp);
            Add(b + "ffn_gate_inp_shexp.weight", [H], Randn(H, 1f / MathF.Sqrt(H)), QuantizationType.F32);
            Mat(b + "ffn_gate_shexp.weight", H, g.SharedInter, q.Proj);
            Mat(b + "ffn_up_shexp.weight", H, g.SharedInter, q.Proj);
            Mat(b + "ffn_down_shexp.weight", g.SharedInter, H, q.Proj);
        }
        return w.Build();
    }

    /// <summary>Degenerate-shape-proof tiny geometry with 8 experts (top-3).</summary>
    public static Q4eGeometry Tiny => new();

    /// <summary>512 experts / top-10 (the released router) on a tiny hidden size.</summary>
    public static Q4eGeometry Experts512 => new() { Experts = 512, TopK = 10 };

    /// <summary>
    /// Hidden size and expert width 256 so Q4_K / Q5_K banks are legal and the MMVQ (T==1) and grouped coopmat (T&gt;=16) kernels engage; 16
    /// experts top-4, 2 key vs 4 value GDN heads, 4 query heads over 2 KV heads.
    /// </summary>
    public static Q4eGeometry KQuant256 => new()
    {
        Hidden = 256, Heads = 4, KvHeads = 2, HeadDim = 64, RopeDim = 16, Experts = 16, TopK = 4, MoeInter = 256, SharedInter = 256,
        Nk = 2, Nv = 4, DState = 32, HcLowRank = 32, IdxHeads = 3, IdxDim = 32, Vocab = 128,
    };
}
