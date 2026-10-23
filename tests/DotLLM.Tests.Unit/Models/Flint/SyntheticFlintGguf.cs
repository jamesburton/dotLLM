using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models.Flint;

/// <summary>
/// Deterministic random-weight <c>flint</c> GGUF writer for hermetic tests (no fixture files). Keys and tensor names follow
/// llama.cpp <c>src/models/flint.cpp</c> (ggml-org/llama.cpp#30928). Every ratio is chosen to be discriminating: GQA 8:2 (a
/// head-broadcast bug cannot hide behind NKHead = 1), Q width 128 wider than hidden 64 (as Flint's 4096 vs 2048), a sliding window
/// of 4 over a 16-token prompt, NoPE on full-attention layers that include a MoE layer (layer 4), sinks on every layer, two leading
/// dense layers, top-3 of 8 sigmoid routing with no selection bias, and one ungated shared expert whose width (64) is
/// <c>expert_shared_count</c> (2) times the expert width (32).
/// </summary>
internal static class SyntheticFlintGguf
{
    public const int Hidden = 64, Layers = 6, Heads = 8, Kv = 2, HeadDim = 16, Window = 4;
    public const int Experts = 8, TopK = 3, MoeFf = 32, SharedCount = 2, SharedFf = MoeFf * SharedCount, DenseFf = 96;
    public const int LeadingDense = 2, Vocab = 40, Context = 1024;

    /// <summary>true = sliding-window layer. Full layers (false) are NoPE.</summary>
    public static readonly bool[] Sliding = [false, false, true, true, false, true];

    public sealed record Options
    {
        public int Seed { get; init; } = 20261023;

        /// <summary>When false every sink is -30 (no effect), to prove the sinks are read.</summary>
        public bool Sinks { get; init; } = true;

        /// <summary>Routed gate/up/down bank type (F32 or Q4_1).</summary>
        public QuantizationType RoutedQuant { get; init; } = QuantizationType.F32;

        /// <summary>Shared-expert projection type (F32, Q8_0 or Q4_1).</summary>
        public QuantizationType SharedQuant { get; init; } = QuantizationType.F32;

        /// <summary>Attention head dim (real Flint: 128). The small default keeps the golden in the test source.</summary>
        public int HeadDim { get; init; } = SyntheticFlintGguf.HeadDim;

        /// <summary>Sliding window (real Flint: 513).</summary>
        public int Window { get; init; } = SyntheticFlintGguf.Window;

        /// <summary>Attention / dense projection type (real Flint: Q4_1).</summary>
        public QuantizationType DenseQuant { get; init; } = QuantizationType.F32;

        /// <summary>Hidden size (real Flint: 2048). Vulkan's packed Q4_1 routed banks need multiples of 256 (#1171).</summary>
        public int Hidden { get; init; } = SyntheticFlintGguf.Hidden;

        /// <summary>Routed expert width (real Flint: 512); the shared expert is <see cref="SharedCount"/> times this.</summary>
        public int MoeFf { get; init; } = SyntheticFlintGguf.MoeFf;

        /// <summary>Leading dense FFN width (real Flint: 6144).</summary>
        public int DenseFf { get; init; } = SyntheticFlintGguf.DenseFf;

        /// <summary>When false the FULL-attention layers' sinks are -30 (no effect) while the sliding layers keep theirs (#1171 mutant).</summary>
        public bool FullLayerSinks { get; init; } = true;

        /// <summary>Std of the router logits weights (smaller = fewer near-tie expert flips between engines).</summary>
        public float RouterStd { get; init; } = 0.6f;

        /// <summary>Multiplier on the routed gate / up bank std: large values push the SwiGLU output (the down input) past the F16 range (#1171).</summary>
        public float ExpertGateUpScale { get; init; } = 1f;
    }

    public static byte[] Build(Options? options = null)
    {
        options ??= new Options();
        var rng = new Random(options.Seed);
        var w = new GgufWriter();
        const string arch = "flint";
        int headDim = options.HeadDim;
        int hidden = options.Hidden, moeFf = options.MoeFf, sharedFf = moeFf * SharedCount, denseFf = options.DenseFf;
        w.AddString("general.architecture", arch);
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.context_length", Context);
        w.AddUInt32($"{arch}.embedding_length", (uint)hidden);
        w.AddUInt32($"{arch}.block_count", Layers);
        w.AddUInt32($"{arch}.feed_forward_length", (uint)denseFf);
        w.AddUInt32($"{arch}.attention.head_count", Heads);
        w.AddUInt32($"{arch}.attention.head_count_kv", Kv);
        w.AddUInt32($"{arch}.attention.key_length", (uint)headDim);
        w.AddUInt32($"{arch}.attention.value_length", (uint)headDim);
        w.AddUInt32($"{arch}.rope.dimension_count", (uint)headDim);
        w.AddFloat32($"{arch}.rope.freq_base", 1e4f);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
        w.AddUInt32($"{arch}.attention.sliding_window", (uint)options.Window);
        w.AddBoolArray($"{arch}.attention.sliding_window_pattern", Sliding);
        w.AddBoolArray($"{arch}.attention.rope_pattern", Sliding);   // RoPE on sliding layers only
        w.AddUInt32($"{arch}.leading_dense_block_count", LeadingDense);
        w.AddUInt32($"{arch}.expert_count", Experts);
        w.AddUInt32($"{arch}.expert_used_count", TopK);
        w.AddUInt32($"{arch}.expert_feed_forward_length", (uint)moeFf);
        w.AddUInt32($"{arch}.expert_shared_count", SharedCount);
        w.AddUInt32($"{arch}.expert_shared_feed_forward_length", (uint)sharedFf);
        w.AddBool($"{arch}.expert_weights_norm", true);
        w.AddFloat32($"{arch}.expert_weights_scale", 1.0f);
        w.AddUInt32($"{arch}.expert_gating_func", 2);

        var tokens = new string[Vocab];
        for (int i = 0; i < Vocab; i++)
            tokens[i] = ((char)('a' + i % 26)).ToString() + (i >= 26 ? ((char)('a' + (i / 26) % 26)).ToString() : "");
        w.AddString("tokenizer.ggml.model", "gpt2");
        w.AddString("tokenizer.ggml.pre", "flint");
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddInt32Array("tokenizer.ggml.token_type", Enumerable.Repeat(1, Vocab).ToArray());
        w.AddStringArray("tokenizer.ggml.merges", []);

        Matrix(w, rng, "token_embd.weight", hidden, Vocab, 0.6f, QuantizationType.F32);
        Norm(w, rng, "output_norm.weight", hidden);
        Matrix(w, rng, "output.weight", hidden, Vocab, 0.25f, QuantizationType.F32);

        int qWidth = Heads * headDim;
        var dq = options.DenseQuant;
        for (int i = 0; i < Layers; i++)
        {
            string p = $"blk.{i}";
            Norm(w, rng, $"{p}.attn_norm.weight", hidden);
            Matrix(w, rng, $"{p}.attn_q.weight", hidden, qWidth, 1.4f / MathF.Sqrt(hidden), dq);
            Matrix(w, rng, $"{p}.attn_k.weight", hidden, Kv * headDim, 1.4f / MathF.Sqrt(hidden), dq);
            Matrix(w, rng, $"{p}.attn_v.weight", hidden, Kv * headDim, 1.4f / MathF.Sqrt(hidden), dq);
            Matrix(w, rng, $"{p}.attn_output.weight", qWidth, hidden, 1.4f / MathF.Sqrt(qWidth), dq);
            Norm(w, rng, $"{p}.attn_q_norm.weight", headDim);
            Norm(w, rng, $"{p}.attn_k_norm.weight", headDim);
            var sinks = new float[Heads];
            for (int h = 0; h < Heads; h++)
            {
                // Draw whenever Sinks is on so FullLayerSinks = false changes only the full layers' sinks, not the rest of the rng stream.
                float g = options.Sinks ? (float)(Gauss(rng) * 1.5) : -30f;
                sinks[h] = Sliding[i] || options.FullLayerSinks ? g : -30f;
            }
            w.AddTensor($"{p}.attn_sinks.weight", [Heads], (uint)QuantizationType.F32, Bytes(sinks));
            Norm(w, rng, $"{p}.ffn_norm.weight", hidden);

            if (i < LeadingDense)
            {
                Matrix(w, rng, $"{p}.ffn_gate.weight", hidden, denseFf, 1.4f / MathF.Sqrt(hidden), dq);
                Matrix(w, rng, $"{p}.ffn_up.weight", hidden, denseFf, 1.4f / MathF.Sqrt(hidden), dq);
                Matrix(w, rng, $"{p}.ffn_down.weight", denseFf, hidden, 1.4f / MathF.Sqrt(denseFf), dq);
                continue;
            }

            Matrix(w, rng, $"{p}.ffn_gate_inp.weight", hidden, Experts, options.RouterStd, QuantizationType.F32);
            Bank(w, rng, $"{p}.ffn_gate_exps.weight", hidden, moeFf, options.ExpertGateUpScale * 1.4f / MathF.Sqrt(hidden), options.RoutedQuant);
            Bank(w, rng, $"{p}.ffn_up_exps.weight", hidden, moeFf, options.ExpertGateUpScale * 1.4f / MathF.Sqrt(hidden), options.RoutedQuant);
            Bank(w, rng, $"{p}.ffn_down_exps.weight", moeFf, hidden, 1.4f / MathF.Sqrt(moeFf), options.RoutedQuant);
            Matrix(w, rng, $"{p}.ffn_gate_shexp.weight", hidden, sharedFf, 1.4f / MathF.Sqrt(hidden), options.SharedQuant);
            Matrix(w, rng, $"{p}.ffn_up_shexp.weight", hidden, sharedFf, 1.4f / MathF.Sqrt(hidden), options.SharedQuant);
            Matrix(w, rng, $"{p}.ffn_down_shexp.weight", sharedFf, hidden, 1.4f / MathF.Sqrt(sharedFf), options.SharedQuant);
        }
        return w.Build();
    }

    private static double Gauss(Random r)
    {
        double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static byte[] Bytes(float[] f) => MemoryMarshal.AsBytes(f.AsSpan()).ToArray();

    private static void Matrix(GgufWriter w, Random rng, string name, int inK, int outM, float std, QuantizationType qt)
    {
        var f = new float[(long)inK * outM];
        for (long i = 0; i < f.Length; i++) f[i] = (float)(Gauss(rng) * std);
        w.AddTensor(name, [inK, outM], (uint)qt, Encode(f, inK, qt));
    }

    private static void Bank(GgufWriter w, Random rng, string name, int inK, int outM, float std, QuantizationType qt)
    {
        var f = new float[(long)inK * outM * Experts];
        for (long i = 0; i < f.Length; i++) f[i] = (float)(Gauss(rng) * std);
        w.AddTensor(name, [inK, outM, Experts], (uint)qt, Encode(f, inK, qt));
    }

    private static unsafe byte[] Encode(float[] data, int rowElems, QuantizationType qt)
    {
        switch (qt)
        {
            case QuantizationType.F32:
                return Bytes(data);
            case QuantizationType.Q4_1:
                return EncodeQ4_1(data);
            case QuantizationType.Q8_0:
            {
                long rows = data.Length / rowElems;
                long rowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(rowElems, QuantizationType.Q8_0);
                var bytes = new byte[rows * rowBytes];
                fixed (float* src = data)
                fixed (byte* dst = bytes)
                {
                    for (long r = 0; r < rows; r++)
                        DotLLM.Cpu.Kernels.MatMul.QuantizeF32ToQ8_0(src + r * rowElems, dst + r * rowBytes, rowElems);
                }
                return bytes;
            }
            default:
                throw new NotSupportedException($"Synthetic Flint writer has no encoder for {qt}.");
        }
    }

    /// <summary>ggml block_q4_1: f16 d, f16 m, 16 bytes of nibbles (element j low, j + 16 high); x = q * d + m.</summary>
    internal static byte[] EncodeQ4_1(float[] data)
    {
        const int qk = 32, blockBytes = 20;
        var bytes = new byte[data.Length / qk * blockBytes];
        for (int b = 0; b < data.Length / qk; b++)
        {
            var x = data.AsSpan(b * qk, qk);
            float min = float.MaxValue, max = float.MinValue;
            foreach (float v in x) { min = MathF.Min(min, v); max = MathF.Max(max, v); }
            float d = (max - min) / 15f;
            float id = d != 0 ? 1f / d : 0f;
            int o = b * blockBytes;
            BitConverter.TryWriteBytes(bytes.AsSpan(o, 2), (Half)d);
            BitConverter.TryWriteBytes(bytes.AsSpan(o + 2, 2), (Half)min);
            for (int j = 0; j < qk / 2; j++)
            {
                int lo = Math.Min(15, (int)((x[j] - min) * id + 0.5f));
                int hi = Math.Min(15, (int)((x[j + qk / 2] - min) * id + 0.5f));
                bytes[o + 4 + j] = (byte)(lo | (hi << 4));
            }
        }
        return bytes;
    }

    private static void Norm(GgufWriter w, Random rng, string name, int n)
    {
        var f = new float[n];
        for (int i = 0; i < n; i++) f[i] = 1.0f + 0.2f * (float)Gauss(rng);
        w.AddTensor(name, [n], (uint)QuantizationType.F32, Bytes(f));
    }
}
