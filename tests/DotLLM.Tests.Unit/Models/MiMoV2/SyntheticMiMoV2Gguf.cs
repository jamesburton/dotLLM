using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models.MiMoV2;

/// <summary>
/// Deterministic random-weight <c>mimo2</c> GGUF writer for hermetic tests (no fixture files). The geometry mirrors the
/// HF/llama.cpp oracle model of <c>tools/mimo2-tiny-oracle/make_tiny.py</c> with a 40-token vocabulary so the expected
/// llama.cpp logits fit in the test source: every ratio is distinct (K 48 vs V 32, rotary 16 of 48, GQA 4 on global layers vs 2 on
/// sliding layers, window 4 over a longer prompt, 2 global + 4 sliding layers, dense layer 0, 3 trailing NextN blocks that the
/// trunk must ignore).
/// </summary>
internal static class SyntheticMiMoV2Gguf
{
    public const int Hidden = 64, Layers = 6, NextN = 3, Heads = 8, GlobalKv = 2, SlidingKv = 4, KDim = 48, VDim = 32, RopeDim = 16;
    public const int Window = 4, Experts = 8, TopK = 2, MoeFf = 32, DenseFf = 96, Vocab = 40, Context = 256;
    public static readonly int[] Pattern = [0, 1, 1, 1, 0, 1];   // 1 = sliding-window layer

    public sealed record Options
    {
        public int Seed { get; init; } = 20260713;
        /// <summary>Scale applied to attn_sinks (0 draws a constant very negative sink = "no sink").</summary>
        public bool Sinks { get; init; } = true;
        public bool WriteTokenizer { get; init; } = true;
        /// <summary>Hidden size (default 64). IQ / K-quant banks need a multiple of 256.</summary>
        public int HiddenSize { get; init; } = Hidden;
        /// <summary>Per-expert FFN width (default 32). IQ / K-quant down banks need a multiple of 256.</summary>
        public int ExpertFf { get; init; } = MoeFf;
        /// <summary>Dense FFN width (layer 0 and the MTP blocks).</summary>
        public int DenseFfWidth { get; init; } = DenseFf;
        /// <summary>Routed gate/up bank type (F32 unless a packed-bank test says otherwise; random blocks for the IQ types).</summary>
        public QuantizationType GateUpQuant { get; init; } = QuantizationType.F32;
        /// <summary>Routed down bank type.</summary>
        public QuantizationType DownQuant { get; init; } = QuantizationType.F32;
        /// <summary>Router weight std (0.6 reproduces the llama.cpp golden). Large hidden sizes need ~2.4/sqrt(hidden): sigmoid saturates otherwise and
        /// near-1.0 ties make CPU and GPU pick different experts (a fixture artefact, not an engine difference).</summary>
        public float RouterStd { get; init; } = 0.6f;
    }

    public static byte[] Build(Options? options = null)
    {
        options ??= new Options();
        var rng = new Random(options.Seed);
        var w = new GgufWriter();
        int hidden = options.HiddenSize, moeFf = options.ExpertFf, denseFf = options.DenseFfWidth;
        const string arch = "mimo2";
        int blocks = Layers + NextN;
        w.AddString("general.architecture", arch);
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.context_length", Context);
        w.AddUInt32($"{arch}.embedding_length", (uint)hidden);
        w.AddUInt32($"{arch}.block_count", (uint)blocks);
        w.AddUInt32($"{arch}.feed_forward_length", (uint)denseFf);
        w.AddUInt32($"{arch}.attention.head_count", Heads);
        w.AddInt32Array($"{arch}.attention.head_count_kv", Enumerable.Range(0, blocks).Select(i => IsSliding(i) ? SlidingKv : GlobalKv).ToArray());
        w.AddFloat32($"{arch}.rope.freq_base", 1e7f);
        w.AddFloat32($"{arch}.rope.freq_base_swa", 1e4f);
        w.AddUInt32($"{arch}.expert_used_count", TopK);
        w.AddUInt32($"{arch}.expert_group_count", 1);
        w.AddUInt32($"{arch}.expert_group_used_count", 1);
        w.AddUInt32($"{arch}.expert_gating_func", 2);
        w.AddUInt32($"{arch}.attention.key_length", KDim);
        w.AddUInt32($"{arch}.attention.value_length", VDim);
        w.AddUInt32($"{arch}.attention.sliding_window", Window);
        w.AddInt32Array($"{arch}.attention.sliding_window_pattern", Enumerable.Range(0, blocks).Select(i => IsSliding(i) ? 1 : 0).ToArray());
        w.AddUInt32($"{arch}.expert_count", Experts);
        w.AddUInt32($"{arch}.expert_feed_forward_length", (uint)moeFf);
        w.AddUInt32($"{arch}.rope.dimension_count", RopeDim);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
        w.AddFloat32($"{arch}.attention.value_scale", 0.707f);
        w.AddUInt32($"{arch}.nextn_predict_layers", NextN);

        if (options.WriteTokenizer)
        {
            var tokens = new string[Vocab];
            for (int i = 0; i < Vocab; i++) tokens[i] = ((char)('a' + i % 26)).ToString() + (i >= 26 ? ((char)('a' + (i / 26) % 26)).ToString() : "");
            w.AddString("tokenizer.ggml.model", "gpt2");
            w.AddString("tokenizer.ggml.pre", "qwen2");
            w.AddStringArray("tokenizer.ggml.tokens", tokens);
            w.AddInt32Array("tokenizer.ggml.token_type", Enumerable.Repeat(1, Vocab).ToArray());
            w.AddStringArray("tokenizer.ggml.merges", []);
        }

        Matrix(w, rng, "token_embd.weight", hidden, Vocab, 0.6f);
        Norm(w, rng, "output_norm.weight", hidden);
        Matrix(w, rng, "output.weight", hidden, Vocab, 0.25f);

        for (int i = 0; i < blocks; i++)
        {
            string p = $"blk.{i}";
            bool sliding = IsSliding(i);
            bool trunk = i < Layers;
            int kv = sliding ? SlidingKv : GlobalKv;
            Norm(w, rng, $"{p}.attn_norm.weight", hidden);
            Matrix(w, rng, $"{p}.attn_qkv.weight", hidden, Heads * KDim + kv * KDim + kv * VDim, 1.4f / MathF.Sqrt(hidden));
            Matrix(w, rng, $"{p}.attn_output.weight", Heads * VDim, hidden, 1.4f / MathF.Sqrt(Heads * VDim));
            if (sliding)
            {
                var sinks = new float[Heads];
                for (int h = 0; h < Heads; h++) sinks[h] = options.Sinks ? (float)(Gauss(rng) * 1.5) : -30f;
                w.AddTensor($"{p}.attn_sinks.weight", [Heads], (uint)QuantizationType.F32, Bytes(sinks));
            }
            Norm(w, rng, $"{p}.ffn_norm.weight", hidden);
            bool dense = i == 0 || !trunk;
            if (dense)
            {
                Matrix(w, rng, $"{p}.ffn_gate.weight", hidden, denseFf, 1.4f / MathF.Sqrt(hidden));
                Matrix(w, rng, $"{p}.ffn_up.weight", hidden, denseFf, 1.4f / MathF.Sqrt(hidden));
                Matrix(w, rng, $"{p}.ffn_down.weight", denseFf, hidden, 1.4f / MathF.Sqrt(denseFf));
            }
            else
            {
                Matrix(w, rng, $"{p}.ffn_gate_inp.weight", hidden, Experts, options.RouterStd);
                Bank(w, rng, $"{p}.ffn_gate_exps.weight", hidden, moeFf, 1.4f / MathF.Sqrt(hidden), options.GateUpQuant);
                Bank(w, rng, $"{p}.ffn_up_exps.weight", hidden, moeFf, 1.4f / MathF.Sqrt(hidden), options.GateUpQuant);
                Bank(w, rng, $"{p}.ffn_down_exps.weight", moeFf, hidden, 1.4f / MathF.Sqrt(moeFf), options.DownQuant);
                var bias = new float[Experts];
                for (int e = 0; e < Experts; e++) bias[e] = (float)(Gauss(rng) * 0.15);
                w.AddTensor($"{p}.exp_probs_b.bias", [Experts], (uint)QuantizationType.F32, Bytes(bias));
            }
            if (!trunk)
            {
                Matrix(w, rng, $"{p}.nextn.eh_proj.weight", 2 * hidden, hidden, 1.4f / MathF.Sqrt(2 * hidden));
                Norm(w, rng, $"{p}.nextn.enorm.weight", hidden);
                Norm(w, rng, $"{p}.nextn.hnorm.weight", hidden);
                Norm(w, rng, $"{p}.layer_output_norm.weight", hidden);
            }
        }
        return w.Build();
    }

    internal static bool IsSliding(int block) => block >= Pattern.Length || Pattern[block] == 1;

    private static double Gauss(Random r)
    {
        double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static byte[] Bytes(float[] f) => MemoryMarshal.AsBytes(f.AsSpan()).ToArray();

    private static void Matrix(GgufWriter w, Random rng, string name, int inK, int outM, float std)
    {
        var f = new float[(long)inK * outM];
        for (long i = 0; i < f.Length; i++) f[i] = (float)(Gauss(rng) * std);
        w.AddTensor(name, [inK, outM], (uint)QuantizationType.F32, Bytes(f));
    }

    private static void Bank(GgufWriter w, Random rng, string name, int inK, int outM, float std, QuantizationType qt)
    {
        var f = new float[(long)inK * outM * Experts];
        for (long i = 0; i < f.Length; i++) f[i] = (float)(Gauss(rng) * std);
        if (qt == QuantizationType.F32)
            w.AddTensor(name, [inK, outM, Experts], (uint)QuantizationType.F32, Bytes(f));
        else if (qt == QuantizationType.Q8_0)
            w.AddTensor(name, [inK, outM, Experts], (uint)qt, EncodeQ8_0(f, inK));
        else
            w.AddTensor(name, [inK, outM, Experts], (uint)qt, EncodeRandomIq(f, inK, qt));
    }

    /// <summary>
    /// Random IQ blocks (the test tree has no IQ quantizer): any code / sign / scale byte decodes, so the bytes are random with the leading
    /// fp16 super-scale of every block rescaled so the dequantized rms matches the Gaussian the float data was drawn with (same recipe as
    /// the qwen4exp fixtures). CPU and Vulkan read the same bytes, which is all a parity test needs.
    /// </summary>
    private static unsafe byte[] EncodeQ8_0(float[] data, int rowElems)
    {
        long rows = data.Length / rowElems;
        long rowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(rowElems, QuantizationType.Q8_0);
        var bytes = new byte[rows * rowBytes];
        fixed (float* src = data)
        fixed (byte* dst = bytes)
            for (long r = 0; r < rows; r++)
                DotLLM.Cpu.Kernels.MatMul.QuantizeF32ToQ8_0(src + r * rowElems, dst + r * rowBytes, rowElems);
        return bytes;
    }

    private static unsafe byte[] EncodeRandomIq(float[] data, int rowElems, QuantizationType qt)
    {
        long n = data.Length;
        long rowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(rowElems, qt);
        long rows = n / rowElems;
        int blockBytes = DotLLM.Core.Configuration.QuantFormat.TryGetInfo(qt)!.Value.BlockBytes;
        var rng = new Random(unchecked((int)(n * 31 + (int)qt * 7919 + rowElems)));
        var bytes = new byte[rows * rowBytes];
        rng.NextBytes(bytes);
        double target = 0;
        for (long i = 0; i < n; i++) target += (double)data[i] * data[i];
        target = Math.Sqrt(target / n);
        for (long b = 0; b + blockBytes <= bytes.Length; b += blockBytes)
            DotLLM.Tests.Unit.Vulkan.IqBlockScale.Set(bytes, b, blockBytes, (Half)0.01f);
        long probeRows = Math.Min(rows, 8);
        var probe = new float[rowElems];
        double ss = 0;
        fixed (byte* p = bytes)
            for (long r = 0; r < probeRows; r++)
            {
                DotLLM.Cpu.Kernels.Dequantize.ToFloat32((nint)(p + r * rowBytes), rowElems, qt, probe);
                foreach (float v in probe) ss += (double)v * v;
            }
        double rms = Math.Sqrt(ss / (probeRows * rowElems));
        float scale = rms > 0 ? (float)(0.01 * target / rms) : 0.01f;
        for (long b = 0; b + blockBytes <= bytes.Length; b += blockBytes)
            DotLLM.Tests.Unit.Vulkan.IqBlockScale.Set(bytes, b, blockBytes, (Half)scale);
        return bytes;
    }

    private static void Norm(GgufWriter w, Random rng, string name, int n)
    {
        var f = new float[n];
        for (int i = 0; i < n; i++) f[i] = 1.0f + 0.2f * (float)Gauss(rng);
        w.AddTensor(name, [n], (uint)QuantizationType.F32, Bytes(f));
    }
}
