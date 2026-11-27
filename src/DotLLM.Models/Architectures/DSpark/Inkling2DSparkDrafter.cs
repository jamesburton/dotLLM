using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures.DSpark;

/// <summary>
/// The pieces of the Inkling 2 target the DSpark drafter borrows: the token-embedding row (raw, before the target's own norm) and the LM head
/// over the unpadded vocabulary (uncapped logits; the drafter applies the softcap after its Markov bias).
/// </summary>
internal unsafe interface IInkling2DSparkTarget
{
    /// <summary>Dequantises the <c>token_embd</c> row of <paramref name="token"/> into <paramref name="dst"/> (hidden floats), no norm.</summary>
    void EmbedRaw(int token, float* dst);

    /// <summary>Projects final-normed rows <c>[rows, hidden]</c> through the target LM head to UNCAPPED logits <c>[rows, vocab]</c>.</summary>
    void ProjectHead(float* hidden, int rows, float* logits);
}

/// <summary>
/// The Inkling 2 DSpark drafter (llama.cpp <c>dflash</c> GGUF with a Markov head, <c>rnn_hidden</c> variant), issue #2039. Semantics:
/// <c>src/models/dspark.cpp</c>, <c>dflash.cpp</c>, <c>common/speculative.cpp</c> and <c>conversion/inkling2.py</c> (see <c>docs/INKLING2.md</c> section 13).
/// </summary>
/// <remarks>
/// Weights stay in their stored types (Q8_0 projections, BF16 Markov/joint matrices) and are read in place. All activations are native scratch, F32 only
/// within a call; the per-position K/V context is F16 (<see cref="Inkling2DSparkState"/>).
/// </remarks>
internal sealed unsafe class Inkling2DSparkDrafter : IDisposable
{
    internal readonly record struct Mat(nint Ptr, QuantizationType Qt, int In, int Out);

    private sealed class Layer
    {
        public required nint AttnNorm, FfnNorm, QNorm, KNorm, KCtxNorm, CtxMix, CtxNorm;   // F32 vectors (native)
        public required Mat Q, K, V, O, KCtx, VCtx, Gate, Up, Down;
        public required bool Local;
    }

    private readonly GgufFile? _ownedFile;
    private readonly ComputeThreadPool? _pool;
    private readonly List<nint> _native = new();
    private readonly Layer[] _layers;
    private readonly Mat _w1, _w2, _joint;
    private readonly nint _w2Bias, _jointBias, _frnnMix, _frnnNorm, _outNorm, _tokNorm, _maskEmbd;
    private readonly int _ffn, _rank, _vocab, _maskRows;
    private readonly float[] _invFreq;

    /// <summary>Hidden size.</summary>
    public int HiddenSize { get; }
    /// <summary>Trained block size B (anchor + B-1 drafts).</summary>
    public int BlockSize { get; }
    /// <summary>Drafter layer count.</summary>
    public int NumLayers => _layers.Length;
    /// <summary>Query heads.</summary>
    public int NumHeads { get; }
    /// <summary>KV heads.</summary>
    public int NumKvHeads { get; }
    /// <summary>Head dimension.</summary>
    public int HeadDim { get; }
    /// <summary>Sliding window of the local layers (llama.cpp <c>n_swa</c>: a key is visible iff <c>anchor - key &lt; Window</c>).</summary>
    public int Window { get; }
    /// <summary>RMSNorm epsilon.</summary>
    public float Eps { get; }
    /// <summary>Final logit softcap, applied AFTER the Markov bias.</summary>
    public float Softcap { get; }
    /// <summary>Target layer VISITS whose layer input is tapped (the last one, = the visit count, is the final normed output).</summary>
    public int[] TargetVisits { get; }
    /// <summary>Width of one captured tap row (<c>targets x hidden</c>).</summary>
    public int TapRowWidth => TargetVisits.Length * HiddenSize;
    /// <summary>Markov rank.</summary>
    public int Rank => _rank;
    /// <summary>Rows of the <c>f_rnn</c> ring a state keeps.</summary>
    public int FrnnRows => 16;

    /// <summary>Test-only: deliberately wrong variants the mutant tests run (see <see cref="DSparkMutant"/>).</summary>
    internal DSparkMutant Mutant { get; set; }

    private const int InjectChunk = 64;

    /// <summary>Rows one injection call writes at most (the ring slack of the windowed layers' context K/V).</summary>
    internal const int InjectChunkRows = InjectChunk;

    /// <summary>True for the sliding-window layers (the ones whose context K/V is a ring).</summary>
    internal bool IsLocalLayer(int layer) => _layers[layer].Local;
    private readonly NativeBuffer<float> _x, _norm, _q, _k, _v, _attn, _o, _gate, _up, _hid, _logits, _bias, _feat, _mix, _tmpK, _tmpV, _taps32, _joinIn, _jp, _w1Row;

    /// <summary>Drafts per round (cap), <c>DOTLLM_DSPARK_DRAFT</c> or 3 (the value llama.cpp's docs use for Inkling 2).</summary>
    public int DraftTokensPerRound
        => Math.Min(BlockSize - 1, int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_DSPARK_DRAFT"), out int n) && n >= 1 ? n : 3);

    /// <summary>Opens a DSpark GGUF and binds it to a target of <paramref name="targetHidden"/> width, <paramref name="targetVisits"/> layer visits and <paramref name="targetVocab"/> tokens.</summary>
    public static Inkling2DSparkDrafter Open(string path, int targetHidden, int targetVisits, int targetVocab, ComputeThreadPool? pool)
    {
        var file = GgufFile.Open(path);
        try { return new Inkling2DSparkDrafter(file, true, targetHidden, targetVisits, targetVocab, pool); }
        catch { file.Dispose(); throw; }
    }

    /// <summary>True when <paramref name="path"/> is an Inkling-style DSpark GGUF (dflash arch with a Markov head); no tensors are read.</summary>
    public static bool IsDSparkFile(string path)
    {
        try
        {
            using var f = GgufFile.Open(path);
            return f.Metadata.GetStringOrDefault("general.architecture") == "dflash" && f.TensorsByName.ContainsKey("markov_w1.weight");
        }
        catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException) { return false; }
    }

    internal Inkling2DSparkDrafter(GgufFile file, bool ownsFile, int targetHidden, int targetVisits, int targetVocab, ComputeThreadPool? pool)
    {
        _pool = pool;
        var md = file.Metadata;
        if (md.GetStringOrDefault("general.architecture") != "dflash") throw new NotSupportedException("Not a DFlash-family drafter GGUF.");
        var t = file.TensorsByName;
        if (!t.ContainsKey("markov_w1.weight") || !t.ContainsKey("joint_proj.weight") || !t.ContainsKey("mask_embd.weight") || !t.ContainsKey("blk.0.ctx_mix.weight"))
            throw new NotSupportedException("Not an Inkling 2 DSpark drafter (needs markov_w1, joint_proj, mask_embd and per-layer ctx_mix).");
        if (!md.GetBool("dflash.has_markov_context")) throw new NotSupportedException("Only the rnn_hidden (has_markov_context) DSpark head is supported.");
        if (md.ContainsKey("dflash.sample_from_anchor") && md.GetBool("dflash.sample_from_anchor")) throw new NotSupportedException("sample_from_anchor DSpark drafters are not supported.");
        if (md.ContainsKey("dflash.has_confidence_head") && md.GetBool("dflash.has_confidence_head")) throw new NotSupportedException("Confidence-head DSpark drafters are not supported.");
        if (t.ContainsKey("d2t")) throw new NotSupportedException("Reduced-vocabulary DSpark drafters are not supported.");

        HiddenSize = (int)md.GetUInt32("dflash.embedding_length");
        if (HiddenSize != targetHidden) throw new InvalidOperationException($"Drafter hidden size {HiddenSize} does not match the target's {targetHidden}.");
        int nLayer = (int)md.GetUInt32("dflash.block_count");
        NumHeads = (int)md.GetUInt32("dflash.attention.head_count");
        NumKvHeads = (int)md.GetUInt32("dflash.attention.head_count_kv");
        HeadDim = (int)md.GetUInt32("dflash.attention.key_length");
        _ffn = (int)md.GetUInt32("dflash.feed_forward_length");
        Eps = md.GetFloat32("dflash.attention.layer_norm_rms_epsilon");
        BlockSize = (int)md.GetUInt32("dflash.block_size");
        Window = (int)md.GetUInt32("dflash.attention.sliding_window");
        Softcap = md.GetFloat32OrDefault("dflash.final_logit_softcapping", 0f);
        if (md.GetFloat32OrDefault("dflash.logit_scale", 0f) != 0f || md.GetFloat32OrDefault("dflash.embedding_scale", 0f) != 0f)
            throw new NotSupportedException("Drafter logit/embedding scales are not supported (Inkling 2 uses none).");
        float theta = md.GetFloat32OrDefault("dflash.rope.freq_base", 10000f);
        if ((int)md.GetUInt32OrDefault("dflash.rope.dimension_count", (uint)HeadDim) != HeadDim) throw new NotSupportedException("Partial-rotary drafters are not supported.");
        if (md.ContainsKey("dflash.attention.causal") && md.GetBool("dflash.attention.causal")) throw new NotSupportedException("Causal drafters are not supported.");
        TargetVisits = ReadInts(md, "dflash.target_layers");
        if (TargetVisits.Length < 2 || TargetVisits[^1] != targetVisits)
            throw new InvalidOperationException($"Drafter target_layers {string.Join(",", TargetVisits)} must end at the target's {targetVisits} layer visits (the final normed output).");
        bool[] swa = ReadBools(md, "dflash.sliding_window_pattern", nLayer);
        _vocab = targetVocab;

        _invFreq = new float[HeadDim / 2];
        for (int i = 0; i < _invFreq.Length; i++) _invFreq[i] = (float)(1.0 / Math.Pow(theta, 2.0 * i / HeadDim));

        int qDim = NumHeads * HeadDim, kvDim = NumKvHeads * HeadDim, nT = TargetVisits.Length;
        Mat M(string name, int @in, int @out) => ReadMat(file, name, @in, @out);
        nint V(string name, int count) => ReadVec(file, name, count);

        _layers = new Layer[nLayer];
        for (int i = 0; i < nLayer; i++)
        {
            string B(string s) => $"blk.{i}.{s}";
            foreach (string bad in new[] { "attn_post_norm", "ffn_post_norm", "layer_output_scale", "rope_freqs", "attn_sinks" })
                if (t.ContainsKey(B(bad + ".weight"))) throw new NotSupportedException($"Drafter tensor {B(bad)} is not supported.");
            _layers[i] = new Layer
            {
                AttnNorm = V(B("attn_norm.weight"), HiddenSize), FfnNorm = V(B("ffn_norm.weight"), HiddenSize),
                QNorm = V(B("attn_q_norm.weight"), HeadDim), KNorm = V(B("attn_k_norm.weight"), HeadDim), KCtxNorm = V(B("attn_k_ctx_norm.weight"), HeadDim),
                CtxMix = V(B("ctx_mix.weight"), nT), CtxNorm = V(B("ctx_norm.weight"), HiddenSize),
                Q = M(B("attn_q.weight"), HiddenSize, qDim), K = M(B("attn_k.weight"), HiddenSize, kvDim), V = M(B("attn_v.weight"), HiddenSize, kvDim),
                O = M(B("attn_output.weight"), qDim, HiddenSize), KCtx = M(B("attn_k_ctx.weight"), HiddenSize, kvDim), VCtx = M(B("attn_v_ctx.weight"), HiddenSize, kvDim),
                Gate = M(B("ffn_gate.weight"), HiddenSize, _ffn), Up = M(B("ffn_up.weight"), HiddenSize, _ffn), Down = M(B("ffn_down.weight"), _ffn, HiddenSize),
                Local = swa[i],
            };
        }

        _outNorm = V("output_norm.weight", HiddenSize);
        _tokNorm = V("token_embd_norm.weight", HiddenSize);
        _frnnMix = V("markov_ctx_mix.weight", nT);
        _frnnNorm = V("markov_ctx_norm.weight", HiddenSize);
        _rank = t["markov_w1.weight"].Shape[0];
        _w1 = M("markov_w1.weight", _rank, _vocab);
        _w2 = M("markov_w2.weight", _rank, _vocab);
        _w2Bias = t.ContainsKey("markov_w2.bias") ? V("markov_w2.bias", _vocab) : 0;
        if (t.ContainsKey("markov_w2.scale")) throw new NotSupportedException("markov_w2.scale is not supported.");
        _joint = M("joint_proj.weight", 2 * HiddenSize + 2 * _rank, 3 * _rank);
        _jointBias = V("joint_proj.bias", 3 * _rank);
        _maskRows = t["mask_embd.weight"].Shape[1];
        if (_maskRows < BlockSize - 1) throw new InvalidOperationException($"mask_embd has {_maskRows} rows, the block needs {BlockSize - 1}.");
        _maskEmbd = V("mask_embd.weight", HiddenSize * _maskRows);

        _ownedFile = ownsFile ? file : null;
        int b = BlockSize;
        NativeBuffer<float> F(long n) => new(n);
        _x = F((long)b * HiddenSize); _norm = F((long)b * HiddenSize); _q = F((long)b * qDim); _k = F((long)b * kvDim); _v = F((long)b * kvDim);
        _attn = F((long)b * qDim); _o = F((long)b * HiddenSize); _gate = F((long)b * _ffn); _up = F((long)b * _ffn); _hid = F((long)b * HiddenSize);
        _logits = F((long)(b - 1) * _vocab); _bias = F(_vocab);
        _feat = F((long)InjectChunk * HiddenSize); _mix = F((long)InjectChunk * HiddenSize);
        _tmpK = F((long)InjectChunk * kvDim); _tmpV = F((long)InjectChunk * kvDim); _taps32 = F((long)InjectChunk * nT * HiddenSize);
        _joinIn = F(2 * HiddenSize + 2 * _rank); _jp = F(3 * _rank); _w1Row = F(_rank);
    }

    private static Mat ReadMat(GgufFile file, string name, int expectIn, int expectOut)
    {
        if (!file.TensorsByName.TryGetValue(name, out var d)) throw new InvalidOperationException($"Drafter is missing tensor '{name}'.");
        int inDim = d.Shape[0], outDim = d.Shape.Rank > 1 ? d.Shape[1] : 1;
        if (inDim != expectIn || outDim != expectOut) throw new InvalidOperationException($"Drafter tensor '{name}' is [{inDim}, {outDim}], expected [{expectIn}, {expectOut}].");
        string stem = name.EndsWith(".weight", StringComparison.Ordinal) ? name[..^".weight".Length] : name;
        if (file.TensorsByName.ContainsKey(stem + ".scale") || file.TensorsByName.ContainsKey(stem + ".input_scale"))
            throw new NotSupportedException($"Drafter tensor '{name}' carries a scale; not supported.");
        return new Mat(file.TensorDataPointer(d), d.QuantizationType, inDim, outDim);
    }

    private nint ReadVec(GgufFile file, string name, int count)
    {
        if (!file.TensorsByName.TryGetValue(name, out var d)) throw new InvalidOperationException($"Drafter is missing tensor '{name}'.");
        if (d.Shape.ElementCount != count) throw new InvalidOperationException($"Drafter tensor '{name}' has {d.Shape.ElementCount} elements, expected {count}.");
        nint p = (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
        _native.Add(p);
        Dequantize.ToFloat32(file.TensorDataPointer(d), count, d.QuantizationType, new Span<float>((void*)p, count));
        return p;
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
        if (!md.TryGetValue(key, out var v)) { var all = new bool[n]; Array.Fill(all, true); return all; }
        bool[] r = v.Value switch
        {
            bool[] a => a,
            object[] a => Array.ConvertAll(a, x => Convert.ToBoolean(x, System.Globalization.CultureInfo.InvariantCulture)),
            _ => throw new InvalidOperationException($"'{key}' is not a bool array."),
        };
        if (r.Length != n) throw new InvalidOperationException($"'{key}' has {r.Length} entries, expected {n}.");
        return r;
    }

    /// <summary>Tap slot per target layer VISIT <c>0..visits</c> (-1 = not tapped).</summary>
    public int[] SlotForVisit(int visits)
    {
        var slots = new int[visits + 1];
        Array.Fill(slots, -1);
        for (int k = 0; k < TargetVisits.Length; k++) slots[TargetVisits[k]] = k;
        return slots;
    }

    private void Gemm(in Mat w, float* x, float* y, int rows) => Inkling2TransformerModel.GemmPublic(w.Ptr, w.Qt, x, y, w.Out, w.In, rows, _pool);

    private void RmsRows(float* x, nint weight, int rows, int width)
    {
        for (int r = 0; r < rows; r++)
            Inkling2Ops.RmsNorm(new ReadOnlySpan<float>(x + (long)r * width, width), new ReadOnlySpan<float>((void*)weight, width), Eps, new Span<float>(x + (long)r * width, width));
    }

    private void NormRopeHeads(float* x, int heads, nint norm, ReadOnlySpan<int> positions, bool rope = true)
    {
        int hd = HeadDim;
        Span<float> cos = stackalloc float[hd / 2], sin = stackalloc float[hd / 2];
        for (int t = 0; t < positions.Length; t++)
        {
            for (int i = 0; i < cos.Length; i++)
            {
                double a = (double)positions[t] * _invFreq[i];
                cos[i] = (float)Math.Cos(a); sin[i] = (float)Math.Sin(a);
            }
            for (int h = 0; h < heads; h++)
            {
                var v = new Span<float>(x + ((long)t * heads + h) * hd, hd);
                Inkling2Ops.RmsNorm(v, new ReadOnlySpan<float>((void*)norm, hd), Eps, v);
                if (rope) RoPE.ApplyRotationNeoX(v, cos, sin, hd);
            }
        }
    }

    // ── Context injection ───────────────────────────────────────────────

    /// <summary>
    /// Projects captured tap rows (F16, <see cref="TapRowWidth"/> wide) into the per-layer context K/V of <paramref name="state"/> at
    /// <paramref name="positions"/> and keeps the <c>f_rnn</c> row of each position (llama.cpp <c>build_inject_ctx_mix</c>): per layer
    /// <c>rms(sum_k ctx_mix[k] * tap_k) * ctx_norm</c> -> <c>attn_k_ctx</c> / <c>attn_v_ctx</c>; K gets a per-head RMSNorm (<c>attn_k_ctx_norm</c>) and RoPE, V neither.
    /// <c>ctx_mix</c> is already softmaxed by the converter.
    /// </summary>
    public void Inject(Inkling2DSparkState state, Half* taps, ReadOnlySpan<int> positions)
    {
        int h = HiddenSize, nT = TargetVisits.Length, kvDim = NumKvHeads * HeadDim;
        for (int start = 0; start < positions.Length; start += InjectChunk)
        {
            int n = Math.Min(InjectChunk, positions.Length - start);
            var pos = positions.Slice(start, n);
            for (int k = 0; k < nT; k++)
                for (int r = 0; r < n; r++)
                    TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(taps + (long)(start + r) * TapRowWidth + (long)k * h, h),
                        new Span<float>(_taps32.Ptr + ((long)k * InjectChunk + r) * h, h));   // [tap][row][h]

            // f_rnn of every injected row (the Markov head reads the one of the position before the block).
            MixRows(_frnnMix, n, _mix.Ptr);
            RmsRows(_mix.Ptr, _frnnNorm, n, h);
            for (int r = 0; r < n; r++) state.StoreFrnn(pos[r], new ReadOnlySpan<float>(_mix.Ptr + (long)r * h, h));

            for (int li = 0; li < _layers.Length; li++)
            {
                var l = _layers[li];
                MixRows(l.CtxMix, n, _feat.Ptr);
                RmsRows(_feat.Ptr, l.CtxNorm, n, h);
                Gemm(l.KCtx, _feat.Ptr, _tmpK.Ptr, n);
                Gemm(l.VCtx, _feat.Ptr, _tmpV.Ptr, n);
                NormRopeHeads(_tmpK.Ptr, NumKvHeads, l.KCtxNorm, pos);
                state.WriteKv(li, pos, _tmpK.Ptr, _tmpV.Ptr);
            }
            foreach (int p in pos) state.NoteInjected(p);
        }
    }

    /// <summary>out[r] = sum_k mix[k] * tap_k[r] over the F32 taps staged as <c>[tap][row][h]</c>.</summary>
    private void MixRows(nint mix, int n, float* dst)
    {
        int h = HiddenSize, nT = TargetVisits.Length;
        var m = new ReadOnlySpan<float>((void*)mix, nT);
        if (Mutant == DSparkMutant.DoubleSoftmaxCtxMix)
        {
            var sm = new float[nT]; m.CopyTo(sm);
            TensorPrimitives.SoftMax(sm, sm);
            m = sm;
        }
        for (int r = 0; r < n; r++)
        {
            var o = new Span<float>(dst + (long)r * h, h);
            TensorPrimitives.Multiply(new ReadOnlySpan<float>(_taps32.Ptr + (long)r * h, h), m[0], o);
            for (int k = 1; k < nT; k++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(_taps32.Ptr + ((long)k * InjectChunk + r) * h, h), m[k], o, o);
        }
    }

    // ── Draft ───────────────────────────────────────────────────────────

    /// <summary>Logits of the drafted rows of the last <see cref="Draft"/> (softcapped, <c>[draft, vocab]</c>).</summary>
    public ReadOnlySpan<float> LastDraftLogits(int row) => new(_logits.Ptr + (long)row * _vocab, _vocab);

    /// <summary>
    /// Drafts <paramref name="tokens"/>.Length tokens after <paramref name="anchor"/> (position <paramref name="pos0"/>): ONE pass over the block
    /// <c>[anchor, mask x n]</c> (non-causal, every block row uses the anchor's window), then the Markov head walks the rows left to right.
    /// </summary>
    public void Draft(Inkling2DSparkState state, IInkling2DSparkTarget target, int anchor, int pos0, Span<int> tokens)
    {
        int nDraft = tokens.Length, b = nDraft + 1, h = HiddenSize;
        if (nDraft < 1 || b > BlockSize) throw new ArgumentOutOfRangeException(nameof(tokens));
        int fpos = Mutant == DSparkMutant.FrnnWrongPos ? pos0 - 2 : pos0 - 1;   // the Markov head reads the context state of the token BEFORE the anchor
        if (!state.TryGetFrnn(fpos, out var frnn)) throw new InvalidOperationException($"DSpark has no context features for position {fpos} (the position before the block).");

        float* x = _x.Ptr;
        // Anchor: the raw target embedding through the DRAFTER's token_embd_norm. Masks: mask_embd rows, no norm.
        target.EmbedRaw(anchor, x);
        if (Mutant != DSparkMutant.NoAnchorNorm)
        {
            if (Mutant == DSparkMutant.DoubleAnchorNorm) RmsRows(x, _tokNorm, 1, h);
            RmsRows(x, _tokNorm, 1, h);
        }
        for (int i = 1; i < b; i++) new ReadOnlySpan<float>((float*)_maskEmbd + (long)(i - 1) * h, h).CopyTo(new Span<float>(x + (long)i * h, h));

        Span<int> positions = stackalloc int[b];
        for (int i = 0; i < b; i++) positions[i] = pos0 + i;
        int qDim = NumHeads * HeadDim, kvDim = NumKvHeads * HeadDim;

        for (int li = 0; li < _layers.Length; li++)
        {
            var l = _layers[li];
            CopyNorm(x, _norm.Ptr, b, l.AttnNorm);
            Gemm(l.Q, _norm.Ptr, _q.Ptr, b);
            Gemm(l.K, _norm.Ptr, _k.Ptr, b);
            Gemm(l.V, _norm.Ptr, _v.Ptr, b);
            NormRopeHeads(_q.Ptr, NumHeads, l.QNorm, positions);
            NormRopeHeads(_k.Ptr, NumKvHeads, l.KNorm, positions);
            // The block's K/V go through the F16 cache in llama.cpp before attention reads them.
            RoundThroughHalf(_k.Ptr, b * kvDim); RoundThroughHalf(_v.Ptr, b * kvDim);
            bool windowed = l.Local && Mutant != DSparkMutant.AllLayersGlobal || Mutant == DSparkMutant.AllLayersLocal;
            int lo = windowed ? Math.Max(0, pos0 - Window + 1) : 0;
            if (Mutant == DSparkMutant.PerQueryWindow) lo = windowed ? Math.Max(0, pos0 + 1 - Window + 1) : 0;
            lo = Math.Max(lo, state.FirstContextPosition);
            Attend(state, li, lo, pos0, b);
            Gemm(l.O, _attn.Ptr, _o.Ptr, b);
            TensorPrimitives.Add(new ReadOnlySpan<float>(x, b * h), new ReadOnlySpan<float>(_o.Ptr, b * h), new Span<float>(x, b * h));

            CopyNorm(x, _norm.Ptr, b, l.FfnNorm);
            Gemm(l.Gate, _norm.Ptr, _gate.Ptr, b);
            Gemm(l.Up, _norm.Ptr, _up.Ptr, b);
            var g = new Span<float>(_gate.Ptr, b * _ffn);
            SiLu.Execute(g, g);
            TensorPrimitives.Multiply(g, new ReadOnlySpan<float>(_up.Ptr, b * _ffn), g);
            Gemm(l.Down, _gate.Ptr, _o.Ptr, b);
            TensorPrimitives.Add(new ReadOnlySpan<float>(x, b * h), new ReadOnlySpan<float>(_o.Ptr, b * h), new Span<float>(x, b * h));
        }

        // Rows 1..n: output_norm -> hidden_i, and the target LM head -> UNCAPPED base logits.
        CopyNorm(x + h, _hid.Ptr, nDraft, _outNorm);
        target.ProjectHead(_hid.Ptr, nDraft, _logits.Ptr);

        MarkovWalk(target, anchor, nDraft, frnn, tokens);
    }

    private void CopyNorm(float* src, float* dst, int rows, nint weight)
    {
        int h = HiddenSize;
        for (int r = 0; r < rows; r++)
            Inkling2Ops.RmsNorm(new ReadOnlySpan<float>(src + (long)r * h, h), new ReadOnlySpan<float>((void*)weight, h), Eps, new Span<float>(dst + (long)r * h, h));
    }

    private static void RoundThroughHalf(float* p, int n)
    {
        var s = new Span<float>(p, n);
        for (int i = 0; i < n; i++) s[i] = (float)(Half)s[i];
    }

    /// <summary>The semi-autoregressive Markov head (llama.cpp <c>build_dspark_markov_head</c>, <c>is_rnn</c>, <c>sample_from_anchor = false</c>), greedy.</summary>
    private void MarkovWalk(IInkling2DSparkTarget target, int anchor, int nDraft, ReadOnlySpan<float> frnn, Span<int> tokens)
    {
        int h = HiddenSize, r = _rank, vocab = _vocab;
        Span<float> state = stackalloc float[r];   // transient accumulator of this call (1 KB), never stored across calls
        state.Clear();
        float* join = _joinIn.Ptr, jp = _jp.Ptr;
        int prev = anchor;
        long w1RowBytes = Dequantize.RowByteSize(r, _w1.Qt);
        for (int i = 1; i <= nDraft; i++)
        {
            // joint = [state | markov_w1[prev] | hidden_i | f_rnn]
            if (Mutant == DSparkMutant.NoMarkovState) state.Clear();
            state.CopyTo(new Span<float>(join, r));
            Dequantize.ToFloat32(_w1.Ptr + (nint)(prev * w1RowBytes), r, _w1.Qt, new Span<float>(join + r, r));
            new ReadOnlySpan<float>(_hid.Ptr + (long)(i - 1) * h, h).CopyTo(new Span<float>(join + 2 * r, h));
            frnn.CopyTo(new Span<float>(join + 2 * r + h, h));
            Gemm(_joint, join, jp, 1);
            TensorPrimitives.Add(new ReadOnlySpan<float>(jp, 3 * r), new ReadOnlySpan<float>((void*)_jointBias, 3 * r), new Span<float>(jp, 3 * r));
            for (int j = 0; j < r; j++)
            {
                float gate = 1f / (1f + MathF.Exp(-jp[j]));
                float cand = MathF.Tanh(jp[r + j]);
                state[j] = cand + gate * (state[j] - cand);   // gate * state + (1 - gate) * cand
                jp[2 * r + j] = MathF.Tanh(jp[2 * r + j]);
            }
            // bias = markov_w2 . tanh(c2) + b ; col = softcap(base + bias)
            Gemm(_w2, jp + 2 * r, _bias.Ptr, 1);
            var col = new Span<float>(_logits.Ptr + (long)(i - 1) * vocab, vocab);
            var bias = new Span<float>(_bias.Ptr, vocab);
            if (_w2Bias != 0) TensorPrimitives.Add(bias, new ReadOnlySpan<float>((void*)_w2Bias, vocab), bias);
            if (Mutant == DSparkMutant.SoftcapBeforeBias && Softcap > 0f)
            {
                TensorPrimitives.Multiply(col, 1f / Softcap, col); TensorPrimitives.Tanh(col, col); TensorPrimitives.Multiply(col, Softcap, col);
                TensorPrimitives.Add(col, bias, col);
            }
            else
            {
                TensorPrimitives.Add(col, bias, col);
                if (Softcap > 0f) { TensorPrimitives.Multiply(col, 1f / Softcap, col); TensorPrimitives.Tanh(col, col); TensorPrimitives.Multiply(col, Softcap, col); }
            }
            prev = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)col);
            tokens[i - 1] = prev;
        }
    }

    // ── Attention ───────────────────────────────────────────────────────

    private struct AttendCtx
    {
        public float* Q, K, V, Out;
        public Half* RingK, RingV;
        public int Lo, Pos0, B, Heads, KvHeads, HeadDim, KvDim, Ring;
        public float Scale;
    }

    /// <summary>
    /// Non-causal attention of the <paramref name="b"/> block rows over the context keys <c>[lo, pos0)</c> (F16 ring) and the block's own K/V.
    /// Every row sees the same keys (llama.cpp <c>block_uniform</c>: the whole block shares the mask row of its first query, the anchor).
    /// </summary>
    private void Attend(Inkling2DSparkState state, int layer, int lo, int pos0, int b)
    {
        var ctx = new AttendCtx
        {
            Q = _q.Ptr, K = _k.Ptr, V = _v.Ptr, Out = _attn.Ptr,
            RingK = state.Keys(layer), RingV = state.Values(layer),
            Ring = state.RingFor(layer), Lo = lo, Pos0 = pos0, B = b, Heads = NumHeads, KvHeads = NumKvHeads, HeadDim = HeadDim, KvDim = NumKvHeads * HeadDim,
            Scale = 1f / MathF.Sqrt(HeadDim),
        };
        if (_pool is null) { for (int hh = 0; hh < ctx.Heads; hh++) AttendHead(&ctx, hh); return; }
        _pool.Dispatch((nint)(&ctx), &AttendWorker);
    }

    private static void AttendWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        var c = (AttendCtx*)ctxPtr;
        ComputeThreadPool.PartitionRange(c->Heads, threadIdx, threadCount, out int s, out int e);
        for (int hh = s; hh < e; hh++) AttendHead(c, hh);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SlotOf(AttendCtx* c, int pos) => c->Ring == 0 ? pos : pos % c->Ring;

    [SkipLocalsInit]
    private static void AttendHead(AttendCtx* c, int head)
    {
        int d = c->HeadDim, kvh = head / (c->Heads / c->KvHeads), b = c->B, qStride = c->Heads * d, kvStride = c->KvDim;
        int nCtx = c->Pos0 - c->Lo, nKeys = nCtx + b;
        float* scores = (float*)NativeMemory.Alloc((nuint)((long)b * nKeys * sizeof(float)));
        try
        {
            Span<float> kf = stackalloc float[d];
            for (int j = 0; j < nCtx; j++)
            {
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(c->RingK + (long)SlotOf(c, c->Lo + j) * kvStride + kvh * d, d), kf);
                for (int i = 0; i < b; i++)
                    scores[(long)i * nKeys + j] = TensorPrimitives.Dot(new ReadOnlySpan<float>(c->Q + (long)i * qStride + head * d, d), kf) * c->Scale;
            }
            for (int j = 0; j < b; j++)
            {
                var kr = new ReadOnlySpan<float>(c->K + (long)j * kvStride + kvh * d, d);
                for (int i = 0; i < b; i++)
                    scores[(long)i * nKeys + nCtx + j] = TensorPrimitives.Dot(new ReadOnlySpan<float>(c->Q + (long)i * qStride + head * d, d), kr) * c->Scale;
            }
            for (int i = 0; i < b; i++)
            {
                var row = new Span<float>(scores + (long)i * nKeys, nKeys);
                float m = TensorPrimitives.Max((ReadOnlySpan<float>)row);
                TensorPrimitives.Add((ReadOnlySpan<float>)row, -m, row);
                TensorPrimitives.Exp((ReadOnlySpan<float>)row, row);
                float denom = TensorPrimitives.Sum((ReadOnlySpan<float>)row);
                TensorPrimitives.Multiply((ReadOnlySpan<float>)row, 1f / denom, row);
                new Span<float>(c->Out + (long)i * qStride + head * d, d).Clear();
            }
            float* vfp = stackalloc float[d];
            var vf = new Span<float>(vfp, d);
            for (int j = 0; j < nKeys; j++)
            {
                ReadOnlySpan<float> vr;
                if (j < nCtx)
                {
                    TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(c->RingV + (long)SlotOf(c, c->Lo + j) * kvStride + kvh * d, d), vf);
                    vr = new ReadOnlySpan<float>(vfp, d);
                }
                else vr = new ReadOnlySpan<float>(c->V + (long)(j - nCtx) * kvStride + kvh * d, d);
                for (int i = 0; i < b; i++)
                {
                    float w = scores[(long)i * nKeys + j];
                    var o = new Span<float>(c->Out + (long)i * qStride + head * d, d);
                    TensorPrimitives.MultiplyAdd(vr, w, o, o);
                }
            }
        }
        finally { NativeMemory.Free(scores); }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (nint p in _native) NativeMemory.AlignedFree((void*)p);
        _native.Clear();
        foreach (var bf in new[] { _x, _norm, _q, _k, _v, _attn, _o, _gate, _up, _hid, _logits, _bias, _feat, _mix, _tmpK, _tmpV, _taps32, _joinIn, _jp, _w1Row }) bf.Dispose();
        _ownedFile?.Dispose();
    }
}

/// <summary>Deliberately wrong drafter variants (test-only mutants, issue #2039).</summary>
internal enum DSparkMutant
{
    None, NoAnchorNorm, DoubleAnchorNorm, NoMarkovState, SoftcapBeforeBias, AllLayersGlobal, AllLayersLocal, PerQueryWindow, FrnnWrongPos, DoubleSoftmaxCtxMix,
}
