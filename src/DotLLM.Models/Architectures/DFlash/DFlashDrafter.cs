using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures.DFlash;

/// <summary>
/// The pieces of a target model a DFlash drafter borrows (issue #1175): the drafter has no token embedding or output head of
/// its own, it embeds its block with the target's <c>token_embd</c> and projects its hidden rows with the target's lm-head
/// (by reference, never copied).
/// </summary>
internal unsafe interface IDFlashTargetOps
{
    /// <summary>Embeds <paramref name="tokens"/> into <paramref name="dst"/> (<c>[tokens, hidden]</c> F32), exactly as the target's own forward does.</summary>
    /// <param name="tokens">Token ids.</param>
    /// <param name="dst">Destination rows.</param>
    void EmbedForDraft(ReadOnlySpan<int> tokens, float* dst);

    /// <summary>Projects final-normed hidden rows to logits with the target's output head (and any activation rotation it folds in).</summary>
    /// <param name="hidden">Rows <c>[rows, hidden]</c>; may be overwritten.</param>
    /// <param name="rows">Row count.</param>
    /// <param name="logits">Destination <c>[rows, vocab]</c>.</param>
    void ProjectForDraft(float* hidden, int rows, float* logits);

    /// <summary>The pool the target computes on; the drafter reuses it.</summary>
    ComputeThreadPool? DraftPool { get; }
}

/// <summary>
/// A DFlash / DFlash2 block-diffusion drafter (z-lab, arXiv 2602.06036; GGUF arch <c>dflash</c> as converted by llama.cpp) on
/// the CPU (issue #1175). Weights stay in their stored quantisation (memory-mapped). The per-sequence context K/V lives in
/// <see cref="DFlashMtpState"/>; this class holds the weights and the per-call scratch.
/// </summary>
/// <remarks>
/// <para>Reference: <c>dflash/model.py</c> (pin 07ebd93d) and llama.cpp <c>src/models/dflash.cpp</c>. Context features are the
/// target's layer inputs at <c>dflash.target_layers</c>, concatenated, through <c>fc</c> and <c>enc.output_norm</c>; each layer
/// projects them with its own <c>attn_k</c>/<c>attn_v</c> (+ k-norm, RoPE) into the context K/V ("KV injection"). A draft
/// block <c>[anchor, mask x (B-1)]</c> then runs the 5 layers with non-causal attention over <c>[context ; block]</c> (sliding
/// window), and rows 1..B-1 give the drafted tokens. DFlash2 adds the grouped dynamic causal convolutions around attention and
/// the MLP and the low-rank candidate selector; plain DFlash takes the argmax.</para>
/// </remarks>
internal sealed unsafe class DFlashDrafter : IDisposable
{
    private const int InjectChunkRows = 64;

    private readonly GgufFile? _ownedFile;

    /// <summary>A matrix in its stored type: rows of <see cref="In"/> elements, <see cref="Out"/> rows.</summary>
    internal readonly record struct Matrix(nint Ptr, QuantizationType Qt, int In, int Out);

    internal sealed class Layer
    {
        public required float[] AttnNorm, FfnNorm, QNorm, KNorm;
        public required Matrix Q, K, V, O, Gate, Up, Down;
        public Matrix? AttnConvProj, FfnConvProj;
        public float[]? AttnConvBase, FfnConvBase; // [side][tap][hidden]
    }

    private readonly Layer[] _layers;
    private readonly Matrix _fc;
    private readonly float[] _encNorm, _outNorm;
    private readonly Matrix? _selHidden;
    private readonly Matrix? _selPrev, _selNext;   // [vocab rows of rank]
    private readonly float[] _invFreq;

    /// <summary>Hidden size (equals the target's).</summary>
    public int HiddenSize { get; }

    /// <summary>Trained block size B (anchor + B-1 drafts).</summary>
    public int BlockSize { get; }

    /// <summary>Target layer indices whose <em>input</em> is tapped (llama.cpp <c>dflash.target_layers</c>); equal to the layer count = the last layer's output.</summary>
    public int[] TargetLayers { get; }

    /// <summary>Sliding window of the drafter's attention (positions).</summary>
    public int Window { get; }

    /// <summary>Mask token id filling the drafted block positions.</summary>
    public int MaskTokenId { get; }

    /// <summary>Drafter layer count.</summary>
    public int NumLayers => _layers.Length;

    /// <summary>KV heads of the drafter attention.</summary>
    public int NumKvHeads { get; }

    /// <summary>Query heads of the drafter attention.</summary>
    public int NumHeads { get; }

    /// <summary>Head dimension.</summary>
    public int HeadDim { get; }

    /// <summary>True for a DFlash2 drafter (convolutions + selector).</summary>
    public bool IsDFlash2 => _selHidden is not null;

    private readonly int _ffn, _convKernel, _convGroup, _selRank, _selTopK, _vocab;
    private readonly float _eps;

    // Scratch, sized for max(BlockSize, InjectChunkRows) rows.
    private readonly int _scratchRows;
    private nint _scratch;
    private float* _x, _norm, _dyn, _q, _k, _v, _attn, _o, _gate, _up, _feat, _taps32, _logits, _selGate;

    /// <summary>Bytes held by the drafter's scratch (host private).</summary>
    public long ScratchBytes { get; }

    /// <summary>Opens a <c>dflash</c> GGUF and binds it to a target of <paramref name="targetHidden"/> width, <paramref name="targetLayers"/> layers and <paramref name="targetVocab"/> tokens.</summary>
    /// <param name="path">Drafter GGUF path.</param>
    /// <param name="targetHidden">Target hidden size.</param>
    /// <param name="targetLayers">Target layer count.</param>
    /// <param name="targetVocab">Target vocabulary size.</param>
    /// <returns>The drafter, owning the file.</returns>
    public static DFlashDrafter Open(string path, int targetHidden, int targetLayers, int targetVocab)
    {
        var file = GgufFile.Open(path);
        try { return new DFlashDrafter(file, ownsFile: true, targetHidden, targetLayers, targetVocab); }
        catch { file.Dispose(); throw; }
    }

    /// <summary>Binds an opened <c>dflash</c> GGUF (see <see cref="Open"/>).</summary>
    /// <param name="file">The drafter GGUF; kept alive for the drafter's lifetime.</param>
    /// <param name="ownsFile">Dispose <paramref name="file"/> with the drafter.</param>
    /// <param name="targetHidden">Target hidden size.</param>
    /// <param name="targetLayers">Target layer count.</param>
    /// <param name="targetVocab">Target vocabulary size.</param>
    public DFlashDrafter(GgufFile file, bool ownsFile, int targetHidden, int targetLayers, int targetVocab)
    {
        var md = file.Metadata;
        string arch = md.GetStringOrDefault("general.architecture");
        if (arch != "dflash")
            throw new NotSupportedException($"Not a DFlash drafter GGUF (general.architecture = '{arch}').");
        if (file.TensorsByName.ContainsKey("d2t") || file.TensorsByName.ContainsKey("markov_w1.weight")
            || md.ContainsKey("dflash.lilicorr_hidden_size") || md.ContainsKey("dflash.hyper_connection_count"))
            throw new NotSupportedException("Reduced-vocabulary (d2t), DSpark, LiliCorr and DSV4 DFlash variants are not supported yet.");
        if (file.TensorsByName.ContainsKey("token_embd.weight") || file.TensorsByName.ContainsKey("output.weight"))
            throw new NotSupportedException("DFlash drafters with their own embeddings/output head are not supported yet; only target-shared heads.");

        HiddenSize = (int)md.GetUInt32("dflash.embedding_length");
        if (HiddenSize != targetHidden)
            throw new InvalidOperationException($"DFlash drafter hidden size {HiddenSize} does not match the target's {targetHidden}.");
        int nLayer = (int)md.GetUInt32("dflash.block_count");
        NumHeads = (int)md.GetUInt32("dflash.attention.head_count");
        NumKvHeads = (int)md.GetUInt32("dflash.attention.head_count_kv");
        HeadDim = (int)md.GetUInt32OrDefault("dflash.attention.key_length", (uint)(HiddenSize / NumHeads));
        _ffn = (int)md.GetUInt32("dflash.feed_forward_length");
        _eps = md.GetFloat32("dflash.attention.layer_norm_rms_epsilon");
        BlockSize = (int)md.GetUInt32OrDefault("dflash.block_size", 16);
        if (BlockSize < 2) throw new InvalidOperationException($"DFlash block size {BlockSize} < 2.");
        Window = md.ContainsKey("dflash.attention.sliding_window") ? (int)md.GetUInt32("dflash.attention.sliding_window") : int.MaxValue / 4;
        if (md.ContainsKey("dflash.attention.causal") && md.GetBool("dflash.attention.causal"))
            throw new NotSupportedException("Causal DFlash drafters are not supported (every published DFlash/DFlash2 drafter is non-causal).");
        MaskTokenId = md.ContainsKey("tokenizer.ggml.mask_token_id") ? (int)md.GetUInt32("tokenizer.ggml.mask_token_id")
            : throw new InvalidOperationException("DFlash drafter GGUF has no tokenizer.ggml.mask_token_id.");
        if ((uint)MaskTokenId >= (uint)targetVocab)
            throw new InvalidOperationException($"DFlash mask token {MaskTokenId} is outside the target vocabulary ({targetVocab}).");
        if (md.GetFloat32OrDefault("dflash.logit_scale", 0f) != 0f || md.GetFloat32OrDefault("dflash.final_logit_softcapping", 0f) != 0f
            || md.GetFloat32OrDefault("dflash.embedding_scale", 0f) != 0f || md.GetFloat32OrDefault("dflash.attention.value_scale", 0f) != 0f)
            throw new NotSupportedException("DFlash logit/embedding/value scales and softcapping are not supported yet (Qwen-family drafters use none).");
        string hiddenAct = md.GetStringOrDefault("dflash.hidden_act", "silu");
        if (hiddenAct != "silu") throw new NotSupportedException($"DFlash hidden activation '{hiddenAct}' is not supported.");

        TargetLayers = ReadIntArray(md, "dflash.target_layers");
        foreach (int t in TargetLayers)
            if (t < 0 || t > targetLayers)
                throw new InvalidOperationException($"DFlash target layer {t} is outside the target's 0..{targetLayers} layer inputs.");
        _vocab = targetVocab;

        float theta = md.GetFloat32OrDefault("dflash.rope.freq_base", 10000f);
        int ropeDim = (int)md.GetUInt32OrDefault("dflash.rope.dimension_count", (uint)HeadDim);
        if (ropeDim != HeadDim) throw new NotSupportedException("Partial-rotary DFlash drafters are not supported yet.");
        _invFreq = new float[HeadDim / 2];
        for (int i = 0; i < _invFreq.Length; i++)
            _invFreq[i] = (float)(1.0 / Math.Pow(theta, 2.0 * i / HeadDim));

        Matrix M(string name, int expectIn, int expectOut)
        {
            if (!file.TensorsByName.TryGetValue(name, out var d))
                throw new InvalidOperationException($"DFlash drafter is missing tensor '{name}'.");
            int inDim = d.Shape[0], outDim = d.Shape.Rank > 1 ? d.Shape[1] : 1;
            if (inDim != expectIn || outDim != expectOut)
                throw new InvalidOperationException($"DFlash tensor '{name}' is [{inDim}, {outDim}], expected [{expectIn}, {expectOut}].");
            return new Matrix(file.TensorDataPointer(d), d.QuantizationType, inDim, outDim);
        }
        float[] F(string name, int count)
        {
            if (!file.TensorsByName.TryGetValue(name, out var d))
                throw new InvalidOperationException($"DFlash drafter is missing tensor '{name}'.");
            if (d.Shape.ElementCount != count)
                throw new InvalidOperationException($"DFlash tensor '{name}' has {d.Shape.ElementCount} elements, expected {count}.");
            var dst = new float[count];
            Dequantize.ToFloat32(file.TensorDataPointer(d), count, d.QuantizationType, dst);
            return dst;
        }

        int qDim = NumHeads * HeadDim, kvDim = NumKvHeads * HeadDim;
        _fc = M("fc.weight", TargetLayers.Length * HiddenSize, HiddenSize);
        _encNorm = F("enc.output_norm.weight", HiddenSize);
        _outNorm = F("output_norm.weight", HiddenSize);

        bool hasConv = md.ContainsKey("dflash.conv_kernel_size");
        _convKernel = hasConv ? (int)md.GetUInt32("dflash.conv_kernel_size") : 0;
        _convGroup = hasConv ? (int)md.GetUInt32("dflash.conv_group_size") : 0;
        if (hasConv && HiddenSize % _convGroup != 0)
            throw new InvalidOperationException("DFlash2 hidden size must be divisible by conv_group_size.");
        int convProj = hasConv ? 2 * _convKernel * (HiddenSize / _convGroup) : 0;

        _layers = new Layer[nLayer];
        for (int i = 0; i < nLayer; i++)
        {
            if (file.TensorsByName.ContainsKey($"blk.{i}.attn_sinks.weight") || file.TensorsByName.ContainsKey($"blk.{i}.attn_post_norm.weight"))
                throw new NotSupportedException("DFlash drafters with attention sinks or post-norms are not supported yet.");
            var l = new Layer
            {
                AttnNorm = F($"blk.{i}.attn_norm.weight", HiddenSize),
                FfnNorm = F($"blk.{i}.ffn_norm.weight", HiddenSize),
                QNorm = F($"blk.{i}.attn_q_norm.weight", HeadDim),
                KNorm = F($"blk.{i}.attn_k_norm.weight", HeadDim),
                Q = M($"blk.{i}.attn_q.weight", HiddenSize, qDim),
                K = M($"blk.{i}.attn_k.weight", HiddenSize, kvDim),
                V = M($"blk.{i}.attn_v.weight", HiddenSize, kvDim),
                O = M($"blk.{i}.attn_output.weight", qDim, HiddenSize),
                Gate = M($"blk.{i}.ffn_gate.weight", HiddenSize, _ffn),
                Up = M($"blk.{i}.ffn_up.weight", HiddenSize, _ffn),
                Down = M($"blk.{i}.ffn_down.weight", _ffn, HiddenSize),
            };
            if (hasConv)
            {
                l.AttnConvProj = M($"blk.{i}.attn_conv_proj.weight", HiddenSize, convProj);
                l.FfnConvProj = M($"blk.{i}.ffn_conv_proj.weight", HiddenSize, convProj);
                l.AttnConvBase = F($"blk.{i}.attn_conv_base", HiddenSize * _convKernel * 2);
                l.FfnConvBase = F($"blk.{i}.ffn_conv_base", HiddenSize * _convKernel * 2);
            }
            _layers[i] = l;
        }

        if (file.TensorsByName.ContainsKey("selector_hidden.weight"))
        {
            _selRank = (int)md.GetUInt32("dflash.selector_rank");
            _selTopK = (int)md.GetUInt32("dflash.selector_top_k");
            if (!hasConv) throw new InvalidOperationException("DFlash2 selector without conv metadata.");
            _selHidden = M("selector_hidden.weight", HiddenSize, _selRank);
            _selPrev = M("selector_predecessor.weight", _selRank, targetVocab);
            _selNext = M("selector_successor.weight", _selRank, targetVocab);
        }

        _ownedFile = ownsFile ? file : null;

        // Scratch: one aligned block carved into the per-call buffers.
        _scratchRows = Math.Max(BlockSize, InjectChunkRows);
        int r = _scratchRows, b = BlockSize;
        long[] sizes =
        [
            (long)r * HiddenSize,               // _x
            (long)r * HiddenSize,               // _norm
            (long)r * Math.Max(convProj, 1),    // _dyn (attention conv coefficients, kept to the finish)
            (long)b * qDim,                     // _q
            (long)r * kvDim,                    // _k
            (long)r * kvDim,                    // _v
            (long)b * qDim,                     // _attn
            (long)r * HiddenSize,               // _o (also the FFN conv coefficients' partner)
            (long)b * _ffn,                     // _gate
            (long)b * _ffn,                     // _up
            (long)r * HiddenSize,               // _feat (context features)
            (long)r * TargetLayers.Length * HiddenSize, // _taps32
            (long)(b - 1) * targetVocab,        // _logits
            (long)b * Math.Max(_selRank, 1),    // _selGate
            (long)r * Math.Max(convProj, 1),    // FFN conv coefficients
        ];
        long total = 0;
        foreach (long s in sizes) total += (s + 15) & ~15L;
        ScratchBytes = total * sizeof(float);
        _scratch = (nint)NativeMemory.AlignedAlloc((nuint)ScratchBytes, 64);
        float* p = (float*)_scratch;
        float* Next(long n) { float* cur = p; p += (n + 15) & ~15L; return cur; }
        _x = Next(sizes[0]); _norm = Next(sizes[1]); _dyn = Next(sizes[2]); _q = Next(sizes[3]); _k = Next(sizes[4]);
        _v = Next(sizes[5]); _attn = Next(sizes[6]); _o = Next(sizes[7]); _gate = Next(sizes[8]); _up = Next(sizes[9]);
        _feat = Next(sizes[10]); _taps32 = Next(sizes[11]); _logits = Next(sizes[12]); _selGate = Next(sizes[13]);
        _ffnDyn = Next(sizes[14]);
    }

    private readonly float* _ffnDyn;

    private static int[] ReadIntArray(GgufMetadata md, string key)
    {
        if (!md.TryGetValue(key, out var v))
            throw new InvalidOperationException($"DFlash drafter GGUF has no '{key}'.");
        return v.Value switch
        {
            int[] a => a,
            uint[] a => Array.ConvertAll(a, x => checked((int)x)),
            long[] a => Array.ConvertAll(a, x => checked((int)x)),
            ulong[] a => Array.ConvertAll(a, x => checked((int)x)),
            _ => throw new InvalidOperationException($"DFlash '{key}' is not an integer array."),
        };
    }

    /// <summary>Width of one captured tap row (<c>targets x hidden</c>).</summary>
    public int TapRowWidth => TargetLayers.Length * HiddenSize;

    /// <summary>Logits of the last <see cref="Draft"/>'s drafted rows, <c>[draft, vocab]</c>.</summary>
    public ReadOnlySpan<float> LastDraftLogits(int row) => new(_logits + (long)row * _vocab, _vocab);

    // ── Context injection ───────────────────────────────────────────────

    /// <summary>
    /// Projects captured target taps (F16 rows, <see cref="TapRowWidth"/> wide) into the drafter's context K/V of
    /// <paramref name="state"/> at <paramref name="positions"/> (llama.cpp's KV-injection pass).
    /// </summary>
    /// <param name="state">The sequence state.</param>
    /// <param name="taps">Captured rows.</param>
    /// <param name="positions">Sequence position of each row.</param>
    /// <param name="pool">Compute pool.</param>
    public void Inject(DFlashMtpState state, Half* taps, ReadOnlySpan<int> positions, ComputeThreadPool? pool)
    {
        int width = TapRowWidth, kvDim = NumKvHeads * HeadDim;
        Span<float> cos = stackalloc float[HeadDim / 2];
        Span<float> sin = stackalloc float[HeadDim / 2];
        for (int start = 0; start < positions.Length; start += InjectChunkRows)
        {
            int n = Math.Min(InjectChunkRows, positions.Length - start);
            TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(taps + (long)start * width, n * width),
                new Span<float>(_taps32, n * width));
            Gemm(_fc, _taps32, _feat, n, pool);
            for (int t = 0; t < n; t++)
            {
                var row = new Span<float>(_feat + (long)t * HiddenSize, HiddenSize);
                RmsNorm.Execute(row, _encNorm, _eps, row);
            }
            for (int li = 0; li < _layers.Length; li++)
            {
                var l = _layers[li];
                Gemm(l.K, _feat, _k, n, pool);
                Gemm(l.V, _feat, _v, n, pool);
                for (int t = 0; t < n; t++)
                {
                    int pos = positions[start + t];
                    RopeTable(pos, cos, sin);
                    NormRopeHeads(_k + (long)t * kvDim, NumKvHeads, l.KNorm, cos, sin);
                    state.WriteContextRow(li, pos, new ReadOnlySpan<float>(_k + (long)t * kvDim, kvDim),
                        new ReadOnlySpan<float>(_v + (long)t * kvDim, kvDim));
                }
            }
            for (int t = 0; t < n; t++)
                state.NoteInjected(positions[start + t]);
        }
    }

    // ── Draft ───────────────────────────────────────────────────────────

    /// <summary>
    /// Drafts <paramref name="tokens"/>.Length tokens after <paramref name="anchor"/> at <paramref name="start"/>: one parallel
    /// pass over the block <c>[anchor, mask x n]</c> with non-causal attention over the context K/V of positions
    /// <c>&lt; start</c>.
    /// </summary>
    /// <param name="state">The sequence state (context K/V; read only).</param>
    /// <param name="target">The target ops (embedding, lm-head).</param>
    /// <param name="anchor">The last committed token, at <paramref name="start"/>.</param>
    /// <param name="start">Position of <paramref name="anchor"/>.</param>
    /// <param name="tokens">Receives the drafts, <c>1 &lt;= Length &lt; BlockSize</c>.</param>
    public void Draft(DFlashMtpState state, IDFlashTargetOps target, int anchor, int start, Span<int> tokens)
    {
        int nDraft = tokens.Length, b = nDraft + 1;
        if (nDraft < 1 || b > BlockSize) throw new ArgumentOutOfRangeException(nameof(tokens));
        var pool = target.DraftPool;
        int h = HiddenSize, qDim = NumHeads * HeadDim, kvDim = NumKvHeads * HeadDim;

        Span<int> block = stackalloc int[b];
        block.Fill(MaskTokenId);
        block[0] = anchor;
        target.EmbedForDraft(block, _x);

        Span<float> cos = stackalloc float[HeadDim / 2];
        Span<float> sin = stackalloc float[HeadDim / 2];
        for (int li = 0; li < _layers.Length; li++)
        {
            var l = _layers[li];
            NormRows(_x, _norm, b, l.AttnNorm);
            if (l.AttnConvProj is { } acp)
            {
                Gemm(acp, _norm, _dyn, b, pool);
                ConvInPlace(_norm, _dyn, l.AttnConvBase!, 0, b);
            }
            Gemm(l.Q, _norm, _q, b, pool);
            Gemm(l.K, _norm, _k, b, pool);
            Gemm(l.V, _norm, _v, b, pool);
            for (int t = 0; t < b; t++)
            {
                RopeTable(start + t, cos, sin);
                NormRopeHeads(_q + (long)t * qDim, NumHeads, l.QNorm, cos, sin);
                NormRopeHeads(_k + (long)t * kvDim, NumKvHeads, l.KNorm, cos, sin);
            }
            Attend(state, li, start, b, pool);
            Gemm(l.O, _attn, _o, b, pool);
            if (l.AttnConvProj is not null)
                ConvInPlace(_o, _dyn, l.AttnConvBase!, 1, b);
            TensorPrimitives.Add(new ReadOnlySpan<float>(_x, b * h), new ReadOnlySpan<float>(_o, b * h), new Span<float>(_x, b * h));

            NormRows(_x, _norm, b, l.FfnNorm);
            if (l.FfnConvProj is { } fcp)
            {
                Gemm(fcp, _norm, _ffnDyn, b, pool);
                ConvInPlace(_norm, _ffnDyn, l.FfnConvBase!, 0, b);
            }
            Gemm(l.Gate, _norm, _gate, b, pool);
            Gemm(l.Up, _norm, _up, b, pool);
            var g = new Span<float>(_gate, b * _ffn);
            SiLu.Execute(g, g);
            TensorPrimitives.Multiply(g, new ReadOnlySpan<float>(_up, b * _ffn), g);
            Gemm(l.Down, _gate, _o, b, pool);
            if (l.FfnConvProj is not null)
                ConvInPlace(_o, _ffnDyn, l.FfnConvBase!, 1, b);
            TensorPrimitives.Add(new ReadOnlySpan<float>(_x, b * h), new ReadOnlySpan<float>(_o, b * h), new Span<float>(_x, b * h));
        }

        // Rows 1..b-1 predict the masked positions; row 0 (the anchor) is not a draft.
        float* hid = _x + h;
        NormRows(hid, hid, nDraft, _outNorm);
        if (_selHidden is { } sh)
            Gemm(sh, hid, _selGate, nDraft, pool);   // before the head: ProjectForDraft may overwrite hid
        target.ProjectForDraft(hid, nDraft, _logits);

        if (_selHidden is null)
        {
            for (int i = 0; i < nDraft; i++)
                tokens[i] = TensorPrimitives.IndexOfMax(LastDraftLogits(i));
            return;
        }
        SelectorWalk(anchor, tokens);
    }

    /// <summary>
    /// DFlash2's candidate selector (greedy): per drafted row, the top-k logits are re-scored with
    /// <c>unary + &lt;prev[pred] * gate_i, next[cand]&gt;</c> and the argmax becomes the next row's predecessor.
    /// </summary>
    private void SelectorWalk(int anchor, Span<int> tokens)
    {
        int k = _selTopK, rank = _selRank;
        Span<int> cand = stackalloc int[k];
        Span<float> unary = stackalloc float[k];
        Span<float> prev = stackalloc float[rank];
        Span<float> next = stackalloc float[rank];
        long rowBytes = Dequantize.RowByteSize(rank, _selPrev!.Value.Qt);
        int predecessor = anchor;
        for (int i = 0; i < tokens.Length; i++)
        {
            TopK(LastDraftLogits(i), cand, unary);
            Dequantize.ToFloat32(_selPrev.Value.Ptr + (nint)(predecessor * rowBytes), rank, _selPrev.Value.Qt, prev);
            TensorPrimitives.Multiply(prev, new ReadOnlySpan<float>(_selGate + (long)i * rank, rank), prev);
            int best = 0;
            float bestScore = float.NegativeInfinity;
            for (int j = 0; j < k; j++)
            {
                Dequantize.ToFloat32(_selNext!.Value.Ptr + (nint)(cand[j] * rowBytes), rank, _selNext.Value.Qt, next);
                float s = unary[j] + TensorPrimitives.Dot((ReadOnlySpan<float>)prev, next);
                if (s > bestScore) { bestScore = s; best = j; }
            }
            predecessor = cand[best];
            tokens[i] = predecessor;
        }
    }

    /// <summary>Top-k (unordered) of <paramref name="row"/> by a single pass with a min-replacement buffer.</summary>
    internal static void TopK(ReadOnlySpan<float> row, Span<int> idx, Span<float> val)
    {
        int k = idx.Length;
        for (int j = 0; j < k; j++) { idx[j] = j; val[j] = row[j]; }
        int minAt = 0;
        for (int j = 1; j < k; j++) if (val[j] < val[minAt]) minAt = j;
        for (int i = k; i < row.Length; i++)
        {
            float x = row[i];
            if (x <= val[minAt]) continue;
            idx[minAt] = i; val[minAt] = x;
            minAt = 0;
            for (int j = 1; j < k; j++) if (val[j] < val[minAt]) minAt = j;
        }
    }

    // ── Kernels ─────────────────────────────────────────────────────────

    private void Gemm(in Matrix w, float* input, float* output, int rows, ComputeThreadPool? pool)
        => NemotronHTransformerModel.GemmDispatch(w.Ptr, w.Qt, input, output, w.Out, w.In, rows, pool, null);

    private void NormRows(float* src, float* dst, int rows, float[] weight)
    {
        for (int t = 0; t < rows; t++)
            RmsNorm.Execute(new ReadOnlySpan<float>(src + (long)t * HiddenSize, HiddenSize), weight, _eps,
                new Span<float>(dst + (long)t * HiddenSize, HiddenSize));
    }

    private void RopeTable(int pos, Span<float> cos, Span<float> sin)
    {
        for (int i = 0; i < cos.Length; i++)
        {
            double a = (double)pos * _invFreq[i];
            cos[i] = (float)Math.Cos(a);
            sin[i] = (float)Math.Sin(a);
        }
    }

    private void NormRopeHeads(float* row, int heads, float[] norm, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        for (int hh = 0; hh < heads; hh++)
        {
            var v = new Span<float>(row + (long)hh * HeadDim, HeadDim);
            RmsNorm.Execute(v, norm, _eps, v);
            RoPE.ApplyRotationNeoX(v, cos, sin, HeadDim);
        }
    }

    /// <summary>
    /// DFlash2 grouped dynamic causal convolution along the block, in place:
    /// <c>x'[p][c] = sum_tap (base[side][tap][c] + dyn[p][side][tap][c / group]) * x[p - tap][c]</c>, zero before the block.
    /// </summary>
    private void ConvInPlace(float* x, float* dyn, float[] baseKernel, int side, int rows)
    {
        int h = HiddenSize, k = _convKernel, groups = h / _convGroup, taps = Math.Min(k, rows);
        int dynWidth = 2 * k * groups;
        // Later rows read earlier rows' inputs, so walk backwards.
        for (int p = rows - 1; p >= 0; p--)
        {
            float* outRow = x + (long)p * h;
            float* d = dyn + (long)p * dynWidth + side * k * groups;
            for (int c0 = 0; c0 < h; c0 += _convGroup)
            {
                int g = c0 / _convGroup;
                for (int c = c0; c < c0 + _convGroup; c++)
                {
                    float s = 0f;
                    for (int tap = 0; tap < taps && tap <= p; tap++)
                        s += (baseKernel[(side * k + tap) * h + c] + d[tap * groups + g]) * x[(long)(p - tap) * h + c];
                    outRow[c] = s;
                }
            }
        }
    }

    /// <summary>
    /// Non-causal attention of the block's queries over the context K/V (positions within the window, read from
    /// <paramref name="state"/>) and the block's own K/V. Writes <c>_attn</c>.
    /// </summary>
    private void Attend(DFlashMtpState state, int layer, int start, int b, ComputeThreadPool? pool)
    {
        int lo = state.FirstVisible(start, Window);   // keys for the earliest query; later queries skip a few more
        var ctx = new AttendCtx
        {
            Q = _q, K = _k, V = _v, Out = _attn,
            RingK = state.RingKeys(layer), RingV = state.RingValues(layer), RingCap = state.RingCapacity,
            Lo = lo, Start = start, B = b, Window = Window,
            Heads = NumHeads, KvHeads = NumKvHeads, HeadDim = HeadDim,
            Scale = 1f / MathF.Sqrt(HeadDim),
        };
        if (pool is null)
        {
            for (int hh = 0; hh < NumHeads; hh++) AttendHead(&ctx, hh);
            return;
        }
        pool.Dispatch((nint)(&ctx), &AttendWorker);
    }

    private struct AttendCtx
    {
        public float* Q, K, V, Out;
        public Half* RingK, RingV;
        public int RingCap, Lo, Start, B, Window, Heads, KvHeads, HeadDim;
        public float Scale;
    }

    private static void AttendWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        var ctx = (AttendCtx*)ctxPtr;
        ComputeThreadPool.PartitionRange(ctx->Heads, threadIdx, threadCount, out int s, out int e);
        for (int hh = s; hh < e; hh++) AttendHead(ctx, hh);
    }

    [SkipLocalsInit]
    private static void AttendHead(AttendCtx* c, int head)
    {
        int d = c->HeadDim, kvh = head / (c->Heads / c->KvHeads), b = c->B;
        int qStride = c->Heads * d, kvStride = c->KvHeads * d;
        int nCtx = c->Start - c->Lo, nKeys = nCtx + b;
        float* scores = (float*)NativeMemory.Alloc((nuint)((long)b * nKeys * sizeof(float)));
        try
        {
            Span<float> kf = stackalloc float[d];
            // Context keys (F16 ring): convert each once, score every query row against it.
            for (int j = 0; j < nCtx; j++)
            {
                int pos = c->Lo + j;
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(c->RingK + (long)(pos % c->RingCap) * kvStride + kvh * d, d), kf);
                for (int i = 0; i < b; i++)
                {
                    // Per-query window: positions at or beyond Window behind the query are masked.
                    scores[(long)i * nKeys + j] = c->Start + i - pos >= c->Window
                        ? float.NegativeInfinity
                        : TensorPrimitives.Dot(new ReadOnlySpan<float>(c->Q + (long)i * qStride + head * d, d), kf) * c->Scale;
                }
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
                TensorPrimitives.Multiply((ReadOnlySpan<float>)row, 1f / TensorPrimitives.Sum((ReadOnlySpan<float>)row), row);
                new Span<float>(c->Out + (long)i * qStride + head * d, d).Clear();
            }
            float* vfp = stackalloc float[d];
            var vf = new Span<float>(vfp, d);
            for (int j = 0; j < nKeys; j++)
            {
                ReadOnlySpan<float> vr;
                if (j < nCtx)
                {
                    int pos = c->Lo + j;
                    TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(c->RingV + (long)(pos % c->RingCap) * kvStride + kvh * d, d), vf);
                    vr = new ReadOnlySpan<float>(vfp, d);
                }
                else
                {
                    vr = new ReadOnlySpan<float>(c->V + (long)(j - nCtx) * kvStride + kvh * d, d);
                }
                for (int i = 0; i < b; i++)
                {
                    float w = scores[(long)i * nKeys + j];
                    if (w == 0f) continue;
                    var o = new Span<float>(c->Out + (long)i * qStride + head * d, d);
                    TensorPrimitives.MultiplyAdd(vr, w, o, o);
                }
            }
        }
        finally
        {
            NativeMemory.Free(scores);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_scratch != 0) { NativeMemory.AlignedFree((void*)_scratch); _scratch = 0; }
        _ownedFile?.Dispose();
    }
}
