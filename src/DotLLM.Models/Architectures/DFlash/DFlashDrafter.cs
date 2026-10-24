using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures.DFlash;

/// <summary>
/// A DFlash / DFlash2 block-diffusion drafter (z-lab, arXiv 2602.06036; GGUF arch <c>dflash</c> as converted by llama.cpp),
/// issue #1175. Weights stay in their stored quantisation (memory-mapped; a GPU backend uploads them as stored). The
/// per-sequence context K/V lives in <see cref="DFlashMtpState"/>; this class holds the weights, the per-call scratch and the
/// forward flow, and runs it on an <see cref="IDFlashBackend"/> (CPU, or the target's GPU).
/// </summary>
/// <remarks>
/// <para>Reference: <c>dflash/model.py</c> (pin 07ebd93d) and llama.cpp <c>src/models/dflash.cpp</c>. Context features are the
/// target's layer inputs at <c>dflash.target_layers</c>, concatenated, through <c>fc</c> and <c>enc.output_norm</c>; each layer
/// projects them with its own <c>attn_k</c>/<c>attn_v</c> (+ k-norm, RoPE) into the context K/V ("KV injection"). A draft
/// block <c>[anchor, mask x (B-1)]</c> then runs the 5 layers with non-causal attention over <c>[context ; block]</c> (sliding
/// window), and rows 1..B-1 give the drafted tokens. DFlash2 adds the grouped dynamic causal convolutions around attention and
/// the MLP (<c>y[p][c] = sum_tap (base[side][tap][c] + dyn[p][side][tap][c / group]) * x[p - tap][c]</c>) and the low-rank
/// candidate selector; plain DFlash takes the argmax.</para>
/// </remarks>
internal sealed unsafe class DFlashDrafter : IDisposable
{
    private const int InjectChunkRows = 64;

    private readonly GgufFile? _ownedFile;
    private readonly IDFlashBackend _backend;

    /// <summary>
    /// A matrix in its stored type: rows of <see cref="In"/> elements, <see cref="Out"/> rows, and the per-tensor output scale of
    /// its <c>X.scale</c> sibling (NVFP4, #1237; 1 when absent) that every product is multiplied by, as llama.cpp's <c>build_lora_mm</c>.
    /// </summary>
    internal readonly record struct Matrix(nint Ptr, QuantizationType Qt, int In, int Out, float Scale = 1f);

    internal sealed class Layer
    {
        public required float[] AttnNorm, FfnNorm, QNorm, KNorm;
        public required Matrix Q, K, V, O, Gate, Up, Down;
        public Matrix? AttnConvProj, FfnConvProj;
        public float[]? Sinks;   // per query head (#1237: Nemotron drafters)
        public float[]? AttnConvBase, FfnConvBase; // [side][tap][hidden]
    }

    private readonly Layer[] _layers;
    private readonly Matrix _fc;
    private readonly float[] _encNorm, _outNorm;
    private readonly Matrix? _selHidden;
    private readonly Matrix? _selPrev, _selNext;   // [vocab rows of rank], read on the host

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

    /// <summary>RMSNorm epsilon.</summary>
    public float Eps { get; }

    /// <summary>RoPE base (NeoX, full head).</summary>
    public float RopeTheta { get; }

    /// <summary>DFlash2 convolution taps (0 without convolutions).</summary>
    public int ConvKernel { get; }

    /// <summary>DFlash2 convolution channel group.</summary>
    public int ConvGroup { get; }

    /// <summary>The backend the drafter computes on.</summary>
    public IDFlashBackend Backend => _backend;

    /// <summary>Drafts per round: the block's <c>BlockSize - 1</c>, capped by <c>DOTLLM_DFLASH_DRAFT</c> when set.</summary>
    public int DraftTokensPerRound
        => int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_DFLASH_DRAFT"), out int n) && n >= 1
            ? Math.Min(n, BlockSize - 1) : BlockSize - 1;

    /// <summary>True for a DFlash2 drafter (convolutions + selector).</summary>
    public bool IsDFlash2 => _selHidden is not null;

    /// <summary>True for a LiLiCorr drafter (#1237): DFlash2 convolutions and the lattice head instead of the selector.</summary>
    public bool IsLiLiCorr => _lilicorr is not null;

    /// <summary>Rows of the per-sequence anchor-feature ring a LiLiCorr drafter keeps (<c>BlockSize + 1</c>, as llama.cpp).</summary>
    public int AnchorRows => BlockSize + 1;

    /// <summary>The weights the backend runs (the selector codebooks stay on the host).</summary>
    public IEnumerable<Matrix> DeviceMatrices
    {
        get
        {
            yield return _fc;
            if (_selHidden is { } sh) yield return sh;
            foreach (var l in _layers)
            {
                yield return l.Q; yield return l.K; yield return l.V; yield return l.O;
                yield return l.Gate; yield return l.Up; yield return l.Down;
                if (l.AttnConvProj is { } a) yield return a;
                if (l.FfnConvProj is { } f) yield return f;
            }
        }
    }

    /// <summary>The F32 vectors the backend uses: row norms (hidden wide), head norms (head wide) and conv bases.</summary>
    public IEnumerable<float[]> DeviceVectors
    {
        get
        {
            yield return _encNorm;
            yield return _outNorm;
            foreach (var l in _layers)
            {
                yield return l.AttnNorm; yield return l.FfnNorm; yield return l.QNorm; yield return l.KNorm;
                if (l.Sinks is { } s) yield return s;
                if (l.AttnConvBase is { } a) yield return a;
                if (l.FfnConvBase is { } f) yield return f;
            }
        }
    }

    private readonly int _ffn, _selRank, _selTopK, _vocab;
    private readonly LiLiCorrHead? _lilicorr;
    private float* _hiddenHost;   // LiLiCorr: the drafted rows' normed hidden states, read back for the head
    private readonly float[] _anchorHost = [];

    // Backend scratch, sized for max(BlockSize, InjectChunkRows) rows, and host buffers for the selector.
    private readonly nint[] _buffers;
    private readonly nint _x, _norm, _conv, _dyn, _ffnDyn, _q, _k, _v, _attn, _o, _gate, _up, _feat, _taps32, _gateSel;
    private float* _logits, _selGate;

    /// <summary>Opens a <c>dflash</c> GGUF and binds it to a target of <paramref name="targetHidden"/> width, <paramref name="targetLayers"/> layers and <paramref name="targetVocab"/> tokens.</summary>
    /// <param name="path">Drafter GGUF path.</param>
    /// <param name="targetHidden">Target hidden size.</param>
    /// <param name="targetLayers">Target layer count.</param>
    /// <param name="targetVocab">Target vocabulary size.</param>
    /// <param name="backend">The kernels to run on; disposed with the drafter.</param>
    /// <param name="hostPool">Host thread pool for a LiLiCorr head (null = the head's own pool); not owned.</param>
    /// <returns>The drafter, owning the file.</returns>
    public static DFlashDrafter Open(string path, int targetHidden, int targetLayers, int targetVocab, IDFlashBackend backend,
        ComputeThreadPool? hostPool = null)
    {
        var file = GgufFile.Open(path);
        try { return new DFlashDrafter(file, ownsFile: true, targetHidden, targetLayers, targetVocab, backend, hostPool); }
        catch { file.Dispose(); backend.Dispose(); throw; }
    }

    /// <summary>Binds an opened <c>dflash</c> GGUF (see <see cref="Open"/>).</summary>
    /// <param name="file">The drafter GGUF; kept alive for the drafter's lifetime.</param>
    /// <param name="ownsFile">Dispose <paramref name="file"/> with the drafter.</param>
    /// <param name="targetHidden">Target hidden size.</param>
    /// <param name="targetLayers">Target layer count.</param>
    /// <param name="targetVocab">Target vocabulary size.</param>
    /// <param name="backend">The kernels to run on; disposed with the drafter.</param>
    /// <param name="hostPool">Host thread pool for a LiLiCorr head (null = the head's own pool); not owned.</param>
    public DFlashDrafter(GgufFile file, bool ownsFile, int targetHidden, int targetLayers, int targetVocab, IDFlashBackend backend,
        ComputeThreadPool? hostPool = null)
    {
        _backend = backend;
        var md = file.Metadata;
        string arch = md.GetStringOrDefault("general.architecture");
        if (arch != "dflash")
            throw new NotSupportedException($"Not a DFlash drafter GGUF (general.architecture = '{arch}').");
        if (file.TensorsByName.ContainsKey("d2t") || file.TensorsByName.ContainsKey("markov_w1.weight")
            || md.ContainsKey("dflash.hyper_connection_count"))
            throw new NotSupportedException("Reduced-vocabulary (d2t), DSpark and DSV4 DFlash variants are not supported yet.");
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
        Eps = md.GetFloat32("dflash.attention.layer_norm_rms_epsilon");
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

        RopeTheta = md.GetFloat32OrDefault("dflash.rope.freq_base", 10000f);
        int ropeDim = (int)md.GetUInt32OrDefault("dflash.rope.dimension_count", (uint)HeadDim);
        if (ropeDim != HeadDim) throw new NotSupportedException("Partial-rotary DFlash drafters are not supported yet.");

        Matrix M(string name, int expectIn, int expectOut) => ReadMatrix(file, name, expectIn, expectOut);
        float[] F(string name, int count) => ReadVector(file, name, count);

        int qDim = NumHeads * HeadDim, kvDim = NumKvHeads * HeadDim;
        _fc = M("fc.weight", TargetLayers.Length * HiddenSize, HiddenSize);
        _encNorm = F("enc.output_norm.weight", HiddenSize);
        _outNorm = F("output_norm.weight", HiddenSize);

        bool hasConv = md.ContainsKey("dflash.conv_kernel_size");
        ConvKernel = hasConv ? (int)md.GetUInt32("dflash.conv_kernel_size") : 0;
        ConvGroup = hasConv ? (int)md.GetUInt32("dflash.conv_group_size") : 0;
        if (hasConv && HiddenSize % ConvGroup != 0)
            throw new InvalidOperationException("DFlash2 hidden size must be divisible by conv_group_size.");
        int convProj = hasConv ? 2 * ConvKernel * (HiddenSize / ConvGroup) : 0;

        _layers = new Layer[nLayer];
        for (int i = 0; i < nLayer; i++)
        {
            if (file.TensorsByName.ContainsKey($"blk.{i}.attn_post_norm.weight") || file.TensorsByName.ContainsKey($"blk.{i}.ffn_post_norm.weight")
                || file.TensorsByName.ContainsKey($"blk.{i}.layer_output_scale.weight") || file.TensorsByName.ContainsKey($"blk.{i}.rope_freqs.weight"))
                throw new NotSupportedException("DFlash drafters with post-norms, output scales or rope factors are not supported yet.");
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
                Sinks = file.TensorsByName.ContainsKey($"blk.{i}.attn_sinks.weight") ? F($"blk.{i}.attn_sinks.weight", NumHeads) : null,
            };
            if (hasConv)
            {
                l.AttnConvProj = M($"blk.{i}.attn_conv_proj.weight", HiddenSize, convProj);
                l.FfnConvProj = M($"blk.{i}.ffn_conv_proj.weight", HiddenSize, convProj);
                l.AttnConvBase = F($"blk.{i}.attn_conv_base", HiddenSize * ConvKernel * 2);
                l.FfnConvBase = F($"blk.{i}.ffn_conv_base", HiddenSize * ConvKernel * 2);
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

        if (LiLiCorrHead.IsPresent(file))
        {
            if (_selHidden is not null) throw new InvalidOperationException("A DFlash drafter cannot carry both a DFlash2 selector and a LiLiCorr head.");
            _lilicorr = new LiLiCorrHead(file, HiddenSize, BlockSize, nLayer, Eps, targetVocab, hostPool);
            _hiddenHost = (float*)NativeMemory.AlignedAlloc((nuint)((long)(BlockSize - 1) * HiddenSize * sizeof(float)), 64);
            _anchorHost = new float[HiddenSize];
        }

        _ownedFile = ownsFile ? file : null;

        int r = Math.Max(BlockSize, InjectChunkRows), bs = BlockSize;
        int convWidth = Math.Max(convProj, 1);
        _buffers =
        [
            _x = backend.Alloc((long)r * HiddenSize),
            _norm = backend.Alloc((long)r * HiddenSize),
            _conv = backend.Alloc((long)bs * HiddenSize),
            _dyn = backend.Alloc((long)bs * convWidth),
            _ffnDyn = backend.Alloc((long)bs * convWidth),
            _q = backend.Alloc((long)bs * qDim),
            _k = backend.Alloc((long)r * kvDim),
            _v = backend.Alloc((long)r * kvDim),
            _attn = backend.Alloc((long)bs * qDim),
            _o = backend.Alloc((long)bs * HiddenSize),
            _gate = backend.Alloc((long)bs * _ffn),
            _up = backend.Alloc((long)bs * _ffn),
            _feat = backend.Alloc((long)r * HiddenSize),
            _taps32 = backend.Alloc((long)r * TargetLayers.Length * HiddenSize),
            _gateSel = backend.Alloc((long)bs * Math.Max(_selRank, 1)),
        ];
        _logits = (float*)NativeMemory.AlignedAlloc((nuint)((long)(bs - 1) * targetVocab * sizeof(float)), 64);
        _selGate = (float*)NativeMemory.AlignedAlloc((nuint)((long)bs * Math.Max(_selRank, 1) * sizeof(float)), 64);
        backend.Bind(this);
    }

    /// <summary>A matrix of <paramref name="file"/> in its stored type, shape-checked, with its optional <c>X.scale</c>.</summary>
    /// <param name="file">The GGUF.</param>
    /// <param name="name">Tensor name (<c>*.weight</c>).</param>
    /// <param name="expectIn">Expected row length (GGUF ne0).</param>
    /// <param name="expectOut">Expected row count (GGUF ne1).</param>
    /// <returns>The matrix view (no copy).</returns>
    internal static Matrix ReadMatrix(GgufFile file, string name, int expectIn, int expectOut)
    {
        if (!file.TensorsByName.TryGetValue(name, out var d))
            throw new InvalidOperationException($"DFlash drafter is missing tensor '{name}'.");
        int inDim = d.Shape[0], outDim = d.Shape.Rank > 1 ? d.Shape[1] : 1;
        if (inDim != expectIn || outDim != expectOut)
            throw new InvalidOperationException($"DFlash tensor '{name}' is [{inDim}, {outDim}], expected [{expectIn}, {expectOut}].");
        string stem = name.EndsWith(".weight", StringComparison.Ordinal) ? name[..^".weight".Length] : name;
        if (file.TensorsByName.ContainsKey(stem + ".input_scale"))
            throw new NotSupportedException($"DFlash tensor '{name}' has an input scale; only output scales (X.scale) are supported.");
        float scale = file.TensorsByName.ContainsKey(stem + ".scale") ? ReadVector(file, stem + ".scale", 1)[0] : 1f;
        return new Matrix(file.TensorDataPointer(d), d.QuantizationType, inDim, outDim, scale);
    }

    /// <summary>A small tensor of <paramref name="file"/> as F32 (norms, biases, conv bases).</summary>
    /// <param name="file">The GGUF.</param>
    /// <param name="name">Tensor name.</param>
    /// <param name="count">Expected element count.</param>
    /// <returns>A host copy.</returns>
    internal static float[] ReadVector(GgufFile file, string name, int count)
    {
        if (!file.TensorsByName.TryGetValue(name, out var d))
            throw new InvalidOperationException($"DFlash drafter is missing tensor '{name}'.");
        if (d.Shape.ElementCount != count)
            throw new InvalidOperationException($"DFlash tensor '{name}' has {d.Shape.ElementCount} elements, expected {count}.");
        var dst = new float[count];
        Dequantize.ToFloat32(file.TensorDataPointer(d), count, d.QuantizationType, dst);
        return dst;
    }

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

    /// <summary>Tap slot per target layer input <c>0..numLayers</c> (-1 = not tapped).</summary>
    /// <param name="numLayers">Target layer count.</param>
    /// <returns>The slot map.</returns>
    public int[] SlotForLayerInput(int numLayers)
    {
        var slots = new int[numLayers + 1];
        Array.Fill(slots, -1);
        for (int k = 0; k < TargetLayers.Length; k++)
        {
            if (slots[TargetLayers[k]] >= 0)
                throw new InvalidOperationException($"DFlash target layer {TargetLayers[k]} is listed twice.");
            slots[TargetLayers[k]] = k;
        }
        return slots;
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
    public void Inject(DFlashMtpState state, Half* taps, ReadOnlySpan<int> positions)
    {
        int width = TapRowWidth;
        for (int start = 0; start < positions.Length; start += InjectChunkRows)
        {
            int n = Math.Min(InjectChunkRows, positions.Length - start);
            var pos = positions.Slice(start, n);
            _backend.LoadTaps(taps + (long)start * width, n * width, _taps32);
            Gemm(_fc, _taps32, _feat, n);
            _backend.RmsNorm(_feat, _feat, n, _encNorm);
            if (_lilicorr is not null)
            {
                // The head's anchor is the fused context feature of the position before the block: keep the newest ones.
                for (int r = Math.Max(0, positions.Length - AnchorRows - start); r < n; r++)
                    state.StoreAnchor(_feat + (nint)((long)r * HiddenSize * sizeof(float)), pos[r]);
            }
            for (int li = 0; li < _layers.Length; li++)
            {
                var l = _layers[li];
                Gemm(l.K, _feat, _k, n);
                Gemm(l.V, _feat, _v, n);
                _backend.NormRopeHeads(_k, NumKvHeads, l.KNorm, pos);
                _backend.WriteRing(state.RingKeys(li), state.RingValues(li), state.RingCapacity, pos, _k, _v);
            }
            foreach (int p in pos)
                state.NoteInjected(p);
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
        int nWalk = tokens.Length;
        if (nWalk < 1 || nWalk + 1 > BlockSize) throw new ArgumentOutOfRangeException(nameof(tokens));
        // LiLiCorr scores the whole trained block (its slot embeddings and bias table are per slot); the walk takes the first nWalk.
        int nDraft = _lilicorr is null ? nWalk : BlockSize - 1, b = nDraft + 1;
        int h = HiddenSize;

        Span<int> block = stackalloc int[b];
        block.Fill(MaskTokenId);
        block[0] = anchor;
        target.EmbedForDraft(block, _x);
        Span<int> positions = stackalloc int[b];
        for (int i = 0; i < b; i++) positions[i] = start + i;
        int lo = state.FirstVisible(start, Window);   // keys for the earliest query; later queries skip a few more

        for (int li = 0; li < _layers.Length; li++)
        {
            var l = _layers[li];
            _backend.RmsNorm(_x, _norm, b, l.AttnNorm);
            nint attnIn = _norm;
            if (l.AttnConvProj is { } acp)
            {
                Gemm(acp, _norm, _dyn, b);
                _backend.Conv(_norm, _dyn, l.AttnConvBase!, 0, b, _conv);
                attnIn = _conv;
            }
            Gemm(l.Q, attnIn, _q, b);
            Gemm(l.K, attnIn, _k, b);
            Gemm(l.V, attnIn, _v, b);
            _backend.NormRopeHeads(_q, NumHeads, l.QNorm, positions);
            _backend.NormRopeHeads(_k, NumKvHeads, l.KNorm, positions);
            long ta = System.Diagnostics.Stopwatch.GetTimestamp();
            _backend.Attend(state.RingKeys(li), state.RingValues(li), state.RingCapacity, lo, start, b, _q, _k, _v, _attn, l.Sinks);
            _attnTicks += System.Diagnostics.Stopwatch.GetTimestamp() - ta;
            Gemm(l.O, _attn, _o, b);
            nint attnOut = _o;
            if (l.AttnConvProj is not null)
            {
                _backend.Conv(_o, _dyn, l.AttnConvBase!, 1, b, _conv);
                attnOut = _conv;
            }
            _backend.Add(_x, attnOut, b * h);

            _backend.RmsNorm(_x, _norm, b, l.FfnNorm);
            nint ffnIn = _norm;
            if (l.FfnConvProj is { } fcp)
            {
                Gemm(fcp, _norm, _ffnDyn, b);
                _backend.Conv(_norm, _ffnDyn, l.FfnConvBase!, 0, b, _conv);
                ffnIn = _conv;
            }
            Gemm(l.Gate, ffnIn, _gate, b);
            Gemm(l.Up, ffnIn, _up, b);
            _backend.SwiGlu(_gate, _up, b * _ffn);
            Gemm(l.Down, _gate, _o, b);
            nint ffnOut = _o;
            if (l.FfnConvProj is not null)
            {
                _backend.Conv(_o, _ffnDyn, l.FfnConvBase!, 1, b, _conv);
                ffnOut = _conv;
            }
            _backend.Add(_x, ffnOut, b * h);
        }

        // Rows 1..b-1 predict the masked positions; row 0 (the anchor) is not a draft.
        nint hid = _x + (nint)((long)h * sizeof(float));
        _backend.RmsNorm(hid, hid, nDraft, _outNorm);
        if (_selHidden is { } sh)
        {
            Gemm(sh, hid, _gateSel, nDraft);   // before the head: ProjectForDraft may overwrite hid
            _backend.Download(_gateSel, _selGate, nDraft * _selRank);
        }
        if (_lilicorr is not null)
            _backend.Download(hid, _hiddenHost, nDraft * h);   // before the head, for the same reason
        long th = System.Diagnostics.Stopwatch.GetTimestamp();
        target.ProjectForDraft(hid, nDraft, _logits);
        long ts = System.Diagnostics.Stopwatch.GetTimestamp();
        _headTicks += ts - th;

        if (_lilicorr is { } head)
        {
            state.LoadAnchor(start - 1, _anchorHost);
            head.Walk(_logits, _hiddenHost, _anchorHost, target, tokens);
            _selectTicks += System.Diagnostics.Stopwatch.GetTimestamp() - ts;
            return;
        }
        if (_selHidden is null)
        {
            for (int i = 0; i < nDraft; i++)
                tokens[i] = TensorPrimitives.IndexOfMax(LastDraftLogits(i));
            return;
        }
        SelectorWalk(anchor, tokens);
        _selectTicks += System.Diagnostics.Stopwatch.GetTimestamp() - ts;
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

    /// <summary>Wall-time split of the drafter's work since the last call, in ms (for <c>DOTLLM_DFLASH_TRACE</c>).</summary>
    /// <returns>GEMM, attention, target lm-head and selector milliseconds.</returns>
    internal (double Gemm, double Attn, double Head, double Select) TakeProfile()
    {
        double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        var r = (_gemmTicks * f, _attnTicks * f, _headTicks * f, _selectTicks * f);
        _gemmTicks = _attnTicks = _headTicks = _selectTicks = 0;
        return r;
    }

    private long _gemmTicks, _attnTicks, _headTicks, _selectTicks;

    private void Gemm(in Matrix w, nint input, nint output, int rows)
    {
        long t = System.Diagnostics.Stopwatch.GetTimestamp();
        _backend.Gemm(w, input, output, rows);
        _gemmTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (nint buf in _buffers) _backend.Free(buf);
        if (_logits != null) { NativeMemory.AlignedFree(_logits); _logits = null; }
        if (_selGate != null) { NativeMemory.AlignedFree(_selGate); _selGate = null; }
        if (_hiddenHost != null) { NativeMemory.AlignedFree(_hiddenHost); _hiddenHost = null; }
        _lilicorr?.Dispose();
        _backend.Dispose();
        _ownedFile?.Dispose();
    }
}
