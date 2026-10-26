using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures.DFlash;

/// <summary>
/// NVIDIA's LiLiCorr lattice head (issue #1237), the draft selector of the <c>*-LiLiCorr</c> DFlash drafters (Nemotron-3.5 Bolt).
/// It replaces DFlash2's low-rank selector: per drafted slot it takes the top-k candidates of the drafter's logits, builds one
/// node per (slot, candidate) from the candidate's embedding, the slot's drafter hidden state and five rank/probability
/// features, runs a small bidirectional transformer over all <c>slots x k</c> nodes (an additive per-head slot-distance bias),
/// and scores candidate-to-candidate transitions with factorised unit vectors. The draft is the greedy walk through that
/// lattice from the anchor. It runs on the host: the nodes are few (7 x 8 on Bolt) and its weights stay in the mmap'd GGUF.
/// </summary>
/// <remarks>
/// Reference: llama.cpp <c>src/models/dflash.cpp</c> <c>build_lilicorr_scores</c> and the walk in
/// <c>common/speculative.cpp</c> (master 0ae0e1976), converted by <c>conversion/nemotron.py</c> <c>LiLiCorrDraftModel</c>.
/// The head's weights are stored F32 (the converter forces it) except the NVFP4 projections; nothing is widened or copied
/// beyond the bias/norm vectors and the 25 KB attention-bias table.
/// </remarks>
internal sealed unsafe class LiLiCorrHead : IDisposable
{
    private const int Features = 5;
    private const float FeatureNormEps = 1e-5f;

    private readonly ComputeThreadPool? _pool;
    private readonly bool _ownsPool;
    private readonly int _h, _d, _k, _heads, _slots, _nodes, _factor, _mlp, _vocab;
    private readonly float _rmsEps, _vecEps, _logitScale;

    private readonly DFlashDrafter.Matrix? _tokenProj;
    private readonly DFlashDrafter.Matrix _passProj, _contextProj, _featUp, _featDown, _factorIn, _outHead, _inHead, _anchorOut;
    private readonly float[]? _tokenProjB;
    private readonly float[] _passB, _contextB, _featNormW, _featNormB, _featUpB, _featDownB, _slotEmb, _rankEmb, _bias;
    private readonly float[] _outNorm, _anchorNorm, _factorInB, _outHeadB, _inHeadB, _anchorOutB;

    private sealed class Layer
    {
        public required float[] AttnNorm, MlpNorm, InProjB, OutProjB, Mlp0B, Mlp2B;
        public required DFlashDrafter.Matrix InProj, OutProj, Mlp0, Mlp2;
    }

    private readonly Layer[] _layers;

    // Host scratch, sized once for the full lattice.
    private readonly int[] _cand;
    private float* _x, _a, _qkv, _att, _tmp, _mlpBuf, _feat, _fs, _emb, _pass, _fin, _fac, _out, _in;
    private readonly float[] _anchorState = [], _anchorNormed = [], _anchorOutV = [], _scores = [];

    /// <summary>Candidates per drafted slot.</summary>
    public int TopK => _k;

    /// <summary>Drafted slots the head scores (<c>block_size - 1</c>).</summary>
    public int Slots => _slots;

    /// <summary>True when <paramref name="file"/> carries a LiLiCorr head.</summary>
    /// <param name="file">The drafter GGUF.</param>
    /// <returns>Whether <c>lilicorr.hidden_size</c> is present.</returns>
    public static bool IsPresent(GgufFile file) => file.Metadata.ContainsKey("lilicorr.hidden_size");

    /// <summary>Binds the head stored in <paramref name="file"/>.</summary>
    /// <param name="file">The drafter GGUF (kept alive by the drafter).</param>
    /// <param name="hidden">Drafter (and target) hidden size.</param>
    /// <param name="blockSize">Drafter block size; the head scores <c>blockSize - 1</c> slots.</param>
    /// <param name="drafterLayers">Drafter trunk layer count: the head's layers are stored as <c>blk.{drafterLayers + i}</c>.</param>
    /// <param name="rmsEps">The drafter's RMSNorm epsilon (the head's norms use it).</param>
    /// <param name="vocab">Target vocabulary size.</param>
    /// <param name="pool">Host thread pool for the head's GEMMs (null = a private pool sized to the machine).</param>
    public LiLiCorrHead(GgufFile file, int hidden, int blockSize, int drafterLayers, float rmsEps, int vocab, ComputeThreadPool? pool)
    {
        var md = file.Metadata;
        _h = hidden;
        _d = (int)md.GetUInt32("lilicorr.hidden_size");
        _k = (int)md.GetUInt32("lilicorr.candidate_topk");
        _heads = (int)md.GetUInt32("lilicorr.num_heads");
        int nLayers = (int)md.GetUInt32("lilicorr.num_layers");
        _factor = (int)md.GetUInt32("lilicorr.factor_dim");
        _mlp = (int)md.GetUInt32("lilicorr.mlp_hidden_size");
        _vecEps = md.GetFloat32("lilicorr.vector_eps");
        _logitScale = md.GetFloat32("lilicorr.logit_scale");
        _rmsEps = rmsEps;
        _vocab = vocab;
        _slots = blockSize - 1;
        _nodes = _slots * _k;
        if (_slots < 1 || _d <= 0 || _heads <= 0 || _d % _heads != 0 || _k < 1 || _k > vocab || nLayers < 1 || _factor <= 0 || _mlp <= 0)
            throw new InvalidOperationException("Invalid LiLiCorr head metadata.");

        DFlashDrafter.Matrix M(string name, int inDim, int outDim) => DFlashDrafter.ReadMatrix(file, name, inDim, outDim);
        float[] V(string name, int count) => DFlashDrafter.ReadVector(file, name, count);

        if (_d != _h)
        {
            _tokenProj = M("lilicorr.token_proj.weight", _h, _d);
            _tokenProjB = V("lilicorr.token_proj.bias", _d);
        }
        _passProj = M("lilicorr.pass_hidden_proj.weight", _h, _d);
        _passB = V("lilicorr.pass_hidden_proj.bias", _d);
        _contextProj = M("lilicorr.context_proj.weight", _h, _d);
        _contextB = V("lilicorr.context_proj.bias", _d);
        int featBid = drafterLayers + nLayers;
        _featNormW = V($"blk.{featBid}.ffn_norm.weight", Features);
        _featNormB = V($"blk.{featBid}.ffn_norm.bias", Features);
        _featUp = M($"blk.{featBid}.ffn_up.weight", Features, _d);
        _featUpB = V($"blk.{featBid}.ffn_up.bias", _d);
        _featDown = M($"blk.{featBid}.ffn_down.weight", _d, _d);
        _featDownB = V($"blk.{featBid}.ffn_down.bias", _d);
        _slotEmb = V("lilicorr.slot_embedding.weight", _slots * _d);
        _rankEmb = V("lilicorr.rank_embedding.weight", _k * _d);
        _bias = V("lilicorr.attention_bias.bias", _heads * _nodes * _nodes);   // [head][query][key]
        _layers = new Layer[nLayers];
        for (int i = 0; i < nLayers; i++)
        {
            int bid = drafterLayers + i;
            _layers[i] = new Layer
            {
                AttnNorm = V($"blk.{bid}.attn_norm.weight", _d),
                MlpNorm = V($"blk.{bid}.ffn_norm.weight", _d),
                InProj = M($"lilicorr.layers.{i}.attn.in_proj.weight", _d, 3 * _d),
                InProjB = V($"lilicorr.layers.{i}.attn.in_proj.bias", 3 * _d),
                OutProj = M($"blk.{bid}.attn_output.weight", _d, _d),
                OutProjB = V($"blk.{bid}.attn_output.bias", _d),
                Mlp0 = M($"blk.{bid}.ffn_up.weight", _d, _mlp),
                Mlp0B = V($"blk.{bid}.ffn_up.bias", _mlp),
                Mlp2 = M($"blk.{bid}.ffn_down.weight", _mlp, _d),
                Mlp2B = V($"blk.{bid}.ffn_down.bias", _d),
            };
        }
        _outNorm = V("lilicorr.output_norm.weight", _d);
        _anchorNorm = V("lilicorr.anchor_norm.weight", _d);
        _factorIn = M("lilicorr.factor_input_proj.weight", 3 * _d, _d);
        _factorInB = V("lilicorr.factor_input_proj.bias", _d);
        _outHead = M("lilicorr.out_head.weight", _d, _factor);
        _outHeadB = V("lilicorr.out_head.bias", _factor);
        _inHead = M("lilicorr.in_head.weight", _d, _factor);
        _inHeadB = V("lilicorr.in_head.bias", _factor);
        _anchorOut = M("lilicorr.anchor_out_head.weight", _d, _factor);
        _anchorOutB = V("lilicorr.anchor_out_head.bias", _factor);

        _ownsPool = pool is null;
        _pool = pool ?? new ComputeThreadPool(Math.Max(1, Environment.ProcessorCount / 2));
        _cand = new int[_nodes];
        _x = Alloc((long)_nodes * _d);
        _a = Alloc((long)_nodes * _d);
        _qkv = Alloc((long)_nodes * 3 * _d);
        _att = Alloc((long)_nodes * _d);
        _tmp = Alloc((long)_nodes * Math.Max(_d, _factor));
        _mlpBuf = Alloc((long)_nodes * _mlp);
        _feat = Alloc((long)_nodes * Features);
        _fs = Alloc((long)_nodes * _d);
        _emb = Alloc((long)_nodes * _h);
        _pass = Alloc((long)_slots * _d);
        _fin = Alloc((long)_nodes * 3 * _d);
        _fac = Alloc((long)_nodes * _d);
        _out = Alloc((long)_nodes * _factor);
        _in = Alloc((long)_nodes * _factor);
        _anchorState = new float[_d];
        _anchorNormed = new float[_d];
        _anchorOutV = new float[_factor];
        _scores = new float[_nodes];
    }

    private static float* Alloc(long floats) => (float*)NativeMemory.AlignedAlloc((nuint)(Math.Max(floats, 1) * sizeof(float)), 64);

    /// <summary>The weights the head reads (for a byte count in traces).</summary>
    public long WeightBytes
    {
        get
        {
            long b = 0;
            void Add(DFlashDrafter.Matrix m) => b += Dequantize.RowByteSize(m.In, m.Qt) * m.Out;
            if (_tokenProj is { } tp) Add(tp);
            Add(_passProj); Add(_contextProj); Add(_featUp); Add(_featDown); Add(_factorIn); Add(_outHead); Add(_inHead); Add(_anchorOut);
            foreach (var l in _layers) { Add(l.InProj); Add(l.OutProj); Add(l.Mlp0); Add(l.Mlp2); }
            return b;
        }
    }

    /// <summary>
    /// Scores the lattice of one drafted block and walks it greedily: <paramref name="tokens"/>[i] is the drafted token of slot i.
    /// </summary>
    /// <param name="logits">Host logits of the <see cref="Slots"/> drafted rows, <c>[slots, vocab]</c>.</param>
    /// <param name="hidden">Host final-normed drafter hidden states of those rows, <c>[slots, hidden]</c>.</param>
    /// <param name="anchor">Host context feature (fc + enc.output_norm) of the position before the anchor token.</param>
    /// <param name="target">The target, for the candidates' embeddings.</param>
    /// <param name="tokens">Receives the walk, <c>1 &lt;= Length &lt;= Slots</c>.</param>
    public void Walk(float* logits, float* hidden, ReadOnlySpan<float> anchor, IDFlashTargetOps target, Span<int> tokens)
    {
        if (tokens.Length < 1 || tokens.Length > _slots) throw new ArgumentOutOfRangeException(nameof(tokens));
        int d = _d, k = _k, n = _nodes;

        // ── Candidates and their five features, per slot ──
        Span<int> idx = stackalloc int[k];
        Span<float> val = stackalloc float[k];
        for (int s = 0; s < _slots; s++)
        {
            var row = new ReadOnlySpan<float>(logits + (long)s * _vocab, _vocab);
            DFlashDrafter.TopK(row, idx, val);
            SortDescending(idx, val);
            float max = val[0];
            double sum = 0;
            for (int v = 0; v < row.Length; v++) sum += MathF.Exp(row[v] - max);
            float logSum = (float)Math.Log(sum);
            float lp0 = val[0] - max - logSum;
            for (int r = 0; r < k; r++)
            {
                int node = s * k + r;
                _cand[node] = idx[r];
                float lp = val[r] - max - logSum;
                float* f = _feat + (long)node * Features;
                f[0] = lp;
                f[1] = MathF.Exp(lp);
                f[2] = lp - lp0;
                f[3] = k == 1 ? 0f : (float)r / (k - 1);
                f[4] = r == 0 ? 1f : 0f;
            }
        }

        // feature_states = mlp3(silu(mlp1(LayerNorm(features))))
        for (int node = 0; node < n; node++)
        {
            float* f = _feat + (long)node * Features;
            float mean = 0f;
            for (int j = 0; j < Features; j++) mean += f[j];
            mean /= Features;
            float var = 0f;
            for (int j = 0; j < Features; j++) { f[j] -= mean; var += f[j] * f[j]; }
            float inv = 1f / MathF.Sqrt(var / Features + FeatureNormEps);
            for (int j = 0; j < Features; j++) f[j] = f[j] * inv * _featNormW[j] + _featNormB[j];
        }
        Linear(_featUp, _feat, _tmp, n, _featUpB);
        Silu(_tmp, (long)n * d);
        Linear(_featDown, _tmp, _fs, n, _featDownB);

        // token_states: the target's embedding of each candidate (projected when the head is narrower than the trunk).
        target.EmbedHost(_cand, _emb);
        if (_tokenProj is { } tp) Linear(tp, _emb, _x, n, _tokenProjB);
        else new Span<float>(_emb, n * d).CopyTo(new Span<float>(_x, n * d));

        // pass_states: the slot's drafter hidden state, shared by its k candidates.
        Linear(_passProj, hidden, _pass, _slots, _passB);
        for (int s = 0; s < _slots; s++)
            for (int r = 0; r < k; r++)
            {
                var x = new Span<float>(_x + (long)(s * k + r) * d, d);
                TensorPrimitives.Add(x, new ReadOnlySpan<float>(_pass + (long)s * d, d), x);
                TensorPrimitives.Add(x, new ReadOnlySpan<float>(_fs + (long)(s * k + r) * d, d), x);
                TensorPrimitives.Add(x, _slotEmb.AsSpan(s * d, d), x);
                TensorPrimitives.Add(x, _rankEmb.AsSpan(r * d, d), x);
            }

        // anchor_state = context_proj(anchor feature)
        fixed (float* ap = anchor)
        fixed (float* asp = _anchorState)
            Linear(_contextProj, ap, asp, 1, _contextB);

        // ── The transformer over all nodes ──
        int hd = d / _heads;
        float scale = 1f / MathF.Sqrt(hd);
        float[] scores = _scores;
        foreach (var l in _layers)
        {
            RmsRows(_x, _a, n, l.AttnNorm);
            Linear(l.InProj, _a, _qkv, n, l.InProjB);
            for (int h = 0; h < _heads; h++)
                for (int q = 0; q < n; q++)
                {
                    var qv = new ReadOnlySpan<float>(_qkv + (long)q * 3 * d + h * hd, hd);
                    float m = float.NegativeInfinity;
                    for (int key = 0; key < n; key++)
                    {
                        var kv = new ReadOnlySpan<float>(_qkv + (long)key * 3 * d + d + h * hd, hd);
                        float sc = TensorPrimitives.Dot(qv, kv) * scale + _bias[((long)h * n + q) * n + key];
                        scores[key] = sc;
                        if (sc > m) m = sc;
                    }
                    float sum = 0f;
                    for (int key = 0; key < n; key++) { scores[key] = MathF.Exp(scores[key] - m); sum += scores[key]; }
                    var o = new Span<float>(_att + (long)q * d + h * hd, hd);
                    o.Clear();
                    for (int key = 0; key < n; key++)
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(_qkv + (long)key * 3 * d + 2 * d + h * hd, hd), scores[key] / sum, o, o);
                }
            Linear(l.OutProj, _att, _tmp, n, l.OutProjB);
            TensorPrimitives.Add(new ReadOnlySpan<float>(_x, n * d), new ReadOnlySpan<float>(_tmp, n * d), new Span<float>(_x, n * d));
            RmsRows(_x, _a, n, l.MlpNorm);
            Linear(l.Mlp0, _a, _mlpBuf, n, l.Mlp0B);
            Silu(_mlpBuf, (long)n * _mlp);
            Linear(l.Mlp2, _mlpBuf, _tmp, n, l.Mlp2B);
            TensorPrimitives.Add(new ReadOnlySpan<float>(_x, n * d), new ReadOnlySpan<float>(_tmp, n * d), new Span<float>(_x, n * d));
        }
        RmsRows(_x, _x, n, _outNorm);
        RmsNorm.Execute(_anchorState, _anchorNorm, _rmsEps, _anchorNormed);

        // factors = silu(factor_input_proj([x, anchor, x * anchor]))
        for (int node = 0; node < n; node++)
        {
            float* fin = _fin + (long)node * 3 * d;
            var x = new ReadOnlySpan<float>(_x + (long)node * d, d);
            x.CopyTo(new Span<float>(fin, d));
            _anchorNormed.CopyTo(new Span<float>(fin + d, d));
            TensorPrimitives.Multiply(x, _anchorNormed, new Span<float>(fin + 2 * d, d));
        }
        Linear(_factorIn, _fin, _fac, n, _factorInB);
        Silu(_fac, (long)n * d);
        Linear(_outHead, _fac, _out, n, _outHeadB);
        Linear(_inHead, _fac, _in, n, _inHeadB);
        L2Rows(_out, n, _factor);
        L2Rows(_in, n, _factor);
        fixed (float* an = _anchorNormed)
        fixed (float* ao = _anchorOutV)
        {
            Linear(_anchorOut, an, ao, 1, _anchorOutB);
            L2Rows(ao, 1, _factor);
        }

        // ── Greedy walk: slot 0 is scored against the anchor, slot i against slot i-1's chosen candidate ──
        int pred = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            ReadOnlySpan<float> from = i == 0 ? _anchorOutV : new ReadOnlySpan<float>(_out + (long)((i - 1) * k + pred) * _factor, _factor);
            int best = 0;
            float bestScore = float.NegativeInfinity;
            for (int j = 0; j < k; j++)
            {
                float sc = _logitScale * TensorPrimitives.Dot(new ReadOnlySpan<float>(_in + (long)(i * k + j) * _factor, _factor), from);
                if (sc > bestScore) { bestScore = sc; best = j; }   // first maximum, as std::max_element
            }
            pred = best;
            tokens[i] = _cand[i * k + best];
        }
    }

    /// <summary>Insertion sort by value, descending (k is small); equal values keep the lower token id first.</summary>
    private static void SortDescending(Span<int> idx, Span<float> val)
    {
        for (int i = 1; i < idx.Length; i++)
        {
            int ti = idx[i];
            float tv = val[i];
            int j = i - 1;
            while (j >= 0 && (val[j] < tv || (val[j] == tv && idx[j] > ti)))
            {
                idx[j + 1] = idx[j];
                val[j + 1] = val[j];
                j--;
            }
            idx[j + 1] = ti;
            val[j + 1] = tv;
        }
    }

    private void Linear(in DFlashDrafter.Matrix w, float* x, float* y, int rows, float[]? bias)
    {
        NemotronHTransformerModel.GemmDispatch(w.Ptr, w.Qt, x, y, w.Out, w.In, rows, _pool, null);
        for (int r = 0; r < rows; r++)
        {
            var o = new Span<float>(y + (long)r * w.Out, w.Out);
            if (w.Scale != 1f) TensorPrimitives.Multiply(o, w.Scale, o);
            if (bias is not null) TensorPrimitives.Add(o, bias, o);
        }
    }

    private void RmsRows(float* src, float* dst, int rows, float[] w)
    {
        for (int r = 0; r < rows; r++)
            RmsNorm.Execute(new ReadOnlySpan<float>(src + (long)r * _d, _d), w, _rmsEps, new Span<float>(dst + (long)r * _d, _d));
    }

    private static void Silu(float* x, long n)
    {
        var s = new Span<float>(x, checked((int)n));
        SiLu.Execute(s, s);
    }

    /// <summary>ggml <c>l2_norm</c>: <c>x / max(|x|, eps)</c> per row.</summary>
    private void L2Rows(float* x, int rows, int width)
    {
        for (int r = 0; r < rows; r++)
        {
            var v = new Span<float>(x + (long)r * width, width);
            float norm = MathF.Sqrt(TensorPrimitives.Dot((ReadOnlySpan<float>)v, v));
            TensorPrimitives.Multiply(v, 1f / MathF.Max(norm, _vecEps), v);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        float*[] all = [_x, _a, _qkv, _att, _tmp, _mlpBuf, _feat, _fs, _emb, _pass, _fin, _fac, _out, _in];
        for (int i = 0; i < all.Length; i++)
            if (all[i] != null) NativeMemory.AlignedFree(all[i]);
        _x = _a = _qkv = _att = _tmp = _mlpBuf = _feat = _fs = _emb = _pass = _fin = _fac = _out = _in = null;
        if (_ownsPool) _pool?.Dispose();
    }
}
