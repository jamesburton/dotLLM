using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Architectures.DFlash;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// DFlash / DFlash2 drafter on the CPU (issue #1175): the drafter's cached forward (context ring, rollback, window,
/// convolutions, selector) against a stateless reference written from z-lab's <c>dflash/model.py</c>, and the end-to-end
/// gate: speculative decoding with the drafter is token-identical to plain greedy decode on a recurrent (GDN) trunk.
/// </summary>
public sealed unsafe class DFlashDrafterTests : IDisposable
{
    // Tiny but non-degenerate: KV heads > 1 with GQA, a window shorter than the context, a 2-tap conv with
    // several groups, a selector whose top-k is smaller than the vocabulary.
    private const int H = 32, Heads = 4, KvHeads = 2, Hd = 8, Ffn = 24, Layers = 2, Block = 4, Window = 5;
    private const int Group = 8, Kernel = 2, Rank = 8, TopK = 4, Vocab = 12, Mask = 11;
    private static readonly int[] TargetLayers = [1, 2];   // a mid-trunk input and the last layer's output

    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-dflash-{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _out;

    public DFlashDrafterTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    // ── Fixture ─────────────────────────────────────────────────────────

    private sealed class Weights
    {
        public readonly Dictionary<string, float[]> T = new();
        public float[] this[string n] => T[n];
    }

    private static (byte[] Gguf, Weights W) BuildDrafter(uint seed, bool dflash2)
    {
        var rng = new SyntheticGemma4Gguf.Xorshift(seed);
        var w = new GgufWriter();
        var weights = new Weights();
        w.AddString("general.architecture", "dflash");
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32("dflash.embedding_length", H);
        w.AddUInt32("dflash.block_count", Layers);
        w.AddUInt32("dflash.attention.head_count", Heads);
        w.AddUInt32("dflash.attention.head_count_kv", KvHeads);
        w.AddUInt32("dflash.attention.key_length", Hd);
        w.AddUInt32("dflash.attention.value_length", Hd);
        w.AddUInt32("dflash.feed_forward_length", Ffn);
        w.AddFloat32("dflash.attention.layer_norm_rms_epsilon", 1e-6f);
        w.AddFloat32("dflash.rope.freq_base", 10000f);
        w.AddBool("dflash.attention.causal", false);
        w.AddUInt32("dflash.block_size", Block);
        w.AddUInt32("dflash.attention.sliding_window", Window);
        w.AddInt32Array("dflash.target_layers", TargetLayers);
        w.AddUInt32("tokenizer.ggml.mask_token_id", Mask);
        if (dflash2)
        {
            w.AddUInt32("dflash.conv_kernel_size", Kernel);
            w.AddUInt32("dflash.conv_group_size", Group);
            w.AddUInt32("dflash.selector_rank", Rank);
            w.AddUInt32("dflash.selector_top_k", TopK);
        }

        void Add(string name, int[] dims, float scale, bool norm = false)
        {
            int n = 1;
            foreach (int d in dims) n *= d;
            var v = new float[n];
            for (int i = 0; i < n; i++) v[i] = norm ? 1f + 0.2f * rng.NextSigned(1f) : scale * rng.NextSigned(1f);
            weights.T[name] = v;
            var bytes = new byte[n * 4];
            Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length);
            w.AddTensor(name, dims, 0, bytes);
        }

        int convProj = 2 * Kernel * (H / Group);
        Add("fc.weight", [TargetLayers.Length * H, H], 0.15f);
        Add("enc.output_norm.weight", [H], 0, norm: true);
        Add("output_norm.weight", [H], 0, norm: true);
        for (int i = 0; i < Layers; i++)
        {
            Add($"blk.{i}.attn_norm.weight", [H], 0, norm: true);
            Add($"blk.{i}.ffn_norm.weight", [H], 0, norm: true);
            Add($"blk.{i}.attn_q_norm.weight", [Hd], 0, norm: true);
            Add($"blk.{i}.attn_k_norm.weight", [Hd], 0, norm: true);
            Add($"blk.{i}.attn_q.weight", [H, Heads * Hd], 0.3f);
            Add($"blk.{i}.attn_k.weight", [H, KvHeads * Hd], 0.3f);
            Add($"blk.{i}.attn_v.weight", [H, KvHeads * Hd], 0.3f);
            Add($"blk.{i}.attn_output.weight", [Heads * Hd, H], 0.2f);
            Add($"blk.{i}.ffn_gate.weight", [H, Ffn], 0.3f);
            Add($"blk.{i}.ffn_up.weight", [H, Ffn], 0.3f);
            Add($"blk.{i}.ffn_down.weight", [Ffn, H], 0.2f);
            if (dflash2)
            {
                Add($"blk.{i}.attn_conv_base", [H, Kernel, 2], 0.5f);
                Add($"blk.{i}.attn_conv_proj.weight", [H, convProj], 0.2f);
                Add($"blk.{i}.ffn_conv_base", [H, Kernel, 2], 0.5f);
                Add($"blk.{i}.ffn_conv_proj.weight", [H, convProj], 0.2f);
            }
        }
        if (dflash2)
        {
            Add("selector_hidden.weight", [H, Rank], 0.4f);
            Add("selector_predecessor.weight", [Rank, Vocab], 0.8f);
            Add("selector_successor.weight", [Rank, Vocab], 0.8f);
        }
        return (w.Build(), weights);
    }

    /// <summary>A stand-in target: an embedding table and an output head, both random.</summary>
    private sealed class FakeTarget : IDFlashTargetOps
    {
        public readonly float[] Embed = new float[Vocab * H], Head = new float[Vocab * H];

        public FakeTarget(uint seed)
        {
            var rng = new SyntheticGemma4Gguf.Xorshift(seed);
            for (int i = 0; i < Embed.Length; i++) Embed[i] = rng.NextSigned(1f);
            for (int i = 0; i < Head.Length; i++) Head[i] = 0.5f * rng.NextSigned(1f);
        }

        public void EmbedForDraft(ReadOnlySpan<int> tokens, nint dst)
        {
            for (int t = 0; t < tokens.Length; t++)
                Embed.AsSpan(tokens[t] * H, H).CopyTo(new Span<float>((float*)dst + t * H, H));
        }

        public void ProjectForDraft(nint hiddenRows, int rows, float* logits)
        {
            float* hidden = (float*)hiddenRows;
            for (int r = 0; r < rows; r++)
                for (int v = 0; v < Vocab; v++)
                {
                    float s = 0;
                    for (int i = 0; i < H; i++) s += Head[v * H + i] * hidden[r * H + i];
                    logits[r * Vocab + v] = s;
                }
        }
    }

    // ── Stateless reference (dflash/model.py) ───────────────────────────

    private static float[] MatVec(float[] w, int inDim, int outDim, ReadOnlySpan<float> x)
    {
        var y = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            double s = 0;
            for (int i = 0; i < inDim; i++) s += (double)w[o * inDim + i] * x[i];
            y[o] = (float)s;
        }
        return y;
    }

    private static float[] Rms(ReadOnlySpan<float> x, float[] g)
    {
        double ss = 0;
        foreach (float v in x) ss += (double)v * v;
        float inv = (float)(1.0 / Math.Sqrt(ss / x.Length + 1e-6));
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * inv * g[i];
        return y;
    }

    private static void NormRope(float[] v, int heads, float[] g, int pos)
    {
        for (int h = 0; h < heads; h++)
        {
            var n = Rms(v.AsSpan(h * Hd, Hd), g);
            for (int i = 0; i < Hd / 2; i++)
            {
                double a = pos / Math.Pow(10000.0, 2.0 * i / Hd);
                float c = (float)Math.Cos(a), s = (float)Math.Sin(a);
                float x0 = n[i], x1 = n[i + Hd / 2];
                v[h * Hd + i] = x0 * c - x1 * s;
                v[h * Hd + i + Hd / 2] = x1 * c + x0 * s;
            }
        }
    }

    private static float RoundHalf(float x) => (float)(Half)x;

    private static float[][] Conv(float[][] x, float[][] dyn, float[] baseK, int side)
    {
        int groups = H / Group;
        var y = new float[x.Length][];
        for (int p = 0; p < x.Length; p++)
        {
            y[p] = new float[H];
            for (int c = 0; c < H; c++)
                for (int tap = 0; tap < Kernel && tap <= p; tap++)
                    y[p][c] += (baseK[(side * Kernel + tap) * H + c] + dyn[p][(side * Kernel + tap) * groups + c / Group]) * x[p - tap][c];
        }
        return y;
    }

    /// <summary>Drafts for a block anchored at <paramref name="start"/> from the full context taps (no cache).</summary>
    private static (int[] Tokens, float[][] Logits) Reference(Weights w, FakeTarget target, bool dflash2,
        float[][] taps, int anchor, int start, int nDraft)
    {
        int b = nDraft + 1, kvDim = KvHeads * Hd;
        var feat = new float[start][];
        for (int p = 0; p < start; p++)
            feat[p] = Rms(MatVec(w["fc.weight"], TargetLayers.Length * H, H, taps[p]), w["enc.output_norm.weight"]);

        var x = new float[b][];
        for (int i = 0; i < b; i++) x[i] = target.Embed.AsSpan((i == 0 ? anchor : Mask) * H, H).ToArray();

        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}";
            var kc = new float[start][]; var vc = new float[start][];
            for (int j = 0; j < start; j++)
            {
                kc[j] = MatVec(w[$"{p}.attn_k.weight"], H, kvDim, feat[j]);
                NormRope(kc[j], KvHeads, w[$"{p}.attn_k_norm.weight"], j);
                vc[j] = MatVec(w[$"{p}.attn_v.weight"], H, kvDim, feat[j]);
                for (int i = 0; i < kvDim; i++) { kc[j][i] = RoundHalf(kc[j][i]); vc[j][i] = RoundHalf(vc[j][i]); }
            }

            var n = x.Select(r => Rms(r, w[$"{p}.attn_norm.weight"])).ToArray();
            float[][]? dyn = null;
            int convProj = 2 * Kernel * (H / Group);
            if (dflash2)
            {
                dyn = n.Select(r => MatVec(w[$"{p}.attn_conv_proj.weight"], H, convProj, r)).ToArray();
                n = Conv(n, dyn, w[$"{p}.attn_conv_base"], 0);
            }
            var q = n.Select(r => MatVec(w[$"{p}.attn_q.weight"], H, Heads * Hd, r)).ToArray();
            var k = n.Select(r => MatVec(w[$"{p}.attn_k.weight"], H, kvDim, r)).ToArray();
            var v = n.Select(r => MatVec(w[$"{p}.attn_v.weight"], H, kvDim, r)).ToArray();
            for (int i = 0; i < b; i++)
            {
                NormRope(q[i], Heads, w[$"{p}.attn_q_norm.weight"], start + i);
                NormRope(k[i], KvHeads, w[$"{p}.attn_k_norm.weight"], start + i);
            }

            var attn = new float[b][];
            for (int i = 0; i < b; i++)
            {
                attn[i] = new float[Heads * Hd];
                for (int h = 0; h < Heads; h++)
                {
                    int kh = h / (Heads / KvHeads);
                    var keys = new List<(float[] K, float[] V)>();
                    for (int j = 0; j < start; j++)
                        if (start + i - j < Window) keys.Add((kc[j], vc[j]));
                    for (int j = 0; j < b; j++) keys.Add((k[j], v[j]));   // non-causal within the block
                    var s = keys.Select(kv =>
                    {
                        double d = 0;
                        for (int e = 0; e < Hd; e++) d += (double)q[i][h * Hd + e] * kv.K[kh * Hd + e];
                        return d / Math.Sqrt(Hd);
                    }).ToArray();
                    double m = s.Max(), sum = s.Sum(z => Math.Exp(z - m));
                    for (int j = 0; j < keys.Count; j++)
                    {
                        double a = Math.Exp(s[j] - m) / sum;
                        for (int e = 0; e < Hd; e++) attn[i][h * Hd + e] += (float)(a * keys[j].V[kh * Hd + e]);
                    }
                }
            }
            var o = attn.Select(r => MatVec(w[$"{p}.attn_output.weight"], Heads * Hd, H, r)).ToArray();
            if (dflash2) o = Conv(o, dyn!, w[$"{p}.attn_conv_base"], 1);
            for (int i = 0; i < b; i++) for (int c = 0; c < H; c++) x[i][c] += o[i][c];

            n = x.Select(r => Rms(r, w[$"{p}.ffn_norm.weight"])).ToArray();
            float[][]? dyn2 = null;
            if (dflash2)
            {
                dyn2 = n.Select(r => MatVec(w[$"{p}.ffn_conv_proj.weight"], H, convProj, r)).ToArray();
                n = Conv(n, dyn2, w[$"{p}.ffn_conv_base"], 0);
            }
            var f = n.Select(r =>
            {
                var g = MatVec(w[$"{p}.ffn_gate.weight"], H, Ffn, r);
                var u = MatVec(w[$"{p}.ffn_up.weight"], H, Ffn, r);
                for (int e = 0; e < Ffn; e++) g[e] = g[e] / (1f + MathF.Exp(-g[e])) * u[e];
                return MatVec(w[$"{p}.ffn_down.weight"], Ffn, H, g);
            }).ToArray();
            if (dflash2) f = Conv(f, dyn2!, w[$"{p}.ffn_conv_base"], 1);
            for (int i = 0; i < b; i++) for (int c = 0; c < H; c++) x[i][c] += f[i][c];
        }

        var hid = x.Skip(1).Select(r => Rms(r, w["output_norm.weight"])).ToArray();
        var logits = new float[nDraft][];
        for (int i = 0; i < nDraft; i++)
        {
            logits[i] = new float[Vocab];
            fixed (float* hp = hid[i]) fixed (float* lp = logits[i]) target.ProjectForDraft((nint)hp, 1, lp);
        }

        var tokens = new int[nDraft];
        if (!dflash2)
        {
            for (int i = 0; i < nDraft; i++) tokens[i] = Array.IndexOf(logits[i], logits[i].Max());
            return (tokens, logits);
        }
        int pred = anchor;
        for (int i = 0; i < nDraft; i++)
        {
            var gate = MatVec(w["selector_hidden.weight"], H, Rank, hid[i]);
            var cand = logits[i].Select((val, idx) => (val, idx)).OrderByDescending(t => t.val).Take(TopK).ToArray();
            double best = double.NegativeInfinity; int bestTok = -1;
            foreach (var (val, idx) in cand)
            {
                double sc = val;
                for (int r = 0; r < Rank; r++)
                    sc += (double)w["selector_predecessor.weight"][pred * Rank + r] * gate[r] * w["selector_successor.weight"][idx * Rank + r];
                if (sc > best) { best = sc; bestTok = idx; }
            }
            tokens[i] = pred = bestTok;
        }
        return (tokens, logits);
    }

    // ── Tests ───────────────────────────────────────────────────────────

    /// <summary>
    /// The cached drafter equals the stateless reference across rounds: rows injected in chunks, a rejected verify's rows
    /// injected and rolled back (then overwritten by different taps), a context longer than the window (and than the ring's
    /// slack, so the ring wraps), and blocks shorter than the trained size.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Draft_MatchesStatelessReference_AcrossInjectRollbackAndWindow(bool dflash2)
        => AssertMatchesReference(_dir, dflash2, () => new DFlashCpuBackend(null), t => t, 2e-4f, _out);

    /// <summary>Host buffers for the stand-in target, whatever memory the backend uses.</summary>
    internal interface ITargetMemory
    {
        /// <summary>Copies host floats into backend memory.</summary>
        void Up(float* src, nint dst, int count);

        /// <summary>Copies backend floats to the host.</summary>
        void Down(nint src, float* dst, int count);
    }

    /// <summary>The stand-in target over a backend's memory: rows are staged through the host.</summary>
    internal sealed class StagedTarget(IDFlashTargetOps host, ITargetMemory mem) : IDFlashTargetOps
    {
        public void EmbedForDraft(ReadOnlySpan<int> tokens, nint dst)
        {
            var tmp = new float[tokens.Length * H];
            fixed (float* p = tmp) { host.EmbedForDraft(tokens, (nint)p); mem.Up(p, dst, tmp.Length); }
        }

        public void ProjectForDraft(nint hidden, int rows, float* logits)
        {
            var tmp = new float[rows * H];
            fixed (float* p = tmp) { mem.Down(hidden, p, tmp.Length); host.ProjectForDraft((nint)p, rows, logits); }
        }
    }

    internal static void AssertMatchesReference(string dir, bool dflash2, Func<IDFlashBackend> backend,
        Func<IDFlashTargetOps, IDFlashTargetOps> wrapTarget, float tol, ITestOutputHelper output)
    {
        var (bytes, w) = BuildDrafter(0xD1F1A5u + (dflash2 ? 1u : 0u), dflash2);
        string path = Path.Combine(dir, $"drafter-{dflash2}.gguf");
        File.WriteAllBytes(path, bytes);
        using var drafter = DFlashDrafter.Open(path, H, 2, Vocab, backend());
        Assert.Equal(dflash2, drafter.IsDFlash2);
        var target = new FakeTarget(0x7A5Eu);
        var ops = wrapTarget(target);
        using var state = new DFlashMtpState(drafter);
        var rng = new SyntheticGemma4Gguf.Xorshift(0x5EEDu);
        int width = drafter.TapRowWidth;

        float[] Row() { var r = new float[width]; for (int i = 0; i < width; i++) r[i] = RoundHalf(2f * rng.NextSigned(1f)); return r; }
        var committed = new List<float[]>();
        void Inject(List<float[]> rows, int firstPos)
        {
            Half* buf = state.EnsureTapRows(rows.Count);
            var positions = new int[rows.Count];
            for (int r = 0; r < rows.Count; r++)
            {
                for (int s = 0; s < TargetLayers.Length; s++) state.CaptureTap(r, s, rows[r].AsSpan(s * H, H));
                positions[r] = firstPos + r;
            }
            state.InjectCaptured(positions);
        }

        // Prefill of 3 rows, then rounds: inject a 4-row "verify", roll back to accept 1, re-inject different rows, draft.
        var prefill = Enumerable.Range(0, 3).Select(_ => Row()).ToList();
        Inject(prefill, 0);
        committed.AddRange(prefill);
        int checks = 0;
        for (int round = 0; round < 6; round++)
        {
            int start = committed.Count;
            Assert.Equal(start, state.CurrentLength);
            int nDraft = 1 + round % (Block - 1);
            var got = new int[nDraft];
            int anchor = (round * 5 + 2) % Vocab;
            got[0] = state.NextDraft(ops, anchor, start, nDraft);
            for (int i = 1; i < nDraft; i++) got[i] = state.NextDraft(ops, got[i - 1], start + i, nDraft);
            var (want, wantLogits) = Reference(w, target, dflash2, committed.ToArray(), anchor, start, nDraft);
            for (int i = 0; i < nDraft; i++)
            {
                var gl = state.DraftLogits(start + i);
                for (int v = 0; v < Vocab; v++)
                    Assert.True(Math.Abs(gl[v] - wantLogits[i][v]) <= tol + tol * Math.Abs(wantLogits[i][v]),
                        $"round {round} draft {i} logit {v}: {gl[v]} vs {wantLogits[i][v]}");
            }
            Assert.Equal(want, got);
            checks++;

            // A verify of 4 rows of which 1 is accepted: inject all, roll back, the accepted row survives.
            var verify = Enumerable.Range(0, 4).Select(_ => Row()).ToList();
            Inject(verify, start);
            state.Rollback(start + 1);
            committed.Add(verify[0]);
            // The next round re-injects the following positions with different taps (overwriting the rejected rows).
            var more = Enumerable.Range(0, 2).Select(_ => Row()).ToList();
            Inject(more, start + 1);
            committed.AddRange(more);
        }
        Assert.True(committed.Count > drafter.Window + 2 * drafter.BlockSize, "the context must outgrow the ring");
        output.WriteLine($"{checks} rounds matched; context {committed.Count} rows, window {drafter.Window}, ring {state.RingCapacity}");
    }

    /// <summary>A target backend for the end-to-end gate: the model and a KV-cache for it.</summary>
    internal delegate (IModel Model, Core.Attention.IKvCache Kv) BackendFactory(GgufFile gguf, ModelConfig config);

    internal static (IModel, Core.Attention.IKvCache) Cpu(GgufFile gguf, ModelConfig config)
    {
        var m = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        return (m, new SimpleKvCache(m.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength));
    }

    /// <summary>
    /// End-to-end gate: MTP-decoder rounds drafted by DFlash on a GDN + attention trunk (taps from a mid-trunk layer input
    /// and the last layer's output) emit exactly the plain greedy sequence, with per-row recurrent snapshots and with
    /// checkpoint + replay, and some drafts are accepted.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SpeculativeDecode_WithDFlash_IsTokenIdenticalToGreedy(bool snapshots)
        => AssertTokenIdentical(_dir, Cpu, snapshots, _out);

    /// <summary>
    /// #1360: a draft-vocabulary limit only restricts which tokens are PROPOSED. The MTP decoder's verify still uses the full head,
    /// so the emitted sequence stays the plain greedy one, and the drafter never proposes a token id at or above the limit.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void DraftVocabLimit_IsTokenIdenticalToGreedy_AndNeverDraftsAboveLimit(bool snapshots)
    {
        const int Limit = 6;
        DFlashDrafter.DraftVocabLimitOverride = Limit;
        try
        {
            AssertTokenIdentical(_dir, Cpu, snapshots, _out);

            string trunk = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_dir, "trunk-limit.gguf"), withMtp: false, contextLength: 96);
            var (bytes, _) = BuildDrafter(0xBEEFu, dflash2: true);
            string drafterPath = Path.Combine(_dir, "drafter-limit.gguf");
            File.WriteAllBytes(drafterPath, bytes);
            using var gguf = GgufFile.Open(trunk);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            var (model, kv) = Cpu(gguf, config);
            using var _m = model as IDisposable;
            using var _k = kv;
            ((IDFlashDraftAttachable)model).AttachDFlashDrafter(drafterPath);
            using var mtp = model.CreateMtpState()!;
            int[] prompt = [1, 5, 3, 7, 2, 9];
            using (model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, kv, adapter: null, mtp)) { }
            var state = (DFlashMtpState)mtp;
            int position = prompt.Length;
            int tok = 3;
            for (int i = 0; i < Block - 1; i++)
            {
                tok = state.NextDraft((IDFlashTargetOps)model, tok, position + i, Block - 1);
                Assert.InRange(tok, 0, Limit - 1);
                Assert.Equal(Limit, state.DraftLogits(position + i).Length);   // the restricted head ran, not the full one
            }
        }
        finally { DFlashDrafter.DraftVocabLimitOverride = null; }
    }

    /// <summary>The gate on any backend: the greedy reference is the same backend's plain decode.</summary>
    internal static void AssertTokenIdentical(string dir, BackendFactory backend, bool snapshots, ITestOutputHelper output)
    {
        string trunk = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(dir, "trunk.gguf"), withMtp: false, contextLength: 96);
        int[] prompt = [1, 5, 3, 7, 2, 9];
        const int NewTokens = 40;

        List<int> greedy;
        using (var g = GgufFile.Open(trunk))
        {
            var cfg = GgufModelConfigExtractor.Extract(g.Metadata);
            var (plain, kv) = backend(g, cfg);
            using (plain as IDisposable) using (kv)
            {
                greedy = [.. prompt];
                using (var l = plain.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, kv)) greedy.Add(ArgMaxLast(l, cfg.VocabSize));
                while (greedy.Count < prompt.Length + NewTokens + Block)
                    using (var l = plain.Forward([greedy[^1]], [greedy.Count - 1], -1, kv)) greedy.Add(ArgMaxLast(l, cfg.VocabSize));
            }
        }

        // A random drafter rarely agrees with the trunk; sweep seeds (each one must stay token-identical) until one also
        // gets drafts accepted, so both the accept and the rollback paths are exercised.
        bool accepted = false, rejected = false;
        for (uint seed = 0xBEEFu; seed < 0xBEEFu + 32 && !(accepted && rejected); seed++)
        {
            var (bytes, _) = BuildDrafter(seed, dflash2: true);
            string drafterPath = Path.Combine(dir, $"drafter-{seed}.gguf");
            File.WriteAllBytes(drafterPath, bytes);
            var perRound = RunDFlash(trunk, drafterPath, backend, prompt, NewTokens, snapshots, greedy);
            accepted |= perRound.Any(n => n > 1);
            rejected |= perRound.Any(n => n < Block);
            output.WriteLine($"seed {seed:X}: emitted/round [{string.Join(",", perRound)}]");
        }
        Assert.True(accepted, "no seed got a draft accepted: the accept path never ran");
        Assert.True(rejected, "no round rejected a draft: the rollback path never ran");
    }

    private static List<int> RunDFlash(string trunk, string drafterPath, BackendFactory backend, int[] prompt, int newTokens,
        bool snapshots, List<int> greedy)
    {
        using var gguf = GgufFile.Open(trunk);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (model, kvCache) = backend(gguf, config);
        using var _m = model as IDisposable;
        using var _k = kvCache;
        Assert.False(model.SupportsMtp);
        ((IDFlashDraftAttachable)model).AttachDFlashDrafter(drafterPath);
        Assert.True(model.SupportsMtp && model.SupportsMtpArgMax);
        Assert.Equal(Block - 1, model.MtpPreferredDraftTokens);

        using var mtp = model.CreateMtpState()!;
        Assert.IsType<DFlashMtpState>(mtp);
        var ids = new List<int>(prompt);
        using (var l = model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, kvCache, adapter: null, mtp))
            ids.Add(ArgMaxLast(l, config.VocabSize));
        Assert.Equal(prompt.Length, mtp.CurrentLength);

        var decoder = new MtpSpeculativeDecoder(greedy: true) { UseRecurrentRowSnapshots = snapshots };
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var outBuf = new int[Block];
        var perRound = new List<int>();
        while (ids.Count < prompt.Length + newTokens)
        {
            int position = ids.Count - 1;
            var r = decoder.DraftAndVerify(model, kvCache, mtp, pipeline, ids, null, position, config.VocabSize, Block - 1, outBuf);
            Assert.True(r.AcceptedCount >= 1);
            for (int i = 0; i < r.AcceptedCount; i++) ids.Add(outBuf[i]);
            perRound.Add(r.AcceptedCount);
            Assert.Equal(ids.Count - 1, mtp.CurrentLength);   // the drafter context tracks the committed tokens
        }
        Assert.Equal(greedy.Take(ids.Count), ids);
        return perRound;
    }

    internal static int ArgMaxLast(ITensor logits, int vocab)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, logits.Shape[0] * vocab).Slice((logits.Shape[0] - 1) * vocab, vocab);
        return System.Numerics.Tensors.TensorPrimitives.IndexOfMax(span);
    }
}
