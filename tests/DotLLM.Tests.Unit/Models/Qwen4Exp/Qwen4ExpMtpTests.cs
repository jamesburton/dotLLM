using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// qwen4exp MTP draft head on the CPU oracle (#820), on the tiny synthetic checkpoint: trunk file + a STANDALONE head file shaped like
/// the released <c>mtp-*.gguf</c> (<see cref="SyntheticQwen4ExpGguf.BuildMtpOnly"/>).
/// </summary>
/// <remarks>
/// There is no HF reference for the head (HF ignores <c>mtp.*</c>), so these tests pin the STRUCTURE: the pairing (cell for token p =
/// (R_{p-1}, token_p)), rollback, the absorb path agreeing with the draft path, and - the contract that matters - that speculative
/// decode emits exactly the plain greedy sequence. Draft QUALITY is measured on the real file (see the PR).
/// </remarks>
public sealed class Qwen4ExpMtpTests : IDisposable
{
    private const int Vocab = SyntheticQwen4ExpGguf.VocabSize;
    private readonly ITestOutputHelper _out;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-mtp-" + Guid.NewGuid().ToString("N"));
    private readonly string _trunkPath, _headPath;
    private readonly List<IDisposable> _disposables = [];

    public Qwen4ExpMtpTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_dir);
        _trunkPath = SyntheticQwen4ExpGguf.Write(Path.Combine(_dir, "trunk.gguf"));
        _headPath = SyntheticQwen4ExpGguf.WriteMtpOnly(Path.Combine(_dir, "mtp-syn-F32.gguf"));
    }

    public void Dispose()
    {
        foreach (var d in _disposables) d.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private (Qwen4ExpTransformerModel Model, ModelConfig Config) Load(bool attach = true)
    {
        var (m, g, c) = ModelLoader.LoadFromGguf(_trunkPath);
        _disposables.Add(g); _disposables.Add(m);
        var model = (Qwen4ExpTransformerModel)m;
        if (attach) model.AttachMtpHead(_headPath);
        return (model, c);
    }

    private static unsafe float[] Rows(ITensor t)
    {
        using (t) return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    private static int Argmax(ReadOnlySpan<float> row) { int b = 0; for (int i = 1; i < row.Length; i++) if (row[i] > row[b]) b = i; return b; }

    private static int[] Prompt(int n, int seed) { var r = new Random(seed); return Enumerable.Range(0, n).Select(_ => r.Next(4, Vocab)).ToArray(); }

    private static int[] Pos(int n, int start = 0) => Enumerable.Range(start, n).ToArray();

    private static IKvCache NewKv(ModelConfig c, int len = 96) => new SimpleKvCache(KvGeometry.FromConfig(c), len);

    private static float MaxRel(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float scale = 1e-12f, worst = 0;
        foreach (float x in a) scale = Math.Max(scale, Math.Abs(x));
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst / scale;
    }

    [Fact]
    public void AttachMtpHead_SeparateFile_EnablesMtp_AndCreatesState()
    {
        var (model, _) = Load(attach: false);
        Assert.False(model.SupportsMtp);
        Assert.Null(model.CreateMtpState(64));
        model.AttachMtpHead(_headPath);
        Assert.True(model.SupportsMtp);
        using var st = model.CreateMtpState(64)!;
        Assert.Equal(0, st.CurrentLength);
        Assert.Equal(SyntheticQwen4ExpGguf.HcCount * SyntheticQwen4ExpGguf.HiddenSize, st.HiddenSize);   // the head reads the 4-stream residual
        Assert.Throws<InvalidOperationException>(() => model.AttachMtpHead(_headPath));
    }

    [Fact]
    public void AttachMtpHead_FileWithoutTheHeadBlock_IsRejected()
    {
        var (model, _) = Load(attach: false);
        Assert.Throws<InvalidDataException>(() => model.AttachMtpHead(_trunkPath));   // the trunk has no blk.4.*
    }

    [Fact]
    public void Forward_WithMtpState_ReturnsTheSameLogits_AndAbsorbsTheBatch()
    {
        var (model, cfg) = Load();
        int[] ids = Prompt(9, 1);

        model.ResetSequenceState();
        using var kv0 = NewKv(cfg);
        float[] plain = Rows(model.Forward(ids, Pos(9), -1, kv0));

        model.ResetSequenceState();
        using var kv1 = NewKv(cfg);
        using var st = model.CreateMtpState(64)!;
        float[] withMtp = Rows(model.Forward(ids, Pos(9), -1, kv1, null, st));

        Assert.Equal(plain, withMtp);   // capture / absorb are side effects on the MTP state only
        Assert.Equal(9, st.CurrentLength);
        Assert.Equal(9, st.CapturedRowCount);
        Assert.Equal(8, ((Qwen4ExpMtpState)st).CellCount);   // token 0 has no preceding residual and owns no cell
    }

    [Fact]
    public void ForwardWithLastTokenHint_StillAbsorbsEveryPosition()
    {
        var (model, cfg) = Load();
        int[] ids = Prompt(7, 2);
        model.ResetSequenceState();
        using var kv = NewKv(cfg);
        using var st = model.CreateMtpState(64)!;
        var lg = model.Forward(ids, Pos(7), -1, kv, null, st, lastTokenLogitsOnly: true);
        Assert.Equal(1, lg.Shape[0]);
        lg.Dispose();
        Assert.Equal(7, st.CapturedRowCount);   // the hint trims logits, never the MTP capture
        Assert.Equal(7, st.CurrentLength);
    }

    [Fact]
    public void DraftStep_WritesTheSameCellAsAbsorbingThatToken()
    {
        // The pairing contract: the cell for token p is (R_{p-1}, token_p), whether the head reaches it by absorbing a trunk batch
        // (pairing row p-1 of the capture) or by a draft step (pending = the carried row, seeded after the previous batch).
        var (model, cfg) = Load();
        int[] ids = Prompt(8, 3);

        model.ResetSequenceState();
        using var kvA = NewKv(cfg);
        using var full = (Qwen4ExpMtpState)model.CreateMtpState(64)!;
        model.Forward(ids, Pos(8), -1, kvA, null, full).Dispose();     // cells for tokens 1..7

        model.ResetSequenceState();
        using var kvB = NewKv(cfg);
        using var part = (Qwen4ExpMtpState)model.CreateMtpState(64)!;
        model.Forward(ids[..7], Pos(7), -1, kvB, null, part).Dispose();  // cells for tokens 1..6, carry = R_6
        Assert.Equal(6, part.CellCount);
        model.ForwardMtp(part, ids[7], 7).Dispose();                    // draft step pairs (R_6, token_7): writes cell 6
        Assert.Equal(7, part.CellCount);

        Assert.True(MaxRel(full.Kv.Keys, part.Kv.Keys) < 1e-4f, "draft-step cell must equal the absorbed cell (keys)");
        Assert.True(MaxRel(full.Kv.Values, part.Kv.Values) < 1e-4f, "draft-step cell must equal the absorbed cell (values)");
    }

    [Fact]
    public void DraftLogits_DependOnThePairedResidual_NotOnAnyRow()
    {
        // Sensitivity control: seeding from a different captured row must change the draft (otherwise the pairing tests above are blind).
        var (model, cfg) = Load();
        int[] ids = Prompt(9, 4);
        model.ResetSequenceState();
        using var kv = NewKv(cfg);
        using var st = (Qwen4ExpMtpState)model.CreateMtpState(64)!;
        model.Forward(ids, Pos(9), -1, kv, null, st).Dispose();

        st.SeedFromCapturedRow(8);
        float[] atLast = Rows(model.ForwardMtp(st, 5, 9));
        st.Rollback(9);
        st.SeedFromCapturedRow(7);
        float[] atPrev = Rows(model.ForwardMtp(st, 5, 9));
        Assert.True(MaxRel(atLast, atPrev) > 1e-3f, "the draft must depend on which trunk residual it is seeded with");
    }

    [Fact]
    public void Rollback_DiscardsSpeculativeCells_AndRedoingThemIsDeterministic()
    {
        var (model, cfg) = Load();
        int[] ids = Prompt(8, 5);
        model.ResetSequenceState();
        using var kv = NewKv(cfg);
        using var st = (Qwen4ExpMtpState)model.CreateMtpState(64)!;
        model.Forward(ids, Pos(8), -1, kv, null, st).Dispose();
        st.SeedFromCapturedRow(7);

        float[] first = Rows(model.ForwardMtp(st, 6, 8));
        model.ForwardMtp(st, 7, 9).Dispose();
        model.ForwardMtp(st, 8, 10).Dispose();
        Assert.Equal(11, st.CurrentLength);
        st.Rollback(8);
        Assert.Equal(8, st.CurrentLength);
        Assert.Equal(7, st.CellCount);
        st.SeedFromCapturedRow(7);
        float[] again = Rows(model.ForwardMtp(st, 6, 8));
        Assert.Equal(first, again);
        Assert.Throws<InvalidOperationException>(() => model.ForwardMtp(st, 6, 12));   // a gap: the head holds no cell 11
    }

    [Fact]
    public void HnormConvention_ChangesDrafts()
    {
        var (model, cfg) = Load();
        int[] ids = Prompt(8, 6);
        model.ResetSequenceState();
        using var kv = NewKv(cfg);
        using var st = (Qwen4ExpMtpState)model.CreateMtpState(64)!;
        model.Forward(ids, Pos(8), -1, kv, null, st).Dispose();
        st.SeedFromCapturedRow(7);
        model.MtpHnormPerStream = false;
        float[] joint = Rows(model.ForwardMtp(st, 5, 8));
        st.Rollback(8); st.SeedFromCapturedRow(7);
        model.MtpHnormPerStream = true;
        float[] stream = Rows(model.ForwardMtp(st, 5, 8));
        Assert.True(MaxRel(joint, stream) > 1e-4f);
    }

    private List<int> PlainGreedy(Qwen4ExpTransformerModel model, ModelConfig cfg, int[] prompt, int n)
    {
        model.ResetSequenceState();
        using var kv = NewKv(cfg);
        var gen = new List<int>();
        gen.Add(Argmax(Rows(model.Forward(prompt, Pos(prompt.Length), -1, kv)).AsSpan((prompt.Length - 1) * Vocab, Vocab)));
        while (gen.Count < n)
            gen.Add(Argmax(Rows(model.Forward([gen[^1]], [prompt.Length + gen.Count - 1], -1, kv))));
        return gen;
    }

    [Theory]
    [InlineData(1, 11, 3)]
    [InlineData(2, 5, 4)]
    [InlineData(3, 17, 2)]
    [InlineData(4, 8, 4)]
    public void SpeculativeDecode_EmitsExactlyThePlainGreedySequence(int seed, int promptLen, int k)
    {
        var (model, cfg) = Load();
        int[] prompt = Prompt(promptLen, seed);
        const int N = 30;
        List<int> expected = PlainGreedy(model, cfg, prompt, N);

        model.ResetSequenceState();
        using var kv = NewKv(cfg);
        using var st = model.CreateMtpState(kv.MaxLength)!;
        var dec = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var gen = new List<int>();
        float[] first = Rows(model.Forward(prompt, Pos(prompt.Length), -1, kv, null, st, true));
        gen.Add(Argmax(first.AsSpan(first.Length - Vocab, Vocab)));
        var buf = new int[k + 1];
        var perRound = new List<int>();
        while (gen.Count < N)
        {
            int pos = prompt.Length + gen.Count - 1;
            var r = dec.DraftAndVerify(model, kv, st, pipeline, gen, null, pos, Vocab, Math.Min(k, N - gen.Count), buf);
            Assert.True(r.AcceptedCount >= 1);
            perRound.Add(r.AcceptedCount);
            for (int i = 0; i < r.AcceptedCount && gen.Count < N; i++) gen.Add(buf[i]);
        }
        _out.WriteLine($"seed {seed} k {k}: rounds={perRound.Count} tokens/round={(double)(N - 1) / perRound.Count:F2} replaysAvoided={dec.ReplaysAvoided} replays={dec.Replays} perRound=[{string.Join(",", perRound)}]");
        Assert.Equal(expected, gen);
        Assert.True(dec.ReplaysAvoided + dec.Replays > 0 || perRound.All(x => x == k + 1),
            "a rejection must have been rolled back through the recurrent snapshots (or every draft was accepted)");
    }

    [Fact]
    public void Resolver_FindsSiblingHead_InTheModelDirectoryAndTheOneAbove()
    {
        string root = Path.Combine(_dir, "snap");
        string quant = Path.Combine(root, "UD-Q4_K_XL");
        Directory.CreateDirectory(quant);
        string shard = Path.Combine(quant, "m-00001-of-00004.gguf");
        File.WriteAllBytes(shard, [0]);
        Assert.Null(Qwen4ExpMtpHeadResolver.Find(shard));

        string small = Path.Combine(root, "mtp-m-Q4_K_M.gguf"), big = Path.Combine(root, "mtp-m-Q8_0.gguf");
        File.WriteAllBytes(small, new byte[10]);
        File.WriteAllBytes(big, new byte[100]);
        Assert.Equal(big, Qwen4ExpMtpHeadResolver.Find(shard));   // above the quantisation folder; the largest wins

        string near = Path.Combine(quant, "mtp-near.gguf");
        File.WriteAllBytes(near, new byte[5]);
        Assert.Equal(near, Qwen4ExpMtpHeadResolver.Find(shard));  // the model's own directory takes precedence
    }

    [Fact]
    public void Resolver_TryAttach_UsesTheExplicitPath_AndIgnoresOtherModels()
    {
        var (model, _) = Load(attach: false);
        string? attached = Qwen4ExpMtpHeadResolver.TryAttach(model, _trunkPath, _headPath);
        Assert.Equal(_headPath, attached);
        Assert.True(model.SupportsMtp);
        Assert.Null(Qwen4ExpMtpHeadResolver.TryAttach(model, _trunkPath, _headPath));   // already attached: nothing to do
        Assert.Throws<FileNotFoundException>(() =>
            Qwen4ExpMtpHeadResolver.TryAttach(Load(attach: false).Model, _trunkPath, Path.Combine(_dir, "nope.gguf")));
    }
}
