using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1185: a <c>qwen35</c> MTP head shipped as a separate <c>mtp-*.gguf</c> (ggml-org's Qwen3.8-27B layout) must behave exactly like the
/// same head embedded in the trunk. The fixtures split one seeded checkpoint into a trunk-only file and a head-only file, so any difference
/// in draft logits or accepted tokens is the attach path's fault.
/// </summary>
public sealed class Qwen3HybridDenseMtpHeadAttachTests : IDisposable
{
    private const int NewTokens = 12;
    private const int DraftTokens = 3;
    private const int Context = 32;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35-mtp-attach-{Guid.NewGuid():N}");

    public Qwen3HybridDenseMtpHeadAttachTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // The real head file carries no nextn.embed_tokens / shared_head_head: the trunk's token_embd / output are used.
    private string Embedded() => SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "embedded.gguf"), withMtp: true,
        mtpHasOwnHeadTensors: false, contextLength: Context);

    private static string Trunk(string dir)
        => SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(dir, "trunk.gguf"), withMtp: false, contextLength: Context);

    private static string Head(string dir, string name = "mtp-trunk.gguf", int blockCount = SyntheticQwen35HybridDenseMtpGguf.BlockCount)
        => SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(dir, name), withMtp: true, mtpHasOwnHeadTensors: false,
            contextLength: Context, blockCount: blockCount, headOnly: true);

    /// <summary>A loaded model plus the GGUF it maps; GDN state is model-owned, so every run gets a fresh one.</summary>
    private sealed class Loaded : IDisposable
    {
        private readonly GgufFile _gguf;

        public Loaded(string path, string? headPath)
        {
            _gguf = GgufFile.Open(path);
            Model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(_gguf, GgufModelConfigExtractor.Extract(_gguf.Metadata));
            if (headPath is not null)
                Model.AttachMtpHead(headPath);
        }

        public Qwen3HybridDenseTransformerModel Model { get; }

        public void Dispose()
        {
            Model.Dispose();
            _gguf.Dispose();
        }
    }

    [Fact]
    public void SeparateHead_DraftsAndDecodesExactlyLikeTheEmbeddedHead()
    {
        string embedded = Embedded();
        string trunk = Trunk(_scratch);
        string head = Head(_scratch);

        using (var bare = new Loaded(trunk, headPath: null))
            Assert.False(bare.Model.SupportsMtp);

        float[] embLogits, splitLogits;
        using (var m = new Loaded(embedded, null)) embLogits = DraftLogits(m.Model);
        using (var m = new Loaded(trunk, head))
        {
            Assert.True(m.Model.SupportsMtp);
            Assert.True(m.Model.HasMtpHead);
            splitLogits = DraftLogits(m.Model);
        }

        // Draft logits, bit for bit, over a prefill plus three autoregressive head steps.
        Assert.Equal(embLogits, splitLogits);

        // Greedy speculation: the same tokens and the same per-round acceptance (so the same drafts), equal to plain greedy.
        (List<int> Tokens, List<int> Accepts) emb, split;
        using (var m = new Loaded(embedded, null)) emb = Speculate(m.Model);
        using (var m = new Loaded(trunk, head)) split = Speculate(m.Model);
        List<int> plain;
        using (var m = new Loaded(trunk, null)) plain = PlainGreedy(m.Model);

        Assert.Equal(emb.Tokens, split.Tokens);
        Assert.Equal(emb.Accepts, split.Accepts);
        Assert.Equal(plain, split.Tokens);
    }

    [Fact]
    public void Resolver_AutoAttachesSiblingHead_AndLeavesEmbeddedHeadsAlone()
    {
        string dir = Directory.CreateDirectory(Path.Combine(_scratch, "snapshot")).FullName;
        string trunk = Trunk(dir);
        string head = Head(dir);

        using (var m = new Loaded(trunk, null))
        {
            Assert.Equal(head, Qwen4ExpMtpHeadResolver.TryAttach(m.Model, trunk));
            Assert.True(m.Model.SupportsMtp);
        }

        string embedded = Embedded();
        using (var m = new Loaded(embedded, null))
            Assert.Null(Qwen4ExpMtpHeadResolver.TryAttach(m.Model, embedded, explicitPath: head));
    }

    [Fact]
    public void HeadFromADifferentTrunkDepth_IsRejected()
    {
        string trunk = Trunk(_scratch);
        string wrong = Head(_scratch, "mtp-deeper.gguf", blockCount: SyntheticQwen35HybridDenseMtpGguf.BlockCount + 2);

        using var m = new Loaded(trunk, null);
        var ex = Assert.Throws<InvalidDataException>(() => m.Model.AttachMtpHead(wrong));
        Assert.Contains("trunk layers", ex.Message);
        Assert.False(m.Model.SupportsMtp);
    }

    private static float[] DraftLogits(Qwen3HybridDenseTransformerModel model)
    {
        var cfg = model.Config;
        using var kv = new SimpleKvCache(model.AttentionLayerCount, cfg.NumKvHeads, cfg.HeadDim, cfg.MaxSequenceLength);
        using var mtp = model.CreateMtpState()!;
        using (model.Forward([1, 5, 3, 7], [0, 1, 2, 3], deviceId: -1, kv, adapter: null, mtp)) { }
        mtp.SeedFromCapturedRow(3);
        var all = new List<float>();
        int token = 4;
        for (int step = 0; step < 3; step++)
        {
            using ITensor t = model.ForwardMtp(mtp, token, 4 + step);
            float[] row = Row(t, cfg.VocabSize);
            all.AddRange(row);
            token = ArgMax(row);
        }
        return all.ToArray();
    }

    private static (List<int> Tokens, List<int> Accepts) Speculate(Qwen3HybridDenseTransformerModel model)
    {
        var cfg = model.Config;
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new DotLLM.Core.Configuration.InferenceOptions { Temperature = 0f });
        using var kv = new SimpleKvCache(model.AttentionLayerCount, cfg.NumKvHeads, cfg.HeadDim, cfg.MaxSequenceLength);
        using var mtp = model.CreateMtpState()!;
        var tokens = new List<int> { 1 };

        // As TextGenerator does: prefill the prompt WITH the head state, take t1 from it, enter the rounds at t1's slot.
        using (ITensor prefill = model.Forward([1], [0], deviceId: -1, kv, adapter: null, mtp))
            tokens.Add(ArgMax(Row(prefill, cfg.VocabSize)));
        var accepts = new List<int>();
        var buffer = new int[DraftTokens + 1];
        int position = 1;
        while (tokens.Count - 1 < NewTokens)
        {
            var r = decoder.DraftAndVerify(model, kv, mtp, pipeline, tokens, constraint: null, position,
                vocabSize: cfg.VocabSize, numCandidates: DraftTokens, buffer);
            Assert.True(r.AcceptedCount > 0);
            accepts.Add(r.AcceptedCount);
            for (int i = 0; i < r.AcceptedCount; i++) tokens.Add(buffer[i]);
            position += r.AcceptedCount;
        }
        return (tokens.Take(NewTokens + 1).ToList(), accepts);
    }

    private static List<int> PlainGreedy(Qwen3HybridDenseTransformerModel model)
    {
        var cfg = model.Config;
        using var kv = new SimpleKvCache(model.AttentionLayerCount, cfg.NumKvHeads, cfg.HeadDim, cfg.MaxSequenceLength);
        var tokens = new List<int> { 1 };
        for (int pos = 0; pos < NewTokens; pos++)
        {
            using ITensor t = model.Forward([tokens[^1]], [pos], deviceId: -1, kv);
            tokens.Add(ArgMax(Row(t, cfg.VocabSize)));
        }
        return tokens;
    }

    private static unsafe float[] Row(ITensor t, int vocab) => new ReadOnlySpan<float>((void*)t.DataPointer, vocab).ToArray();

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }
}
