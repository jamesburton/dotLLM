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
/// Issue #1188: the <c>qwen35moe</c> MTP head on the CPU oracle. The head must leave the trunk's logits untouched, draft through the MoE FFN,
/// and greedy speculation (which rolls the GDN recurrence back through the checkpoint pair) must equal plain greedy decode.
/// </summary>
public sealed class Qwen3MoeHybridMtpTests : IDisposable
{
    private const int Context = 32;
    private const int NewTokens = 14;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35moe-mtp-{Guid.NewGuid():N}");

    public Qwen3MoeHybridMtpTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Fixture(bool withMtp)
        => SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, $"moe-mtp{withMtp}.gguf"), withMtp: withMtp, contextLength: Context);

    private sealed class Loaded : IDisposable
    {
        private readonly GgufFile _gguf;

        public Loaded(string path)
        {
            _gguf = GgufFile.Open(path);
            Model = (Qwen3MoeHybridTransformerModel)ModelLoader.CreateCpuModelFromGguf(_gguf, GgufModelConfigExtractor.Extract(_gguf.Metadata));
        }

        public Qwen3MoeHybridTransformerModel Model { get; }

        public SimpleKvCache Kv() => new(Model.AttentionLayerCount, Model.Config.NumKvHeads, Model.Config.HeadDim, Model.Config.MaxSequenceLength);

        public void Dispose()
        {
            Model.Dispose();
            _gguf.Dispose();
        }
    }

    [Fact]
    public void HeadIsDetected_AndTrunkLogitsAreUnchangedByCaptureAndAbsorb()
    {
        using var plain = new Loaded(Fixture(withMtp: false));
        using var mtp = new Loaded(Fixture(withMtp: true));
        Assert.False(plain.Model.SupportsMtp);
        Assert.True(mtp.Model.SupportsMtp);
        Assert.Equal(plain.Model.Config.NumLayers, mtp.Model.Config.NumLayers);

        int[] tokens = [1, 5, 3, 7, 2];
        int[] positions = [0, 1, 2, 3, 4];
        using var kvA = plain.Kv();
        using var kvB = mtp.Kv();
        using var state = mtp.Model.CreateMtpState()!;
        using ITensor a = plain.Model.Forward(tokens, positions, -1, kvA);
        using ITensor b = mtp.Model.Forward(tokens, positions, -1, kvB, adapter: null, state);
        Assert.Equal(Row(a, 0, tokens.Length * plain.Model.Config.VocabSize), Row(b, 0, tokens.Length * plain.Model.Config.VocabSize));
        Assert.Equal(tokens.Length, state.CurrentLength);
        Assert.Equal(tokens.Length, state.CapturedRowCount);
    }

    [Fact]
    public void AbsorbedKv_EqualsTheKvADraftStepWrites()
    {
        // Batched absorb computes K/V only; a draft step at the same position computes the whole block. Their K/V rows must agree (to GEMM
        // reduction order: the absorb multiplies 4 columns at once, and to one F16 ulp because the head stores F16, #1141).
        using var m = new Loaded(Fixture(withMtp: true));
        int vocab = m.Model.Config.VocabSize;
        using var kv = m.Kv();
        using var state = m.Model.CreateMtpState()!;
        using (m.Model.Forward([1, 5, 3, 7], [0, 1, 2, 3], -1, kv, adapter: null, state)) { }

        float[] absorbedK = KeyRow((CpuMtpState)state, 3);

        // Re-run position 3 as a draft step paired with h_2: rebuild the pairing exactly as the absorb did.
        m.Model.ResetSequenceState();   // the GDN recurrence is model-owned
        using var kv2 = m.Kv();
        using var state2 = m.Model.CreateMtpState()!;
        using (m.Model.Forward([1, 5, 3], [0, 1, 2], -1, kv2, adapter: null, state2)) { }
        using ITensor draft = m.Model.ForwardMtp(state2, 7, 3);
        float[] draftK = KeyRow((CpuMtpState)state2, 3);
        for (int i = 0; i < absorbedK.Length; i++)
            Assert.True(MathF.Abs(absorbedK[i] - draftK[i]) <= 1e-5f + 1.2e-3f * MathF.Abs(absorbedK[i]), $"K[{i}] absorb {absorbedK[i]} vs draft {draftK[i]}");
        Assert.All(Row(draft, 0, vocab), x => Assert.True(float.IsFinite(x)));
    }

    [Fact]
    public void GreedySpeculation_EqualsPlainGreedy()
    {
        string path = Fixture(withMtp: true);
        List<int> plain;
        using (var m = new Loaded(path)) plain = PlainGreedy(m);
        foreach (int k in new[] { 1, 3, 5 })
        {
            using var m = new Loaded(path);
            Assert.Equal(plain, Speculate(m, k));
        }
    }

    [Fact]
    public void CheckpointRestore_RollsTheGdnRecurrenceBack()
    {
        using var m = new Loaded(Fixture(withMtp: false));
        int vocab = m.Model.Config.VocabSize;
        using var kv = m.Kv();
        using (m.Model.Forward([1, 5], [0, 1], -1, kv)) { }
        object? cp = m.Model.CheckpointRecurrentState();
        float[] first;
        using (ITensor t = m.Model.Forward([3], [2], -1, kv)) first = Row(t, 0, vocab);
        kv.Rollback(2);
        using (ITensor t = m.Model.Forward([3], [2], -1, kv))
            Assert.NotEqual(first, Row(t, 0, vocab));   // the state advanced: the forward really depends on it
        kv.Rollback(2);
        m.Model.RestoreRecurrentState(cp);
        ((IDisposable)cp!).Dispose();
        using (ITensor t = m.Model.Forward([3], [2], -1, kv))
            Assert.Equal(first, Row(t, 0, vocab));
    }

    private static List<int> PlainGreedy(Loaded m)
    {
        int vocab = m.Model.Config.VocabSize;
        using var kv = m.Kv();
        var ids = new List<int> { 1 };
        for (int pos = 0; pos < NewTokens; pos++)
        {
            using ITensor t = m.Model.Forward([ids[^1]], [pos], -1, kv);
            ids.Add(ArgMax(Row(t, 0, vocab)));
        }
        return ids;
    }

    private static List<int> Speculate(Loaded m, int k)
    {
        int vocab = m.Model.Config.VocabSize;
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new DotLLM.Core.Configuration.InferenceOptions { Temperature = 0f });
        using var kv = m.Kv();
        using var state = m.Model.CreateMtpState()!;
        var ids = new List<int> { 1 };
        using (ITensor prefill = m.Model.Forward([1], [0], -1, kv, adapter: null, state))
            ids.Add(ArgMax(Row(prefill, 0, vocab)));
        var buffer = new int[k + 1];
        int position = 1, drafted = 0;
        while (ids.Count - 1 < NewTokens)
        {
            var r = decoder.DraftAndVerify(m.Model, kv, state, pipeline, ids, constraint: null, position, vocabSize: vocab, numCandidates: k, buffer);
            Assert.True(r.AcceptedCount > 0);
            drafted += r.DraftedCount;
            for (int i = 0; i < r.AcceptedCount; i++) ids.Add(buffer[i]);
            position += r.AcceptedCount;
        }
        Assert.True(drafted > 0, "the decoder must actually have drafted");
        return ids.Take(NewTokens + 1).ToList();
    }

    private static unsafe float[] KeyRow(CpuMtpState s, int row)
    {
        var bytes = s.GetKeyRowBytes(row);
        return s.KvFormat switch
        {
            CpuMtpKvFormat.F32 => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes).ToArray(),
            CpuMtpKvFormat.F16 => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(bytes).ToArray().Select(h => (float)h).ToArray(),
            _ => throw new NotSupportedException(),
        };
    }

    private static unsafe float[] Row(ITensor t, int start, int count) => new ReadOnlySpan<float>((float*)t.DataPointer + start, count).ToArray();

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }
}
