using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// #844: several PLE modules and image-token PLE stand-ins, against HF <c>Qwen4ExpTextModel</c> (transformers 5.19, eager) on a tiny
/// random checkpoint with PLE modules on layers 1 and 2 (<c>Reference/gen_model_ple2.py</c>). Each module has its own table, projections,
/// norms and conv, and its own hash constants (HF derives them from the module's position in <c>ple_layer_ids</c>).
/// </summary>
public sealed unsafe class Qwen4ExpMultiPleAndImageTests : IDisposable
{
    private static readonly Qwen4ExpReferenceFixture Fx = Qwen4ExpReferenceFixture.Load("tiny_model_ple2.json");
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-ple2-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _d = [];

    public Qwen4ExpMultiPleAndImageTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        foreach (var x in _d) x.Dispose();
        try { Directory.Delete(_dir, true); } catch (IOException) { }
    }

    private (Qwen4ExpTransformerModel Model, ModelConfig Config) Load(bool omitImageKey = false)
    {
        string path = Path.Combine(_dir, omitImageKey ? "noimg.gguf" : "tiny.gguf");
        if (!File.Exists(path)) File.WriteAllBytes(path, Qwen4ExpTinyGguf.Build(Fx, omitImageTokenId: omitImageKey));
        var (m, g, c) = ModelLoader.LoadFromGguf(path);
        _d.Add(g); _d.Add(m);
        return ((Qwen4ExpTransformerModel)m, c);
    }

    private static float[] ToArray(ITensor t) { using (t) return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray(); }
    private static int[] Positions(int n, int start = 0) => Enumerable.Range(start, n).ToArray();
    private static int[] Ids(string name) => Fx.I64(name).Select(v => (int)v).ToArray();

    private static float MaxRel(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]) / (1f + MathF.Abs(expected[i])));
        return worst;
    }

    [Fact]
    public void Config_ReadsTwoModulesWithTheirOwnHashConstants()
    {
        var (_, cfg) = Load();
        var ple = cfg.Qwen4Exp!.Ple!;
        Assert.Equal([1, 2], ple.Layers);
        Assert.Equal(7, ple.ImageTokenId);
        int ng = ple.NgramSize, nh = ple.NumHeads;
        Assert.Equal(2 * ng, ple.LayerMultipliers.Count);
        Assert.Equal(2 * nh, ple.HeadOffsets.Count);
        // HF derives different multipliers per module, so the two sets must differ (else the test could not tell a shared set from two).
        Assert.NotEqual(ple.MultipliersOf(0), ple.MultipliersOf(1));
        Assert.NotEqual(ple.HeadVocabSizesOf(0), ple.HeadVocabSizesOf(1));
        // Module 1's ranges sit after module 0's in the one concatenated table.
        Assert.True(ple.HeadOffsetsOf(1).Min() >= ple.HeadOffsetsOf(0).Zip(ple.HeadVocabSizesOf(0), (o, v) => o + v).Max());
        Assert.True(ple.MinTableRows <= (ulong)Fx.Shape("per_layer_token_embd.weight")[0]);
    }

    [Fact]
    public void TwoPleModules_TextOnly_MatchHf_PerLayerAndLogits()
    {
        var (model, _) = Load();
        var trace = new Dictionary<string, float[]>();
        model.Trace = (name, data, rows, cols) => trace[name] = data.ToArray();
        int T = Fx.Int("seq_len");

        var logits = ToArray(model.Forward(Ids("ids"), Positions(T), -1));

        for (int il = 0; il < Fx.Int("num_layers"); il++)
        {
            float rel = MaxRel(Fx.F32($"l_out.{il}"), trace[$"blk.{il}.l_out"]);
            Assert.True(rel < 2e-4f, $"layer {il} residual: worst rel diff {rel:E3}");
        }
        float lrel = MaxRel(Fx.F32("logits"), logits);
        Assert.True(lrel < 5e-4f, $"logits: worst rel diff {lrel:E3}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(16)]
    public void TwoPleModules_ChunkedAndDecode_EqualSingleShot(int chunk)
    {
        var (model, _) = Load();
        int T = Fx.Int("seq_len"), V = Fx.Int("vocab");
        var ids = Ids("ids");
        var whole = ToArray(model.Forward(ids, Positions(T), -1));
        using var state = model.CreateState();
        var chunked = new float[T * V];
        for (int i = 0; i < T; i += chunk)
        {
            int n = Math.Min(chunk, T - i);
            ToArray(model.Forward(ids.AsSpan(i, n), Positions(n, i), -1, state)).CopyTo(chunked, i * V);
        }
        Assert.True(MaxRel(whole, chunked) < 1e-4f);
    }

    [Fact]
    public void TwoPleModules_CheckpointRowSnapshotsAndAccounting_CoverBothStates()
    {
        var (model, cfg) = Load();
        int T = 24;
        var ids = Ids("ids").AsSpan(0, T + 8).ToArray();
        model.ResetSequenceState();
        model.Forward(ids.AsSpan(0, T), Positions(T), -1).Dispose();
        var cp = model.CheckpointRecurrentState();
        var want = ToArray(model.Forward(ids.AsSpan(T, 8), Positions(8, T), -1));
        model.RestoreRecurrentState(cp);
        (cp as IDisposable)?.Dispose();
        Assert.Equal(want, ToArray(model.Forward(ids.AsSpan(T, 8), Positions(8, T), -1)));   // both modules' windows + conv histories restored

        // Accounting: two PLE states, and the estimate equals the allocation.
        var bytes = model.DefaultStateBytes;
        var est = model.EstimateSequenceStateBytes(T + 8);
        Assert.Equal(est.Ple, bytes.Ple);
        long perModule = (long)(cfg.Qwen4Exp!.Ple!.NgramSize - 1) * sizeof(int)
                         + (long)(cfg.Qwen4Exp.Ple.ConvKernel - 1) * cfg.Qwen4Exp.Ple.NgramSize * cfg.Qwen4Exp.HyperConnectionCount * cfg.HiddenSize * sizeof(float);
        Assert.Equal(2 * perModule, est.Ple);
    }

    [Fact]
    public void ImageRows_ExternalEmbeddingsAndTheStandInId_MatchHf()
    {
        var (model, cfg) = Load();
        int T = Fx.Int("seq_len"), V = Fx.Int("vocab"), H = cfg.HiddenSize;
        var idsImg = Ids("ids_img");
        var pos = Fx.I64("img_positions").Select(v => (int)v).ToArray();
        var tokens = (int[])idsImg.Clone();
        foreach (int p in pos) tokens[p] = Qwen4ExpTransformerModel.ExternalEmbeddingToken;
        var embeds = Fx.F32("img_embeds");
        Assert.Equal(pos.Length * H, embeds.Length);

        using var state = model.CreateState();
        var logits = ToArray(model.Forward(tokens, Positions(T), -1, state, null, lastTokenLogitsOnly: false, embeds));
        float rel = MaxRel(Fx.F32("logits_img"), logits);
        Assert.True(rel < 5e-4f, $"image-row logits: worst rel diff {rel:E3}");

        // Controls (each would pass the bound above if the mechanism were a no-op):
        // (1) the placeholder id as an ordinary token, no external embeddings -> wrong embeddings
        var plain = ToArray(model.Forward(idsImg, Positions(T), -1));
        Assert.True(MaxRel(Fx.F32("logits_img"), plain) > 1e-2f, "image embeddings did not matter");

        // (2) a file without ple.image_token_id falls back to the EOS stand-in -> the PLE hash sees different ids at the image rows
        var (noKey, cfg2) = Load(omitImageKey: true);
        Assert.Null(cfg2.Qwen4Exp!.Ple!.ImageTokenId);
        using var state2 = noKey.CreateState();
        var eosFallback = ToArray(noKey.Forward(tokens, Positions(T), -1, state2, null, lastTokenLogitsOnly: false, embeds));
        Assert.True(MaxRel(Fx.F32("logits_img"), eosFallback) > 1e-3f, "the PLE stand-in id did not matter");
    }

    [Fact]
    public void ImageRows_Chunked_EqualsSingleShot_AndTheTextPathIsUntouched()
    {
        var (model, cfg) = Load();
        int T = Fx.Int("seq_len"), V = Fx.Int("vocab"), H = cfg.HiddenSize;
        var tokens = Ids("ids_img");
        var pos = Fx.I64("img_positions").Select(v => (int)v).ToArray();
        foreach (int p in pos) tokens[p] = Qwen4ExpTransformerModel.ExternalEmbeddingToken;
        var embeds = Fx.F32("img_embeds");

        using var whole = model.CreateState();
        var one = ToArray(model.Forward(tokens, Positions(T), -1, whole, null, false, embeds));
        using var chunked = model.CreateState();
        var parts = new float[T * V];
        int used = 0;
        for (int i = 0; i < T; i += 5)
        {
            int n = Math.Min(5, T - i);
            int imgs = tokens.AsSpan(i, n).ToArray().Count(t => t == Qwen4ExpTransformerModel.ExternalEmbeddingToken);
            ToArray(model.Forward(tokens.AsSpan(i, n), Positions(n, i), -1, chunked, null, false, embeds.AsSpan(used * H, imgs * H)))
                .CopyTo(parts, i * V);
            used += imgs;
        }
        Assert.True(MaxRel(one, parts) < 1e-4f);

        // text-only through the new overload (no sentinel, empty embeddings) is bit-identical to the original entry point
        using var a = model.CreateState(); using var b = model.CreateState();
        var ids = Ids("ids");
        Assert.Equal(ToArray(model.Forward(ids, Positions(T), -1, a, null, false)),
                     ToArray(model.Forward(ids, Positions(T), -1, b, null, false, default(ReadOnlySpan<float>))));
    }

    [Fact]
    public void ExternalEmbeddings_CountMismatch_IsRejected()
    {
        var (model, cfg) = Load();
        using var s = model.CreateState();
        int sentinel = Qwen4ExpTransformerModel.ExternalEmbeddingToken;
        Assert.Throws<ArgumentException>(() => model.Forward([8, sentinel], [0, 1], -1, s, null, false, new float[cfg.HiddenSize - 1]));
        Assert.Throws<ArgumentException>(() => model.Forward([8, 9], [0, 1], -1, s, null, false, new float[cfg.HiddenSize]));
    }
}
