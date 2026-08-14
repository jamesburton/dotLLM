using DotLLM.Core.Configuration;
using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Evaluation;

// THROWAWAY verification harness for issue #395 — delete after reporting.
public sealed class TempWindowParityScratchTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _scratch = CreateScratch();

    private static string CreateScratch()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"dotllm-395-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void HybridSplitWindows_MatchWholeModel()
    {
        string path = SyntheticQwen35MoeGguf.Write(
            Path.Combine(_scratch, "qwen35moe-4layer.gguf"), blockCount: 4);

        using GgufFile gguf = GgufFile.Open(path);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using IModel model = DotLLM.Models.ModelLoader.CreateCpuModelFromGguf(gguf, config);

        Assert.Equal(4, config.NumLayers);
        Assert.True(model.RequiresPerSequenceState);

        int[] tokens = [1, 2, 3, 4, 5, 6];
        int[] positions = [0, 1, 2, 3, 4, 5];
        int seqLen = tokens.Length;
        int hidden = config.HiddenSize;
        int vocab = config.VocabSize;

        model.ResetSequenceState();
        using ITensor reference = model.Forward(tokens, positions, -1);
        var refLogits = new float[seqLen * vocab];
        unsafe { new ReadOnlySpan<float>((void*)reference.DataPointer, refLogits.Length).CopyTo(refLogits); }

        using var wm = new CpuLayerWindowModel(model, config);
        var h1 = new float[seqLen * hidden];
        var h2 = new float[seqLen * hidden];
        using (ILayerWindowExecutor a = wm.CreateWindow(0, 2))
        {
            a.ResetState();
            a.Run(tokens, ReadOnlySpan<float>.Empty, positions, h1);
        }

        using (ILayerWindowExecutor b = wm.CreateWindow(2, 2))
        {
            b.ResetState();
            b.Run(ReadOnlySpan<int>.Empty, h1, positions, h2);
        }

        using ITensor cycled = wm.ApplyOutputHead(h2, seqLen);
        Report("hybrid cut@2", refLogits, cycled);
    }

    [Fact]
    public void DenseSplitWindows_MatchWholeModel()
    {
        const int hidden = 64, numHeads = 4, headDim = 16, intermediate = 128, vocab = 32, layers = 4;
        var rng = new Random(42);
        var bld = new SafetensorsFixtureBuilder();
        bld.AddFloat32("model.embed_tokens.weight", [vocab, hidden], Rand(rng, vocab * hidden));
        bld.AddFloat32("model.norm.weight", [hidden], Rand(rng, hidden));
        for (int i = 0; i < layers; i++)
        {
            string p = $"model.layers.{i}";
            bld.AddFloat32($"{p}.input_layernorm.weight", [hidden], Rand(rng, hidden));
            bld.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Rand(rng, hidden));
            bld.AddFloat32($"{p}.self_attn.q_proj.weight", [numHeads * headDim, hidden], Rand(rng, numHeads * headDim * hidden));
            bld.AddFloat32($"{p}.self_attn.k_proj.weight", [numHeads * headDim, hidden], Rand(rng, numHeads * headDim * hidden));
            bld.AddFloat32($"{p}.self_attn.v_proj.weight", [numHeads * headDim, hidden], Rand(rng, numHeads * headDim * hidden));
            bld.AddFloat32($"{p}.self_attn.o_proj.weight", [hidden, numHeads * headDim], Rand(rng, hidden * numHeads * headDim));
            bld.AddFloat32($"{p}.mlp.gate_proj.weight", [intermediate, hidden], Rand(rng, intermediate * hidden));
            bld.AddFloat32($"{p}.mlp.up_proj.weight", [intermediate, hidden], Rand(rng, intermediate * hidden));
            bld.AddFloat32($"{p}.mlp.down_proj.weight", [hidden, intermediate], Rand(rng, hidden * intermediate));
        }
        bld.AddFloat32("lm_head.weight", [vocab, hidden], Rand(rng, vocab * hidden));

        string path = Path.Combine(_scratch, "dense.safetensors");
        bld.WriteTo(path);

        var cfg = new ModelConfig
        {
            Architecture = Architecture.Llama,
            VocabSize = vocab,
            HiddenSize = hidden,
            IntermediateSize = intermediate,
            NumLayers = layers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numHeads,
            HeadDim = headDim,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: headDim, Type: RoPEType.Norm),
        };

        using var file = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(file, cfg);

        int[] tokens = [1, 5, 9, 3, 7];
        int[] positions = [0, 1, 2, 3, 4];
        int seqLen = tokens.Length;

        using ITensor reference = model.Forward(tokens, positions, -1);
        var refLogits = new float[seqLen * vocab];
        unsafe { new ReadOnlySpan<float>((void*)reference.DataPointer, refLogits.Length).CopyTo(refLogits); }

        using var wm = new CpuLayerWindowModel(model, cfg);
        var h1 = new float[seqLen * hidden];
        var h2 = new float[seqLen * hidden];
        using (ILayerWindowExecutor a = wm.CreateWindow(0, 1)) a.Run(tokens, ReadOnlySpan<float>.Empty, positions, h1);
        using (ILayerWindowExecutor b = wm.CreateWindow(1, 3)) b.Run(ReadOnlySpan<int>.Empty, h1, positions, h2);
        using ITensor cycled = wm.ApplyOutputHead(h2, seqLen);
        Report("dense cut@1", refLogits, cycled);
    }

    private void Report(string label, float[] refLogits, ITensor cycled)
    {
        var cyc = new float[refLogits.Length];
        unsafe { new ReadOnlySpan<float>((void*)cycled.DataPointer, cyc.Length).CopyTo(cyc); }

        double maxAbs = 0, maxRel = 0;
        for (int i = 0; i < refLogits.Length; i++)
        {
            double abs = Math.Abs(refLogits[i] - cyc[i]);
            double rel = abs / Math.Max(1e-9, Math.Abs(refLogits[i]));
            if (abs > maxAbs) maxAbs = abs;
            if (rel > maxRel) maxRel = rel;
        }

        output.WriteLine($"{label}: maxAbs={maxAbs:E6} maxRel={maxRel:E6}");
        Assert.True(maxRel < 1e-4, $"{label}: maxAbs={maxAbs:E6} maxRel={maxRel:E6}");
    }

    private static float[] Rand(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() - 0.5) * 0.1);
        return a;
    }
}
