using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// Numeric gates for the diffusion-LM spike (issue: text-diffusion mechanism). Validates that dotLLM
/// loads open-dcoder-0.5B (a plain Qwen2ForCausalLM checkpoint finetuned for masked diffusion) and that
/// its forward pass matches a PyTorch/transformers reference — first the standard CAUSAL forward (proves
/// the Qwen2 SafeTensors load + forward is numerically correct), then the BIDIRECTIONAL forward (proves
/// the non-causal attention path the denoising loop needs).
///
/// <para>
/// Gated on the model + reference artifacts produced by the Python harness in
/// <c>C:\Development\dotllm-diffusion-ref\</c> (snapshot + <c>reference.py</c> + <c>dump_for_csharp.py</c>):
/// <c>open-dcoder-0.5B\</c>, <c>ref_meta.json</c>, <c>ref_causal.bin</c>, <c>ref_bidir.bin</c>. Skips
/// cleanly when absent so CI stays green. Override the root via <c>DOTLLM_DIFFUSION_REF_DIR</c>.
/// </para>
/// </summary>
public sealed class OpenDcoderDiffusionTests
{
    private readonly ITestOutputHelper _output;

    public OpenDcoderDiffusionTests(ITestOutputHelper output) => _output = output;

    private static string RefDir =>
        Environment.GetEnvironmentVariable("DOTLLM_DIFFUSION_REF_DIR")
        ?? @"C:\Development\dotllm-diffusion-ref";

    private sealed record RefMeta(int[] input_ids, int L, int vocab);

    private (RefMeta meta, float[] causal, float[] bidir)? TryLoadReference()
    {
        string metaPath = Path.Combine(RefDir, "ref_meta.json");
        string causalPath = Path.Combine(RefDir, "ref_causal.bin");
        string bidirPath = Path.Combine(RefDir, "ref_bidir.bin");
        if (!File.Exists(metaPath) || !File.Exists(causalPath))
            return null;

        var meta = JsonSerializer.Deserialize<RefMeta>(File.ReadAllText(metaPath))!;
        float[] ReadBin(string p)
        {
            byte[] b = File.ReadAllBytes(p);
            float[] f = new float[b.Length / 4];
            Buffer.BlockCopy(b, 0, f, 0, b.Length);
            return f;
        }
        return (meta, ReadBin(causalPath), File.Exists(bidirPath) ? ReadBin(bidirPath) : Array.Empty<float>());
    }

    [Fact]
    public void OpenDcoder_Loads_AsQwen2()
    {
        string modelDir = Path.Combine(RefDir, "open-dcoder-0.5B");
        if (!Directory.Exists(modelDir))
        {
            _output.WriteLine($"[SKIP] model dir not found: {modelDir}");
            return;
        }

        var (model, source, config) = ModelLoader.LoadFromSafetensors(modelDir);
        try
        {
            _output.WriteLine($"arch={config.Architecture} vocab={config.VocabSize} hidden={config.HiddenSize} "
                + $"layers={config.NumLayers} heads={config.NumAttentionHeads} kv={config.NumKvHeads} tied={config.TiedEmbeddings}");
            Assert.Equal(Architecture.Qwen, config.Architecture);
            Assert.Equal(24, config.NumLayers);
            Assert.Equal(896, config.HiddenSize);
            Assert.Equal(14, config.NumAttentionHeads);
            Assert.Equal(2, config.NumKvHeads);
            Assert.Equal(151936, config.VocabSize);
        }
        finally
        {
            model.Dispose();
            (source as IDisposable)?.Dispose();
        }
    }

    [Fact]
    public unsafe void OpenDcoder_CausalForward_MatchesPyTorch()
    {
        string modelDir = Path.Combine(RefDir, "open-dcoder-0.5B");
        var reference = Directory.Exists(modelDir) ? TryLoadReference() : null;
        if (reference is null)
        {
            _output.WriteLine($"[SKIP] model or reference artifacts not found under {RefDir}");
            return;
        }
        var (meta, refCausal, _) = reference.Value;

        var (model, source, config) = ModelLoader.LoadFromSafetensors(modelDir);
        try
        {
            int[] ids = meta.input_ids;
            int[] pos = new int[ids.Length];
            for (int i = 0; i < ids.Length; i++) pos[i] = i;

            using ITensor logits = model.Forward(ids, pos, deviceId: -1);
            int L = logits.Shape[0], V = logits.Shape[1];
            Assert.Equal(meta.L, L);
            Assert.Equal(meta.vocab, V);

            var got = new ReadOnlySpan<float>((void*)logits.DataPointer, L * V);
            CompareLogits("CAUSAL", got, refCausal, L, V);
        }
        finally
        {
            model.Dispose();
            (source as IDisposable)?.Dispose();
        }
    }

    [Fact]
    public unsafe void OpenDcoder_BidirectionalForward_MatchesPyTorch()
    {
        string modelDir = Path.Combine(RefDir, "open-dcoder-0.5B");
        var reference = Directory.Exists(modelDir) ? TryLoadReference() : null;
        if (reference is null || reference.Value.bidir.Length == 0)
        {
            _output.WriteLine($"[SKIP] model or bidirectional reference not found under {RefDir}");
            return;
        }
        var (meta, refCausal, refBidir) = reference.Value;

        var (model, source, config) = ModelLoader.LoadFromSafetensors(modelDir);
        try
        {
            Assert.True(model is DotLLM.Models.Architectures.TransformerModel,
                "expected TransformerModel to toggle BidirectionalAttention");
            ((DotLLM.Models.Architectures.TransformerModel)model).BidirectionalAttention = true;

            int[] ids = meta.input_ids;
            int[] pos = new int[ids.Length];
            for (int i = 0; i < ids.Length; i++) pos[i] = i;

            using ITensor logits = model.Forward(ids, pos, deviceId: -1);
            int L = logits.Shape[0], V = logits.Shape[1];
            var got = new ReadOnlySpan<float>((void*)logits.DataPointer, L * V);

            // Discriminating: bidirectional must DIFFER from the causal reference at position 0 (which under
            // causal sees only itself), confirming the mask was actually dropped — not a vacuous match.
            int causal0 = Argmax(refCausal, 0, V);
            int bidir0 = Argmax(refBidir, 0, V);
            _output.WriteLine($"position-0 argmax: causal-ref={causal0} bidir-ref={bidir0} (differ={causal0 != bidir0})");

            CompareLogits("BIDIRECTIONAL", got, refBidir, L, V);
        }
        finally
        {
            model.Dispose();
            (source as IDisposable)?.Dispose();
        }
    }

    private static int Argmax(float[] data, int row, int v)
    {
        int a = 0; float m = data[row * v];
        for (int j = 1; j < v; j++) if (data[row * v + j] > m) { m = data[row * v + j]; a = j; }
        return a;
    }

    /// <summary>
    /// Compares dotLLM logits to the PyTorch reference: hard assertion on per-position argmax agreement
    /// (robust to small fp / fast-softmax differences) plus a reported max-abs-diff for diagnostics.
    /// </summary>
    private void CompareLogits(string label, ReadOnlySpan<float> got, float[] reference, int L, int V)
    {
        int argmaxMatches = 0;
        float maxAbsDiff = 0, refMaxAbs = 0;
        for (int i = 0; i < L; i++)
        {
            int gA = 0, rA = 0;
            float gMax = got[i * V], rMax = reference[i * V];
            for (int j = 1; j < V; j++)
            {
                float g = got[i * V + j], r = reference[i * V + j];
                if (g > gMax) { gMax = g; gA = j; }
                if (r > rMax) { rMax = r; rA = j; }
                float d = MathF.Abs(g - r);
                if (d > maxAbsDiff) maxAbsDiff = d;
                if (MathF.Abs(r) > refMaxAbs) refMaxAbs = MathF.Abs(r);
            }
            if (gA == rA) argmaxMatches++;
        }
        _output.WriteLine($"{label}: argmax match {argmaxMatches}/{L}  maxAbsDiff={maxAbsDiff:F4}  refMaxAbs={refMaxAbs:F4}  rel={maxAbsDiff / MathF.Max(refMaxAbs, 1e-6f):P1}");
        Assert.Equal(L, argmaxMatches); // every position's greedy token must match the reference
    }
}
