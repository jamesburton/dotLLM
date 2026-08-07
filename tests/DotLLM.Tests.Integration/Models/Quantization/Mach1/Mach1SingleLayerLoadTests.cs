using System.Text.Json;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Quantization.Mach1;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Quantization.Mach1;

/// <summary>
/// Narrow, fast confirmation that <c>Qwen3MoeHybridTransformerModel</c>'s internal
/// Mach-1 per-layer loader helpers (<c>LoadLayerFromMach1</c> and friends) build
/// correctly-shaped weight bundles from the real fixture, WITHOUT paying the full
/// <c>LoadFromMach1Packed</c> public entry point's embedding (~2 GB) + LM head
/// (~2 GB) decode cost. Exercises exactly ONE GDN layer and ONE full-attention
/// layer directly against the real 40-layer config (no truncation needed — the
/// per-layer helpers only look at <c>config.HybridLayout.LayerKind[layerIdx]</c>,
/// not <c>config.NumLayers</c>).
/// </summary>
/// <remarks>
/// This is deliberately the SMALLEST test that still proves the full HF-&gt;dotLLM
/// tensor mapping (issue #266 Phase B) end-to-end: one GDN layer's 256-expert MoE
/// decode is ~3 GB / a few minutes of CPU-bound trellis decode — the full
/// <c>LoadFromMach1Packed_Truncated4Layers</c> test in
/// <c>Mach1PackedCheckpointLoaderTests</c> covers 4 layers plus embed/head and is
/// consequently much slower; this test is the fast, always-run confirmation, while
/// that one remains available as an occasional deeper check.
/// </remarks>
public sealed class Mach1SingleLayerLoadTests
{
    [SkippableFact]
    public void LoadLayerFromMach1_GdnLayer0_BuildsCorrectlyShapedWeights()
    {
        string? root = ResolveFixtureRoot();
        Skip.If(root is null, SkipReason);
        Skip.If(!File.Exists(Path.Combine(root!, "packed", "experts", "L00.safetensors")),
            "packed/experts/L00.safetensors not staged.");
        Skip.If(!File.Exists(Path.Combine(root!, "packed", "ne", "L00.safetensors")),
            "packed/ne/L00.safetensors not staged.");
        Skip.If(!File.Exists(Path.Combine(root!, "extras.safetensors")), "extras.safetensors not staged.");

        ModelConfig config = LoadRealConfig(root!);

        using var checkpoint = Mach1PackedCheckpoint.Open(root!);
        using var extras = Mach1ExtrasReader.Open(root!);
        var owned = new List<nint>();

        try
        {
            Qwen3MoeLayerWeights layer = Qwen3MoeHybridTransformerModel.LoadLayerFromMach1(
                0, checkpoint, extras, config, owned);

            Assert.NotNull(layer.Gdn);
            Assert.Null(layer.FullAttn);
            Assert.Equal(2048, layer.AttnNormWeight.Length);
            Assert.Equal(2048, layer.PostAttnNormWeight.Length);
            Assert.All(layer.AttnNormWeight, v => Assert.True(float.IsFinite(v)));

            GdnTokenMixingWeights gdn = layer.Gdn!;
            Assert.Equal(2048, gdn.QkvInputDim);
            Assert.Equal(8192, gdn.QkvOutputDim); // 2*16*128 + 32*128
            Assert.Equal(2048, gdn.GateInputDim);
            Assert.Equal(4096, gdn.GateOutputDim); // 32*128
            Assert.Equal(4096, gdn.OutInputDim);
            Assert.Equal(2048, gdn.OutOutputDim);
            Assert.Equal(32, gdn.A.Length);
            Assert.Equal(32, gdn.DtBias.Length);
            Assert.Equal(128, gdn.SsmNormWeight.Length);
            Assert.Equal(4 * (2 * 16 + 32) * 128, gdn.Conv1dWeight.Length); // dConv * convDim
            Assert.All(gdn.A, v => Assert.True(float.IsFinite(v)));

            Assert.NotNull(layer.Moe);
            MoeLayerWeights moe = layer.Moe;
            Assert.Equal(256, moe.NumExperts);
            Assert.Equal(8, moe.NumExpertsPerTok);
            Assert.Equal(512, moe.IntermediateSize);
            Assert.Equal(256, moe.W1.Length);
            Assert.Equal(256, moe.W2.Length);
            Assert.Equal(256, moe.W3.Length);
            Assert.True(moe.HasSharedExpert);
            Assert.Equal(512, moe.SharedIntermediateSize);
            Assert.NotNull(moe.SharedExpertGate);

            // Cross-check expert 0's gate projection (W1[0]) against the vendor golden —
            // proves LoadMoeLayerFromMach1's loop wiring (not just Mach1PackedCheckpoint
            // directly, already covered elsewhere) decodes the right expert/proj/dims.
            string goldenPath = Path.Combine(root!, "goldens", "L0_e0_fp32.safetensors");
            if (File.Exists(goldenPath))
            {
                using var goldenFile = SafetensorsFile.Open(goldenPath);
                var golden = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(
                    goldenFile.GetTensorSpan("L0.e0.gate.fp32"));
                unsafe
                {
                    var decoded = new ReadOnlySpan<float>((void*)moe.W1[0], 512 * 2048);
                    Assert.Equal(golden.Length, decoded.Length);
                    int mismatches = 0;
                    for (int i = 0; i < decoded.Length; i++)
                    {
                        if (BitConverter.SingleToInt32Bits(golden[i]) != BitConverter.SingleToInt32Bits(decoded[i]))
                            mismatches++;
                    }
                    Assert.True(mismatches == 0, $"W1[0] (expert 0 gate): {mismatches}/{decoded.Length} not bit-exact vs golden.");
                }
            }
        }
        finally
        {
            unsafe
            {
                foreach (nint ptr in owned)
                    if (ptr != 0)
                        System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)ptr);
            }
        }
    }

    [SkippableFact]
    public void LoadLayerFromMach1_FullAttnLayer3_BuildsCorrectlyShapedWeights()
    {
        string? root = ResolveFixtureRoot();
        Skip.If(root is null, SkipReason);
        Skip.If(!File.Exists(Path.Combine(root!, "packed", "experts", "L03.safetensors")),
            "packed/experts/L03.safetensors not staged.");
        Skip.If(!File.Exists(Path.Combine(root!, "packed", "ne", "L03.safetensors")),
            "packed/ne/L03.safetensors not staged.");
        Skip.If(!File.Exists(Path.Combine(root!, "extras.safetensors")), "extras.safetensors not staged.");

        ModelConfig config = LoadRealConfig(root!);

        using var checkpoint = Mach1PackedCheckpoint.Open(root!);
        using var extras = Mach1ExtrasReader.Open(root!);
        var owned = new List<nint>();

        try
        {
            Qwen3MoeLayerWeights layer = Qwen3MoeHybridTransformerModel.LoadLayerFromMach1(
                3, checkpoint, extras, config, owned);

            Assert.Null(layer.Gdn);
            Assert.NotNull(layer.FullAttn);

            Qwen3FullAttnWeights attn = layer.FullAttn!;
            Assert.Equal(2048, attn.QInputDim);
            Assert.Equal(8192, attn.QOutputDim); // 2*16*256 (Q+Gate fused)
            Assert.Equal(2048, attn.KInputDim);
            Assert.Equal(512, attn.KOutputDim); // 2*256
            Assert.Equal(2048, attn.VInputDim);
            Assert.Equal(512, attn.VOutputDim);
            Assert.Equal(4096, attn.OInputDim); // 16*256
            Assert.Equal(2048, attn.OOutputDim);
            Assert.Equal(2, attn.NumKvHeads);
            Assert.Equal(256, attn.QNormWeight.Length);
            Assert.Equal(256, attn.KNormWeight.Length);
            Assert.All(attn.QNormWeight, v => Assert.True(float.IsFinite(v)));

            Assert.NotNull(layer.Moe);
            Assert.Equal(256, layer.Moe.NumExperts);
        }
        finally
        {
            unsafe
            {
                foreach (nint ptr in owned)
                    if (ptr != 0)
                        System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)ptr);
            }
        }
    }

    private static ModelConfig LoadRealConfig(string root)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "config.json")));
        return Qwen35MoeConfigExtractor.Extract(doc.RootElement);
    }

    private const string SkipReason =
        "Mach-1-Additive-35B fixture not found. Set DOTLLM_MACH1_35B_DIR or populate " +
        "~/.dotllm/test-cache/SyzygyResearch/Mach-1-Additive-35B/ (see docs/QUANTIZATION.md).";

    private static string? ResolveFixtureRoot()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_MACH1_35B_DIR");
        if (!string.IsNullOrWhiteSpace(env) && Directory.Exists(env))
            return env;

        string conventional = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache", "SyzygyResearch", "Mach-1-Additive-35B");
        return Directory.Exists(conventional) ? conventional : null;
    }
}
