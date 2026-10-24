using System;
using System.IO;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity: runs a real model on the Vulkan backend with the full-precision
/// <see cref="VulkanKvCache"/> vs the GPU <see cref="VulkanTurboQuantKvCache"/> (tq4) and checks the
/// next-token distribution barely moves. This exercises the whole GPU TurboQuant path through the
/// fence-pipelined forward graph: per-layer encode (fresh K/V → codes), dequant (codes → scratch),
/// and attention over the dequantized cache. 4-bit MSE TurboQuant is quality-neutral (the CPU
/// model-level benchmark measured PPL +0.05 / 98% argmax), so the GPU top-1 token must match and the
/// last-token logits must stay close.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanTurboQuantKvEndToEndTests
{
    private const ulong Seed = 0xC0FFEE_4B2CUL;
    private const ulong VSeedXor = 0xD1B54A32D192ED03UL;

    private readonly ITestOutputHelper _output;
    public VulkanTurboQuantKvEndToEndTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Vulkan_TurboQuantKv_TracksFullPrecision_OnSmolLM()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"SmolLM-135M Q8_0 GGUF not found at {modelPath}");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int headDim = config.HeadDim;
        Skip.If(headDim > DotLLM.Vulkan.Kernels.TurboQuantDequantF32Kernel.MaxHeadDim || (headDim & (headDim - 1)) != 0,
            $"headDim {headDim} unsupported by the TurboQuant kernels (power of two ≤ 256).");

        int[] tokens = tokenizer.Encode("The capital of France is");
        var positions = new int[tokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;
        int vocab = config.VocabSize;
        int maxSeqLen = tokens.Length + 4;

        // Codec constants for the GPU cache (K/V independent rotations — same derivation as the CPU cache).
        var codecK = new TurboQuantCodec(headDim, 4, Seed, useQjl: false);
        var codecV = new TurboQuantCodec(headDim, 4, Seed ^ VSeedXor, useQjl: false);

        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);

        float[] refRow, tqRow;
        using (var f32 = model.CreateKvCache(maxSeqLen))
        using (var logits = model.Forward(tokens, positions, -1, f32))
            refRow = LastRow(logits, vocab);

        using (var tq = model.CreateTurboQuantKvCache(
                   spvDir, maxSeqLen, codecK.MseBits,
                   codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD))
        using (var logits = model.Forward(tokens, positions, -1, tq))
            tqRow = LastRow(logits, vocab);

        int refArg = Argmax(refRow), tqArg = Argmax(tqRow);
        double meanAbs = 0, maxAbs = 0, dot = 0, nr = 0, nt = 0;
        for (int i = 0; i < vocab; i++)
        {
            double d = Math.Abs((double)refRow[i] - tqRow[i]);
            meanAbs += d; if (d > maxAbs) maxAbs = d;
            dot += (double)refRow[i] * tqRow[i]; nr += (double)refRow[i] * refRow[i]; nt += (double)tqRow[i] * tqRow[i];
        }
        meanAbs /= vocab;
        double cos = dot / (Math.Sqrt(nr) * Math.Sqrt(nt) + 1e-12);
        _output.WriteLine($"SmolLM headDim={headDim}: ref argmax={refArg} ('{tokenizer.Decode([refArg])}'), " +
                          $"tq4 argmax={tqArg} ('{tokenizer.Decode([tqArg])}'); meanAbs={meanAbs:F4} maxAbs={maxAbs:F4} cos={cos:F5}");

        // Primary signals: the top token is preserved and the logit vectors stay tightly aligned.
        // (Raw |Δlogit| scales with the model's logit magnitude — a 135M model's single-position
        // logits are noisier than the averaged Llama-8B benchmark; cosine is the robust measure.)
        Assert.Equal(refArg, tqArg);                       // 4-bit MSE preserves the top token
        Assert.True(cos > 0.99, $"last-token logits diverged: cos={cos:F5}");
        Assert.True(meanAbs < 1.5, $"mean |Δlogit| unexpectedly high: {meanAbs:F4}");
    }

    // The Vulkan model returns ONLY the last token's logits as a host UnmanagedTensor [1, vocab].
    private static float[] LastRow(DotLLM.Core.Tensors.ITensor logits, int vocab)
    {
        Assert.Equal(1, logits.Shape[0]);
        var row = new float[vocab];
        float* p = (float*)logits.DataPointer;
        for (int i = 0; i < vocab; i++) row[i] = p[i];
        return row;
    }

    private static int Argmax(float[] row)
    {
        int best = 0; float bv = row[0];
        for (int i = 1; i < row.Length; i++) if (row[i] > bv) { bv = row[i]; best = i; }
        return best;
    }
}
