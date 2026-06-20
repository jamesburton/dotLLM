using System;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// End-to-end parity: runs SmolLM-135M on the CUDA backend with the full-precision
/// <see cref="CudaKvCache"/> vs the GPU <see cref="CudaTurboQuantKvCache"/> (tq4) and checks the
/// next-token distribution barely moves. Exercises the whole CUDA TurboQuant path — per-layer FP16
/// encode (fresh K/V → codes), FP16 dequant (codes → scratch), and attention over the dequantized
/// cache. Mirrors the Vulkan end-to-end test. Gated on a CUDA GPU + turboquant.ptx (runs on T5500).
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class CudaTurboQuantKvEndToEndTests
{
    private const ulong Seed = 0xC0FFEE_4B2CUL;
    private const ulong VSeedXor = 0xD1B54A32D192ED03UL;

    private readonly ITestOutputHelper _out;
    public CudaTurboQuantKvEndToEndTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    [SkippableFact]
    public void Cuda_TurboQuantKv_TracksFullPrecision_OnSmolLM()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"SmolLM-135M Q8_0 GGUF not found at {modelPath}");

        var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int headDim = config.HeadDim;
        Skip.If(headDim > 256 || (headDim & (headDim - 1)) != 0, $"headDim {headDim} unsupported by TurboQuant kernels.");

        int[] tokens = tokenizer.Encode("The capital of France is");
        var positions = new int[tokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;
        int vocab = config.VocabSize;
        int maxSeqLen = tokens.Length + 4;

        var codecK = new TurboQuantCodec(headDim, 4, Seed, useQjl: false);
        var codecV = new TurboQuantCodec(headDim, 4, Seed ^ VSeedXor, useQjl: false);

        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir);

        float[] refRow, tqRow;
        using (var f16 = model.CreateKvCache(maxSeqLen))
        using (var logits = model.Forward(tokens, positions, 0, f16))
            refRow = LastRow(logits, vocab);

        using (var tq = model.CreateTurboQuantKvCache(
                   maxSeqLen, codecK.MseBits, codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD))
        using (var logits = model.Forward(tokens, positions, 0, tq))
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
        _out.WriteLine($"CUDA SmolLM headDim={headDim}: ref argmax={refArg} ('{tokenizer.Decode([refArg])}'), " +
                       $"tq4 argmax={tqArg} ('{tokenizer.Decode([tqArg])}'); meanAbs={meanAbs:F4} maxAbs={maxAbs:F4} cos={cos:F5}");

        Assert.Equal(refArg, tqArg);
        Assert.True(cos > 0.99, $"last-token logits diverged: cos={cos:F5}");
        Assert.True(meanAbs < 1.5, $"mean |Δlogit| unexpectedly high: {meanAbs:F4}");
    }

    // CUDA Forward returns the last token's logits as a host UnmanagedTensor [1, vocab].
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
