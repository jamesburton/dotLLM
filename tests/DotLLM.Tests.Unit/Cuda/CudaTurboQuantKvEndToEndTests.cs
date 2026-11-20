using System;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
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
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaTurboQuantKvEndToEndTests
{
    private const ulong Seed = 0xC0FFEE_4B2CUL;
    private const ulong VSeedXor = TurboQuantCodec.VSeedMask;

    private readonly ITestOutputHelper _out;
    public CudaTurboQuantKvEndToEndTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
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

    /// <summary>
    /// #1892 Task 8: <c>CreateKvCache</c> serves a TurboQuant request in place or refuses it; it never hands back the F16 cache. Before the gate a
    /// request this host cannot run (tq*q, a window, mixed, V-only) fell through to <c>config.IsQuantized</c> (false for TurboQuant) and built a
    /// plain <see cref="CudaKvCache"/>, silently. Kills the mutant that lets the gate fall through.
    /// </summary>
    [SkippableFact]
    public void Cuda_CreateKvCache_TurboQuant_IsInPlaceOrRefused_NeverF16()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_SMOLLM_Q8_GGUF")
            ?? TestFixtureResolver.ResolveFile([], "QuantFactory", "SmolLM-135M-GGUF", ["SmolLM-135M.Q8_0.gguf"], null).Path ?? "";
        Skip.If(!File.Exists(modelPath), $"SmolLM-135M Q8_0 GGUF not found at {modelPath}");
        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Skip.If(config.HeadDim > 256 || (config.HeadDim & (config.HeadDim - 1)) != 0, "head dim unsupported by TurboQuant kernels");
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir);

        var tq = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, TurboQuantBits: 4);
        Assert.True(model.SupportsInPlaceTurboQuant(tq));
        using (var ok = model.CreateKvCache(32, tq))
            Assert.IsType<CudaTurboQuantInPlaceKvCache>(ok);

        foreach (var (name, bad) in new (string, KvCacheConfig)[]
        {
            ("tq*q", tq with { TurboQuantUseQjl = true }),
            ("window", tq with { MixedPrecisionWindowSize = 16 }),
            ("mixed", new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.Q8_0)),
            ("V-only", new KvCacheConfig(KvCacheDType.F32, KvCacheDType.TurboQuant)),
        })
        {
            Assert.False(model.SupportsInPlaceTurboQuant(bad), name);
            Assert.Throws<NotSupportedException>(() => model.CreateKvCache(32, bad).Dispose());
        }
    }

    [SkippableFact]
    public void Cuda_Dispose_IsIdempotent()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_SMOLLM_Q8_GGUF")
            ?? TestFixtureResolver.ResolveFile([], "QuantFactory", "SmolLM-135M-GGUF", ["SmolLM-135M.Q8_0.gguf"], null).Path ?? "";
        Skip.If(!File.Exists(modelPath), $"SmolLM-135M Q8_0 GGUF not found at {modelPath}");
        using var gguf = GgufFile.Open(modelPath);
        var model = CudaTransformerModel.LoadFromGguf(gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), 0, ptxDir);
        model.Dispose();
        model.Dispose();   // #1892: a refusal path and a finally may both dispose; the second must be a no-op
    }

    [SkippableFact]
    public void Cuda_TurboQuantKv_TracksFullPrecision_OnSmolLM()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_SMOLLM_Q8_GGUF")
            ?? TestFixtureResolver.ResolveFile([], "QuantFactory", "SmolLM-135M-GGUF", ["SmolLM-135M.Q8_0.gguf"], null).Path ?? "";
        Skip.If(!File.Exists(modelPath), $"SmolLM-135M Q8_0 GGUF not found at {modelPath}");

        using var gguf = GgufFile.Open(modelPath);
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
