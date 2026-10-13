using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #962: model-level validation of the head_dim-256 tensor-core flash prefill attention on
/// real Bonsai-2 27B weights. One model instance is loaded and the dispatch is toggled in-process
/// (<see cref="CudaQwen3HybridDenseTransformerModel.FlashPrefillHd256ForTest"/>) between the
/// <c>attention_f32</c> reference path and the flash kernel. Last-token prefill logits must agree
/// (cosine), the flash kernel must actually have served every attention layer (and staged no F32
/// KV), and a teacher-forced short greedy continuation must keep top-1 agreement.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaQwen3HybridDenseFlashPrefillHd256Test
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const int DecodeSteps = 8;

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDenseFlashPrefillHd256Test(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private sealed record Run(float[] First, float[][] Seq, int[] Fed, int Staged, int Flash);

    [SkippableFact]
    public void FlashPrefill_MatchesF32Prefill_OnRealWeights()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? path = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        Skip.If(string.IsNullOrWhiteSpace(path) || !File.Exists(path), $"Set {ModelPathEnvVar} to a Bonsai-2 PQ2_0 GGUF.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);

        foreach (int prefillLen in new[] { 300, 2000 })
        {
            var tokens = new int[prefillLen];
            var pos = new int[prefillLen];
            for (int i = 0; i < prefillLen; i++) { tokens[i] = 1 + (i * 7919 + i / 13) % 30000; pos[i] = i; }

            model.FlashPrefillHd256ForTest = false;           // reference: attention_f32 prefill (F32 staged)
            var rf = Execute(model, config.VocabSize, tokens, pos, null, prefillLen);
            model.FlashPrefillHd256ForTest = true;            // candidate, teacher-forced with the reference tokens
            var fl = Execute(model, config.VocabSize, tokens, pos, rf.Fed, prefillLen);

            Skip.If(fl.Flash == 0, "flash prefill kernel not loaded / not eligible on this device");
            double cos = Cosine(rf.First, fl.First);
            int agree = 0;
            for (int s = 0; s < DecodeSteps; s++)
                if (Argmax(rf.Seq[s]) == Argmax(fl.Seq[s])) agree++;
            _out.WriteLine($"prefill {prefillLen}: logits cosine {cos:F6}; prefill top-1 {Argmax(rf.First)} vs {Argmax(fl.First)}; " +
                           $"decode top-1 agree {agree}/{DecodeSteps}; staged ref {rf.Staged} flash {fl.Staged}; " +
                           $"flash calls {fl.Flash} (layers {model.AttentionLayerCount})");

            Assert.Equal(0, rf.Flash);
            // Long prompts are chunked by the model: one flash call per attention layer per chunk.
            Assert.True(fl.Flash >= model.AttentionLayerCount && fl.Flash % model.AttentionLayerCount == 0,
                $"flash calls {fl.Flash} not a whole number of per-layer passes");
            Assert.Equal(rf.Staged, fl.Flash);   // reference staged exactly where flash served
            Assert.Equal(0, fl.Staged);
            Assert.True(cos > 0.999, $"prefill {prefillLen}: logits cosine {cos}");
            Assert.Equal(Argmax(rf.First), Argmax(fl.First));
            Assert.True(agree >= DecodeSteps - 1, $"prefill {prefillLen}: teacher-forced top-1 agreement {agree}/{DecodeSteps}");
        }
    }

    private static Run Execute(CudaQwen3HybridDenseTransformerModel model, int vocab,
        int[] tokens, int[] pos, int[]? forced, int prefillLen)
    {
        // The GDN recurrent state is model-owned and persists across forwards: reset it so every run
        // is an independent sequence (otherwise run 2 starts from run 1's state).
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(maxSeqLen: prefillLen + DecodeSteps + 8);
        int s0 = model.DebugF32StagedAttentionCallsForTest, f0 = model.DebugFlashPrefillHd256CallsForTest;
        float[] first;
        using (ITensor t = model.Forward(tokens, pos, deviceId: -1, kv, lastTokenLogitsOnly: true))
            first = Copy(t, vocab);
        int staged = model.DebugF32StagedAttentionCallsForTest - s0;
        int flash = model.DebugFlashPrefillHd256CallsForTest - f0;

        var seq = new float[DecodeSteps][];
        var fed = new int[DecodeSteps];
        int next = Argmax(first);
        for (int s = 0; s < DecodeSteps; s++)
        {
            fed[s] = forced is null ? next : forced[s];
            using ITensor t = model.Forward([fed[s]], [prefillLen + s], deviceId: -1, kv);
            seq[s] = Copy(t, vocab);
            next = Argmax(seq[s]);
        }
        return new Run(first, seq, fed, staged, flash);
    }

    private static unsafe float[] Copy(ITensor t, int count)
        => new ReadOnlySpan<float>((void*)t.DataPointer, count).ToArray();

    private static int Argmax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
