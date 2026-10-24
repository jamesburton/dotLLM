using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1222: one-token steps and short verifies run the layer loop as a CUDA graph, with step <c>t+1</c> captured
/// ahead while step <c>t</c> runs. The graph replays the eager launch sequence, so every logit must be bit-identical to
/// the eager run — across the attention-tier switch at <see cref="CudaKernels.AttentionGqaSplitMinSeqKv"/> (a topology
/// change: the executable is re-instantiated), a 3-row verify, and a position jump that must invalidate the step
/// captured ahead.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseDecodeGraphTests
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDenseDecodeGraphTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void GraphedSteps_AreBitIdenticalToEager_AcrossTierSwitchVerifyAndJump()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        Skip.IfNot(NativeLibrary.TryLoad(lib, out nint h) && CudaDevice.IsAvailable(), "No CUDA GPU available");
        NativeLibrary.Free(h);
        string? path = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), $"Bonsai 2 PQ2_0 GGUF not found (set {ModelPathEnvVar})");
        string ptxDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "ptx"));
        Skip.IfNot(Directory.Exists(ptxDir), "PTX files not found");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        try
        {
            CudaQwen3HybridDenseTransformerModel.DecodeGraphOverride = false;
            List<float[]> eager = Run(model, config.VocabSize);
            var before = model.DecodeGraphCounts;
            Assert.Equal(0, before.Launches);

            CudaQwen3HybridDenseTransformerModel.DecodeGraphOverride = true;
            List<float[]> graphed = Run(model, config.VocabSize);
            var counts = model.DecodeGraphCounts;
            _out.WriteLine($"graph launches {counts.Launches}, captured ahead {counts.AheadHits}, " +
                           $"instantiations {counts.Instantiations}, eager fallbacks {counts.EagerFallbacks}");

            Assert.Equal(eager.Count, graphed.Count);
            for (int s = 0; s < eager.Count; s++)
                for (int i = 0; i < eager[s].Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(eager[s][i]) == BitConverter.SingleToInt32Bits(graphed[s][i]),
                        $"call {s}, logit {i}: eager {eager[s][i]:R} vs graph {graphed[s][i]:R}");

            Assert.True(counts.Launches >= 14, "the one-token steps and the verify did not run through the graph");
            Assert.True(counts.AheadHits >= 8, "consecutive steps were not served by the graph captured ahead");
        }
        finally
        {
            CudaQwen3HybridDenseTransformerModel.DecodeGraphOverride = null;
        }
    }

    // 250-token prefill, 12 steps across seqKv 256 (attention tier switch), a 3-row verify, then a position JUMP (a step
    // at +2: the step captured ahead for +1 must not be launched) and two more steps.
    private static List<float[]> Run(CudaQwen3HybridDenseTransformerModel model, int vocab)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(maxSeqLen: 512);
        var logits = new List<float[]>();
        const int prefill = 250;
        int[] tokens = Enumerable.Range(0, prefill).Select(i => 1 + (i * 7919) % 30000).ToArray();
        using (model.Forward(tokens, Enumerable.Range(0, prefill).ToArray(), deviceId: -1, kv, lastTokenLogitsOnly: true)) { }

        int pos = prefill;
        void Step(int[] ids)
        {
            using ITensor t = model.Forward(ids, Enumerable.Range(pos, ids.Length).ToArray(), deviceId: -1, kv);
            logits.Add(Copy(t, ids.Length * vocab));
            pos += ids.Length;
        }

        for (int s = 0; s < 12; s++) Step([(7 + s * 13) % 30000]);
        Step([101, 202, 303]);
        pos += 1;   // the jump
        Step([404]);
        Step([505]);
        return logits;
    }

    private static unsafe float[] Copy(ITensor t, int count)
        => new ReadOnlySpan<float>((void*)t.DataPointer, count).ToArray();
}
