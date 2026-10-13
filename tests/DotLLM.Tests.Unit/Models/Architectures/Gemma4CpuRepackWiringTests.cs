using System.Reflection;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #933. <c>RunGemma4Layer</c> / <c>Gemma4DenseFfn</c> used to call the row-major <c>Gemm</c> directly,
/// so the R4 repack that <c>TransformerModel.LoadFromGguf</c> builds for every Gemma-4 projection was never
/// read (E2B CPU decode 11 -> 16-19 tok/s once wired). This test proves the repacked copies are CONSUMED by
/// the Gemma-4 layer by perturbation (a fast path "is wired" only if corrupting its buffer changes the
/// output) and that using them is bit-identical to not repacking at all (#530).
/// </summary>
[Collection("SequentialFileIO")]
public sealed class Gemma4CpuRepackWiringTests
{
    private static SyntheticGemma4Config Config() => new()
    {
        // Dense E2B-like shape with K-quant projections whose rows (K = 256 -> 144 B) are SHORTER than the
        // 1 KiB R4 threshold - the K-quant exemption must still route them through the repacked kernels.
        IncludeExperts = false,
        BlockCount = 6,
        HiddenSize = 256,
        DenseFeedForward = 256,
        PerLayerDim = 32,
        SharedKvLayers = 2,
        GlobalLayerStride = 4,
        AttnQuant = QuantizationType.Q4_K,
        DenseFfnQuant = QuantizationType.Q4_K,
        TokenEmbdQuant = QuantizationType.Q8_0,
        SlidingHeadDim = 64,
        GlobalHeadDim = 64,
        SlidingKvHeads = 2,
        GlobalKvHeads = 2,
        HeadCount = 4,
    };

    private static float[] Logits(TransformerModel model, int[] ids)
    {
        var pos = new int[ids.Length];
        for (int i = 0; i < pos.Length; i++) pos[i] = i;
        using ITensor t = model.Forward(ids, pos, deviceId: -1, kvCache: null);
        int vocab = (int)t.Shape[1];
        var all = new float[ids.Length * vocab];
        unsafe { new ReadOnlySpan<float>((void*)t.DataPointer, all.Length).CopyTo(all); }
        return all;
    }

    private sealed class Loaded : IDisposable
    {
        public required GgufFile Gguf { get; init; }
        public required TransformerModel Model { get; init; }
        public void Dispose() { Model.Dispose(); Gguf.Dispose(); }
    }

    private static Loaded Load(string path, string repackMode)
    {
        string? prev = Environment.GetEnvironmentVariable("DOTLLM_CPU_REPACK");
        Environment.SetEnvironmentVariable("DOTLLM_CPU_REPACK", repackMode);
        try
        {
            var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            return new Loaded { Gguf = gguf, Model = TransformerModel.LoadFromGguf(gguf, cfg) };
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_CPU_REPACK", prev);
        }
    }

    [Fact]
    public void Gemma4Layer_ConsumesRepackedWeights_AndMatchesUnrepacked()
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_g4_repack_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, Config());
        int[] ids = { 2, 7, 8, 9, 5, 6, 3, 11, 12, 13 }; // >= 4 tokens: exercises the 4-token kernel + a tail
        try
        {
            using var plain = Load(path, "never");
            Assert.Equal(0, plain.Model.RepackedWeightBytes);
            float[] expected = Logits(plain.Model, ids);

            using var repacked = Load(path, "always");
            Assert.True(repacked.Model.RepackedWeightBytes > 0, "fixture must actually repack (K-quant rows < 1 KiB)");
            float[] actual = Logits(repacked.Model, ids);

            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                    $"logit {i}: repacked {actual[i]:R} != unrepacked {expected[i]:R}");

            // Perturbation: corrupt every repacked buffer the layers own; if RunGemma4Layer really reads
            // them (and only them) the logits must move. Without the wiring they would not change.
            CorruptRepackedLayers(repacked.Model);
            float[] poisoned = Logits(repacked.Model, ids);
            int changed = 0;
            for (int i = 0; i < expected.Length; i++)
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(poisoned[i])) changed++;
            Assert.True(changed > expected.Length / 2,
                $"corrupting the repacked weights changed only {changed}/{expected.Length} logits - the Gemma-4 layer is not reading them");
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static unsafe void CorruptRepackedLayers(TransformerModel model)
    {
        var weights = (TransformerWeights)typeof(TransformerModel)
            .GetField("_weights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var layers = weights.RepackedLayers!;
        foreach (var rl in layers)
        {
            foreach (var rw in new[] { rl.Q, rl.K, rl.V, rl.O, rl.Gate, rl.Up, rl.Down })
            {
                if (rw.Ptr == 0) continue;
                byte* p = (byte*)rw.Ptr;
                // Flip bits across the buffer (leaves fp16 scales mostly sane-looking; the nibbles/values change).
                for (long i = 16; i < rw.AllocatedBytes; i += 7) p[i] ^= 0x5A;
            }
        }
    }
}
