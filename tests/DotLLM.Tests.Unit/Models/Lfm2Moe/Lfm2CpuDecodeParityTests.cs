using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lfm2Moe;

[CollectionDefinition("Lfm2CpuDecodeFlags", DisableParallelization = true)]
public sealed class Lfm2CpuDecodeFlagsCollection;

[Collection("Lfm2CpuDecodeFlags")]
public sealed unsafe class Lfm2CpuDecodeParityTests
{
    [SkippableFact]
    public void RealQ4Model_DecodeFastPathPreservesEveryLogit()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_LFM2_GGUF");
        Skip.If(path is null || !File.Exists(path), "Set DOTLLM_LFM2_GGUF to the LFM2 QAD-Q4_0 fixture.");
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Lfm2MoeTransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(6));
        bool previous = MoeSwiGluMlp.DecodeN1Enabled;
        try
        {
            float[][] Run(bool enabled)
            {
                MoeSwiGluMlp.DecodeN1Enabled = enabled;
                model.ResetSequenceState();
                using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, 16);
                using var prefill = model.Forward([124894, 9707, 11], [0, 1, 2], -1, kv);
                var rows = new float[3][];
                for (int i = 0; i < rows.Length; i++)
                {
                    using var logits = model.Forward([100 + i * 13], [3 + i], -1, kv);
                    rows[i] = new ReadOnlySpan<float>((void*)logits.DataPointer, config.VocabSize).ToArray();
                }
                return rows;
            }
            float[][] expected = Run(false);
            long before = MoeSwiGluMlp.DecodeN1Calls;
            float[][] actual = Run(true);
            Assert.Equal(3 * 38, MoeSwiGluMlp.DecodeN1Calls - before);
            for (int row = 0; row < expected.Length; row++)
                for (int i = 0; i < config.VocabSize; i++)
                {
                    Assert.True(float.IsFinite(actual[row][i]));
                    Assert.Equal(BitConverter.SingleToInt32Bits(expected[row][i]), BitConverter.SingleToInt32Bits(actual[row][i]));
                }
        }
        finally { MoeSwiGluMlp.DecodeN1Enabled = previous; }
    }
}
