using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1485: the CPU Qwen3 hybrid hosts honour <c>lastTokenLogitsOnly</c> — the LM head runs over the last row only, the
/// result is <c>[1, vocab]</c> and bit-identical to the last row of the full-logits forward. They also no longer keep a
/// model-state <c>cap x vocab x 4</c> logits buffer (the head writes straight into the caller-owned result).
/// </summary>
public sealed class HybridLastRowLogitsTests : IDisposable
{
    private static readonly int[] Prompt = [1, 3, 5, 7, 9];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-lastrow-{Guid.NewGuid():N}");

    public HybridLastRowLogitsTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>The rotated head (Hadamard fold) is the shape-sensitive path: only the kept row may be rotated.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void DenseHybrid_LastRowHint_SingleRowIdenticalToFullLastRow(bool fold)
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold.gguf"));
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (fold) config = config with { HadamardFold = SyntheticHadamardFold.For(config) };
        int[] pos = Enumerable.Range(0, Prompt.Length).ToArray();
        int v = config.VocabSize;

        float[] full;
        using (var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded))
        using (ITensor lg = model.Forward(Prompt, pos, -1))
        {
            Assert.Equal(Prompt.Length, lg.Shape[0]);
            full = new ReadOnlySpan<float>((float*)lg.DataPointer + (long)(Prompt.Length - 1) * v, v).ToArray();
        }

        using var model2 = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        using ITensor last = ((IModel)model2).Forward(Prompt, pos, -1, null, lastTokenLogitsOnly: true);
        Assert.Equal(1, last.Shape[0]);
        Assert.Equal(v, last.Shape[1]);
        Assert.Equal(full, new ReadOnlySpan<float>((float*)last.DataPointer, v).ToArray());
    }

    /// <summary>MTP capture must still see every row: the draft after a last-row prefill equals the draft after a full one.</summary>
    [Fact]
    public unsafe void DenseHybrid_LastRowHint_KeepsMtpCaptureOfEveryRow()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold.gguf"));
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] pos = Enumerable.Range(0, Prompt.Length).ToArray();
        int v = config.VocabSize;

        float[] Draft(bool lastOnly)
        {
            using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using var mtp = model.CreateMtpState()!;
            using (ITensor t = ((IModel)model).Forward(Prompt, pos, -1, kv, null, mtp, lastOnly))
                Assert.Equal(lastOnly ? 1 : Prompt.Length, t.Shape[0]);
            using ITensor draft = model.ForwardMtp(mtp, tokenId: 4, position: Prompt.Length);
            return new ReadOnlySpan<float>((float*)draft.DataPointer, v).ToArray();
        }

        Assert.Equal(Draft(false), Draft(true));
    }

    [Fact]
    public unsafe void MoeHybrid_LastRowHint_SingleRowIdenticalToFullLastRow()
    {
        string path = SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "qwen35moe-tiny.gguf"));
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] tokens = [0, 1, 2, 3];
        int[] pos = [0, 1, 2, 3];
        int v = config.VocabSize;

        float[] full;
        using (var model = (Qwen3MoeHybridTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config))
        using (ITensor lg = model.Forward(tokens, pos, -1))
        {
            Assert.Equal(tokens.Length, lg.Shape[0]);
            full = new ReadOnlySpan<float>((float*)lg.DataPointer + (long)(tokens.Length - 1) * v, v).ToArray();
        }

        using var model2 = (Qwen3MoeHybridTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using ITensor last = ((IModel)model2).Forward(tokens, pos, -1, null, lastTokenLogitsOnly: true);
        Assert.Equal(1, last.Shape[0]);
        Assert.Equal(full, new ReadOnlySpan<float>((float*)last.DataPointer, v).ToArray());
    }
}
