using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// CPU-side guards for the <c>pq2_0Projections</c> variant of
/// <see cref="SyntheticQwen35HybridDenseMtpGguf"/> (issue #482), which the CUDA small-S GEMV
/// model test (<c>CudaQwen3HybridDenseSmallSGemvTests</c>) relies on to reach the multi-column
/// PQ2_0 kernel. These run anywhere: they pin that the fixture really stores its projections as
/// PQ2_0, loads, and yields finite, non-degenerate logits whose S = 3 and 3 x S = 1 forwards agree
/// on the CPU oracle.
/// </summary>
public sealed class Qwen3HybridDenseSyntheticPq2_0FixtureTests : IDisposable
{
    private static readonly int[] Prompt = [1, 3, 5, 7];
    private static readonly int[] Continuation = [9, 4, 6];

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public Qwen3HybridDenseSyntheticPq2_0FixtureTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35-pq2_0-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Pq2_0Variant_StoresProjectionsAsPq2_0_AndDefaultStaysF32()
    {
        string pq2 = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "pq2.gguf"), pq2_0Projections: true);
        string f32 = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "f32.gguf"));

        using (var gguf = GgufFile.Open(pq2))
        {
            var t = gguf.TensorsByName;
            foreach (string name in new[]
                     {
                         "token_embd.weight", "output.weight",
                         "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
                         "blk.1.attn_q.weight", "blk.1.attn_k.weight", "blk.1.attn_v.weight", "blk.1.attn_output.weight",
                         "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.1.ffn_down.weight",
                     })
            {
                Assert.Equal(QuantizationType.PQ2_0, t[name].QuantizationType);
                Assert.Equal(0, t[name].Shape[0] % 128);
            }
            Assert.Equal(QuantizationType.F32, t["blk.0.ssm_alpha.weight"].QuantizationType);
            Assert.Equal(QuantizationType.F32, t["blk.2.nextn.eh_proj.weight"].QuantizationType);
        }

        using (var gguf = GgufFile.Open(f32))
            Assert.DoesNotContain(gguf.Tensors, d => d.QuantizationType != QuantizationType.F32);
    }

    [Fact]
    public void Pq2_0Variant_S3ForwardMatchesThreeS1Forwards_OnCpu()
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "pq2.gguf"), withMtp: false, pq2_0Projections: true);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        int vocab = config.VocabSize;

        float[][] batched = Run(model, config, batched: true);
        float[][] singles = Run(model, config, batched: false);

        for (int r = 0; r < Continuation.Length; r++)
        {
            float max = 0, lo = float.MaxValue, hi = float.MinValue;
            for (int c = 0; c < vocab; c++)
            {
                Assert.True(float.IsFinite(batched[r][c]) && float.IsFinite(singles[r][c]));
                max = MathF.Max(max, MathF.Abs(batched[r][c] - singles[r][c]));
                lo = MathF.Min(lo, singles[r][c]);
                hi = MathF.Max(hi, singles[r][c]);
            }
            _out.WriteLine($"row {r}: max|S3-S1| {max:E3}, logit range {hi - lo:F4}");
            Assert.True(hi - lo > 1e-2f, $"row {r}: logits are degenerate (range {hi - lo})");
            // Loose on purpose: on an AVX2 host the CPU's single-row PQ2_0 GEMV takes the W2A8
            // int8-activation tier while the multi-row path does not (see CudaPQ2_0GemvTest's
            // remarks), so the CPU's own S=3 vs S=1 gap is activation-quantization error (~1e-2
            // measured on a ~1.8-wide logit range). This guards the fixture's sanity, not a kernel.
            Assert.True(max <= 5e-2f, $"row {r}: CPU S=3 vs S=1 differ by {max:E3}");
        }
    }

    private static unsafe float[][] Run(Qwen3HybridDenseTransformerModel model, ModelConfig config, bool batched)
    {
        model.ResetSequenceState();
        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        for (int i = 0; i < Prompt.Length; i++)
            using (model.Forward([Prompt[i]], [i], deviceId: -1, kv)) { }

        int vocab = config.VocabSize;
        var rows = new float[Continuation.Length][];
        if (batched)
        {
            int[] positions = Enumerable.Range(Prompt.Length, Continuation.Length).ToArray();
            using ITensor logits = model.Forward(Continuation, positions, deviceId: -1, kv);
            for (int r = 0; r < rows.Length; r++)
                rows[r] = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)r * vocab, vocab).ToArray();
        }
        else
        {
            for (int r = 0; r < rows.Length; r++)
            {
                using ITensor logits = model.Forward([Continuation[r]], [Prompt.Length + r], deviceId: -1, kv);
                rows[r] = new ReadOnlySpan<float>((float*)logits.DataPointer, vocab).ToArray();
            }
        }
        return rows;
    }
}
