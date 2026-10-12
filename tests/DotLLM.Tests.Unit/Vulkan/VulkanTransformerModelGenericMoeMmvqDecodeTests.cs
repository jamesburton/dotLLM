using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #912: the generic MoE layer (<c>RecordMoeLayer</c>: GLM-4.7-Flash, gpt-oss, Mixtral, OLMoE ...) used to run the scalar one-thread-per-cell
/// indexed kernels at every decode-sized row count. It now routes rows below the grouped-prefill gate onto the indexed dp4a MMVQ kernels. These tests
/// PIN THE ARM: they read the layer counters, so a regression that silently falls back to the scalar kernels (the original bug) fails here even though
/// the logits would still be right. The fixture is a Mixtral-shaped model with 8 experts and top-2 routing (numExperts &gt; topK, not every expert is hit
/// by a 1-token forward) and the GLM-style mixed bank types (Q4_K gate/up with a Q6_K or Q4_K down).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelGenericMoeMmvqDecodeTests : IDisposable
{
    private const int Hidden = 256, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 64, Vocab = 16, Interm = 256, Experts = 8, TopK = 2;
    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_K, QuantizationType.Q6_K, 11)]   // GLM-4.7-Flash's Q6_K-down layers
    [InlineData(QuantizationType.Q4_K, QuantizationType.Q4_K, 12)]   // ... and its Q4_K-down layers
    [InlineData(QuantizationType.Q5_K, QuantizationType.Q5_K, 13)]
    public void Decode_TakesMmvqArm_AndMatchesScalarPathAndCpu(QuantizationType gateUp, QuantizationType down, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(gateUp, down, seed);

        int[][] sequences = [[3], [5], [1, 2, 3], [7, 1, 4, 9, 2, 6]];   // 1-row decode and 3/6-row small-row steps
        var cpuLogits = new float[sequences.Length][];
        using (var gguf = GgufFile.Open(path))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
            using var cpu = TransformerModel.BuildFromPrebuiltWeights(cpuWeights, config);
            for (int i = 0; i < sequences.Length; i++)
            {
                using ITensor l = cpu.Forward(sequences[i], Enumerable.Range(0, sequences[i].Length).ToArray(), deviceId: -1);
                cpuLogits[i] = LastRow(l);
            }
        }

        using var g2 = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(g2.Metadata);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var vk = VulkanTransformerModel.LoadFromGguf(device, g2, cfg, spvDir);

        for (int i = 0; i < sequences.Length; i++)
        {
            int[] ids = sequences[i];
            int[] pos = Enumerable.Range(0, ids.Length).ToArray();

            vk.GenericMoeMmvqEnabled = true;
            long mm0 = vk.GenericMoeMmvqLayerCount, sc0 = vk.GenericMoeScalarDecodeLayerCount;
            float[] viaMmvq;
            using (ITensor l = vk.Forward(ids, pos, deviceId: -1)) viaMmvq = LastRow(l);
            // The arm ran on every MoE layer and the scalar decode kernels on none.
            Assert.Equal(Layers, vk.GenericMoeMmvqLayerCount - mm0);
            Assert.Equal(0, vk.GenericMoeScalarDecodeLayerCount - sc0);

            vk.GenericMoeMmvqEnabled = false;
            mm0 = vk.GenericMoeMmvqLayerCount; sc0 = vk.GenericMoeScalarDecodeLayerCount;
            float[] viaScalar;
            using (ITensor l = vk.Forward(ids, pos, deviceId: -1)) viaScalar = LastRow(l);
            // Control: switched off, the SAME model takes the scalar kernels (so the counters above can tell the two apart).
            Assert.Equal(0, vk.GenericMoeMmvqLayerCount - mm0);
            Assert.Equal(Layers, vk.GenericMoeScalarDecodeLayerCount - sc0);

            float maxAbs = Math.Max(viaScalar.Max(MathF.Abs), 1e-3f);
            double dScalar = 0, dCpu = 0, dScalarCpu = 0;
            for (int c = 0; c < Vocab; c++)
            {
                dScalar = Math.Max(dScalar, MathF.Abs(viaMmvq[c] - viaScalar[c]));
                dCpu = Math.Max(dCpu, MathF.Abs(viaMmvq[c] - cpuLogits[i][c]));
                dScalarCpu = Math.Max(dScalarCpu, MathF.Abs(viaScalar[c] - cpuLogits[i][c]));
            }
            Assert.True(dScalar <= 0.05 * maxAbs + 0.02, $"{gateUp}/{down} rows={ids.Length}: mmvq vs scalar-Vulkan max|diff| {dScalar:G4} (logit maxAbs {maxAbs:G4})");
            Assert.True(dCpu <= 0.05 * maxAbs + 0.08, $"{gateUp}/{down} rows={ids.Length}: mmvq vs CPU max|diff| {dCpu:G4} (scalar-vs-CPU {dScalarCpu:G4}, logit maxAbs {maxAbs:G4})");
            Assert.Equal(ArgMax(cpuLogits[i]), ArgMax(viaMmvq));
        }
    }

    /// <summary>The arm has a row ceiling: above <c>GenericMoeMmvqMaxRows</c> the layer must NOT take it (it would re-read the bank per routed row).</summary>
    [SkippableFact]
    public void RowsAboveTheCeiling_DoNotTakeTheArm()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(QuantizationType.Q4_K, QuantizationType.Q6_K, 14);
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var vk = VulkanTransformerModel.LoadFromGguf(device, gguf, cfg, spvDir);
        vk.GenericMoeMmvqMaxRows = 2;
        long mm0 = vk.GenericMoeMmvqLayerCount;
        using (ITensor l = vk.Forward([1, 2, 3], [0, 1, 2], deviceId: -1)) { }
        Assert.Equal(0, vk.GenericMoeMmvqLayerCount - mm0);
        using (ITensor l = vk.Forward([1, 2], [0, 1], deviceId: -1)) { }
        Assert.Equal(Layers, vk.GenericMoeMmvqLayerCount - mm0);
    }

    private static unsafe float[] LastRow(ITensor logits)
    {
        int rows = logits.Shape[0];
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * Vocab, Vocab).ToArray();
    }

    private static int ArgMax(float[] v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    private string WriteFixture(QuantizationType gateUp, QuantizationType down, int seed)
    {
        var b = new GgufTestData(version: 3);
        var rng = new Random(seed);
        int qOut = Heads * HeadDim, kvOut = KvHeads * HeadDim;
        b.AddString("general.architecture", "mixtral");
        b.AddUInt32("mixtral.embedding_length", Hidden);
        b.AddUInt32("mixtral.block_count", Layers);
        b.AddUInt32("mixtral.attention.head_count", Heads);
        b.AddUInt32("mixtral.attention.head_count_kv", KvHeads);
        b.AddUInt32("mixtral.attention.key_length", HeadDim);
        b.AddUInt32("mixtral.context_length", 16);
        b.AddFloat32("mixtral.attention.layer_norm_rms_epsilon", 1e-5f);
        b.AddUInt32("mixtral.vocab_size", Vocab);
        b.AddFloat32("mixtral.rope.freq_base", 10000.0f);
        b.AddUInt32("mixtral.rope.dimension_count", HeadDim);
        b.AddUInt32("mixtral.expert_count", Experts);
        b.AddUInt32("mixtral.expert_used_count", TopK);
        b.AddUInt32("mixtral.expert_feed_forward_length", Interm);

        AddF32(b, "token_embd.weight", [Hidden, Vocab], rng, amplitude: 0.5f);
        AddF32(b, "output_norm.weight", [Hidden], rng, center: 1.0f, jitter: 0.05f);
        AddF32(b, "output.weight", [Hidden, Vocab], rng, amplitude: 0.5f);
        for (int i = 0; i < Layers; i++)
        {
            string p = $"blk.{i}";
            AddF32(b, $"{p}.attn_norm.weight", [Hidden], rng, center: 1.0f, jitter: 0.05f);
            AddF32(b, $"{p}.ffn_norm.weight", [Hidden], rng, center: 1.0f, jitter: 0.05f);
            AddF32(b, $"{p}.attn_q.weight", [Hidden, qOut], rng);
            AddF32(b, $"{p}.attn_k.weight", [Hidden, kvOut], rng);
            AddF32(b, $"{p}.attn_v.weight", [Hidden, kvOut], rng);
            AddF32(b, $"{p}.attn_output.weight", [qOut, Hidden], rng);
            AddF32(b, $"{p}.ffn_gate_inp.weight", [Hidden, Experts], rng, amplitude: 0.5f);
            AddBank(b, $"{p}.ffn_gate_exps.weight", k: Hidden, m: Interm, gateUp, rng);
            AddBank(b, $"{p}.ffn_up_exps.weight", k: Hidden, m: Interm, gateUp, rng);
            AddBank(b, $"{p}.ffn_down_exps.weight", k: Interm, m: Hidden, down, rng);
        }
        string path = b.WriteToTempFile();
        _tempFiles.Add(path);
        return path;
    }

    private static void AddBank(GgufTestData b, string name, int k, int m, QuantizationType qt, Random rng)
    {
        int rowBytes = qt switch
        {
            QuantizationType.Q4_K => (k / Q4KFixture.Q4KGroupSize) * Q4KFixture.Q4KBlockBytes,
            QuantizationType.Q5_K => (k / Q5KFixture.Q5KGroupSize) * Q5KFixture.Q5KBlockBytes,
            QuantizationType.Q6_K => (k / Q6KFixture.Q6KGroupSize) * Q6KFixture.Q6KBlockBytes,
            _ => throw new NotSupportedException($"{qt}"),
        };
        var all = new byte[(long)Experts * m * rowBytes];
        for (int e = 0; e < Experts; e++)
        {
            float[] f32 = Q4KFixture.RandomFloats(rng, m * k, range: 0.1f);
            byte[] q = qt switch
            {
                QuantizationType.Q4_K => Q4KFixture.QuantizeRows(f32, m, k),
                QuantizationType.Q5_K => Q5KFixture.QuantizeRows(f32, m, k),
                _ => Q6KFixture.QuantizeRows(f32, m, k),
            };
            Buffer.BlockCopy(q, 0, all, e * m * rowBytes, q.Length);
        }
        b.AddTensor(name, [k, m, Experts], (uint)qt, all);
    }

    private static void AddF32(GgufTestData b, string name, int[] shape, Random rng, float amplitude = 0.1f, float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        foreach (int d in shape) n *= d;
        byte[] bytes = new byte[n * sizeof(float)];
        for (long i = 0; i < n; i++)
        {
            float raw = (float)(rng.NextDouble() * 2.0 - 1.0);
            float v = jitter > 0f ? center + jitter * raw : amplitude * raw;
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan((int)(i * sizeof(float)), sizeof(float)), v);
        }
        b.AddTensor(name, shape, quantType: 0, bytes);
    }
}
