using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

[Collection(CudaCollection.Name)]
[Trait("Category", "GPU")]
public sealed unsafe class CudaLfm2ParityTests(Xunit.Abstractions.ITestOutputHelper output)
{
    [SkippableTheory]
    [InlineData(false, false, 1)]
    [InlineData(false, false, 2)]
    [InlineData(false, false, 3)]
    [InlineData(true, false, 2)]
    [InlineData(true, true, 1)]
    [InlineData(true, true, 2)]
    [InlineData(true, true, 3)]
    public void DedicatedHostMatchesCpuPrefillDecodeAndReset(bool quantized, bool quantizeProjections, int topk)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "CUDA device unavailable");
        string ptx = Path.Combine(AppContext.BaseDirectory, "ptx");
        string path = Path.Combine(Path.GetTempPath(), "dotllm-cuda-lfm-" + Guid.NewGuid().ToString("N") + ".gguf");
        File.WriteAllBytes(path, Build(quantized, quantizeProjections));
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = Config with { Moe = Config.Moe! with { NumExpertsPerTok = topk } };
            using var cpu = Lfm2MoeTransformerModel.LoadFromGguf(gguf, config);
            var (loaded, cacheFactory) = CudaModelLoader.CreateFromGguf(gguf, config, 0, ptx);
            using var gpu = loaded;
            Assert.True(gpu.TrySetAllRowLogitsLimit(16));
            Assert.Equal("CudaLfm2MoeTransformerModel", gpu.GetType().Name);
            using var gpuKv = cacheFactory(16);
            using var cpuKv = new SimpleKvCache(1, 2, 8, 16);
            using var c = cpu.Forward([1, 4, 7], [0, 1, 2], -1, cpuKv);
            using var g = gpu.Forward([1, 4, 7], [0, 1, 2], -1, gpuKv);
            Close(All(c), All(g), quantized);
            using var cd = cpu.Forward([3], [3], -1, cpuKv);
            using var gd = gpu.Forward([3], [3], -1, gpuKv);
            Close(All(cd), All(gd), quantized);
            gpu.ResetSequenceState(); cpu.ResetSequenceState();
            using var cr = cpu.Forward([1, 4, 7], [0, 1, 2], -1);
            using var gr = gpu.Forward([1, 4, 7], [0, 1, 2], -1);
            Close(All(cr), All(gr), quantized);
            gpu.ResetSequenceState();
            using var freshCache = cacheFactory(16);
            using var fresh = gpu.Forward([1, 4, 7], [0, 1, 2], -1, freshCache);
            Assert.Equal(All(g), All(fresh));
            Assert.Throws<NotSupportedException>(() => gpu.Forward([3], [4], -1, gpuKv));
        }
        finally { File.Delete(path); }
    }
    private static float[] All(ITensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    private void Close(float[] expected, float[] actual, bool quantized)
    {
        Assert.Equal(expected.Length, actual.Length);
        float max = expected.Max(x => MathF.Abs(x));
        float err = expected.Zip(actual).Max(p => MathF.Abs(p.First - p.Second));
        output.WriteLine($"Logit max error={err:E6}, max scale={max:E6}, relative={err/max:E6}, quantized={quantized}");
        // Q4_0 projections and experts use the same packed Q8_1 activation contract on both backends.
        Assert.True(actual.All(float.IsFinite) && err < 1e-4f * max + 1e-5f, $"max logit error {err}; scale {max}; quantized={quantized}");
    }
    internal static ModelConfig Config => new()
    {
        Architecture = Architecture.Lfm2Moe, HiddenSize = 32, VocabSize = 48, IntermediateSize = 64,
        NumLayers = 3, NumAttentionHeads = 4, NumKvHeads = 2, HeadDim = 8, MaxSequenceLength = 32,
        ShortConvCacheLength = 3,
        HybridLayout = new() { LayerKind = [HybridLayerKind.ShortConv, HybridLayerKind.Attention, HybridLayerKind.ShortConv], HeadCountKv = [0, 2, 0], FeedForwardLength = [64, 0, 0] },
        Moe = new() { NumExperts = 4, NumExpertsPerTok = 2, MoeIntermediateSize = 64, MlpOnlyLayers = [0], SigmoidGating = true, NormTopKProb = true, NormalizeExpertWeights = true, HasSelectionBias = true }
    };

    internal static byte[] Build(bool quantized = true, bool quantizeProjections = false)
    {
        var w = new GgufWriter();
        const string arch = "lfm2moe";
        w.AddString("general.architecture", arch);
        w.AddUInt32($"{arch}.embedding_length", 32); w.AddUInt32($"{arch}.block_count", 3);
        w.AddUInt32($"{arch}.attention.head_count", 4); w.AddInt32Array($"{arch}.attention.head_count_kv", [0, 2, 0]);
        w.AddUInt32($"{arch}.attention.key_length", 8); w.AddUInt32($"{arch}.context_length", 32);
        w.AddUInt32($"{arch}.feed_forward_length", 64); w.AddUInt32($"{arch}.leading_dense_block_count", 1);
        w.AddUInt32($"{arch}.shortconv.l_cache", 3); w.AddUInt32($"{arch}.expert_count", 4);
        w.AddUInt32($"{arch}.expert_used_count", 2); w.AddUInt32($"{arch}.expert_feed_forward_length", 64);
        w.AddUInt32($"{arch}.expert_gating_func", 2); w.AddUInt32($"{arch}.rope.dimension_count", 8);
        w.AddString("tokenizer.ggml.model", "gpt2");
        w.AddStringArray("tokenizer.ggml.tokens", Enumerable.Range(0, 48).Select(i => i switch { 0 => "<unk>", 1 => "<bos>", 2 => "<eos>", _ => $"tok{i}" }).ToArray());
        w.AddStringArray("tokenizer.ggml.merges", []);
        w.AddInt32Array("tokenizer.ggml.token_type", Enumerable.Range(0, 48).Select(i => i < 3 ? 3 : 1).ToArray());
        w.AddUInt32("tokenizer.ggml.unknown_token_id", 0); w.AddUInt32("tokenizer.ggml.bos_token_id", 1); w.AddUInt32("tokenizer.ggml.eos_token_id", 2);
        w.AddString("tokenizer.chat_template", "{% for message in messages %}{{ message['content'] }}{% endfor %}");
        var rng = new Random(1030);
        void Add(string name, int[] shape, bool norm = false)
        {
            int n = shape.Aggregate(1, (a, b) => a * b);
            float[] values = Enumerable.Range(0, n).Select(_ => norm ? 1f : (float)(rng.NextDouble() * 2 - 1) * 0.12f).ToArray();
            if (quantizeProjections && shape.Length >= 2 && shape[0] % 32 == 0)
            {
                byte[] packed = new byte[n / 32 * 18]; rng.NextBytes(packed);
                for (int b = 0; b < n / 32; b++) BitConverter.TryWriteBytes(packed.AsSpan(b * 18, 2), BitConverter.HalfToInt16Bits((Half)0.012f));
                w.AddTensor(name, shape, (uint)QuantizationType.Q4_0, packed);
            }
            else w.AddTensor(name, shape, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
        }
        void Bank(string name, int[] shape)
        {
            if (!quantized) { Add(name, shape); return; }
            int blocks = shape.Aggregate(1, (a, b) => a * b) / 32;
            byte[] bytes = new byte[blocks * 18];
            rng.NextBytes(bytes);
            for (int i = 0; i < blocks; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 18, 2), BitConverter.HalfToInt16Bits((Half)0.012f));
            w.AddTensor(name, shape, (uint)QuantizationType.Q4_0, bytes);
        }
        Add("token_embd.weight", [32, 48]); Add("token_embd_norm.weight", [32], true);
        for (int l = 0; l < 3; l++)
        {
            string p = $"blk.{l}.";
            Add(p + "attn_norm.weight", [32], true); Add(p + "ffn_norm.weight", [32], true);
            if (l != 1) { Add(p + "shortconv.in_proj.weight", [32, 96]); Add(p + "shortconv.out_proj.weight", [32, 32]); Add(p + "shortconv.conv.weight", [3, 32]); }
            else { Add(p + "attn_q.weight", [32, 32]); Add(p + "attn_k.weight", [32, 16]); Add(p + "attn_v.weight", [32, 16]); Add(p + "attn_output.weight", [32, 32]); Add(p + "attn_q_norm.weight", [8], true); Add(p + "attn_k_norm.weight", [8], true); }
            if (l == 0) { Add(p + "ffn_gate.weight", [32, 64]); Add(p + "ffn_up.weight", [32, 64]); Add(p + "ffn_down.weight", [64, 32]); }
            else { Add(p + "ffn_gate_inp.weight", [32, 4]); Add(p + "exp_probs_b.bias", [4]); Bank(p + "ffn_gate_exps.weight", [32, 64, 4]); Bank(p + "ffn_up_exps.weight", [32, 64, 4]); Bank(p + "ffn_down_exps.weight", [64, 32, 4]); }
        }
        return w.Build();
    }
}

