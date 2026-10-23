using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

[Collection("VulkanKernels")]
[Trait("Category", "GPU")]
public sealed unsafe class VulkanLfm2PackedParityTests(ITestOutputHelper output)
{
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void PortablePackedWeightsMatchExplicitTranscodeAcrossF32AttentionCausalChunks(int topK)
        => VerifyPackedAndTranscodedWeights(topK, f32Attention: true);

    [SkippableFact]
    public void DefaultAttentionPackedAndTranscodedWeightsMatchForSameSchedule()
        => VerifyPackedAndTranscodedWeights(topK: 1, f32Attention: false);

    private void VerifyPackedAndTranscodedWeights(int topK, bool f32Attention)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string temp = Path.Combine(Path.GetTempPath(), "dotllm-lfm-portable-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        foreach (string file in Directory.EnumerateFiles(spvDir, "*.spv"))
            if (!(Path.GetFileName(file).Contains("q4_0", StringComparison.OrdinalIgnoreCase) && Path.GetFileName(file).Contains("coopmat", StringComparison.OrdinalIgnoreCase)))
                File.Copy(file, Path.Combine(temp, Path.GetFileName(file)));
        string path = Path.Combine(temp, "tiny.gguf");
        File.WriteAllBytes(path, Build(quantizeProjections: false));
        string? previous = Environment.GetEnvironmentVariable("DOTLLM_VK_Q4_0_NATIVE");
        string? previousAttention = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION");
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION", f32Attention ? "1" : null);
            using var gguf = GgufFile.Open(path);
            using var device = VulkanDevice.Create();
            Skip.IfNot(device.HasIntegerDotProduct, "Packed routed experts require integer dot products; this test removes only the Q4 cooperative-matrix kernels.");
            var config = Config with { Moe = Config.Moe! with { NumExpertsPerTok = topK } };
            (long Bytes, float[] Prefill, float[] Decode, float[] Repeated) Run(bool packed)
            {
                Environment.SetEnvironmentVariable("DOTLLM_VK_Q4_0_NATIVE", packed ? "1" : "0");
                var (loaded, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, temp);
                using var model = Assert.IsType<VulkanLfm2MoeTransformerModel>(loaded);
                using var kv = model.CreateKvCache(16);
                using var p = model.Forward([1, 4, 7], [0, 1, 2], -1, kv);
                float[] prefill = Row(p);
                using var d = model.Forward([3], [3], -1, kv);
                float[] decode = Row(d);
                model.ResetSequenceState();
                using var fresh = model.CreateKvCache(16);
                using var repeated = model.Forward([1, 4, 7, 3], [0, 1, 2, 3], -1, fresh);
                return (model.ComputeMemoryBytes, prefill, decode, Row(repeated));
            }
            var packed = Run(true);
            var transcode = Run(false);
            // Two routed layers, three banks each: Q4_0 blocks18B vs Q8_0 blocks34B.
            const long expertSavings = 2L * 3 * (32 * 64 * 4 / 32) * (34 - 18);
            Assert.True(transcode.Bytes - packed.Bytes >= expertSavings,
                $"No-coopmat device must retain packed banks: packed={packed.Bytes}, transcode={transcode.Bytes}, expectedSavings>={expertSavings}");
            Close(packed.Prefill, transcode.Prefill);
            Close(packed.Decode, transcode.Decode);
            Close(packed.Repeated, transcode.Repeated);
            if (f32Attention)
            {
                Close(transcode.Decode, transcode.Repeated);
                Close(packed.Decode, packed.Repeated);
            }
            else
            {
                // Default cooperative attention prefill uses F16, decode uses F32. The legacy
                // transcode reproduces this schedule gap; this contract compares like schedules.
                output.WriteLine("Default fast attention: cross-schedule parity is not asserted (F16 prefill / F32 decode).");
                LogDifference(transcode.Decode, transcode.Repeated);
                LogDifference(packed.Decode, packed.Repeated);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_Q4_0_NATIVE", previous);
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION", previousAttention);
            Directory.Delete(temp, true);
        }
    }

    [SkippableFact]
    public void PortablePackedDensePrefillMatchesSameWeightsExpandedToF32()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string temp = Path.Combine(Path.GetTempPath(), "dotllm-lfm-dense-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        foreach (string file in Directory.EnumerateFiles(spvDir, "*.spv"))
            if (!(Path.GetFileName(file).Contains("q4_0", StringComparison.OrdinalIgnoreCase) && Path.GetFileName(file).Contains("coopmat", StringComparison.OrdinalIgnoreCase)))
                File.Copy(file, Path.Combine(temp, Path.GetFileName(file)));
        string? previous = Environment.GetEnvironmentVariable("DOTLLM_VK_Q4_0_NATIVE");
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_Q4_0_NATIVE", "1");
            using var device = VulkanDevice.Create();
            Skip.IfNot(device.HasIntegerDotProduct, "Packed routed experts require integer dot products.");
            float[] Run(bool expand)
            {
                string path = Path.Combine(temp, expand ? "expanded.gguf" : "packed.gguf");
                File.WriteAllBytes(path, Build(expandProjections: expand));
                using var gguf = GgufFile.Open(path);
                var (loaded, _) = VulkanModelLoader.CreateFromGguf(device, gguf, Config, temp);
                using var model = Assert.IsType<VulkanLfm2MoeTransformerModel>(loaded);
                Assert.True(model.TrySetAllRowLogitsLimit(4));
                using var kv = model.CreateKvCache(16);
                using var logits = model.Forward([1, 4, 7, 3], [0, 1, 2, 3], -1, kv);
                return new ReadOnlySpan<float>((void*)logits.DataPointer, 4 * 48).ToArray();
            }
            // Both paths use identical packed experts. The reference expands only dense projection
            // bytes to F32, matching the portable GEMM's activation arithmetic instead of MMVQ's Q8_1 tier.
            Close(Run(true), Run(false));
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_Q4_0_NATIVE", previous);
            Directory.Delete(temp, true);
        }
    }
    private static float[] Row(ITensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, 48).ToArray();
    private (float Error, float Scale) LogDifference(float[] a, float[] b)
    {
        float scale = Math.Max(1e-6f, a.Max(MathF.Abs));
        float maxError = a.Zip(b, (x, y) => MathF.Abs(x - y)).Max();
        output.WriteLine($"max abs error={maxError}, reference max={scale}, normalized={maxError / scale}");
        return (maxError, scale);
    }
    private void Close(float[] a, float[] b)
    {
        var (maxError, scale) = LogDifference(a, b);
        Assert.True(b.All(float.IsFinite) && maxError < 3e-4f + scale * 1e-3f,
            $"max abs error={maxError}, reference max={scale}, normalized={maxError / scale}");
    }
    internal static ModelConfig Config => new()
    {
        Architecture = Architecture.Lfm2Moe, HiddenSize = 32, VocabSize = 48, IntermediateSize = 64,
        NumLayers = 3, NumAttentionHeads = 4, NumKvHeads = 2, HeadDim = 8, MaxSequenceLength = 32,
        ShortConvCacheLength = 3,
        HybridLayout = new() { LayerKind = [HybridLayerKind.ShortConv, HybridLayerKind.Attention, HybridLayerKind.ShortConv], HeadCountKv = [0, 2, 0], FeedForwardLength = [64, 0, 0] },
        Moe = new() { NumExperts = 4, NumExpertsPerTok = 2, MoeIntermediateSize = 64, MlpOnlyLayers = [0], SigmoidGating = true, NormTopKProb = true, NormalizeExpertWeights = true, HasSelectionBias = true }
    };

    internal static byte[] Build(bool quantizeProjections = true, bool expandProjections = false)
    {
        var w = new GgufWriter();
        var rng = new Random(1030);
        void Add(string name, int[] shape, bool norm = false)
        {
            if (quantizeProjections && shape.Length == 2 && !name.Contains("embd") && !name.Contains("conv.weight") && !name.Contains("gate_inp")) { Bank(name, shape, expandProjections); return; }
            int n = shape.Aggregate(1, (a, b) => a * b);
            float[] values = Enumerable.Range(0, n).Select(_ => norm ? 1f : (float)(rng.NextDouble() * 2 - 1) * 0.12f).ToArray();
            w.AddTensor(name, shape, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
        }
        void Bank(string name, int[] shape, bool expand = false)
        {
            int blocks = shape.Aggregate(1, (a, b) => a * b) / 32;
            byte[] bytes = new byte[blocks * 18];
            rng.NextBytes(bytes);
            for (int i = 0; i < blocks; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 18, 2), BitConverter.HalfToInt16Bits((Half)0.012f));
            if (expand)
            {
                float[] floats = new float[blocks * 32];
                fixed (byte* source = bytes)
                    Dequantize.ToFloat32((nint)source, floats.Length, QuantizationType.Q4_0, floats);
                w.AddTensor(name, shape, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(floats.AsSpan()).ToArray());
            }
            else w.AddTensor(name, shape, (uint)QuantizationType.Q4_0, bytes);
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






