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

/// <summary>
/// #921 on a real hybrid-dense file (opt-in: <c>DOTLLM_TEV1_GGUF</c> = bartowski's Tev1-4B Q4_K_M, whose last two attention layers' q/k/v/o are
/// Q4_0 - 8 tensors, 63 MiB packed, ~0.45 GiB widened). The file is loaded twice on one device, with the legacy keep-packed arms off (the
/// pre-#921 widening) and on, and the device heap is read IN PROCESS (<see cref="VulkanDevice.MemorySnapshot"/>; vulkaninfo and the working set
/// cannot see device-local memory). The resident delta must equal the widened-minus-packed bytes of the file's Q4_0 tensors, and greedy decode
/// over the same prompt must agree (argmax identity over 6 steps; KL within the Q8_1-activation floor).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanHybridDenseLegacyPackedRealFileTests(ITestOutputHelper output)
{
    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    private static double Kl(float[] p, float[] q)
    {
        static double[] LogSoftmax(float[] x)
        {
            double mx = x.Max(), s = 0;
            foreach (float f in x) s += Math.Exp(f - mx);
            double ls = Math.Log(s);
            return x.Select(f => f - mx - ls).ToArray();
        }
        var lp = LogSoftmax(p); var lq = LogSoftmax(q);
        double kl = 0;
        for (int i = 0; i < lp.Length; i++) kl += Math.Exp(lp[i]) * (lp[i] - lq[i]);
        return kl;
    }

    private static long HeapOurs(VulkanDevice device)
    {
        foreach (var line in device.MemorySnapshot().Split("; "))
            if (line.StartsWith("heap1 ours=", StringComparison.Ordinal)) return long.Parse(line.Split(' ')[1].Split('=')[1]) << 20;
        return -1;
    }

    [SkippableFact]
    public void Q4_0Attention_StaysPacked_SameGreedyTokens()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_TEV1_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_TEV1_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3HybridDense, cfg.Architecture);
        long legacyPacked = 0, legacyWidened = 0; int legacyTensors = 0, trunkLegacy = 0;
        foreach (var kv in gguf.TensorsByName)
        {
            var qt = kv.Value.QuantizationType;
            if (qt is not (QuantizationType.Q4_0 or QuantizationType.Q4_1 or QuantizationType.Q5_0 or QuantizationType.Q5_1)) continue;
            long rows = kv.Value.Shape.ElementCount / kv.Value.Shape[0];
            legacyPacked += Dequantize.RowByteSize((int)kv.Value.Shape[0], qt) * rows;
            legacyWidened += kv.Value.Shape.ElementCount * sizeof(float);
            legacyTensors++;
            // blk.N with N >= NumLayers is the NextN / MTP block: uploaded (and now kept packed) but not run by a plain forward.
            if (kv.Key.StartsWith("blk.", StringComparison.Ordinal) && int.TryParse(kv.Key.Split('.')[1], out int blk) && blk < cfg.NumLayers) trunkLegacy++;
            output.WriteLine($"  {kv.Key} {qt} [{string.Join(",", Enumerable.Range(0, kv.Value.Shape.Rank).Select(i => kv.Value.Shape[i]))}]");
        }
        Skip.If(legacyTensors == 0, "the file has no legacy-quant tensors");
        output.WriteLine($"{legacyTensors} legacy-quant tensors ({trunkLegacy} in the trunk): packed {legacyPacked >> 20} MiB, widened {legacyWidened >> 20} MiB");

        int[] ids = [12516, 5193, 264, 882, 11, 304, 264, 2613, 14117, 553, 279, 9392, 11, 1052, 12157, 264];
        const int steps = 6;
        bool saved = VulkanQwen3MoeHybridWeights.LegacyPackedEnabled;
        using var device = VulkanDevice.Create();
        try
        {
            (long heap, long alloc, int[] greedy, float[] lastLogits, long dispatches) Run(bool packed)
            {
                VulkanQwen3MoeHybridWeights.LegacyPackedEnabled = packed;
                long before = HeapOurs(device);
                using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir);
                long after = HeapOurs(device);
                output.WriteLine($"packed={packed}: {device.MemorySnapshot()}");
                using var cache = model.CreateKvCache(64);
                int[] pos = Enumerable.Range(0, ids.Length).ToArray();
                using ITensor p = model.Forward(ids, pos, deviceId: -1, cache);
                float[] pl = Copy(p);
                int vocab = p.Shape[p.Shape.Rank - 1];
                int next = Argmax(pl.AsSpan(pl.Length - vocab, vocab));
                var greedy = new int[steps];
                float[] last = [];
                for (int s = 0; s < steps; s++)
                {
                    greedy[s] = next;
                    using ITensor d = model.Forward([next], [ids.Length + s], deviceId: -1, cache);
                    last = Copy(d);
                    next = Argmax(last);
                }
                return (after - before, model.Weights.AllocatedBytes, greedy, last, model.LegacyQuantDispatchCount);
            }

            var widened = Run(packed: false);
            var packedRun = Run(packed: true);
            long heapDelta = widened.heap - packedRun.heap, allocDelta = widened.alloc - packedRun.alloc, expected = legacyWidened - legacyPacked;
            double kl = Kl(widened.lastLogits, packedRun.lastLogits);
            output.WriteLine($"heap1 ours: widened {widened.heap >> 20} MiB, packed {packedRun.heap >> 20} MiB, delta {heapDelta >> 20} MiB (expected {expected >> 20} MiB); "
                + $"loader AllocatedBytes delta {allocDelta >> 20} MiB; greedy widened [{string.Join(",", widened.greedy)}] packed [{string.Join(",", packedRun.greedy)}]; "
                + $"last-step KL {kl:E3}; legacy dispatches widened {widened.dispatches} packed {packedRun.dispatches}");
            Assert.Equal(0, widened.dispatches);
            if (trunkLegacy > 0) Assert.True(packedRun.dispatches > 0, "the packed legacy-quant arm never ran");
            else output.WriteLine("legacy-quant tensors sit in the MTP block only: the plain forward never dispatches them (memory delta is still the proof)");
            Assert.True(Math.Abs(heapDelta - expected) <= 64L << 20, $"device heap delta {heapDelta >> 20} MiB vs expected {expected >> 20} MiB");   // allocator granularity
            Assert.Equal(widened.greedy, packedRun.greedy);
            Assert.True(kl < 5e-3, $"KL(widened||packed) {kl:E3}");
        }
        finally { VulkanQwen3MoeHybridWeights.LegacyPackedEnabled = saved; }
    }
}
