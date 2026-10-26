using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in A/B harness (issue #1102) for split-KV decode attention at Bonsai 2's full-attention shape (24 query heads over
/// 4 KV heads, head_dim 256): the per-query-head pass against the GQA-grouped pass, each over an F32 and an F16 cache.
/// </summary>
/// <remarks>
/// <para>One layer's decode attention per launch. Every arm runs in one process, interleaved, with the order reversed on
/// alternate rounds; the metric is the minimum wall time per arm (submit and wait included). The effective bandwidth column
/// counts each cached K and V byte once (the unique bytes a decode step must read).</para>
/// <para>Each arm's output is first compared with the F32 per-query-head arm on the same rows (F16 arms on F16-rounded rows),
/// so a fast arm that computes something else fails instead of reporting.</para>
/// <para>Enable with <c>DOTLLM_SPLITKV_DECODE_AB=1</c>; <c>DOTLLM_SPLITKV_DECODE_AB_KV</c> overrides the KV lengths
/// (comma-separated, default 8192,32768,131072).</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanSplitKvDecodeBench
{
    private const int Rounds = 9;

    // DOTLLM_SPLITKV_DECODE_AB_SHAPE=heads,kvHeads,headDim overrides the Bonsai 2 shape (Flint: 32,4,128, issue #1161).
    private static readonly int[] Shape = (Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB_SHAPE") ?? "24,4,256")
        .Split(',').Select(int.Parse).ToArray();
    private static readonly int NumHeads = Shape[0], NumKvHeads = Shape[1], HeadDim = Shape[2];
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the bench with the xUnit output sink.</summary>
    /// <param name="output">Sink for the results table.</param>
    public VulkanSplitKvDecodeBench(ITestOutputHelper output) => _output = output;

    /// <summary>Times the four arms at each KV length.</summary>
    [SkippableFact]
    public void AB_SplitKvDecode_Bonsai2Shape()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB") == "1", "DOTLLM_SPLITKV_DECODE_AB=1 to enable this A/B.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        int[] lengths = (Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB_KV") ?? "8192,32768,131072")
            .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();

        using var device = VulkanDevice.Create();
        using var k32 = VulkanSplitKvAttentionKernel.Create(device, spvDir, VulkanKvStorage.F32);
        using var k16 = VulkanSplitKvAttentionKernel.Create(device, spvDir, VulkanKvStorage.F16);
        _output.WriteLine($"device {device.DeviceName}");
        _output.WriteLine("seqKv\tarm\tmin_ms\tunique_GB/s\tmax|diff| vs f32-classic");

        foreach (int seqKv in lengths)
        {
            int kvElems = seqKv * NumKvHeads * HeadDim;
            var rng = new Random(seqKv);
            float[] q = new float[NumHeads * HeadDim];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);

            // Rows already rounded to F16, so every arm computes on the same values.
            float[] kv32 = new float[kvElems];
            ushort[] kv16 = new ushort[kvElems];
            for (int i = 0; i < kvElems; i++)
            {
                Half h = (Half)(rng.NextDouble() * 2 - 1);
                kv16[i] = BitConverter.HalfToUInt16Bits(h);
                kv32[i] = (float)h;
            }

            using var bq = device.Allocate((long)q.Length * sizeof(float));
            // The cache lives in device-local memory in the model (#1161: host-visible buffers read at a fraction of the bandwidth on
            // UMA and made every arm look latency-bound).
            using var bk32 = DeviceLocal(device, MemoryMarshal.AsBytes(kv32.AsSpan()));
            using var bv32 = DeviceLocal(device, MemoryMarshal.AsBytes(kv32.AsSpan()));
            using var bk16 = DeviceLocal(device, MemoryMarshal.AsBytes(kv16.AsSpan()));
            using var bv16 = DeviceLocal(device, MemoryMarshal.AsBytes(kv16.AsSpan()));
            using var bo = device.Allocate((long)q.Length * sizeof(float));
            device.Upload(q.AsSpan(), bq);
            kv32 = null!;
            kv16 = null!;

            var arms = new (string Name, VulkanSplitKvAttentionKernel Kernel, bool Gqa, bool V2, VulkanDevice.Buffer K, VulkanDevice.Buffer V, int Bytes)[]
            {
                ("f32-classic", k32, false, false, bk32, bv32, 4),
                ("f32-gqa", k32, true, false, bk32, bv32, 4),
                ("f16-classic", k16, false, false, bk16, bv16, 2),
                ("f16-gqa", k16, true, false, bk16, bv16, 2),
                ("f32-gqa2", k32, true, true, bk32, bv32, 4),
                ("f16-gqa2", k16, true, true, bk16, bv16, 2),
            };
            k16.PreferGqa = true;
            k16.PreferGqa2 = true;
            if (!k16.UsesGqa2(NumHeads, NumKvHeads, HeadDim)) arms = arms[..4];

            float[]? reference = null;
            var outputs = new float[arms.Length][];
            for (int a = 0; a < arms.Length; a++)
            {
                outputs[a] = new float[q.Length];
                Run(arms[a]);
                device.Download(bo, outputs[a]);
                reference ??= outputs[a];
            }

            var best = Enumerable.Repeat(double.MaxValue, arms.Length).ToArray();
            for (int round = 0; round < Rounds; round++)
            {
                for (int i = 0; i < arms.Length; i++)
                {
                    int a = (round & 1) == 0 ? i : arms.Length - 1 - i;
                    var sw = Stopwatch.StartNew();
                    Run(arms[a]);
                    best[a] = Math.Min(best[a], sw.Elapsed.TotalMilliseconds);
                }
            }

            for (int a = 0; a < arms.Length; a++)
            {
                float maxDiff = 0;
                for (int i = 0; i < q.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(outputs[a][i] - reference![i]));
                double gbs = 2.0 * kvElems * arms[a].Bytes / (best[a] / 1000) / 1e9;
                _output.WriteLine($"{seqKv}\t{arms[a].Name}\t{best[a]:F3}\t{gbs:F1}\t{maxDiff:E2}");
                Assert.True(maxDiff <= 1e-3f, $"{arms[a].Name} at seqKv {seqKv} disagrees with f32-classic by {maxDiff}");
            }

            void Run((string Name, VulkanSplitKvAttentionKernel Kernel, bool Gqa, bool V2, VulkanDevice.Buffer K, VulkanDevice.Buffer V, int Bytes) arm)
            {
                arm.Kernel.PreferGqa = arm.Gqa;
                arm.Kernel.PreferGqa2 = arm.V2;
                Assert.Equal(arm.Gqa, arm.Kernel.UsesGqa(NumHeads, NumKvHeads, HeadDim));
                arm.Kernel.Launch(bq, arm.K, arm.V, bo, 1, seqKv, NumHeads, NumKvHeads, HeadDim, seqKv - 1);
            }
        }
    }

    private static VulkanDevice.Buffer DeviceLocal(VulkanDevice device, ReadOnlySpan<byte> data)
    {
        using var staging = device.Allocate(data.Length);
        device.Upload(data, staging);
        var dst = device.AllocateDeviceLocal(data.Length);
        device.CopyBufferSynchronous(staging, dst, (ulong)data.Length);
        return dst;
    }
}
