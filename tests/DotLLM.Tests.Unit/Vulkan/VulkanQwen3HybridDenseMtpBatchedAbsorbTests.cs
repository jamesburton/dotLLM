using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #472: the Vulkan batched, KV-only MTP absorb (one submit for S rows) must leave the head's
/// device KV-cache where the #469 per-token absorb leaves it — and where the CPU per-token
/// reference leaves it.
/// </summary>
/// <remarks>
/// Two trunk forwards, then a verify-shaped third after three speculative draft steps, so the
/// carried row (row 0 of a later batch), the captured-row pairing (rows 1..) and the rollback over
/// drafted slots are all exercised. The fixture's random weights make every hidden row distinct,
/// so a pairing or offset slip moves the K/V rows by O(1).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseMtpBatchedAbsorbTests : IDisposable
{
    // Vulkan vs Vulkan: the n=1 GEMV and the n=S GEMM differ in reduction order only.
    private const float VkTol = 1e-4f;
    // Vulkan vs CPU oracle: a ceiling, see the test body.
    private const float CpuDriftCeiling = 2e-2f;

    private static readonly int KvStride =
        SyntheticQwen35HybridDenseMtpGguf.NumKvHeads * SyntheticQwen35HybridDenseMtpGguf.HeadDim;

    private readonly string _scratch;
    private readonly Xunit.Abstractions.ITestOutputHelper _out;

    public VulkanQwen3HybridDenseMtpBatchedAbsorbTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-mtp-absorb-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void BatchedAbsorb_KvCacheMatchesPerTokenAbsorbAndCpuOracle(bool mtpHasOwnHeadTensors)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, $"mtp-own{mtpHasOwnHeadTensors}.gguf"),
            withMtp: true, mtpHasOwnHeadTensors: mtpHasOwnHeadTensors);

        var cpu = RunCpu(path);
        var vkPerToken = RunVulkan(path, spvDir, perToken: true);
        var vkBatched = RunVulkan(path, spvDir, perToken: false);

        Assert.Equal(cpu.Length, vkBatched.Length);
        Assert.Equal(vkPerToken.Length, vkBatched.Length);
        AssertClose(vkPerToken.Keys, vkBatched.Keys, VkTol, "K (vk per-token vs batched)");
        AssertClose(vkPerToken.Values, vkBatched.Values, VkTol, "V (vk per-token vs batched)");

        // Against the CPU oracle the Vulkan cache drifts by up to ~5e-3 BEFORE this change (the
        // trunk's captured rows already differ across backends; K-norm makes K rows O(1)). Batching
        // must not add to that: its distance from the oracle may exceed the per-token path's only
        // by the Vulkan-vs-Vulkan bound, and both stay under a sanity ceiling far below the O(1)
        // error a mispairing produces.
        float perTokenDrift = Math.Max(MaxDiff(cpu.Keys, vkPerToken.Keys), MaxDiff(cpu.Values, vkPerToken.Values));
        float batchedDrift = Math.Max(MaxDiff(cpu.Keys, vkBatched.Keys), MaxDiff(cpu.Values, vkBatched.Values));
        _out.WriteLine($"max |cpu - vk|: per-token {perTokenDrift:E3}, batched {batchedDrift:E3}");
        Assert.True(batchedDrift <= perTokenDrift + VkTol,
            $"batched absorb drifts further from the CPU oracle ({batchedDrift:E3}) than per-token ({perTokenDrift:E3})");
        Assert.True(batchedDrift <= CpuDriftCeiling, $"CPU-oracle drift {batchedDrift:E3} > {CpuDriftCeiling:E3}");
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private sealed record Snapshot(int Length, float[] Keys, float[] Values);

    private static readonly int[][] Batches = [[1, 2, 3], [4, 5, 6, 7]];
    private static readonly int[] Verify = [8, 10, 11];

    private static Snapshot RunCpu(string path)
    {
        MtpAbsorbDispatch.PerTokenOverride = true;
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using var state = (CpuMtpState)model.CreateMtpState()!;
            int p = Drive(model, kv, state);
            unsafe
            {
                return new Snapshot(state.CurrentLength,
                    new ReadOnlySpan<float>(state.KeyCachePtr, p * KvStride).ToArray(),
                    new ReadOnlySpan<float>(state.ValueCachePtr, p * KvStride).ToArray());
            }
        }
        finally
        {
            MtpAbsorbDispatch.PerTokenOverride = null;
        }
    }

    private static Snapshot RunVulkan(string path, string spvDir, bool perToken)
    {
        MtpAbsorbDispatch.PerTokenOverride = perToken;
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
            using var kv = model.CreateKvCache(config.MaxSequenceLength);
            using var state = (VulkanMtpState)model.CreateMtpState()!;
            int p = Drive(model, kv, state);

            var all = new float[state.MaxSteps * KvStride];
            device.Download(state.KeyCache, all);
            var keys = all.AsSpan(0, p * KvStride).ToArray();
            device.Download(state.ValueCache, all);
            var values = all.AsSpan(0, p * KvStride).ToArray();
            return new Snapshot(state.CurrentLength, keys, values);
        }
        finally
        {
            MtpAbsorbDispatch.PerTokenOverride = null;
        }
    }

    private static int Drive(DotLLM.Core.Models.IModel model, DotLLM.Core.Attention.IKvCache kv,
                             DotLLM.Core.Models.IMtpState state)
    {
        int p = 0;
        foreach (int[] batch in Batches)
        {
            int[] pos = Enumerable.Range(p, batch.Length).ToArray();
            using (ITensor _ = model.Forward(batch, pos, deviceId: -1, kv, adapter: null, state)) { }
            p += batch.Length;
        }
        // Draft three speculative slots, then verify over the same positions: the absorb must
        // overwrite them rather than append.
        for (int i = 0; i < 3; i++)
            using (ITensor _ = model.ForwardMtp(state, 9 - i, p + i)) { }
        Assert.Equal(p + 3, state.CurrentLength);
        int[] vpos = Enumerable.Range(p, Verify.Length).ToArray();
        using (ITensor _ = model.Forward(Verify, vpos, deviceId: -1, kv, adapter: null, state)) { }
        p += Verify.Length;
        Assert.Equal(p, state.CurrentLength);
        return p;
    }

    private static void AssertClose(float[] expected, float[] actual, float tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what}[{i}] is not finite");
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"{what}[{i}]: expected {expected[i]} vs actual {actual[i]}");
        }
    }
}
