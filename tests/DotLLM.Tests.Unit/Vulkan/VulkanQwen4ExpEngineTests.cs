using DotLLM.Models.Gguf;
using DotLLM.Core.Models;
using DotLLM.Tests.Unit.Models.Qwen4Exp;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Engine integration of the Vulkan Qwen4-Exp model (#871): per-sequence state (<c>CreateSequenceState</c>), <c>ForwardBatch</c> with the QSA
/// K/V rows in an engine KV cache, opt-in all-row logits (perplexity), and the clear dense-attention-limit failure. Synthetic checkpoints
/// against the CPU oracle, like <see cref="VulkanQwen4ExpParityTests"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpEngineTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpEngineTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void AllRowLogits_OptIn_MatchOracleRowByRow()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(0), spvDir);
        int T = 37;     // not a multiple of the 32-row LM-head chunk
        var ids = VulkanQwen4ExpParityTests.Ids(T, rig.Config.VocabSize, seed: 9);
        var pos = Enumerable.Range(0, T).ToArray();

        Assert.Equal(1, rig.Vk.MaxAllRowLogitsLength);
        using (var lastOnly = rig.Vk.Forward(ids, pos, -1))
            Assert.Equal(1, lastOnly.Shape[0]);                  // default: last row only
        rig.Vk.ResetSequenceState();

        Assert.True(rig.Vk.TrySetAllRowLogitsLimit(T + 1));
        using var all = rig.Vk.Forward(ids, pos, -1);
        using var cpu = rig.Cpu.Forward(ids, pos, -1);
        Assert.Equal(T, all.Shape[0]);
        for (int r = 0; r < T; r++)
        {
            var (rl2, kl, _) = VulkanQwen4ExpParityTests.Compare(Q4eRig.Row(cpu, r), Q4eRig.Row(all, r));
            Assert.True(rl2 < 3e-3 && kl < 1e-4, $"row {r}: relL2 {rl2:E3}, KL {kl:E3}");
        }

        // The last-row hint still short-circuits the head over a long chunk.
        rig.Vk.ResetSequenceState();
        using var hinted = rig.Vk.Forward(ids, pos, -1, null, lastTokenLogitsOnly: true);
        Assert.Equal(1, hinted.Shape[0]);
    }

    [SkippableFact]
    public void InterleavedSequences_ThroughForwardBatch_EqualSeparateRunsAndOracle()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(0), spvDir);
        Assert.True(rig.Vk.SupportsThreadedSequenceState);
        int V = rig.Config.VocabSize;
        var a = VulkanQwen4ExpParityTests.Ids(20, V, seed: 1);
        var b = VulkanQwen4ExpParityTests.Ids(14, V, seed: 2);

        // Oracle: each sequence alone on a fresh model-owned state.
        rig.Cpu.ResetSequenceState();
        var cpuA = Q4eRig.Row(rig.Cpu.Forward(a, Enumerable.Range(0, a.Length).ToArray(), -1), a.Length - 1);
        rig.Cpu.ResetSequenceState();
        var cpuB = Q4eRig.Row(rig.Cpu.Forward(b, Enumerable.Range(0, b.Length).ToArray(), -1), b.Length - 1);

        using var sa = (VulkanQwen4ExpSequenceState)rig.Vk.CreateSequenceState()!;
        using var sb = (VulkanQwen4ExpSequenceState)rig.Vk.CreateSequenceState()!;
        using var ka = rig.Vk.CreateKvCache(64);
        using var kb = rig.Vk.CreateKvCache(64);

        // Interleave: A[0..10) B[0..7) A[10..20) B[7..14) - as separate chunks in alternating batches.
        float[] la = [], lb = [];
        void Step(VulkanQwen4ExpSequenceState s, DotLLM.Core.Attention.IKvCache k, int[] ids, int from, int to, bool isA)
        {
            var req = new SequenceForwardRequest
            {
                TokenIds = ids.AsMemory(from, to - from), Positions = Enumerable.Range(from, to - from).ToArray(), KvCache = k, GdnState = s,
            };
            // Two requests in one call to exercise the multi-seq path: the other sequence's current chunk rides along below.
            using var r = rig.Vk.ForwardBatch([req], -1)[0];
            var row = Q4eRig.Row(r, 0);
            if (isA) la = row; else lb = row;
        }
        Step(sa, ka, a, 0, 10, true);
        Step(sb, kb, b, 0, 7, false);
        Step(sa, ka, a, 10, 20, true);
        Step(sb, kb, b, 7, 14, false);

        var (ra, kla, _) = VulkanQwen4ExpParityTests.Compare(cpuA, la);
        var (rb, klb, _) = VulkanQwen4ExpParityTests.Compare(cpuB, lb);
        _out.WriteLine($"interleaved A relL2 {ra:E3} KL {kla:E3}; B relL2 {rb:E3} KL {klb:E3}");
        Assert.True(ra < 3e-3 && rb < 3e-3, $"interleaved sequences diverge from the oracle: A {ra:E3}, B {rb:E3}");
        Assert.Equal(20, sa.Length); Assert.Equal(14, sb.Length);
        Assert.Equal(20, ka.CurrentLength); Assert.Equal(14, kb.CurrentLength);

        // A multi-sequence batch needs each request's own state.
        var bad = new SequenceForwardRequest { TokenIds = new[] { 1 }, Positions = new[] { 20 }, KvCache = ka, GdnState = sa };
        var noState = new SequenceForwardRequest { TokenIds = new[] { 1 }, Positions = new[] { 14 }, KvCache = kb };
        Assert.Throws<ArgumentException>(() => rig.Vk.ForwardBatch([bad, noState], -1));
    }

    [SkippableFact]
    public void PastTheContextCapacity_FailsWithAClearMessage_AndTheDenseLimitIsNoLongerAWall()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const string Env = "DOTLLM_VK_QWEN4EXP_CONTEXT";
        string? old = Environment.GetEnvironmentVariable(Env);
        Environment.SetEnvironmentVariable(Env, "40");
        try
        {
            using var rig = new Q4eRig(SyntheticQwen4ExpGguf.Build(), spvDir);
            int dense = rig.Vk.DenseContextLimit;
            Assert.Equal(40, rig.Vk.ContextCapacity);
            Assert.True(dense < 40);
            // Past the dense limit but inside the capacity: sparse QSA runs (#819) instead of refusing.
            var ok = VulkanQwen4ExpParityTests.Ids(dense + 5, rig.Config.VocabSize);
            using (rig.Vk.Forward(ok, Enumerable.Range(0, ok.Length).ToArray(), -1)) { }
            // Past the capacity: a clear message naming the knob.
            var ids = VulkanQwen4ExpParityTests.Ids(41, rig.Config.VocabSize);
            var ex = Assert.Throws<NotSupportedException>(() => rig.Vk.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1));
            Assert.Contains(Env, ex.Message);
            Assert.Contains("40", ex.Message);
            // The engine KV cache is clamped to the capacity.
            using var kv = rig.Vk.CreateKvCache(1_000_000);
            Assert.Equal(40, kv.MaxLength);
        }
        finally { Environment.SetEnvironmentVariable(Env, old); }
    }
}
