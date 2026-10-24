using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Speculative decoding on the Vulkan <c>nemotron_h_moe</c> host (issue #1237), on a synthetic trunk in Nemotron-3.5 Bolt's formats
/// (NVFP4 + scale SSM projections, IQ2_XXS experts and shared expert, Q8_0 latent/attention/head, F32 router): a small batch is
/// bit-identical to the same rows decoded one at a time, the recurrent checkpoint rewinds exactly, and greedy speculation with a
/// LiLiCorr drafter (on the device) is token-identical to plain greedy decoding.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNemotronHDFlashTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-vk-lilicorr-{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _out;
    private readonly List<VulkanDevice> _devices = [];
    private static string _spvDir = string.Empty;

    public VulkanNemotronHDFlashTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        foreach (var d in _devices) d.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    private (IModel, IKvCache) Vk(GgufFile gguf, ModelConfig config)
    {
        var device = VulkanDevice.Create();
        _devices.Add(device);
        var m = VulkanNemotronHTransformerModel.BuildFromGguf(device, gguf, config, _spvDir);
        return (m, m.CreateKvCache(config.MaxSequenceLength));
    }

    private static unsafe float[] Rows(ITensor t, int rows, int vocab)
    {
        Assert.Equal(rows, t.Shape[0]);
        return new ReadOnlySpan<float>((void*)t.DataPointer, rows * vocab).ToArray();
    }

    /// <summary>
    /// The exactness core: after a prompt, an 8-row forward (all-row logits) gives, row for row, the same bits as 8 decode steps
    /// on a fresh state — through the column GEMVs (NVFP4, IQ2_XXS, Q8_0, F32), the row-at-a-time attention, the indexed experts,
    /// the scan and the LM head. A tolerance would hide exactly the drift that flips a greedy argmax.
    /// </summary>
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]   // negative control: the prefill kernels are NOT decode-identical, so the comparison discriminates
    public void SmallBatch_EveryRow_IsBitIdenticalToDecode(bool rowExact)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _spvDir);
        string trunk = SyntheticBoltLiLiCorrGguf.WriteTarget(Path.Combine(_dir, "trunk.gguf"));
        using var gguf = GgufFile.Open(trunk);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (model, kv) = Vk(gguf, config);
        using var _m = (IDisposable)model;
        using var _k = kv;
        int vocab = config.VocabSize;
        Assert.True(((VulkanNemotronHTransformerModel)model).TrySetAllRowLogitsLimit(16));

        int[] prompt = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8];   // 19 > 16: a batched prefill
        int[] rows = [2, 7, 1, 8, 2, 8, 1, 8];
        int[] P(int start, int n) => Enumerable.Range(start, n).ToArray();

        using (model.Forward(prompt, P(0, prompt.Length), -1, kv)) { }
        float[] batched;
        VulkanNemotronHTransformerModel.RowExactEnabled = rowExact;
        try
        {
            using var l = model.Forward(rows, P(prompt.Length, rows.Length), -1, kv);
            batched = Rows(l, rows.Length, vocab);
        }
        finally { VulkanNemotronHTransformerModel.RowExactEnabled = true; }

        model.ResetSequenceState();
        kv.Rollback(0);
        using (model.Forward(prompt, P(0, prompt.Length), -1, kv)) { }
        int differing = 0;
        for (int r = 0; r < rows.Length; r++)
        {
            float[] one;
            using (var l = model.Forward([rows[r]], [prompt.Length + r], -1, kv)) one = Rows(l, 1, vocab);
            for (int v = 0; v < vocab; v++)
            {
                bool same = BitConverter.SingleToInt32Bits(one[v]) == BitConverter.SingleToInt32Bits(batched[r * vocab + v]);
                if (!same) differing++;
                Assert.True(same || !rowExact, $"row {r} logit {v}: decode {one[v]:R} vs batch {batched[r * vocab + v]:R}");
            }
        }
        _out.WriteLine($"rowExact={rowExact}: {differing} of {rows.Length * vocab} logits differ from decode");
        if (!rowExact) Assert.True(differing > 0, "the prefill kernels matched decode bit for bit: the test would not catch a regression");
    }

    /// <summary>Checkpoint before a forward, restore: the replayed forward equals the original, bit for bit.</summary>
    [SkippableFact]
    public void Checkpoint_Restore_RewindsTheMambaState()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _spvDir);
        string trunk = SyntheticBoltLiLiCorrGguf.WriteTarget(Path.Combine(_dir, "trunk.gguf"));
        using var gguf = GgufFile.Open(trunk);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (model, kv) = Vk(gguf, config);
        using var _m = (IDisposable)model;
        using var _k = kv;
        ((VulkanNemotronHTransformerModel)model).TrySetAllRowLogitsLimit(8);
        int vocab = config.VocabSize;
        using (model.Forward([1, 2, 3], [0, 1, 2], -1, kv)) { }
        object? ckpt = model.CheckpointRecurrentState();
        float[] first;
        using (var l = model.Forward([4, 5, 6], [3, 4, 5], -1, kv)) first = Rows(l, 3, vocab);
        model.RestoreRecurrentState(ckpt);
        kv.Rollback(3);
        float[] again;
        using (var l = model.Forward([4, 5, 6], [3, 4, 5], -1, kv)) again = Rows(l, 3, vocab);
        Assert.Equal(first, again);
        // And the checkpoint really captured something: without the restore the state has moved on.
        kv.Rollback(3);
        using (var l = model.Forward([4, 5, 6], [3, 4, 5], -1, kv)) Assert.NotEqual(first, Rows(l, 3, vocab));
    }

    /// <summary>
    /// The per-row snapshot rollback: a 6-row snapshot forward, roll back to row 2 (scan replay from the pre-verify state), then
    /// the next rows give the same bits as a run that only ever saw rows 0..2 (with the conv window and SSM state both rewound).
    /// </summary>
    [SkippableFact]
    public void RowSnapshot_RestoreToRow_EqualsTheShorterRun()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _spvDir);
        string trunk = SyntheticBoltLiLiCorrGguf.WriteTarget(Path.Combine(_dir, "trunk.gguf"));
        using var gguf = GgufFile.Open(trunk);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (model, kv) = Vk(gguf, config);
        using var _m = (IDisposable)model;
        using var _k = kv;
        ((VulkanNemotronHTransformerModel)model).TrySetAllRowLogitsLimit(8);
        Assert.True(model.SupportsRecurrentRowSnapshots);
        int vocab = config.VocabSize;
        int[] prompt = [1, 2, 3, 4];
        int[] verify = [9, 8, 7, 6, 5, 4];
        int[] next = [11, 12, 13];
        int[] P(int start, int n) => Enumerable.Range(start, n).ToArray();

        using (model.Forward(prompt, P(0, 4), -1, kv)) { }
        using (model.ForwardWithRecurrentSnapshots(verify, P(4, 6), -1, kv, null)) { }
        model.RestoreRecurrentStateToRow(2);
        kv.Rollback(7);
        float[] got;
        using (var l = model.Forward(next, P(7, 3), -1, kv)) got = Rows(l, 3, vocab);

        model.ResetSequenceState();
        kv.Rollback(0);
        using (model.Forward(prompt, P(0, 4), -1, kv)) { }
        using (model.Forward(verify[..3], P(4, 3), -1, kv)) { }
        float[] want;
        using (var l = model.Forward(next, P(7, 3), -1, kv)) want = Rows(l, 3, vocab);
        Assert.Equal(want, got);
        _out.WriteLine($"row-snapshot scratch {((VulkanNemotronHTransformerModel)model).RecurrentRowSnapshotBytes} B, checkpoint {((VulkanNemotronHTransformerModel)model).RecurrentCheckpointBytes} B");
    }

    /// <summary>The end-to-end gate on the Vulkan host: GPU verify, GPU drafter (sinks, scales), host lattice head.</summary>
    [SkippableFact]
    public void SpeculativeDecode_WithLiLiCorr_IsTokenIdenticalToGreedy()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _spvDir);
        LiLiCorrDrafterTests.AssertTokenIdentical(_dir, Vk, _out);
    }
}

/// <summary>Sends short Nemotron-H batches down the prefill kernels for the duration of a test that checks those kernels (#1237).</summary>
internal sealed class RowExactOffScope : IDisposable
{
    public RowExactOffScope() => VulkanNemotronHTransformerModel.RowExactEnabled = false;

    public void Dispose() => VulkanNemotronHTransformerModel.RowExactEnabled = true;
}
