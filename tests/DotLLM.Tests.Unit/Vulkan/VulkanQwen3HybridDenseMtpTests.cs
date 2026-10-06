using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Vulkan MTP ("NextN") parity against the CPU oracle — issue #435.
/// </summary>
/// <remarks>
/// <para>
/// The CPU host (<see cref="Qwen3HybridDenseTransformerModel.ForwardMtp"/>) is the reference
/// implementation and has been exercised end-to-end since #253, so these tests assert Vulkan
/// against it rather than against hand-computed expectations. <b>Draft token IDs must match
/// exactly</b> — an argmax over the same weights is the quantity speculative decoding actually
/// consumes, and any disagreement there is a correctness bug, not numerical drift. The logits
/// themselves are compared with a tolerance: the two backends reduce the same GEMM in a different
/// order.
/// </para>
/// <para>
/// Both fixture arms are covered because they take structurally different paths: with head-local
/// <c>nextn.*</c> tensors the MTP head embeds from and projects through its own matrices, and
/// without them it falls back to the trunk's <c>token_embd</c> / <c>output</c> — the arm that, on
/// a Hadamard-folded checkpoint, has to apply the trunk's own rotations. The synthetic fixture
/// carries no fold, so this pins the fallback plumbing; the rotation arms themselves need the real
/// Bonsai 2 MTP checkpoint (see the issue report).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseMtpTests : IDisposable
{
    // The token after the prompt the first draft step starts from (issue #469).
    private const int NextToken = 4;

    private const int VocabSize = SyntheticQwen35HybridDenseMtpGguf.VocabSize;

    // Loose enough for a different GEMM reduction order, tight enough that a wrong tensor,
    // a missing norm or a mis-indexed KV row fails by orders of magnitude.
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;
    private readonly Xunit.Abstractions.ITestOutputHelper _out;

    public VulkanQwen3HybridDenseMtpTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-qwen35-mtp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string WriteFixture(bool withMtp, bool mtpHasOwnHeadTensors = true, string name = "qwen35-mtp.gguf")
        => SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, name), withMtp: withMtp, mtpHasOwnHeadTensors: mtpHasOwnHeadTensors);

    /// <summary>
    /// The capability surface <c>MtpSpeculativeDecoder</c> gates on. Before #435 the Vulkan model
    /// reported <c>SupportsMtp == false</c> for a checkpoint that plainly carries an MTP head, and
    /// <c>SupportsRecurrentStateCheckpoint == false</c> — which would have let a rejected draft
    /// silently corrupt the trunk's GDN recurrence.
    /// </summary>
    [SkippableFact]
    public void Model_WithMtpCheckpoint_ExposesMtpAndRecurrentCheckpoint()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);

        Assert.True(model.SupportsMtp);
        Assert.True(model.SupportsRecurrentStateCheckpoint);

        using IMtpState? state = model.CreateMtpState();
        var vkState = Assert.IsType<VulkanMtpState>(state);
        Assert.Equal(config.HiddenSize, vkState.HiddenSize);
        Assert.Equal(0, vkState.CurrentLength);

        // The checkpoint is opaque and caller-disposed (pooled since #469); a second one taken after
        // disposing the first must work too, since it reuses the returned buffers.
        object? checkpoint = model.CheckpointRecurrentState();
        Assert.IsAssignableFrom<IDisposable>(checkpoint);
        model.RestoreRecurrentState(checkpoint);
        ((IDisposable)checkpoint!).Dispose();
        object? second = model.CheckpointRecurrentState();
        model.RestoreRecurrentState(second);
        ((IDisposable)second!).Dispose();
    }

    /// <summary>A checkpoint without an MTP head must be completely unaffected.</summary>
    [SkippableFact]
    public void Model_WithoutMtpCheckpoint_ReportsNoMtpSupport()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: false, name: "qwen35-nomtp.gguf");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);

        Assert.False(model.SupportsMtp);
        Assert.Null(model.CreateMtpState());
    }

    /// <summary>
    /// The verify phase indexes one logit row per drafted position, so a short batch must come
    /// back as <c>[seq, vocab]</c>. Vulkan previously returned <c>[1, vocab]</c> for every batch
    /// length, which would have made every verify comparison read row 0's logits.
    /// </summary>
    [SkippableFact]
    public void Forward_ShortBatch_ReturnsOneLogitRowPerPosition_MatchingCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true);

        int[] tokenIds = [1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3];

        float[] cpuLogits;
        int cpuRows;
        using (var gguf = GgufFile.Open(path))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv);
            cpuRows = logits.Shape[0];
            cpuLogits = Copy(logits, cpuRows * VocabSize);
        }

        Assert.Equal(tokenIds.Length, cpuRows);

        float[] vkLogits;
        using (var gguf = GgufFile.Open(path))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
            using var kv = model.CreateKvCache(config.MaxSequenceLength);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv);
            Assert.Equal(tokenIds.Length, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            vkLogits = Copy(logits, tokenIds.Length * VocabSize);
        }

        AssertClose(cpuLogits, vkLogits, "all-row logits");
    }

    /// <summary>
    /// The all-row LM head deviates from <c>IModel</c>'s <c>[seq, vocab]</c> contract above
    /// <see cref="VulkanQwen3HybridDenseTransformerModel.MaxAllRowLogitsSeqLen"/>, and this pins
    /// the model declaring that honestly through <see cref="IModel.MaxAllRowLogitsLength"/>.
    /// </summary>
    /// <remarks>
    /// Not bookkeeping — this is the regression test for a heap over-read. <c>BackendPerplexityModel.Probe</c>
    /// decides between the single-pass and growing-prefix perplexity strategies from a <b>two-token</b>
    /// forward. Two tokens is under the threshold, so the probe sees two rows, would conclude "this
    /// backend returns every row", and the single-pass evaluator would then index rows 1..n-1 of a
    /// buffer holding exactly one row at a 512- or 2048-token context — past the end of the
    /// allocation, reporting a fabricated perplexity rather than throwing. The probe therefore has
    /// to consult the declared bound as well as measure, and this asserts the bound is declared.
    /// </remarks>
    [SkippableFact]
    public void Model_DeclaresItsAllRowLogitsBound()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        // The default fixture's context (16) equals the bound, so it cannot express "one token past
        // it". Build this one with a longer context so both sides of the bound are reachable.
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-mtp-bound.gguf"), withMtp: true, contextLength: 64);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);

        IModel asInterface = model;
        Assert.NotEqual(int.MaxValue, asInterface.MaxAllRowLogitsLength);
        Assert.Equal(VulkanQwen3HybridDenseTransformerModel.MaxAllRowLogitsSeqLen,
            asInterface.MaxAllRowLogitsLength);

        // And the bound must actually describe the behaviour: at the threshold, every row; one
        // past it, a single row. A declared bound that does not match is worse than none.
        int atBound = asInterface.MaxAllRowLogitsLength;
        Assert.True(atBound + 1 <= config.MaxSequenceLength, "Fixture context is too short for this check.");

        var tokens = new int[atBound + 1];
        var positions = new int[atBound + 1];
        for (int i = 0; i < tokens.Length; i++) { tokens[i] = i % VocabSize; positions[i] = i; }

        using (var kv = model.CreateKvCache(config.MaxSequenceLength))
        using (ITensor logits = model.Forward(tokens.AsSpan(0, atBound), positions.AsSpan(0, atBound),
                                              deviceId: -1, kv))
        {
            Assert.Equal(atBound, logits.Shape[0]);
        }

        model.ResetSequenceState();
        using (var kv = model.CreateKvCache(config.MaxSequenceLength))
        using (ITensor logits = model.Forward(tokens, positions, deviceId: -1, kv))
        {
            Assert.Equal(1, logits.Shape[0]);
        }
    }

    /// <summary>
    /// The acceptance criterion: the same draft tokens, at the same positions, from the same
    /// checkpoint. Runs the full round shape <c>MtpSpeculativeDecoder</c> uses — a trunk forward
    /// that captures the pre-final-norm hidden state, a seed from the captured row, then K
    /// autoregressive head steps chained through their own argmax.
    /// </summary>
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void ForwardMtp_DraftTokens_MatchCpuOracle(bool mtpHasOwnHeadTensors)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true, mtpHasOwnHeadTensors: mtpHasOwnHeadTensors,
            name: $"qwen35-mtp-own{mtpHasOwnHeadTensors}.gguf");

        int[] tokenIds = [1, 2, 3];
        int[] positions = [0, 1, 2];
        const int DraftSteps = 4;

        var (cpuTokens, cpuLogits) = RunCpuDraft(path, tokenIds, positions, DraftSteps);
        var (vkTokens, vkLogits) = RunVulkanDraft(path, spvDir, tokenIds, positions, DraftSteps);

        Assert.Equal(cpuTokens, vkTokens);
        for (int i = 0; i < DraftSteps; i++)
            AssertClose(cpuLogits[i], vkLogits[i], $"draft step {i}");
    }

    private static (int[] tokens, float[][] logits) RunCpuDraft(
        string path, int[] tokenIds, int[] positions, int draftSteps)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;

        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kv, adapter: null, mtpState)) { }

        var tokens = new int[draftSteps];
        var logits = new float[draftSteps][];
        // Prefill absorbed positions 0..n-1 (issue #469); drafting starts at position n.
        int draftToken = NextToken;
        for (int i = 0; i < draftSteps; i++)
        {
            using ITensor step = model.ForwardMtp(mtpState, draftToken, positions[^1] + 1 + i);
            logits[i] = Copy(step, VocabSize);
            draftToken = ArgMax(logits[i]);
            tokens[i] = draftToken;
        }
        return (tokens, logits);
    }

    private static (int[] tokens, float[][] logits) RunVulkanDraft(
        string path, string spvDir, int[] tokenIds, int[] positions, int draftSteps)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        using var kv = model.CreateKvCache(config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;

        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kv, adapter: null, mtpState)) { }
        Assert.Equal(tokenIds.Length, mtpState.CapturedRowCount);

        var tokens = new int[draftSteps];
        var logits = new float[draftSteps][];
        Assert.Equal(tokenIds.Length, mtpState.CurrentLength);   // the prefill absorbed the prompt
        int draftToken = NextToken;
        for (int i = 0; i < draftSteps; i++)
        {
            using ITensor step = model.ForwardMtp(mtpState, draftToken, positions[^1] + 1 + i);
            Assert.Equal(1, step.Shape[0]);
            Assert.Equal(VocabSize, step.Shape[1]);
            logits[i] = Copy(step, VocabSize);
            draftToken = ArgMax(logits[i]);
            tokens[i] = draftToken;
            Assert.Equal(tokenIds.Length + i + 1, mtpState.CurrentLength);
        }
        return (tokens, logits);
    }

    /// <summary>
    /// The end-to-end claim: <c>MtpSpeculativeDecoder</c> — completely unmodified by this work —
    /// driving the real Vulkan model must produce the exact same token sequence as plain greedy
    /// decode of that model alone. That is the whole point of self-speculation: same output, fewer
    /// trunk forwards.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses the DEFAULT mixed GDN + full-attention fixture, unlike the equivalent CUDA test which
    /// had to fall back to an all-full-attention trunk.
    /// </para>
    /// <para>
    /// <b>Shape.</b> The run mirrors <c>TextGenerator</c>: prefill <c>[start]</c> with the MTP state,
    /// sample <c>t1</c>, then rounds from <c>t1</c>'s own slot (issue #475). The helper asserts at
    /// every round entry that neither the trunk KV nor the head has seen <c>lastToken</c> yet — the
    /// only check here that catches a double-forwarded start token, because on this fixture one
    /// extra GDN step shifts logits by ~1.6e-3 (CPU, measured), which flips no argmax and sits
    /// inside the Vulkan tolerance. The bit-tight CPU counterpart is
    /// <c>MtpSpeculativeDecoderGdnStateTests.DraftAndVerify_PrefillThenRounds_LikeTextGenerator_NextLogitsMatchSerialReplay</c>.
    /// </para>
    /// <para>
    /// <b>What this covers.</b> The emitted-per-round histogram is (almost) all 1s:
    /// the MTP head and the trunk are independently random on a synthetic fixture, so the head
    /// agrees with the trunk's argmax at chance (1 in 12 here) and nearly every round rejects
    /// draft 1. Since #469 the verify batch runs every round regardless, so each round rolls the
    /// GDN state back to row 0 — through the #473 row snapshot by default. Rollback to rows 1 and
    /// up needs scripted drafts: see
    /// <see cref="DraftAndVerify_ScriptedPartialRejections_SnapshotsMatchGreedyAndReplay"/>. The
    /// checkpoint pair is covered by <see cref="CheckpointRestoreRecurrentState_RestoresGdnStateExactly"/>.
    /// </para>
    /// </remarks>
    [SkippableFact]
    public void DraftAndVerify_MatchesPlainGreedyDecode_OnRealVulkanModel()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true, name: "qwen35-mtp-e2e.gguf");

        const int StartToken = 1;
        const int TotalNewTokens = 10;
        const int K = 3;

        var run = RunSpeculative(path, spvDir, StartToken, TotalNewTokens, K);
        List<int> plain = RunPlainGreedy(path, spvDir, StartToken, TotalNewTokens);

        Assert.Equal(plain, run.Tokens);
        Assert.True(run.Drafted > 0, "The decoder must actually have drafted something.");
        Assert.True(run.Emitted > 0, "Every round emits at least the corrected/bonus token.");

        // On this fixture the emitted-per-round histogram comes back mostly 1s: the random MTP
        // head agrees with the trunk at chance (1/12), so nearly every round rejects draft 1.
        // Since #469 the verify batch still runs every round, so each of those rounds rolls the
        // GDN state back to row 0 (via the #473 row snapshot by default). Rollback to rows >= 1
        // is covered by DraftAndVerify_ScriptedPartialRejections_SnapshotsMatchGreedyAndReplay,
        // and the checkpoint pair by CheckpointRestoreRecurrentState_RestoresGdnStateExactly.
        _out.WriteLine($"emitted-per-round histogram: [{string.Join(",", run.EmittedPerRound)}]  " +
                       $"(K={K}, drafted={run.Drafted}, emitted={run.Emitted}, " +
                       $"post-verify rejections={run.PartialRejectionRounds})");
    }

    /// <summary>
    /// The deterministic test for the recurrent-state checkpoint pair added in #435, and the reason
    /// speculative decoding is safe against a GDN trunk on Vulkan at all.
    /// </summary>
    /// <remarks>
    /// Structure matters here. Asserting only "restore then re-run reproduces the first run" would
    /// pass against a <b>no-op</b> restore whenever the forward happens not to depend on the
    /// recurrent state. So this asserts both halves: the restored run must match bit-for-bit, AND
    /// a run from the <i>mutated</i> state must differ. The second half is what makes the first
    /// half mean something — it proves the GDN state genuinely affects this forward, so matching
    /// after a restore is evidence the state really was rolled back.
    /// </remarks>
    [SkippableFact]
    public void CheckpointRestoreRecurrentState_RestoresGdnStateExactly()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true, name: "qwen35-mtp-ckpt.gguf");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        Assert.True(model.SupportsRecurrentStateCheckpoint);

        using var kv = model.CreateKvCache(config.MaxSequenceLength);

        // Advance the GDN recurrence with a prefix, then snapshot it.
        using (ITensor _ = model.Forward([1, 2, 3], [0, 1, 2], deviceId: -1, kv)) { }
        object? checkpoint = model.CheckpointRecurrentState();
        Assert.NotNull(checkpoint);

        try
        {
            // First continuation from the snapshot point.
            float[] first;
            using (ITensor logits = model.Forward([4], [3], deviceId: -1, kv))
                first = Copy(logits, VocabSize);

            // A SECOND continuation without restoring: the GDN state has moved on, so this must
            // differ. If it does not, the forward is insensitive to the recurrent state and the
            // restore assertion below would be vacuous.
            float[] mutated;
            kv.Rollback(3);
            using (ITensor logits = model.Forward([4], [3], deviceId: -1, kv))
                mutated = Copy(logits, VocabSize);

            bool anyDifference = false;
            for (int i = 0; i < VocabSize && !anyDifference; i++)
                anyDifference = first[i] != mutated[i];
            Assert.True(anyDifference,
                "Re-running the same token from a MUTATED recurrent state produced identical logits, " +
                "so this forward does not depend on GDN state and the restore assertion below would " +
                "prove nothing. The test needs a different shape.");

            // Now restore and re-run: must reproduce the first continuation exactly. Bit-exactness
            // is fair to demand — identical inputs, identical kernels, identical dispatch order.
            kv.Rollback(3);
            model.RestoreRecurrentState(checkpoint);
            float[] restored;
            using (ITensor logits = model.Forward([4], [3], deviceId: -1, kv))
                restored = Copy(logits, VocabSize);

            for (int i = 0; i < VocabSize; i++)
                Assert.True(first[i] == restored[i],
                    $"index {i}: before={first[i]:R} after restore={restored[i]:R} — " +
                    "VulkanGdnStateCache.CopyTo did not reproduce the snapshot exactly.");

            // The checkpoint must survive being restored FROM: the decoder holds one object across
            // a restore-and-replay and only disposes it at the end of the round. A CopyTo that
            // consumed or aliased its source would pass the first restore and fail here.
            kv.Rollback(3);
            model.RestoreRecurrentState(checkpoint);
            using (ITensor logits = model.Forward([4], [3], deviceId: -1, kv))
            {
                float[] restoredAgain = Copy(logits, VocabSize);
                for (int i = 0; i < VocabSize; i++)
                    Assert.True(first[i] == restoredAgain[i],
                        $"index {i}: second restore from the SAME checkpoint diverged " +
                        $"({first[i]:R} vs {restoredAgain[i]:R}) — the checkpoint is not reusable.");
            }
        }
        finally
        {
            (checkpoint as IDisposable)?.Dispose();
        }
    }

    /// <summary>
    /// Issue #473, row selection on Vulkan: restoring the GDN snapshot of verify row <c>n</c> and
    /// probing the next token must match a checkpoint restore + replay of rows <c>0..n</c>, and must
    /// miss the replay of rows <c>n±1</c> — so a snapshot index off by one (state or conv window)
    /// fails. Tolerance, not bit-equality: the replay is a shorter batch, and the matmul pipeline
    /// is chosen by batch width (#470), which contracts FMAs differently.
    /// </summary>
    [SkippableFact]
    public void RestoreRecurrentStateToRow_MatchesPrefixReplay_AndNotItsNeighbours()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true, name: "qwen35-mtp-rowsnap.gguf");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        Assert.True(model.SupportsRecurrentRowSnapshots);

        float[][] snap = Engine.MtpRecurrentRowSnapshotTests.RowProbes(
            model, config, useSnapshots: true, kvFactory: () => model.CreateKvCache(config.MaxSequenceLength));
        float[][] replay = Engine.MtpRecurrentRowSnapshotTests.RowProbes(
            model, config, useSnapshots: false, kvFactory: () => model.CreateKvCache(config.MaxSequenceLength));
        Engine.MtpRecurrentRowSnapshotTests.AssertRowSelection(snap, replay, AbsTol, RelTol, _out);
        _out.WriteLine($"snapshot scratch: {model.RecurrentRowSnapshotBytes} bytes for 3 rows");
    }

    /// <summary>
    /// Issue #473, decoder level on Vulkan: drafts scripted so rounds roll back to rows 0, 1 and 2
    /// of K=3 (and one clean round). Snapshots on and off must both reproduce plain greedy decode,
    /// restore exactly where the other replays, and leave next-token logits within tolerance.
    /// </summary>
    [SkippableFact]
    public void DraftAndVerify_ScriptedPartialRejections_SnapshotsMatchGreedyAndReplay()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true, name: "qwen35-mtp-rowsnap-dec.gguf");
        const int StartToken = 3, NewTokens = 12, K = 3;

        List<int> greedy = RunPlainGreedy(path, spvDir, StartToken, NewTokens + K);
        var on = RunScripted(path, spvDir, greedy, StartToken, NewTokens, K, snapshots: true);
        var off = RunScripted(path, spvDir, greedy, StartToken, NewTokens, K, snapshots: false);
        _out.WriteLine($"greedy: [{string.Join(",", greedy.Take(NewTokens + 1))}]");
        _out.WriteLine($"on : emitted/round [{string.Join(",", on.PerRound)}] avoided={on.Avoided} replays={on.Replays}");
        _out.WriteLine($"off: emitted/round [{string.Join(",", off.PerRound)}] avoided={off.Avoided} replays={off.Replays}");

        Assert.Equal(greedy.Take(NewTokens + 1), on.Tokens);
        Assert.Equal(greedy.Take(NewTokens + 1), off.Tokens);
        for (int emitted = 1; emitted <= K + 1; emitted++)
            Assert.Contains(emitted, on.PerRound);
        Assert.True(on.Avoided > 0 && on.Replays == 0, "snapshot run must roll back without replaying");
        Assert.True(off.Avoided == 0 && off.Replays == on.Avoided);
        AssertClose(off.NextLogits, on.NextLogits, "next-token logits, snapshots vs replay");
    }

    private sealed record ScriptedRun(List<int> Tokens, List<int> PerRound, int Avoided, int Replays, float[] NextLogits);

    private static ScriptedRun RunScripted(
        string path, string spvDir, List<int> greedy, int start, int newTokens, int k, bool snapshots)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var real = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        var model = new Engine.ScriptedDraftMtpModel(real, greedy, wrongAt: p => p is 1 or 3 or 6 or 11 or 13);
        using var kv = real.CreateKvCache(config.MaxSequenceLength);
        using var mtp = model.CreateMtpState()!;
        var decoder = new MtpSpeculativeDecoder(greedy: true) { UseRecurrentRowSnapshots = snapshots };
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        // No prefill: position 0 is the start token's own slot, and the first verify forwards it.
        var ids = new List<int> { start };
        var perRound = new List<int>();
        int position = 0;
        Span<int> outBuf = stackalloc int[k + 1];
        while (ids.Count - 1 < newTokens)
        {
            var r = decoder.DraftAndVerify(model, kv, mtp, pipeline, ids, null, position, config.VocabSize, k, outBuf);
            perRound.Add(r.AcceptedCount);
            for (int i = 0; i < r.AcceptedCount; i++) ids.Add(outBuf[i]);
            position += r.AcceptedCount;
        }

        using ITensor next = model.Forward([ids[position]], [position], deviceId: -1, kv);
        float[] nextRow = Engine.MtpRecurrentRowSnapshotTests.LastRow(next, config.VocabSize);
        return new ScriptedRun(ids.Take(newTokens + 1).ToList(), perRound, decoder.ReplaysAvoided, decoder.Replays, nextRow);
    }

    private sealed record SpeculativeRun(
        List<int> Tokens, int Drafted, int Emitted, int PartialRejectionRounds, List<int> EmittedPerRound);

    private static SpeculativeRun RunSpeculative(
        string path, string spvDir, int startToken, int totalNewTokens, int k)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        Assert.True(model.SupportsMtp);

        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        var generatedIds = new List<int> { startToken };
        using var kvCache = model.CreateKvCache(config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;

        // Mirror TextGenerator (issue #475): prefill the prompt [startToken] WITH the MTP state, so
        // the head absorbs it and is seeded from h_0, sample t1 from the prefill, and enter the round
        // loop at t1's own slot. Every trunk forward of an MTP sequence carries the state (#469), and
        // DraftAndVerify's verify forwards `lastToken` itself, so the round's lastToken must be one
        // the trunk has never seen. The old helper prefilled startToken without the state and then
        // started the loop at position 0 — the start token went through the GDN recurrence twice and
        // the first draft came from a never-seeded head, a path production never takes.
        using (ITensor prefill = model.Forward([startToken], [0], deviceId: -1, kvCache, adapter: null, mtpState))
        {
            int rows = prefill.Shape[0];
            generatedIds.Add(ArgMax(Copy(prefill, rows * config.VocabSize)
                .AsSpan((rows - 1) * config.VocabSize, config.VocabSize).ToArray()));
        }
        Assert.Equal(1, mtpState.CurrentLength);

        int position = 1, drafted = 0, emitted = 0, partialRejections = 0, guard = 0;
        var emittedPerRound = new List<int>();
        Span<int> outputBuffer = stackalloc int[k + 1];
        while (generatedIds.Count - 1 < totalNewTokens && guard++ < totalNewTokens * 4)
        {
            Engine.MtpSpeculativeDecoderGdnStateTests.AssertRoundEntryInvariants(kvCache, mtpState, position);
            var result = decoder.DraftAndVerify(
                model, kvCache, mtpState, pipeline, generatedIds,
                constraint: null, position, vocabSize: config.VocabSize, numCandidates: k, outputBuffer);

            Assert.True(result.AcceptedCount > 0, "Every round must emit at least the corrected/bonus token.");
            drafted += result.DraftedCount;
            emitted += result.AcceptedCount;
            emittedPerRound.Add(result.AcceptedCount);
            if (result.AcceptedCount >= 2 && result.AcceptedCount <= k)
                partialRejections++;

            for (int i = 0; i < result.AcceptedCount && generatedIds.Count - 1 < totalNewTokens; i++)
                generatedIds.Add(outputBuffer[i]);

            position += result.AcceptedCount;
        }

        return new SpeculativeRun(
            generatedIds.Take(totalNewTokens + 1).ToList(), drafted, emitted, partialRejections, emittedPerRound);
    }

    private static List<int> RunPlainGreedy(string path, string spvDir, int startToken, int totalNewTokens)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        using var kvCache = model.CreateKvCache(config.MaxSequenceLength);

        var ids = new List<int> { startToken };
        for (int step = 0; step < totalNewTokens; step++)
        {
            using ITensor logits = model.Forward([ids[^1]], [step], deviceId: -1, kvCache);
            int rows = logits.Shape[0];
            var last = Copy(logits, rows * config.VocabSize)
                .AsSpan((rows - 1) * config.VocabSize, config.VocabSize).ToArray();
            ids.Add(ArgMax(last));
        }
        return ids;
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what}: index {i} is not finite ({actual[i]}).");
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(diff <= bar,
                $"{what}: index {i}: cpu={expected[i]:F6} vs vulkan={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static float[] Copy(ITensor t, int count)
    {
        var copy = new float[count];
        unsafe
        {
            new ReadOnlySpan<float>((void*)t.DataPointer, count).CopyTo(copy);
        }
        return copy;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
