using System.Globalization;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Engine.Scheduler;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// Unit tests for <see cref="ContinuousBatchScheduler"/>. Uses a deterministic mock model that
/// emits a scripted token sequence per (prompt-last-token, step) so the test can verify
/// admission, decode iteration, eviction, and KV-cache release in isolation from any real model.
/// </summary>
public sealed class ContinuousBatchSchedulerTests
{
    private const int VocabSize = 32;
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int KvStride = NumKvHeads * HeadDim;
    private const int BlockSize = 4;
    private const int MaxSeqLen = 64;
    private const int EosTokenId = 0;

    [Fact]
    public void Submit_NewRequest_QueuedNotActive()
    {
        using var fix = new TestFixture();
        var req = MakeRequest(promptLen: 3, maxTokens: 4);

        var handle = fix.Scheduler.Submit(req);

        Assert.Equal(SequenceState.Queued, handle.State);
        Assert.False(fix.Scheduler.IsIdle);
        Assert.Equal(1, fix.Scheduler.QueueDepth);
        Assert.Equal(0, fix.Scheduler.ActiveCount);
    }

    [Fact]
    public async Task SingleSequence_RunsToCompletionOnEos()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 3)); // 3 normal tokens then EOS
        var req = MakeRequest(promptLen: 4, maxTokens: 16);

        var handle = fix.Scheduler.Submit(req);
        DriveUntilIdle(fix.Scheduler);

        var response = await handle.Completion;
        Assert.Equal(FinishReason.Stop, response.FinishReason);
        // 3 emitted tokens (EOS excluded per Stop semantics)
        Assert.Equal(3, response.GeneratedTokenCount);
        Assert.True(fix.Scheduler.IsIdle);
    }

    [Fact]
    public async Task FourSequences_DifferentPromptLengths_AllFinish()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 2));

        int[] promptLens = [2, 5, 7, 3];
        var handles = new ISchedulerRequest[promptLens.Length];

        for (int i = 0; i < promptLens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 16));

        DriveUntilIdle(fix.Scheduler);

        for (int i = 0; i < handles.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Stop, r.FinishReason);
            Assert.Equal(2, r.GeneratedTokenCount);
            Assert.Equal(promptLens[i], r.PromptTokenCount);
        }
        Assert.True(fix.Scheduler.IsIdle);
        Assert.Equal(0, fix.Scheduler.ActiveCount);
    }

    [Fact]
    public async Task Sequence_HitsMaxTokens_FinishReasonLength()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 7)); // emit 7 forever
        var req = MakeRequest(promptLen: 2, maxTokens: 5);

        var handle = fix.Scheduler.Submit(req);
        DriveUntilIdle(fix.Scheduler);

        var r = await handle.Completion;
        Assert.Equal(FinishReason.Length, r.FinishReason);
        Assert.Equal(5, r.GeneratedTokenCount);
    }

    [Fact]
    public async Task KvCacheBlocks_ReleasedAfterCompletion()
    {
        // Need 1 prompt token + 2 generated tokens = 3 positions → ceil(3/4) = 1 block
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 2));

        int totalBlocksBefore = fix.PagedPool.FreeBlocks;
        Assert.Equal(fix.PagedPool.TotalBlocks, totalBlocksBefore);

        for (int i = 0; i < 3; i++)
        {
            var handle = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 8));
            DriveUntilIdle(fix.Scheduler);
            await handle.Completion;
        }

        Assert.Equal(totalBlocksBefore, fix.PagedPool.FreeBlocks);
        Assert.True(fix.Scheduler.IsIdle);
    }

    [Fact]
    public void Backpressure_WhenCapacityExhausted_RequestStaysQueued()
    {
        // Cap to 1 active sequence. Submit 3 and verify only one runs at a time.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 5), // long-running, never EOS
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1 });

        var h1 = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 4));
        var h2 = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 4));
        var h3 = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 4));

        // First Step: admit h1, prefill, no further admission (active==1 cap).
        fix.Scheduler.Step();
        Assert.Equal(1, fix.Scheduler.ActiveCount);
        Assert.Equal(2, fix.Scheduler.QueueDepth);

        // Drain h1 by stepping decode until it completes (max-tokens=4 → 4 decode steps).
        DriveUntilCompleted(fix.Scheduler, h1);
        Assert.True(h1.Completion.IsCompletedSuccessfully);

        // Now h2 can be admitted. Verify backpressure progresses.
        fix.Scheduler.Step();
        Assert.Equal(1, fix.Scheduler.ActiveCount);
        Assert.Equal(1, fix.Scheduler.QueueDepth);

        DriveUntilCompleted(fix.Scheduler, h2);
        DriveUntilCompleted(fix.Scheduler, h3);
        Assert.True(fix.Scheduler.IsIdle);
    }

    [Fact]
    public async Task Cancellation_BeforeAdmission_PropagatesToCompletion()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 5));

        using var cts = new CancellationTokenSource();
        var h = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 8), cts.Token);

        cts.Cancel();
        fix.Scheduler.Step(); // sweeps the cancelled queued request

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await h.Completion.ConfigureAwait(false));
    }

    [Fact]
    public async Task Cancellation_DuringDecode_ReleasesKvBlocks()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 5));

        using var cts = new CancellationTokenSource();
        var h = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 100), cts.Token);

        // Admit + first decode.
        fix.Scheduler.Step();
        Assert.Equal(SequenceState.Decoding, h.State);
        int freeBeforeCancel = fix.PagedPool.FreeBlocks;

        cts.Cancel();
        fix.Scheduler.Step(); // sweeps cancellation

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await h.Completion.ConfigureAwait(false));
        Assert.True(fix.Scheduler.IsIdle);
        Assert.True(fix.PagedPool.FreeBlocks >= freeBeforeCancel);
    }

    [Fact]
    public async Task ExplicitStopCondition_MaxTokensZeroBlock_Honored()
    {
        // Provide an explicit stop-condition list including MaxTokens(2) — should override
        // the request's MaxTokens.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 9));

        var opts = new InferenceOptions
        {
            Temperature = 0f,
            MaxTokens = 100,
            StopConditions = [new MaxTokensStopCondition(2)],
        };
        var req = new InferenceRequest
        {
            TokenIds = new int[] { 1, 2 },
            Options = opts,
        };

        var h = fix.Scheduler.Submit(req);
        DriveUntilIdle(fix.Scheduler);

        var r = await h.Completion;
        Assert.Equal(2, r.GeneratedTokenCount);
        Assert.Equal(FinishReason.Length, r.FinishReason);
    }

    // ── Priority-ordered admission (Step 59 first piece) ──

    [Fact]
    public async Task Priority_HighAfterLowQueue_AdmittedFirst()
    {
        // Cap to 1 active sequence so admission order is observable. Submit three Low
        // requests, then a High request; verify the High request admits before the
        // two queued Lows.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 1),
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1 });

        var hLow1 = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4, priority: RequestPriority.Low));
        var hLow2 = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4, priority: RequestPriority.Low));
        var hLow3 = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4, priority: RequestPriority.Low));
        var hHigh = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4, priority: RequestPriority.High));

        // Drain in admission order via DriveUntilCompleted on each handle — the order
        // of completion mirrors admission order since MaxActiveSequences=1.
        DriveUntilCompleted(fix.Scheduler, hHigh);
        var rHigh = await hHigh.Completion;
        Assert.Equal(FinishReason.Stop, rHigh.FinishReason);

        // At this point at most one Low should have started; the others are still queued.
        // hLow1 (oldest Low) should be next.
        DriveUntilCompleted(fix.Scheduler, hLow1);
        DriveUntilCompleted(fix.Scheduler, hLow2);
        DriveUntilCompleted(fix.Scheduler, hLow3);

        // All four completed, no exceptions.
        Assert.True(fix.Scheduler.IsIdle);
    }

    [Fact]
    public async Task Priority_SameTier_DrainsInSubmissionOrder()
    {
        // Three Normal requests with MaxActiveSequences=1; verify FIFO order
        // by checking which handle completes first via Task.WhenAny.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 1),
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1 });

        var h1 = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4));
        var h2 = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4));
        var h3 = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4));

        // Drain to completion; with MaxActiveSequences=1 the completion order is
        // the admission order.
        DriveUntilCompleted(fix.Scheduler, h1);
        Assert.True(h1.Completion.IsCompletedSuccessfully);
        Assert.False(h2.Completion.IsCompleted);
        Assert.False(h3.Completion.IsCompleted);

        DriveUntilCompleted(fix.Scheduler, h2);
        Assert.True(h2.Completion.IsCompletedSuccessfully);
        Assert.False(h3.Completion.IsCompleted);

        DriveUntilCompleted(fix.Scheduler, h3);
        Assert.True(h3.Completion.IsCompletedSuccessfully);

        // Sanity: all three actually finished cleanly.
        var r1 = await h1.Completion;
        var r2 = await h2.Completion;
        var r3 = await h3.Completion;
        Assert.Equal(FinishReason.Stop, r1.FinishReason);
        Assert.Equal(FinishReason.Stop, r2.FinishReason);
        Assert.Equal(FinishReason.Stop, r3.FinishReason);
    }

    [Fact]
    public void Priority_InferenceRequest_DefaultsToNormal()
    {
        // Constructed without an explicit Priority, the field defaults to Normal.
        // Guards against accidental default changes in the InferenceRequest record.
        var req = new InferenceRequest { TokenIds = new[] { 1, 2, 3 } };
        Assert.Equal(RequestPriority.Normal, req.Priority);
    }

    [Fact]
    public async Task Priority_CriticalBeatsHighBeatsNormalBeatsLow()
    {
        // Submit Low, Normal, High, Critical in that order; with MaxActiveSequences=1
        // the completion order should reverse priority (Critical, High, Normal, Low).
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 1),
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1 });

        var hLow = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4, priority: RequestPriority.Low));
        var hNormal = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4, priority: RequestPriority.Normal));
        var hHigh = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4, priority: RequestPriority.High));
        var hCritical = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4, priority: RequestPriority.Critical));

        DriveUntilCompleted(fix.Scheduler, hCritical);
        Assert.True(hCritical.Completion.IsCompletedSuccessfully);
        Assert.False(hHigh.Completion.IsCompleted);
        Assert.False(hNormal.Completion.IsCompleted);
        Assert.False(hLow.Completion.IsCompleted);

        DriveUntilCompleted(fix.Scheduler, hHigh);
        Assert.True(hHigh.Completion.IsCompletedSuccessfully);
        Assert.False(hNormal.Completion.IsCompleted);
        Assert.False(hLow.Completion.IsCompleted);

        DriveUntilCompleted(fix.Scheduler, hNormal);
        Assert.True(hNormal.Completion.IsCompletedSuccessfully);
        Assert.False(hLow.Completion.IsCompleted);

        DriveUntilCompleted(fix.Scheduler, hLow);
        Assert.True(hLow.Completion.IsCompletedSuccessfully);

        // Drain remaining task results to ensure no exceptions hide.
        await hCritical.Completion;
        await hHigh.Completion;
        await hNormal.Completion;
        await hLow.Completion;
    }

    // ── Priority-based preemption (Step 59) ──

    [Fact]
    public void Preemption_LowEvictedForHigh_UnderBlockPressure()
    {
        // Pool of 3 blocks; reserve gate of 3 forces "block pressure" the moment any sequence is
        // holding blocks. A Low sequence is admitted and starts decoding (2 blocks held); a High
        // request then arrives and must preempt the Low to be admitted.
        using var fix = new TestFixture(
            options: new ContinuousBatchSchedulerOptions
            {
                MaxActiveSequences = 4,
                ReserveBlocksPerSequence = 3,
                EnablePreemption = true,
            },
            totalBlocks: 3,
            inputEmitter: Ramp);

        var low = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.Low));
        fix.Scheduler.Step(); // admit + first decode → Low holds 2 blocks, FreeBlocks == 1

        Assert.Equal(SequenceState.Decoding, low.State);
        Assert.Equal(0L, fix.Scheduler.GetMetrics().PreemptionCount);
        Assert.Equal(1, fix.PagedPool.FreeBlocks);

        var high = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.High));
        fix.Scheduler.Step(); // block pressure → preempt Low, admit High

        Assert.Equal(1L, fix.Scheduler.GetMetrics().PreemptionCount);
        Assert.Equal(1, fix.Scheduler.ActiveCount);   // only High is active
        Assert.Equal(1, fix.Scheduler.QueueDepth);    // Low was re-queued
        Assert.Equal(SequenceState.Queued, low.State);
        Assert.Equal(SequenceState.Decoding, high.State);
        Assert.False(low.Completion.IsCompleted);

        // Drain both. High finishes first; then Low resumes (recompute) and finishes; blocks return.
        DriveUntilCompleted(fix.Scheduler, high);
        DriveUntilCompleted(fix.Scheduler, low);
        Assert.True(fix.Scheduler.IsIdle);
        Assert.Equal(fix.PagedPool.TotalBlocks, fix.PagedPool.FreeBlocks);
    }

    [Fact]
    public async Task Preemption_ResumedSequence_MatchesUnpreemptedOutput()
    {
        // Control: run the prompt alone with no preemption and capture the exact output.
        int[] controlTokens;
        FinishReason controlReason;
        using (var ctl = new TestFixture(totalBlocks: 8, inputEmitter: Ramp))
        {
            var h = ctl.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16));
            DriveUntilIdle(ctl.Scheduler);
            var r = await h.Completion;
            controlTokens = r.GeneratedTokenIds;
            controlReason = r.FinishReason;
        }
        Assert.Equal(FinishReason.Stop, controlReason);
        Assert.Equal(new[] { 5, 6, 7, 8, 9 }, controlTokens); // ramp 5..9 then EOS

        // Preemption: the same sequence is preempted mid-decode, then resumes via recompute.
        using var fix = new TestFixture(
            options: new ContinuousBatchSchedulerOptions
            {
                MaxActiveSequences = 4,
                ReserveBlocksPerSequence = 3,
                EnablePreemption = true,
            },
            totalBlocks: 3,
            inputEmitter: Ramp);

        var low = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.Low));
        fix.Scheduler.Step();
        Assert.Equal(SequenceState.Decoding, low.State); // genuinely started before preemption

        var high = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.High));
        fix.Scheduler.Step();
        Assert.Equal(1L, fix.Scheduler.GetMetrics().PreemptionCount);
        Assert.Equal(SequenceState.Queued, low.State);

        DriveUntilCompleted(fix.Scheduler, high);
        DriveUntilCompleted(fix.Scheduler, low);

        var lowResp = await low.Completion;
        // Recompute-on-resume must reproduce the unpreempted result exactly, token-for-token.
        Assert.Equal(controlReason, lowResp.FinishReason);
        Assert.Equal(controlTokens, lowResp.GeneratedTokenIds);
        Assert.Equal(4, lowResp.PromptTokenCount);
    }

    [Fact]
    public async Task Preemption_Disabled_HigherPriorityWaitsInsteadOfPreempting()
    {
        // Same pressure setup, but preemption is gated OFF: the High request must wait for the Low
        // to free its blocks rather than evicting it. PreemptionCount stays 0.
        using var fix = new TestFixture(
            options: new ContinuousBatchSchedulerOptions
            {
                MaxActiveSequences = 4,
                ReserveBlocksPerSequence = 3,
                EnablePreemption = false,
            },
            totalBlocks: 3,
            inputEmitter: Ramp);

        var low = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.Low));
        fix.Scheduler.Step();
        Assert.Equal(SequenceState.Decoding, low.State);

        var high = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.High));
        fix.Scheduler.Step();

        Assert.Equal(0L, fix.Scheduler.GetMetrics().PreemptionCount);
        Assert.Equal(SequenceState.Queued, high.State);     // High waits
        Assert.Equal(SequenceState.Decoding, low.State);    // Low keeps running

        DriveUntilIdle(fix.Scheduler);
        var rLow = await low.Completion;
        var rHigh = await high.Completion;
        Assert.Equal(FinishReason.Stop, rLow.FinishReason);
        Assert.Equal(FinishReason.Stop, rHigh.FinishReason);
        Assert.Equal(0L, fix.Scheduler.GetMetrics().PreemptionCount);
    }

    [Fact]
    public async Task Preemption_NeverEvictsCriticalActiveSequence()
    {
        // A High request cannot preempt a Critical (or any same-or-higher tier) active sequence —
        // the victim rule only selects strictly-lower priority. High waits instead.
        using var fix = new TestFixture(
            options: new ContinuousBatchSchedulerOptions
            {
                MaxActiveSequences = 4,
                ReserveBlocksPerSequence = 3,
                EnablePreemption = true,
            },
            totalBlocks: 3,
            inputEmitter: Ramp);

        var critical = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.Critical));
        fix.Scheduler.Step();
        Assert.Equal(SequenceState.Decoding, critical.State);

        var high = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.High));
        fix.Scheduler.Step();

        Assert.Equal(0L, fix.Scheduler.GetMetrics().PreemptionCount);
        Assert.Equal(SequenceState.Decoding, critical.State); // never preempted
        Assert.Equal(SequenceState.Queued, high.State);       // High waits

        DriveUntilIdle(fix.Scheduler);
        await critical.Completion;
        await high.Completion;
        Assert.Equal(0L, fix.Scheduler.GetMetrics().PreemptionCount);
    }

    // ── Batched decode (Step 59) ─────────────────────────────────────────

    // Ramp emitter: a prompt ending in token k decodes k+1, k+2, ... until 9 then EOS, so the
    // generated-token count is a deterministic function of the prompt's last token. Distinct prompt
    // lengths therefore yield distinct counts — a batched decode that cross-wired sequences' logits
    // would produce the wrong counts, so this discriminates correctness (not just "it ran").
    private static int RampExpectedGenerated(int promptLen)
    {
        // last input token == promptLen (MakeRequest builds 1..promptLen); ramp to 9, EOS excluded.
        int n = 0;
        for (int t = promptLen; t < 9; t++) n++; // emits promptLen+1 .. 9
        return n;
    }

    [Fact]
    public async Task BatchedDecode_MultipleSequences_MatchPerSeqBaseline()
    {
        int[] promptLens = [3, 5, 7]; // distinct ramp counts: 6, 4, 2

        // Batched: all admitted together (default MaxActiveSequences) → fused ForwardBatch.
        using var batched = new TestFixture(inputEmitter: Ramp);
        var bHandles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            bHandles[i] = batched.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 32));
        DriveUntilIdle(batched.Scheduler);

        // Per-seq baseline: MaxActiveSequences=1 → one sequence active at a time → never batched.
        using var perSeq = new TestFixture(
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1 },
            inputEmitter: Ramp);
        var pHandles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            pHandles[i] = perSeq.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 32));
        DriveUntilIdle(perSeq.Scheduler);

        Assert.True(batched.Model.ForwardBatchCount > 0, "batched run never fused ForwardBatch");
        Assert.Equal(0, perSeq.Model.ForwardBatchCount);

        for (int i = 0; i < promptLens.Length; i++)
        {
            var b = await bHandles[i].Completion;
            var p = await pHandles[i].Completion;
            int expected = RampExpectedGenerated(promptLens[i]);
            Assert.Equal(FinishReason.Stop, b.FinishReason);
            Assert.Equal(expected, b.GeneratedTokenCount);          // correct per-seq output under batching
            Assert.Equal(p.FinishReason, b.FinishReason);           // batched == per-seq
            Assert.Equal(p.GeneratedTokenCount, b.GeneratedTokenCount);
        }
    }

    [Fact]
    public async Task BatchedDecode_PerSeqStopAndMaxTokens_Honored()
    {
        // Never emits EOS → each sequence stops only at its own MaxTokens, so the batch shrinks
        // 3→2→1 as sequences finish at different steps within the batched-decode loop.
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(tokenId: 7));
        int[] maxTokens = [2, 4, 6];
        var handles = new ISchedulerRequest[maxTokens.Length];
        for (int i = 0; i < maxTokens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: maxTokens[i]));

        DriveUntilIdle(fix.Scheduler);

        Assert.True(fix.Model.ForwardBatchCount > 0, "expected fused decode for 3 concurrent sequences");
        for (int i = 0; i < maxTokens.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Length, r.FinishReason);
            Assert.Equal(maxTokens[i], r.GeneratedTokenCount);
        }
    }

    [Fact]
    public async Task BatchedDecode_CallsForwardBatch_OnlyWhenTwoPlusActive()
    {
        // Two concurrent sequences → batched.
        using var multi = new TestFixture(tokenScript: TokenScript.Constant(tokenId: 7));
        var h1 = multi.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4));
        var h2 = multi.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4));
        DriveUntilIdle(multi.Scheduler);
        await h1.Completion; await h2.Completion;
        Assert.True(multi.Model.ForwardBatchCount > 0);

        // Single sequence → never batched (per-seq Forward path).
        using var single = new TestFixture(tokenScript: TokenScript.Constant(tokenId: 7));
        var h = single.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4));
        DriveUntilIdle(single.Scheduler);
        await h.Completion;
        Assert.Equal(0, single.Model.ForwardBatchCount);
    }

    [Fact]
    public async Task BatchedDecode_StatefulModel_FallsBackToPerSeq()
    {
        // A model that needs per-seq recurrent state must NOT be batch-decoded (the scheduler can't
        // supply that state) — it falls back to per-sequence Forward, still producing correct output.
        int[] promptLens = [3, 5, 7];
        using var fix = new TestFixture(inputEmitter: Ramp, requiresPerSequenceState: true);
        var handles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 32));

        DriveUntilIdle(fix.Scheduler);

        Assert.Equal(0, fix.Model.ForwardBatchCount); // gated off for stateful models
        for (int i = 0; i < promptLens.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Stop, r.FinishReason);
            Assert.Equal(RampExpectedGenerated(promptLens[i]), r.GeneratedTokenCount);
        }
    }

    // ── Batched prefill (Step 59) ────────────────────────────────────────

    [Fact]
    public async Task BatchedPrefill_MultipleSequences_MatchPerSeqBaseline()
    {
        // Distinct prompt lengths → distinct ramp first-tokens AND distinct generated counts. A
        // batched prefill that cross-wired sequences' last-position logits would sample the wrong
        // first token and diverge, so this discriminates correctness (not just "it ran").
        int[] promptLens = [3, 5, 7];

        // Batched: all admitted in one Step (default MaxActiveSequences) → fused prefill ForwardBatch.
        using var batched = new TestFixture(inputEmitter: Ramp);
        var bHandles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            bHandles[i] = batched.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 32));

        // First Step admits + prefills all three together. Assert the fused prefill happened before
        // any decode could (so ForwardBatchCount > 0 is attributable to prefill, not decode).
        batched.Scheduler.Step();
        Assert.True(batched.Model.ForwardBatchCount > 0, "batched run never fused prefill ForwardBatch");
        DriveUntilIdle(batched.Scheduler);

        // Per-seq baseline: MaxActiveSequences=1 → one sequence prefills at a time → never batched.
        using var perSeq = new TestFixture(
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1 },
            inputEmitter: Ramp);
        var pHandles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            pHandles[i] = perSeq.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 32));
        DriveUntilIdle(perSeq.Scheduler);
        Assert.Equal(0, perSeq.Model.ForwardBatchCount);

        for (int i = 0; i < promptLens.Length; i++)
        {
            var b = await bHandles[i].Completion;
            var p = await pHandles[i].Completion;
            int expected = RampExpectedGenerated(promptLens[i]);
            Assert.Equal(FinishReason.Stop, b.FinishReason);
            Assert.Equal(expected, b.GeneratedTokenCount);              // correct per-seq output under batched prefill
            Assert.Equal(p.FinishReason, b.FinishReason);              // batched == per-seq
            Assert.Equal(p.GeneratedTokenCount, b.GeneratedTokenCount);
            Assert.Equal(p.GeneratedTokenIds, b.GeneratedTokenIds);    // exact token-for-token parity
        }
    }

    [Fact]
    public async Task BatchedPrefill_FirstTokenStop_CompletesWithoutDecode()
    {
        // A prompt whose ramp first-token is EOS (last input token == 9 → Ramp emits EOS) must
        // complete during prefill (zero generated tokens, Stop), even when batched with sequences
        // that keep going — verifying per-seq stop handling inside the fused prefill path.
        using var fix = new TestFixture(inputEmitter: Ramp);
        var hStop = fix.Scheduler.Submit(MakeRequest(promptLen: 9, maxTokens: 16)); // last token 9 → EOS first
        var hGo = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 16));   // ramps 4..9

        fix.Scheduler.Step(); // admit + fused prefill of both
        Assert.True(fix.Model.ForwardBatchCount > 0, "expected fused prefill for 2 admitted sequences");
        DriveUntilIdle(fix.Scheduler);

        var rStop = await hStop.Completion;
        Assert.Equal(FinishReason.Stop, rStop.FinishReason);
        Assert.Equal(0, rStop.GeneratedTokenCount);   // stopped on the first (EOS) token, no decode

        var rGo = await hGo.Completion;
        Assert.Equal(FinishReason.Stop, rGo.FinishReason);
        Assert.Equal(RampExpectedGenerated(3), rGo.GeneratedTokenCount);
    }

    [Fact]
    public async Task BatchedPrefill_StatefulModel_FallsBackToPerSeq()
    {
        // A recurrent model (needs per-seq state the scheduler can't supply) must NOT batch-prefill —
        // it falls back to per-seq Forward, still producing correct output.
        int[] promptLens = [3, 5, 7];
        using var fix = new TestFixture(inputEmitter: Ramp, requiresPerSequenceState: true);
        var handles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 32));

        DriveUntilIdle(fix.Scheduler);

        Assert.Equal(0, fix.Model.ForwardBatchCount); // gated off for stateful models (prefill + decode)
        for (int i = 0; i < promptLens.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Stop, r.FinishReason);
            Assert.Equal(RampExpectedGenerated(promptLens[i]), r.GeneratedTokenCount);
        }
    }

    [Fact]
    public void BatchedPrefill_SingleAdmission_NotBatched()
    {
        // Exactly one admission in a Step must take the per-seq Forward path (no ForwardBatch),
        // matching the decode-side "only fuse for ≥2" rule.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 7),
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1 });
        fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 4));

        fix.Scheduler.Step(); // single admit + prefill
        Assert.Equal(0, fix.Model.ForwardBatchCount);
    }

    // ── Recurrent batched decode/prefill (Step 59, threaded per-seq state) ──

    [Fact]
    public async Task RecurrentBatched_MultipleSequences_ThreadsPerSeqStateAndMatchesBaseline()
    {
        // A recurrent (threaded-state) model: the scheduler must allocate one per-seq state per
        // sequence and thread it through ForwardBatch. The mock model THROWS if a batched request has
        // a null or shared MambaState, so correct independent output here proves per-seq threading
        // (not just "it ran"). Distinct prompt lengths → distinct ramp counts (discriminates cross-wiring).
        int[] promptLens = [3, 5, 7];
        using var fix = new RecurrentTestFixture();
        var handles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 32));

        DriveUntilIdle(fix.Scheduler);

        Assert.True(fix.Model.ForwardBatchCount > 0, "recurrent decode/prefill never used ForwardBatch");
        // Exactly one state allocated per sequence (threaded across all its steps, not re-allocated),
        // and every one disposed by completion.
        Assert.Equal(promptLens.Length, fix.Model.StateCreateCount);
        Assert.Equal(promptLens.Length, fix.Model.StateDisposeCount);

        for (int i = 0; i < promptLens.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Stop, r.FinishReason);
            Assert.Equal(RampExpectedGenerated(promptLens[i]), r.GeneratedTokenCount);
        }
    }

    [Fact]
    public async Task RecurrentBatched_SingleSequence_UsesForwardBatchWithState()
    {
        // Even a single recurrent sequence dispatches via ForwardBatch (the only entrypoint that
        // carries the per-seq state) — unlike a dense single sequence, which stays on per-seq Forward.
        using var fix = new RecurrentTestFixture();
        var h = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 32));
        DriveUntilIdle(fix.Scheduler);

        Assert.True(fix.Model.ForwardBatchCount > 0, "single recurrent sequence must route via ForwardBatch");
        Assert.Equal(1, fix.Model.StateCreateCount);
        Assert.Equal(1, fix.Model.StateDisposeCount);
        var r = await h.Completion;
        Assert.Equal(FinishReason.Stop, r.FinishReason);
        Assert.Equal(RampExpectedGenerated(4), r.GeneratedTokenCount);
    }

    [Fact]
    public void RecurrentBatched_MaxRecurrentSequences_CapsConcurrency()
    {
        // MaxRecurrentSequences caps how many threaded recurrent sequences run at once (bounds
        // aggregate per-seq state memory). With a cap of 1, only one is admitted at a time.
        using var fix = new RecurrentTestFixture(
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 8, MaxRecurrentSequences = 1 });

        var h1 = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4));
        var h2 = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4));
        var h3 = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4));

        fix.Scheduler.Step();
        Assert.Equal(1, fix.Scheduler.ActiveCount);   // capped to 1 despite MaxActiveSequences=8
        Assert.Equal(2, fix.Scheduler.QueueDepth);

        DriveUntilCompleted(fix.Scheduler, h1);
        DriveUntilCompleted(fix.Scheduler, h2);
        DriveUntilCompleted(fix.Scheduler, h3);
        Assert.True(fix.Scheduler.IsIdle);
    }

    [Fact]
    public async Task RecurrentBatched_PerSeqMaxTokens_Honored()
    {
        // Never emits EOS → each sequence stops at its own MaxTokens; the batch shrinks 3→2→1 as
        // sequences finish at different steps, exercising per-seq finish inside the threaded batch.
        using var fix = new RecurrentTestFixture(emitter: _ => 7); // constant token, never EOS
        int[] maxTokens = [2, 4, 6];
        var handles = new ISchedulerRequest[maxTokens.Length];
        for (int i = 0; i < maxTokens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: maxTokens[i]));

        DriveUntilIdle(fix.Scheduler);

        Assert.True(fix.Model.ForwardBatchCount > 0);
        Assert.Equal(maxTokens.Length, fix.Model.StateDisposeCount);
        for (int i = 0; i < maxTokens.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Length, r.FinishReason);
            Assert.Equal(maxTokens[i], r.GeneratedTokenCount);
        }
    }

    // ── Prefill/decode disaggregation seam (Step 59) ─────────────────────

    [Fact]
    public void Disaggregated_AdmittedInStepPrefill_DecodesOnlyInStepDecode()
    {
        // StepPrefill admits + prefills (samples the first token → Decoding) but must NOT decode;
        // only StepDecode advances decode tokens. Driving StepPrefill alone can never complete a
        // sequence that finishes by max-tokens.
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(tokenId: 7)); // never EOS
        var h = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4));

        Assert.True(fix.Scheduler.StepPrefill());      // admit + prefill
        Assert.Equal(SequenceState.Decoding, h.State);
        Assert.Equal(1, fix.Scheduler.ActiveCount);

        // More prefill-only steps do no further work for this sequence and never complete it.
        for (int i = 0; i < 5; i++) fix.Scheduler.StepPrefill();
        Assert.False(h.Completion.IsCompleted);

        // Decode phase drives it to the max-tokens cap.
        for (int i = 0; i < 10 && !h.Completion.IsCompleted; i++) fix.Scheduler.StepDecode();
        Assert.True(h.Completion.IsCompletedSuccessfully);
        Assert.True(fix.Scheduler.IsIdle);
    }

    [Fact]
    public async Task Disaggregated_SeparatePhases_MatchCombinedStepOutput()
    {
        int[] promptLens = [3, 5, 7];

        // Baseline: combined Step().
        int[][] combined = new int[promptLens.Length][];
        using (var baseFix = new TestFixture(inputEmitter: Ramp))
        {
            var hs = new ISchedulerRequest[promptLens.Length];
            for (int i = 0; i < promptLens.Length; i++)
                hs[i] = baseFix.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 32));
            DriveUntilIdle(baseFix.Scheduler);
            for (int i = 0; i < promptLens.Length; i++)
                combined[i] = (await hs[i].Completion).GeneratedTokenIds;
        }

        // Disaggregated driver: run StepPrefill and StepDecode as separate phases, with several
        // decode phases between prefills (mimicking a decode worker outpacing a prefill worker).
        using var fix = new TestFixture(inputEmitter: Ramp);
        var handles = new ISchedulerRequest[promptLens.Length];
        for (int i = 0; i < promptLens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 32));

        for (int iter = 0; iter < 1000 && !fix.Scheduler.IsIdle; iter++)
        {
            fix.Scheduler.StepPrefill();
            // Decode runs more often than prefill admission.
            fix.Scheduler.StepDecode();
            fix.Scheduler.StepDecode();
        }
        Assert.True(fix.Scheduler.IsIdle);

        for (int i = 0; i < promptLens.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Stop, r.FinishReason);
            Assert.Equal(combined[i], r.GeneratedTokenIds); // phase-split is byte-identical to combined Step
        }
    }

    [Fact]
    public async Task Disaggregated_DecodePhaseAloneIsNoOp_WhenNothingAdmitted()
    {
        // A decode worker that runs ahead of any admission must be a safe no-op (no prefill happened).
        using var fix = new TestFixture(inputEmitter: Ramp);
        Assert.False(fix.Scheduler.StepDecode());      // nothing active yet
        Assert.False(fix.Scheduler.StepDecode());

        var h = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 32));
        Assert.False(fix.Scheduler.StepDecode());      // submitted but not yet admitted → still no-op
        Assert.Equal(SequenceState.Queued, h.State);

        // Now drive both phases to completion.
        for (int i = 0; i < 1000 && !fix.Scheduler.IsIdle; i++) { fix.Scheduler.StepPrefill(); fix.Scheduler.StepDecode(); }
        var r = await h.Completion;
        Assert.Equal(FinishReason.Stop, r.FinishReason);
        Assert.Equal(RampExpectedGenerated(4), r.GeneratedTokenCount);
    }

    // ── Per-API-key fairness (Step 59, SFQ admission) ───────────────────

    [Fact]
    public void InferenceRequest_ApiKey_DefaultsNull()
    {
        var req = new InferenceRequest { TokenIds = new[] { 1, 2, 3 } };
        Assert.Null(req.ApiKey);
    }

    [Fact]
    public async Task Fairness_Enabled_LightKeyInterleavesAheadOfHammerBacklog()
    {
        // A hammer client floods the queue (5 requests), then a light client submits 1 — all before
        // any admission. With SFQ fairness, the light request interleaves right after the hammer's
        // FIRST request (its key is idle → low start tag), instead of waiting behind all 5.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 1),
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1, EnableFairness = true });

        var hammer = new ISchedulerRequest[5];
        for (int i = 0; i < hammer.Length; i++)
            hammer[i] = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, apiKey: "hammer"));
        var light = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, apiKey: "light"));

        DriveUntilCompleted(fix.Scheduler, light);

        int hammerDoneBeforeLight = 0;
        foreach (var h in hammer) if (h.Completion.IsCompleted) hammerDoneBeforeLight++;
        // SFQ admits hammer#1 then light; the rest of the hammer backlog follows.
        Assert.True(hammerDoneBeforeLight <= 1,
            $"fairness failed: light was starved behind {hammerDoneBeforeLight} hammer requests");

        DriveUntilIdle(fix.Scheduler);
        await light.Completion;
        foreach (var h in hammer) await h.Completion;
    }

    [Fact]
    public void Fairness_Disabled_LightKeyWaitsBehindBacklog_Fifo()
    {
        // Same setup, fairness OFF: admission is pure FIFO-by-submission-order, so the light request
        // (submitted last) waits behind ALL 5 hammer requests — the behaviour fairness fixes.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 1),
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1, EnableFairness = false });

        var hammer = new ISchedulerRequest[5];
        for (int i = 0; i < hammer.Length; i++)
            hammer[i] = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, apiKey: "hammer"));
        var light = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, apiKey: "light"));

        DriveUntilCompleted(fix.Scheduler, light);

        int hammerDoneBeforeLight = 0;
        foreach (var h in hammer) if (h.Completion.IsCompleted) hammerDoneBeforeLight++;
        Assert.Equal(5, hammerDoneBeforeLight); // FIFO: light is dead last
    }

    [Fact]
    public void Fairness_PriorityStillDominatesAcrossTiers()
    {
        // Fairness only reorders WITHIN a priority tier. A High-priority light request still beats a
        // backlog of Low-priority hammer requests regardless of submission order / fairness tags.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 1),
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1, EnableFairness = true });

        var hammer = new ISchedulerRequest[4];
        for (int i = 0; i < hammer.Length; i++)
            hammer[i] = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, priority: RequestPriority.Low, apiKey: "hammer"));
        var light = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, priority: RequestPriority.High, apiKey: "light"));

        DriveUntilCompleted(fix.Scheduler, light);

        int hammerDoneBeforeLight = 0;
        foreach (var h in hammer) if (h.Completion.IsCompleted) hammerDoneBeforeLight++;
        Assert.Equal(0, hammerDoneBeforeLight); // High light admitted first; priority dominates fairness
    }

    [Fact]
    public void Fairness_HigherWeightKey_AdmittedProportionallyAhead()
    {
        // Two equal-priority keys flood the queue with equal backlogs (4 each), but "heavy" carries a
        // fairness weight of 4 vs "light" weight 1. SFQ charges heavy cost/4, so its start tags grow 4×
        // slower and it races ahead: by the time the WHOLE heavy backlog has drained, at most ONE light
        // request has slipped through. With equal weights the interleave is ~1:1 (3 light done by then),
        // so this asserts the weight is actually applied — not just that SFQ runs.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 1),
            options: new ContinuousBatchSchedulerOptions
            {
                MaxActiveSequences = 1,
                EnableFairness = true,
                FairnessWeightProvider = key => string.Equals(key, "heavy", StringComparison.Ordinal) ? 4.0 : 1.0,
            });

        var heavy = new ISchedulerRequest[4];
        for (int i = 0; i < heavy.Length; i++)
            heavy[i] = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, apiKey: "heavy"));
        var light = new ISchedulerRequest[4];
        for (int i = 0; i < light.Length; i++)
            light[i] = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, apiKey: "light"));

        DriveUntilCompleted(fix.Scheduler, heavy[3]); // drive until the LAST heavy request completes

        int lightDone = 0;
        foreach (var l in light) if (l.Completion.IsCompleted) lightDone++;
        Assert.True(lightDone <= 1,
            $"weight fairness failed: {lightDone} light requests slipped ahead before the heavy backlog drained");

        DriveUntilIdle(fix.Scheduler);
    }

    [Fact]
    public void Fairness_UniformWeightProvider_MatchesUnweightedSfq()
    {
        // A provider that returns the SAME weight for every key (here 2.0) scales all charges equally,
        // so it must not change relative ordering — admission matches the unweighted SFQ interleave:
        // the light request still slots in right after the hammer's first request.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 1),
            options: new ContinuousBatchSchedulerOptions
            {
                MaxActiveSequences = 1,
                EnableFairness = true,
                FairnessWeightProvider = _ => 2.0,
            });

        var hammer = new ISchedulerRequest[5];
        for (int i = 0; i < hammer.Length; i++)
            hammer[i] = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, apiKey: "hammer"));
        var light = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 4, apiKey: "light"));

        DriveUntilCompleted(fix.Scheduler, light);

        int hammerDoneBeforeLight = 0;
        foreach (var h in hammer) if (h.Completion.IsCompleted) hammerDoneBeforeLight++;
        Assert.True(hammerDoneBeforeLight <= 1,
            $"uniform weight changed ordering: light starved behind {hammerDoneBeforeLight} hammer requests");

        DriveUntilIdle(fix.Scheduler);
    }

    [Fact]
    public async Task PerKeyTokenUsage_AccruesGeneratedTokensPerApiKey()
    {
        // Each request generates exactly 3 tokens (emit 9,9,9 then EOS). Per-key accounting sums
        // generated tokens by ApiKey; null-key requests are not attributed.
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 3));

        var a1 = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 16, apiKey: "alice"));
        var a2 = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 16, apiKey: "alice"));
        var b1 = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 16, apiKey: "bob"));
        var anon = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 16)); // null key

        DriveUntilIdle(fix.Scheduler);
        await Task.WhenAll(a1.Completion, a2.Completion, b1.Completion, anon.Completion);

        var usage = fix.Scheduler.GetPerKeyTokenUsage();
        Assert.Equal(6, usage["alice"]); // 2 requests × 3 tokens
        Assert.Equal(3, usage["bob"]);
        Assert.Equal(2, usage.Count); // null-key request is not attributed
    }

    // ── Helpers ──

    /// <summary>
    /// Content-driven token emitter for preemption tests: emits <c>input + 1</c> (a deterministic
    /// ramp) until the input token reaches 9, then emits EOS. Because it depends only on the
    /// generated-token chain — not on any per-cache step counter — it produces identical output
    /// whether or not the sequence was preempted and recomputed mid-flight.
    /// </summary>
    private static int Ramp(int lastInputToken) => lastInputToken >= 9 ? EosTokenId : lastInputToken + 1;

    private static void DriveUntilIdle(IBatchScheduler scheduler, int maxIterations = 1000)
    {
        for (int i = 0; i < maxIterations; i++)
        {
            if (scheduler.IsIdle) return;
            scheduler.Step();
        }
        Assert.Fail("Scheduler did not reach idle within iteration cap.");
    }

    private static void DriveUntilCompleted(IBatchScheduler scheduler, ISchedulerRequest handle, int maxIterations = 1000)
    {
        for (int i = 0; i < maxIterations; i++)
        {
            if (handle.Completion.IsCompleted) return;
            scheduler.Step();
        }
        Assert.Fail("Sequence did not complete within iteration cap.");
    }

    private static InferenceRequest MakeRequest(int promptLen, int maxTokens,
                                                 RequestPriority priority = RequestPriority.Normal,
                                                 string? apiKey = null)
    {
        // Build prompt: avoid 0 (EOS) to keep things clean. Tokens 1..promptLen.
        var tokens = new int[promptLen];
        for (int i = 0; i < promptLen; i++) tokens[i] = i + 1;

        return new InferenceRequest
        {
            TokenIds = tokens,
            Options = new InferenceOptions { Temperature = 0f, MaxTokens = maxTokens },
            Priority = priority,
            ApiKey = apiKey,
        };
    }

    // Scripted token emission. A script returns the token to emit given the prefill step counter.
    private sealed class TokenScript
    {
        private readonly Func<int, int> _emit;
        public TokenScript(Func<int, int> emit) => _emit = emit;
        public int Emit(int step) => _emit(step);

        /// <summary>Emit <paramref name="tokenId"/> on every step.</summary>
        public static TokenScript Constant(int tokenId) => new(_ => tokenId);

        /// <summary>Emit <paramref name="afterToken"/> on the first <paramref name="afterNTokens"/> steps,
        /// then emit <paramref name="tokenId"/>.</summary>
        public static TokenScript Constant(int tokenId, int afterNTokens, int afterToken = 9)
            => new(step => step < afterNTokens ? afterToken : tokenId);
    }

    private sealed class TestFixture : IDisposable
    {
        public PagedKvCacheFactory PagedFactory { get; }
        public KvBlockPool PagedPool => PagedFactory.Pool;
        public MockModel Model { get; }
        public MockTokenizer Tokenizer { get; }
        public ContinuousBatchScheduler Scheduler { get; }

        public TestFixture(
            TokenScript? tokenScript = null,
            ContinuousBatchSchedulerOptions? options = null,
            int totalBlocks = 64,
            Func<int, int>? inputEmitter = null,
            bool requiresPerSequenceState = false)
        {
            tokenScript ??= TokenScript.Constant(EosTokenId, afterNTokens: 1);
            PagedFactory = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize,
                maxTotalTokens: totalBlocks * BlockSize);
            Model = new MockModel(tokenScript, inputEmitter, requiresPerSequenceState);
            Tokenizer = new MockTokenizer();
            Scheduler = new ContinuousBatchScheduler(
                Model,
                Tokenizer,
                (_, maxSeq) => PagedFactory.Create(maxSeq),
                options,
                pagedPool: PagedFactory.Pool);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            PagedFactory.Dispose();
            Model.Dispose();
        }
    }

    /// <summary>
    /// Deterministic model. Tracks a per-sequence decode-step counter using the KV-cache instance
    /// identity (each PagedKvCache is unique). Forward emits scripted logits so argmax = scripted token.
    /// </summary>
    private sealed class MockModel : IModel
    {
        private readonly TokenScript _script;
        // Optional content-driven emitter: given the last input token, returns the token to emit.
        // Independent of the per-cache step counter, so it survives a preempt→recompute resume
        // (a fresh KvCache resets the step counter, but the generated-token chain is preserved).
        private readonly Func<int, int>? _inputEmitter;
        private readonly bool _requiresPerSeqState;
        private readonly Dictionary<IKvCache, int> _stepCounters = new(ReferenceEqualityComparer.Instance);

        /// <summary>Number of times <see cref="ForwardBatch"/> was invoked (i.e. the scheduler took
        /// the fused batched-decode path), for asserting batching is engaged / bypassed.</summary>
        public int ForwardBatchCount { get; private set; }

        public MockModel(TokenScript script, Func<int, int>? inputEmitter = null,
            bool requiresPerSequenceState = false)
        {
            _script = script;
            _inputEmitter = inputEmitter;
            _requiresPerSeqState = requiresPerSequenceState;
        }

        public bool RequiresPerSequenceState => _requiresPerSeqState;

        // Mirror IModel's default loop ForwardBatch, but count invocations so tests can assert the
        // scheduler actually fused the decode (vs the per-sequence Forward path).
        public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
        {
            ForwardBatchCount++;
            var results = new ITensor[requests.Count];
            for (int i = 0; i < requests.Count; i++)
            {
                var r = requests[i];
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache);
            }
            return results;
        }

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HeadDim * NumKvHeads,
            IntermediateSize = HeadDim * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
        {
            int batchSize = tokenIds.Length;

            // Allocate logits [batchSize, VocabSize]
            long totalFloats = (long)batchSize * VocabSize;
            nint logitsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            NativeMemory.Clear((void*)logitsPtr, (nuint)(totalFloats * sizeof(float)));

            // Determine current step for this sequence (prefill is step 0; decode increments).
            int step;
            if (kvCache is not null)
            {
                if (!_stepCounters.TryGetValue(kvCache, out step)) step = 0;
                _stepCounters[kvCache] = step + 1;
            }
            else
            {
                step = 0;
            }

            int emitToken;
            if (_inputEmitter is not null)
            {
                // Emit from the last input token of this forward (the token whose successor the
                // scheduler samples). Deterministic in the generated-token chain, not in call count.
                int lastInput = tokenIds.Length > 0 ? tokenIds[^1] : 0;
                emitToken = _inputEmitter(lastInput);
            }
            else
            {
                emitToken = _script.Emit(step);
            }
            if ((uint)emitToken >= VocabSize) emitToken = 1;

            float* dst = (float*)logitsPtr;
            for (int b = 0; b < batchSize; b++)
            {
                // Set argmax for the last position only; for prefill batches the scheduler reads
                // logitRows-1 anyway, so set the same argmax across the batch for safety.
                float* row = dst + (long)b * VocabSize;
                for (int v = 0; v < VocabSize; v++) row[v] = -10f;
                row[emitToken] = 10f;
            }

            // Update KV-cache (write zeros — the scheduler doesn't inspect content).
            if (kvCache is not null)
            {
                int kvStride = NumKvHeads * HeadDim;
                long kvBytes = (long)batchSize * kvStride * sizeof(float);
                nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
                NativeMemory.Clear((void*)kPtr, (nuint)kvBytes);
                NativeMemory.Clear((void*)vPtr, (nuint)kvBytes);
                try
                {
                    for (int layer = 0; layer < NumLayers; layer++)
                    {
                        var kRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, kPtr);
                        var vRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, vPtr);
                        kvCache.Update(kRef, vRef, positions, layer);
                    }
                }
                finally
                {
                    NativeMemory.AlignedFree((void*)kPtr);
                    NativeMemory.AlignedFree((void*)vPtr);
                }
            }

            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, logitsPtr);
        }

        public void Dispose() { }
    }

    /// <summary>
    /// Fixture for the recurrent (threaded-state) scheduler tests — wires a <see cref="MockRecurrentModel"/>
    /// (RequiresPerSequenceState + SupportsThreadedSequenceState) so the scheduler exercises the
    /// allocate-per-seq-state + always-ForwardBatch path.
    /// </summary>
    private sealed class RecurrentTestFixture : IDisposable
    {
        public PagedKvCacheFactory PagedFactory { get; }
        public MockRecurrentModel Model { get; }
        public ContinuousBatchScheduler Scheduler { get; }

        public RecurrentTestFixture(
            ContinuousBatchSchedulerOptions? options = null,
            int totalBlocks = 64,
            Func<int, int>? emitter = null)
        {
            PagedFactory = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize,
                maxTotalTokens: totalBlocks * BlockSize);
            Model = new MockRecurrentModel(emitter ?? Ramp);
            Scheduler = new ContinuousBatchScheduler(
                Model,
                new MockTokenizer(),
                (_, maxSeq) => PagedFactory.Create(maxSeq),
                options,
                pagedPool: PagedFactory.Pool);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            PagedFactory.Dispose();
            Model.Dispose();
        }
    }

    /// <summary>Mock per-sequence recurrent state. Implements <see cref="IMambaState"/> so it flows
    /// through the scheduler's <c>MambaState</c> slot; disposal is counted via a callback so tests can
    /// assert every allocated state is freed.</summary>
    private sealed class MockRecurrentState : IMambaState
    {
        private readonly Action _onDispose;
        public int Step;
        public MockRecurrentState(Action onDispose) => _onDispose = onDispose;
        public int NumLayers => ContinuousBatchSchedulerTests.NumLayers;
        public void Reset() => Step = 0;
        public void Dispose() => _onDispose();
    }

    /// <summary>
    /// Deterministic recurrent model. Reports <c>RequiresPerSequenceState</c> AND
    /// <c>SupportsThreadedSequenceState</c>, so the scheduler allocates one <see cref="MockRecurrentState"/>
    /// per sequence and dispatches every forward (incl. single-sequence) through <see cref="ForwardBatch"/>.
    /// Its <see cref="ForwardBatch"/> THROWS if a multi-seq request carries a null or shared MambaState —
    /// so a scheduler that fails to thread per-seq state surfaces as a test failure, not silent corruption.
    /// </summary>
    private sealed class MockRecurrentModel : IModel
    {
        private readonly Func<int, int> _emit; // content-driven (ramp): last input token -> emitted token
        public int ForwardBatchCount { get; private set; }
        public int StateCreateCount { get; private set; }
        public int StateDisposeCount { get; private set; }

        public MockRecurrentModel(Func<int, int> emit) => _emit = emit;

        public bool RequiresPerSequenceState => true;
        public bool SupportsThreadedSequenceState => true;
        public IRecurrentSequenceState? CreateSequenceState()
        {
            StateCreateCount++;
            return new MockRecurrentState(() => StateDisposeCount++);
        }

        public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
        {
            ForwardBatchCount++;
            if (requests.Count >= 2)
            {
                var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
                for (int i = 0; i < requests.Count; i++)
                {
                    var st = requests[i].MambaState;
                    if (st is null)
                        throw new InvalidOperationException(
                            $"recurrent ForwardBatch[{i}] has null MambaState — scheduler did not thread per-seq state");
                    if (!seen.Add(st))
                        throw new InvalidOperationException(
                            "recurrent ForwardBatch got the SAME MambaState across sequences — state is shared, not per-seq");
                }
            }
            var results = new ITensor[requests.Count];
            for (int i = 0; i < requests.Count; i++)
            {
                var r = requests[i];
                results[i] = ForwardOne(r.TokenIds.Span, deviceId, r.MambaState);
            }
            return results;
        }

        private unsafe ITensor ForwardOne(ReadOnlySpan<int> tokenIds, int deviceId, IMambaState? state)
        {
            // Touch the per-seq state so threading is genuinely exercised (and a null would NRE on a
            // single-seq dispatch too).
            if (state is MockRecurrentState s) s.Step++;

            int batchSize = tokenIds.Length;
            long totalFloats = (long)batchSize * VocabSize;
            nint logitsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);

            int lastInput = tokenIds.Length > 0 ? tokenIds[^1] : 0;
            int emit = _emit(lastInput);
            if ((uint)emit >= VocabSize) emit = 1;

            float* dst = (float*)logitsPtr;
            for (int b = 0; b < batchSize; b++)
            {
                float* row = dst + (long)b * VocabSize;
                for (int v = 0; v < VocabSize; v++) row[v] = -10f;
                row[emit] = 10f;
            }
            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, logitsPtr);
        }

        // Plain Forward is part of the interface but NOT used by the scheduler for a threaded-state
        // model (it always routes through ForwardBatch). Provide a functional fallback anyway.
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => ForwardOne(tokenIds, deviceId, null);
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
            => ForwardOne(tokenIds, deviceId, null);

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HeadDim * NumKvHeads,
            IntermediateSize = HeadDim * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;
        public void Dispose() { }
    }

    private sealed class MockTokenizer : ITokenizer
    {
        public int VocabSize => ContinuousBatchSchedulerTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => ContinuousBatchSchedulerTests.EosTokenId;

        public int[] Encode(string text) => Array.Empty<int>();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => 0;
    }
}
