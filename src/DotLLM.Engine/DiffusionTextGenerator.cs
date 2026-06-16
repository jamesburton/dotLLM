using System.Buffers;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Samplers;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Masked text-diffusion generator (DiffusionGemma). Mirrors <see cref="TextGenerator"/> for the
/// autoregressive case but, instead of emitting one token per forward pass, it refines a fixed-length
/// <i>canvas</i> of masked positions over up to <see cref="DiffusionConfig.MaxDenoisingSteps"/>
/// denoising steps, committing the most-confident positions each step until the canvas stabilises.
/// </summary>
/// <remarks>
/// <para>
/// <b>Per-step forward.</b> Each step rebuilds the working sequence
/// <c>[prompt tokens … | canvas tokens …]</c> with positions <c>[0 … promptLen+canvasLen)</c> and runs
/// a <i>cacheless</i> forward under <see cref="AttentionMaskSpec.Hybrid(int)"/> with
/// <c>prefixLen = promptLen</c>: the prompt prefix stays causal among itself, the canvas is
/// bidirectional and cross-attends the whole prompt. PR-3 forbids a non-null KV-cache with a
/// non-causal mask (it throws <see cref="NotSupportedException"/>), so no KV-cache is ever passed
/// here and the full prompt+canvas is recomputed every step.
/// </para>
/// <para>
/// <b>Cost / follow-up.</b> Recomputing the prompt prefix on every denoise step is the
/// correct-but-unoptimised path. Reusing a cached prompt-KV prefix across steps (the canvas region
/// alone is recomputed) is a deferred optimisation that depends on KV-cache wiring for non-causal
/// masks — tracked separately. Real-weight end-to-end validation against a downloaded DiffusionGemma
/// checkpoint is issue #32; this type is validated on a synthetic model only.
/// </para>
/// <para>
/// <b>Multi-canvas (block-autoregressive).</b> When the requested target length exceeds
/// <see cref="DiffusionConfig.CanvasLength"/>, a finished canvas is appended to the prompt context
/// (becoming a causal prefix for the next block) and a fresh all-mask canvas is started, until the
/// target length is reached or an EOS token appears in a finished canvas.
/// </para>
/// </remarks>
public sealed class DiffusionTextGenerator
{
    private readonly IModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly DiffusionConfig _diffusion;
    private readonly IDiffusionUnmaskSampler _sampler;
    private readonly float _logitSoftCap;

    /// <summary>
    /// Creates a diffusion text generator.
    /// </summary>
    /// <param name="model">Model to run cacheless hybrid forward passes. Its
    /// <see cref="ModelConfig.DiffusionConfig"/> is used when <paramref name="diffusionConfig"/> is null.</param>
    /// <param name="tokenizer">Tokenizer for encoding the prompt and decoding finished canvases.</param>
    /// <param name="sampler">Canvas-level unmasking sampler (e.g. <see cref="EntropyBoundSampler"/>).
    /// Null uses a default argmax <see cref="EntropyBoundSampler"/>.</param>
    /// <param name="diffusionConfig">Diffusion decode config. Null falls back to the model's
    /// <see cref="ModelConfig.DiffusionConfig"/>; if that is also null an exception is thrown.</param>
    /// <exception cref="ArgumentException">The model has no diffusion configuration and none was supplied.</exception>
    public DiffusionTextGenerator(
        IModel model,
        ITokenizer tokenizer,
        IDiffusionUnmaskSampler? sampler = null,
        DiffusionConfig? diffusionConfig = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        _model = model;
        _tokenizer = tokenizer;
        _diffusion = diffusionConfig
            ?? model.Config.DiffusionConfig
            ?? throw new ArgumentException(
                "No diffusion configuration: the model's ModelConfig.DiffusionConfig is null and none was supplied.",
                nameof(diffusionConfig));
        _sampler = sampler ?? new EntropyBoundSampler();
        _logitSoftCap = model.Config.FinalLogitSoftcap ?? 0f;
    }

    /// <summary>
    /// Generates text by masked diffusion from the supplied prompt text.
    /// </summary>
    /// <param name="prompt">Prompt text; encoded with the tokenizer.</param>
    /// <param name="targetLength">Optional total generated-token target. Values greater than the
    /// canvas length trigger multi-canvas (block-autoregressive) decoding. Null/≤0 uses a single
    /// canvas of <see cref="DiffusionConfig.CanvasLength"/>.</param>
    /// <param name="onCanvasStep">Optional streaming callback invoked after each denoise step and on
    /// canvas completion, with the live canvas snapshot.</param>
    /// <returns>The decoded text and per-run statistics.</returns>
    public DiffusionResult Generate(
        string prompt,
        int? targetLength = null,
        Action<DiffusionCanvasState>? onCanvasStep = null)
    {
        int[] promptIds = _tokenizer.Encode(prompt);
        return Generate(promptIds, targetLength, onCanvasStep);
    }

    /// <summary>
    /// Generates text by masked diffusion from pre-tokenized prompt ids.
    /// </summary>
    /// <param name="promptTokens">Prompt token ids (already encoded).</param>
    /// <param name="targetLength">Optional total generated-token target (see the text overload).</param>
    /// <param name="onCanvasStep">Optional per-step streaming callback (see the text overload).</param>
    /// <returns>The decoded text and per-run statistics.</returns>
    public DiffusionResult Generate(
        ReadOnlySpan<int> promptTokens,
        int? targetLength = null,
        Action<DiffusionCanvasState>? onCanvasStep = null)
    {
        int canvasLength = _diffusion.CanvasLength;
        if (canvasLength < 1)
            throw new InvalidOperationException("DiffusionConfig.CanvasLength must be at least 1.");

        int target = targetLength is > 0 ? targetLength.Value : canvasLength;
        int vocabSize = _model.Config.VocabSize;
        int maskTokenId = _diffusion.MaskTokenId;
        int eos = _tokenizer.EosTokenId;

        // Growing prompt context: starts as the user prompt, and after each finished canvas the
        // committed canvas tokens are appended (block-autoregressive). The prefix is always causal.
        var context = new List<int>(promptTokens.Length + target);
        for (int i = 0; i < promptTokens.Length; i++)
            context.Add(promptTokens[i]);
        int promptLen = context.Count;

        var generatedIds = new List<int>(target);
        var canvasStats = new List<DiffusionCanvasStats>();
        int totalSteps = 0;
        bool hitEos = false;

        while (generatedIds.Count < target && !hitEos)
        {
            // This block's canvas length never overshoots the remaining target.
            int blockLen = Math.Min(canvasLength, target - generatedIds.Count);
            int prefixLen = context.Count;

            var blockResult = RunCanvas(context, prefixLen, blockLen, vocabSize, maskTokenId,
                onCanvasStep, canvasStats.Count);

            totalSteps += blockResult.Steps;
            canvasStats.Add(new DiffusionCanvasStats(
                CanvasIndex: canvasStats.Count,
                CanvasLength: blockLen,
                Steps: blockResult.Steps,
                StopReason: blockResult.StopReason));

            // Absorb the finished canvas: it becomes causal prefix for the next block and part of
            // the generated output. Stop at an EOS inside the canvas (exclude EOS from output).
            for (int i = 0; i < blockLen; i++)
            {
                int tok = blockResult.Canvas[i];
                if (tok == eos)
                {
                    hitEos = true;
                    break;
                }
                context.Add(tok);
                generatedIds.Add(tok);
            }
        }

        string text = generatedIds.Count > 0
            ? _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false)
            : string.Empty;

        var finishReason = hitEos ? FinishReason.Stop : FinishReason.Length;

        return new DiffusionResult
        {
            Text = text,
            GeneratedTokenIds = generatedIds.ToArray(),
            PromptTokenCount = promptLen,
            GeneratedTokenCount = generatedIds.Count,
            TotalDenoisingSteps = totalSteps,
            CanvasCount = canvasStats.Count,
            FinishReason = finishReason,
            CanvasStats = canvasStats.ToArray(),
        };
    }

    /// <summary>
    /// The attention-mask spec for a canvas forward, per <see cref="DiffusionConfig.CanvasAttentionMode"/>:
    /// fully <see cref="AttentionMaskSpec.Bidirectional"/> (LLaDA-style) or
    /// <see cref="AttentionMaskSpec.Hybrid(int)"/> with a causal prompt prefix (DiffusionGemma block-AR).
    /// </summary>
    private AttentionMaskSpec CanvasMaskSpec(int prefixLen) =>
        _diffusion.CanvasAttentionMode == AttentionMaskMode.Bidirectional
            ? AttentionMaskSpec.Bidirectional
            : AttentionMaskSpec.Hybrid(prefixLen);

    /// <summary>
    /// Runs the denoising loop for a single canvas appended after the current causal context.
    /// </summary>
    private CanvasRun RunCanvas(
        List<int> context, int prefixLen, int blockLen, int vocabSize, int maskTokenId,
        Action<DiffusionCanvasState>? onCanvasStep, int canvasIndex)
    {
        // canvas[i] is the committed token at canvas position i, or maskTokenId while masked.
        int[] canvas = new int[blockLen];
        Array.Fill(canvas, maskTokenId);
        int maskedCount = blockLen;

        var scheduler = new DenoiseScheduler(_diffusion);

        int seqLen = prefixLen + blockLen;
        // Working buffers for the cacheless forward (prompt-context prefix + canvas).
        int[] seqTokens = ArrayPool<int>.Shared.Rent(seqLen);
        int[] positions = ArrayPool<int>.Shared.Rent(seqLen);
        // Scatter buffers for the still-masked rows handed to the canvas sampler.
        long logitElems = (long)blockLen * vocabSize;
        if (logitElems > int.MaxValue)
            throw new InvalidOperationException("Canvas logit buffer (blockLen * vocabSize) exceeds Int32 range.");
        int[] maskedPositions = ArrayPool<int>.Shared.Rent(blockLen);
        float[] maskedLogits = ArrayPool<float>.Shared.Rent((int)logitElems);

        // Self-conditioning (DiffusionGemma): the FULL canvas-region logits [blockLen × vocab]
        // (post-softcap, NOT mask-suppressed) from the PREVIOUS step, fed back into the next
        // step's canvas embedding. _scValid flips true after the first forward populates it.
        // On non-SC models SetDiffusionSelfCond is a no-op so this is harmless overhead-free
        // bookkeeping (the buffer fills but the model ignores it).
        float[] scPrevCanvasLogits = ArrayPool<float>.Shared.Rent((int)logitElems);
        bool scValid = false;

        // Canvas attention pattern is model-specific: DiffusionGemma is block-AR (Hybrid —
        // causal prompt prefix + bidirectional canvas); LLaDA is fully Bidirectional over
        // [prompt | canvas] (Hybrid yields degenerate all-EOS output on LLaDA).
        AttentionMaskSpec hybrid = CanvasMaskSpec(prefixLen);

        int step = 0;
        DenoiseStopResult stop = DenoiseStopResult.Continue;

        try
        {
            for (int i = 0; i < prefixLen; i++)
                positions[i] = i;
            for (int i = 0; i < blockLen; i++)
                positions[prefixLen + i] = prefixLen + i;
            for (int i = 0; i < prefixLen; i++)
                seqTokens[i] = context[i];

            for (step = 0; step < _diffusion.MaxDenoisingSteps; step++)
            {
                // Rebuild the canvas region of the working sequence from the live canvas state.
                for (int i = 0; i < blockLen; i++)
                    seqTokens[prefixLen + i] = canvas[i];

                // Self-conditioning: feed the PREVIOUS step's canvas logits into THIS forward's
                // canvas embedding. Step 0 (no prior forward) → scUse=0 (zero-SC, the AR/LLaDA-
                // identical path); steps > 0 → scUse=1 + the prior forward's canvas-region logits.
                // No-op on non-DiffusionGemma models (default IModel implementation).
                if (scValid)
                    _model.SetDiffusionSelfCond(
                        scPrevCanvasLogits.AsSpan(0, blockLen * vocabSize), blockLen, scUse: 1f);
                else
                    _model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);

                // ── Cacheless hybrid forward. NEVER pass a KV-cache here: PR-3 throws for a
                //    non-causal mask + non-null cache. Recompute the whole prompt+canvas each step.
                int beforeMasked = maskedCount;
                UnmaskDecision decision;
                using (ITensor logits = _model.Forward(
                           seqTokens.AsSpan(0, seqLen),
                           positions.AsSpan(0, seqLen),
                           deviceId: -1, kvCache: null, adapter: null, hybrid))
                {
                    // Capture the FULL canvas-region logits ([blockLen × vocab], rows
                    // prefixLen..prefixLen+blockLen) for the NEXT step's self-conditioning.
                    // These are post-softcap and NOT mask-suppressed (SC uses the full
                    // distribution; mask-suppression is only for the unmask COMMIT below).
                    CaptureCanvasLogits(logits, prefixLen, blockLen, vocabSize, scPrevCanvasLogits);
                    scValid = true;

                    // Gather logit rows for the still-masked canvas positions only.
                    int rows = GatherMaskedRows(canvas, maskTokenId, prefixLen, vocabSize, logits,
                        maskedPositions, maskedLogits);

                    var ctx = scheduler.CreateStepContext(step, rows, _logitSoftCap);
                    decision = _sampler.SelectAndSample(
                        maskedLogits.AsSpan(0, rows * vocabSize),
                        maskedPositions.AsSpan(0, rows),
                        vocabSize, rows, ctx);
                }

                // Scatter committed tokens back into the canvas (absorbing: frozen thereafter).
                var committed = decision.UnmaskedPositions;
                var tokens = decision.TokenIds;
                for (int i = 0; i < committed.Count; i++)
                {
                    int pos = committed[i];
                    if (canvas[pos] == maskTokenId)
                    {
                        canvas[pos] = tokens[i];
                        maskedCount--;
                    }
                }

                bool canvasChanged = maskedCount < beforeMasked;
                if (onCanvasStep is not null)
                    EmitCanvasState(onCanvasStep, canvas, maskTokenId, prefixLen, blockLen, canvasIndex,
                        step, maskedCount, decision.AverageEntropy, completed: false);

                stop = scheduler.ShouldStop(step, maskedCount, decision.AverageEntropy, canvasChanged);
                if (stop != DenoiseStopResult.Continue)
                    break;
            }

            // `step` is the zero-based index of the last executed step; the scheduler's MaxSteps cap
            // guarantees a break by step == MaxDenoisingSteps - 1, but clamp defensively in case the
            // for-loop ever exits naturally (then step == MaxDenoisingSteps).
            int executedSteps = Math.Min(step + 1, _diffusion.MaxDenoisingSteps);
            int lastStepIndex = executedSteps - 1;
            if (stop == DenoiseStopResult.Continue)
                stop = DenoiseStopResult.MaxSteps;

            // Any positions still masked at stop (e.g. a Stability/Confidence early stop) are not
            // emitted as the mask token: commit their argmax so the block is fully materialised.
            // In practice the proportional budget unmasks everything by MaxSteps; this is a guard.
            if (maskedCount > 0)
                FillRemainingMasked(context, prefixLen, blockLen, vocabSize, maskTokenId, canvas, ref maskedCount);

            // Final completion snapshot for the streaming consumer.
            if (onCanvasStep is not null)
                EmitCanvasState(onCanvasStep, canvas, maskTokenId, prefixLen, blockLen, canvasIndex,
                    lastStepIndex, maskedCount, averageEntropy: 0f, completed: true);

            return new CanvasRun(canvas, executedSteps, stop);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(seqTokens);
            ArrayPool<int>.Shared.Return(positions);
            ArrayPool<int>.Shared.Return(maskedPositions);
            ArrayPool<float>.Shared.Return(maskedLogits);
            ArrayPool<float>.Shared.Return(scPrevCanvasLogits);
            // Clear the model's SC state so a later non-diffusion / next-canvas forward
            // starts clean (the next RunCanvas re-primes step 0 with scUse=0 anyway).
            _model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
        }
    }

    /// <summary>
    /// Copies the canvas-region logit rows ([blockLen × vocab], sequence rows
    /// prefixLen..prefixLen+blockLen) out of the forward result into
    /// <paramref name="dst"/> for the next denoise step's self-conditioning. Logits are
    /// taken AS-IS (post-softcap, full distribution — no mask suppression).
    /// </summary>
    private static unsafe void CaptureCanvasLogits(
        ITensor logits, int prefixLen, int blockLen, int vocabSize, float[] dst)
    {
        float* basePtr = (float*)logits.DataPointer;
        var src = new ReadOnlySpan<float>(basePtr + (long)prefixLen * vocabSize, blockLen * vocabSize);
        src.CopyTo(dst.AsSpan(0, blockLen * vocabSize));
    }

    /// <summary>
    /// Copies the logit rows of every still-masked canvas position into <paramref name="maskedLogits"/>
    /// (row-major, one row per masked position) and records their canvas indices in
    /// <paramref name="maskedPositions"/>. Returns the number of masked rows.
    /// </summary>
    private static unsafe int GatherMaskedRows(
        int[] canvas, int maskTokenId, int prefixLen, int vocabSize,
        ITensor logits, int[] maskedPositions, float[] maskedLogits)
    {
        // CPU TransformerModel returns [seqLen, vocabSize]; the canvas row for canvas index i is at
        // sequence row prefixLen + i.
        float* basePtr = (float*)logits.DataPointer;
        int rows = 0;
        for (int i = 0; i < canvas.Length; i++)
        {
            if (canvas[i] != maskTokenId)
                continue;
            int seqRow = prefixLen + i;
            var src = new ReadOnlySpan<float>(basePtr + (long)seqRow * vocabSize, vocabSize);
            Span<float> dst = maskedLogits.AsSpan(rows * vocabSize, vocabSize);
            src.CopyTo(dst);
            // Suppress the mask token from the candidate distribution: a denoise step must
            // commit a REAL token, never the mask token. Models whose vocab contains the
            // mask token and predict it as the argmax at still-masked positions (e.g.
            // DiffusionGemma, mask id 4) would otherwise "unmask" a position back to the
            // mask token — a no-op that stalls the canvas at all-masked. Harmless for
            // absorbing-state models (LLaDA) that never rank the mask token highly.
            if ((uint)maskTokenId < (uint)vocabSize)
                dst[maskTokenId] = float.NegativeInfinity;
            maskedPositions[rows] = i;
            rows++;
        }
        return rows;
    }

    /// <summary>
    /// Final-step guard: forces any positions left masked at loop exit to their argmax token from one
    /// more cacheless hybrid forward, so a finished canvas never carries the mask token.
    /// </summary>
    private void FillRemainingMasked(
        List<int> context, int prefixLen, int blockLen, int vocabSize, int maskTokenId,
        int[] canvas, ref int maskedCount)
    {
        int seqLen = prefixLen + blockLen;
        int[] seqTokens = ArrayPool<int>.Shared.Rent(seqLen);
        int[] positions = ArrayPool<int>.Shared.Rent(seqLen);
        try
        {
            for (int i = 0; i < prefixLen; i++) { seqTokens[i] = context[i]; positions[i] = i; }
            for (int i = 0; i < blockLen; i++) { seqTokens[prefixLen + i] = canvas[i]; positions[prefixLen + i] = prefixLen + i; }

            using ITensor logits = _model.Forward(
                seqTokens.AsSpan(0, seqLen), positions.AsSpan(0, seqLen),
                deviceId: -1, kvCache: null, adapter: null, CanvasMaskSpec(prefixLen));

            unsafe
            {
                float* basePtr = (float*)logits.DataPointer;
                for (int i = 0; i < blockLen; i++)
                {
                    if (canvas[i] != maskTokenId)
                        continue;
                    var row = new ReadOnlySpan<float>(basePtr + (long)(prefixLen + i) * vocabSize, vocabSize);
                    canvas[i] = System.Numerics.Tensors.TensorPrimitives.IndexOfMax(row);
                    maskedCount--;
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(seqTokens);
            ArrayPool<int>.Shared.Return(positions);
        }
    }

    private void EmitCanvasState(
        Action<DiffusionCanvasState> onCanvasStep, int[] canvas, int maskTokenId,
        int prefixLen, int blockLen, int canvasIndex, int step, int maskedCount,
        float averageEntropy, bool completed)
    {
        // Snapshot the canvas (the live array is mutated across steps). Decode the committed prefix
        // — the contiguous run of unmasked positions from the start — for a stable partial preview.
        int[] snapshot = canvas.AsSpan(0, blockLen).ToArray();
        int decodeLen = 0;
        while (decodeLen < blockLen && snapshot[decodeLen] != maskTokenId)
            decodeLen++;
        string partial = decodeLen > 0
            ? _tokenizer.Decode(snapshot.AsSpan(0, decodeLen), stripBosSpace: false)
            : string.Empty;

        onCanvasStep(new DiffusionCanvasState(
            CanvasIndex: canvasIndex,
            Step: step,
            MaskedCount: maskedCount,
            CanvasLength: blockLen,
            AverageEntropy: averageEntropy,
            Completed: completed,
            Canvas: snapshot,
            PartialText: partial));
    }

    private readonly record struct CanvasRun(int[] Canvas, int Steps, DenoiseStopResult StopReason);
}

/// <summary>
/// Live snapshot of a diffusion canvas, delivered to the streaming callback after each denoise step
/// and on canvas completion.
/// </summary>
/// <param name="CanvasIndex">Zero-based index of this canvas within a multi-canvas run.</param>
/// <param name="Step">Zero-based denoise step that produced this snapshot.</param>
/// <param name="MaskedCount">Number of positions still masked after this step.</param>
/// <param name="CanvasLength">Number of positions in this canvas.</param>
/// <param name="AverageEntropy">Average canvas entropy reported this step (0 on the completion snapshot).</param>
/// <param name="Completed">True for the final snapshot emitted when the canvas finished.</param>
/// <param name="Canvas">Defensive copy of the canvas token ids (mask token for still-masked positions).</param>
/// <param name="PartialText">Decoded text of the leading contiguous run of committed tokens.</param>
public readonly record struct DiffusionCanvasState(
    int CanvasIndex,
    int Step,
    int MaskedCount,
    int CanvasLength,
    float AverageEntropy,
    bool Completed,
    int[] Canvas,
    string PartialText);

/// <summary>Per-canvas statistics for a completed diffusion run.</summary>
/// <param name="CanvasIndex">Zero-based canvas index within the run.</param>
/// <param name="CanvasLength">Number of positions in this canvas.</param>
/// <param name="Steps">Number of denoise steps this canvas consumed.</param>
/// <param name="StopReason">Why this canvas stopped denoising.</param>
public readonly record struct DiffusionCanvasStats(
    int CanvasIndex,
    int CanvasLength,
    int Steps,
    DenoiseStopResult StopReason);

/// <summary>Result of a masked text-diffusion generation run.</summary>
public sealed record DiffusionResult
{
    /// <summary>Decoded generated text (excludes the prompt; excludes a terminating EOS).</summary>
    public required string Text { get; init; }

    /// <summary>Generated token ids in canvas order across all canvases (EOS excluded).</summary>
    public required int[] GeneratedTokenIds { get; init; }

    /// <summary>Number of prompt tokens fed into the first canvas's causal prefix.</summary>
    public required int PromptTokenCount { get; init; }

    /// <summary>Number of generated tokens (== <see cref="GeneratedTokenIds"/> length).</summary>
    public required int GeneratedTokenCount { get; init; }

    /// <summary>Total denoise steps summed across every canvas.</summary>
    public required int TotalDenoisingSteps { get; init; }

    /// <summary>Number of canvases produced (≥ 2 for multi-canvas runs).</summary>
    public required int CanvasCount { get; init; }

    /// <summary>Why generation finished overall (Stop on EOS, Length otherwise).</summary>
    public required FinishReason FinishReason { get; init; }

    /// <summary>Per-canvas statistics, in canvas order.</summary>
    public required DiffusionCanvasStats[] CanvasStats { get; init; }
}
