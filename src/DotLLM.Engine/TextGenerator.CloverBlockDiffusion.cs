using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers.StopConditions;

namespace DotLLM.Engine;

/// <summary>How a Clover (<c>clover_moe</c>) model decodes (#1828).</summary>
public enum CloverDecodeMode
{
    /// <summary>Ordinary autoregressive decoding (the default).</summary>
    Causal,

    /// <summary>LowConfidence block diffusion (<see cref="CloverBlockDiffusion"/>): 32-token blocks resolved by bidirectional previews.</summary>
    Block,
}

/// <summary>Per-request statistics of a block-diffusion generation.</summary>
/// <param name="Blocks">Blocks resolved.</param>
/// <param name="Steps">Preview forwards (denoising steps) over all blocks.</param>
/// <param name="TokensReturned">Tokens returned to the caller.</param>
/// <param name="PrefillMs">Causal prompt commit.</param>
/// <param name="PreviewMs">Preview forwards.</param>
/// <param name="SelectMs">Host-side prediction, confidence and transfer.</param>
/// <param name="CommitMs">Causal block commits.</param>
/// <param name="DecodeMs">Everything after the prompt commit (previews, selection, commits, detokenizing).</param>
public readonly record struct CloverBlockStats(int Blocks, int Steps, int TokensReturned, double PrefillMs, double PreviewMs, double SelectMs,
                                              double CommitMs, double DecodeMs)
{
    /// <summary>Mean preview steps per block.</summary>
    public double StepsPerBlock => Blocks > 0 ? (double)Steps / Blocks : 0;

    /// <summary>Mean wall milliseconds per block (previews + selection + commit).</summary>
    public double MsPerBlock => Blocks > 0 ? DecodeMs / Blocks : 0;

    /// <summary>Returned tokens per second over the decode phase.</summary>
    public double TokensPerSecond => DecodeMs > 0 ? TokensReturned / (DecodeMs / 1000.0) : 0;
}

public sealed partial class TextGenerator
{
    private const string CloverDecodeEnv = "DOTLLM_CLOVER_DECODE";
    private CloverDecodeMode? _cloverDecodeMode;
    private bool _cloverUnsupportedWarned;
    private static readonly bool CloverTrace = Environment.GetEnvironmentVariable("DOTLLM_CLOVER_TRACE") == "1";

    /// <summary>
    /// Decoding mode for Clover models (#1828). Defaults to <see cref="CloverDecodeMode.Causal"/>; <c>DOTLLM_CLOVER_DECODE=block</c>
    /// selects block diffusion process-wide (so <c>run</c>, <c>chat</c> and <c>serve</c> share one switch). Ignored for other models,
    /// and for a Clover file without the preview tensors (a warning is printed once).
    /// </summary>
    public CloverDecodeMode CloverDecode
    {
        get => _cloverDecodeMode ??= ParseCloverDecode(Environment.GetEnvironmentVariable(CloverDecodeEnv));
        set => _cloverDecodeMode = value;
    }

    /// <summary>Block-diffusion fill threshold (HF default 0.95; <c>DOTLLM_CLOVER_THRESHOLD</c>).</summary>
    public float CloverThreshold { get; set; } =
        float.TryParse(Environment.GetEnvironmentVariable("DOTLLM_CLOVER_THRESHOLD"), System.Globalization.NumberStyles.Float,
                       System.Globalization.CultureInfo.InvariantCulture, out float t) ? t : CloverBlockDiffusion.DefaultThreshold;

    /// <summary>Statistics of the most recent block-diffusion request (null when none ran).</summary>
    public CloverBlockStats? LastCloverBlockStats { get; private set; }

    /// <summary>Parses a <c>--clover-decode</c> value (<c>causal</c> | <c>block</c>).</summary>
    public static CloverDecodeMode ParseCloverDecodeOption(string value) => ParseCloverDecode(value);

    internal static CloverDecodeMode ParseCloverDecode(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        null or "" or "causal" or "ar" => CloverDecodeMode.Causal,
        "block" or "diffusion" or "block-diffusion" => CloverDecodeMode.Block,
        _ => throw new ArgumentException($"{CloverDecodeEnv}='{value}' is not a decode mode (causal | block)."),
    };

    private bool UseCloverBlockDiffusion()
    {
        if (CloverDecode != CloverDecodeMode.Block || _model.Config.Clover is null) return false;
        if (_model is ICloverBlockPreview { SupportsBlockPreview: true }) return true;
        if (!_cloverUnsupportedWarned)
        {
            _cloverUnsupportedWarned = true;
            Console.Error.WriteLine("[clover] block-diffusion decoding was requested but this model/backend has no block preview "
                                    + "(it needs the official DiffuSpace GGUF on the CPU or Vulkan host); decoding causally.");
        }
        return false;
    }

    private async IAsyncEnumerable<GenerationToken> GenerateCloverBlockStreamingAsync(string prompt, InferenceOptions? options,
        [EnumeratorCancellation] CancellationToken cancellationToken, ILoraAdapter? adapter)
    {
        options ??= new InferenceOptions();
        if (adapter is not null) throw new NotSupportedException("Clover block-diffusion decoding does not support LoRA adapters.");
        if (_draftModel is not null) throw new NotSupportedException("Clover block-diffusion decoding does not combine with a draft model.");
        if (options.Logprobs) throw new NotSupportedException("Clover block-diffusion decoding does not return logprobs.");
        if (options.ResponseFormat is not null and not ResponseFormat.Text)
            throw new NotSupportedException("Clover block-diffusion decoding does not support constrained decoding (response_format).");

        int[] promptIds = _tokenizer.Encode(prompt);
        if (promptIds.Length == 0) promptIds = [_tokenizer.BosTokenId];
        int promptLen = promptIds.Length, maxTokens = options.MaxTokens;
        if (maxTokens <= 0) yield break;

        var sampling = options.Temperature > 0f
            ? new CloverBlockSampling(options.Temperature, options.TopK, options.TopP, options.Seed)
            : new CloverBlockSampling();
        var decoder = new CloverBlockDiffusion(_model, CloverThreshold, blockSize: 0, sampling);
        int needed = decoder.RequiredKvLength(promptLen, maxTokens);
        if (needed > _model.Config.MaxSequenceLength)
            throw new ArgumentException($"Prompt plus full preview blocks ({needed} tokens) exceeds the context length {_model.Config.MaxSequenceLength}.");

        List<IStopCondition> stopConditions = options.StopConditions is not null ? new(options.StopConditions) :
        [
            EndOfGenerationTokens.CreateStopCondition(_tokenizer),
            new MaxTokensStopCondition(maxTokens),
            .. StopStringCondition.CreateAll(options),
        ];
        // HF stops on generation_config eos_token_id = [<|im_end|>, <|endoftext|>]; the GGUF's EOS is only <|im_end|>, and block
        // diffusion fills block tails with <|endoftext|> (the pad id), which would otherwise be returned (and committed) as text.
        if (options.StopConditions is null && _model.Config.Clover is { PadTokenId: >= 0 } cc)
            stopConditions.Insert(1, new EosStopCondition(cc.PadTokenId));
        int stopTailSize = ComputeStopTailSize(stopConditions);
        char[] stopScratch = ArrayPool<char>.Shared.Rent(stopTailSize);

        // The preview parks its K/V in rows of this request's own cache and the recurrent prefix snapshots may share device storage
        // with it: drop them (block diffusion is opt-in; it trades the prefix cache away) and start from a zeroed state.
        ClearRecurrentPrefixCacheCore();
        _model.ResetSequenceState();
        var kvCache = AllocateKvCache(needed);
        long kvBytes = GetKvCacheBytes(kvCache);
        var generatedIds = new List<int>(maxTokens);
        var detok = new IncrementalDetokenizer(_tokenizer, initialCapacity: Math.Max(64, maxTokens * 4));
        long prefillTicks = 0, previewTicks = 0, selectTicks = 0, commitTicks = 0;
        int blocks = 0, steps = 0;
        long decodeStart = 0;
        try
        {
            // The loop itself (prompt commit, block resolution, causal commit between blocks) is CloverBlockDiffusion.Blocks, the one the
            // HF oracle and the mutants exercise; this consumer only applies the request's stop conditions and streams.
            void Prefill(int[] ids, Core.Attention.IKvCache kv)
            {
                long t0 = Stopwatch.GetTimestamp();
                _requestImages = null;
                ForwardPrefill(ids, 0, ids.Length, kv, adapter: null, cancellationToken: cancellationToken).Dispose();
                decodeStart = Stopwatch.GetTimestamp();
                prefillTicks = decodeStart - t0;
            }

            CloverBlockTurn? previous = null;
            foreach (var turn in decoder.Blocks(promptIds, maxTokens, kvCache, Prefill))
            {
                cancellationToken.ThrowIfCancellationRequested();
                commitTicks += previous?.CommitTicks ?? 0;   // the previous block was committed just before this one was resolved
                previous = turn;
                var resolved = turn.Block;
                blocks++;
                steps += resolved.Steps.Count;
                previewTicks += resolved.PreviewTicks;
                selectTicks += resolved.SelectTicks;
                if (CloverTrace)
                    Console.Error.WriteLine($"[clover] block {blocks} @{turn.Position}: " + string.Join(" ", resolved.Steps.Select(s =>
                        "[" + string.Join(",", s.Filled.Select((pos, n) => $"{pos}:{s.Tokens[n]}")) + (s.Fallback ? "]*" : "]"))));

                // HF: stop only after the whole block resolved; tokens past the stop are dropped.
                for (int i = 0; i < turn.Remaining; i++)
                {
                    int token = resolved.Tokens[i];
                    generatedIds.Add(token);
                    detok.Append(token);
                    var stop = CheckStopConditions(stopConditions, token, generatedIds, detok.GetTailView(stopTailSize, stopScratch));
                    if (stop != StopResult.Continue)
                    {
                        turn.Stop = true;
                        long end = Stopwatch.GetTimestamp();
                        LastCloverBlockStats = Stats(end);
                        if (stop == StopResult.Stop)
                        {
                            generatedIds.RemoveAt(generatedIds.Count - 1);
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, end - decodeStart, selectTicks, kvBytes);
                            yield return new GenerationToken(token, detok.TakeWithheld(), FinishReason.Stop, timings, GeneratedTokenCount: generatedIds.Count);
                        }
                        else
                        {
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, end - decodeStart, selectTicks, kvBytes);
                            yield return new GenerationToken(token, detok.TakeDelta(flushIncomplete: true), FinishReason.Length, timings,
                                                             GeneratedTokenCount: generatedIds.Count);
                        }
                        yield break;
                    }
                    yield return new GenerationToken(token, detok.TakeDelta(flushIncomplete: false), null);
                }
                if (generatedIds.Count >= maxTokens)
                {
                    long end = Stopwatch.GetTimestamp();
                    LastCloverBlockStats = Stats(end);
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, end - decodeStart, selectTicks, kvBytes);
                    yield return new GenerationToken(generatedIds[^1], detok.TakeDelta(flushIncomplete: true), FinishReason.Length, timings,
                                                     GeneratedTokenCount: generatedIds.Count);
                    yield break;
                }
            }
        }
        finally
        {
            ArrayPool<char>.Shared.Return(stopScratch);
            kvCache.Dispose();
            _model.ResetSequenceState();   // the next request (causal or not) must not inherit this sequence's recurrent state
            if (LastCloverBlockStats is { } s && Environment.GetEnvironmentVariable("DOTLLM_CLOVER_STATS") != "0")
                Console.Error.WriteLine($"[clover] block diffusion: {s.TokensReturned} tokens, {s.Blocks} blocks, {s.StepsPerBlock:F1} steps/block, "
                                        + $"{s.MsPerBlock:F0} ms/block (preview {s.PreviewMs:F0} ms, select {s.SelectMs:F0} ms, commit {s.CommitMs:F0} ms), "
                                        + $"{s.TokensPerSecond:F2} tok/s decode; prefill {s.PrefillMs:F0} ms");
        }

        CloverBlockStats Stats(long end)
        {
            double f = 1000.0 / Stopwatch.Frequency;
            return new CloverBlockStats(blocks, steps, generatedIds.Count, prefillTicks * f, previewTicks * f, selectTicks * f, commitTicks * f,
                                        (end - decodeStart) * f);
        }
    }

    private InferenceResponse GenerateCloverBlock(string prompt, InferenceOptions? options, Action<int>? onTokenGenerated, ILoraAdapter? adapter,
        Action<ReadOnlySpan<char>>? onTextGenerated, CancellationToken cancellationToken)
    {
        var ids = new List<int>();
        var text = new StringBuilder();
        FinishReason finish = FinishReason.Length;
        InferenceTimings timings = default;
        foreach (var token in GenerateCloverBlockStreamingAsync(prompt, options, cancellationToken, adapter).ToBlockingEnumerable(cancellationToken))
        {
            if (token.FinishReason is null || token.FinishReason == FinishReason.Length)
            {
                ids.Add(token.TokenId);
                onTokenGenerated?.Invoke(token.TokenId);
            }
            text.Append(token.Text);
            if (token.Text.Length > 0) onTextGenerated?.Invoke(token.Text);
            if (token.FinishReason is { } fr) { finish = fr; timings = token.Timings ?? default; }
        }
        string result = text.ToString();
        string? matched = null;
        if (finish == FinishReason.Stop && (options ?? new InferenceOptions()) is { } o)
        {
            var conditions = new List<IStopCondition>(StopStringCondition.CreateAll(o));
            matched = StopSuffixTrimmer.MatchedSuffix(result.AsSpan(), conditions);
            result = StopSuffixTrimmer.TrimMatchedSuffix(result, conditions);
        }
        return new InferenceResponse
        {
            GeneratedTokenIds = ids.ToArray(),
            Text = result,
            FinishReason = finish,
            PromptTokenCount = timings.PrefillTokenCount,
            GeneratedTokenCount = ids.Count,
            Timings = timings,
            MatchedStopSequence = matched,
        };
    }
}
