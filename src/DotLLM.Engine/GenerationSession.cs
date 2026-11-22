using System.Buffers;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Samplers;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// A resumable single-sequence generation (issue #1919). One prefill, then any number of: generate until a boundary,
/// append observation tokens (full-layer prefill, never sampled), roll back, without re-prefilling the committed prefix.
/// </summary>
/// <remarks>
/// <para><b>Invariant.</b> After every pause the KV length equals <see cref="CurrentPosition"/> and the next-token logits are valid.
/// A boundary token is therefore committed and forwarded (one extra forward at the end of a segment that stops on
/// <see cref="StopResult.StopInclude"/>, the token limit or a stop-string match). An end-of-generation or
/// <see cref="StopResult.Stop"/> token is not committed, exactly as <see cref="TextGenerator"/> drops it.</para>
/// <para><b>Threading.</b> One operation at a time; a concurrent call throws <see cref="InvalidOperationException"/>.
/// Operations complete synchronously (the model forward is synchronous); cancellation is checked per token and per prefill chunk.
/// A cancelled or failed prefill is undone to the position it started from when the cache and model can do that, otherwise the
/// session is faulted and only RollbackTo(0) or disposal remain.</para>
/// <para>Paused sessions are local to the process and model instance. Sampling history for repetition penalties is the current segment.</para>
/// </remarks>
public sealed class GenerationSession : IAsyncDisposable
{
    private readonly TextGenerator _owner;
    private readonly IModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly InferenceOptions _options;
    private readonly SamplerPipeline _pipeline;
    private readonly IKvCache _kv;
    private readonly int _vocab;
    private readonly int[] _tokBuf = new int[1];
    private readonly int[] _posBuf = new int[1];

    private int[] _history = new int[64];
    private int _position;
    private readonly float[] _nextLogits;
    private int _logitsPos = -1;           // position _nextLogits belongs to; -1 = none
    private readonly float[] _sampleScratch;
    private int _busy;
    private int _disposed;
    private bool _faulted;

    internal GenerationSession(TextGenerator owner, InferenceOptions options, CacheProvenanceTag tag, IKvCache kv)
    {
        _owner = owner;
        _model = owner.SessionModel;
        _tokenizer = owner.SessionTokenizer;
        _options = options;
        ProvenanceTag = tag;
        _kv = kv;
        _vocab = _model.Config.VocabSize;
        _nextLogits = new float[_vocab];
        _sampleScratch = new float[_vocab];
        _pipeline = new SamplerPipeline(options, _tokenizer);
    }

    /// <summary>Number of tokens committed to the session; equals the KV length at every pause.</summary>
    public int CurrentPosition => _position;

    /// <summary>The provenance the session's cached state was created under.</summary>
    public CacheProvenanceTag ProvenanceTag { get; }

    /// <summary>KV-cache length; equals <see cref="CurrentPosition"/> at every pause (tests assert it).</summary>
    internal int KvLength => _kv.CurrentLength;

    /// <summary>Width of the logits vector <see cref="CopyNextLogits"/> fills.</summary>
    public int VocabSize => _vocab;

    /// <summary>Prefills <paramref name="tokens"/> (the prompt). Identical to <see cref="AppendObservationTokensAsync"/>; the name documents intent.</summary>
    public ValueTask PrefillAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken = default)
        => AppendCore(tokens, cancellationToken);

    /// <summary>
    /// Runs full-layer prefill (KV and recurrent state, every layer) of externally supplied <paramref name="tokens"/> after the
    /// committed prefix. Observation tokens are never sampled and never constrained. Throws <see cref="InvalidOperationException"/>
    /// before any forward when they would not fit the KV capacity.
    /// </summary>
    public ValueTask AppendObservationTokensAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken = default)
        => AppendCore(tokens, cancellationToken);

    private ValueTask AppendCore(ReadOnlyMemory<int> tokens, CancellationToken ct)
    {
        Begin();
        try
        {
            ct.ThrowIfCancellationRequested();
            if (tokens.Length == 0) return ValueTask.CompletedTask;
            int start = _position;
            if (start + tokens.Length > _kv.MaxLength)
                throw new InvalidOperationException(
                    $"{tokens.Length} tokens at position {start} exceed the session's KV capacity of {_kv.MaxLength}.");
            EnsureHistory(start + tokens.Length);
            tokens.Span.CopyTo(_history.AsSpan(start));
            try
            {
                using ITensor logits = _owner.SessionForwardPrefill(_history, start, tokens.Length, _kv, ct);
                StoreLogits(logits, start + tokens.Length);
            }
            catch
            {
                Recover(start);
                throw;
            }
            _position = start + tokens.Length;
            return ValueTask.CompletedTask;
        }
        finally { End(); }
    }

    /// <summary>
    /// Samples and commits tokens until <paramref name="stopCondition"/>, an end-of-generation token, <see cref="InferenceOptions.MaxTokens"/>
    /// (per segment) or the KV capacity ends the segment, then pauses with KV length and next-token logits valid at
    /// <see cref="GenerationSegmentResult.EndPosition"/>.
    /// </summary>
    public ValueTask<GenerationSegmentResult> GenerateUntilBoundaryAsync(IStopCondition stopCondition, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stopCondition);
        Begin();
        try
        {
            EnsureLogitsValid();
            var conditions = new List<IStopCondition> { EndOfGenerationTokens.CreateStopCondition(_tokenizer), stopCondition };
            int tailSize = TextGenerator.ComputeStopTailSize(conditions);
            char[] scratch = ArrayPool<char>.Shared.Rent(tailSize);
            try
            {
                int startPos = _position;
                int max = _options.MaxTokens;
                var generated = new List<int>(Math.Clamp(max, 1, 256));
                var detok = new IncrementalDetokenizer(_tokenizer, initialCapacity: 64);
                var finish = FinishReason.Length;

                while (generated.Count < max)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (_position >= _kv.MaxLength) { finish = FinishReason.Length; break; }

                    _nextLogits.CopyTo(_sampleScratch, 0);
                    int token = _pipeline.Sample(_sampleScratch, generated);
                    generated.Add(token);
                    detok.Append(token);
                    var tail = detok.GetTailView(tailSize, scratch);
                    var result = TextGenerator.CheckStopConditions(conditions, token, generated, tail);

                    if (result == StopResult.Stop && !TextGenerator.HasStopStringSuffix(tail, conditions))
                    {
                        generated.RemoveAt(generated.Count - 1);   // EOS-like: not committed, logits at _position stay valid
                        finish = FinishReason.Stop;
                        break;
                    }

                    CommitAndForward(token);
                    if (result != StopResult.Continue)
                    {
                        finish = result == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                        break;
                    }
                }

                string text = generated.Count > 0
                    ? _tokenizer.Decode(CollectionsMarshal.AsSpan(generated), stripBosSpace: false)
                    : string.Empty;
                if (finish == FinishReason.Stop)
                    text = StopSuffixTrimmer.TrimMatchedSuffix(text, conditions);

                return ValueTask.FromResult(new GenerationSegmentResult(text, generated.ToArray(), finish, startPos, _position));
            }
            finally { ArrayPool<char>.Shared.Return(scratch); }
        }
        finally { End(); }
    }

    /// <summary>Index of the highest next-token logit (the raw model logits, before sampler steps or masks).</summary>
    public int NextTokenArgMax()
    {
        Begin();
        try
        {
            EnsureLogitsValid();
            int best = 0;
            for (int i = 1; i < _nextLogits.Length; i++)
                if (_nextLogits[i] > _nextLogits[best]) best = i;
            return best;
        }
        finally { End(); }
    }

    /// <summary>Copies the raw next-token logits (F32, <see cref="VocabSize"/> values) into <paramref name="destination"/>.</summary>
    public void CopyNextLogits(Span<float> destination)
    {
        Begin();
        try
        {
            EnsureLogitsValid();
            if (destination.Length < _vocab)
                throw new ArgumentException($"destination holds {destination.Length} floats; {_vocab} needed.", nameof(destination));
            _nextLogits.CopyTo(destination);
        }
        finally { End(); }
    }

    /// <summary>Releases the KV-cache, detaches from the generator and resets model-owned recurrent state.</summary>
    public ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return ValueTask.CompletedTask;
        _kv.Dispose();
        if (_model.RequiresPerSequenceState) _model.ResetSequenceState();
        _owner.ReleaseSession();
        return ValueTask.CompletedTask;
    }

    // ── internals ───────────────────────────────────────────────────────────────

    private void CommitAndForward(int token)
    {
        int pos = _position;
        EnsureHistory(pos + 1);
        _history[pos] = token;
        _tokBuf[0] = token;
        _posBuf[0] = pos;
        try
        {
            using ITensor logits = _model.Forward(_tokBuf, _posBuf, deviceId: -1, _kv, adapter: null, mtpState: null, lastTokenLogitsOnly: true);
            StoreLogits(logits, pos + 1);
        }
        catch
        {
            Recover(pos);
            throw;
        }
        _position = pos + 1;
    }

    private unsafe void StoreLogits(ITensor logits, int forPosition)
    {
        int rows = logits.Shape[0];
        var span = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * _vocab, _vocab);
        span.CopyTo(_nextLogits);
        _logitsPos = forPosition;
    }

    /// <summary>
    /// Puts KV and model state back at <paramref name="position"/> after a failed or cancelled forward. The position and logits
    /// fields were not advanced, so they already describe it; a state that cannot be restored faults the session.
    /// </summary>
    private void Recover(int position)
    {
        try
        {
            if (_kv.CurrentLength > position) _kv.Rollback(position);
            if (_model.RequiresPerSequenceState)
            {
                if (position == 0) _model.ResetSequenceState();
                else _faulted = true;
            }
        }
        catch
        {
            _faulted = true;
        }
    }

    private void EnsureHistory(int needed)
    {
        if (_history.Length < needed)
            Array.Resize(ref _history, Math.Max(needed, _history.Length * 2));
    }

    private void EnsureLogitsValid()
    {
        if (_logitsPos != _position || _position == 0)
            throw new InvalidOperationException("No next-token logits yet: prefill at least one token first.");
    }

    private void Begin()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
            throw new InvalidOperationException("A GenerationSession runs one operation at a time.");
        if (_faulted)
        {
            Volatile.Write(ref _busy, 0);
            throw new InvalidOperationException("The session is faulted (a failed forward could not be undone); dispose it.");
        }
    }

    private void End() => Volatile.Write(ref _busy, 0);
}
