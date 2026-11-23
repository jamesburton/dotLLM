using System.Buffers;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Engine.KvCache;
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
/// session is faulted and only <see cref="RollbackTo"/>(0) or disposal remain.</para>
/// <para>Paused sessions are local to the process and model instance. Sampler history (repetition penalties, DRY, Gumbel index) is session-wide: every committed token, prompt and observation included.</para>
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
    // Repeat-guard state per position (#1919): _runAt[k] = length of the run of identical SAMPLED tokens ending at position k-1 (0 when token k-1 was
    // prompt / observation, or k == 0). The guard's live (token, run) is always derivable from (_history, _runAt, _position), which is what makes
    // RollbackTo restore it exactly.
    private int[] _runAt = new int[65];
    private readonly RepeatGuardStopCondition? _repeatGuard;
    private readonly CommittedView _committed;
    private int _position;
    private readonly float[] _nextLogits;
    private int _logitsPos = -1;           // position _nextLogits belongs to; -1 = none
    private readonly float[] _sampleScratch;
    private int _busy;                   // 0 idle, 1 an operation is running, 2 disposing / disposed
    private int _disposed;               // set when disposal is requested; new operations are refused from then on
    private bool _faulted;
    private readonly int _maxContext;    // hard KV capacity ceiling the session may grow to
    private readonly Func<CacheProvenanceTag>? _currentProvenance;

    /// <summary>Most boundary records (position, recurrent checkpoint, logits) a recurrent session keeps; the oldest are evicted.</summary>
    public const int MaxRollbackBoundaries = 16;
    /// <summary>Default checkpoint budget: 1 GiB. On a GPU the recurrent-state checkpoints are DEVICE memory, so this is VRAM the session may hold.</summary>
    public const long DefaultCheckpointBudgetBytes = 1L << 30;

    private readonly bool _recurrent;
    private readonly bool _checkpointing;                // recurrent AND the model can checkpoint
    private readonly int _maxBoundaries;
    private readonly List<Boundary> _boundaries = [];
    private int _skippedBoundaries;

    /// <summary>Sampler history: every committed token (prompt, generated, observation), session-wide, as the model saw them.</summary>
    private sealed class CommittedView(GenerationSession s) : IReadOnlyList<int>
    {
        public int Count => s._position;
        public int this[int index] => (uint)index < (uint)s._position ? s._history[index] : throw new ArgumentOutOfRangeException(nameof(index));
        public IEnumerator<int> GetEnumerator() { for (int i = 0; i < s._position; i++) yield return s._history[i]; }
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    private sealed class Boundary(int position, object? checkpoint, float[] logits) : IDisposable
    {
        public int Position { get; } = position;
        public object? Checkpoint { get; } = checkpoint;
        public float[] Logits { get; } = logits;
        public void Dispose() => (Checkpoint as IDisposable)?.Dispose();
    }

    internal GenerationSession(TextGenerator owner, InferenceOptions options, CacheProvenanceTag tag,
        Func<CacheProvenanceTag>? currentProvenance, IKvCache kv, int maxContext, long checkpointBudgetBytes)
    {
        _maxContext = Math.Max(maxContext, kv.MaxLength);
        _currentProvenance = currentProvenance;
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
        _repeatGuard = RepeatGuardStopCondition.Create(options.RepeatGuardLimit);   // null limit = the engine default, 0 disables, as TextGenerator
        _committed = new CommittedView(this);
        _recurrent = _model.RequiresPerSequenceState;
        _checkpointing = _recurrent && _model.SupportsRecurrentStateCheckpoint;
        long perBoundary = _model.RecurrentStateCheckpointBytes + (long)_vocab * sizeof(float);
        _maxBoundaries = _checkpointing
            ? (int)Math.Clamp(checkpointBudgetBytes / Math.Max(1, perBoundary), 1, MaxRollbackBoundaries)
            : 0;
    }

    /// <summary>Boundaries whose checkpoint failed (admission or allocation) and were not recorded: those positions are not rollback targets.</summary>
    internal int SkippedBoundaries => _skippedBoundaries;

    /// <summary>Current KV capacity in tokens (grows on demand up to the model maximum / <c>maxContextTokens</c>).</summary>
    internal int KvCapacity => _kv.MaxLength;

    /// <summary>Number of tokens committed to the session; equals the KV length at every pause.</summary>
    public int CurrentPosition => _position;

    /// <summary>The provenance the session's cached state was created under.</summary>
    public CacheProvenanceTag ProvenanceTag { get; }

    /// <summary>
    /// Throws <see cref="InvalidOperationException"/> naming the differing fields when <paramref name="current"/> is not
    /// identical to the <see cref="ProvenanceTag"/> the session was created under. Called automatically with the live view
    /// on every operation when a provider was given to <see cref="TextGenerator.CreateSession"/>.
    /// </summary>
    public void ValidateProvenance(CacheProvenanceTag current)
    {
        ArgumentNullException.ThrowIfNull(current);
        var diff = ProvenanceTag.Diff(current);
        if (diff.Count > 0)
            throw new InvalidOperationException(
                $"Cache provenance mismatch: {string.Join(", ", diff)} differ{(diff.Count == 1 ? "s" : "")} from the tag this session's KV / recurrent state was built under. "
                + "Open a new session (or restore the original identity) instead of reusing stale state.");
    }

    /// <summary>KV-cache length; equals <see cref="CurrentPosition"/> at every pause (tests assert it).</summary>
    internal int KvLength => _kv.CurrentLength;

    /// <summary>The KV-cache this session owns (tests assert its disposal).</summary>
    internal IKvCache KvForTest => _kv;

    /// <summary>The recurrent-state checkpoints currently held at boundaries (tests assert their disposal).</summary>
    internal IReadOnlyList<object?> CheckpointsForTest => [.. _boundaries.Select(b => b.Checkpoint)];

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
            if (!EnsureCapacity(start + tokens.Length))
                throw new InvalidOperationException(
                    $"{tokens.Length} tokens at position {start} exceed the session's KV capacity of {_kv.MaxLength} (ceiling {_maxContext}; a cache that cannot grow needs an explicit maxContextTokens).");
            EnsureHistory(start + tokens.Length);
            tokens.Span.CopyTo(_history.AsSpan(start));
            Array.Clear(_runAt, start + 1, tokens.Length);   // externally supplied tokens are not sampled: they end a run
            try
            {
                using ITensor logits = _owner.SessionForwardPrefill(_history, start, tokens.Length, _kv, ct);
                StoreLogits(logits, start + tokens.Length);
            }
            catch (Exception ex)
            {
                // A model that restored its own state (IForwardStateRestored) at the first chunk needs no recovery; a later chunk
                // left the earlier chunks in KV, which Recover cuts back to the start.
                if (!(ex is IForwardStateRestored && _kv.CurrentLength == start)) Recover(start);
                throw;
            }
            _position = start + tokens.Length;
            SyncGuard();   // the appended tokens end any sampled run
            RecordBoundary();
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
            // options.StopSequences (with ReasoningStopGate) apply per segment, as TextGenerator applies them per request (#1931 review I-4).
            conditions.AddRange(StopStringCondition.CreateAll(_options));
               // #1291 semantics, state carried across segments
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
                    if (!EnsureCapacity(_position + 1)) { finish = FinishReason.Length; break; }

                    _nextLogits.CopyTo(_sampleScratch, 0);
                    int token = _pipeline.Sample(_sampleScratch, _committed);
                    generated.Add(token);
                    detok.Append(token);
                    var tail = detok.GetTailView(tailSize, scratch);
                    // The guard sees EVERY sampled token (CheckStopConditions short-circuits, and a segment's own stop condition often ends it first); its
                    // verdict applies only when nothing else stopped the token. #1291 semantics: the limit-th identical token in a row ends the segment, kept.
                    var guard = _repeatGuard?.ShouldStop(token, generated, tail) ?? StopResult.Continue;
                    var result = TextGenerator.CheckStopConditions(conditions, token, generated, tail);
                    if (result == StopResult.Continue) result = guard;

                    if (result == StopResult.Stop && !TextGenerator.HasStopStringSuffix(tail, conditions))
                    {
                        generated.RemoveAt(generated.Count - 1);   // EOS-like: not committed, logits at _position stay valid
                        finish = FinishReason.Stop;
                        break;
                    }

                    CommitAndForward(token);
                    if (_repeatGuard is not null) _runAt[_position] = _repeatGuard.State.Run;
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

                SyncGuard();   // an EOS-like stop evaluated the guard on a token that was not committed
                RecordBoundary();
                return ValueTask.FromResult(new GenerationSegmentResult(text, generated.ToArray(), finish, startPos, _position));
            }
            catch (Exception ex) when (ex is IForwardStateRestored)
            {
                // The model undid the failed forward: the session sits at a clean pause (tokens committed before it stay committed).
                SyncGuard();
                RecordBoundary();
                throw;
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

    /// <summary>
    /// Releases the KV-cache, detaches from the generator and resets model-owned recurrent state. Serialised with the busy guard:
    /// when an operation is in flight on another thread, new operations are refused at once and the release runs when that
    /// operation ends (never while a forward reads the buffers); the returned task completes immediately either way. The generator
    /// is detached even when releasing a buffer throws.
    /// </summary>
    public ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return ValueTask.CompletedTask;
        TryFinishDispose();
        return ValueTask.CompletedTask;
    }

    /// <summary>Runs the release if no operation holds the guard (CAS 0 to 2); otherwise <see cref="End"/> runs it when the operation ends.</summary>
    private void TryFinishDispose()
    {
        if (Interlocked.CompareExchange(ref _busy, 2, 0) != 0) return;
        try { _kv.Dispose(); }
        finally
        {
            try { DropBoundariesAfter(-1); }
            finally
            {
                try { if (_model.RequiresPerSequenceState) _model.ResetSequenceState(); }
                finally { _owner.ReleaseSession(); }
            }
        }
    }

    /// <summary>
    /// Positions a recurrent session can roll back to (the pauses it recorded a recurrent-state checkpoint at, oldest first).
    /// Empty for an attention-only model (any position is legal there, see <see cref="CanRollbackTo"/>).
    /// </summary>
    public IReadOnlyList<int> RollbackBoundaries
    {
        get
        {
            var list = new List<int>(_boundaries.Count);
            foreach (var b in _boundaries) list.Add(b.Position);
            return list;
        }
    }

    /// <summary>
    /// True when <see cref="RollbackTo"/> would succeed for <paramref name="tokenPosition"/> (same provenance and disposal checks as every operation): always for 0 and the current position;
    /// otherwise, for an attention-only model, when the KV-cache can be cut to <c>tokenPosition - 1</c>; for a recurrent model, when
    /// <paramref name="tokenPosition"/> is a recorded boundary and the cache can be cut to it.
    /// </summary>
    public bool CanRollbackTo(int tokenPosition)
    {
        Begin(allowFaulted: true);
        try { return CheckRollback(tokenPosition) is null; }
        finally { End(); }
    }

    /// <summary>
    /// Discards every token at or after <paramref name="tokenPosition"/> from KV, recurrent state and next-token logits, leaving the
    /// session as it was at that position; nothing of the kept prefix is re-prefilled.
    /// <list type="bullet">
    /// <item>Attention-only models: any <c>0 &lt;= k &lt;= CurrentPosition</c>; one token is re-forwarded to regenerate the logits.</item>
    /// <item>Recurrent models: only a recorded boundary (<see cref="RollbackBoundaries"/>) or 0; state, KV and logits are restored with no forward.</item>
    /// </list>
    /// Throws <see cref="ArgumentOutOfRangeException"/> outside <c>[0, CurrentPosition]</c> and <see cref="NotSupportedException"/>
    /// (changing nothing) where the model or KV-cache cannot do it exactly.
    /// </summary>
    public void RollbackTo(int tokenPosition)
    {
        Begin(allowFaulted: true);
        try
        {
            if ((uint)tokenPosition > (uint)_position)
                throw new ArgumentOutOfRangeException(nameof(tokenPosition), tokenPosition, $"Must be in [0, {_position}].");
            if (CheckRollback(tokenPosition) is { } refusal)
                throw new NotSupportedException(refusal);
            if (tokenPosition == _position && !(_faulted && tokenPosition == 0)) return;

            if (tokenPosition == 0)
            {
                // A full reset: the one recovery for a faulted session (it clears the fault only after every reset step succeeded).
                _kv.Rollback(0);
                if (_recurrent) _model.ResetSequenceState();
                DropBoundariesAfter(-1);
                _position = 0;
                _logitsPos = -1;
                _faulted = false;
                return;
            }

            if (_recurrent)
            {
                var b = FindBoundary(tokenPosition)!;
                _model.RestoreRecurrentState(b.Checkpoint);
                _kv.Rollback(tokenPosition);
                b.Logits.CopyTo(_nextLogits, 0);
                _logitsPos = tokenPosition;
                _position = tokenPosition;
                DropBoundariesAfter(tokenPosition);
                return;
            }

            // Attention only: KV rows [0, k-1) stay, token k-1 is re-forwarded for the logits (it rewrites its own KV row).
            int last = tokenPosition - 1;
            _kv.Rollback(last);
            _tokBuf[0] = _history[last];
            _posBuf[0] = last;
            try
            {
                using ITensor logits = _model.Forward(_tokBuf, _posBuf, deviceId: -1, _kv, adapter: null, mtpState: null, lastTokenLogitsOnly: true);
                StoreLogits(logits, tokenPosition);
            }
            catch (Exception ex) when (ex is IForwardStateRestored)
            {
                // The model undid the failed forward. The KV cut is part of this rollback and cannot be undone, so the session sits
                // cleanly at last (KV length == position) with no next-token logits: append history[last] again, or roll back further.
                _position = last;
                _logitsPos = -1;
                throw;
            }
            catch
            {
                _faulted = true;   // KV already cut, position not: cannot be put back
                throw;
            }
            _position = tokenPosition;
        }
        finally { SyncGuard(); End(); }
    }

    /// <summary>Sets the live repeat guard to the state it had at <see cref="CurrentPosition"/>: the trailing sampled run recorded in <c>_runAt</c>.</summary>
    private void SyncGuard()
    {
        if (_repeatGuard is null) return;
        int run = _position == 0 ? 0 : _runAt[_position];
        _repeatGuard.Restore(run == 0 ? -1 : _history[_position - 1], run);
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
        catch (Exception ex) when (ex is IForwardStateRestored)
        {
            // The model restored KV and recurrent state to their values at forward entry: the token was never committed here either
            // (position and logits not advanced), so there is nothing to recover and the session is not faulted.
            throw;
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
                else if (FindBoundary(position) is { } b) _model.RestoreRecurrentState(b.Checkpoint);
                else _faulted = true;
            }
        }
        catch
        {
            _faulted = true;
        }
    }

    /// <summary>Null when rolling back to <paramref name="k"/> is possible, else the reason it is not.</summary>
    private string? CheckRollback(int k)
    {
        if (k < 0 || k > _position) return $"Position {k} is outside [0, {_position}].";
        if (_faulted) return k == 0 ? null : "The session is faulted (a failed forward could not be undone); only RollbackTo(0) or disposal remain.";
        if (k == _position || k == 0) return null;
        if (_recurrent)
        {
            if (!_checkpointing)
                return $"{_model.GetType().Name} is a recurrent model without recurrent-state checkpoint support; only RollbackTo(0) or the current position are possible.";
            if (FindBoundary(k) is null)
                return $"Position {k} is not a recorded boundary of this recurrent session (boundaries: {string.Join(", ", _boundaries.Select(b => b.Position))}); recurrent state exists only at pauses.";
            return KvCanTruncateTo(k);
        }
        return KvCanTruncateTo(k - 1);
    }

    private string? KvCanTruncateTo(int length) => KvTruncationRefusal(_kv, length);

    /// <summary>Null when <paramref name="kv"/> can be cut to <paramref name="length"/> exactly, else the reason (naming the cache type).</summary>
    internal static string? KvTruncationRefusal(IKvCache kv, int length)
    {
        bool ok = kv switch
        {
            SimpleKvCache simple => simple.CanTruncateTo(length),
            IPrefixReusableKvCache reusable => reusable.CanTruncateTo(length),
            IKvLengthRollback rollback => rollback.CanTruncateTo(length),
            _ => false,
        };
        return ok ? null
            : $"The session's KV-cache ({kv.GetType().Name}) cannot be cut back to {length} positions (rows already dropped or no truncation support).";
    }

    private Boundary? FindBoundary(int position)
    {
        foreach (var b in _boundaries)
            if (b.Position == position) return b;
        return null;
    }

    private void DropBoundariesAfter(int position)
    {
        for (int i = _boundaries.Count - 1; i >= 0 && _boundaries[i].Position > position; i--)
        {
            _boundaries[i].Dispose();
            _boundaries.RemoveAt(i);
        }
    }

    /// <summary>Records recurrent state + logits at the current pause so it can be rolled back to; no-op for attention-only models.</summary>
    private void RecordBoundary()
    {
        if (!_checkpointing || _logitsPos != _position || _position == 0 || FindBoundary(_position) is not null)
            return;
        object? checkpoint = null;
        for (int attempt = 0; ; attempt++)
        {
            try { checkpoint = _model.CheckpointRecurrentState(); break; }
            catch (Exception ex)
            {
                // Never throw after commit: the segment's tokens are already in KV. Free the oldest checkpoint and retry once,
                // else skip this boundary (the position is simply not a rollback target).
                if (attempt == 0 && _boundaries.Count > 0)
                {
                    _boundaries[0].Dispose();
                    _boundaries.RemoveAt(0);
                    continue;
                }
                _skippedBoundaries++;
                System.Diagnostics.Trace.TraceWarning(
                    $"GenerationSession: recurrent-state checkpoint at position {_position} failed ({ex.GetType().Name}: {ex.Message}); the position is not a rollback target.");
                return;
            }
        }
        _boundaries.Add(new Boundary(_position, checkpoint, (float[])_nextLogits.Clone()));
        while (_boundaries.Count > _maxBoundaries)
        {
            _boundaries[0].Dispose();
            _boundaries.RemoveAt(0);
        }
    }

    private void EnsureHistory(int needed)
    {
        if (_history.Length < needed)
            Array.Resize(ref _history, Math.Max(needed, _history.Length * 2));
        if (_runAt.Length < _history.Length + 1)
            Array.Resize(ref _runAt, _history.Length + 1);
    }

    private void EnsureLogitsValid()
    {
        if (_logitsPos != _position || _position == 0)
            throw new InvalidOperationException("No next-token logits yet: prefill at least one token first.");
    }

    /// <summary>Grows the KV-cache (up to the ceiling) so it holds <paramref name="needed"/> positions; false when it cannot.</summary>
    private bool EnsureCapacity(int needed)
    {
        if (needed <= _kv.MaxLength) return true;
        if (needed > _maxContext) return false;
        int target = (int)Math.Min(_maxContext, Math.Max((long)needed, (long)_kv.MaxLength * 2));
        bool ok;
        try
        {
            ok = _kv.TryGrowTo(target) || (target > needed && _kv.TryGrowTo(needed));
        }
        catch (Exception ex) when (ex is not OutOfMemoryException && ex is not ObjectDisposedException)
        {
            // Device admission refused the growth: report it as a refusal (growth is all-or-nothing).
            System.Diagnostics.Trace.TraceWarning($"GenerationSession: KV growth to {target} positions refused ({ex.GetType().Name}: {ex.Message}).");
            return false;
        }
        if (!ok)
            System.Diagnostics.Trace.TraceWarning($"GenerationSession: the KV-cache ({_kv.GetType().Name}) could not grow from {_kv.MaxLength} to {needed} positions.");
        return ok && _kv.MaxLength >= needed;
    }

    private void Begin(bool allowFaulted = false)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);   // lost the race against a disposal
            throw new InvalidOperationException("A GenerationSession runs one operation at a time.");
        }
        try
        {
            if (_faulted && !allowFaulted)
                throw new InvalidOperationException("The session is faulted (a failed forward could not be undone); dispose it.");
            if (_currentProvenance is not null)
                ValidateProvenance(_currentProvenance() ?? throw new InvalidOperationException("The provenance provider returned null."));
        }
        catch
        {
            End();
            throw;
        }
    }

    private void End()
    {
        Volatile.Write(ref _busy, 0);
        if (Volatile.Read(ref _disposed) != 0) TryFinishDispose();   // a dispose that arrived mid-operation runs now
    }
}
