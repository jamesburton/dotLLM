using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Thinking budget (#1292), llama.cpp <c>--reasoning-budget</c> semantics (<c>common/reasoning-budget.cpp</c>): once a
/// think block has run for <see cref="ThinkingBudget.Budget"/> tokens without closing, the next tokens are forced to
/// <see cref="ThinkingBudget.Message"/> followed by the close tag, and the answer continues normally. Budget 0 closes the
/// block immediately. A block that closes on its own before the budget is left alone.
/// </summary>
/// <remarks>
/// Stateful per sequence: it advances over the tokens generated since its last call, so a pipeline (one per sequence)
/// owns one instance. A rollback (fewer tokens than already seen, e.g. rejected drafts) replays from the start. Forcing
/// masks every other logit to -infinity, so greedy and stochastic samplers alike pick the forced token.
/// </remarks>
public sealed class ThinkingBudgetProcessor : ILogitProcessor
{
    private enum State
    {
        Idle,
        Counting,
        WaitingUtf8,
        Forcing,
        Done,
    }

    private readonly ITokenizer _tokenizer;
    private readonly int[] _open;
    private readonly int[] _close;
    private readonly int[] _forced;
    private readonly int _budget;
    private readonly bool _startsInside;

    private State _state;
    private int _remaining;
    private int _openPos;
    private int _closePos;
    private int _forcePos;
    private int _seen;

    /// <summary>Creates the processor for one sequence.</summary>
    /// <param name="budget">The budget and the model's think markup.</param>
    /// <param name="tokenizer">Tokenizer used to encode the tags and the message, and to check UTF-8 completeness.</param>
    public ThinkingBudgetProcessor(ThinkingBudget budget, ITokenizer tokenizer)
    {
        _tokenizer = tokenizer;
        _open = tokenizer.EncodeRaw(budget.OpenTag);
        _close = tokenizer.EncodeRaw(budget.CloseTag);
        _forced = [.. tokenizer.EncodeRaw(budget.Message ?? ""), .. _close];
        _budget = Math.Max(0, budget.Budget);
        _startsInside = budget.StartsInside;
        Reset();
    }

    /// <summary>True once the budget forced the close sequence (for logging and tests).</summary>
    public bool Forced { get; private set; }

    /// <inheritdoc/>
    public void Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context)
    {
        if (previousTokens.Count < _seen)
            Reset();
        for (; _seen < previousTokens.Count; _seen++)
            Accept(previousTokens[_seen]);

        if (_state != State.Forcing || _forcePos >= _forced.Length)
            return;

        int forced = _forced[_forcePos];
        if ((uint)forced >= (uint)logits.Length)
            return;
        float keep = logits[forced];
        logits.Fill(float.NegativeInfinity);
        logits[forced] = float.IsFinite(keep) ? keep : 0f;
        Forced = true;
    }

    private void Reset()
    {
        _seen = 0;
        _openPos = _closePos = _forcePos = 0;
        _state = State.Idle;
        if (_startsInside)
            Activate();
    }

    private void Activate()
    {
        _remaining = _budget;
        _state = _remaining <= 0 ? State.Forcing : State.Counting;
        _forcePos = 0;
    }

    private void Accept(int token)
    {
        switch (_state)
        {
            case State.Idle:
                if (Advance(_open, ref _openPos, token))
                    Activate();
                break;

            case State.Counting:
            case State.WaitingUtf8:
                if (Advance(_close, ref _closePos, token))
                {
                    _state = State.Done; // closed on its own
                    break;
                }

                // Do not force mid-character: a budget can land inside a multi-token UTF-8 sequence.
                bool complete = !_tokenizer.DecodeToken(token).Contains('�');
                if (_state == State.WaitingUtf8)
                {
                    if (complete)
                        _state = State.Forcing;
                }
                else if (--_remaining <= 0)
                {
                    _state = complete ? State.Forcing : State.WaitingUtf8;
                }

                break;

            case State.Forcing:
                if (++_forcePos >= _forced.Length)
                    _state = State.Done;
                break;
        }
    }

    /// <summary>Advances a token-sequence matcher; true when <paramref name="seq"/> just completed.</summary>
    private static bool Advance(int[] seq, ref int pos, int token)
    {
        if (seq.Length == 0)
            return false;
        if (token == seq[pos])
        {
            if (++pos >= seq.Length)
            {
                pos = 0;
                return true;
            }

            return false;
        }

        pos = token == seq[0] ? 1 : 0;
        return false;
    }
}
