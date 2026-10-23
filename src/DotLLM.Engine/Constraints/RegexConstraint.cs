using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using DotLLM.Core.Constraints;
using DotLLM.Engine.Constraints.Regex;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Constraints;

/// <summary>
/// Constrained decoding constraint that guarantees output matches a regex pattern.
/// Compiles the pattern to a minimized DFA at construction time.
/// Token masks are cached by DFA state ID — typically &lt;50 entries for practical patterns.
/// </summary>
public sealed class RegexConstraint : IDecodingConstraint
{
    private DfaSimulator _simulator;
    private readonly CompiledDfa _dfa;
    private readonly ITokenizer _tokenizer;
    private readonly int _vocabSize;
    private readonly int _eosTokenId;

    // Shared across clones AND (via Create) across requests with the same tokenizer and pattern. Masks are
    // immutable once built, so a concurrent dictionary is all the synchronisation needed.
    private readonly ConcurrentDictionary<int, TokenMask> _maskCache;

    /// <summary>Compiled DFA plus the per-state token masks, shared by every constraint made for one (tokenizer, pattern).</summary>
    private sealed class SharedState(CompiledDfa dfa)
    {
        public readonly CompiledDfa Dfa = dfa;
        public readonly ConcurrentDictionary<int, TokenMask> Masks = new();
    }

    private const int MaxCachedPatternsPerTokenizer = 64;
    private static readonly ConditionalWeakTable<ITokenizer, ConcurrentDictionary<string, SharedState>> Shared = new();

    /// <summary>
    /// Creates a constraint for <paramref name="pattern"/>, reusing the compiled DFA and — more importantly — the
    /// per-state token masks of earlier constraints for the same tokenizer and pattern. A mask build decodes and
    /// DFA-steps every token in the vocabulary (~10 ms on a 248k vocab), so rebuilding it per request cost more than
    /// the forward pass a one-letter classifier answer is trying to save.
    /// </summary>
    public static RegexConstraint Create(ITokenizer tokenizer, string pattern)
    {
        var perTokenizer = Shared.GetValue(tokenizer, _ => new ConcurrentDictionary<string, SharedState>());
        if (perTokenizer.Count >= MaxCachedPatternsPerTokenizer && !perTokenizer.ContainsKey(pattern))
            perTokenizer.Clear();   // unbounded distinct patterns must not grow memory without limit
        var shared = perTokenizer.GetOrAdd(pattern, p => new SharedState(CompilePattern(p)));
        return new RegexConstraint(tokenizer, shared.Dfa, shared.Masks);
    }

    /// <summary>
    /// Creates a regex constraint from a pattern string.
    /// Compiles: pattern → AST → Thompson NFA → minimized DFA.
    /// </summary>
    /// <param name="tokenizer">Tokenizer for decoding token IDs to text.</param>
    /// <param name="pattern">Regex pattern (anchored: entire output must match).</param>
    public RegexConstraint(ITokenizer tokenizer, string pattern)
        : this(tokenizer, CompilePattern(pattern))
    {
    }

    /// <summary>
    /// Creates a regex constraint from a pre-compiled DFA (for reuse across requests).
    /// </summary>
    internal RegexConstraint(ITokenizer tokenizer, CompiledDfa dfa)
        : this(tokenizer, dfa, new ConcurrentDictionary<int, TokenMask>())
    {
    }

    private RegexConstraint(ITokenizer tokenizer, CompiledDfa dfa, ConcurrentDictionary<int, TokenMask> maskCache)
    {
        _dfa = dfa;
        _simulator = new DfaSimulator(dfa);
        _tokenizer = tokenizer;
        _vocabSize = tokenizer.VocabSize;
        _eosTokenId = tokenizer.EosTokenId;
        _maskCache = maskCache;
    }

    /// <summary>Copy constructor for <see cref="Clone"/>.</summary>
    private RegexConstraint(RegexConstraint source)
    {
        _simulator = source._simulator; // struct — copies by value (single int)
        _dfa = source._dfa;
        _tokenizer = source._tokenizer;
        _vocabSize = source._vocabSize;
        _eosTokenId = source._eosTokenId;
        _maskCache = source._maskCache; // shared cache — masks are immutable once built
    }

    /// <inheritdoc/>
    public void Advance(int tokenId)
    {
        if (tokenId == _eosTokenId)
            return;

        string text = _tokenizer.DecodeToken(tokenId);
        foreach (char c in text)
        {
            bool ok = _simulator.TryAdvance(c);
            Debug.Assert(ok, $"Constraint allowed token that advances to dead state at char '{c}'");
        }
    }

    /// <inheritdoc/>
    public TokenMask GetAllowedTokens()
    {
        int stateKey = _simulator.State;

        // Dead state: nothing allowed
        if (stateKey < 0)
        {
            var deadMask = new TokenMask(_vocabSize);
            // Only EOS if we somehow got here (shouldn't happen with proper masking)
            return deadMask;
        }

        if (_maskCache.TryGetValue(stateKey, out var cached))
            return cached;
        return BuildAndCacheMask(stateKey);
    }

    /// <inheritdoc/>
    public bool IsComplete() => _simulator.IsAccepting;

    /// <inheritdoc/>
    public IDecodingConstraint Clone() => new RegexConstraint(this);

    /// <inheritdoc/>
    public void Reset()
    {
        _simulator.Reset();
        // Don't clear cache — it's still valid for the same DFA
    }

    private TokenMask BuildAndCacheMask(int stateKey)
    {
        var mask = new TokenMask(_vocabSize);

        for (int tokenId = 0; tokenId < _vocabSize; tokenId++)
        {
            if (tokenId == _eosTokenId)
            {
                // EOS allowed only when DFA is in an accepting state
                if (_simulator.IsAccepting)
                    mask.Allow(tokenId);
                continue;
            }

            if (IsTokenValid(tokenId))
                mask.Allow(tokenId);
        }

        _maskCache[stateKey] = mask;
        return mask;
    }

    private bool IsTokenValid(int tokenId)
    {
        string tokenText = _tokenizer.DecodeToken(tokenId);
        if (tokenText.Length == 0)
            return false;

        // Struct copy — zero allocations (DfaSimulator is just an int + reference)
        var clone = _simulator;
        foreach (char c in tokenText)
        {
            if (!clone.TryAdvance(c))
                return false;
        }
        return true;
    }

    private static CompiledDfa CompilePattern(string pattern)
    {
        var ast = RegexParser.Parse(pattern);
        var nfa = NfaBuilder.Build(ast);
        return DfaBuilder.Build(nfa);
    }
}
