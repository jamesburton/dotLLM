using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace DotLLM.Engine.Constraints.Schema;

/// <summary>
/// Holds all per-position schema-tracking state for a single JSON generation branch.
/// Extracted from <see cref="SchemaTracker"/> so that Task 4 can generalise to a
/// set of parallel branches without changing any tracking logic.
/// </summary>
internal struct BranchState
{
    private const int MaxDepth = 64;
    private const int MaxKeyLength = 128;

    private readonly CompiledSchema _schema;

    // Schema node index stack — parallel to parser's nesting stack.
    // Each entry is the schema node index for the containing object/array.
    private SchemaNodeIdxStack _nodeStack;
    private int _stackDepth;

    // Current schema node index for the value being generated.
    private int _currentNodeIndex;

    // Emitted property bitmask per object nesting level.
    private PropertyBitStack _emittedProps;

    // Key character buffer for matching property names after key string closes.
    private KeyCharBuffer _keyBuffer;
    private int _keyLength;

    // Trie position during key string generation.
    private int _trieNodeIndex;

    // Array item index per array nesting level (for future minItems/maxItems).
    private ArrayIndexStack _arrayIndices;

    // Enum/const trie position during value string generation.
    private int _enumTrieNodeIndex;

    // Track whether we are inside a key string (set on entry, cleared on exit).
    private bool _inKeyString;

    // Track whether we are inside a value string with enum/const constraint.
    private bool _inEnumString;

    /// <summary>
    /// Creates a new branch state for the given compiled schema.
    /// </summary>
    /// <param name="schema">The compiled schema (immutable, shared).</param>
    public BranchState(CompiledSchema schema)
    {
        _schema = schema;
        _currentNodeIndex = 0; // root node
        _stackDepth = 0;
        _keyLength = 0;
        _trieNodeIndex = 0;
        _enumTrieNodeIndex = 0;
        _inKeyString = false;
        _inEnumString = false;
    }

    /// <summary>
    /// Whether the schema is fully satisfied.
    /// </summary>
    public readonly bool IsComplete(in JsonCharParser parser) => parser.IsComplete;

    /// <summary>
    /// Checks if a character is allowed by the schema at the current position.
    /// Called BEFORE <see cref="JsonCharParser.TryAdvance"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IsCharAllowed(char c, in JsonCharParser parser)
    {
        var state = parser.State;

        return state switch
        {
            JsonParserState.Start => IsValueStartCharAllowed(c, _currentNodeIndex),
            JsonParserState.ValueStart => IsValueStartCharAllowed(c, _currentNodeIndex) || IsWhitespace(c),
            JsonParserState.ObjectOpen => IsObjectOpenCharAllowed(c),
            JsonParserState.ObjectNextKey => IsObjectNextKeyCharAllowed(c),
            JsonParserState.ObjectCommaOrClose => IsObjectCommaOrCloseCharAllowed(c),
            JsonParserState.ObjectColon => true, // parser handles syntax
            JsonParserState.InString => IsInStringCharAllowed(c, parser),
            JsonParserState.InStringEscape => true, // parser handles escape validation
            JsonParserState.InStringUnicode => true, // parser handles hex validation
            JsonParserState.InNumberSign => true,
            JsonParserState.InNumberZero => IsNumberContinuationAllowed(c),
            JsonParserState.InNumberIntDigits => IsNumberContinuationAllowed(c),
            JsonParserState.InNumberDot => true,
            JsonParserState.InNumberFracDigits => IsNumberContinuationAllowed(c),
            JsonParserState.InNumberExp => true,
            JsonParserState.InNumberExpSign => true,
            JsonParserState.InNumberExpDigits => IsNumberContinuationAllowed(c),
            JsonParserState.InLiteral => true, // parser validates literal chars
            JsonParserState.ArrayOpen => IsArrayOpenCharAllowed(c),
            JsonParserState.ArrayCommaOrClose => true, // parser handles syntax
            JsonParserState.ArrayNextValue => IsValueStartCharAllowed(c, GetArrayItemNodeIndex()) || IsWhitespace(c),
            JsonParserState.Done => true, // parser rejects everything at Done
            _ => true,
        };
    }

    /// <summary>
    /// Called AFTER a character has been successfully accepted by the JSON parser.
    /// Detects structural events and updates schema position accordingly.
    /// </summary>
    /// <param name="c">The character that was just accepted.</param>
    /// <param name="parser">The parser state AFTER advancing.</param>
    public void OnCharAdvanced(char c, in JsonCharParser parser)
    {
        var newState = parser.State;

        // Object opened: '{' → parser is now in ObjectOpen
        if (c == '{' && newState == JsonParserState.ObjectOpen)
        {
            PushObject();
            return;
        }

        // Array opened: '[' → parser is now in ArrayOpen
        if (c == '[' && newState == JsonParserState.ArrayOpen)
        {
            PushArray();
            return;
        }

        // Key string started: '"' in ObjectOpen/ObjectNextKey → parser moves to InString
        if (c == '"' && newState == JsonParserState.InString && parser.IsKeyString)
        {
            StartKeyString();
            return;
        }

        // Value string started: '"' when not key → parser moves to InString
        if (c == '"' && newState == JsonParserState.InString && !parser.IsKeyString && !_inKeyString)
        {
            StartValueString();
            return;
        }

        // Character inside key string
        if (_inKeyString && newState == JsonParserState.InString)
        {
            AppendKeyChar(c);
            return;
        }

        // Character inside enum/const value string
        if (_inEnumString && newState == JsonParserState.InString)
        {
            AdvanceEnumTrie(c);
            return;
        }

        // Key string closed: parser transitioned out of InString to ObjectColon
        if (_inKeyString && newState == JsonParserState.ObjectColon)
        {
            FinishKeyString();
            return;
        }

        // Value string closed (non-key)
        if (_inEnumString && newState != JsonParserState.InString &&
            newState != JsonParserState.InStringEscape && newState != JsonParserState.InStringUnicode)
        {
            _inEnumString = false;
            // Value complete at current level — handled by PopIfValueComplete below
        }

        // Object closed: '}' → depth decreased
        if (c == '}' && parser.Depth < _stackDepth)
        {
            PopObject();
            return;
        }

        // Array closed: ']' → depth decreased
        if (c == ']' && parser.Depth < _stackDepth)
        {
            PopArray();
            return;
        }

        // Array comma: ',' in array context → next item
        if (c == ',' && newState == JsonParserState.ArrayNextValue)
        {
            AdvanceArrayItem();
            return;
        }

        // Value complete in container context — restore _currentNodeIndex to parent container.
        // After FinishKeyString sets _currentNodeIndex to a property's value node, we need to
        // restore it to the parent object node when the value finishes (string closes, number
        // terminates + comma, literal completes, etc.). Container close (PopObject/PopArray)
        // and array comma (AdvanceArrayItem) have their own return paths above.
        if (_stackDepth > 0 &&
            newState is JsonParserState.ObjectCommaOrClose
                     or JsonParserState.ObjectNextKey
                     or JsonParserState.ArrayCommaOrClose)
        {
            _currentNodeIndex = _nodeStack[_stackDepth - 1];
        }
    }

    /// <summary>
    /// Returns a composite state key for mask caching incorporating schema position.
    /// Uses a struct key to avoid hash collisions (all fields compared exactly).
    /// </summary>
    public readonly SchemaStateKey GetSchemaStateKey(in JsonCharParser parser)
    {
        int parserKey = parser.GetEffectiveStateKey();
        ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
        int triePos = _inKeyString ? _trieNodeIndex : (_inEnumString ? _enumTrieNodeIndex : 0);

        return new SchemaStateKey(parserKey, _currentNodeIndex, emitted, triePos);
    }

    /// <summary>Resets to initial state.</summary>
    public void Reset()
    {
        _currentNodeIndex = 0;
        _stackDepth = 0;
        _keyLength = 0;
        _trieNodeIndex = 0;
        _enumTrieNodeIndex = 0;
        _inKeyString = false;
        _inEnumString = false;
    }

    // ── anyOf forking support (Task 4) ──────────────────────────────

    /// <summary>
    /// Reports whether the value this branch is about to generate is governed by an
    /// <c>anyOf</c> node, yielding the alternative node indices. Used by
    /// <see cref="SchemaTracker"/> to expand a single branch into one parallel branch
    /// per alternative just before the value's opening structural character is applied.
    /// </summary>
    /// <param name="alternatives">The alternative node indices when the result is <c>true</c>.</param>
    /// <returns><c>true</c> when the current value node is an <c>anyOf</c> union.</returns>
    /// <remarks>
    /// The current value node is only an <c>anyOf</c> node at a value-start position
    /// (root value, or a property/array-item value that has just been resolved). While
    /// inside a key or enum/const string, no fork is possible, so those are excluded.
    /// </remarks>
    public readonly bool TryGetAnyOfAlternatives(out ReadOnlySpan<int> alternatives)
    {
        if (!_inKeyString && !_inEnumString)
        {
            int[]? alts = GetNode(_currentNodeIndex).AnyOfNodeIndices;
            if (alts != null)
            {
                alternatives = alts;
                return true;
            }
        }

        alternatives = default;
        return false;
    }

    /// <summary>
    /// Returns a value copy of this branch re-seeded at <paramref name="nodeIndex"/> as its
    /// current value node, with key/enum string cursors reset. The object/array frame is
    /// established normally when the value's opening <c>{</c>/<c>[</c> is consumed, so the
    /// existing node stack and emitted-property bitmasks are preserved.
    /// </summary>
    /// <param name="nodeIndex">The alternative node index to seed the copy with.</param>
    public readonly BranchState WithCurrentNode(int nodeIndex)
    {
        var copy = this;
        copy._currentNodeIndex = nodeIndex;
        copy._inKeyString = false;
        copy._inEnumString = false;
        copy._keyLength = 0;
        copy._trieNodeIndex = 0;
        copy._enumTrieNodeIndex = 0;
        return copy;
    }

    // ── Value start type restriction ────────────────────────────────

    private readonly bool IsValueStartCharAllowed(char c, int nodeIndex)
    {
        if (IsWhitespace(c))
            return true;

        ref readonly var node = ref GetNode(nodeIndex);
        var types = node.AllowedTypes;

        // If anyOf, merge types from all alternatives. This is intentionally an
        // overapproximation: nested branch constraints would require parallel
        // tracker states after the first disambiguating character.
        if (node.AnyOfNodeIndices != null)
        {
            types = JsonSchemaType.None;
            foreach (int altIdx in node.AnyOfNodeIndices)
                types |= GetNode(altIdx).AllowedTypes;
        }

        return c switch
        {
            '{' => types.HasFlag(JsonSchemaType.Object),
            '[' => types.HasFlag(JsonSchemaType.Array),
            '"' => types.HasFlag(JsonSchemaType.String),
            '-' => types.HasFlag(JsonSchemaType.Number) || types.HasFlag(JsonSchemaType.Integer),
            >= '0' and <= '9' => types.HasFlag(JsonSchemaType.Number) || types.HasFlag(JsonSchemaType.Integer),
            't' or 'f' => types.HasFlag(JsonSchemaType.Boolean),
            'n' => types.HasFlag(JsonSchemaType.Null),
            _ => false,
        };
    }

    // ── Object state restrictions ───────────────────────────────────

    private readonly bool IsObjectOpenCharAllowed(char c)
    {
        if (IsWhitespace(c))
            return true;

        ref readonly var node = ref GetNode(_currentNodeIndex);

        if (c == '}')
        {
            // Can close empty object only if no required properties
            return node.RequiredBitmask == 0;
        }

        if (c == '"')
        {
            // Start key — if additional properties allowed, any key is fine.
            if (!node.AdditionalPropertiesForbidden)
                return true;
            // additionalProperties:false — only allow if trie root can reach an un-emitted property.
            if (node.PropertyTrieIndex < 0)
                return false;
            ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
            var trie = _schema.PropertyTries[node.PropertyTrieIndex];
            return (trie.ReachableTerminalBits(0) & ~emitted) != 0;
        }

        return false;
    }

    private readonly bool IsObjectNextKeyCharAllowed(char c)
    {
        if (IsWhitespace(c))
            return true;

        if (c == '"')
        {
            ref readonly var node = ref GetNode(_currentNodeIndex);
            // If additional properties allowed, any key is fine.
            if (!node.AdditionalPropertiesForbidden)
                return true;
            // additionalProperties:false — only allow if trie root can reach an un-emitted property.
            if (node.PropertyTrieIndex < 0)
                return false;
            ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
            var trie = _schema.PropertyTries[node.PropertyTrieIndex];
            return (trie.ReachableTerminalBits(0) & ~emitted) != 0;
        }

        return false;
    }

    private readonly bool IsObjectCommaOrCloseCharAllowed(char c)
    {
        if (IsWhitespace(c))
            return true;

        if (c == '}')
        {
            // Can close only if all required properties emitted
            ref readonly var node = ref GetNode(_currentNodeIndex);
            ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
            return (node.RequiredBitmask & ~emitted) == 0;
        }

        if (c == ',')
        {
            // Can continue if there are more properties possible
            ref readonly var node = ref GetNode(_currentNodeIndex);
            if (node.AdditionalPropertiesForbidden && node.PropertyNames != null)
            {
                ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
                ulong allProps = node.PropertyNames.Length < 64
                    ? (1UL << node.PropertyNames.Length) - 1
                    : ~0UL;
                return (allProps & ~emitted) != 0;
            }
            return true;
        }

        return false;
    }

    // ── Array state restrictions ────────────────────────────────────

    private readonly bool IsArrayOpenCharAllowed(char c)
    {
        if (IsWhitespace(c))
            return true;

        if (c == ']')
            return true; // empty array always allowed (no minItems in MVP)

        // First item — check items schema type
        int itemsNode = GetArrayItemNodeIndex();
        if (itemsNode >= 0)
            return IsValueStartCharAllowed(c, itemsNode);

        return true; // unconstrained items
    }

    // ── String content restrictions ─────────────────────────────────

    private readonly bool IsInStringCharAllowed(char c, in JsonCharParser parser)
    {
        // Key string: restrict to trie. Note: '\' (escape) is NOT unconditionally
        // allowed — it must be a valid trie edge like any other char. Property names
        // contain no backslashes, so allowing '\' here let a weak model escape-flood
        // ('"a\"\"\"...') and break out of the constraint (BitNet tool-call derail).
        if (_inKeyString)
        {
            if (c == '"')
            {
                // Closing quote — property name must be complete (terminal)
                return _schema.PropertyTries.Length > 0 && IsTrieTerminal();
            }
            return IsTrieCharValid(c);
        }

        // Enum/const value string: restrict to enum trie (same escape-flood reasoning).
        if (_inEnumString)
        {
            if (c == '"')
            {
                // Closing quote — value must be complete
                return IsEnumTrieTerminal();
            }
            return IsEnumTrieCharValid(c);
        }

        // Unconstrained string — parser handles
        return true;
    }

    // ── Number restrictions (integer type) ──────────────────────────

    private readonly bool IsNumberContinuationAllowed(char c)
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        var types = node.AllowedTypes;

        // If only Integer (no Number), reject fractional/exponent parts
        if (types.HasFlag(JsonSchemaType.Integer) && !types.HasFlag(JsonSchemaType.Number))
        {
            if (c is '.' or 'e' or 'E')
                return false;
        }

        return true;
    }

    // ── Stack operations ────────────────────────────────────────────

    private void PushObject()
    {
        if (_stackDepth >= MaxDepth) return;
        _nodeStack[_stackDepth] = _currentNodeIndex;
        _emittedProps[_stackDepth] = 0;
        _stackDepth++;
    }

    private void PushArray()
    {
        if (_stackDepth >= MaxDepth) return;
        _nodeStack[_stackDepth] = _currentNodeIndex;
        _arrayIndices[_stackDepth] = 0;
        _stackDepth++;

        // Set current node to items schema for first element
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.ItemsNodeIndex >= 0)
            _currentNodeIndex = node.ItemsNodeIndex;
    }

    private void PopObject()
    {
        if (_stackDepth <= 0) return;
        _stackDepth--;
        // Restore parent's current node (will be set by parent context)
        if (_stackDepth > 0)
        {
            _currentNodeIndex = _nodeStack[_stackDepth - 1];
        }
        else
        {
            _currentNodeIndex = 0; // back to root
        }
    }

    private void PopArray()
    {
        if (_stackDepth <= 0) return;
        _stackDepth--;
        if (_stackDepth > 0)
        {
            _currentNodeIndex = _nodeStack[_stackDepth - 1];
        }
        else
        {
            _currentNodeIndex = 0;
        }
    }

    private void AdvanceArrayItem()
    {
        if (_stackDepth > 0)
        {
            _arrayIndices[_stackDepth - 1]++;
            // Reset current node to items schema for next element
            int parentNode = _nodeStack[_stackDepth - 1];
            ref readonly var node = ref GetNode(parentNode);
            if (node.ItemsNodeIndex >= 0)
                _currentNodeIndex = node.ItemsNodeIndex;
        }
    }

    // ── Key string tracking ─────────────────────────────────────────

    private void StartKeyString()
    {
        _inKeyString = true;
        _keyLength = 0;
        _trieNodeIndex = 0; // root of property name trie

        // Resolve the property trie for the current object
        // _currentNodeIndex should be the object node (top of stack)
    }

    private void AppendKeyChar(char c)
    {
        if (_keyLength < MaxKeyLength)
            _keyBuffer[_keyLength++] = c;

        // Advance trie
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.PropertyTrieIndex >= 0)
        {
            var trie = _schema.PropertyTries[node.PropertyTrieIndex];
            if (trie.TryGetChild(_trieNodeIndex, c, out int child))
                _trieNodeIndex = child;
        }
    }

    private void FinishKeyString()
    {
        _inKeyString = false;

        // Build key name from buffer (inline to avoid ref-escape issues with InlineArray)
        string keyName = new(((ReadOnlySpan<char>)_keyBuffer)[.._keyLength]);

        // Look up property in schema and set current node to the property's value schema
        ref readonly var objectNode = ref GetNode(_currentNodeIndex);
        if (objectNode.Properties != null && objectNode.Properties.TryGetValue(keyName, out int valueNodeIndex))
        {
            // Mark property as emitted
            if (objectNode.PropertyNames != null)
            {
                int bitPos = Array.IndexOf(objectNode.PropertyNames, keyName);
                if (bitPos >= 0 && bitPos < 64 && _stackDepth > 0)
                    _emittedProps[_stackDepth - 1] |= 1UL << bitPos;
            }

            _currentNodeIndex = valueNodeIndex;
        }
        // If property not in schema and additionalProperties allowed, keep unconstrained
    }

    // ── Value string (enum/const) tracking ──────────────────────────

    // Non-string enum/const values (e.g. {"const":1}, {"enum":[true,false]}) are
    // type-constrained but not character-sequence constrained. Literal/number
    // matching belongs in a separate tracker mechanism from string-prefix matching.
    private void StartValueString()
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.EnumTrieIndex >= 0)
        {
            _inEnumString = true;
            _enumTrieNodeIndex = 0; // root of enum trie
        }
    }

    private void AdvanceEnumTrie(char c)
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.EnumTrieIndex >= 0)
        {
            var trie = _schema.PropertyTries[node.EnumTrieIndex];
            if (trie.TryGetChild(_enumTrieNodeIndex, c, out int child))
                _enumTrieNodeIndex = child;
        }
    }

    // ── Trie helpers ────────────────────────────────────────────────

    private readonly bool IsTrieCharValid(char c)
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.PropertyTrieIndex < 0) return true; // no trie = unconstrained

        var trie = _schema.PropertyTries[node.PropertyTrieIndex];

        if (node.AdditionalPropertiesForbidden)
        {
            // Char must exist in trie AND the resulting child must reach at least one un-emitted property.
            if (!trie.TryGetChild(_trieNodeIndex, c, out int child))
                return false;
            ulong emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
            return (trie.ReachableTerminalBits(child) & ~emitted) != 0;
        }

        // If additional properties allowed, any char is valid even if not in trie
        return trie.TryGetChild(_trieNodeIndex, c, out _) || !node.AdditionalPropertiesForbidden;
    }

    private readonly bool IsTrieTerminal()
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.PropertyTrieIndex < 0) return true;

        var trie = _schema.PropertyTries[node.PropertyTrieIndex];

        // Terminal in trie = complete property name
        if (!trie.IsTerminal(_trieNodeIndex))
        {
            // Not a complete property name in the trie
            // If additional properties are allowed, it could still be valid
            return !node.AdditionalPropertiesForbidden;
        }

        // Check it's not already emitted (only if additionalProperties is forbidden,
        // otherwise duplicates are technically valid JSON though unusual)
        if (node.AdditionalPropertiesForbidden)
        {
            string? name = trie.GetCompleteName(_trieNodeIndex);
            if (name != null && node.PropertyNames != null)
            {
                int bitPos = Array.IndexOf(node.PropertyNames, name);
                if (bitPos >= 0 && bitPos < 64 && _stackDepth > 0)
                {
                    ulong emitted = _emittedProps[_stackDepth - 1];
                    if ((emitted & (1UL << bitPos)) != 0)
                        return false; // already emitted
                }
            }
        }

        return true;
    }

    private readonly bool IsEnumTrieCharValid(char c)
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.EnumTrieIndex < 0) return true;

        var trie = _schema.PropertyTries[node.EnumTrieIndex];
        return trie.TryGetChild(_enumTrieNodeIndex, c, out _);
    }

    private readonly bool IsEnumTrieTerminal()
    {
        ref readonly var node = ref GetNode(_currentNodeIndex);
        if (node.EnumTrieIndex < 0) return true;

        var trie = _schema.PropertyTries[node.EnumTrieIndex];
        return trie.IsTerminal(_enumTrieNodeIndex);
    }

    private readonly int GetArrayItemNodeIndex()
    {
        if (_stackDepth <= 0) return -1;
        int parentNode = _nodeStack[_stackDepth - 1];
        ref readonly var node = ref GetNode(parentNode);
        return node.ItemsNodeIndex;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private readonly ref readonly SchemaNode GetNode(int index) => ref _schema.Nodes[index];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsWhitespace(char c) => c is ' ' or '\t' or '\n' or '\r';
}

/// <summary>
/// Tracks schema position during JSON generation. Advances in lockstep with
/// <see cref="JsonCharParser"/>, observing structural events to enforce schema constraints.
/// </summary>
/// <remarks>
/// <para>
/// Value type — copies by value for zero-alloc cloning. Holds a bounded set of up to
/// <see cref="MaxParallelBranches"/> parallel <see cref="BranchState"/> values in an <c>InlineArray</c>, so a
/// whole tracker still copies by value with no heap allocation.
/// </para>
/// <para>
/// A single branch is the common case (any schema without an active <c>anyOf</c> at the
/// current value position). When the value about to be generated is governed by an
/// <c>anyOf</c> node with at most <see cref="MaxParallelBranches"/> alternatives, the single branch forks
/// into one branch per alternative. Thereafter <see cref="IsCharAllowedBySchema"/> is the
/// OR over live branches and <see cref="OnCharAdvanced"/> prunes branches that reject an
/// accepted string character, narrowing the set until it collapses back to one. An
/// <c>anyOf</c> with more than <see cref="MaxParallelBranches"/> alternatives is not forked; the tracker
/// keeps the historical single-branch union overapproximation.
/// </para>
/// </remarks>
internal struct SchemaTracker
{
    /// <summary>
    /// Maximum number of parallel <c>anyOf</c> branches the tracker forks into. Matches
    /// <see cref="BranchStateArray"/> length. Shared with <see cref="ToolCallSchemaBuilder"/>
    /// so the builder's <c>&gt; K</c> degradation threshold and this fork cap can never drift.
    /// </summary>
    internal const int MaxParallelBranches = 8;

    private BranchStateArray _branches;
    private int _liveCount;
    private readonly CompiledSchema _schema;

    // Logged at most once per process when an anyOf exceeds the branch cap.
    private static int _degradationLogged;

    /// <summary>
    /// Creates a new schema tracker for the given compiled schema.
    /// </summary>
    /// <param name="schema">The compiled schema (immutable, shared).</param>
    public SchemaTracker(CompiledSchema schema)
    {
        _schema = schema;
        _branches[0] = new BranchState(schema);
        _liveCount = 1;
    }

    /// <summary>
    /// Whether this tracker can ever fork into multiple branches. A schema with no <c>anyOf</c>
    /// node anywhere is always exactly one live branch, so callers may simulate per-token on a
    /// single <see cref="BranchState"/> copy instead of cloning the whole (wide) tracker.
    /// </summary>
    public readonly bool CanFork => _schema.HasAnyOf;

    /// <summary>
    /// Returns a value copy of the single live branch. Only meaningful when the tracker cannot
    /// fork (<see cref="CanFork"/> is <c>false</c>), in which case branch 0 is the sole branch.
    /// </summary>
    public readonly BranchState GetSingleBranch()
    {
        Debug.Assert(!CanFork, "GetSingleBranch must only be called on a non-forking tracker");
        return _branches[0];
    }

    /// <summary>
    /// Whether the schema is fully satisfied (any live branch is complete).
    /// </summary>
    public readonly bool IsComplete(in JsonCharParser p)
    {
        for (int i = 0; i < _liveCount; i++)
            if (_branches[i].IsComplete(in p))
                return true;
        return false;
    }

    /// <summary>
    /// Checks if a character is allowed by the schema at the current position — the union
    /// (OR) over all live branches. Called BEFORE <see cref="JsonCharParser.TryAdvance"/>.
    /// </summary>
    public readonly bool IsCharAllowedBySchema(char c, in JsonCharParser p)
    {
        for (int i = 0; i < _liveCount; i++)
            if (_branches[i].IsCharAllowed(c, in p))
                return true;
        return false;
    }

    /// <summary>
    /// Called AFTER a character has been successfully accepted by the JSON parser.
    /// Forks the single branch when a governing <c>anyOf</c> value is about to start,
    /// prunes branches that did not allow the accepted character, and advances the
    /// survivors (collapsing to a single branch when the set narrows to one).
    /// </summary>
    /// <param name="c">The character that was just accepted.</param>
    /// <param name="p">The parser state AFTER advancing past <paramref name="c"/>.</param>
    /// <param name="pre">The parser state BEFORE advancing past <paramref name="c"/>.</param>
    /// <remarks>
    /// Pruning uses the PRE-advance allow decision (<c>branch.IsCharAllowed(c, in pre)</c>),
    /// which is exactly the predicate the mask build OR-ed across branches to permit
    /// <paramref name="c"/>. It is correct for ALL characters — string content, a value/key
    /// string's CLOSING quote, and structural characters alike — so a branch whose const/enum
    /// trie is non-terminal at a closing quote (e.g. a tool name that is a strict prefix of
    /// another) is pruned. Because the mask's OR guarantees at least one branch allowed
    /// <paramref name="c"/> pre-advance, the live set never empties on a syntactically valid
    /// character.
    /// </remarks>
    public void OnCharAdvanced(char c, in JsonCharParser p, in JsonCharParser pre)
    {
        // Fork BEFORE applying the character, so each new branch processes the value's
        // opening structural character against its own alternative node (the object/array
        // frame is then pushed normally per branch).
        MaybeFork();

        if (_liveCount <= 1)
        {
            // Single branch: it allowed the char (it is the sole term of the mask's OR), so it
            // never prunes. Behaviour- and cost-identical to the historical single-branch path.
            _branches[0].OnCharAdvanced(c, in p);
            return;
        }

        int w = 0;
        for (int i = 0; i < _liveCount; i++)
        {
            // Prune by the PRE-advance decision — correct for content, closing quotes and
            // structural chars alike.
            if (!_branches[i].IsCharAllowed(c, in pre))
                continue;

            _branches[i].OnCharAdvanced(c, in p);
            if (w != i)
                _branches[w] = _branches[i];
            w++;
        }

        if (w > 0)
        {
            _liveCount = w;
        }
        else
        {
            // Defensive: every branch rejected a character the parser accepted. The mask build
            // never emits such a token (the OR guarantees ≥1 allowing branch), but stay in
            // lockstep with the parser by advancing all branches without pruning.
            for (int i = 0; i < _liveCount; i++)
                _branches[i].OnCharAdvanced(c, in p);
        }
    }

    /// <summary>
    /// Returns a composite state key for mask caching incorporating schema position.
    /// The single-branch case returns the byte-identical key of the underlying branch so
    /// existing cache behaviour is unchanged. The multi-branch case folds all live branches
    /// into a sentinel composite (<c>NodeIdx = -1</c>) that never collides with a single
    /// branch (whose node index is always non-negative).
    /// </summary>
    public readonly SchemaStateKey GetSchemaStateKey(in JsonCharParser p)
    {
        if (_liveCount == 1)
            return _branches[0].GetSchemaStateKey(in p);

        var hash = new HashCode();
        hash.Add(_liveCount);
        for (int i = 0; i < _liveCount; i++)
        {
            var k = _branches[i].GetSchemaStateKey(in p);
            hash.Add(k.ParserKey);
            hash.Add(k.NodeIdx);
            hash.Add(k.EmittedProps);
            hash.Add(k.TriePos);
        }

        return new SchemaStateKey(hash.ToHashCode(), -1, 0, _liveCount);
    }

    /// <summary>Resets to a single fresh branch at the schema root.</summary>
    public void Reset()
    {
        _branches[0].Reset();
        _liveCount = 1;
    }

    /// <summary>
    /// Expands the single live branch into one branch per <c>anyOf</c> alternative when the
    /// value about to be generated is a union of at most <see cref="MaxParallelBranches"/> alternatives.
    /// </summary>
    private void MaybeFork()
    {
        if (_liveCount != 1)
            return;

        if (!_branches[0].TryGetAnyOfAlternatives(out var alts) || alts.Length < 2)
            return;

        if (alts.Length > MaxParallelBranches)
        {
            // Degradation: keep the single-branch union overapproximation (no narrowing).
            LogDegradationOnce(alts.Length);
            return;
        }

        var src = _branches[0];
        for (int i = 0; i < alts.Length; i++)
            _branches[i] = src.WithCurrentNode(alts[i]);
        _liveCount = alts.Length;
    }

    private static void LogDegradationOnce(int altCount)
    {
        if (Interlocked.Exchange(ref _degradationLogged, 1) == 0)
            Debug.WriteLine(
                $"[SchemaTracker] anyOf with {altCount} alternatives exceeds branch cap {MaxParallelBranches}; " +
                "falling back to single-branch union overapproximation (no per-branch narrowing).");
    }
}

/// <summary>InlineArray of parallel branch states (anyOf narrowing capacity).</summary>
[InlineArray(8)]
internal struct BranchStateArray
{
    private BranchState _element;
}

/// <summary>InlineArray for schema node index stack.</summary>
[InlineArray(64)]
internal struct SchemaNodeIdxStack
{
    private int _element;
}

/// <summary>InlineArray for emitted property bitmask stack.</summary>
[InlineArray(64)]
internal struct PropertyBitStack
{
    private ulong _element;
}

/// <summary>InlineArray for key character buffer.</summary>
[InlineArray(128)]
internal struct KeyCharBuffer
{
    private char _element;
}

/// <summary>InlineArray for array item index stack.</summary>
[InlineArray(64)]
internal struct ArrayIndexStack
{
    private int _element;
}

/// <summary>
/// Collision-free cache key for schema constraint mask lookup.
/// All fields are compared exactly — no hash compression.
/// </summary>
internal readonly record struct SchemaStateKey(
    int ParserKey,
    int NodeIdx,
    ulong EmittedProps,
    int TriePos);
