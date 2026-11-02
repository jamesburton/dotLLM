// SPDX-License-Identifier: MIT
using System.Diagnostics.CodeAnalysis;

namespace DotLLM.ChunkCache.Client;

/// <summary>
/// Part text to spans, registered by the file-read tool when it produces a result (spec 8.3). Keyed by the part's
/// <b>text</b>, never by a tool-call id: the decorator runs below the invocation loop and does not see which tool
/// produced a result (spec 12.1).
/// </summary>
public interface IChunkSpanRegistry
{
    /// <summary>Registers <paramref name="spans"/> (UTF-16 offsets into <paramref name="partText"/>).</summary>
    void Register(string partText, IReadOnlyList<ChunkSpan> spans);

    /// <summary>Exact lookup by part text.</summary>
    bool TryGet(string partText, out IReadOnlyList<ChunkSpan> spans);

    /// <summary>
    /// Finds the registration for a part whose text is <paramref name="partText"/>, or a registered text plus an appended
    /// suffix (a hook that appends context keeps the spans of the registered prefix valid, spec 11 item 5).
    /// <paramref name="registeredText"/> is the text the spans index into. The default is exact lookup only.
    /// </summary>
    bool TryFind(string partText, [NotNullWhen(true)] out string? registeredText, out IReadOnlyList<ChunkSpan> spans)
    {
        if (TryGet(partText, out spans))
        {
            registeredText = partText;
            return true;
        }

        registeredText = null;
        return false;
    }
}

/// <summary>
/// The default <see cref="IChunkSpanRegistry"/>: a thread-safe least-recently-used map from part text to spans, bounded
/// both in texts (4,096 by default) and in total UTF-16 chars (32 Mi chars, 64 MiB, by default), with prefix lookup for
/// texts that later had a suffix appended.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>Registering the same text again replaces its spans; registering it with no spans removes it.</item>
/// <item>When either bound is exceeded, the least recently used texts are evicted first. A single text larger than the
/// char budget is not kept (its spans are simply not annotated).</item>
/// <item>Span ranges are checked against the text at registration (a programming error throws). Hashes are not: the body
/// handler re-hashes every span against the serialised body, so a wrong hash costs that span, never a wrong annotation.</item>
/// <item>Prefix lookup applies to texts of at least <see cref="PrefixKeyLength"/> UTF-16 chars (chunks are tens of tokens
/// at least) and returns the longest registered prefix.</item>
/// </list>
/// </remarks>
public sealed class ChunkSpanRegistry : IChunkSpanRegistry
{
    /// <summary>The default capacity (plan Task 8).</summary>
    public const int DefaultCapacity = 4096;

    /// <summary>The default budget of registered text, in UTF-16 chars (32 Mi chars = 64 MiB).</summary>
    public const long DefaultMaxTotalChars = 32L * 1024 * 1024;

    /// <summary>Texts at least this long (UTF-16 chars) are also found as a prefix of a longer part text.</summary>
    public const int PrefixKeyLength = 32;

    private readonly int _capacity;
    private readonly long _maxTotalChars;
    private long _totalChars;
    private readonly object _gate = new();
    private readonly Dictionary<string, LinkedListNode<Entry>> _byText = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<Entry>> _byPrefix = new(StringComparer.Ordinal);
    private readonly LinkedList<Entry> _lru = new(); // most recently used first

    /// <summary>Creates a registry holding at most <paramref name="capacity"/> part texts and <paramref name="maxTotalChars"/> chars of text.</summary>
    public ChunkSpanRegistry(int capacity = DefaultCapacity, long maxTotalChars = DefaultMaxTotalChars)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTotalChars, 1L);
        _capacity = capacity;
        _maxTotalChars = maxTotalChars;
    }

    /// <summary>Total UTF-16 chars of the registered texts.</summary>
    public long TotalChars
    {
        get
        {
            lock (_gate) return _totalChars;
        }
    }

    /// <summary>The number of registered part texts.</summary>
    public int Count
    {
        get
        {
            lock (_gate) return _byText.Count;
        }
    }

    /// <inheritdoc />
    public void Register(string partText, IReadOnlyList<ChunkSpan> spans)
    {
        ArgumentNullException.ThrowIfNull(partText);
        ArgumentNullException.ThrowIfNull(spans);
        foreach (ChunkSpan s in spans)
        {
            ArgumentNullException.ThrowIfNull(s, nameof(spans));
            int end = s.End ?? partText.Length;
            if (s.Start < 0 || end > partText.Length || s.Start > end)
                throw new ArgumentOutOfRangeException(nameof(spans), $"span '{s.Id}' [{s.Start}, {end}) is outside the part text (length {partText.Length}).");
        }

        ChunkSpan[] copy = [.. spans];
        lock (_gate)
        {
            if (_byText.TryGetValue(partText, out LinkedListNode<Entry>? existing))
                RemoveLocked(existing);
            if (copy.Length == 0 || partText.Length > _maxTotalChars)
                return;

            var entry = new Entry(partText, copy);
            _byText[partText] = _lru.AddFirst(entry);
            _totalChars += partText.Length;
            if (partText.Length >= PrefixKeyLength)
            {
                string key = partText[..PrefixKeyLength];
                if (!_byPrefix.TryGetValue(key, out List<Entry>? bucket))
                    _byPrefix[key] = bucket = [];
                bucket.Add(entry);
            }

            while (_byText.Count > _capacity || _totalChars > _maxTotalChars)
                RemoveLocked(_lru.Last!);
        }
    }

    /// <inheritdoc />
    public bool TryGet(string partText, out IReadOnlyList<ChunkSpan> spans)
    {
        ArgumentNullException.ThrowIfNull(partText);
        lock (_gate)
        {
            if (_byText.TryGetValue(partText, out LinkedListNode<Entry>? node))
            {
                Touch(node);
                spans = node.Value.Spans;
                return true;
            }
        }

        spans = [];
        return false;
    }

    /// <inheritdoc />
    public bool TryFind(string partText, [NotNullWhen(true)] out string? registeredText, out IReadOnlyList<ChunkSpan> spans)
    {
        ArgumentNullException.ThrowIfNull(partText);
        lock (_gate)
        {
            if (_byText.TryGetValue(partText, out LinkedListNode<Entry>? node))
            {
                Touch(node);
                registeredText = node.Value.Text;
                spans = node.Value.Spans;
                return true;
            }

            if (partText.Length > PrefixKeyLength && _byPrefix.TryGetValue(partText[..PrefixKeyLength], out List<Entry>? bucket))
            {
                Entry? best = null;
                foreach (Entry e in bucket)
                {
                    if (e.Text.Length < partText.Length && (best is null || e.Text.Length > best.Text.Length)
                        && partText.StartsWith(e.Text, StringComparison.Ordinal))
                        best = e;
                }

                if (best is not null)
                {
                    Touch(_byText[best.Text]);
                    registeredText = best.Text;
                    spans = best.Spans;
                    return true;
                }
            }
        }

        registeredText = null;
        spans = [];
        return false;
    }

    private void Touch(LinkedListNode<Entry> node)
    {
        if (node != _lru.First)
        {
            _lru.Remove(node);
            _lru.AddFirst(node);
        }
    }

    private void RemoveLocked(LinkedListNode<Entry> node)
    {
        Entry e = node.Value;
        _lru.Remove(node);
        _byText.Remove(e.Text);
        _totalChars -= e.Text.Length;
        if (e.Text.Length >= PrefixKeyLength)
        {
            string key = e.Text[..PrefixKeyLength];
            List<Entry> bucket = _byPrefix[key];
            bucket.Remove(e);
            if (bucket.Count == 0)
                _byPrefix.Remove(key);
        }
    }

    private sealed record Entry(string Text, ChunkSpan[] Spans);
}
