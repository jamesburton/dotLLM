using DotLLM.Tokenizers;

namespace ChunkKvLab;

/// <summary>A fixed-length token chunk of one corpus document.</summary>
/// <param name="Index">Position in <see cref="ContextBuilder.Chunks"/>.</param>
/// <param name="Document">Document index.</param>
/// <param name="Start">Token offset of the chunk inside its document.</param>
/// <param name="TokenIds">The chunk's token ids, fixed at corpus load (never re-tokenised).</param>
public sealed record CorpusChunk(int Index, int Document, int Start, int[] TokenIds);

/// <summary>One E1 request.</summary>
/// <param name="Seed">Seed the plan was built from.</param>
/// <param name="BaseOrder">The earlier ("cached") request's chunk order: the first k chunks of the seed's permutation.</param>
/// <param name="RequestOrder">This request's chunk order: the base's first half kept, the rest re-shuffled.</param>
/// <param name="ContextIds">Concatenation of the request chunks' token ids, in request order.</param>
/// <param name="Continuation">Held-out tokens after the context.</param>
/// <param name="MatchedChunks">Longest identical leading chunk run of <see cref="BaseOrder"/> and <see cref="RequestOrder"/>.</param>
/// <param name="ContinuationIsSuccessor">True when <see cref="Continuation"/> is the document text right after the last request chunk.</param>
public sealed record ContextPlan(
    int Seed, int[] BaseOrder, int[] RequestOrder, int[] ContextIds, int[] Continuation, int MatchedChunks, bool ContinuationIsSuccessor);

/// <summary>
/// Splits a corpus into fixed-size token chunks (each document tokenised exactly once) and builds E1 contexts as
/// concatenations of chunk token ids. Nothing is re-tokenised after construction, so the reference prefill and the chunk
/// methods see the same ids.
/// </summary>
public sealed class ContextBuilder
{
    private readonly List<int[]> _documents = [];
    private readonly List<CorpusChunk> _chunks = [];

    /// <summary>Tokenises every document once (no BOS) and cuts it into <paramref name="chunkTokens"/>-token chunks; tails are dropped.</summary>
    public ContextBuilder(IEnumerable<(string Name, string Text)> documents, ITokenizer tokenizer, int chunkTokens = 512)
    {
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkTokens);
        ChunkTokens = chunkTokens;
        foreach ((string _, string text) in documents)
        {
            int[] ids = tokenizer.EncodeRaw(text);
            int doc = _documents.Count;
            _documents.Add(ids);
            for (int start = 0; start + chunkTokens <= ids.Length; start += chunkTokens)
                _chunks.Add(new CorpusChunk(_chunks.Count, doc, start, ids.AsSpan(start, chunkTokens).ToArray()));
        }
    }

    /// <summary>Reads every file matching <paramref name="pattern"/> under <paramref name="directory"/> (ordinal path order, CRLF → LF).</summary>
    public static ContextBuilder FromDirectory(string directory, string pattern, ITokenizer tokenizer, int chunkTokens = 512)
    {
        var files = Directory.EnumerateFiles(directory, pattern, SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                     && !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .Select(f => Path.GetRelativePath(directory, f))
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToList();
        return new ContextBuilder(
            files.Select(f => (f, File.ReadAllText(Path.Combine(directory, f)).Replace("\r\n", "\n", StringComparison.Ordinal))),
            tokenizer, chunkTokens);
    }

    /// <summary>Tokens per chunk.</summary>
    public int ChunkTokens { get; }

    /// <summary>All chunks in corpus order.</summary>
    public IReadOnlyList<CorpusChunk> Chunks => _chunks;

    /// <summary>Number of documents read.</summary>
    public int DocumentCount => _documents.Count;

    /// <summary>
    /// Builds the request for <paramref name="seed"/> with <paramref name="contextTokens"/> / <see cref="ChunkTokens"/> chunks.
    /// The seed fixes one permutation of all chunks, so contexts of different sizes for a seed are nested (their chunks can be
    /// encoded once). Base order = the permutation's first k chunks; request order = the first k/2 kept, the rest shuffled.
    /// Continuation = the document tokens following the last request chunk when they exist and do not belong to a context
    /// chunk; otherwise the start of the next unused chunk of the permutation.
    /// </summary>
    public ContextPlan Build(int seed, int contextTokens, int continuationTokens)
    {
        if (contextTokens % ChunkTokens != 0 || contextTokens <= 0)
            throw new ArgumentException($"Context {contextTokens} is not a positive multiple of the chunk size {ChunkTokens}.");
        int k = contextTokens / ChunkTokens;
        if (k + 1 > _chunks.Count)
            throw new ArgumentException($"Context needs {k} chunks plus a held-out one; the corpus has {_chunks.Count}.");
        if (continuationTokens <= 0 || continuationTokens > ChunkTokens)
            throw new ArgumentOutOfRangeException(nameof(continuationTokens));

        int[] perm = Enumerable.Range(0, _chunks.Count).ToArray();
        new Random(seed).Shuffle(perm);
        int[] baseOrder = perm[..k];

        int keep = k / 2;
        int[] request = (int[])baseOrder.Clone();
        new Random(unchecked(seed * 7919 + k)).Shuffle(request.AsSpan(keep));
        int matched = 0;
        while (matched < k && request[matched] == baseOrder[matched]) matched++;

        var ids = new int[k * ChunkTokens];
        for (int c = 0; c < k; c++)
            _chunks[request[c]].TokenIds.CopyTo(ids, c * ChunkTokens);

        CorpusChunk last = _chunks[request[^1]];
        int[] doc = _documents[last.Document];
        int after = last.Start + ChunkTokens;
        var inContext = new HashSet<int>(request);
        bool successorUsed = after < doc.Length && _chunks.Any(ch => ch.Document == last.Document && ch.Start == after && inContext.Contains(ch.Index));
        int[] continuation;
        bool isSuccessor;
        if (after + continuationTokens <= doc.Length && !successorUsed)
        {
            continuation = doc.AsSpan(after, continuationTokens).ToArray();
            isSuccessor = true;
        }
        else
        {
            continuation = _chunks[perm[k]].TokenIds.AsSpan(0, continuationTokens).ToArray();
            isSuccessor = false;
        }
        return new ContextPlan(seed, baseOrder, request, ids, continuation, matched, isSuccessor);
    }
}
