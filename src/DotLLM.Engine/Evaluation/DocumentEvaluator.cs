using DotLLM.Core.Evaluation;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.Evaluation;

/// <summary>Per-document hook: bracket the forward passes of one document (routing capture uses it).</summary>
public interface IDocumentObserver
{
    /// <summary>Called before the first forward of document <paramref name="doc"/>; <paramref name="tokens"/> is the sequence fed to the model.</summary>
    void BeginDocument(int doc, ReadOnlySpan<int> tokens);

    /// <summary>Called after the last forward of the document.</summary>
    void EndDocument(int doc);
}

/// <summary>Options for <see cref="DocumentEvaluator"/>.</summary>
/// <param name="BosTokenId">Token prepended to every document, or null for none (the default; ids are scored as supplied).</param>
/// <param name="MaxDocTokens">Truncate each document's sequence (BOS included) to this many tokens; 0 = never truncate.</param>
public readonly record struct DocumentEvalOptions(int? BosTokenId = null, int MaxDocTokens = 0);

/// <summary>
/// Teacher-forced per-document scoring (<c>dotllm perplexity --docs-file</c>, #1706).
/// </summary>
/// <remarks>
/// Each document is an independent sequence from position 0. Token 0 of the sequence (the BOS if one is prepended, else the
/// first document token) has no prediction and is NOT scored; token <c>i</c> (i &gt;= 1) is scored from row <c>i-1</c>, so a
/// document of <c>n</c> tokens scores <c>n - 1</c> targets. The context limit is the model's
/// <see cref="IPerplexityModel.MaxContextLength"/>; a longer sequence is an error unless <see cref="DocumentEvalOptions.MaxDocTokens"/>
/// truncates it. NLL and KL are accumulated in double, sequentially in position order (see <see cref="DocumentScoring"/>).
/// </remarks>
public static class DocumentEvaluator
{
    /// <summary>Scores every document, writing one JSONL result line per document to <paramref name="output"/>.</summary>
    /// <param name="model">An already-constructed model. Not owned.</param>
    /// <param name="documents">Token-id documents.</param>
    /// <param name="options">BOS / truncation options.</param>
    /// <param name="output">Receives one JSON line per document.</param>
    /// <param name="references">Optional reference log-probs, one per document in order; null = NLL only.</param>
    /// <param name="observer">Optional per-document hook.</param>
    /// <returns>The results, in document order.</returns>
    /// <exception cref="ArgumentException">A document is longer than the context limit, has an id outside the vocabulary, or does not match its reference.</exception>
    public static unsafe List<DocResult> Run(
        IPerplexityModel model, IEnumerable<int[]> documents, DocumentEvalOptions options, TextWriter output,
        RefLogProbsReader? references = null, IDocumentObserver? observer = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        int vocab = model.VocabSize;
        int context = model.MaxContextLength;
        var results = new List<DocResult>();
        var positions = new int[context];
        for (int i = 0; i < context; i++) positions[i] = i;

        int doc = -1;
        foreach (int[] raw in documents)
        {
            doc++;
            int[] seq = BuildSequence(raw, options);
            if (seq.Length > context)
                throw new ArgumentException(
                    $"Document {doc} has {seq.Length} tokens (BOS included), over the model's context limit of {context}. Pass --max-doc-tokens to truncate.");
            for (int i = 0; i < seq.Length; i++)
                if ((uint)seq[i] >= (uint)vocab)
                    throw new ArgumentException($"Document {doc}: token id {seq[i]} at position {i} is outside the model vocabulary of {vocab}.");

            int scored = Math.Max(0, seq.Length - 1);
            RefDocument? reference = null;
            if (references is not null)
            {
                reference = references.Next()
                    ?? throw new ArgumentException($"--ref-logprobs has no entry for document {doc}.");
                if (reference.Positions != scored)
                    throw new ArgumentException(
                        $"Document {doc}: reference has {reference.Positions} positions but the document scores {scored} (check BOS / truncation match the reference).");
            }

            double nll = 0, kl = 0;
            observer?.BeginDocument(doc, seq);
            if (seq.Length >= 2)
            {
                if (model.ReturnsAllRows)
                {
                    model.ResetState();
                    using ITensor logits = model.Forward(seq, positions.AsSpan(0, seq.Length));
                    for (int i = 0; i < scored; i++)
                    {
                        var row = new ReadOnlySpan<float>((void*)(logits.DataPointer + (nint)i * vocab * sizeof(float)), vocab);
                        nll -= DocumentScoring.TargetLogProb(row, seq[i + 1]);
                        if (reference is not null) kl += reference.Kl(i, row);
                    }
                }
                else
                {
                    // Last-row-only backend: one prefill per target over the growing prefix (O(n^2) forwards).
                    for (int i = 0; i < scored; i++)
                    {
                        model.ResetState();
                        using ITensor last = model.Forward(seq.AsSpan(0, i + 1), positions.AsSpan(0, i + 1));
                        var row = new ReadOnlySpan<float>((void*)last.DataPointer, vocab);
                        nll -= DocumentScoring.TargetLogProb(row, seq[i + 1]);
                        if (reference is not null) kl += reference.Kl(i, row);
                    }
                }
            }
            observer?.EndDocument(doc);

            var result = new DocResult(doc, seq.Length,
                new DocumentScore(scored, nll, reference is null ? null : kl, reference?.Mode));
            results.Add(result);
            output.WriteLine(DocumentJsonl.FormatResultLine(result));
            output.Flush();
        }

        if (references is not null && references.Next() is not null)
            throw new ArgumentException($"--ref-logprobs has more entries than the {doc + 1} documents scored.");
        return results;
    }

    /// <summary>Applies the BOS prepend and truncation to one document.</summary>
    internal static int[] BuildSequence(int[] doc, DocumentEvalOptions options)
    {
        int bos = options.BosTokenId.HasValue ? 1 : 0;
        int n = doc.Length + bos;
        if (options.MaxDocTokens > 0 && n > options.MaxDocTokens) n = options.MaxDocTokens;
        var seq = new int[n];
        if (bos == 1 && n > 0) seq[0] = options.BosTokenId!.Value;
        for (int i = bos; i < n; i++) seq[i] = doc[i - bos];
        return seq;
    }
}
