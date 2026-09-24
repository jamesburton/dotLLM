namespace DotLLM.Engine.Evaluation;

/// <summary>
/// Detects CR bytes in a perplexity corpus, the defect behind issue #506.
/// </summary>
/// <remarks>
/// <para>
/// A corpus stored with CRLF line endings is read differently by the two engines a quality
/// comparison usually involves: <c>llama-perplexity</c> built with MSVC opens the prompt file in
/// <b>text mode</b>, so the C runtime collapses <c>\r\n</c> to <c>\n</c> before the tokenizer ever
/// sees it, while dotLLM reads the bytes as they are and keeps the <c>\r</c>. The two engines then
/// tokenize different text — measured on <c>wiki.test.raw</c> and Llama-3.2-1B, 501 of the 512
/// tokens in chunk 0 differed — and every aggregate agreement figure produced that way is
/// meaningless.
/// </para>
/// <para>
/// The reader is deliberately <b>not</b> fixed to strip CRs by default: llama.cpp on Linux keeps
/// the <c>\r</c> too, so unconditional stripping would trade a Windows mismatch for a Linux one.
/// The corpus is the thing to fix; this type exists to make the problem loud.
/// </para>
/// </remarks>
public static class CorpusLineEndings
{
    /// <summary>Bytes read per scan iteration.</summary>
    private const int ScanBufferSize = 1 << 16;

    /// <summary>
    /// Counts CR (<c>0x0D</c>) bytes in a file.
    /// </summary>
    /// <remarks>
    /// The scan is byte-wise and therefore exact for UTF-8 input: <c>0x0D</c> is an ASCII byte and
    /// can never occur as a continuation byte of a multi-byte sequence, so no decoding is needed.
    /// </remarks>
    /// <param name="path">Corpus file to scan.</param>
    /// <returns>Number of CR bytes in the file.</returns>
    public static long CountCarriageReturns(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using FileStream stream = File.OpenRead(path);
        return CountCarriageReturns(stream);
    }

    /// <summary>
    /// Counts CR (<c>0x0D</c>) bytes read from <paramref name="stream"/> to its end.
    /// </summary>
    /// <param name="stream">Corpus bytes, read from the current position.</param>
    /// <returns>Number of CR bytes seen.</returns>
    public static long CountCarriageReturns(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        byte[] buffer = new byte[ScanBufferSize];
        long count = 0;
        int read;
        while ((read = stream.Read(buffer, 0, buffer.Length)) > 0)
        {
            var span = new ReadOnlySpan<byte>(buffer, 0, read);
            int index;
            while ((index = span.IndexOf((byte)'\r')) >= 0)
            {
                count++;
                span = span[(index + 1)..];
            }
        }

        return count;
    }

    /// <summary>
    /// Builds the operator-facing warning for a corpus that contains CR bytes.
    /// </summary>
    /// <param name="path">Corpus path, quoted back to the operator.</param>
    /// <param name="carriageReturns">CR byte count from <see cref="CountCarriageReturns(string)"/>.</param>
    /// <returns>
    /// The multi-line warning text, or <see langword="null"/> when the corpus is CR-free and
    /// nothing needs saying.
    /// </returns>
    public static string? DescribeMismatchRisk(string path, long carriageReturns)
    {
        if (carriageReturns <= 0) return null;

        return $"""
            WARNING: corpus '{path}' contains {carriageReturns:N0} CR (0x0D) bytes - most likely CRLF line endings.
            llama.cpp on Windows reads its prompt file in MSVC text mode, which strips the CR of every
            CRLF before tokenizing; dotLLM keeps them. (A lone CR survives text mode, so a corpus whose
            CRs are all bare is safe - scripts/make_lf_corpus.py reports the CRLF count separately.) The two engines therefore score DIFFERENT TEXT and any
            dotLLM-vs-llama.cpp perplexity comparison made from this file is NOT like-for-like.
            (llama.cpp on Linux keeps the CRs, which is why dotLLM does not strip them by default.)
            Fix the corpus, not the reader: use an LF copy (scripts/make_lf_corpus.py writes one into
            ~/.dotllm/test-cache/corpora/), or feed both engines the same token ids via
            llama-perplexity --kl-divergence-base plus dotllm perplexity --tokens-file.
            See issue #506 and docs/PERPLEXITY.md.
            """;
    }

    /// <summary>Short label for the results table: how the corpus was encoded and what was done about it.</summary>
    /// <param name="carriageReturns">CR byte count.</param>
    /// <param name="normalized">Whether <c>--normalize-line-endings</c> was in effect.</param>
    /// <returns>A one-line summary suitable for a report row.</returns>
    public static string SummarizeForReport(long carriageReturns, bool normalized)
    {
        if (carriageReturns <= 0) return "LF";
        return normalized
            ? $"CRLF ({carriageReturns:N0} CR) - normalized to LF"
            : $"CRLF ({carriageReturns:N0} CR) - NOT like-for-like vs llama.cpp/Windows (#506)";
    }
}
