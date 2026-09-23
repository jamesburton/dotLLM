namespace DotLLM.Engine.Evaluation;

/// <summary>
/// A <see cref="TextReader"/> decorator that collapses <c>\r\n</c> to <c>\n</c>, reproducing what
/// the MSVC C runtime does to a file opened in text mode.
/// </summary>
/// <remarks>
/// <para>
/// Opt-in only (<c>dotllm perplexity --normalize-line-endings</c>). It exists so a CRLF corpus can
/// be scored against a Windows <c>llama-perplexity</c> run without first rewriting the file; it is
/// off by default because llama.cpp on Linux does <b>not</b> strip the CRs, so normalizing
/// unconditionally would simply move the mismatch (issue #506).
/// </para>
/// <para>
/// Only <c>\r</c> immediately followed by <c>\n</c> is removed — a lone <c>\r</c> is passed through,
/// exactly as text mode does. A CRLF straddling two reads is handled by carrying the pending
/// <c>\r</c> across the boundary.
/// </para>
/// </remarks>
public sealed class CrlfNormalizingTextReader : TextReader
{
    private readonly TextReader _inner;
    private readonly bool _ownsInner;
    private readonly char[] _raw = new char[4096];
    private int _rawPos;
    private int _rawLen;
    private bool _pendingCr;
    private bool _innerExhausted;

    /// <summary>Wraps <paramref name="inner"/>.</summary>
    /// <param name="inner">Underlying reader supplying raw corpus characters.</param>
    /// <param name="ownsInner">When <see langword="true"/>, disposing this also disposes <paramref name="inner"/>.</param>
    public CrlfNormalizingTextReader(TextReader inner, bool ownsInner = false)
    {
        ArgumentNullException.ThrowIfNull(inner);
        _inner = inner;
        _ownsInner = ownsInner;
    }

    /// <inheritdoc />
    public override int Read()
    {
        Span<char> one = stackalloc char[1];
        return Read(one) == 0 ? -1 : one[0];
    }

    /// <inheritdoc />
    public override int Read(char[] buffer, int index, int count)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        return Read(buffer.AsSpan(index, count));
    }

    /// <inheritdoc />
    public override int Read(Span<char> buffer)
    {
        if (buffer.Length == 0) return 0;

        int produced = 0;
        // Loop because a refill can consist entirely of removed CRs; returning 0 then would be
        // indistinguishable from end-of-input and would silently truncate the corpus.
        while (produced == 0)
        {
            if (_rawPos >= _rawLen)
            {
                if (_innerExhausted)
                {
                    // A CR that turned out to be the last character is real input, not a CRLF.
                    if (_pendingCr)
                    {
                        _pendingCr = false;
                        buffer[produced++] = '\r';
                    }

                    return produced;
                }

                _rawLen = _inner.Read(_raw, 0, _raw.Length);
                _rawPos = 0;
                if (_rawLen == 0)
                {
                    _innerExhausted = true;
                    continue;
                }
            }

            while (_rawPos < _rawLen && produced < buffer.Length)
            {
                char c = _raw[_rawPos++];

                // A CR carried over from the previous character (possibly from the previous read
                // call) is resolved here: dropped when this is the LF of a CRLF, emitted otherwise.
                if (_pendingCr)
                {
                    _pendingCr = false;
                    if (c == '\n')
                    {
                        buffer[produced++] = '\n';
                        continue;
                    }

                    buffer[produced++] = '\r';
                    if (produced == buffer.Length)
                    {
                        _rawPos--;   // no room for c; re-examine it on the next call
                        break;
                    }
                }

                if (c == '\r')
                {
                    _pendingCr = true;
                    continue;
                }

                buffer[produced++] = c;
            }
        }

        return produced;
    }

    /// <inheritdoc />
    public override int Peek() => throw new NotSupportedException(
        "CrlfNormalizingTextReader is a forward-only streaming decorator.");

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing && _ownsInner) _inner.Dispose();
        base.Dispose(disposing);
    }
}
