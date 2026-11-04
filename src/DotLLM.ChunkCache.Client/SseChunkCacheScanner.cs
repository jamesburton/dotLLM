// SPDX-License-Identifier: MIT
using System.Text.Json.Nodes;

namespace DotLLM.ChunkCache.Client;

/// <summary>
/// A pass-through read stream over a server-sent-events body that finds the top-level <c>chunk_cache</c> report as the
/// bytes are read (spec 5.9: the OpenAI final chunk next to <c>usage</c> and <c>timings</c>, or the Anthropic
/// <c>message_delta</c> event), without buffering the stream. The reader above sees exactly the inner stream's bytes.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>Lazy, like CodeSharp's <c>SseTimingsScanner</c>: only the current line is held, and only a <c>data:</c> line
/// whose bytes contain <c>"chunk_cache"</c> is parsed as JSON; every other event costs a byte scan.</item>
/// <item>A line longer than <see cref="MaxLineBytes"/> is skipped, so a hostile stream cannot grow the buffer without
/// bound. Single-line <c>data:</c> fields only (both surfaces emit one JSON object per line).</item>
/// <item><c>\n</c>, <c>\r\n</c> and bare <c>\r</c> line ends (the SSE spec allows all three); a final line without a
/// line end is examined at the real end of stream. A zero-byte read (a caller polling with an empty buffer, as pipeline
/// readers with zero-byte reads do) is passed through and is not end of stream.</item>
/// <item>A report line over <see cref="MaxLineBytes"/> is lost (counted in <see cref="SkippedLines"/>); a summary report is
/// about 1 KB.</item>
/// </list>
/// </remarks>
public sealed class SseChunkCacheScanner : Stream
{
    /// <summary>The longest line examined (1 MiB).</summary>
    public const int MaxLineBytes = 1 << 20;

    private static ReadOnlySpan<byte> Needle => "\"chunk_cache\""u8;

    private static ReadOnlySpan<byte> DataField => "data:"u8;

    private readonly Stream _inner;
    private readonly Action<JsonObject> _onReport;
    private byte[] _line = new byte[256];
    private int _length;
    private bool _overflow;
    private bool _ended;
    private bool _afterCr;
    private int _parsed;
    private int _skipped;

    /// <summary>Wraps <paramref name="inner"/>; <paramref name="onReport"/> receives each report found (normally one).</summary>
    public SseChunkCacheScanner(Stream inner, Action<JsonObject> onReport)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentNullException.ThrowIfNull(onReport);
        _inner = inner;
        _onReport = onReport;
    }

    /// <summary>JSON parses so far: only <c>data:</c> lines that contain <c>"chunk_cache"</c>.</summary>
    public int ParsedEvents => _parsed;

    /// <summary>Lines skipped for exceeding <see cref="MaxLineBytes"/>.</summary>
    public int SkippedLines => _skipped;

    /// <inheritdoc />
    public override bool CanRead => true;

    /// <inheritdoc />
    public override bool CanSeek => false;

    /// <inheritdoc />
    public override bool CanWrite => false;

    /// <inheritdoc />
    public override long Length => throw new NotSupportedException();

    /// <inheritdoc />
    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

    /// <inheritdoc />
    public override void Flush()
    {
    }

    /// <inheritdoc />
    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

    /// <inheritdoc />
    public override int Read(Span<byte> buffer)
    {
        if (buffer.IsEmpty) return _inner.Read(buffer); // a zero-byte read is not end of stream
        int n = _inner.Read(buffer);
        Scan(buffer[..n]);
        return n;
    }

    /// <inheritdoc />
    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

    /// <inheritdoc />
    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
    {
        if (buffer.IsEmpty) return await _inner.ReadAsync(buffer, cancellationToken).ConfigureAwait(false);
        int n = await _inner.ReadAsync(buffer, cancellationToken).ConfigureAwait(false);
        Scan(buffer.Span[..n]);
        return n;
    }

    /// <inheritdoc />
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

    /// <inheritdoc />
    public override void SetLength(long value) => throw new NotSupportedException();

    /// <inheritdoc />
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing) _inner.Dispose();
        base.Dispose(disposing);
    }

    /// <inheritdoc />
    public override async ValueTask DisposeAsync()
    {
        await _inner.DisposeAsync().ConfigureAwait(false);
        await base.DisposeAsync().ConfigureAwait(false);
    }

    private void Scan(ReadOnlySpan<byte> bytes)
    {
        if (bytes.IsEmpty)
        {
            // End of stream: a last line without a newline still counts.
            if (!_ended)
            {
                _ended = true;
                EndLine();
            }

            return;
        }

        while (!bytes.IsEmpty)
        {
            if (_afterCr)
            {
                // The \n of a \r\n pair whose \r ended the previous read: the line has already ended.
                _afterCr = false;
                if (bytes[0] == (byte)'\n')
                {
                    bytes = bytes[1..];
                    continue;
                }
            }

            int end = bytes.IndexOfAny((byte)'\r', (byte)'\n');
            if (end < 0)
            {
                Append(bytes);
                return;
            }

            Append(bytes[..end]);
            EndLine();
            if (bytes[end] == (byte)'\r')
            {
                if (end + 1 < bytes.Length && bytes[end + 1] == (byte)'\n')
                    end++;
                else if (end + 1 == bytes.Length)
                    _afterCr = true;
            }

            bytes = bytes[(end + 1)..];
        }
    }

    private void Append(ReadOnlySpan<byte> piece)
    {
        if (_overflow || piece.IsEmpty) return;
        if (_length + piece.Length > MaxLineBytes)
        {
            _overflow = true;
            return;
        }

        if (_length + piece.Length > _line.Length)
            Array.Resize(ref _line, Math.Min(MaxLineBytes, Math.Max(_line.Length * 2, _length + piece.Length)));
        piece.CopyTo(_line.AsSpan(_length));
        _length += piece.Length;
    }

    private void EndLine()
    {
        ReadOnlySpan<byte> line = _line.AsSpan(0, _length);
        bool skip = _overflow;
        _length = 0;
        _overflow = false;
        if (skip)
        {
            _skipped++;
            return;
        }

        if (!line.StartsWith(DataField)) return;
        line = line[DataField.Length..];
        if (!line.IsEmpty && line[0] == (byte)' ') line = line[1..];
        if (line.IndexOf(Needle) < 0) return;

        _parsed++;
        if (ChunkCacheReport.FromJson(line) is { } report)
            _onReport(report);
    }
}
