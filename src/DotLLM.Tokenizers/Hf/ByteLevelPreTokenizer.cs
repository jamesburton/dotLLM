using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.Hf;

/// <summary>
/// GPT-2 / HuggingFace <c>ByteLevel</c> pre-tokenizer. Converts raw text to a
/// byte-level Unicode representation where every byte of the UTF-8 encoding
/// maps to a single printable Unicode character via the GPT-2
/// <c>bytes_to_unicode</c> mapping. The resulting characters are the
/// alphabet over which BPE merges are declared in Qwen / Llama-3 / Granite /
/// GPT-2 style <c>tokenizer.json</c> files.
/// </summary>
/// <remarks>
/// <para>
/// The GPT-2 mapping is:
/// </para>
/// <list type="bullet">
///   <item><description>Printable ASCII (33–126) → same code point.</description></item>
///   <item><description>Latin-1 supplement (161–172, 174–255) → same code point.</description></item>
///   <item><description>All other bytes (0–32, 127–160, 173) → U+0100 + n where n
///     is the running index of unassigned bytes in ascending order. This
///     pushes control / whitespace bytes out of their normal range so the BPE
///     regex does not re-split them.</description></item>
/// </list>
/// <para>
/// <b>Where this runs in the pipeline.</b> For a pure <c>ByteLevel</c>
/// pre-tokenizer with <c>use_regex: true</c>, the GPT-2 pre-tokenization
/// regex is applied to the raw text first, then each match is byte-encoded
/// and fed to BPE. For a <c>Sequence[Split(regex), ByteLevel(use_regex:false)]</c>
/// pre-tokenizer (Qwen-2/Qwen-2.5), the model-specific Split regex replaces
/// the default GPT-2 regex — <see cref="ByteLevelPreTokenizer"/> only
/// provides the byte-to-unicode transform in that case.
/// </para>
/// <para>
/// <b>Zero per-call allocations on the hot byte-mapping path.</b>
/// <see cref="MapBytesToUnicode(ReadOnlySpan{byte}, Span{char})"/> is a
/// straight table lookup; callers pass a pre-rented <see cref="Span{T}"/>
/// (typically <see cref="ArrayPool{T}"/>-rented).
/// </para>
/// <para>
/// <b>Actual encoding.</b> The BPE merge loop itself lives in
/// <see cref="DotLLM.Tokenizers.Bpe.BpeTokenizer"/> via <c>CreateTiktoken</c>;
/// this class supplies the byte alphabet and regex routing that the HF
/// <c>tokenizer.json</c> adapter uses to wire itself to that code path.
/// </para>
/// </remarks>
public static class ByteLevelPreTokenizer
{
    /// <summary>
    /// 256-entry table mapping a raw byte (0–255) to its GPT-2 Unicode
    /// character. Identical to HF <c>tokenizers</c> / <c>transformers</c>
    /// <c>bytes_to_unicode()</c>.
    /// </summary>
    public static ReadOnlySpan<char> BytesToUnicode => ByteToCharTable;

    private static readonly char[] ByteToCharTable = BuildByteToCharTable();

    /// <summary>
    /// Reverse table: Unicode char → byte (0–255), or -1 if the char is not
    /// one produced by the forward mapping. Sized to the largest mapped
    /// code point (U+0144 = 324).
    /// </summary>
    internal static readonly short[] UnicodeToByteTable = BuildUnicodeToByteTable();

    private static char[] BuildByteToCharTable()
    {
        char[] table = new char[256];
        for (int b = 33; b <= 126; b++) table[b] = (char)b;
        for (int b = 161; b <= 172; b++) table[b] = (char)b;
        for (int b = 174; b <= 255; b++) table[b] = (char)b;
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (table[b] == 0)
                table[b] = (char)(0x100 + n++);
        }
        return table;
    }

    private static short[] BuildUnicodeToByteTable()
    {
        char[] byteToChar = BuildByteToCharTable();
        int maxChar = 0;
        foreach (char c in byteToChar) if (c > maxChar) maxChar = c;
        short[] table = new short[maxChar + 1];
        for (int i = 0; i < table.Length; i++) table[i] = -1;
        for (int b = 0; b < 256; b++) table[(int)byteToChar[b]] = (short)b;
        return table;
    }

    /// <summary>
    /// Default GPT-2 pre-tokenization regex — the exact pattern HF
    /// <c>ByteLevel</c> uses when <c>use_regex: true</c>.
    /// </summary>
    /// <remarks>
    /// <c>(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+</c>
    /// </remarks>
    public static readonly Regex DefaultGpt2Regex = new(
        @"(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    /// <summary>
    /// Writes the GPT-2 byte-level Unicode encoding of <paramref name="bytes"/>
    /// into <paramref name="destination"/>. One output char per input byte.
    /// </summary>
    /// <param name="bytes">Source bytes (typically UTF-8 of a text chunk).</param>
    /// <param name="destination">
    /// Output buffer; must be at least <paramref name="bytes"/>.Length chars.
    /// </param>
    /// <returns>Number of chars written (equal to <paramref name="bytes"/>.Length).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int MapBytesToUnicode(ReadOnlySpan<byte> bytes, Span<char> destination)
    {
        if (destination.Length < bytes.Length)
            throw new ArgumentException("destination too small", nameof(destination));
        for (int i = 0; i < bytes.Length; i++)
            destination[i] = ByteToCharTable[bytes[i]];
        return bytes.Length;
    }

    /// <summary>
    /// Convenience allocating overload: UTF-8 encodes <paramref name="text"/>
    /// then applies the byte-to-unicode mapping, returning a fresh string.
    /// Primarily intended for tests and tooling — the hot tokenizer path
    /// uses rented buffers via <see cref="MapBytesToUnicode"/>.
    /// </summary>
    public static string ApplyByteLevel(ReadOnlySpan<char> text)
    {
        int utf8Len = Encoding.UTF8.GetByteCount(text);
        byte[] utf8 = ArrayPool<byte>.Shared.Rent(utf8Len);
        try
        {
            Encoding.UTF8.GetBytes(text, utf8);
            return string.Create(utf8Len, utf8, static (dst, src) =>
            {
                for (int i = 0; i < dst.Length; i++)
                    dst[i] = ByteToCharTable[src[i]];
            });
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(utf8);
        }
    }
}
