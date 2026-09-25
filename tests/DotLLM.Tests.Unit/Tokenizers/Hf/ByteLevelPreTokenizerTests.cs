using System.Text;
using DotLLM.Tokenizers.Hf;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Hf;

/// <summary>
/// Unit tests for the ByteLevel helpers. These cover only the stand-alone
/// behaviour of <see cref="ByteLevelPreTokenizer"/> and
/// <see cref="ByteLevelDecoder"/> — the end-to-end HF <c>tokenizer.json</c>
/// integration tests live in <see cref="HfByteLevelBpeTokenizerTests"/>.
/// </summary>
public class ByteLevelPreTokenizerTests
{
    [Fact]
    public void BytesToUnicode_PrintableAsciiMapsToItself()
    {
        ReadOnlySpan<char> table = ByteLevelPreTokenizer.BytesToUnicode;
        for (int b = 33; b <= 126; b++)
            Assert.Equal((char)b, table[b]);
    }

    [Fact]
    public void BytesToUnicode_Latin1SupplementMapsToItself()
    {
        ReadOnlySpan<char> table = ByteLevelPreTokenizer.BytesToUnicode;
        for (int b = 161; b <= 172; b++)
            Assert.Equal((char)b, table[b]);
        for (int b = 174; b <= 255; b++)
            Assert.Equal((char)b, table[b]);
    }

    [Fact]
    public void BytesToUnicode_ControlBytesMapIntoPrivateRange()
    {
        ReadOnlySpan<char> table = ByteLevelPreTokenizer.BytesToUnicode;
        // 0x00..0x20, 0x7F..0xA0, 0xAD must map to U+0100+n (ascending).
        // First unassigned byte (0x00) → U+0100.
        Assert.Equal((char)0x0100, table[0]);
        // 0x20 (space) → should be the 33rd unassigned byte → U+0120.
        Assert.Equal((char)0x0120, table[0x20]);
    }

    [Fact]
    public void BytesToUnicode_AllCharsAreUnique()
    {
        ReadOnlySpan<char> table = ByteLevelPreTokenizer.BytesToUnicode;
        var seen = new HashSet<char>();
        for (int i = 0; i < 256; i++)
            Assert.True(seen.Add(table[i]), $"duplicate char for byte {i:X2}");
    }

    [Fact]
    public void ApplyByteLevel_HelloWorld_MatchesHfReference()
    {
        // HF `bytes_to_unicode()` applied to "Hello world" (UTF-8 = ASCII) →
        // "Hello" + U+0120 + "world". Validates the mapping we will feed to
        // the BPE merge table.
        string result = ByteLevelPreTokenizer.ApplyByteLevel("Hello world");
        Assert.Equal("Hello\u0120world", result);
    }

    [Fact]
    public void ApplyByteLevel_NonAscii_EncodesMultipleBytes()
    {
        // 'é' = 0xC3 0xA9 (both Latin-1 supplement → identity).
        string result = ByteLevelPreTokenizer.ApplyByteLevel("é");
        Assert.Equal("\u00c3\u00a9", result);
    }

    [Fact]
    public void ApplyByteLevel_EmptyString_Roundtrips()
    {
        string result = ByteLevelPreTokenizer.ApplyByteLevel(string.Empty);
        Assert.Equal(string.Empty, result);
    }

    [Fact]
    public void MapBytesToUnicode_WritesOneCharPerByte()
    {
        ReadOnlySpan<byte> bytes = [0x48, 0x65, 0x6C, 0x6C, 0x6F]; // "Hello"
        Span<char> dest = stackalloc char[bytes.Length];
        int written = ByteLevelPreTokenizer.MapBytesToUnicode(bytes, dest);
        Assert.Equal(5, written);
        Assert.Equal("Hello", dest.ToString());
    }

    [Fact]
    public void Decoder_RoundTrip_ConcatenatesBytesAcrossTokens()
    {
        // Build a miniature "vocab": each entry is a single-byte-encoded token.
        // Token 0 = ByteLevel('H'), token 1 = ByteLevel('i'), token 2 = ByteLevel(' ').
        string[] idToToken = new string[3];
        byte[] raw = [(byte)'H', (byte)'i', (byte)' '];
        for (int i = 0; i < raw.Length; i++)
            idToToken[i] = ByteLevelPreTokenizer.BytesToUnicode[raw[i]].ToString();

        // Decode [0, 2, 1] → "H" + " " + "i" = "H i".
        string decoded = ByteLevelDecoder.Decode([0, 2, 1], idToToken);
        Assert.Equal("H i", decoded);
    }

    [Fact]
    public void Decoder_MultiByteUtf8Spanning_DecodesCorrectly()
    {
        // Encode 'é' (0xC3 0xA9) as two separate tokens. The decoder must
        // concatenate the byte stream first, THEN UTF-8 decode — otherwise
        // the individual tokens would be invalid UTF-8 and produce
        // replacement chars.
        string[] idToToken = [
            ByteLevelPreTokenizer.BytesToUnicode[0xC3].ToString(),
            ByteLevelPreTokenizer.BytesToUnicode[0xA9].ToString(),
        ];
        string decoded = ByteLevelDecoder.Decode([0, 1], idToToken);
        Assert.Equal("é", decoded);
    }

    [Fact]
    public void Decoder_EmptyInput_ReturnsEmpty()
    {
        Assert.Equal(string.Empty, ByteLevelDecoder.Decode(ReadOnlySpan<int>.Empty, []));
    }

    [Fact]
    public void Decoder_IgnoresOutOfRangeIds()
    {
        string[] idToToken = [ByteLevelPreTokenizer.BytesToUnicode[(byte)'A'].ToString()];
        // id 1 is out of range → skipped; id 0 emits 'A'.
        string decoded = ByteLevelDecoder.Decode([0, 1, 0], idToToken);
        Assert.Equal("AA", decoded);
    }

    [Fact]
    public void DefaultGpt2Regex_SplitsContractionsAndWords()
    {
        var r = ByteLevelPreTokenizer.DefaultGpt2Regex;
        var matches = r.Matches("Hello world's");
        Assert.Equal("Hello", matches[0].Value);
        Assert.Equal(" world", matches[1].Value);
        Assert.Equal("'s", matches[2].Value);
    }
}
