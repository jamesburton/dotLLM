using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>One token per UTF-16 char (id = char code); counts <see cref="Encode"/> calls.</summary>
internal sealed class CharTokenizer : ITokenizer
{
    public int EncodeCalls { get; private set; }
    public int[] Encode(string text) { EncodeCalls++; return text.Select(ch => (int)ch).ToArray(); }
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var chars = new char[tokenIds.Length];
        for (int i = 0; i < chars.Length; i++) chars[i] = (char)tokenIds[i];
        return new string(chars);
    }
    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
    public string DecodeToken(int tokenId) => ((char)tokenId).ToString();
    public int BosTokenId => -1;
    public int EosTokenId => -1;
    public int VocabSize => 65536;
    public int CountTokens(string text) => text.Length;
}

/// <summary>
/// One token per UTF-8 byte, like a byte-level BPE without merges: a multi-byte character spans several tokens, and decoding
/// part of one gives U+FFFD replacement characters.
/// </summary>
internal sealed class ByteTokenizer : ITokenizer
{
    public int[] Encode(string text) => Encoding.UTF8.GetBytes(text).Select(b => (int)b).ToArray();
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var bytes = new byte[tokenIds.Length];
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)tokenIds[i];
        return Encoding.UTF8.GetString(bytes);
    }
    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
    public string DecodeToken(int tokenId) => Decode([tokenId]);
    public int BosTokenId => -1;
    public int EosTokenId => -1;
    public int VocabSize => 256;
    public int CountTokens(string text) => Encoding.UTF8.GetByteCount(text);
}
