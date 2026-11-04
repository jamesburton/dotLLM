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

/// <summary>
/// BPE-like word tokenizer: an identifier is one token, and a space or '.' directly before it is glued on (" Add", ".Add"),
/// so a name usually starts inside its token; every other character is its own token. Ids are assigned on first sight.
/// </summary>
internal sealed class GluingTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _ids = new(StringComparer.Ordinal);
    private readonly List<string> _pieces = [];

    public int[] Encode(string text)
    {
        var ids = new List<int>();
        for (int i = 0; i < text.Length;)
        {
            int start = i;
            if ((text[i] == ' ' || text[i] == '.') && i + 1 < text.Length && IsStart(text[i + 1])) i++;
            if (IsStart(text[i]))
                while (i < text.Length && (char.IsLetterOrDigit(text[i]) || text[i] == '_')) i++;
            else i++;
            ids.Add(Id(text[start..i]));
        }
        return ids.ToArray();
    }

    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var sb = new StringBuilder();
        foreach (int id in tokenIds) sb.Append(_pieces[id]);
        return sb.ToString();
    }

    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
    public string DecodeToken(int tokenId) => _pieces[tokenId];
    public int BosTokenId => -1;
    public int EosTokenId => -1;
    public int VocabSize => _pieces.Count;
    public int CountTokens(string text) => Encode(text).Length;

    private static bool IsStart(char c) => char.IsLetter(c) || c == '_';

    private int Id(string piece)
    {
        if (_ids.TryGetValue(piece, out int id)) return id;
        _ids[piece] = _pieces.Count;
        _pieces.Add(piece);
        return _pieces.Count - 1;
    }
}

/// <summary>
/// One token per char except that a maximal run of '\n' is a single token (a "\n\n" merge, as in Qwen's and Llama's BPE).
/// Two chunks that meet at a newline on each side therefore tokenise differently shown back to back than alone.
/// </summary>
internal sealed class NewlineRunTokenizer : ITokenizer
{
    private const int RunBase = 100_000;

    public int[] Encode(string text)
    {
        var ids = new List<int>();
        for (int i = 0; i < text.Length;)
        {
            if (text[i] != '\n') { ids.Add(text[i++]); continue; }
            int s = i;
            while (i < text.Length && text[i] == '\n') i++;
            ids.Add(RunBase + (i - s));
        }
        return ids.ToArray();
    }

    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var sb = new StringBuilder();
        foreach (int id in tokenIds) sb.Append(id >= RunBase ? new string('\n', id - RunBase) : ((char)id).ToString());
        return sb.ToString();
    }

    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
    public string DecodeToken(int tokenId) => Decode([tokenId]);
    public int BosTokenId => -1;
    public int EosTokenId => -1;
    public int VocabSize => 200_000;
    public int CountTokens(string text) => Encode(text).Length;
}

/// <summary>Every encode ends with one marker token, so no two texts ever concatenate stably (no stable edge exists).</summary>
internal sealed class EndMarkerTokenizer : ITokenizer
{
    public int[] Encode(string text) => text.Select(ch => (int)ch).Append(-7).ToArray();
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var sb = new StringBuilder();
        foreach (int id in tokenIds) if (id >= 0) sb.Append((char)id);
        return sb.ToString();
    }

    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
    public string DecodeToken(int tokenId) => Decode([tokenId]);
    public int BosTokenId => -1;
    public int EosTokenId => -1;
    public int VocabSize => 65536;
    public int CountTokens(string text) => text.Length + 1;
}

/// <summary>One token per char except that a '\n' directly followed by 'x' is one token (a merge across a line end).</summary>
internal sealed class NewlineXTokenizer : ITokenizer
{
    private const int Merged = 100_000;

    public int[] Encode(string text)
    {
        var ids = new List<int>();
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] == '\n' && i + 1 < text.Length && text[i + 1] == 'x') { ids.Add(Merged); i++; }
            else ids.Add(text[i]);
        }
        return ids.ToArray();
    }

    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var sb = new StringBuilder();
        foreach (int id in tokenIds) sb.Append(id == Merged ? "\nx" : ((char)id).ToString());
        return sb.ToString();
    }

    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
    public string DecodeToken(int tokenId) => Decode([tokenId]);
    public int BosTokenId => -1;
    public int EosTokenId => -1;
    public int VocabSize => 200_000;
    public int CountTokens(string text) => Encode(text).Length;
}
