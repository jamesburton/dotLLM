using System.Text.Json;

namespace DotLLM.Tokenizers.WordPiece;

/// <summary>
/// Builds a <see cref="WordPieceTokenizer"/> from a HuggingFace <c>tokenizer.json</c>
/// whose <c>model.type</c> is <c>WordPiece</c>.
/// </summary>
/// <remarks>
/// Only the settings that change token ids are honoured: <c>model.vocab</c>, <c>model.unk_token</c>,
/// <c>model.max_input_chars_per_word</c>, and the normalizer's lower-casing / accent stripping /
/// CJK handling (a <c>BertNormalizer</c>, or a <c>Sequence</c> containing <c>Lowercase</c>,
/// <c>StripAccents</c>). <c>[CLS]</c>/<c>[SEP]</c> ids are resolved by string, matching the
/// <c>BertProcessing</c>/<c>TemplateProcessing</c> post-processors every BERT checkpoint ships.
/// </remarks>
public static class HfWordPieceLoader
{
    /// <summary>True when the JSON declares a WordPiece model.</summary>
    public static bool IsWordPiece(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.TryGetProperty("model", out var m)
            && m.TryGetProperty("type", out var t) && t.GetString() == "WordPiece";
    }

    /// <summary>Parses a <c>tokenizer.json</c> document.</summary>
    /// <exception cref="InvalidDataException">Not a WordPiece tokenizer, or [CLS]/[SEP]/unk missing.</exception>
    public static WordPieceTokenizer Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (!root.TryGetProperty("model", out var model)
            || !model.TryGetProperty("type", out var type) || type.GetString() != "WordPiece")
            throw new InvalidDataException("tokenizer.json is not a WordPiece tokenizer (model.type != WordPiece).");

        var vocabObj = model.GetProperty("vocab");
        int max = -1;
        foreach (var p in vocabObj.EnumerateObject())
            max = Math.Max(max, p.Value.GetInt32());

        // added_tokens may extend the vocab beyond model.vocab.
        if (root.TryGetProperty("added_tokens", out var added))
            foreach (var a in added.EnumerateArray())
                max = Math.Max(max, a.GetProperty("id").GetInt32());

        var tokens = new string[max + 1];
        Array.Fill(tokens, string.Empty);
        foreach (var p in vocabObj.EnumerateObject())
            tokens[p.Value.GetInt32()] = p.Name;
        if (root.TryGetProperty("added_tokens", out added))
            foreach (var a in added.EnumerateArray())
                tokens[a.GetProperty("id").GetInt32()] = a.GetProperty("content").GetString()!;

        int Find(string s) => Array.IndexOf(tokens, s);

        string unk = model.TryGetProperty("unk_token", out var u) && u.ValueKind == JsonValueKind.String
            ? u.GetString()! : "[UNK]";
        int unkId = Find(unk), clsId = Find("[CLS]"), sepId = Find("[SEP]");
        if (unkId < 0 || clsId < 0 || sepId < 0)
            throw new InvalidDataException("WordPiece tokenizer.json lacks [CLS]/[SEP]/unk tokens.");

        int maxChars = model.TryGetProperty("max_input_chars_per_word", out var mc) && mc.ValueKind == JsonValueKind.Number
            ? mc.GetInt32() : 100;

        bool lower = false, chinese = false;
        bool? strip = null;
        if (root.TryGetProperty("normalizer", out var norm) && norm.ValueKind == JsonValueKind.Object)
            ReadNormalizer(norm, ref lower, ref strip, ref chinese);

        return new WordPieceTokenizer(
            tokens, clsId, sepId, unkId, Find("[PAD]"),
            lowerCase: lower, stripAccents: strip ?? lower, tokenizeChinese: chinese, maxCharsPerWord: maxChars);
    }

    private static void ReadNormalizer(JsonElement n, ref bool lower, ref bool? strip, ref bool chinese)
    {
        string? kind = n.TryGetProperty("type", out var t) ? t.GetString() : null;
        switch (kind)
        {
            case "BertNormalizer":
                lower = n.TryGetProperty("lowercase", out var l) && l.ValueKind != JsonValueKind.Null && l.GetBoolean();
                chinese = !n.TryGetProperty("handle_chinese_chars", out var c)
                          || c.ValueKind == JsonValueKind.Null || c.GetBoolean();
                if (n.TryGetProperty("strip_accents", out var s) && s.ValueKind is JsonValueKind.True or JsonValueKind.False)
                    strip = s.GetBoolean();
                break;
            case "Lowercase":
                lower = true;
                break;
            case "StripAccents":
                strip = true;
                break;
            case "Sequence":
                if (n.TryGetProperty("normalizers", out var arr))
                    foreach (var child in arr.EnumerateArray())
                        ReadNormalizer(child, ref lower, ref strip, ref chinese);
                break;
        }
    }
}
