using System.Text.Json;

namespace DotLLM.Tokenizers.Hf;

/// <summary>
/// Supported HuggingFace pre-tokenizer kinds. Only the variants dotLLM understands
/// are enumerated; unrecognised values surface as <see cref="Unknown"/>.
/// </summary>
public enum HfPreTokenizerKind
{
    /// <summary>No pre-tokenizer or unrecognised type.</summary>
    None = 0,
    /// <summary>
    /// SentencePiece-style leading-space marker (U+2581). Maps to
    /// <see cref="Bpe.BpeTokenizer.CreateSentencePiece"/> semantics.
    /// </summary>
    Metaspace,
    /// <summary>GPT-2 byte-level pre-tokenizer (not yet implemented in this adapter).</summary>
    ByteLevel,
    /// <summary>Whitespace splitting (not yet implemented).</summary>
    Whitespace,
    /// <summary>An unrecognised pre-tokenizer type — caller may warn or fall back.</summary>
    Unknown,
}

/// <summary>
/// Supported HuggingFace decoder kinds present in the pipeline. Only the ones
/// dotLLM handles are enumerated.
/// </summary>
public enum HfDecoderStage
{
    /// <summary>Unrecognised decoder stage — ignored.</summary>
    Unknown = 0,
    /// <summary>Metaspace replace (▁ → ' ').</summary>
    Replace,
    /// <summary>Byte-fallback (<c>&lt;0xNN&gt;</c> → raw byte).</summary>
    ByteFallback,
    /// <summary>Fuse adjacent byte-fallback tokens.</summary>
    Fuse,
    /// <summary>Strip leading/trailing whitespace.</summary>
    Strip,
    /// <summary>Sequence wrapper (unwrapped).</summary>
    Sequence,
}

/// <summary>
/// A HuggingFace-declared added token (usually a special token like
/// <c>&lt;s&gt;</c> / <c>&lt;/s&gt;</c> / <c>&lt;unk&gt;</c>).
/// </summary>
/// <param name="Id">Vocabulary ID.</param>
/// <param name="Content">Literal token string.</param>
/// <param name="Special">
/// <see langword="true"/> when the token is flagged <c>special: true</c> in the
/// HF JSON — such tokens should pre-split the input during encoding rather than
/// going through the BPE merge loop.
/// </param>
public readonly record struct HfAddedToken(int Id, string Content, bool Special);

/// <summary>
/// Parsed representation of a HuggingFace <c>tokenizer.json</c>. Populated by
/// <see cref="HfTokenizerJsonParser.Parse(string)"/>; consumed by
/// <see cref="HfBpeTokenizerFactory.Create(HfTokenizerSpec, int, int)"/>.
/// </summary>
/// <remarks>
/// Only the fields dotLLM currently uses are materialised. Normalizer and
/// post-processor sections are not represented — normalizer is null for
/// Llama-style tokenizers (the only variant supported today), and the
/// post-processor's BOS insertion is handled by callers who know their prompt
/// conventions.
/// </remarks>
public sealed class HfTokenizerSpec
{
    /// <summary>Raw <c>model.vocab</c> map (token text → id).</summary>
    public required IReadOnlyDictionary<string, int> Vocab { get; init; }

    /// <summary>
    /// Merge pairs in rank order. Index 0 = highest priority. Each entry is the
    /// <c>(left, right)</c> pair whose concatenation is the merged token; the
    /// merged token is expected to be present in <see cref="Vocab"/>.
    /// </summary>
    public required IReadOnlyList<(string Left, string Right)> Merges { get; init; }

    /// <summary>Declared added tokens (usually special tokens).</summary>
    public required IReadOnlyList<HfAddedToken> AddedTokens { get; init; }

    /// <summary>Pre-tokenizer kind declared in the JSON (<see cref="HfPreTokenizerKind.None"/> if absent).</summary>
    public required HfPreTokenizerKind PreTokenizerKind { get; init; }

    /// <summary>
    /// When <see cref="PreTokenizerKind"/> is <see cref="HfPreTokenizerKind.Metaspace"/>,
    /// the replacement character (usually <c>▁</c>, U+2581). Null otherwise.
    /// </summary>
    public required string? MetaspaceReplacement { get; init; }

    /// <summary>
    /// When <see cref="PreTokenizerKind"/> is <see cref="HfPreTokenizerKind.Metaspace"/>,
    /// the prepend scheme (<c>"first"</c>, <c>"always"</c>, or <c>"never"</c>).
    /// Null otherwise.
    /// </summary>
    public required string? MetaspacePrependScheme { get; init; }

    /// <summary>Decoder stages in declared order; only the kind flag matters today.</summary>
    public required IReadOnlyList<HfDecoderStage> DecoderStages { get; init; }

    /// <summary>
    /// <see langword="true"/> when <c>model.byte_fallback</c> is set — the
    /// vocabulary contains <c>&lt;0x00&gt;</c>…<c>&lt;0xFF&gt;</c> byte tokens
    /// used to render UTF-8 bytes that don't tokenize.
    /// </summary>
    public required bool ByteFallback { get; init; }
}

/// <summary>
/// Parses HuggingFace <c>tokenizer.json</c> files into an
/// <see cref="HfTokenizerSpec"/>. Supports the SentencePiece-style BPE layout
/// used by Llama 1/2 derivatives (including Mamba-3 ib-ssm 370M): <c>model.type
/// = "BPE"</c>, Metaspace pre-tokenizer, ByteFallback decoder.
/// </summary>
/// <remarks>
/// The parser is intentionally minimal — normalizer / post-processor sections
/// are skipped because they add no information for the encode/decode path the
/// adapter drives. Both HF merge serialisations are accepted:
/// <list type="bullet">
///   <item><description>Space-separated string form (<c>"a b"</c>) — older tokenizer versions.</description></item>
///   <item><description>JSON array form (<c>["a", "b"]</c>) — tokenizers ≥ 0.15.</description></item>
/// </list>
/// </remarks>
public static class HfTokenizerJsonParser
{
    /// <summary>Extracts the <c>model.vocab</c> dictionary.</summary>
    /// <param name="root">Root JSON element of <c>tokenizer.json</c>.</param>
    /// <returns>Map from token string to integer ID.</returns>
    /// <exception cref="InvalidDataException">Vocab section missing or malformed.</exception>
    public static IReadOnlyDictionary<string, int> ParseVocab(JsonElement root)
    {
        if (!root.TryGetProperty("model", out JsonElement model))
            throw new InvalidDataException("tokenizer.json has no 'model' section.");
        if (!model.TryGetProperty("vocab", out JsonElement vocab) || vocab.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("tokenizer.json 'model.vocab' missing or not an object.");

        var dict = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (JsonProperty entry in vocab.EnumerateObject())
        {
            if (entry.Value.ValueKind != JsonValueKind.Number)
                continue;
            dict[entry.Name] = entry.Value.GetInt32();
        }
        return dict;
    }

    /// <summary>
    /// Extracts the ordered merge list. Accepts both legacy
    /// <c>"a b"</c> strings and modern <c>["a","b"]</c> arrays. Merges whose
    /// encoding can't be decoded are skipped (a corrupt merge would simply
    /// never fire, never crashing the tokenizer).
    /// </summary>
    public static IReadOnlyList<(string Left, string Right)> ParseMerges(JsonElement root)
    {
        if (!root.TryGetProperty("model", out JsonElement model) ||
            !model.TryGetProperty("merges", out JsonElement merges) ||
            merges.ValueKind != JsonValueKind.Array)
        {
            return Array.Empty<(string, string)>();
        }

        var list = new List<(string, string)>(merges.GetArrayLength());
        foreach (JsonElement entry in merges.EnumerateArray())
        {
            if (entry.ValueKind == JsonValueKind.String)
            {
                string? raw = entry.GetString();
                if (string.IsNullOrEmpty(raw)) continue;
                int sp = raw.IndexOf(' ');
                if (sp <= 0 || sp >= raw.Length - 1) continue;
                list.Add((raw[..sp], raw[(sp + 1)..]));
            }
            else if (entry.ValueKind == JsonValueKind.Array && entry.GetArrayLength() == 2)
            {
                string? a = entry[0].GetString();
                string? b = entry[1].GetString();
                if (a is null || b is null) continue;
                list.Add((a, b));
            }
            // Other formats silently ignored — matches HF tolerant behaviour.
        }
        return list;
    }

    /// <summary>Extracts declared added tokens.</summary>
    public static IReadOnlyList<HfAddedToken> ParseAddedTokens(JsonElement root)
    {
        if (!root.TryGetProperty("added_tokens", out JsonElement added) ||
            added.ValueKind != JsonValueKind.Array)
        {
            return Array.Empty<HfAddedToken>();
        }

        var list = new List<HfAddedToken>(added.GetArrayLength());
        foreach (JsonElement entry in added.EnumerateArray())
        {
            if (entry.ValueKind != JsonValueKind.Object) continue;
            if (!entry.TryGetProperty("id", out JsonElement idEl) || idEl.ValueKind != JsonValueKind.Number) continue;
            if (!entry.TryGetProperty("content", out JsonElement contentEl) || contentEl.ValueKind != JsonValueKind.String) continue;
            bool special = entry.TryGetProperty("special", out JsonElement sp) &&
                           sp.ValueKind == JsonValueKind.True;
            list.Add(new HfAddedToken(idEl.GetInt32(), contentEl.GetString()!, special));
        }
        return list;
    }

    /// <summary>Parses the full spec from a JSON string.</summary>
    public static HfTokenizerSpec Parse(string jsonContent)
    {
        using JsonDocument doc = JsonDocument.Parse(jsonContent);
        return ParseElement(doc.RootElement);
    }

    /// <summary>Parses the full spec from a pre-opened <see cref="JsonElement"/>.</summary>
    public static HfTokenizerSpec ParseElement(JsonElement root)
    {
        IReadOnlyDictionary<string, int> vocab = ParseVocab(root);
        IReadOnlyList<(string, string)> merges = ParseMerges(root);
        IReadOnlyList<HfAddedToken> added = ParseAddedTokens(root);

        HfPreTokenizerKind preKind = HfPreTokenizerKind.None;
        string? metaReplacement = null;
        string? metaPrepend = null;
        if (root.TryGetProperty("pre_tokenizer", out JsonElement pre) &&
            pre.ValueKind == JsonValueKind.Object &&
            pre.TryGetProperty("type", out JsonElement preType) &&
            preType.ValueKind == JsonValueKind.String)
        {
            preKind = preType.GetString() switch
            {
                "Metaspace" => HfPreTokenizerKind.Metaspace,
                "ByteLevel" => HfPreTokenizerKind.ByteLevel,
                "Whitespace" => HfPreTokenizerKind.Whitespace,
                _ => HfPreTokenizerKind.Unknown,
            };
            if (preKind == HfPreTokenizerKind.Metaspace)
            {
                if (pre.TryGetProperty("replacement", out JsonElement r) && r.ValueKind == JsonValueKind.String)
                    metaReplacement = r.GetString();
                if (pre.TryGetProperty("prepend_scheme", out JsonElement ps) && ps.ValueKind == JsonValueKind.String)
                    metaPrepend = ps.GetString();
            }
        }

        var stages = new List<HfDecoderStage>();
        if (root.TryGetProperty("decoder", out JsonElement dec) && dec.ValueKind == JsonValueKind.Object)
        {
            CollectDecoderStages(dec, stages);
        }

        bool byteFallback = false;
        if (root.TryGetProperty("model", out JsonElement model) &&
            model.TryGetProperty("byte_fallback", out JsonElement bfEl) &&
            bfEl.ValueKind == JsonValueKind.True)
        {
            byteFallback = true;
        }

        return new HfTokenizerSpec
        {
            Vocab = vocab,
            Merges = merges,
            AddedTokens = added,
            PreTokenizerKind = preKind,
            MetaspaceReplacement = metaReplacement,
            MetaspacePrependScheme = metaPrepend,
            DecoderStages = stages,
            ByteFallback = byteFallback,
        };
    }

    private static void CollectDecoderStages(JsonElement decoder, List<HfDecoderStage> stages)
    {
        if (!decoder.TryGetProperty("type", out JsonElement t) || t.ValueKind != JsonValueKind.String)
            return;

        string? type = t.GetString();
        if (type == "Sequence")
        {
            if (decoder.TryGetProperty("decoders", out JsonElement seq) && seq.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement child in seq.EnumerateArray())
                    CollectDecoderStages(child, stages);
            }
            return;
        }

        HfDecoderStage stage = type switch
        {
            "Replace" => HfDecoderStage.Replace,
            "ByteFallback" => HfDecoderStage.ByteFallback,
            "Fuse" => HfDecoderStage.Fuse,
            "Strip" => HfDecoderStage.Strip,
            _ => HfDecoderStage.Unknown,
        };
        stages.Add(stage);
    }
}
