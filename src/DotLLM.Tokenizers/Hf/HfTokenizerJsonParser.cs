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
    /// <summary>
    /// GPT-2 byte-level pre-tokenizer (Qwen, Llama-3, Granite dense, GPT-2/4).
    /// When present, routes through <see cref="ByteLevelPreTokenizer"/> +
    /// <see cref="Bpe.BpeTokenizer.CreateTiktoken"/>.
    /// </summary>
    ByteLevel,
    /// <summary>Whitespace splitting (not yet implemented).</summary>
    Whitespace,
    /// <summary>
    /// A <c>Sequence</c> wrapper of pre-tokenizers. Currently only the
    /// <c>[Split, ByteLevel]</c> composition (Qwen2/Qwen2.5) is fully wired —
    /// see <see cref="HfTokenizerSpec.ByteLevelSplitRegex"/>.
    /// </summary>
    Sequence,
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
    /// <summary>
    /// GPT-2 byte-level decoder: reverse the byte-to-unicode map and
    /// concatenate. See <see cref="ByteLevelDecoder"/>.
    /// </summary>
    ByteLevel,
    /// <summary>Sequence wrapper (unwrapped).</summary>
    Sequence,
}

/// <summary>
/// HuggingFace text normalizer kind. Only variants dotLLM understands are
/// enumerated; anything else surfaces as <see cref="Unknown"/>.
/// </summary>
public enum HfNormalizerKind
{
    /// <summary>No normalizer declared.</summary>
    None = 0,
    /// <summary>Unicode NFC normalization (Qwen2/Qwen2.5).</summary>
    Nfc,
    /// <summary>Unicode NFD normalization.</summary>
    Nfd,
    /// <summary>Unicode NFKC normalization.</summary>
    Nfkc,
    /// <summary>Unicode NFKD normalization.</summary>
    Nfkd,
    /// <summary>Declared but unrecognised.</summary>
    Unknown,
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
/// Only the fields dotLLM currently uses are materialised. Normalizer kind is
/// captured to apply NFC to the input where required (Qwen2/Qwen2.5). The
/// post-processor section is not represented — BOS/EOS insertion is handled
/// by callers who know their prompt conventions.
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

    /// <summary>
    /// When <see cref="PreTokenizerKind"/> is <see cref="HfPreTokenizerKind.ByteLevel"/>,
    /// whether the declared ByteLevel pre-tokenizer applies the default GPT-2
    /// regex (<c>use_regex: true</c>). Null otherwise.
    /// </summary>
    public required bool? ByteLevelUseRegex { get; init; }

    /// <summary>
    /// When <see cref="PreTokenizerKind"/> is <see cref="HfPreTokenizerKind.ByteLevel"/>,
    /// whether the declared ByteLevel pre-tokenizer forces a leading space
    /// (<c>add_prefix_space: true</c>). Null otherwise.
    /// </summary>
    public required bool? ByteLevelAddPrefixSpace { get; init; }

    /// <summary>
    /// When the pre-tokenizer is a <c>Sequence[Split, ByteLevel]</c> (Qwen2
    /// / Qwen2.5 style), the Split step's regex pattern string. The
    /// consumer compiles this and uses it in place of the default GPT-2
    /// regex. Null when there is no Sequence or no Split regex.
    /// </summary>
    public required string? ByteLevelSplitRegex { get; init; }

    /// <summary>Decoder stages in declared order; only the kind flag matters today.</summary>
    public required IReadOnlyList<HfDecoderStage> DecoderStages { get; init; }

    /// <summary>
    /// <see langword="true"/> when <c>model.byte_fallback</c> is set — the
    /// vocabulary contains <c>&lt;0x00&gt;</c>…<c>&lt;0xFF&gt;</c> byte tokens
    /// used to render UTF-8 bytes that don't tokenize.
    /// </summary>
    public required bool ByteFallback { get; init; }

    /// <summary>
    /// Normalizer kind declared at the top of the tokenizer pipeline. When
    /// not <see cref="HfNormalizerKind.None"/>, the input text must be
    /// normalized before encoding (Qwen2 requires NFC).
    /// </summary>
    public required HfNormalizerKind NormalizerKind { get; init; }
}

/// <summary>
/// Parses HuggingFace <c>tokenizer.json</c> files into an
/// <see cref="HfTokenizerSpec"/>. Supports the SentencePiece-style BPE layout
/// used by Llama 1/2 derivatives (including Mamba-3 ib-ssm 370M): <c>model.type
/// = "BPE"</c>, Metaspace pre-tokenizer, ByteFallback decoder — and the
/// GPT-2 / ByteLevel layout used by Qwen, Llama-3, Granite dense, and GPT-2.
/// </summary>
/// <remarks>
/// The parser is intentionally minimal — post-processor sections are skipped
/// because they add no information for the encode/decode path the adapter
/// drives. Both HF merge serialisations are accepted:
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
        bool? byteLevelUseRegex = null;
        bool? byteLevelAddPrefixSpace = null;
        string? byteLevelSplitRegex = null;

        if (root.TryGetProperty("pre_tokenizer", out JsonElement pre) &&
            pre.ValueKind == JsonValueKind.Object)
        {
            ParsePreTokenizer(
                pre,
                out preKind,
                out metaReplacement, out metaPrepend,
                out byteLevelUseRegex, out byteLevelAddPrefixSpace,
                out byteLevelSplitRegex);
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

        HfNormalizerKind normalizerKind = ParseNormalizerKind(root);

        return new HfTokenizerSpec
        {
            Vocab = vocab,
            Merges = merges,
            AddedTokens = added,
            PreTokenizerKind = preKind,
            MetaspaceReplacement = metaReplacement,
            MetaspacePrependScheme = metaPrepend,
            ByteLevelUseRegex = byteLevelUseRegex,
            ByteLevelAddPrefixSpace = byteLevelAddPrefixSpace,
            ByteLevelSplitRegex = byteLevelSplitRegex,
            DecoderStages = stages,
            ByteFallback = byteFallback,
            NormalizerKind = normalizerKind,
        };
    }

    /// <summary>
    /// Parses the pre_tokenizer object. Handles <c>Metaspace</c>,
    /// <c>ByteLevel</c> (standalone), and <c>Sequence[Split, ByteLevel]</c>
    /// (Qwen2 style) — the last collapses into
    /// <see cref="HfPreTokenizerKind.ByteLevel"/> with
    /// <see cref="HfTokenizerSpec.ByteLevelSplitRegex"/> populated so the
    /// factory can use the Split's regex instead of the default GPT-2
    /// pattern. Other kinds surface as <see cref="HfPreTokenizerKind.Unknown"/>
    /// or <see cref="HfPreTokenizerKind.Sequence"/>.
    /// </summary>
    private static void ParsePreTokenizer(
        JsonElement pre,
        out HfPreTokenizerKind kind,
        out string? metaspaceReplacement,
        out string? metaspacePrepend,
        out bool? byteLevelUseRegex,
        out bool? byteLevelAddPrefixSpace,
        out string? byteLevelSplitRegex)
    {
        kind = HfPreTokenizerKind.None;
        metaspaceReplacement = null;
        metaspacePrepend = null;
        byteLevelUseRegex = null;
        byteLevelAddPrefixSpace = null;
        byteLevelSplitRegex = null;

        if (!pre.TryGetProperty("type", out JsonElement preType) ||
            preType.ValueKind != JsonValueKind.String)
            return;

        string? typeStr = preType.GetString();
        switch (typeStr)
        {
            case "Metaspace":
                kind = HfPreTokenizerKind.Metaspace;
                if (pre.TryGetProperty("replacement", out JsonElement r) && r.ValueKind == JsonValueKind.String)
                    metaspaceReplacement = r.GetString();
                if (pre.TryGetProperty("prepend_scheme", out JsonElement ps) && ps.ValueKind == JsonValueKind.String)
                    metaspacePrepend = ps.GetString();
                break;

            case "ByteLevel":
                kind = HfPreTokenizerKind.ByteLevel;
                byteLevelUseRegex = ReadBoolProp(pre, "use_regex");
                byteLevelAddPrefixSpace = ReadBoolProp(pre, "add_prefix_space");
                break;

            case "Whitespace":
                kind = HfPreTokenizerKind.Whitespace;
                break;

            case "Sequence":
                ParsePreTokenizerSequence(
                    pre,
                    out kind,
                    out byteLevelUseRegex,
                    out byteLevelAddPrefixSpace,
                    out byteLevelSplitRegex);
                break;

            default:
                kind = HfPreTokenizerKind.Unknown;
                break;
        }
    }

    /// <summary>
    /// Parses a <c>Sequence</c> pre-tokenizer. Recognised shape is
    /// <c>[Split(regex), ByteLevel(use_regex:false)]</c> as used by Qwen2 —
    /// this resolves to <see cref="HfPreTokenizerKind.ByteLevel"/> with the
    /// Split regex captured. Any other composition becomes
    /// <see cref="HfPreTokenizerKind.Sequence"/> and is rejected by the
    /// factory as unsupported.
    /// </summary>
    private static void ParsePreTokenizerSequence(
        JsonElement sequence,
        out HfPreTokenizerKind kind,
        out bool? byteLevelUseRegex,
        out bool? byteLevelAddPrefixSpace,
        out string? byteLevelSplitRegex)
    {
        kind = HfPreTokenizerKind.Sequence;
        byteLevelUseRegex = null;
        byteLevelAddPrefixSpace = null;
        byteLevelSplitRegex = null;

        JsonElement subs = default;
        // HF ships both keys ("pretokenizers" and "pre_tokenizers") across versions.
        if (!sequence.TryGetProperty("pretokenizers", out subs) &&
            !sequence.TryGetProperty("pre_tokenizers", out subs))
            return;
        if (subs.ValueKind != JsonValueKind.Array)
            return;

        // Qwen2 shape: exactly one Split then one ByteLevel.
        string? splitRegex = null;
        bool sawByteLevel = false;
        bool? blUseRegex = null;
        bool? blAddPrefix = null;
        int stepCount = 0;

        foreach (JsonElement step in subs.EnumerateArray())
        {
            stepCount++;
            if (step.ValueKind != JsonValueKind.Object) return;
            if (!step.TryGetProperty("type", out JsonElement st) || st.ValueKind != JsonValueKind.String) return;
            string? stepType = st.GetString();

            if (stepType == "Split")
            {
                if (splitRegex is not null || sawByteLevel) return; // only the first Split ahead of ByteLevel
                splitRegex = ReadSplitRegex(step);
                if (splitRegex is null) return;
            }
            else if (stepType == "ByteLevel")
            {
                sawByteLevel = true;
                blUseRegex = ReadBoolProp(step, "use_regex");
                blAddPrefix = ReadBoolProp(step, "add_prefix_space");
            }
            else
            {
                // Unknown composition — bail to Sequence (factory will reject).
                return;
            }
        }

        if (sawByteLevel && stepCount >= 1)
        {
            kind = HfPreTokenizerKind.ByteLevel;
            byteLevelSplitRegex = splitRegex; // may be null if the sequence was [ByteLevel] alone
            byteLevelUseRegex = blUseRegex;
            byteLevelAddPrefixSpace = blAddPrefix;
        }
    }

    private static string? ReadSplitRegex(JsonElement split)
    {
        if (!split.TryGetProperty("pattern", out JsonElement pattern) ||
            pattern.ValueKind != JsonValueKind.Object)
            return null;
        if (pattern.TryGetProperty("Regex", out JsonElement rx) && rx.ValueKind == JsonValueKind.String)
            return rx.GetString();
        // We do not (yet) support literal-String or Char patterns here.
        return null;
    }

    private static bool? ReadBoolProp(JsonElement obj, string name)
    {
        if (!obj.TryGetProperty(name, out JsonElement v)) return null;
        return v.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => (bool?)null,
        };
    }

    private static HfNormalizerKind ParseNormalizerKind(JsonElement root)
    {
        if (!root.TryGetProperty("normalizer", out JsonElement norm) ||
            norm.ValueKind != JsonValueKind.Object)
            return HfNormalizerKind.None;
        if (!norm.TryGetProperty("type", out JsonElement t) || t.ValueKind != JsonValueKind.String)
            return HfNormalizerKind.Unknown;
        return t.GetString() switch
        {
            "NFC" => HfNormalizerKind.Nfc,
            "NFD" => HfNormalizerKind.Nfd,
            "NFKC" => HfNormalizerKind.Nfkc,
            "NFKD" => HfNormalizerKind.Nfkd,
            _ => HfNormalizerKind.Unknown,
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
            "ByteLevel" => HfDecoderStage.ByteLevel,
            _ => HfDecoderStage.Unknown,
        };
        stages.Add(stage);
    }
}
