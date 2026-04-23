using System.Text.Json;

namespace DotLLM.Tokenizers.Hf;

/// <summary>
/// Builds an <see cref="HfTokenizerSpec"/> from the legacy HuggingFace
/// tokenizer trio — <c>vocab.json</c> + <c>merges.txt</c> + optionally
/// <c>tokenizer_config.json</c> — used by checkpoints that predate the
/// consolidated <c>tokenizer.json</c> file (GPT-2 proper, older Llama/Mistral
/// variants, and IBM Granite-3.0 MoE in particular).
/// </summary>
/// <remarks>
/// <para>
/// <b>Pipeline assumed.</b> The legacy trio is effectively always a GPT-2
/// <c>ByteLevel</c> BPE (<c>tokenizer_class: GPT2Tokenizer</c>). The loader
/// therefore synthesises a spec with <see cref="HfPreTokenizerKind.ByteLevel"/>
/// + <c>use_regex: true</c> (default GPT-2 regex) + <see cref="HfDecoderStage.ByteLevel"/>
/// and no normalizer — matching the behaviour of
/// <c>transformers.AutoTokenizer.from_pretrained(...)</c> for these repos.
/// </para>
/// <para>
/// <b>Special tokens.</b> Parsed from <c>tokenizer_config.json</c>'s
/// <c>added_tokens_decoder</c> map (keys = string IDs, values carry
/// <c>content</c> / <c>special</c>). The top-level <c>bos_token</c> /
/// <c>eos_token</c> / <c>unk_token</c> strings are consulted when populating
/// the <see cref="HfAddedToken"/> list — even if they weren't explicitly
/// flagged <c>special: true</c>, they are promoted to specials so that
/// <see cref="HfBpeTokenizerFactory"/> can pre-split inputs on them.
/// </para>
/// <para>
/// <b>Out of scope.</b> Checkpoints shipping only <c>tokenizer.model</c>
/// (SentencePiece binary, e.g. classic Llama-1/2 repos without <c>vocab.json</c>)
/// are not handled here — they require a proto parser and are tracked
/// separately. This loader returns <see langword="null"/> when no
/// <c>vocab.json</c> / <c>merges.txt</c> pair is present.
/// </para>
/// </remarks>
public static class HfLegacyBpeLoader
{
    /// <summary>
    /// Attempts to construct an <see cref="HfTokenizerSpec"/> from a
    /// checkpoint directory that ships the legacy trio. Returns
    /// <see langword="null"/> when either <c>vocab.json</c> or
    /// <c>merges.txt</c> is missing.
    /// </summary>
    /// <param name="directory">Checkpoint directory.</param>
    /// <returns>A populated spec or <see langword="null"/>.</returns>
    public static HfTokenizerSpec? TryLoad(string directory)
    {
        ArgumentNullException.ThrowIfNull(directory);

        string vocabPath = Path.Combine(directory, "vocab.json");
        string mergesPath = Path.Combine(directory, "merges.txt");
        if (!File.Exists(vocabPath) || !File.Exists(mergesPath))
            return null;

        IReadOnlyDictionary<string, int> vocab = ParseVocabFile(vocabPath);
        IReadOnlyList<(string Left, string Right)> merges = ParseMergesFile(mergesPath);

        string tokenizerConfigPath = Path.Combine(directory, "tokenizer_config.json");
        IReadOnlyList<HfAddedToken> added = File.Exists(tokenizerConfigPath)
            ? ParseTokenizerConfig(tokenizerConfigPath, vocab)
            : Array.Empty<HfAddedToken>();

        return new HfTokenizerSpec
        {
            Vocab = vocab,
            Merges = merges,
            AddedTokens = added,
            // GPT-2-style: ByteLevel with default GPT-2 regex, no Split sequence,
            // no normalizer. These are the defaults HF assumes when there is no
            // tokenizer.json to override them.
            PreTokenizerKind = HfPreTokenizerKind.ByteLevel,
            MetaspaceReplacement = null,
            MetaspacePrependScheme = null,
            ByteLevelUseRegex = true,
            ByteLevelAddPrefixSpace = false,
            ByteLevelSplitRegex = null,
            DecoderStages = new[] { HfDecoderStage.ByteLevel },
            ByteFallback = false,
            NormalizerKind = HfNormalizerKind.None,
        };
    }

    /// <summary>Parses <c>vocab.json</c> — a flat <c>{token: id}</c> map.</summary>
    private static IReadOnlyDictionary<string, int> ParseVocabFile(string path)
    {
        using FileStream fs = File.OpenRead(path);
        using JsonDocument doc = JsonDocument.Parse(fs);
        if (doc.RootElement.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException($"vocab.json at '{path}' is not a JSON object.");

        var dict = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (JsonProperty entry in doc.RootElement.EnumerateObject())
        {
            if (entry.Value.ValueKind != JsonValueKind.Number) continue;
            dict[entry.Name] = entry.Value.GetInt32();
        }
        return dict;
    }

    /// <summary>
    /// Parses <c>merges.txt</c>. The first line may be a <c>#version: ...</c>
    /// header (HF emits <c>#version: 0.2</c> by convention); any line
    /// starting with <c>#</c> or that is blank is skipped. Non-comment
    /// lines are expected to be <c>"left right"</c> pairs separated by a
    /// single space.
    /// </summary>
    private static IReadOnlyList<(string Left, string Right)> ParseMergesFile(string path)
    {
        var list = new List<(string, string)>(capacity: 50_000);
        // Explicit UTF-8: merges may contain the Ġ (U+0120) byte-level marker.
        foreach (string rawLine in File.ReadLines(path, System.Text.Encoding.UTF8))
        {
            if (rawLine.Length == 0) continue;
            if (rawLine[0] == '#') continue;

            int sp = rawLine.IndexOf(' ');
            // Skip malformed lines (no space, or space at either end) rather
            // than crash — matches the tolerant behaviour HF tokenizers use.
            if (sp <= 0 || sp >= rawLine.Length - 1) continue;
            list.Add((rawLine[..sp], rawLine[(sp + 1)..]));
        }
        return list;
    }

    /// <summary>
    /// Parses the added-tokens list from <c>tokenizer_config.json</c>. Looks
    /// at <c>added_tokens_decoder</c> (the canonical modern layout: string ID
    /// keys → object with <c>content</c> / <c>special</c>) and additionally
    /// promotes the top-level <c>bos_token</c> / <c>eos_token</c> /
    /// <c>unk_token</c> / <c>pad_token</c> to specials when they are present
    /// in the vocabulary but not already in the decoder list.
    /// </summary>
    private static IReadOnlyList<HfAddedToken> ParseTokenizerConfig(
        string path,
        IReadOnlyDictionary<string, int> vocab)
    {
        using FileStream fs = File.OpenRead(path);
        using JsonDocument doc = JsonDocument.Parse(fs);
        if (doc.RootElement.ValueKind != JsonValueKind.Object)
            return Array.Empty<HfAddedToken>();

        var byId = new Dictionary<int, HfAddedToken>(capacity: 32);
        JsonElement root = doc.RootElement;

        // Primary source: added_tokens_decoder { "id": {content, special, ...} }.
        if (root.TryGetProperty("added_tokens_decoder", out JsonElement decoder) &&
            decoder.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty prop in decoder.EnumerateObject())
            {
                if (!int.TryParse(prop.Name, out int id)) continue;
                if (prop.Value.ValueKind != JsonValueKind.Object) continue;
                if (!prop.Value.TryGetProperty("content", out JsonElement contentEl) ||
                    contentEl.ValueKind != JsonValueKind.String)
                    continue;
                bool special = prop.Value.TryGetProperty("special", out JsonElement sp) &&
                               sp.ValueKind == JsonValueKind.True;
                byId[id] = new HfAddedToken(id, contentEl.GetString()!, special);
            }
        }

        // Promote top-level bos/eos/unk/pad strings into specials when the
        // config only carries them as scalar fields (common in older repos).
        PromoteScalarSpecial(root, "bos_token", vocab, byId);
        PromoteScalarSpecial(root, "eos_token", vocab, byId);
        PromoteScalarSpecial(root, "unk_token", vocab, byId);
        PromoteScalarSpecial(root, "pad_token", vocab, byId);

        // Additional special tokens — rare in legacy, but cheap to support.
        if (root.TryGetProperty("additional_special_tokens", out JsonElement extra) &&
            extra.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement e in extra.EnumerateArray())
            {
                if (e.ValueKind != JsonValueKind.String) continue;
                string? s = e.GetString();
                if (string.IsNullOrEmpty(s)) continue;
                if (!vocab.TryGetValue(s, out int id)) continue;
                if (!byId.ContainsKey(id))
                    byId[id] = new HfAddedToken(id, s, Special: true);
            }
        }

        // Stable ordering by ID (the factory doesn't rely on order but this
        // makes the output deterministic for debugging).
        var list = new List<HfAddedToken>(byId.Values);
        list.Sort(static (a, b) => a.Id.CompareTo(b.Id));
        return list;
    }

    /// <summary>
    /// Promotes a scalar tokenizer_config field (e.g. <c>"bos_token": "&lt;s&gt;"</c>)
    /// into an <see cref="HfAddedToken"/> marked special, provided the token
    /// is present in the vocabulary and not already in <paramref name="byId"/>.
    /// Values shaped as objects (<c>{"content": ..., ...}</c>) are also accepted.
    /// </summary>
    private static void PromoteScalarSpecial(
        JsonElement root,
        string fieldName,
        IReadOnlyDictionary<string, int> vocab,
        Dictionary<int, HfAddedToken> byId)
    {
        if (!root.TryGetProperty(fieldName, out JsonElement field)) return;

        string? content = field.ValueKind switch
        {
            JsonValueKind.String => field.GetString(),
            JsonValueKind.Object when field.TryGetProperty("content", out JsonElement c) &&
                                      c.ValueKind == JsonValueKind.String => c.GetString(),
            _ => null,
        };
        if (string.IsNullOrEmpty(content)) return;
        if (!vocab.TryGetValue(content, out int id)) return;

        if (!byId.ContainsKey(id))
            byId[id] = new HfAddedToken(id, content, Special: true);
        else if (!byId[id].Special)
            byId[id] = byId[id] with { Special = true };
    }
}
