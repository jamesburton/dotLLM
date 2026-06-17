using System.Text.Json;
using DotLLM.Core.Models;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Builds a <see cref="DiffusionConfig"/> for a DiffusionGemma checkpoint from
/// its on-disk metadata. The diffusion fields are split across two HuggingFace
/// files — <c>canvas_length</c> lives in <c>config.json</c> while the denoising
/// schedule (steps, entropy bound, confidence/stability thresholds, temperature
/// range) lives in <c>generation_config.json</c> — and the <b>mask token id</b>
/// lives in neither: it is resolved from the tokenizer metadata.
/// </summary>
/// <remarks>
/// <para>
/// This extractor is deliberately separate from <see cref="HfConfigExtractor"/>:
/// the autoregressive <see cref="HfConfigExtractor.Extract(JsonElement)"/> path
/// stays untouched (it never produces a non-null
/// <see cref="ModelConfig.DiffusionConfig"/>), and the diffusion wrapper loader
/// (issue #29) composes this extractor's output onto the hoisted text-tower
/// config when it lands.
/// </para>
/// <para>
/// <b>Defaults.</b> Every numeric field falls back to the verified DiffusionGemma
/// default carried on <see cref="DiffusionConfig"/> when its source file omits
/// the key, so a partial <c>generation_config.json</c> still yields the reference
/// schedule. Only <see cref="DiffusionConfig.MaskTokenId"/> has no default — it
/// must resolve from the tokenizer or loading fails loudly.
/// </para>
/// <para>
/// <b>Mask-token resolution.</b> The denoising schedule field names mirror the
/// reference <c>EntropyBoundSamplerConfig</c>; the extractor reads each key both
/// at the top level of <c>generation_config.json</c> and (defensively) nested
/// under an <c>entropy_bound_sampler_config</c> / <c>sampler_config</c> object,
/// since the reference checkpoints have shipped both layouts.
/// </para>
/// </remarks>
public static class DiffusionConfigExtractor
{
    /// <summary>
    /// Builds a <see cref="DiffusionConfig"/> from a checkpoint directory.
    /// Reads <c>generation_config.json</c> (optional — absent ⇒ all-defaults
    /// schedule), takes <c>canvas_length</c> from the already-parsed
    /// <paramref name="config"/> root, and resolves the mask token id from the
    /// tokenizer metadata in <paramref name="weightsDir"/>.
    /// </summary>
    /// <param name="weightsDir">Directory holding the checkpoint's JSON metadata.</param>
    /// <param name="config">Parsed <c>config.json</c> root (text-tower-hoisted is fine).</param>
    /// <returns>A fully populated diffusion configuration.</returns>
    /// <exception cref="InvalidDataException">
    /// The mask token id cannot be resolved from the tokenizer metadata.
    /// </exception>
    public static DiffusionConfig ExtractFromDirectory(string weightsDir, JsonElement config)
    {
        ArgumentNullException.ThrowIfNull(weightsDir);

        JsonDocument? genDoc = null;
        try
        {
            string genPath = Path.Combine(weightsDir, "generation_config.json");
            if (File.Exists(genPath))
                genDoc = JsonDocument.Parse(File.ReadAllText(genPath));

            int maskTokenId = ResolveMaskTokenId(weightsDir);

            return Extract(config, genDoc?.RootElement, maskTokenId);
        }
        finally
        {
            genDoc?.Dispose();
        }
    }

    /// <summary>
    /// Builds a <see cref="DiffusionConfig"/> from already-parsed JSON roots and
    /// an externally resolved mask token id. The pure, file-system-free core —
    /// used directly by unit tests and by the directory overload above.
    /// </summary>
    /// <param name="config">Parsed <c>config.json</c> root (supplies <c>canvas_length</c>).</param>
    /// <param name="generationConfig">
    /// Parsed <c>generation_config.json</c> root, or <see langword="null"/> when
    /// the file is absent (all denoising fields then fall back to defaults).
    /// </param>
    /// <param name="maskTokenId">
    /// Mask token id resolved by a higher layer (see
    /// <see cref="ResolveMaskTokenId(string)"/>). Must be non-negative.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="maskTokenId"/> is negative.
    /// </exception>
    public static DiffusionConfig Extract(JsonElement config, JsonElement? generationConfig, int maskTokenId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(maskTokenId);

        // canvas_length is a config.json field; default 256 when omitted.
        int canvasLength = GetInt32OrDefault(config, "canvas_length", 256);

        // Denoising schedule lives in generation_config.json. Read each key at
        // the top level, falling back to a nested sampler-config object, then to
        // the verified default. A `default` DiffusionConfig supplies the
        // canonical fallbacks so this method has a single source of truth.
        DiffusionConfig defaults = new() { MaskTokenId = maskTokenId };

        JsonElement? gen = generationConfig;
        JsonElement? sampler = gen is { } g ? FindSamplerConfig(g) : null;

        int maxSteps = GetSchedInt(gen, sampler, "max_denoising_steps", defaults.MaxDenoisingSteps);
        float entropyBound = GetSchedFloat(gen, sampler, "entropy_bound", defaults.EntropyBound);
        float confidence = GetSchedFloat(gen, sampler, "confidence_threshold", defaults.ConfidenceThreshold);
        int stability = GetSchedInt(gen, sampler, "stability_threshold", defaults.StabilityThreshold);
        float tMax = GetSchedFloat(gen, sampler, "t_max", defaults.TemperatureMax);
        float tMin = GetSchedFloat(gen, sampler, "t_min", defaults.TemperatureMin);

        return new DiffusionConfig
        {
            CanvasLength = canvasLength,
            MaxDenoisingSteps = maxSteps,
            EntropyBound = entropyBound,
            ConfidenceThreshold = confidence,
            StabilityThreshold = stability,
            TemperatureMax = tMax,
            TemperatureMin = tMin,
            MaskTokenId = maskTokenId,
        };
    }

    /// <summary>
    /// Resolves the diffusion mask token id from a checkpoint's tokenizer
    /// metadata. The id is NOT in <c>config.json</c>/<c>generation_config.json</c>,
    /// so it is recovered (in priority order) from:
    /// <list type="number">
    ///   <item><description><c>special_tokens_map.json</c> — a <c>mask_token</c> entry (string or object).</description></item>
    ///   <item><description><c>tokenizer_config.json</c> — a <c>mask_token</c> entry, cross-referenced against
    ///     its <c>added_tokens_decoder</c> id map.</description></item>
    ///   <item><description><c>tokenizer.json</c> — an <c>added_tokens</c> entry whose content is the mask token.</description></item>
    /// </list>
    /// The mask token content defaults to the HuggingFace convention
    /// <c>[MASK]</c> but is overridden by whatever <c>mask_token</c> the special
    /// tokens map / tokenizer config declares.
    /// </summary>
    /// <param name="weightsDir">Directory holding the tokenizer JSON metadata.</param>
    /// <returns>The resolved mask token id.</returns>
    /// <exception cref="InvalidDataException">
    /// No tokenizer file declares a resolvable mask token — a diffusion model
    /// cannot decode without it, so this fails loudly rather than guessing an id.
    /// </exception>
    public static int ResolveMaskTokenId(string weightsDir)
    {
        ArgumentNullException.ThrowIfNull(weightsDir);

        // Default mask-token content; may be overridden by mask_token entries.
        string maskContent = "[MASK]";

        // 1) special_tokens_map.json — authoritative for the mask token content
        //    and may carry the id directly when stored as an AddedToken object.
        string stmPath = Path.Combine(weightsDir, "special_tokens_map.json");
        if (TryReadJson(stmPath, out JsonDocument? stmDoc))
        {
            using (stmDoc)
            {
                if (TryGetMaskToken(stmDoc!.RootElement, out string? content, out int? id))
                {
                    if (content is not null) maskContent = content;
                    if (id is { } directId) return directId;
                }
            }
        }

        // 2) tokenizer_config.json — mask_token + added_tokens_decoder id map.
        string tcPath = Path.Combine(weightsDir, "tokenizer_config.json");
        if (TryReadJson(tcPath, out JsonDocument? tcDoc))
        {
            using (tcDoc)
            {
                JsonElement tcRoot = tcDoc!.RootElement;
                if (TryGetMaskToken(tcRoot, out string? content, out int? id))
                {
                    if (content is not null) maskContent = content;
                    if (id is { } directId) return directId;
                }

                // added_tokens_decoder is a { "<id>": { "content": "..." }, ... } map.
                if (TryResolveFromAddedTokensDecoder(tcRoot, maskContent, out int decId))
                    return decId;
            }
        }

        // 3) tokenizer.json — added_tokens array of { id, content }.
        string tkPath = Path.Combine(weightsDir, "tokenizer.json");
        if (TryReadJson(tkPath, out JsonDocument? tkDoc))
        {
            using (tkDoc)
            {
                if (TryResolveFromAddedTokensArray(tkDoc!.RootElement, maskContent, out int addId))
                    return addId;
            }
        }

        throw new InvalidDataException(
            $"Could not resolve the diffusion mask token id from tokenizer metadata in '{weightsDir}'. "
            + $"Looked for a '{maskContent}' / mask_token entry in special_tokens_map.json, "
            + "tokenizer_config.json (added_tokens_decoder), and tokenizer.json (added_tokens). "
            + "A diffusion model cannot decode without a mask token; provide one of these files "
            + "with a mask special token declared.");
    }

    // -------------------------------------------------------------------------
    // Schedule-field helpers
    // -------------------------------------------------------------------------

    /// <summary>
    /// Locates a nested sampler-config object inside generation_config.json.
    /// Reference DiffusionGemma checkpoints have shipped the entropy-bound
    /// schedule both flat at the top level and nested under a config object —
    /// accept either layout.
    /// </summary>
    private static JsonElement? FindSamplerConfig(JsonElement gen)
    {
        if (gen.ValueKind != JsonValueKind.Object)
            return null;
        if (gen.TryGetProperty("entropy_bound_sampler_config", out var ebsc) && ebsc.ValueKind == JsonValueKind.Object)
            return ebsc;
        if (gen.TryGetProperty("sampler_config", out var sc) && sc.ValueKind == JsonValueKind.Object)
            return sc;
        return null;
    }

    private static int GetSchedInt(JsonElement? gen, JsonElement? sampler, string key, int fallback)
    {
        if (gen is { } g && TryGetInt32(g, key, out int v)) return v;
        if (sampler is { } s && TryGetInt32(s, key, out int vs)) return vs;
        return fallback;
    }

    private static float GetSchedFloat(JsonElement? gen, JsonElement? sampler, string key, float fallback)
    {
        if (gen is { } g && TryGetSingle(g, key, out float v)) return v;
        if (sampler is { } s && TryGetSingle(s, key, out float vs)) return vs;
        return fallback;
    }

    // -------------------------------------------------------------------------
    // Mask-token resolution helpers
    // -------------------------------------------------------------------------

    /// <summary>
    /// Reads a <c>mask_token</c> entry, which HF stores either as a bare string
    /// (<c>"mask_token": "[MASK]"</c>) or as an AddedToken object
    /// (<c>{ "content": "[MASK]", "id": 4 }</c>). Returns the content and, when
    /// present, the explicit id.
    /// </summary>
    private static bool TryGetMaskToken(JsonElement root, out string? content, out int? id)
    {
        content = null;
        id = null;
        if (root.ValueKind != JsonValueKind.Object
            || !root.TryGetProperty("mask_token", out var mt))
            return false;

        switch (mt.ValueKind)
        {
            case JsonValueKind.String:
                content = mt.GetString();
                return content is not null;
            case JsonValueKind.Object:
                if (mt.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
                    content = c.GetString();
                if (mt.TryGetProperty("id", out var i) && i.ValueKind == JsonValueKind.Number
                    && i.TryGetInt32(out int idv))
                    id = idv;
                return content is not null || id is not null;
            default:
                return false;
        }
    }

    /// <summary>
    /// Resolves the id of <paramref name="maskContent"/> from a
    /// <c>tokenizer_config.json</c> <c>added_tokens_decoder</c> map of the form
    /// <c>{ "&lt;id&gt;": { "content": "[MASK]", ... }, ... }</c>.
    /// </summary>
    private static bool TryResolveFromAddedTokensDecoder(JsonElement tcRoot, string maskContent, out int id)
    {
        id = -1;
        if (!tcRoot.TryGetProperty("added_tokens_decoder", out var dec)
            || dec.ValueKind != JsonValueKind.Object)
            return false;

        foreach (JsonProperty entry in dec.EnumerateObject())
        {
            if (entry.Value.ValueKind != JsonValueKind.Object) continue;
            if (!entry.Value.TryGetProperty("content", out var c)
                || c.ValueKind != JsonValueKind.String
                || !string.Equals(c.GetString(), maskContent, StringComparison.Ordinal))
                continue;
            if (int.TryParse(entry.Name, out int parsed))
            {
                id = parsed;
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Resolves the id of <paramref name="maskContent"/> from a
    /// <c>tokenizer.json</c> <c>added_tokens</c> array of the form
    /// <c>[{ "id": 4, "content": "[MASK]" }, ...]</c>.
    /// </summary>
    private static bool TryResolveFromAddedTokensArray(JsonElement tkRoot, string maskContent, out int id)
    {
        id = -1;
        if (tkRoot.ValueKind != JsonValueKind.Object
            || !tkRoot.TryGetProperty("added_tokens", out var added)
            || added.ValueKind != JsonValueKind.Array)
            return false;

        foreach (JsonElement tok in added.EnumerateArray())
        {
            if (tok.ValueKind != JsonValueKind.Object) continue;
            if (!tok.TryGetProperty("content", out var c)
                || c.ValueKind != JsonValueKind.String
                || !string.Equals(c.GetString(), maskContent, StringComparison.Ordinal))
                continue;
            if (tok.TryGetProperty("id", out var i) && i.ValueKind == JsonValueKind.Number
                && i.TryGetInt32(out int idv))
            {
                id = idv;
                return true;
            }
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Low-level JSON helpers
    // -------------------------------------------------------------------------

    private static bool TryReadJson(string path, out JsonDocument? doc)
    {
        doc = null;
        if (!File.Exists(path))
            return false;
        doc = JsonDocument.Parse(File.ReadAllText(path));
        return true;
    }

    private static bool TryGetInt32(JsonElement root, string key, out int value)
    {
        value = 0;
        if (root.ValueKind != JsonValueKind.Object
            || !root.TryGetProperty(key, out var prop)
            || prop.ValueKind != JsonValueKind.Number)
            return false;
        return prop.TryGetInt32(out value);
    }

    private static bool TryGetSingle(JsonElement root, string key, out float value)
    {
        value = 0f;
        if (root.ValueKind != JsonValueKind.Object
            || !root.TryGetProperty(key, out var prop)
            || prop.ValueKind != JsonValueKind.Number)
            return false;
        return prop.TryGetSingle(out value);
    }

    private static int GetInt32OrDefault(JsonElement root, string key, int fallback)
        => TryGetInt32(root, key, out int v) ? v : fallback;
}
