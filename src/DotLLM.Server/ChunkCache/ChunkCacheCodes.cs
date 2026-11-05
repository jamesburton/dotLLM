namespace DotLLM.Server.ChunkCache;

/// <summary>
/// The <c>chunk_cache</c> wire convention's warning and error codes (spec §5.8). Strict-mode errors are
/// <c>400 invalid_request_error</c> with one of these as <c>code</c>, except <see cref="StoreUnavailable"/>
/// (<c>503 api_error</c>). No code maps to 422.
/// </summary>
internal static class ChunkCacheCodes
{
    // Errors when strict, warnings otherwise.

    /// <summary>The convention version (<c>v</c>) or a v1-only field value is not supported.</summary>
    public const string VersionUnsupported = "chunk_version_unsupported";

    /// <summary>The server's hash of the decoded span text differs from the annotation's <c>hash</c>.</summary>
    public const string HashMismatch = "chunk_hash_mismatch";

    /// <summary><c>at</c> out of range, overlapping, off a UTF-8 boundary, on a non-text part, or a malformed annotation.</summary>
    public const string SpanInvalid = "chunk_span_invalid";

    /// <summary>Inline and manifest placements in one request; it is served as unannotated.</summary>
    public const string ManifestConflict = "chunk_manifest_conflict";

    /// <summary>The chat template altered the span's text (Task 11).</summary>
    public const string TemplateAltered = "chunk_template_altered";

    /// <summary>Reserved (spec §5.3 item 6): chunk text is tokenised like plain content, so v1 never emits it.</summary>
    public const string SpecialTokens = "chunk_special_tokens";

    /// <summary>The model or backend cannot assemble chunks (recurrent / hybrid); every policy degrades to <c>exact</c>.</summary>
    public const string ModelUnsupported = "chunk_model_unsupported";

    /// <summary>The requested policy (or selector) is not available here; <c>exact</c> applies.</summary>
    public const string PolicyUnavailable = "chunk_policy_unavailable";

    /// <summary>More annotations than <c>--chunk-cache-max-chunks</c>.</summary>
    public const string TooMany = "chunk_too_many";

    /// <summary>The chunk store is missing, over budget or locked. Strict: <c>503 api_error</c>.</summary>
    public const string StoreUnavailable = "chunk_store_unavailable";

    /// <summary>
    /// The body named a <c>chunk_cache.namespace</c> other than the caller's own. The namespace is always server-assigned
    /// (API key, else the default), so the body value is ignored.
    /// </summary>
    public const string NamespaceIgnored = "chunk_namespace_ignored";

    /// <summary>
    /// A structurally malformed annotation (ruling 23): a wrong JSON type, a missing required field, an unknown
    /// <c>at</c> shape, or a string System.Text.Json cannot decode (a lone-surrogate escape).
    /// </summary>
    public const string AnnotationMalformed = "chunk_annotation_malformed";

    /// <summary>
    /// The named <c>chunker</c> is not text-first (spec §5.3 item 4); the annotation is dropped (ruling 26: a warning,
    /// 400 when strict).
    /// </summary>
    public const string NotWireEligible = "chunk_not_wire_eligible";

    // Warnings only.

    /// <summary>The span is shorter than <c>min_chunk_tokens</c> and is treated as plain text (Task 13).</summary>
    public const string IgnoredTooShort = "chunk_ignored_too_short";

    /// <summary>Encode admission refused (spec §5.10).</summary>
    public const string EncodeSkipped = "chunk_encode_skipped";

    /// <summary>A background encode was queued (spec §5.10).</summary>
    public const string EncodeQueued = "chunk_encode_queued";

    /// <summary>
    /// The server's token-verification budget ran out before this span could be checked (ruling 49): the span is ignored,
    /// never a strict-mode error.
    /// </summary>
    public const string VerificationSkipped = "chunk_verification_skipped";

    /// <summary>
    /// True for codes that never become a strict-mode error (spec §5.8 "warnings only").
    /// </summary>
    public static bool IsWarningOnly(string code) =>
        code is IgnoredTooShort or EncodeSkipped or EncodeQueued or VerificationSkipped;

    /// <summary>
    /// The strict-mode error for a code: <c>503 api_error</c> for <see cref="StoreUnavailable"/>, else
    /// <c>400 invalid_request_error</c>.
    /// </summary>
    public static ChunkCacheError ErrorFor(string code, string message) =>
        code == StoreUnavailable
            ? new ChunkCacheError(503, "api_error", code, message)
            : new ChunkCacheError(400, "invalid_request_error", code, message);
}
