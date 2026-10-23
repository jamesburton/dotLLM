namespace DotLLM.Core.Configuration;

/// <summary>
/// Pure selection logic for the quantized K/V storage owned by the CUDA hybrid-dense model
/// (<c>CudaQwen3HybridDenseTransformerModel</c>, issues #956/#967): flag/env parsing, precedence,
/// validation and the byte math shared by the model's VRAM ledger and the CLI summary line.
/// Kept in Core and free of CUDA types so it is unit-testable without a GPU.
/// </summary>
/// <remarks>
/// <para>Convention: <see cref="KvCacheDType.F32"/> means "unquantized", which on the GPU is an F16
/// cache (2 bytes/element). The CLI accepts <c>f16</c>/<c>fp16</c> as aliases.</para>
/// <para>Precedence, per side (key and value independently):
/// explicit CLI flag, then the <see cref="EnvVar"/> environment variable, then F16.</para>
/// </remarks>
public static class HybridKvStorage
{
    /// <summary>Environment variable selecting the storage format: f16, q8_0, q4_0, k8v4, k4v8.</summary>
    public const string EnvVar = "DOTLLM_HYBRID_KV";

    /// <summary>Elements per quantization block (Q8_0 and Q4_0 KV blocks are both 32 wide).</summary>
    public const int BlockElements = 32;

    /// <summary>
    /// Parses <see cref="EnvVar"/>'s value. Null or empty yields <c>(null, null)</c> (unspecified);
    /// <c>f16</c> yields explicit unquantized on both sides.
    /// </summary>
    /// <exception cref="InvalidOperationException">The value is not one of the supported names.</exception>
    public static (KvCacheDType? Key, KvCacheDType? Value) ParseEnv(string? raw)
    {
        string? v = raw?.Trim().ToLowerInvariant();
        return v switch
        {
            null or "" => (null, null),
            "f16" or "f32" => (KvCacheDType.F32, KvCacheDType.F32),
            "q8_0" or "q8" => (KvCacheDType.Q8_0, KvCacheDType.Q8_0),
            "q4_0" or "q4" => (KvCacheDType.Q4_0, KvCacheDType.Q4_0),
            "k8v4" => (KvCacheDType.Q8_0, KvCacheDType.Q4_0),
            "k4v8" => (KvCacheDType.Q4_0, KvCacheDType.Q8_0),
            _ => throw new InvalidOperationException(
                $"{EnvVar}='{v}' is not one of f16, q8_0, q4_0, k8v4, k4v8."),
        };
    }

    /// <summary>Parses one per-side CLI value (<c>--kv-quant-k</c>, <c>--cache-type-k</c>, ...). Null/empty is unspecified.</summary>
    /// <exception cref="ArgumentException">The value is not a known KV dtype.</exception>
    public static KvCacheDType? ParseFlag(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : KvCacheConfig.ParseDType(value.Trim());

    /// <summary>
    /// Parses a server option string whose default is <c>"f32"</c>: the default is
    /// indistinguishable from "not given", so <c>f32</c> is treated as unspecified (use
    /// <c>f16</c> to force an unquantized cache over the environment variable).
    /// </summary>
    public static KvCacheDType? FromServerOption(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return null;
        string v = value.Trim().ToLowerInvariant();
        return v == "f32" ? null : ParseFlag(v);
    }

    /// <summary>Resolves the effective per-side storage: explicit flag, else environment variable, else F16.</summary>
    public static (KvCacheDType Key, KvCacheDType Value) Resolve(
        KvCacheDType? flagKey, KvCacheDType? flagValue, string? envValue)
    {
        var env = ParseEnv(envValue);
        return (flagKey ?? env.Key ?? KvCacheDType.F32, flagValue ?? env.Value ?? KvCacheDType.F32);
    }

    /// <summary>True for the dtypes the hybrid-dense model can store: F32 (= F16), Q8_0, Q4_0.</summary>
    public static bool IsSupported(KvCacheDType d) => d is KvCacheDType.F32 or KvCacheDType.Q8_0 or KvCacheDType.Q4_0;

    /// <summary>True for the block-quantized dtypes.</summary>
    public static bool IsQuantized(KvCacheDType d) => d is KvCacheDType.Q8_0 or KvCacheDType.Q4_0;

    /// <summary>
    /// Returns a user-facing error when the (key, value) pair cannot be stored for a model whose
    /// per-token K (or V) row has <paramref name="kvElemsPerToken"/> = numKvHeads * headDim elements;
    /// null when valid.
    /// </summary>
    public static string? Validate(KvCacheDType key, KvCacheDType value, int kvElemsPerToken)
    {
        if (!IsSupported(key))
            return $"KV key type {key} is not supported by the CUDA hybrid-dense model; use f16, q8_0 or q4_0.";
        if (!IsSupported(value))
            return $"KV value type {value} is not supported by the CUDA hybrid-dense model; use f16, q8_0 or q4_0.";
        if ((IsQuantized(key) || IsQuantized(value)) && kvElemsPerToken % BlockElements != 0)
            return $"Quantized KV storage needs numKvHeads*headDim ({kvElemsPerToken}) to be a multiple of {BlockElements}.";
        return null;
    }

    /// <summary>Bytes of one stored K or V row of <paramref name="kvElems"/> elements.</summary>
    public static long RowBytes(KvCacheDType d, int kvElems) => d switch
    {
        KvCacheDType.Q8_0 => (long)kvElems / BlockElements * 34,
        KvCacheDType.Q4_0 => (long)kvElems / BlockElements * 18,
        _ => (long)kvElems * sizeof(ushort),
    };

    /// <summary>Total persistent K+V bytes for <paramref name="attentionSlots"/> full-attention layers at <paramref name="maxSeqLen"/> tokens.</summary>
    public static long TotalBytes(int attentionSlots, long maxSeqLen, int kvElems, KvCacheDType key, KvCacheDType value) =>
        (long)attentionSlots * maxSeqLen * (RowBytes(key, kvElems) + RowBytes(value, kvElems));

    /// <summary>Short label: <c>f16</c>, <c>q8_0</c>, <c>q4_0</c> when both sides match, else <c>k8v4</c>-style.</summary>
    public static string Label(KvCacheDType key, KvCacheDType value)
    {
        static string One(KvCacheDType d) => d switch
        {
            KvCacheDType.Q8_0 => "q8_0",
            KvCacheDType.Q4_0 => "q4_0",
            KvCacheDType.F32 => "f16",
            _ => d.ToString().ToLowerInvariant(),
        };
        static string Bits(KvCacheDType d) => d switch { KvCacheDType.Q8_0 => "8", KvCacheDType.Q4_0 => "4", _ => "16" };
        return key == value ? One(key) : $"k{Bits(key)}v{Bits(value)}";
    }
}
