namespace DotLLM.Core.Configuration;

/// <summary>
/// Data types supported for KV-cache storage. Distinct from <see cref="QuantizationType"/>
/// because KV-cache only supports a subset of quantization formats.
/// </summary>
public enum KvCacheDType
{
    /// <summary>Full precision: FP32 on CPU, FP16 on GPU.</summary>
    F32 = 0,

    /// <summary>Q8_0: 34 bytes per 32 elements (Half scale + 32 int8 values).</summary>
    Q8_0 = 1,

    /// <summary>Q4_0: 18 bytes per 32 elements (Half scale + 16 packed nibble bytes).</summary>
    Q4_0 = 2,

    /// <summary>
    /// TurboQuant (arXiv:2504.19874): data-oblivious per-head rotation + Lloyd–Max scalar
    /// quant. Per-vector storage (<c>ceil(headDim*bits/8)</c> code bytes + an fp32 norm per
    /// head), bit-width carried by <c>KvCacheConfig.TurboQuantBits</c>. Applies to both keys
    /// and values together.
    /// </summary>
    TurboQuant = 3
}
