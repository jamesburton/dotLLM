using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Hip;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

[Collection("HipGpu")]
public sealed class HipQ8KvOptInTests
{
    [Theory]
    [InlineData(null, false)]
    [InlineData("0", false)]
    [InlineData("true", false)]
    [InlineData("false", false)]
    [InlineData(" 1 ", false)]
    [InlineData("1", true)]
    public void EitherQ8Plane_RequiresExactAcknowledgement(string? optIn, bool allowed)
    {
        using var environment = new HipQ8Environment(optIn);
        Assert.Null(HipPagedKvCache.ExperimentalQ8Error(KvCacheDType.F32, KvCacheDType.F32));
        foreach (var (key, value) in new[]
        {
            (KvCacheDType.Q8_0, KvCacheDType.F32),
            (KvCacheDType.F32, KvCacheDType.Q8_0),
            (KvCacheDType.Q8_0, KvCacheDType.Q8_0)
        })
        {
            string? error = HipPagedKvCache.ExperimentalQ8Error(key, value);
            Assert.Equal(allowed, error is null);
            if (!allowed)
            {
                Assert.Contains("precision loss", error, StringComparison.Ordinal);
                Assert.Contains("DOTLLM_HIP_ALLOW_Q8_KV=1", error, StringComparison.Ordinal);
            }
        }
    }

    [Theory]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.F32)]
    [InlineData(KvCacheDType.F32, KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0)]
    public void DirectCache_RejectsUnacknowledgedQ8BeforeAnyNativeAccess(KvCacheDType key, KvCacheDType value)
    {
        using var environment = new HipQ8Environment(null);
        var error = Assert.Throws<NotSupportedException>(() =>
        {
            using var rejected = new HipPagedKvCache(null!, 1, 32, 16, 16, keyType: key, valueType: value);
        });
        Assert.Contains("DOTLLM_HIP_ALLOW_Q8_KV=1", error.Message, StringComparison.Ordinal);
    }
}
