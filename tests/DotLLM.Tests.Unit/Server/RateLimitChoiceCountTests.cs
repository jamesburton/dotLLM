using System.Text;
using System.Threading.Tasks;
using DotLLM.Server.RateLimiting;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #460 interaction with #457. The token reservation is taken by the middleware from the raw
/// request body, <i>before</i> the endpoint runs — and <c>TrueUpTokens</c> ignores an actual that
/// exceeds it ("the reservation is the cap"). So the moment <c>n</c> became a real multiplier on
/// the work done, an estimate sized for one choice turned <c>n</c> into a straight TPM bypass:
/// <c>n: 8</c> would run eight times the tokens against a budget reserved for one.
/// </summary>
/// <remarks>
/// This is the same shape of defect as #459's: making a long-dead field live creates interactions
/// that nothing tested, because nothing could have. Rate limiting had only just become reachable
/// at all (#457), so there was no prior run in which this could have been noticed.
/// </remarks>
public sealed class RateLimitChoiceCountTests
{
    private static readonly RateLimitConfig Config = new() { EstimatedCompletionTokensFallback = 256 };

    private static DefaultHttpContext PostWithBody(string body)
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Method = HttpMethods.Post;
        ctx.Request.Path = "/v1/chat/completions";
        ctx.Request.Body = new System.IO.MemoryStream(Encoding.UTF8.GetBytes(body));
        return ctx;
    }

    /// <summary>
    /// The reservation scales with <c>n</c>. Against the pre-fix estimator every one of these
    /// returned the single-choice figure.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(8)]
    public async Task ReservationScalesWithN(int n)
    {
        string prompt = """{"messages":[{"role":"user","content":"hello there"}],"max_tokens":100""";

        int single = await RateLimitMiddleware.EstimateTotalTokensAsync(
            PostWithBody(prompt + "}"), Config);
        int multi = await RateLimitMiddleware.EstimateTotalTokensAsync(
            PostWithBody(prompt + $",\"n\":{n}}}"), Config);

        // Not an exact multiple: `n` lengthens the body, and the prompt estimate is a char count.
        // The load-bearing property is that it scales, not that it scales exactly.
        Assert.True(multi >= single * n,
            $"n={n} reserved {multi} but a single choice reserves {single}; an under-reservation is "
            + "an un-metered request, because an actual above the reservation is never charged.");
    }

    /// <summary>
    /// <c>n: 1</c> reserves exactly what no <c>n</c> at all reserves. Compared against a body
    /// carrying an ignored key of the same length, because the prompt half of the estimate is a
    /// raw character count — comparing against a shorter body would measure the JSON, not the
    /// multiplier.
    /// </summary>
    [Fact]
    public async Task NAbsent_ReservesAsOneChoice()
    {
        int ignoredKey = await RateLimitMiddleware.EstimateTotalTokensAsync(
            PostWithBody("""{"messages":[{"role":"user","content":"hi"}],"max_tokens":50,"z":1}"""), Config);
        int withOne = await RateLimitMiddleware.EstimateTotalTokensAsync(
            PostWithBody("""{"messages":[{"role":"user","content":"hi"}],"max_tokens":50,"n":1}"""), Config);

        Assert.Equal(ignoredKey, withOne);
    }

    /// <summary>
    /// The multiplier is clamped rather than trusted. This estimator runs before the endpoint
    /// validator that rejects an out-of-range <c>n</c>, so a hostile body must not be able to
    /// reserve an absurd number of permits or overflow the arithmetic on the way.
    /// </summary>
    [Theory]
    [InlineData(1000)]
    [InlineData(int.MaxValue)]
    [InlineData(-4)]
    [InlineData(0)]
    public async Task OutOfRangeN_IsClampedNotTrusted(long n)
    {
        int estimate = await RateLimitMiddleware.EstimateTotalTokensAsync(
            PostWithBody($$"""{"messages":[{"role":"user","content":"hi"}],"max_tokens":50,"n":{{n}}}"""),
            Config);

        Assert.InRange(estimate, 1, 50 * 8 + 4096);
    }
}
