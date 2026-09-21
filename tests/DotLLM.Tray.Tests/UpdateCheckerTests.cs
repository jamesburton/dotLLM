using DotLLM.Tray.Updates;
using Xunit;

namespace DotLLM.Tray.Tests;

/// <summary>
/// Covers the opt-in update check and, mostly, the version ordering it rests on.
/// </summary>
/// <remarks>
/// <b>Discrimination (#417), by substitution.</b> The interesting failure mode here is not a wrong
/// assertion but a wrong comparator, so the negative control is the comparator a naive
/// implementation would use. <see cref="NaiveOrdinalComparison_IsWrong_WhichIsWhyReleaseVersionExists"/>
/// demonstrates on the project's own MinVer-shaped strings that ordinal string comparison and
/// <see cref="Version"/> both get the answer wrong, and that
/// <see cref="ReleaseVersion"/> gets it right — a test that would pass against a broken
/// implementation is worthless, and this one names the broken implementations explicitly.
/// </remarks>
public sealed class UpdateCheckerTests
{
    [Theory]
    // Plain ordering.
    [InlineData("1.0.0", "1.0.1", -1)]
    [InlineData("0.9.0", "0.10.0", -1)]
    [InlineData("2.0.0", "1.99.99", 1)]
    // A leading v, as every MinVer tag carries.
    [InlineData("v1.2.3", "1.2.3", 0)]
    // Build metadata is excluded from precedence — AssemblyInformationalVersion carries it.
    [InlineData("1.2.3+abc1234", "1.2.3", 0)]
    [InlineData("1.2.3+abc1234", "1.2.3+zzz9999", 0)]
    // A prerelease precedes the release it leads to. This is the one a naive comparator inverts.
    [InlineData("1.0.0-preview.0.1", "1.0.0", -1)]
    [InlineData("1.0.0", "1.0.0-preview.0.1", 1)]
    // Numeric prerelease identifiers compare numerically, not lexically: 2 < 10.
    [InlineData("0.4.0-preview.0.2", "0.4.0-preview.0.10", -1)]
    // ...which ordinal string comparison gets backwards.
    [InlineData("0.4.0-preview.0.10", "0.4.0-preview.0.2", 1)]
    // A numeric identifier ranks below an alphanumeric one.
    [InlineData("1.0.0-1", "1.0.0-alpha", -1)]
    // A longer identifier list is the later prerelease when the shared prefix matches.
    [InlineData("1.0.0-preview", "1.0.0-preview.1", -1)]
    public void ReleaseVersion_Orders(string left, string right, int expected)
    {
        var a = ReleaseVersion.Parse(left);
        var b = ReleaseVersion.Parse(right);
        Assert.NotNull(a);
        Assert.NotNull(b);
        Assert.Equal(expected, Math.Sign(a!.CompareTo(b)));
    }

    [Fact]
    public void NaiveOrdinalComparison_IsWrong_WhichIsWhyReleaseVersionExists()
    {
        // Negative control. If someone "simplifies" ReleaseVersion into one of these, the theory
        // above goes red — and this test documents exactly what breaks.
        const string Prerelease = "0.4.0-preview.0.2";
        const string Release = "0.4.0";

        // Ordinal: "0.4.0-preview.0.2" > "0.4.0" because '-' > end-of-string. Backwards.
        Assert.True(string.CompareOrdinal(Prerelease, Release) > 0);

        // System.Version cannot even parse it.
        Assert.False(Version.TryParse(Prerelease, out _));

        // ReleaseVersion gets it right.
        Assert.True(ReleaseVersion.Parse(Prerelease)!.CompareTo(ReleaseVersion.Parse(Release)) < 0);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("not-a-version")]
    [InlineData("nightly")]
    public void ReleaseVersion_ReturnsNullForNonVersions(string value) =>
        Assert.Null(ReleaseVersion.Parse(value));

    [Fact]
    public void ReleaseVersion_ParsesAShortVersion()
    {
        var version = ReleaseVersion.Parse("v1");
        Assert.NotNull(version);
        Assert.Equal(1, version!.Major);
        Assert.Equal(0, version.Minor);
        Assert.Equal(0, version.Patch);
    }

    private const string Feed = """
        [
          {"tag_name":"v0.3.0","name":"0.3.0","body":"Older.","prerelease":false,"draft":false,"html_url":"https://example.invalid/0.3.0"},
          {"tag_name":"v0.5.0","name":"0.5.0","body":"## What's new\n- Tray app","prerelease":false,"draft":false,"html_url":"https://example.invalid/0.5.0"},
          {"tag_name":"v0.4.0","name":"0.4.0","body":"Middle.","prerelease":false,"draft":false,"html_url":"https://example.invalid/0.4.0"}
        ]
        """;

    private static UpdateChecker Checker(string json, out HttpClient http)
    {
        http = StubHttpMessageHandler.Client(_ => StubHttpMessageHandler.Json(json), out _);
        return new UpdateChecker(http, "http://localhost:18080/releases");
    }

    [Fact]
    public async Task Check_FindsTheNewestRelease_NotTheFirstListed()
    {
        // GitHub returns newest-first in practice, but the tray must not depend on feed order.
        var checker = Checker(Feed, out var http);
        using (http)
        {
            var result = await checker.CheckAsync("0.4.0", ct: CancellationToken.None);

            Assert.True(result.UpdateAvailable);
            Assert.Equal("v0.5.0", result.LatestVersion);
            Assert.Contains("Tray app", result.Changelog!, StringComparison.Ordinal);
            Assert.Equal("https://example.invalid/0.5.0", result.ReleaseUrl);
        }
    }

    [Fact]
    public async Task Check_ReportsNoUpdateWhenAlreadyCurrent()
    {
        var checker = Checker(Feed, out var http);
        using (http)
        {
            var result = await checker.CheckAsync("0.5.0", ct: CancellationToken.None);
            Assert.False(result.UpdateAvailable);
            Assert.Null(result.Changelog);
        }
    }

    [Fact]
    public async Task Check_ReportsNoUpdateWhenAheadOfTheFeed()
    {
        var checker = Checker(Feed, out var http);
        using (http)
        {
            Assert.False((await checker.CheckAsync("0.6.0", ct: CancellationToken.None)).UpdateAvailable);
        }
    }

    [Fact]
    public async Task Check_IgnoresDraftsAndPrereleasesByDefault()
    {
        const string WithDraftAndPrerelease = """
            [
              {"tag_name":"v0.6.0","body":"Draft.","prerelease":false,"draft":true,"html_url":"d"},
              {"tag_name":"v0.7.0-preview.0.1","body":"Preview.","prerelease":true,"draft":false,"html_url":"p"},
              {"tag_name":"v0.5.0","body":"Stable.","prerelease":false,"draft":false,"html_url":"s"}
            ]
            """;

        var checker = Checker(WithDraftAndPrerelease, out var http);
        using (http)
        {
            var result = await checker.CheckAsync("0.4.0", ct: CancellationToken.None);
            Assert.Equal("v0.5.0", result.LatestVersion);
        }
    }

    [Fact]
    public async Task Check_OffersPrereleasesToAUserAlreadyOnOne()
    {
        // Otherwise a preview user is told they are up to date forever, because every newer
        // build is also a preview.
        const string PreviewFeed = """
            [
              {"tag_name":"v0.7.0-preview.0.5","body":"Newer preview.","prerelease":true,"draft":false,"html_url":"p5"},
              {"tag_name":"v0.7.0-preview.0.1","body":"Older preview.","prerelease":true,"draft":false,"html_url":"p1"}
            ]
            """;

        var checker = Checker(PreviewFeed, out var http);
        using (http)
        {
            var result = await checker.CheckAsync("0.7.0-preview.0.1", ct: CancellationToken.None);
            Assert.True(result.UpdateAvailable);
            Assert.Equal("v0.7.0-preview.0.5", result.LatestVersion);
        }
    }

    [Fact]
    public async Task Check_TreatsAnUnreachableFeedAsNoUpdate()
    {
        // Never a fabricated update: prompting a reinstall because GitHub was down is worse than
        // saying nothing.
        using var http = new HttpClient(
            new FailingHandler(), disposeHandler: true) { BaseAddress = new Uri("http://localhost:18080/") };

        var result = await new UpdateChecker(http, "http://localhost:18080/releases")
            .CheckAsync("0.4.0", ct: CancellationToken.None);

        Assert.False(result.UpdateAvailable);
        Assert.Null(result.ReleaseUrl);
    }

    [Fact]
    public async Task Check_TreatsAMalformedFeedAsNoUpdate()
    {
        var checker = Checker("{not json", out var http);
        using (http)
        {
            Assert.False((await checker.CheckAsync("0.4.0", ct: CancellationToken.None)).UpdateAvailable);
        }
    }

    [Fact]
    public void CurrentVersion_ReadsTheInformationalVersion()
    {
        // MinVer stamps AssemblyInformationalVersion; the assembly version is truncated to
        // major.minor.patch.0 and loses the prerelease suffix entirely.
        var version = UpdateChecker.CurrentVersion(typeof(UpdateChecker).Assembly);
        Assert.False(string.IsNullOrWhiteSpace(version));
        Assert.NotNull(ReleaseVersion.Parse(version));
    }

    private sealed class FailingHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromException<HttpResponseMessage>(new HttpRequestException("unreachable"));
    }
}
