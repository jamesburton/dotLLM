using System.Text;
using DotLLM.Engine.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

/// <summary>
/// Guards the issue #506 detector: a CRLF corpus must be reported, an LF corpus must not be.
/// </summary>
/// <remarks>
/// The defect this protects against was silent for months — dotLLM and a Windows llama.cpp build
/// tokenized different text from the same file — so the test asserts the warning *fires* and that
/// it names the issue, not merely that a count is non-zero.
/// </remarks>
public sealed class CorpusLineEndingsTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    private string WriteBytes(string content)
    {
        string path = Path.Combine(Path.GetTempPath(), $"dotllm-506-{Guid.NewGuid():N}.txt");
        // Byte-level so no writer setting can introduce or remove a CR behind the test's back.
        File.WriteAllBytes(path, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false).GetBytes(content));
        _tempFiles.Add(path);
        return path;
    }

    [Fact]
    public void CrlfCorpus_IsDetectedAndWarned()
    {
        string path = WriteBytes("the quick brown fox\r\njumped over\r\nthe lazy dog\r\n");

        long crs = CorpusLineEndings.CountCarriageReturns(path);
        Assert.Equal(3, crs);

        string? warning = CorpusLineEndings.DescribeMismatchRisk(path, crs);
        Assert.NotNull(warning);
        Assert.Contains("#506", warning);
        Assert.Contains("CRLF", warning);
        Assert.Contains("text mode", warning);
    }

    [Fact]
    public void LfCorpus_ProducesNoWarning()
    {
        string path = WriteBytes("the quick brown fox\njumped over\nthe lazy dog\n");

        long crs = CorpusLineEndings.CountCarriageReturns(path);
        Assert.Equal(0, crs);
        Assert.Null(CorpusLineEndings.DescribeMismatchRisk(path, crs));
    }

    [Fact]
    public void SingleCarriageReturn_StillWarns()
    {
        // One CR in a megabyte still means the two engines tokenize different text, so the
        // threshold is "any", not "many".
        string path = WriteBytes(new string('a', 300_000) + "\r\n" + new string('b', 300_000));

        long crs = CorpusLineEndings.CountCarriageReturns(path);
        Assert.Equal(1, crs);
        Assert.NotNull(CorpusLineEndings.DescribeMismatchRisk(path, crs));
    }

    [Fact]
    public void CountIsExactAcrossBufferBoundaries()
    {
        // 64 KiB scan buffer: a CR landing on the seam must not be double-counted or dropped.
        var sb = new StringBuilder();
        for (int i = 0; i < 20_000; i++) sb.Append("line\r\n");
        string path = WriteBytes(sb.ToString());

        Assert.Equal(20_000, CorpusLineEndings.CountCarriageReturns(path));
    }

    [Fact]
    public void SummarizeForReport_DistinguishesTheThreeStates()
    {
        Assert.Equal("LF", CorpusLineEndings.SummarizeForReport(0, normalized: false));
        Assert.Contains("#506", CorpusLineEndings.SummarizeForReport(12, normalized: false));
        Assert.Contains("normalized", CorpusLineEndings.SummarizeForReport(12, normalized: true));
    }

    public void Dispose()
    {
        foreach (string f in _tempFiles)
        {
            try { File.Delete(f); } catch (IOException) { /* best effort */ }
        }
    }
}

/// <summary>Guards the opt-in <c>--normalize-line-endings</c> decorator.</summary>
public sealed class CrlfNormalizingTextReaderTests
{
    private static string ReadAll(string input, int bufferSize)
    {
        using var reader = new CrlfNormalizingTextReader(new StringReader(input));
        var sb = new StringBuilder();
        char[] buffer = new char[bufferSize];
        int read;
        while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
            sb.Append(buffer, 0, read);
        return sb.ToString();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(64)]
    public void CollapsesCrlfAtEveryBufferSize(int bufferSize)
    {
        Assert.Equal("a\nb\nc\n", ReadAll("a\r\nb\r\nc\r\n", bufferSize));
    }

    [Fact]
    public void CrlfSplitAcrossReadsStillCollapses()
    {
        // The pending-CR carry is the whole point: with a 2-char buffer "a\r" and "\nb" arrive in
        // separate reads, and a naive implementation emits the CR.
        Assert.Equal("a\nb", ReadAll("a\r\nb", 2));
    }

    [Fact]
    public void LoneCarriageReturnSurvives()
    {
        // MSVC text mode only translates CRLF; a bare CR is data.
        Assert.Equal("a\rb", ReadAll("a\rb", 8));
        Assert.Equal("a\r", ReadAll("a\r", 8));
        Assert.Equal("a\r\r\nb", ReadAll("a\r\r\r\nb", 8));
    }

    [Fact]
    public void AllCrInputDoesNotReportEarlyEof()
    {
        // A read whose every character is dropped must not return 0 — that reads as end-of-input
        // and would truncate the corpus.
        Assert.Equal("\n\n\n", ReadAll("\r\n\r\n\r\n", 4));
    }

    [Fact]
    public void LfOnlyInputIsUnchanged()
    {
        Assert.Equal("a\nb\nc", ReadAll("a\nb\nc", 4));
    }
}
