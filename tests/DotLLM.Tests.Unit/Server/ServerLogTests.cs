using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>Issue #1021: <c>[dotllm]</c> console lines carry a timestamp; everything else (Spectre banners, spinners) passes through.</summary>
public sealed class ServerLogTests
{
    private static readonly DateTime Fixed = new(2026, 10, 16, 14, 12, 54, 921);

    private static (ServerLog.TimestampingWriter Writer, StringWriter Sink) Make()
    {
        var sink = new StringWriter { NewLine = "\n" };
        return (new ServerLog.TimestampingWriter(sink, () => Fixed), sink);
    }

    [Fact]
    public void DotllmLines_AreTimestamped()
    {
        var (w, sink) = Make();
        w.WriteLine("[dotllm] Loading model from x.gguf...");
        w.WriteLine("[dotllm] req#3 model=q4_k_m");

        Assert.Equal(
            "2026-10-16 14:12:54.921 [dotllm] Loading model from x.gguf...\n2026-10-16 14:12:54.921 [dotllm] req#3 model=q4_k_m\n",
            sink.ToString());
    }

    [Fact]
    public void OtherLines_AndMidLineText_AreUntouched()
    {
        var (w, sink) = Make();
        w.WriteLine("  Listening on http://localhost:8080");
        w.Write("progress ");
        w.WriteLine("[dotllm] not at line start");   // continues a line already started: no stamp mid-line

        Assert.Equal("  Listening on http://localhost:8080\nprogress [dotllm] not at line start\n", sink.ToString());
    }

    [Fact]
    public void Duration_IsTerse()
    {
        Assert.Equal("850ms", ServerLog.Duration(TimeSpan.FromMilliseconds(850)));
        Assert.Equal("12.3s", ServerLog.Duration(TimeSpan.FromSeconds(12.34)));
        Assert.Equal("4m05s", ServerLog.Duration(TimeSpan.FromSeconds(245)));
    }
}
