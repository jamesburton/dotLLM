using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// GGUF type id 42 is PQ2_0 here (128-element groups, 34 B) but upstream ggml Q2_0 (64-element
/// groups, 18 B). These tests use the SAME shape and id with the two byte extents so only the
/// layout guard can tell them apart (#747).
/// </summary>
public sealed class GgufType42CollisionTests : IDisposable
{
    private const int N = 4096;
    private const int Pq2Bytes = N / 128 * 34;       // 1088
    private const int UpstreamQ2Bytes = N / 64 * 18; // 1152
    private readonly List<string> _files = [];

    public void Dispose() { foreach (var f in _files) try { File.Delete(f); } catch { } }

    private string Build(uint typeId, int bytes, int elements, bool trailingTensor)
    {
        var w = new GgufWriter().AddString("general.architecture", "llama")
            .AddTensor("w", [elements], typeId, new byte[bytes]);
        if (trailingTensor) w.AddTensor("t", [8], 0, new byte[32]);
        string path = Path.Combine(Path.GetTempPath(), $"t42-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, w.Build());
        _files.Add(path);
        return path;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Id42_WithPq2_0Extent_Loads(bool trailing)
    {
        using var f = GgufFile.Open(Build(42, Pq2Bytes, N, trailing));
        Assert.Equal(QuantizationType.PQ2_0, f.Tensors[0].QuantizationType);
    }

    [Fact]
    public void Id142_Loads() // PrismML's own id; never ambiguous
    {
        using var f = GgufFile.Open(Build(142, Pq2Bytes, N, trailingTensor: true));
        Assert.Equal(QuantizationType.PQ2_0, f.Tensors[0].QuantizationType);
    }

    [Theory]
    [InlineData(true)]  // extent measured to next tensor
    [InlineData(false)] // extent measured to end of data section
    public void Id42_WithUpstreamQ2_0Extent_LoadsAsQ2_0(bool trailing)
    {
        // #823: the extent says upstream ggml Q2_0 (64-element groups), so the tensor is re-labelled instead of mis-decoded as PQ2_0.
        using var f = GgufFile.Open(Build(42, UpstreamQ2Bytes, N, trailing));
        Assert.Equal(QuantizationType.Q2_0, f.Tensors[0].QuantizationType);
    }

    [Fact]
    public void Id42_FittingNeitherLayout_IsStillRejected()
    {
        var ex = Assert.Throws<NotSupportedException>(() => GgufFile.Open(Build(42, UpstreamQ2Bytes + 200, N, trailingTensor: true)));
        Assert.Contains("Q2_0", ex.Message);
    }

    [Fact]
    public void Id42_SingleUpstreamQ2_0Group_LoadsAsQ2_0()
    {
        // 64 elements = one upstream Q2_0 group (18 B); impossible as PQ2_0.
        using var f = GgufFile.Open(Build(42, 18, 64, trailingTensor: true));
        Assert.Equal(QuantizationType.Q2_0, f.Tensors[0].QuantizationType);
    }

    /// <summary>
    /// #823 real-file check (opt-in, <c>DOTLLM_ISTA_GGUF</c> = shard 1 of an ISTA Qwen3.8-Flash-Next GSQ-RCO IQ2_XS / IQ3_XXS set): every id-42
    /// expert down bank is upstream Q2_0 (2.25 bpw), nothing is left labelled PQ2_0, and the open succeeds.
    /// </summary>
    [SkippableFact]
    public void RealIstaFile_Id42ExpertBanks_AreQ2_0()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_ISTA_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_ISTA_GGUF not set");
        using var f = GgufFile.Open(path!);
        var q2 = f.Tensors.Where(t => t.QuantizationType == QuantizationType.Q2_0).ToList();
        Assert.NotEmpty(q2);
        Assert.Contains(q2, t => t.Name.Contains("ffn_down_exps", StringComparison.Ordinal));
        Assert.DoesNotContain(f.Tensors, t => t.QuantizationType == QuantizationType.PQ2_0);
    }
}
