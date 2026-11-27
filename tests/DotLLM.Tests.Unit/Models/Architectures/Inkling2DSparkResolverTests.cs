using DotLLM.Models.Architectures.DSpark;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>Sibling auto-detection of the Inkling 2 DSpark drafter (issue #2039); no model needed.</summary>
public sealed class Inkling2DSparkResolverTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dspark-resolver-" + Guid.NewGuid().ToString("N"));
    public Inkling2DSparkResolverTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch (IOException) { } }

    [Fact]
    public void Find_PicksTheLargestDsparkSibling_InTheModelDirectoryOrTheOneAbove()
    {
        string sub = Path.Combine(_dir, "quant"); Directory.CreateDirectory(sub);
        string model = Path.Combine(sub, "Inkling-2-Mini-NVFP4.gguf");
        File.WriteAllBytes(model, [1]);
        Assert.Null(Inkling2DSparkResolver.Find(model));
        File.WriteAllBytes(Path.Combine(_dir, "dspark-small.gguf"), new byte[10]);
        File.WriteAllBytes(Path.Combine(_dir, "dspark-big.gguf"), new byte[100]);
        File.WriteAllBytes(Path.Combine(_dir, "mtp-other.gguf"), new byte[1000]);   // another family's head is not a DSpark drafter
        Assert.Equal(Path.Combine(_dir, "dspark-big.gguf"), Inkling2DSparkResolver.Find(model));
        File.WriteAllBytes(Path.Combine(sub, "dspark-near.gguf"), new byte[5]);        // the model's own directory wins over the one above
        Assert.Equal(Path.Combine(sub, "dspark-near.gguf"), Inkling2DSparkResolver.Find(model));
    }

    [Fact]
    public void AGarbageFile_IsNotADsparkDrafter()
    {
        string f = Path.Combine(_dir, "dspark-garbage.gguf");
        File.WriteAllBytes(f, [0, 1, 2, 3, 4, 5, 6, 7]);
        Assert.False(Inkling2DSparkDrafter.IsDSparkFile(f));
        Assert.False(Inkling2DSparkDrafter.IsDSparkFile(Path.Combine(_dir, "missing.gguf")));
    }
}
