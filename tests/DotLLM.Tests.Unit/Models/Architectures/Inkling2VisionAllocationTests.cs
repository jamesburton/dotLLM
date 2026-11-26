using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// CLAUDE.md "no managed arrays for tensor data" for the vision tower (issue #2035; the text slice had to remove 449 MB per call of <c>new float[]</c>).
/// A warm <c>Encode</c> keeps every activation (patches, Q/K/V, scores, FFN hidden: ~20 MB at 15 tokens, hundreds of MB at the 4096-token limit) in native
/// scratch; what the managed heap sees is the result object and a few small job structs. Runs alone: the measurement is process-wide.
/// </summary>
[Collection("Inkling2Alloc")]
public sealed class Inkling2VisionAllocationTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2VisionAllocationTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData("rainbow190x110", 64 * 1024)]
    [InlineData("rainbow640", 256 * 1024)]
    public void WarmEncode_AllocatesNearZeroManagedBytes(string name, long budgetBytes)
    {
        string? mm = Inkling2VisionFixtures.FindMmproj();
        Skip.If(mm is null, "mmproj-Inkling-2-Mini-Q8_0.gguf not in the HF cache");
        string ppm = Path.Combine(Inkling2VisionFixtures.OracleDir, name, "image.ppm");
        Skip.If(!File.Exists(ppm), "vision oracle images not generated");
        var image = ImageDecoder.Load(ppm);
        using var enc = Inkling2VisionEncoder.Load(mm!, new ThreadingConfig(8));
        for (int i = 0; i < 2; i++) enc.Encode(image).Dispose();   // JIT, scratch growth, kernel scratch

        GC.Collect(); GC.WaitForPendingFinalizers();
        long before = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < 3; i++) enc.Encode(image).Dispose();
        long perCall = (GC.GetTotalAllocatedBytes(precise: true) - before) / 3;
        _output.WriteLine($"{name}: {perCall} managed bytes per warm Encode; native scratch held {enc.ScratchBytes / 1e6:F1} MB");
        Assert.True(enc.ScratchBytes > 1_000_000, "the activations should live in native scratch");
        Assert.True(perCall < budgetBytes, $"a warm Encode allocated {perCall} managed bytes (budget {budgetBytes}); the activations belong in native scratch");
    }
}
