using DotLLM.Core.Configuration;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>Runs alone: the measurement is process-wide (the compute pool allocates on its own threads), so parallel tests would pollute it.</summary>
[CollectionDefinition("Inkling2Alloc", DisableParallelization = true)]
public sealed class Inkling2AllocCollection { }

/// <summary>
/// CLAUDE.md "no managed arrays for tensor data": the Inkling 2 CPU reference keeps its per-call F32 activations in native scratch, so a warm decode
/// step must allocate (almost) no managed memory. Before the fix a 32-token prefill call allocated 449 MB and a decoded token 16.0 MB.
/// </summary>
[Collection("Inkling2Alloc")]
public sealed class Inkling2AllocationTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2AllocationTests(ITestOutputHelper output) => _output = output;

    /// <summary>Budget per decoded token: the returned logits tensor object, spans and pool bookkeeping. One <c>new float[2304]</c> per layer visit is 442 KB.</summary>
    private const long DecodeBudgetBytesPerToken = 32 * 1024;

    [SkippableFact]
    public void WarmDecodeAndPrefill_AllocateNearZeroManagedBytes()
    {
        string? gguf = Inkling2RealModelTests.FindGguf();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var model = Inkling2TransformerModel.LoadFromGguf(file, cfg, ThreadingConfig.Auto, Inkling2Options.FromEnvironment());
        int[] prompt = Enumerable.Range(1000, 32).ToArray();
        int[] pos = Enumerable.Range(0, 32).ToArray();

        // Warm-up: JIT, scratch growth, kernel scratch, then a second prefill so the steady state is measured.
        for (int rep = 0; rep < 2; rep++)
        {
            model.ResetSequenceState();
            using (model.Forward(prompt, pos, -1, null, true)) { }
        }
        for (int j = 0; j < 4; j++) { using var w = model.Forward([1234], [32 + j], -1, null, true); }

        model.ResetSequenceState();
        GC.Collect(); GC.WaitForPendingFinalizers();
        long p0 = GC.GetTotalAllocatedBytes(precise: true);
        using (model.Forward(prompt, pos, -1, null, true)) { }
        long prefill = GC.GetTotalAllocatedBytes(precise: true) - p0;

        const int steps = 8;
        long d0 = GC.GetTotalAllocatedBytes(precise: true);
        for (int j = 0; j < steps; j++) { using var t = model.Forward([1234], [32 + j], -1, null, true); }
        long perToken = (GC.GetTotalAllocatedBytes(precise: true) - d0) / steps;

        _output.WriteLine($"MANAGED ALLOCATION: 32-token prefill call {prefill / 1e6:F2} MB; decode {perToken / 1e6:F3} MB/token ({perToken} B)");
        Assert.True(perToken < DecodeBudgetBytesPerToken, $"decode allocated {perToken} managed bytes per token (budget {DecodeBudgetBytesPerToken})");
        Assert.True(prefill < 1024 * 1024, $"a 32-token prefill call allocated {prefill} managed bytes");
    }
}
