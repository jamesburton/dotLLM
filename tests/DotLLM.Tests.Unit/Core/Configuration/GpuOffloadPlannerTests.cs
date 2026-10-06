using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Configuration;

/// <summary>
/// Issue #729. A partial <c>--gpu-layers</c> request on an architecture whose layers are not dense
/// Llama-style (Nemotron-H, ...) used to reach <c>HybridTransformerModel.LoadFromGguf</c> and die
/// with "does not use the dense Llama-style GGUF tensor naming". The planner is the single predicate.
/// </summary>
public sealed class GpuOffloadPlannerTests
{
    private static ModelConfig ConfigFor(Architecture arch) => new()
    {
        Architecture = arch, VocabSize = 32, HiddenSize = 8, IntermediateSize = 16, NumLayers = 1,
        NumAttentionHeads = 2, NumKvHeads = 2, HeadDim = 4, MaxSequenceLength = 16, NormEpsilon = 1e-5f,
    };

    [Theory]
    [InlineData(Architecture.NemotronH)]
    [InlineData(Architecture.NemotronHMoe)]
    [InlineData(Architecture.Qwen3MoeHybrid)]
    [InlineData(Architecture.Mamba3)]
    public void UnsupportedArchitecture_PartialRequest_IsNotRoutedToTheSplitPath(Architecture arch)
    {
        var plan = GpuOffloadPlanner.Plan(arch, requestedGpuLayers: 10, numLayers: 52);

        Assert.False(GpuOffloadPlanner.SupportsPartialOffload(arch));
        Assert.Equal(GpuOffloadMode.FullGpuOrFail, plan.Mode);   // never Partial
        Assert.Equal(52, plan.GpuLayers);
        Assert.NotNull(plan.Warning);
        Assert.Contains(arch.ToString(), plan.Warning, StringComparison.Ordinal);
        Assert.Contains("not supported", plan.Warning, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Architecture.Llama)]
    [InlineData(Architecture.Qwen)]
    [InlineData(Architecture.Qwen3HybridDense)] // has its own split loader (#291)
    public void SupportedArchitecture_PartialRequest_IsPartialWithoutWarning(Architecture arch)
    {
        var plan = GpuOffloadPlanner.Plan(arch, 10, 52);

        Assert.Equal(GpuOffloadMode.Partial, plan.Mode);
        Assert.Equal(10, plan.GpuLayers);
        Assert.Null(plan.Warning);
    }

    [Theory]
    [InlineData(Architecture.NemotronH)]
    [InlineData(Architecture.Llama)]
    public void ZeroAndFullRequests_AreNeverWarnedAndNeverPartial(Architecture arch)
    {
        Assert.Equal(new GpuOffloadPlan(GpuOffloadMode.Cpu, 0, null), GpuOffloadPlanner.Plan(arch, 0, 52));
        Assert.Equal(new GpuOffloadPlan(GpuOffloadMode.FullGpu, 52, null), GpuOffloadPlanner.Plan(arch, 52, 52));
        // Out-of-range requests clamp, as the callers do.
        Assert.Equal(GpuOffloadMode.FullGpu, GpuOffloadPlanner.Plan(arch, 999, 52).Mode);
        Assert.Equal(GpuOffloadMode.Cpu, GpuOffloadPlanner.Plan(arch, -3, 52).Mode);
    }

    /// <summary>
    /// Drift guard: any architecture the dense loader rejects as needing a dedicated loader cannot be
    /// split by <c>HybridTransformerModel</c>, so the planner must say "unsupported" for it -- unless it
    /// has its own split loader (only Qwen3HybridDense). A new SSM-style architecture added to one list
    /// but not the other fails here instead of at a user's model load.
    /// </summary>
    [Fact]
    public void Predicate_AgreesWithTheDenseLoaderGuard_ForEveryArchitecture()
    {
        foreach (var arch in Enum.GetValues<Architecture>())
        {
            bool needsDedicated;
            try { TransformerWeights.ThrowIfArchitectureNeedsDedicatedLoader(ConfigFor(arch)); needsDedicated = false; }
            catch (NotSupportedException) { needsDedicated = true; }

            bool hasOwnSplitLoader = arch == Architecture.Qwen3HybridDense;
            Assert.True(
                GpuOffloadPlanner.SupportsPartialOffload(arch) == (!needsDedicated || hasOwnSplitLoader),
                $"{arch}: needsDedicatedLoader={needsDedicated}, planner says {GpuOffloadPlanner.SupportsPartialOffload(arch)}");
        }
    }

    [Fact]
    public void UnsatisfiableMessage_IsActionable_AndNeverOffersASilentCpuRun()
    {
        string m = GpuOffloadPlanner.BuildUnsatisfiableMessage(Architecture.NemotronH, 10, 52,
            modelBytes: 24L << 30, gpuTotalBytes: 12L << 30, gpuFreeBytes: 11L << 30, cause: "out of memory");

        Assert.Contains("10/52", m, StringComparison.Ordinal);                 // what was requested
        Assert.Contains("cannot be split", m, StringComparison.Ordinal);        // why
        Assert.Contains("24.0 GiB needed, 11.0 GiB free of 12.0 GiB", m, StringComparison.Ordinal); // needed vs free
        Assert.Contains("--device cpu", m, StringComparison.Ordinal);           // explicit opt-in
        Assert.Contains("Nothing was run on the CPU", m, StringComparison.Ordinal);
        Assert.Contains(GpuOffloadPlanner.PriorityIssueUrl, m, StringComparison.Ordinal);
        Assert.EndsWith("/issues/735.", GpuOffloadPlanner.PriorityIssueUrl + ".", StringComparison.Ordinal);
    }

    [Fact]
    public void UnsatisfiableMessage_StatesWhenFreeVramIsUnknown()
    {
        string m = GpuOffloadPlanner.BuildUnsatisfiableMessage(Architecture.NemotronH, 1, 2, 1L << 30, null, null, "no CUDA");
        Assert.Contains("device memory unknown", m, StringComparison.Ordinal);
    }
}
