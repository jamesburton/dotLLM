using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>Serializes HIP qualification so memory and device-ownership checks do not overlap.</summary>
[CollectionDefinition("HipGpu", DisableParallelization = true)]
public sealed class HipGpuCollection;

// Environment changes must stay in the non-parallel HipGpu collection.
internal sealed class HipQ8Environment : IDisposable
{
    private readonly string? _previous = Environment.GetEnvironmentVariable(DotLLM.Hip.HipPagedKvCache.Q8OptInEnvironmentVariable);

    public HipQ8Environment(string? value)
        => Environment.SetEnvironmentVariable(DotLLM.Hip.HipPagedKvCache.Q8OptInEnvironmentVariable, value);

    public void Dispose()
        => Environment.SetEnvironmentVariable(DotLLM.Hip.HipPagedKvCache.Q8OptInEnvironmentVariable, _previous);
}
