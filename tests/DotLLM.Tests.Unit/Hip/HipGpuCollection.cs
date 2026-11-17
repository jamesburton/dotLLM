using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>Serializes HIP qualification so memory and device-ownership checks do not overlap.</summary>
[CollectionDefinition("HipGpu", DisableParallelization = true)]
public sealed class HipGpuCollection;
