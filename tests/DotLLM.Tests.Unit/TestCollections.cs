using Xunit;

namespace DotLLM.Tests.Unit;

/// <summary>
/// Marker for xUnit collections that must not run in parallel with each
/// other. Apply <c>[Collection("SequentialFileIO")]</c> to any test class
/// whose tests contend on shared file handles (e.g., the 442 KB Granite
/// <c>merges.txt</c>) when the suite is executed with xUnit's default
/// collection parallelism.
/// </summary>
[CollectionDefinition("SequentialFileIO", DisableParallelization = true)]
public class SequentialFileIOCollection
{
}
