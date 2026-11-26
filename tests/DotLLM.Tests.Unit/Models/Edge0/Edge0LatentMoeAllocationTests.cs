using DotLLM.Models.Edge0;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>Collection that runs alone: see <see cref="Edge0LatentMoeAllocationTests"/>.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class Edge0ZeroAllocCollection
{
    /// <summary>Collection name.</summary>
    public const string Name = "Edge0ZeroAlloc";
}

/// <summary>
/// Steady-state allocation of Edge0 latent-MoE (#1972, #1985). This is its own non-parallel collection on purpose: the kernels rent scratch
/// from the process-wide <c>ArrayPool&lt;T&gt;.Shared</c>, and a test class running concurrently that rents the same size bucket takes the buffer
/// Edge0 just returned, so Edge0's next rent misses and allocates one array (the 32792 B = 8192 floats + header seen in CI, worst under the slower
/// DOTLLM_CPU_ISA caps). That is cross-test pool contention, not Edge0's own steady state, which is what this test measures. Isolating the
/// measurement does not weaken it: the assertion is still exactly zero bytes. The production-side exposure is tracked in #1985.
/// </summary>
[Collection(Edge0ZeroAllocCollection.Name)]
public sealed class Edge0LatentMoeAllocationTests
{
    private const int H = SyntheticEdge0Pack.Hidden, K = SyntheticEdge0Pack.TopK;

    private static float[] Random(int n, int seed, float scale)
    {
        var r = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1) * scale;
        return a;
    }

    [Theory]
    [InlineData(1)]    // one chunk
    [InlineData(70)]   // two chunks (crosses the 64-token boundary)
    public void Forward_SteadyState_AllocatesNothing_OnTheCallingThread(int tokens)
    {
        // #1972: the per-chunk scratch used to be a fresh helper object plus two Lists per chunk. Serial path (pool null) so the
        // thread-local counter sees everything; a warmed second pass must not allocate a single byte.
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        var router = Random(SyntheticEdge0Pack.Experts * H, 11, 0.6f);
        var x = Random(tokens * H, 12, 1f);
        var y = new float[tokens * H];
        for (int i = 0; i < 3; i++) Edge0LatentMoe.Forward(pack, 1, router, K, true, null, x, y, tokens);   // warm: JIT, ArrayPool buckets
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 5; i++) Edge0LatentMoe.Forward(pack, 1, router, K, true, null, x, y, tokens);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated == 0, $"{allocated} bytes allocated over 5 steady-state calls ({tokens} tokens)");
    }
}
