using DotLLM.Core.Attention;
using Xunit;

namespace DotLLM.Tests.Unit.AttentionInvariants;

/// <summary>
/// Pins the KV-span rule the GPU attention guard enforces (#532).
/// </summary>
/// <remarks>
/// <para>
/// The cases below are the real shapes the #532 call-site audit classified, so this doubles as
/// an executable record of <i>why</i> each production path is safe. If someone later changes one
/// of those call sites, the shape it must still satisfy is written down here rather than living
/// only in a comment.
/// </para>
/// <para>
/// This tests <see cref="AttentionSpanInvariant.IsTight"/>, not
/// <see cref="AttentionSpanInvariant.AssertTight"/>: a failing <c>Debug.Assert</c> can abort the
/// host process, so tripping it inside the runner is not safe. The assert's own validation is
/// that it stays <i>silent</i> across the Debug suite's model / forward / decode tests — 1094
/// passing there is the evidence it does not false-positive on a real path.
/// </para>
/// <para>
/// Namespace note: this deliberately is NOT <c>DotLLM.Tests.Unit.Core.*</c> — that shadows
/// <c>DotLLM.Core</c> for every other test file and breaks the build. The existing files under
/// <c>Core/</c> drop the segment for the same reason.
/// </para>
/// </remarks>
public class AttentionSpanInvariantTests
{
    /// <summary>Every shape the audit found reachable in production must be tight.</summary>
    [Theory]
    [InlineData(1, 0, 1)]         // first decode step
    [InlineData(301, 300, 1)]     // ordinary decode
    [InlineData(4096, 4095, 1)]   // long-context decode
    [InlineData(512, 0, 512)]     // cold prefill
    [InlineData(640, 512, 128)]   // chunked prefill, second chunk
    [InlineData(100, 99, 1)]      // 100%-prefix-cache-hit re-forward of the last prompt token
    [InlineData(48, 40, 8)]       // speculative replay after rollback to position+acceptedCount
    [InlineData(20, 16, 4)]       // diffusion decode: seqKv = p + c, positionOffset = p, seqQ = c
    public void ReachableProductionShapes_AreTight(int seqKv, int positionOffset, int seqQ)
        => Assert.True(AttentionSpanInvariant.IsTight(seqKv, positionOffset, seqQ));

    /// <summary>
    /// The shapes that would silently reorder the split-KV reduction. These are what the guard
    /// exists to catch, and none is reachable today.
    /// </summary>
    [Theory]
    [InlineData(4096, 300, 1)]    // decode against a cache padded to capacity
    [InlineData(320, 300, 1)]     // padded by a handful of rows
    [InlineData(302, 300, 1)]     // padded by exactly ONE — the knife-edge case
    [InlineData(1024, 0, 512)]    // prefill given a capacity-sized K tensor
    public void PaddedSpans_AreRejected(int seqKv, int positionOffset, int seqQ)
        => Assert.False(AttentionSpanInvariant.IsTight(seqKv, positionOffset, seqQ));

    /// <summary>
    /// A span SHORTER than the visible range is also a violation, and a different bug: it would
    /// drop real keys rather than merely reorder them. Pinned separately so a future "relax the
    /// guard to &lt;=" change has to confront it.
    /// </summary>
    [Theory]
    [InlineData(300, 300, 1)]
    [InlineData(256, 0, 512)]
    public void UnderLongSpans_AreAlsoRejected(int seqKv, int positionOffset, int seqQ)
        => Assert.False(AttentionSpanInvariant.IsTight(seqKv, positionOffset, seqQ));
}
