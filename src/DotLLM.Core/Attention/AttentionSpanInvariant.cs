using System.Diagnostics;

namespace DotLLM.Core.Attention;

/// <summary>
/// Debug-only guard for the KV-span invariant the GPU attention kernels silently depend on
/// (#532): the span they reduce over must be exactly the causally visible one.
/// </summary>
/// <remarks>
/// <para>
/// <b>The invariant.</b> <c>seqKv == positionOffset + seqQ</c> at every GPU attention entry
/// point. It holds today because every KV cache updates its length monotonically
/// (<c>if (maxPos + 1 &gt; _currentLength) _currentLength = maxPos + 1;</c>), so
/// <c>seqKv = max(CurrentLength_before, positionOffset + seqQ)</c> — and equality follows from the
/// fact that no caller ever forwards <i>into the middle of</i> a warm cache.
/// </para>
/// <para>
/// <b>Why it matters.</b> If a caller ever passes a padded <c>seqKv</c>, the extra rows are
/// causally masked and contribute exactly 0.0 — but the <b>split-KV</b> kernels derive their
/// reduction boundaries from the cache length (<c>splitLen = ceil(seqKv/numSplits)</c> on Vulkan,
/// <c>chunk = ceil(seq_kv/kv_split)</c> on CUDA), so the real rows get regrouped into different
/// partials and accumulate in a different order. Measured on both backends: 214-230 of 256
/// output elements differ on Vulkan (worst 2.98E-08) and 5520-5581 of 6144 on CUDA (worst
/// 8.20E-08). #525 showed that a ULP of exactly that scale is digitized by activation
/// quantization into a whole step, amplified ~14,600x by <c>o_proj</c>, and can change the
/// emitted token. <b>The failure mode is silently wrong logits, not a crash.</b>
/// </para>
/// <para>
/// <b>Why an assert rather than a fix.</b> No in-tree caller can violate it — audited across
/// every attention call site in every backend — so fixing the split-KV reduction order would buy
/// nothing today. What the audit did find is that the invariant is held on a knife edge: three
/// paths sit on equality by exactly one (speculative rollback, and both 100%-prefix-cache-hit
/// paths), and two sites are safe by accident rather than design — <c>CudaMlaAttention</c>
/// derives <c>seqKv</c> from the cache's append cursor with no check against
/// <c>positionOffset</c>, and <c>NaiveAttentionStrategy</c> takes it from a caller-supplied
/// tensor shape (dead code today). This makes the next person to break it fail loudly.
/// </para>
/// <para>
/// <b>Not applied to the CPU backend.</b> <c>Cpu.Kernels.Attention</c> was fixed properly by
/// #525 — it confines every reduction to the row's own visible range, "never on how many keys
/// happen to be resident in the KV cache". A padded <c>seqKv</c> there is harmless, so asserting
/// on it would forbid something legal.
/// </para>
/// <para>
/// <b>Cost.</b> <c>[Conditional("DEBUG")]</c> — the call and its arguments are not emitted
/// at all in Release, so this is genuinely zero-cost on the decode hot path.
/// </para>
/// </remarks>
public static class AttentionSpanInvariant
{
    /// <summary>
    /// True when the KV span is exactly the causally visible span.
    /// </summary>
    /// <remarks>
    /// Exposed separately from <see cref="AssertTight"/> so the rule itself can be unit-tested.
    /// A failing <see cref="Debug.Assert(bool)"/> can abort the host process, so a test must not
    /// trip the assert directly — it exercises this predicate instead, and the assert's real
    /// validation is that it stays silent across the Debug test suite's model/forward/decode
    /// paths.
    /// </remarks>
    public static bool IsTight(int seqKv, int positionOffset, int seqQ)
        => seqKv == positionOffset + seqQ;

    /// <summary>
    /// Asserts that the KV span equals the causally visible span. Compiled out in Release.
    /// </summary>
    /// <param name="seqKv">The KV span the kernel will reduce over.</param>
    /// <param name="positionOffset">Absolute position of the first query row.</param>
    /// <param name="seqQ">Number of query rows.</param>
    /// <param name="backend">Backend name, for the failure message.</param>
    [Conditional("DEBUG")]
    public static void AssertTight(int seqKv, int positionOffset, int seqQ, string backend)
    {
        Debug.Assert(
            IsTight(seqKv, positionOffset, seqQ),
            $"{backend} attention received a PADDED KV span: seqKv={seqKv} but " +
            $"positionOffset({positionOffset}) + seqQ({seqQ}) = {positionOffset + seqQ}. " +
            "The split-KV kernels derive their reduction boundaries from seqKv, so padding " +
            "silently changes the accumulation order of the real rows and can change the " +
            "emitted token on a quantized model. See #532 before relaxing this.");
    }
}
