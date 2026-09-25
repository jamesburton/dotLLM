namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Mamba-3 QK-Normalization wrapper for the B and C coefficient tensors. Applies
/// RMSNorm independently to each <c>[d_state]</c> slice of a
/// <c>[seqLen, n_group, d_state]</c> row-major tensor, with a single per-element
/// weight of length <c>d_state</c> shared across all <c>(t, g)</c> slices.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the CPU reference <see cref="DotLLM.Cpu.Kernels.Mamba3QkNorm.Execute"/>:
/// in the VikramKarLex/mamba3-minimal reference, <c>B_norm</c> / <c>C_norm</c> are
/// a pair of <c>RMSNorm(bc_dim)</c> layers applied before group reshape. In our
/// split shape <c>[T, n_group, d_state]</c> the same per-<c>d_state</c>-slice
/// normalization is applied independently across groups; the learnable weight
/// is shared across <c>(t, g)</c> slices, matching the reference's single
/// <c>RMSNorm(bc_dim)</c> layer.
/// </para>
/// <para>
/// <b>Implementation choice (Option A).</b> This is mathematically identical to
/// <see cref="RmsNormF32Kernel"/> with <c>rowCount = seqLen * n_group</c> and
/// <c>n = d_state</c>: every <c>[d_state]</c> slice is one row, the same
/// per-feature weight is broadcast across all rows. Rather than ship a clone
/// of <c>rmsnorm_f32.comp</c>, this wrapper delegates straight into the existing
/// kernel with the reshaped row count. No new shader is needed and the kernel
/// continues to benefit from the subgroup-arithmetic variant when the device
/// supports it.
/// </para>
/// <para>
/// The wrapper holds no GPU resources of its own — it borrows the
/// <see cref="RmsNormF32Kernel"/> passed in at construction. The caller retains
/// ownership of that kernel and is responsible for disposing it; this class is
/// intentionally not <see cref="IDisposable"/>.
/// </para>
/// <para>
/// <b>Alias safety:</b> the underlying <see cref="RmsNormF32Kernel"/> reads from
/// <c>input</c> and writes to <c>output</c>; passing the same buffer for both
/// (in-place QK-norm, the normal mode of use) is safe because each row is read
/// and written independently within a single workgroup.
/// </para>
/// </remarks>
public sealed class Mamba3QkNormF32Kernel
{
    private readonly RmsNormF32Kernel _rmsNorm;

    /// <summary>
    /// Wraps an existing <see cref="RmsNormF32Kernel"/>. The inner kernel is
    /// borrowed, not owned: the caller remains responsible for its lifetime.
    /// </summary>
    /// <param name="rmsNorm">The shared RMSNorm kernel to delegate to.</param>
    public Mamba3QkNormF32Kernel(RmsNormF32Kernel rmsNorm)
    {
        ArgumentNullException.ThrowIfNull(rmsNorm);
        _rmsNorm = rmsNorm;
    }

    /// <summary>
    /// Returns <c>true</c> when the underlying RMSNorm kernel uses the
    /// subgroup-arithmetic variant. Exposed for tests and telemetry.
    /// </summary>
    public bool UsesSubgroupReduce => _rmsNorm.UsesSubgroupReduce;

    /// <summary>
    /// Synchronous launch — wraps <see cref="Record"/>; used by unit tests.
    /// </summary>
    /// <param name="bc">
    /// FP32 buffer of shape <c>[seqLen, nGroup, dState]</c> row-major. Normalized
    /// in place when the same buffer is passed for input and output (the
    /// standard QK-norm usage).
    /// </param>
    /// <param name="weight">FP32 per-element scale, length <c>dState</c>. Shared across all <c>(t, g)</c> slices.</param>
    /// <param name="seqLen">Number of tokens <c>T</c>. Zero is a no-op.</param>
    /// <param name="nGroup">Number of B/C groups.</param>
    /// <param name="dState">State-space width per group; the per-slice norm length.</param>
    /// <param name="eps">RMSNorm stabilizing constant (typically <c>1e-5</c>).</param>
    public void Launch(
        VulkanDevice.Buffer bc, VulkanDevice.Buffer weight,
        int seqLen, int nGroup, int dState, float eps)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nGroup <= 0) throw new ArgumentOutOfRangeException(nameof(nGroup));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen == 0) return; // no-op, matches CPU reference

        // bc is normalized in-place by the caller's convention: pass the same
        // buffer for input and output. RmsNormF32Kernel writes each row from
        // the result of its own per-row reduction, so input == output is safe.
        _rmsNorm.Launch(bc, weight, bc, rowCount: seqLen * nGroup, n: dState, eps: eps);
    }

    /// <summary>
    /// Records QK-norm into <paramref name="cmdBuf"/> without submitting.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="bc">
    /// FP32 buffer of shape <c>[seqLen, nGroup, dState]</c> row-major. Normalized
    /// in place.
    /// </param>
    /// <param name="weight">FP32 per-element scale, length <c>dState</c>.</param>
    /// <param name="seqLen">Number of tokens <c>T</c>. Zero is a no-op.</param>
    /// <param name="nGroup">Number of B/C groups.</param>
    /// <param name="dState">State-space width per group.</param>
    /// <param name="eps">RMSNorm stabilizing constant.</param>
    public void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bc, VulkanDevice.Buffer weight,
        int seqLen, int nGroup, int dState, float eps)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nGroup <= 0) throw new ArgumentOutOfRangeException(nameof(nGroup));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen == 0) return; // no-op, matches CPU reference

        _rmsNorm.Record(
            cmdBuf,
            input: bc, weight: weight, output: bc,
            rowCount: seqLen * nGroup, n: dState, eps: eps);
    }
}
