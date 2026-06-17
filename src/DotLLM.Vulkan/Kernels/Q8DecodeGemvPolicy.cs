namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Per-shape dispatch policy for the Q8_0 decode-path (N=1) GEMV: decides, for a
/// given device and contraction dimension <c>K</c>, whether the DP4a INT8 path
/// (<see cref="MatMulQ8_0Dp4aPqKernel"/>) or the FP32-activation workgroup kernel
/// (<see cref="MatMulQ8_0Kernel"/>) is faster.
/// </summary>
/// <remarks>
/// <para>
/// Background: the optimal decode GEMV is not only <b>vendor</b>-dependent (which
/// <see cref="MatMulQ8_0Kernel"/> already handles) but also <b>shape</b>-dependent
/// along the DP4a-vs-workgroup axis. Measured on the local box (device-local
/// weights, batched-fence + paired per-round median ratios, SmolLM2-135M and
/// Llama-3.2-1B decode shapes):
/// </para>
/// <list type="bullet">
///   <item><description>
///     <b>Intel Arc</b> (DP4a default-on): DP4a-pq wins <i>every</i> shape across
///     both models by 1.6–2.2× over the best workgroup variant — at K=576 and at
///     K=8192 alike. Threshold = 0 (always DP4a when supported).
///   </description></item>
///   <item><description>
///     <b>NVIDIA</b> (DP4a default-off): DP4a-pq wins the long-contraction shapes
///     (K≥2048: attn_q 1.29×, attn_o 1.19×, ffn_down 2.00×, lm_head 1.07× vs the
///     wg64 vendor default) but <i>loses</i> at the short SmolLM shapes (K≤1536),
///     where the wg64 kernel is faster. Crossover ≈ K=2048. Among the workgroup
///     variants themselves wg64 is best or statistically tied on every NVIDIA
///     shape, so per-shape <i>workgroup-variant</i> selection adds nothing — the
///     only shape-dependent lever that matters on NVIDIA is DP4a-vs-workgroup.
///   </description></item>
/// </list>
/// <para>
/// This policy is consulted only when the DP4a kernel is available and DP4a is
/// enabled (Arc by default; any vendor via <c>DOTLLM_VULKAN_ENABLE_DP4A=1</c>).
/// It never turns DP4a <i>on</i> where it was off — it only refines, per shape,
/// whether an already-enabled DP4a path is actually used for a given GEMV, so the
/// NVIDIA "force DP4a on" path stops regressing the short-K shapes. The threshold
/// is overridable with <c>DOTLLM_VULKAN_DP4A_MIN_K</c> for tuning on other
/// hardware.
/// </para>
/// </remarks>
public static class Q8DecodeGemvPolicy
{
    private const uint VendorIntel = 0x8086;
    private const uint VendorNvidia = 0x10DE;

    /// <summary>
    /// Minimum contraction dimension <c>K</c> at or above which the DP4a decode
    /// GEMV is expected to beat the workgroup kernel for <paramref name="vendorId"/>.
    /// Shapes with smaller <c>K</c> should use the workgroup kernel.
    /// </summary>
    /// <param name="vendorId">Vulkan device vendor id.</param>
    /// <returns>
    /// The K crossover threshold. <c>0</c> means DP4a is always preferred when
    /// available (Intel/Arc and the conservative default for unknown vendors).
    /// </returns>
    public static int Dp4aMinK(uint vendorId)
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DP4A_MIN_K");
        if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int forced) && forced >= 0)
            return forced;

        return vendorId switch
        {
            // Arc: DP4a wins at every measured K → always prefer it.
            VendorIntel => 0,
            // NVIDIA: DP4a wins only the long-contraction shapes; wg64 wins K≤1536.
            VendorNvidia => 2048,
            // Unknown vendor: be conservative and keep DP4a wherever it was chosen
            // (threshold 0 = no per-shape veto), matching prior behaviour.
            _ => 0,
        };
    }

    /// <summary>
    /// Decides whether a decode GEMV with contraction dimension <paramref name="k"/>
    /// should take the DP4a path on <paramref name="vendorId"/>.
    /// </summary>
    /// <param name="vendorId">Vulkan device vendor id.</param>
    /// <param name="k">Contraction (input) dimension of the GEMV.</param>
    /// <returns><c>true</c> to use DP4a-pq; <c>false</c> to use the workgroup kernel.</returns>
    public static bool UseDp4a(uint vendorId, int k) => k >= Dp4aMinK(vendorId);
}
