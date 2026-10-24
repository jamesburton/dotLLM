namespace DotLLM.Hip;

/// <summary>
/// HIP/ROCm transformer forward pass — NOT YET IMPLEMENTED.
/// <para>
/// This scaffold commit establishes the project structure, P/Invoke layer,
/// code-object loader, and one working kernel (<see cref="Kernels.RmsNormKernel"/>).
/// The full LLM forward pass lands in follow-up work via mechanical hipify-perl
/// translation of the kernels in <c>native/kernels/</c>:
/// rope, attention, swiglu, softmax, add, bias_add, embedding, convert,
/// dequant (Q4_0/Q4_K/Q5_0/Q5_K/Q6_K/Q8_0), quantized_gemv, fused_add_rmsnorm,
/// per_head_rmsnorm.
/// </para>
/// <para>
/// See <c>docs/HIP.md</c> for the porting workflow and kernel catalog.
/// </para>
/// </summary>
public static class HipTransformerModel
{
    /// <summary>
    /// Placeholder entry point. Always throws <see cref="NotImplementedException"/>
    /// pointing at the follow-up work.
    /// </summary>
    public static void Forward()
    {
        throw new NotImplementedException(
            "HIP transformer forward pass is not implemented yet. " +
            "The DotLLM.Hip assembly currently ships a proof-of-pipeline scaffold " +
            "(P/Invoke layer, module loader, RmsNorm kernel). Port the remaining " +
            "kernels from native/kernels/*.cu via hipify-perl — see docs/HIP.md.");
    }
}
