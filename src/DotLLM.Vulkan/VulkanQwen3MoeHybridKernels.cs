using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Container for the Vulkan kernels needed by the Qwen3MoeHybrid forward path.
/// Created once per model and owned by it; <see cref="Dispose"/> releases every
/// kernel descriptor pool and SPIR-V module. Splitting this off from the model
/// keeps the model's constructor signature manageable and makes the kernel set
/// auditable from one place.
/// </summary>
internal sealed class VulkanQwen3MoeHybridKernels : IDisposable
{
    // ── Matmul kernels (Q8_0 / K-quants / F16 / BF16 / F32) ─────────────────
    public MatMulF32Kernel MatMul { get; }
    public MatMulQ8_0Kernel MatMulQ8 { get; }
    public MatMulQ8_0GemmKernel MatMulQ8Gemm { get; }
    public MatMulQ8_0GemmCoopmatKernel? MatMulQ8GemmCoopmat { get; }
    public MatMulQ2KGemvF32Kernel MatMulQ2K { get; }
    public MatMulQ2KGemmF32Kernel MatMulQ2KGemm { get; }
    public MatMulQ3KGemvF32Kernel MatMulQ3K { get; }
    public MatMulQ3KGemmF32Kernel MatMulQ3KGemm { get; }
    public MatMulQ4KGemvF32Kernel MatMulQ4K { get; }
    public MatMulQ4KGemmF32Kernel MatMulQ4KGemm { get; }
    public MatMulQ5KGemvF32Kernel MatMulQ5K { get; }
    public MatMulQ5KGemmF32Kernel MatMulQ5KGemm { get; }
    public MatMulQ6KGemvF32Kernel MatMulQ6K { get; }
    public MatMulQ6KGemmF32Kernel MatMulQ6KGemm { get; }
    public MatMulIq4NlGemvF32Kernel MatMulIq4Nl { get; }
    public MatMulIq4NlGemmF32Kernel MatMulIq4NlGemm { get; }
    public MatMulIq4XsGemvF32Kernel MatMulIq4Xs { get; }
    public MatMulIq4XsGemmF32Kernel MatMulIq4XsGemm { get; }
    /// <summary>Shared IQ2 codebooks (3 grids + ksigns) backing all 6 IQ2 matmul kernels.</summary>
    public Iq2Codebooks Iq2Codebooks { get; }
    /// <summary>IQ2_XXS GEMV (decode).</summary>
    public MatMulIq2XxsGemvF32Kernel MatMulIq2Xxs { get; }
    /// <summary>IQ2_XXS GEMM (prefill).</summary>
    public MatMulIq2XxsGemmF32Kernel MatMulIq2XxsGemm { get; }
    /// <summary>IQ2_XS GEMV.</summary>
    public MatMulIq2XsGemvF32Kernel MatMulIq2Xs { get; }
    /// <summary>IQ2_XS GEMM.</summary>
    public MatMulIq2XsGemmF32Kernel MatMulIq2XsGemm { get; }
    /// <summary>IQ2_S GEMV.</summary>
    public MatMulIq2SGemvF32Kernel MatMulIq2S { get; }
    /// <summary>IQ2_S GEMM.</summary>
    public MatMulIq2SGemmF32Kernel MatMulIq2SGemm { get; }
    public MatMulIq1SGemvF32Kernel MatMulIq1S { get; }
    public MatMulIq1SGemmF32Kernel MatMulIq1SGemm { get; }
    public MatMulF16GemvF32Kernel MatMulF16 { get; }
    public MatMulF16GemmF32Kernel MatMulF16Gemm { get; }
    public MatMulF16GemmCoopmatKernel? MatMulF16GemmCoopmat { get; }
    public MatMulBf16GemvF32Kernel MatMulBf16 { get; }
    public MatMulBf16GemmF32Kernel MatMulBf16Gemm { get; }

    // ── Shared norms / attention / SwiGLU / Add ─────────────────────────────
    public RmsNormF32Kernel RmsNorm { get; }
    public RopeF32Kernel Rope { get; }
    public AttentionF32Kernel Attention { get; }
    public SwiGluF32Kernel SwiGlu { get; }
    public AddKernel Add { get; }
    public SiluInplaceF32Kernel SiluInplace { get; }
    public Conv1dCausalF32Kernel Conv1dCausal { get; }

    // ── GDN-specific kernels ────────────────────────────────────────────────
    public GdnL2NormalizeHeadsF32Kernel GdnL2Normalize { get; }
    public GdnScanStepF32Kernel GdnScanStep { get; }
    public GdnScanMultiTokenF32Kernel GdnScanMultiToken { get; }
    public GdnPostScanGateF32Kernel GdnPostScanGate { get; }
    public GdnDecayF32Kernel GdnDecay { get; }
    public SigmoidInplaceF32Kernel SigmoidInplace { get; }

    // ── Full-attention sigmoid-gated multiply ───────────────────────────────
    public SigmoidGateMulF32Kernel SigmoidGateMul { get; }

    // ── MoE kernels ─────────────────────────────────────────────────────────
    public MoeTopKSoftmaxF32Kernel MoeTopkSoftmax { get; }
    public MoeBroadcastF32Kernel MoeBroadcast { get; }
    public MoeIndexedMatmulF32Kernel MoeIndexedMatmul { get; }
    public MoeWeightedScatterF32Kernel MoeWeightedScatter { get; }
    public MoeSigmoidGatedAddF32Kernel MoeSigmoidGatedAdd { get; }

    private VulkanQwen3MoeHybridKernels(
        MatMulF32Kernel matmul, MatMulQ8_0Kernel matmulQ8, MatMulQ8_0GemmKernel matmulQ8Gemm,
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat,
        MatMulQ2KGemvF32Kernel matmulQ2K, MatMulQ2KGemmF32Kernel matmulQ2KGemm,
        MatMulQ3KGemvF32Kernel matmulQ3K, MatMulQ3KGemmF32Kernel matmulQ3KGemm,
        MatMulQ4KGemvF32Kernel matmulQ4K, MatMulQ4KGemmF32Kernel matmulQ4KGemm,
        MatMulQ5KGemvF32Kernel matmulQ5K, MatMulQ5KGemmF32Kernel matmulQ5KGemm,
        MatMulQ6KGemvF32Kernel matmulQ6K, MatMulQ6KGemmF32Kernel matmulQ6KGemm,
        MatMulIq4NlGemvF32Kernel matmulIq4Nl, MatMulIq4NlGemmF32Kernel matmulIq4NlGemm,
        MatMulIq4XsGemvF32Kernel matmulIq4Xs, MatMulIq4XsGemmF32Kernel matmulIq4XsGemm,
        Iq2Codebooks iq2Codebooks,
        MatMulIq2XxsGemvF32Kernel matmulIq2Xxs, MatMulIq2XxsGemmF32Kernel matmulIq2XxsGemm,
        MatMulIq2XsGemvF32Kernel matmulIq2Xs, MatMulIq2XsGemmF32Kernel matmulIq2XsGemm,
        MatMulIq2SGemvF32Kernel matmulIq2S, MatMulIq2SGemmF32Kernel matmulIq2SGemm,
        MatMulIq1SGemvF32Kernel matmulIq1S, MatMulIq1SGemmF32Kernel matmulIq1SGemm,
        MatMulF16GemvF32Kernel matmulF16, MatMulF16GemmF32Kernel matmulF16Gemm,
        MatMulF16GemmCoopmatKernel? matmulF16GemmCoopmat,
        MatMulBf16GemvF32Kernel matmulBf16, MatMulBf16GemmF32Kernel matmulBf16Gemm,
        RmsNormF32Kernel rmsnorm, RopeF32Kernel rope, AttentionF32Kernel attention,
        SwiGluF32Kernel swiglu, AddKernel add, SiluInplaceF32Kernel silu, Conv1dCausalF32Kernel conv1d,
        GdnL2NormalizeHeadsF32Kernel gdnL2,
        GdnScanStepF32Kernel gdnScan, GdnScanMultiTokenF32Kernel gdnScanMulti,
        GdnPostScanGateF32Kernel gdnPost,
        GdnDecayF32Kernel gdnDecay, SigmoidInplaceF32Kernel sigmoidInplace,
        SigmoidGateMulF32Kernel sigmoidGateMul,
        MoeTopKSoftmaxF32Kernel moeTopk, MoeBroadcastF32Kernel moeBroadcast,
        MoeIndexedMatmulF32Kernel moeIndexedMatmul, MoeWeightedScatterF32Kernel moeWeightedScatter,
        MoeSigmoidGatedAddF32Kernel moeSigmoidGatedAdd)
    {
        MatMul = matmul; MatMulQ8 = matmulQ8; MatMulQ8Gemm = matmulQ8Gemm;
        MatMulQ8GemmCoopmat = matmulQ8GemmCoopmat;
        MatMulQ2K = matmulQ2K; MatMulQ2KGemm = matmulQ2KGemm;
        MatMulQ3K = matmulQ3K; MatMulQ3KGemm = matmulQ3KGemm;
        MatMulQ4K = matmulQ4K; MatMulQ4KGemm = matmulQ4KGemm;
        MatMulQ5K = matmulQ5K; MatMulQ5KGemm = matmulQ5KGemm;
        MatMulQ6K = matmulQ6K; MatMulQ6KGemm = matmulQ6KGemm;
        MatMulIq4Nl = matmulIq4Nl; MatMulIq4NlGemm = matmulIq4NlGemm;
        MatMulIq4Xs = matmulIq4Xs; MatMulIq4XsGemm = matmulIq4XsGemm;
        Iq2Codebooks = iq2Codebooks;
        MatMulIq2Xxs = matmulIq2Xxs; MatMulIq2XxsGemm = matmulIq2XxsGemm;
        MatMulIq2Xs = matmulIq2Xs; MatMulIq2XsGemm = matmulIq2XsGemm;
        MatMulIq2S = matmulIq2S; MatMulIq2SGemm = matmulIq2SGemm;
        MatMulIq1S = matmulIq1S; MatMulIq1SGemm = matmulIq1SGemm;
        MatMulF16 = matmulF16; MatMulF16Gemm = matmulF16Gemm;
        MatMulF16GemmCoopmat = matmulF16GemmCoopmat;
        MatMulBf16 = matmulBf16; MatMulBf16Gemm = matmulBf16Gemm;
        RmsNorm = rmsnorm; Rope = rope; Attention = attention;
        SwiGlu = swiglu; Add = add; SiluInplace = silu; Conv1dCausal = conv1d;
        GdnL2Normalize = gdnL2;
        GdnScanStep = gdnScan; GdnScanMultiToken = gdnScanMulti;
        GdnPostScanGate = gdnPost;
        GdnDecay = gdnDecay; SigmoidInplace = sigmoidInplace;
        SigmoidGateMul = sigmoidGateMul;
        MoeTopkSoftmax = moeTopk; MoeBroadcast = moeBroadcast;
        MoeIndexedMatmul = moeIndexedMatmul; MoeWeightedScatter = moeWeightedScatter;
        MoeSigmoidGatedAdd = moeSigmoidGatedAdd;
    }

    public static VulkanQwen3MoeHybridKernels Create(VulkanDevice device, string spvDir)
    {
        var matmul = MatMulF32Kernel.Create(device, spvDir);
        var matmulQ8 = MatMulQ8_0Kernel.Create(device, spvDir);
        var matmulQ8Gemm = MatMulQ8_0GemmKernel.Create(device, spvDir);
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat = null;
        if (device.HasCooperativeMatrix)
        {
            try { matmulQ8GemmCoopmat = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir); }
            catch (InvalidOperationException) { /* No usable Q8_0 tile shape — stay on scalar. */ }
        }
        var matmulQ2K = MatMulQ2KGemvF32Kernel.Create(device, spvDir);
        var matmulQ2KGemm = MatMulQ2KGemmF32Kernel.Create(device, spvDir);
        var matmulQ3K = MatMulQ3KGemvF32Kernel.Create(device, spvDir);
        var matmulQ3KGemm = MatMulQ3KGemmF32Kernel.Create(device, spvDir);
        var matmulQ4K = MatMulQ4KGemvF32Kernel.Create(device, spvDir);
        var matmulQ4KGemm = MatMulQ4KGemmF32Kernel.Create(device, spvDir);
        var matmulQ5K = MatMulQ5KGemvF32Kernel.Create(device, spvDir);
        var matmulQ5KGemm = MatMulQ5KGemmF32Kernel.Create(device, spvDir);
        var matmulQ6K = MatMulQ6KGemvF32Kernel.Create(device, spvDir);
        var matmulQ6KGemm = MatMulQ6KGemmF32Kernel.Create(device, spvDir);
        var matmulIq4Nl = MatMulIq4NlGemvF32Kernel.Create(device, spvDir);
        var matmulIq4NlGemm = MatMulIq4NlGemmF32Kernel.Create(device, spvDir);
        var matmulIq4Xs = MatMulIq4XsGemvF32Kernel.Create(device, spvDir);
        var matmulIq4XsGemm = MatMulIq4XsGemmF32Kernel.Create(device, spvDir);
        var iq2Codebooks = Iq2Codebooks.Create(device);
        var matmulIq2Xxs     = MatMulIq2XxsGemvF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2XxsGemm = MatMulIq2XxsGemmF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2Xs      = MatMulIq2XsGemvF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2XsGemm  = MatMulIq2XsGemmF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2S       = MatMulIq2SGemvF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2SGemm   = MatMulIq2SGemmF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq1S = MatMulIq1SGemvF32Kernel.Create(device, spvDir);
        var matmulIq1SGemm = MatMulIq1SGemmF32Kernel.Create(device, spvDir);
        var matmulF16 = MatMulF16GemvF32Kernel.Create(device, spvDir);
        var matmulF16Gemm = MatMulF16GemmF32Kernel.Create(device, spvDir);
        MatMulF16GemmCoopmatKernel? matmulF16GemmCoopmat = null;
        if (device.HasCooperativeMatrix)
        {
            try { matmulF16GemmCoopmat = MatMulF16GemmCoopmatKernel.Create(device, spvDir); }
            catch (InvalidOperationException) { /* No usable F16 tile shape — stay on scalar. */ }
        }
        var matmulBf16 = MatMulBf16GemvF32Kernel.Create(device, spvDir);
        var matmulBf16Gemm = MatMulBf16GemmF32Kernel.Create(device, spvDir);

        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        var rope = RopeF32Kernel.Create(device, spvDir);
        var attention = AttentionF32Kernel.Create(device, spvDir);
        var swiglu = SwiGluF32Kernel.Create(device, spvDir);
        var add = AddKernel.Create(device, spvDir);
        var silu = SiluInplaceF32Kernel.Create(device, spvDir);
        var conv1d = Conv1dCausalF32Kernel.Create(device, spvDir);

        var gdnL2 = GdnL2NormalizeHeadsF32Kernel.Create(device, spvDir);
        var gdnScan = GdnScanStepF32Kernel.Create(device, spvDir);
        var gdnScanMulti = GdnScanMultiTokenF32Kernel.Create(device, spvDir);
        var gdnPost = GdnPostScanGateF32Kernel.Create(device, spvDir);
        var gdnDecay = GdnDecayF32Kernel.Create(device, spvDir);
        var sigmoidInplace = SigmoidInplaceF32Kernel.Create(device, spvDir);
        var sigGateMul = SigmoidGateMulF32Kernel.Create(device, spvDir);

        var moeTopk = MoeTopKSoftmaxF32Kernel.Create(device, spvDir);
        var moeBroadcast = MoeBroadcastF32Kernel.Create(device, spvDir);
        var moeIndexed = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
        var moeScatter = MoeWeightedScatterF32Kernel.Create(device, spvDir);
        var moeSigmoidGatedAdd = MoeSigmoidGatedAddF32Kernel.Create(device, spvDir);

        return new VulkanQwen3MoeHybridKernels(
            matmul, matmulQ8, matmulQ8Gemm, matmulQ8GemmCoopmat,
            matmulQ2K, matmulQ2KGemm,
            matmulQ3K, matmulQ3KGemm,
            matmulQ4K, matmulQ4KGemm,
            matmulQ5K, matmulQ5KGemm,
            matmulQ6K, matmulQ6KGemm,
            matmulIq4Nl, matmulIq4NlGemm,
            matmulIq4Xs, matmulIq4XsGemm,
            iq2Codebooks,
            matmulIq2Xxs, matmulIq2XxsGemm,
            matmulIq2Xs, matmulIq2XsGemm,
            matmulIq2S, matmulIq2SGemm,
            matmulIq1S, matmulIq1SGemm,
            matmulF16, matmulF16Gemm, matmulF16GemmCoopmat,
            matmulBf16, matmulBf16Gemm,
            rmsnorm, rope, attention, swiglu, add, silu, conv1d,
            gdnL2, gdnScan, gdnScanMulti, gdnPost,
            gdnDecay, sigmoidInplace,
            sigGateMul,
            moeTopk, moeBroadcast, moeIndexed, moeScatter, moeSigmoidGatedAdd);
    }

    /// <summary>Invalidates every kernel's cached descriptor sets. Call after scratch buffers re-allocate.</summary>
    public void InvalidateAll()
    {
        MatMul.InvalidateDescriptorCache();
        MatMulQ8.InvalidateDescriptorCache();
        MatMulQ8Gemm.InvalidateDescriptorCache();
        MatMulQ8GemmCoopmat?.InvalidateDescriptorCache();
        MatMulQ2K.InvalidateDescriptorCache();
        MatMulQ2KGemm.InvalidateDescriptorCache();
        MatMulQ3K.InvalidateDescriptorCache();
        MatMulQ3KGemm.InvalidateDescriptorCache();
        MatMulQ4K.InvalidateDescriptorCache();
        MatMulQ4KGemm.InvalidateDescriptorCache();
        MatMulQ5K.InvalidateDescriptorCache();
        MatMulQ5KGemm.InvalidateDescriptorCache();
        MatMulQ6K.InvalidateDescriptorCache();
        MatMulQ6KGemm.InvalidateDescriptorCache();
        MatMulIq4Nl.InvalidateDescriptorCache();
        MatMulIq4NlGemm.InvalidateDescriptorCache();
        MatMulIq4Xs.InvalidateDescriptorCache();
        MatMulIq4XsGemm.InvalidateDescriptorCache();
        MatMulIq2Xxs.InvalidateDescriptorCache();
        MatMulIq2XxsGemm.InvalidateDescriptorCache();
        MatMulIq2Xs.InvalidateDescriptorCache();
        MatMulIq2XsGemm.InvalidateDescriptorCache();
        MatMulIq2S.InvalidateDescriptorCache();
        MatMulIq2SGemm.InvalidateDescriptorCache();
        MatMulIq1S.InvalidateDescriptorCache();
        MatMulIq1SGemm.InvalidateDescriptorCache();
        MatMulF16.InvalidateDescriptorCache();
        MatMulF16Gemm.InvalidateDescriptorCache();
        MatMulF16GemmCoopmat?.InvalidateDescriptorCache();
        MatMulBf16.InvalidateDescriptorCache();
        MatMulBf16Gemm.InvalidateDescriptorCache();
        RmsNorm.InvalidateDescriptorCache();
        Rope.InvalidateDescriptorCache();
        Attention.InvalidateDescriptorCache();
        SwiGlu.InvalidateDescriptorCache();
        Add.InvalidateDescriptorCache();
        SiluInplace.InvalidateDescriptorCache();
        Conv1dCausal.InvalidateDescriptorCache();
        GdnL2Normalize.InvalidateDescriptorCache();
        GdnScanStep.InvalidateDescriptorCache();
        GdnScanMultiToken.InvalidateDescriptorCache();
        GdnPostScanGate.InvalidateDescriptorCache();
        GdnDecay.InvalidateDescriptorCache();
        SigmoidInplace.InvalidateDescriptorCache();
        SigmoidGateMul.InvalidateDescriptorCache();
        MoeTopkSoftmax.InvalidateDescriptorCache();
        MoeBroadcast.InvalidateDescriptorCache();
        MoeIndexedMatmul.InvalidateDescriptorCache();
        MoeWeightedScatter.InvalidateDescriptorCache();
        MoeSigmoidGatedAdd.InvalidateDescriptorCache();
    }

    public void Dispose()
    {
        MoeSigmoidGatedAdd.Dispose();
        MoeWeightedScatter.Dispose();
        MoeIndexedMatmul.Dispose();
        MoeBroadcast.Dispose();
        MoeTopkSoftmax.Dispose();
        SigmoidGateMul.Dispose();
        SigmoidInplace.Dispose();
        GdnDecay.Dispose();
        GdnPostScanGate.Dispose();
        GdnScanMultiToken.Dispose();
        GdnScanStep.Dispose();
        GdnL2Normalize.Dispose();
        Conv1dCausal.Dispose();
        SiluInplace.Dispose();
        Add.Dispose();
        SwiGlu.Dispose();
        Attention.Dispose();
        Rope.Dispose();
        RmsNorm.Dispose();
        MatMulBf16Gemm.Dispose();
        MatMulBf16.Dispose();
        MatMulF16GemmCoopmat?.Dispose();
        MatMulF16Gemm.Dispose();
        MatMulF16.Dispose();
        MatMulIq1SGemm.Dispose();
        MatMulIq1S.Dispose();
        MatMulIq4XsGemm.Dispose();
        MatMulIq4Xs.Dispose();
        MatMulIq4NlGemm.Dispose();
        MatMulIq4Nl.Dispose();
        MatMulIq2SGemm.Dispose();
        MatMulIq2S.Dispose();
        MatMulIq2XsGemm.Dispose();
        MatMulIq2Xs.Dispose();
        MatMulIq2XxsGemm.Dispose();
        MatMulIq2Xxs.Dispose();
        Iq2Codebooks.Dispose();
        MatMulQ6KGemm.Dispose();
        MatMulQ6K.Dispose();
        MatMulQ5KGemm.Dispose();
        MatMulQ5K.Dispose();
        MatMulQ4KGemm.Dispose();
        MatMulQ4K.Dispose();
        MatMulQ3KGemm.Dispose();
        MatMulQ3K.Dispose();
        MatMulQ2KGemm.Dispose();
        MatMulQ2K.Dispose();
        MatMulQ8GemmCoopmat?.Dispose();
        MatMulQ8Gemm.Dispose();
        MatMulQ8.Dispose();
        MatMul.Dispose();
    }
}
