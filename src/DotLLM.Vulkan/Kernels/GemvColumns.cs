using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Dispatch shape of the column-indexed decode GEMVs (#1237): <c>matmul_q8_0[_coalesced]</c>, <c>matmul_nvfp4_*gemv*</c>,
/// <c>matmul_iq2_xxs_f32_gemv</c> and <c>matmul_f32_gemv</c> take a <c>cols</c> push constant, and workgroup
/// <c>w = x + 32768 · y</c> computes weight row(s) <c>w / cols</c> of activation column <c>w % cols</c> with the one-column
/// arithmetic. A K+1-row speculative verify therefore reads each weight row once (its <c>cols</c> workgroups are adjacent and share
/// it through the cache) and every row is bit-identical to a decode step, which dispatches <c>cols = 1</c> through the same pipeline.
/// </summary>
internal static class GemvColumns
{
    /// <summary>Workgroups per dispatch row; the shaders fold <c>gl_WorkGroupID.y</c> in with this stride.</summary>
    public const uint Stride = 32768;

    /// <summary>Largest column count a caller should ask for (a verify of a block of 16).</summary>
    public const int MaxColumns = 16;

    /// <summary>Dispatches <paramref name="groups"/> workgroups as <c>(min(groups, Stride), ceil(groups / Stride))</c>.</summary>
    /// <param name="cmd">Command buffer.</param>
    /// <param name="groups">Workgroups (row groups x columns).</param>
    public static void Dispatch(nint cmd, long groups)
    {
        if (groups <= 0) throw new ArgumentOutOfRangeException(nameof(groups));
        uint x = (uint)Math.Min(groups, Stride);
        uint y = (uint)((groups + Stride - 1) / Stride);
        VulkanApi.vkCmdDispatch(cmd, x, y, 1);
    }
}
