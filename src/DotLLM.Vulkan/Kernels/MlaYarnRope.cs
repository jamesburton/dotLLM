using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// YaRN RoPE parameters for the MLA rope kernels (#1278). DeepSeek-V2/V3 rotate the decoupled rope slice with YaRN-ramped inverse
/// frequencies (<c>inv_freq = inter * ramp + extra * (1 - ramp)</c>) and multiply cos/sin by <c>mscale / mscale_all_dim</c>; the CPU
/// path builds that table in <c>TransformerModel</c>. The Vulkan MLA kernels used plain <c>1 / theta^(2i/d)</c>, so every DeepSeek-V2
/// logit was off by a broad shift (L-inf 4.8 vs the CPU on V2-Lite Q4_K_M). Evaluated in closed form from four push constants:
/// <see cref="Factor"/> &lt;= 1 selects the legacy plain-theta path bit-identically.
/// </summary>
/// <param name="Factor">YaRN scaling factor; &lt;= 1 disables YaRN.</param>
/// <param name="Low">Ramp start (half-dim index).</param>
/// <param name="High">Ramp end (half-dim index).</param>
/// <param name="Mscale">cos/sin multiplier.</param>
public readonly record struct MlaYarnRope(float Factor, float Low, float High, float Mscale)
{
    /// <summary>YaRN off (plain-theta RoPE).</summary>
    public static MlaYarnRope Off => new(1.0f, 0.0f, 1.0f, 1.0f);

    /// <summary>Resolves the parameters with exactly the CPU condition and formulas (TransformerModel MLA + YaRN branch).</summary>
    public static MlaYarnRope From(MlaConfig? mla, int ropeDim, float theta)
    {
        if (mla is not { RopeScalingFactor: float factor } || factor <= 1.0f
            || mla.RopeScalingOriginalMaxPositionEmbeddings is not int origMax || origMax <= 0)
            return Off;
        float mscaleNum = (mla.RopeScalingMscale is float m && m != 0.0f) ? 0.1f * m * MathF.Log(factor) + 1.0f : 1.0f;
        float mscaleDen = (mla.RopeScalingMscaleAllDim is float mad && mad != 0.0f) ? 0.1f * mad * MathF.Log(factor) + 1.0f : 1.0f;
        (float low, float high) = RoPE.YarnRampBounds(ropeDim, theta, origMax, mla.RopeScalingBetaFast, mla.RopeScalingBetaSlow);
        return new MlaYarnRope(factor, low, high, mscaleNum / mscaleDen);
    }
}
