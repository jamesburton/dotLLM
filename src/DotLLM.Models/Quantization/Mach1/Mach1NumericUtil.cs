// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Small numeric helpers shared by the Mach-1 decode primitives.
/// </summary>
public static class Mach1NumericUtil
{
    /// <summary>
    /// Rounds every element through <see cref="Half"/> and back to
    /// <see cref="float"/> in place. Mirrors decode.py's explicit
    /// <c>unit.astype(np.float16).astype(np.float32)</c> round-trip, which the
    /// codec's own bit-exactness gate depends on: skipping this step (treating
    /// it as a no-op) produces values that are numerically close but not
    /// bit-identical to the vendor's golden tensors.
    /// </summary>
    public static void RoundTripThroughHalf(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(Half)data[i];
    }
}
