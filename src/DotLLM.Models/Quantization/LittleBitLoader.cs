using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels.Experimental;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Quantization;

/// <summary>
/// Loads LittleBit factorized linears from a safetensors export (key layout per the clean-room spec, section 1.3):
/// <c>{prefix}U_packed / V_packed / u1 / u2 / v1 / v2</c> plus the <c>_R</c> residual set. Scales are bf16 in the
/// export; they are widened to F32 exactly. The latent scale is <c>l = bf16(v1 * u2)</c>, rounded once to bf16 exactly
/// as the reference forward multiplies the two bf16 vectors, so the factorized product equals the reference's.
/// </summary>
public static class LittleBitLoader
{
    /// <summary>True when <paramref name="prefix"/> (ending in '.') names a LittleBit linear in the file.</summary>
    public static bool HasLayer(ISafetensorsTensorSource f, string prefix) => f.TensorsByName.ContainsKey(prefix + "U_packed");

    /// <summary>Loads the linear at <paramref name="prefix"/> (e.g. <c>model.layers.3.mlp.gate_proj.</c>).</summary>
    public static LittleBitLayer LoadLayer(ISafetensorsTensorSource f, string prefix)
    {
        var paths = new List<LittleBitPath> { LoadPath(f, prefix, "") };
        if (f.TensorsByName.ContainsKey(prefix + "U_R_packed")) paths.Add(LoadPath(f, prefix, "_R"));
        return new LittleBitLayer(paths.ToArray());
    }

    private static LittleBitPath LoadPath(ISafetensorsTensorSource f, string prefix, string sfx)
    {
        var us = I64(f, $"{prefix}U{sfx}_shape");
        var vs = I64(f, $"{prefix}V{sfx}_shape");
        if (us.Length != 2 || vs.Length != 2 || us[1] != vs[0])
            throw new InvalidDataException($"{prefix}{sfx}: inconsistent U/V shapes [{string.Join(',', us)}] [{string.Join(',', vs)}]");
        int dOut = checked((int)us[0]), r = checked((int)us[1]), dIn = checked((int)vs[1]);

        var uw = Words(f, $"{prefix}U{sfx}_packed", dOut, (r + 31) / 32);
        var vw = Words(f, $"{prefix}V{sfx}_packed", r, (dIn + 31) / 32);
        var u1 = Bf16(f, $"{prefix}u1{sfx}", dOut);
        var v2 = Bf16(f, $"{prefix}v2{sfx}", dIn);
        var v1 = Bf16(f, $"{prefix}v1{sfx}", r);
        var u2 = Bf16(f, $"{prefix}u2{sfx}", r);
        var l = new float[r];
        for (int j = 0; j < r; j++) l[j] = RoundToBf16(v1[j] * u2[j]);
        return LittleBitPath.FromPackedWords(dOut, dIn, r, uw, vw, u1, v2, l);
    }

    private static unsafe ReadOnlySpan<byte> Span(ISafetensorsTensorSource f, string name)
    {
        var d = f.TensorsByName[name];
        return new ReadOnlySpan<byte>((void*)f.GetTensorPointer(name), checked((int)d.ByteCount));
    }

    private static long[] I64(ISafetensorsTensorSource f, string name)
        => MemoryMarshal.Cast<byte, long>(Span(f, name)).ToArray();

    private static ReadOnlySpan<int> Words(ISafetensorsTensorSource f, string name, int rows, int words)
    {
        var d = f.TensorsByName[name];
        if (d.DType != SafetensorsDType.I32 || d.Shape.Length != 2 || d.Shape[0] != rows || d.Shape[1] != words)
            throw new InvalidDataException($"{name}: expected I32 [{rows},{words}], got {d.DType} [{string.Join(',', d.Shape)}]");
        return MemoryMarshal.Cast<byte, int>(Span(f, name));
    }

    /// <summary>Reads a bf16 vector tensor (any leading 1 dims) as exact F32.</summary>
    public static float[] Bf16(ISafetensorsTensorSource f, string name, int expectedLength)
    {
        var d = f.TensorsByName[name];
        if (d.DType != SafetensorsDType.BF16 || d.ElementCount != expectedLength)
            throw new InvalidDataException($"{name}: expected BF16 x{expectedLength}, got {d.DType} x{d.ElementCount}");
        var src = MemoryMarshal.Cast<byte, ushort>(Span(f, name));
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
        return dst;
    }

    /// <summary>Round-to-nearest-even float32 -> bf16 -> float32 (finite inputs).</summary>
    public static float RoundToBf16(float v)
    {
        uint b = BitConverter.SingleToUInt32Bits(v);
        if ((b & 0x7F800000u) == 0x7F800000u) return v;
        b += 0x7FFFu + ((b >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(b & 0xFFFF0000u);
    }
}
