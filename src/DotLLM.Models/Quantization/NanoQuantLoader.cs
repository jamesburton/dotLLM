using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Kernels.Experimental;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Quantization;

/// <summary>
/// Reads the community NanoQuant GGUF layout (<c>quantization.nanoquant.version = 2</c>, issue #866): every quantised
/// linear <c>&lt;base&gt;</c> is a composite of <c>&lt;base&gt;.nq_u / nq_v</c> (ggml I32 sign words) and
/// <c>nq_scale_pre / nq_scale_mid / nq_scale_post</c> (BF16, F16 or F32), plus an optional salient-column side path
/// (<c>nq_salient_idx</c> I32, <c>nq_salient_weight</c> F16/BF16/F32). GGUF dims are listed fastest-first, so
/// <c>nq_u = [ceil(r/32), d_out]</c>, <c>nq_v = [ceil(d_in/32), r]</c>, <c>nq_salient_weight = [k, d_out]</c>.
/// The embeddings / output head / norms of such a file are ordinary tensors and are not handled here.
/// </summary>
public static class NanoQuantLoader
{
    /// <summary>Metadata key that marks a NanoQuant GGUF.</summary>
    public const string VersionKey = "quantization.nanoquant.version";

    /// <summary>True when the file declares NanoQuant tensors.</summary>
    public static bool IsNanoQuant(GgufFile file) => file.Metadata.ContainsKey(VersionKey);

    /// <summary>
    /// Checks the layout declaration: version 2, LSB-first bit order, positive bit 0 (clear bit = +1) and the column
    /// side-path outlier format. Anything else would decode to a silently different matrix, so it is rejected.
    /// </summary>
    /// <exception cref="NotSupportedException">The file declares a layout this reader does not implement.</exception>
    public static void ValidateMetadata(GgufMetadata md)
    {
        long version = Int(md, VersionKey, required: true);
        if (version != 2) throw new NotSupportedException($"{VersionKey} = {version}; only version 2 is implemented.");
        string order = md.GetStringOrDefault("quantization.nanoquant.bit_order", "lsb_first");
        if (order != "lsb_first") throw new NotSupportedException($"NanoQuant bit_order '{order}' is not supported (lsb_first only).");
        if (Int(md, "quantization.nanoquant.positive_bit", required: false) is long pb && pb != 0)
            throw new NotSupportedException($"NanoQuant positive_bit = {pb} is not supported (0 only: a clear bit is +1).");
        string outlier = md.GetStringOrDefault("quantization.nanoquant.outlier_format", "column_side_path");
        if (outlier != "column_side_path") throw new NotSupportedException($"NanoQuant outlier_format '{outlier}' is not supported.");
    }

    private static long Int(GgufMetadata md, string key, bool required)
    {
        if (!md.TryGetValue(key, out var v))
            return required ? throw new KeyNotFoundException(key) : 0;
        return Convert.ToInt64(v.Value);
    }

    /// <summary>Base names (e.g. <c>blk.0.ffn_gate</c>) of every NanoQuant composite in the file, in file order.</summary>
    public static IReadOnlyList<string> FindBases(GgufFile file)
    {
        const string suffix = ".nq_u";
        return file.Tensors.Where(t => t.Name.EndsWith(suffix, StringComparison.Ordinal))
                           .Select(t => t.Name[..^suffix.Length]).ToList();
    }

    /// <summary>Loads one composite as a <see cref="NanoQuantLayer"/>.</summary>
    /// <exception cref="InvalidDataException">Missing or inconsistent tensors.</exception>
    /// <exception cref="NotSupportedException">A tensor has a dtype this reader does not accept.</exception>
    public static unsafe NanoQuantLayer Load(GgufFile file, string baseName)
    {
        ValidateMetadata(file.Metadata);
        var u = Get(file, baseName, "nq_u", 2, QuantizationType.I32);
        var v = Get(file, baseName, "nq_v", 2, QuantizationType.I32);
        var pre = GetScale(file, baseName, "nq_scale_pre");
        var mid = GetScale(file, baseName, "nq_scale_mid");
        var post = GetScale(file, baseName, "nq_scale_post");

        int dOut = u.Shape[1], r = v.Shape[1], dIn = pre.Shape[0];
        if (u.Shape[0] != (r + 31) / 32) throw new InvalidDataException($"{baseName}.nq_u has {u.Shape[0]} words/row, rank {r} needs {(r + 31) / 32}.");
        if (v.Shape[0] != (dIn + 31) / 32) throw new InvalidDataException($"{baseName}.nq_v has {v.Shape[0]} words/row, d_in {dIn} needs {(dIn + 31) / 32}.");
        if (mid.Shape[0] != r) throw new InvalidDataException($"{baseName}.nq_scale_mid length {mid.Shape[0]} != rank {r}.");
        if (post.Shape[0] != dOut) throw new InvalidDataException($"{baseName}.nq_scale_post length {post.Shape[0]} != d_out {dOut}.");

        int[] uw = ReadI32(file, u), vw = ReadI32(file, v);
        int[] idx = [];
        float[] sw = [];
        bool hasIdx = file.TensorsByName.ContainsKey(baseName + ".nq_salient_idx");
        bool hasW = file.TensorsByName.ContainsKey(baseName + ".nq_salient_weight");
        if (hasIdx != hasW) throw new InvalidDataException($"{baseName}: salient idx and weight must both be present or both absent.");
        if (file.TensorsByName.ContainsKey(baseName + ".nq_salient_scale"))
            throw new NotSupportedException($"{baseName}.nq_salient_scale (I8 salient weights) is not implemented.");
        if (hasIdx)
        {
            var si = Get(file, baseName, "nq_salient_idx", 1, QuantizationType.I32);
            var swt = Get(file, baseName, "nq_salient_weight", 2, null);
            if (swt.QuantizationType is not (QuantizationType.F16 or QuantizationType.BF16 or QuantizationType.F32))
                throw new NotSupportedException($"{swt.Name} has type {swt.QuantizationType}; F16/BF16/F32 expected.");
            if (swt.Shape[0] != si.Shape[0] || swt.Shape[1] != dOut)
                throw new InvalidDataException($"{swt.Name} shape [{swt.Shape[0]},{swt.Shape[1]}] != [k={si.Shape[0]}, d_out={dOut}].");
            idx = ReadI32(file, si);
            sw = ReadFloats(file, swt);
        }

        return NanoQuantLayer.FromPacked(dOut, dIn, r, uw, vw, ReadFloats(file, pre), ReadFloats(file, mid), ReadFloats(file, post), idx, sw);
    }

    private static GgufTensorDescriptor Get(GgufFile f, string b, string role, int rank, QuantizationType? type)
    {
        if (!f.TensorsByName.TryGetValue(b + "." + role, out var t)) throw new InvalidDataException($"missing tensor '{b}.{role}'.");
        if (t.Shape.Rank != rank) throw new InvalidDataException($"{t.Name} must be {rank}-D, got {t.Shape.Rank}-D.");
        if (type is { } q && t.QuantizationType != q) throw new NotSupportedException($"{t.Name} has type {t.QuantizationType}; {q} expected.");
        return t;
    }

    private static GgufTensorDescriptor GetScale(GgufFile f, string b, string role)
    {
        var t = Get(f, b, role, 1, null);
        if (t.QuantizationType is not (QuantizationType.F16 or QuantizationType.BF16 or QuantizationType.F32))
            throw new NotSupportedException($"{t.Name} has type {t.QuantizationType}; F16/BF16/F32 expected.");
        return t;
    }

    private static unsafe int[] ReadI32(GgufFile f, GgufTensorDescriptor t)
    {
        var a = new int[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<int>((void*)f.TensorDataPointer(t), a.Length).CopyTo(a);   // little-endian host == file order
        return a;
    }

    private static float[] ReadFloats(GgufFile f, GgufTensorDescriptor t)
    {
        var a = new float[checked((int)t.Shape.ElementCount)];
        Dequantize.ToFloat32(f.TensorDataPointer(t), a.Length, t.QuantizationType, a);
        return a;
    }
}
