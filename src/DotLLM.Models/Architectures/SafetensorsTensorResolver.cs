using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Low-level tensor-name resolution primitives shared by the per-layer
/// safetensors loaders (<see cref="TransformerWeightsSafetensorsLoader"/>,
/// <see cref="AttentionTensorLoader"/>).
/// </summary>
/// <remarks>
/// <para>
/// These helpers were previously private static methods on
/// <see cref="TransformerWeightsSafetensorsLoader"/>. Promoted to
/// <c>internal static</c> on a separate type so that sibling loaders
/// (attention, MoE, eventually FFN) can share the same BF16 / F16 / F32
/// resolution + validation + owned-allocation bookkeeping without copying
/// the logic. No behavioral change — callers continue to observe exactly
/// the same error messages, allocation counts, and tensor-pointer
/// semantics as before.
/// </para>
/// </remarks>
internal static class SafetensorsTensorResolver
{
    /// <summary>
    /// Resolves a safetensors tensor as a linear projection weight:
    /// HF shape <c>[out_features, in_features]</c> → (ptr, dtype, M, K).
    /// F32 tensors are zero-copy; BF16 tensors are upcast into an owned
    /// 64-byte-aligned scratch buffer and registered in
    /// <paramref name="owned"/>.
    /// </summary>
    public static unsafe (nint ptr, QuantizationType qt, int m, int k) ResolveLinear(
        ISafetensorsTensorSource file, string name, List<nint> owned)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException(
                $"Safetensors file is missing required tensor '{name}'.");

        if (desc.Shape.Length != 2)
            throw new InvalidDataException(
                $"Tensor '{name}' expected to be rank-2, got rank {desc.Shape.Length}.");

        int m = desc.Shape[0];
        int k = desc.Shape[1];

        nint srcPtr = file.GetTensorPointer(name);

        switch (desc.DType)
        {
            case SafetensorsDType.F32:
                return (srcPtr, QuantizationType.F32, m, k);

            case SafetensorsDType.BF16:
            {
                long elementCount = (long)m * k;
                nint dst = AllocBf16ToF32(srcPtr, elementCount);
                owned.Add(dst);
                return (dst, QuantizationType.F32, m, k);
            }

            case SafetensorsDType.F16:
            {
                // Keep as F16 (kernels support it directly). No copy.
                return (srcPtr, QuantizationType.F16, m, k);
            }

            default:
                throw new NotSupportedException(
                    $"Tensor '{name}' has dtype {desc.DType} which is not yet supported by the safetensors transformer loader (F32/F16/BF16 only).");
        }
    }

    /// <summary>
    /// Resolves a rank-2 projection weight as an F32 pointer. F32 tensors
    /// are returned zero-copy; F16 and BF16 tensors are upcast into
    /// 64-byte-aligned owned scratch and registered in
    /// <paramref name="owned"/>. Similar to <see cref="ResolveLinear"/>
    /// but always hands back F32 — MoE / MLA kernels expect F32 today
    /// (per-expert / per-head quantised GEMM is a follow-up).
    /// </summary>
    public static unsafe (nint ptr, QuantizationType qt, int m, int k) ResolveLinearAsF32(
        ISafetensorsTensorSource file, string name, List<nint> owned)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException($"Safetensors file is missing required tensor '{name}'.");
        if (desc.Shape.Length != 2)
            throw new InvalidDataException($"Tensor '{name}' expected to be rank-2, got rank {desc.Shape.Length}.");

        int m = desc.Shape[0], k = desc.Shape[1];
        long count = (long)m * k;
        nint srcPtr = file.GetTensorPointer(name);

        switch (desc.DType)
        {
            case SafetensorsDType.F32:
                return (srcPtr, QuantizationType.F32, m, k);

            case SafetensorsDType.BF16:
            {
                nint dst = AllocBf16ToF32(srcPtr, count);
                owned.Add(dst);
                return (dst, QuantizationType.F32, m, k);
            }

            case SafetensorsDType.F16:
            {
                nuint byteCount = checked((nuint)count * sizeof(float));
                nint dst = (nint)NativeMemory.AlignedAlloc(byteCount, 64);
                owned.Add(dst);
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>((void*)srcPtr, (int)count),
                    new Span<float>((void*)dst, (int)count));
                return (dst, QuantizationType.F32, m, k);
            }

            default:
                throw new NotSupportedException(
                    $"Tensor '{name}' has dtype {desc.DType} — MoE loader supports F32/F16/BF16 only.");
        }
    }

    /// <summary>
    /// Resolves a rank-2 tensor as a managed <c>float[]</c>, up-casting
    /// F16 / BF16 on the way in. Used for small weights (router gate)
    /// where a copy costs nothing and is simpler than tracking owned
    /// allocations.
    /// </summary>
    public static unsafe float[] ResolveDense2D(
        ISafetensorsTensorSource file, string name, int expectedM, int expectedK)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException($"Safetensors file is missing required tensor '{name}'.");
        if (desc.Shape.Length != 2)
            throw new InvalidDataException($"Tensor '{name}' expected to be rank-2, got rank {desc.Shape.Length}.");
        int m = desc.Shape[0], k = desc.Shape[1];
        if (m != expectedM || k != expectedK)
            throw new InvalidDataException(
                $"Tensor '{name}' shape [{m},{k}] does not match expected [{expectedM},{expectedK}].");

        int count = m * k;
        var result = new float[count];
        nint src = file.GetTensorPointer(name);
        DecodeFloatTensor(src, desc.DType, count, result, name);
        return result;
    }

    /// <summary>
    /// Splits a row-fused HF tensor of shape <c>[sum(partRows), hidden]</c>
    /// into <paramref name="partRows"/>.Length independent F32 allocations
    /// (one per part) and returns their pointers in order. Supports
    /// F32 / BF16 / F16 source dtypes; BF16 is upcast to F32, F16 is
    /// decoded to F32 as well to keep the downstream kernel uniform
    /// (quantised splits would force per-expert dequant, out of scope).
    /// </summary>
    /// <remarks>
    /// Used by the Phi-3 attention loader path to split
    /// <c>self_attn.qkv_proj.weight</c> into Q/K/V and by the dense-FFN
    /// path to split <c>mlp.gate_up_proj.weight</c> into gate/up. Each
    /// allocation is 64-byte-aligned and registered in
    /// <paramref name="owned"/> for the caller's Dispose unwind.
    /// </remarks>
    public static unsafe void SplitFusedProjection(
        ISafetensorsTensorSource file,
        string name,
        int[] partRows,
        int expectedCols,
        List<nint> owned,
        out nint[] partPtrs)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException($"Safetensors file is missing required tensor '{name}'.");
        if (desc.Shape.Length != 2)
            throw new InvalidDataException(
                $"Tensor '{name}' expected to be rank-2, got rank {desc.Shape.Length}.");

        int totalRows = 0;
        for (int i = 0; i < partRows.Length; i++) totalRows += partRows[i];
        int m = desc.Shape[0], k = desc.Shape[1];
        if (m != totalRows || k != expectedCols)
            throw new InvalidDataException(
                $"Tensor '{name}' shape [{m},{k}] does not match expected fused shape [{totalRows},{expectedCols}].");

        nint srcPtr = file.GetTensorPointer(name);
        partPtrs = new nint[partRows.Length];
        int rowCursor = 0;
        for (int i = 0; i < partRows.Length; i++)
        {
            int rows = partRows[i];
            long partCount = (long)rows * expectedCols;
            nuint byteCount = checked((nuint)partCount * sizeof(float));
            nint dst = (nint)NativeMemory.AlignedAlloc(byteCount, 64);
            owned.Add(dst);

            switch (desc.DType)
            {
                case SafetensorsDType.F32:
                {
                    float* srcRow = (float*)srcPtr + (long)rowCursor * expectedCols;
                    new ReadOnlySpan<float>(srcRow, (int)partCount)
                        .CopyTo(new Span<float>((void*)dst, (int)partCount));
                    break;
                }
                case SafetensorsDType.BF16:
                {
                    ushort* srcRow = (ushort*)srcPtr + (long)rowCursor * expectedCols;
                    DecodeBf16(srcRow, (int)partCount, new Span<float>((void*)dst, (int)partCount));
                    break;
                }
                case SafetensorsDType.F16:
                {
                    Half* srcRow = (Half*)srcPtr + (long)rowCursor * expectedCols;
                    System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                        new ReadOnlySpan<Half>(srcRow, (int)partCount),
                        new Span<float>((void*)dst, (int)partCount));
                    break;
                }
                default:
                    throw new NotSupportedException(
                        $"Tensor '{name}' has dtype {desc.DType} — fused-split path supports F32/F16/BF16 only.");
            }
            partPtrs[i] = dst;
            rowCursor += rows;
        }
    }

    /// <summary>
    /// Allocates a 64-byte-aligned F32 buffer of <paramref name="elements"/>
    /// elements, filling it from the fused source pointer starting at
    /// <paramref name="sourceElementOffset"/>. BF16 and F16 sources are
    /// upcast / decoded; F32 sources are copied. Used by the Granite-MoE
    /// fused-per-expert split path.
    /// </summary>
    public static unsafe nint AllocPartAsF32(
        nint source, SafetensorsDType dtype, long sourceElementOffset, long elements,
        List<nint> owned, string sourceName)
    {
        nuint byteCount = checked((nuint)elements * sizeof(float));
        nint dst = (nint)NativeMemory.AlignedAlloc(byteCount, 64);
        owned.Add(dst);

        switch (dtype)
        {
            case SafetensorsDType.F32:
            {
                float* srcRow = (float*)source + sourceElementOffset;
                new ReadOnlySpan<float>(srcRow, (int)elements)
                    .CopyTo(new Span<float>((void*)dst, (int)elements));
                break;
            }
            case SafetensorsDType.BF16:
            {
                ushort* srcRow = (ushort*)source + sourceElementOffset;
                DecodeBf16(srcRow, (int)elements, new Span<float>((void*)dst, (int)elements));
                break;
            }
            case SafetensorsDType.F16:
            {
                Half* srcRow = (Half*)source + sourceElementOffset;
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>(srcRow, (int)elements),
                    new Span<float>((void*)dst, (int)elements));
                break;
            }
            default:
                throw new NotSupportedException(
                    $"Tensor '{sourceName}' has dtype {dtype} — Granite-MoE split path supports F32/F16/BF16 only.");
        }
        return dst;
    }

    public static void ValidateProjectionShape(int actualM, int actualK, int expectedM, int expectedK, string name)
    {
        if (actualM != expectedM || actualK != expectedK)
            throw new InvalidDataException(
                $"{name} shape [M={actualM}, K={actualK}] does not match expected [M={expectedM}, K={expectedK}].");
    }

    /// <summary>
    /// Resolves a norm weight tensor into a managed <c>float[]</c>. Norms
    /// are small and read once per forward call, so the load-time copy has
    /// no measurable inference cost.
    /// </summary>
    public static float[] ResolveNorm(ISafetensorsTensorSource file, string name, int expectedSize)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException(
                $"Safetensors file is missing required tensor '{name}'.");

        long elementCount = desc.ElementCount;
        if (elementCount != expectedSize)
            throw new InvalidDataException(
                $"Tensor '{name}' has {elementCount} elements, expected {expectedSize}.");

        var result = new float[expectedSize];
        nint src = file.GetTensorPointer(name);
        DecodeFloatTensor(src, desc.DType, expectedSize, result, name);
        return result;
    }

    public static float[]? ResolveOptionalNorm(ISafetensorsTensorSource file, string name, int expectedSize)
    {
        if (!file.TensorsByName.ContainsKey(name)) return null;
        return ResolveNorm(file, name, expectedSize);
    }

    public static float[]? ResolveOptionalBias(ISafetensorsTensorSource file, string name, int expectedSize)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc)) return null;

        long elementCount = desc.ElementCount;
        if (elementCount != expectedSize)
            throw new InvalidDataException(
                $"Bias tensor '{name}' has {elementCount} elements, expected {expectedSize}.");

        var result = new float[expectedSize];
        nint src = file.GetTensorPointer(name);
        DecodeFloatTensor(src, desc.DType, expectedSize, result, name);
        return result;
    }

    public static unsafe void DecodeFloatTensor(
        nint src, SafetensorsDType dtype, int elementCount, float[] dest, string name)
    {
        switch (dtype)
        {
            case SafetensorsDType.F32:
                new ReadOnlySpan<float>((void*)src, elementCount).CopyTo(dest);
                break;
            case SafetensorsDType.F16:
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>((void*)src, elementCount), dest);
                break;
            case SafetensorsDType.BF16:
                DecodeBf16((ushort*)src, elementCount, dest);
                break;
            default:
                throw new NotSupportedException(
                    $"Tensor '{name}' has dtype {dtype} which is not supported for norm/bias load (F32/F16/BF16 only).");
        }
    }

    /// <summary>
    /// Upcasts a bf16 tensor to a 64-byte-aligned F32 buffer owned by the
    /// caller. bf16 is "the high 16 bits of an IEEE-754 binary32", so the
    /// upcast is a shift-left-by-16-bits reinterpret — identical to what
    /// llama.cpp does when it normalises HF checkpoints to F32.
    /// </summary>
    public static unsafe nint AllocBf16ToF32(nint srcBf16, long elementCount)
    {
        nuint byteCount = checked((nuint)elementCount * sizeof(float));
        nint dst = (nint)NativeMemory.AlignedAlloc(byteCount, 64);
        DecodeBf16((ushort*)srcBf16, (int)elementCount, new Span<float>((void*)dst, (int)elementCount));
        return dst;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void DecodeBf16(ushort* src, int count, Span<float> dest)
    {
        // bf16 → f32: shift the 16 bits into the high half of a 32-bit
        // word, then reinterpret as float. NaN/Inf bit patterns transfer
        // cleanly.
        fixed (float* dstPtr = dest)
        {
            uint* dw = (uint*)dstPtr;
            for (int i = 0; i < count; i++)
                dw[i] = (uint)src[i] << 16;
        }
    }
}
