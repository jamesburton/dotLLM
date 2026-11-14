using DotLLM.Core.Configuration;
using System.Runtime.InteropServices;

namespace DotLLM.Core.Models;

/// <summary>
/// CPU-only view of a validated HuggingFace AWQ/GPTQ projection. This is a descriptor,
/// not a GGUF block: all four pointers borrow the original SafeTensors mappings.
/// The owner keeps those mappings alive and frees only this 64-byte descriptor.
/// </summary>
[StructLayout(LayoutKind.Sequential, Size = 64)]
public unsafe struct Packed4Weight
{
    /// <summary>Mapped int32 packed codes, in the declared producer layout.</summary>
    public uint* QWeight;
    /// <summary>Mapped F16 scales [groups, output features].</summary>
    public Half* Scales;
    /// <summary>Mapped int32 zero points, eight output lanes per word.</summary>
    public uint* QZeros;
    /// <summary>Mapped GPTQ input-to-group indices; null for AWQ.</summary>
    public int* GroupIndices;
    /// <summary>Logical input dimension K.</summary>
    public int InputFeatures;
    /// <summary>Logical output dimension M.</summary>
    public int OutputFeatures;
    /// <summary>Number of input elements per scale/zero group.</summary>
    public int GroupSize;
    /// <summary>AWQ GEMM4 or legacy GPTQ4 descriptor kind.</summary>
    public QuantizationType Layout;
    /// <summary>Total mapped bytes of all components; excludes this descriptor.</summary>
    public long MappedBytes;
}
