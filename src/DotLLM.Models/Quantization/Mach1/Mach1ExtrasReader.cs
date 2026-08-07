// Loader glue for issue #266 Phase B (load path).
using System.Runtime.InteropServices;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Reads the Mach-1 checkpoint's <c>extras.safetensors</c> — the small
/// (~52 MB, "bf16 keep + norm overrides" per its own <c>__metadata__.note</c>)
/// sidecar of tensors the additive codec does NOT quantize: per-layer RMSNorm
/// gains, GDN scalars (<c>A_log</c>, <c>dt_bias</c>, <c>conv1d.weight</c>,
/// the alpha/beta input projections), the MoE router gate, the shared-expert
/// sigmoid gate, and QK-norm for full-attention layers. These never go
/// through the trellis/RHT codec — Phase A has no decoder for them because
/// none is needed; this reader just upcasts bf16 -&gt; fp32.
/// </summary>
internal sealed class Mach1ExtrasReader : IDisposable
{
    private readonly SafetensorsFile _file;

    private Mach1ExtrasReader(SafetensorsFile file) => _file = file;

    public static Mach1ExtrasReader Open(string checkpointRoot)
    {
        string path = Path.Combine(checkpointRoot, "extras.safetensors");
        return new Mach1ExtrasReader(SafetensorsFile.Open(path));
    }

    public bool Contains(string name) => _file.TensorsByName.ContainsKey(name);

    /// <summary>
    /// Reads a tensor of any rank (flattened) to a new managed <c>float[]</c>
    /// of exactly <paramref name="expectedElementCount"/> elements, upcasting
    /// bf16/f16 as needed. Throws if the tensor is absent or its element
    /// count does not match.
    /// </summary>
    public unsafe float[] ReadF32(string name, int expectedElementCount)
    {
        if (!_file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException($"extras.safetensors is missing required tensor '{name}'.");

        long count = 1;
        foreach (int d in desc.Shape) count *= d;
        if (count != expectedElementCount)
            throw new InvalidDataException(
                $"extras.safetensors tensor '{name}' has {count} elements, expected {expectedElementCount}.");

        var result = new float[expectedElementCount];
        var bytes = _file.GetTensorSpan(name);
        switch (desc.DType)
        {
            case SafetensorsDType.F32:
                MemoryMarshal.Cast<byte, float>(bytes).CopyTo(result);
                break;
            case SafetensorsDType.F16:
            {
                var half = MemoryMarshal.Cast<byte, Half>(bytes);
                for (int i = 0; i < expectedElementCount; i++)
                    result[i] = (float)half[i];
                break;
            }
            case SafetensorsDType.BF16:
            {
                fixed (byte* src = bytes)
                    SafetensorsTensorResolver.DecodeBf16((ushort*)src, expectedElementCount, result);
                break;
            }
            default:
                throw new NotSupportedException(
                    $"extras.safetensors tensor '{name}' has unsupported dtype {desc.DType}.");
        }
        return result;
    }

    public void Dispose() => _file.Dispose();
}
