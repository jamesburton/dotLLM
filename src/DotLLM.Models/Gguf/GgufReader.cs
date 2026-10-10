using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Static binary parser for the GGUF file format. Pure functions: bytes in, structs out.
/// Supports GGUF v2 and v3, which are identical on the wire: tensor/metadata counts,
/// string lengths and array lengths are all <c>uint64</c>. (The <c>uint32</c> form belongs to
/// the obsolete v1, which the header validation rejects.)
/// </summary>
public static class GgufReader
{
    /// <summary>GGUF magic number: "GGUF" in little-endian.</summary>
    public const uint GgufMagic = 0x46554747;

    /// <summary>
    /// Reads and validates the GGUF header from the current reader position.
    /// </summary>
    /// <param name="reader">A <see cref="BinaryReader"/> positioned at the start of a GGUF file.</param>
    /// <returns>The parsed header.</returns>
    /// <exception cref="InvalidDataException">Invalid magic number or unsupported version.</exception>
    public static GgufHeader ReadHeader(BinaryReader reader)
    {
        uint magic = reader.ReadUInt32();
        if (magic != GgufMagic)
            throw new InvalidDataException(
                $"Invalid GGUF magic number: 0x{magic:X8}. Expected 0x{GgufMagic:X8}.");

        uint version = reader.ReadUInt32();
        if (version is not (2 or 3))
            throw new InvalidDataException(
                $"Unsupported GGUF version: {version}. Only versions 2 and 3 are supported.");

        // GGUF v2 and v3 both store these counts as uint64 (the uint32 form was v1 only).
        ulong tensorCount = reader.ReadUInt64();
        ulong metadataKvCount = reader.ReadUInt64();

        return new GgufHeader(version, tensorCount, metadataKvCount);
    }

    /// <summary>
    /// Reads all metadata key-value pairs from the current reader position.
    /// </summary>
    /// <param name="reader">A <see cref="BinaryReader"/> positioned after the header.</param>
    /// <param name="header">The previously parsed header (provides version and KV count).</param>
    /// <returns>Dictionary of metadata entries keyed by name.</returns>
    public static Dictionary<string, GgufMetadataValue> ReadMetadata(BinaryReader reader, GgufHeader header)
    {
        var metadata = new Dictionary<string, GgufMetadataValue>((int)Math.Min(header.MetadataKvCount, int.MaxValue));

        for (ulong i = 0; i < header.MetadataKvCount; i++)
        {
            string key = ReadGgufString(reader, header.Version);
            var valueType = (GgufValueType)reader.ReadUInt32();
            object value = ReadMetadataValue(reader, header.Version, valueType);
            metadata[key] = new GgufMetadataValue(valueType, value);
        }

        return metadata;
    }

    /// <summary>
    /// PrismML's own ggml type id for <see cref="QuantizationType.PQ2_0"/>. Pristine Bonsai GGUFs —
    /// both <c>Ternary-Bonsai-27B</c> and <c>Ternary-Bonsai-2-27B</c> — declare <c>142</c>
    /// (<c>GGML_TYPE_PQ2_0</c> in the PrismML llama.cpp fork), not the <c>42</c> that
    /// <see cref="QuantizationType.PQ2_0"/> was originally derived from. The block layout is
    /// byte-identical either way (<c>fp16</c> scale then 32 code bytes per 128 weights), so this is
    /// purely an id alias; <c>42</c> stays recognized for locally patched artifacts that carry it.
    /// </summary>
    private const uint GgufTypePrismPq2_0 = 142;

    /// <summary>
    /// PrismML <c>GGML_TYPE_PTQ1_0</c> — dense base-3 trit packing at group 128 (28 bytes per 128
    /// weights, 1.75 bpw). Recognized only so that the failure is a clear diagnostic instead of a
    /// downstream size overflow; no kernel consumes it yet.
    /// </summary>
    private const uint GgufTypePrismPtq1_0 = 143;

    /// <summary>
    /// Maps a raw GGUF tensor type id onto a <see cref="QuantizationType"/>, translating the
    /// PrismML-private ids that are not mainline ggml types.
    /// </summary>
    /// <param name="rawType">The raw type id read from the tensor info entry.</param>
    /// <param name="name">Tensor name, used only for diagnostics.</param>
    /// <returns>The mapped quantization type.</returns>
    /// <exception cref="NotSupportedException">
    /// The id is unrecognized, or is a known-but-unimplemented format.
    /// </exception>
    private static QuantizationType MapGgufTensorType(uint rawType, string name)
    {
        if (rawType == GgufTypePrismPq2_0)
            return QuantizationType.PQ2_0;

        if (rawType == GgufTypePrismPtq1_0)
            throw new NotSupportedException(
                $"Tensor '{name}' is PTQ1_0 (PrismML ternary, GGUF type {rawType}), which dotLLM " +
                "does not implement yet. Use the PQ2_0 packing of this model instead.");

        if (!Enum.IsDefined(typeof(QuantizationType), (int)rawType))
            throw new NotSupportedException(
                $"Tensor '{name}' has unrecognized quantization type: {rawType}.");

        return (QuantizationType)rawType;
    }

    /// <summary>
    /// Resolves the GGUF type-id 42 collision in favour of upstream ggml <c>Q2_0</c> where the bytes say so (#823). A tensor read as
    /// <see cref="QuantizationType.PQ2_0"/> (id 42) whose on-disk extent does NOT fit the PQ2_0 byte count (the distance to the next tensor
    /// offset is at least one alignment larger) but DOES fit the upstream Q2_0 count (64-element blocks, 18 B each, 2.25 bpw vs 2.125) is
    /// re-labelled <see cref="QuantizationType.Q2_0"/>. Tensors that fit PQ2_0 (every pristine PrismML tensor, and any ambiguous tiny
    /// one) are left alone, and whatever still fits neither is rejected afterwards by <see cref="ValidatePq2_0Layout"/> with the existing
    /// message. Real example: the ISTA Qwen3.8-Flash-Next GSQ-RCO IQ2_XS / IQ3_XXS files store their expert down banks as id 42 at exactly
    /// 2.25 bits per weight.
    /// </summary>
    public static void ReclassifyUpstreamQ2_0(List<GgufTensorDescriptor> tensors, uint alignment, long dataSectionLength)
    {
        List<ulong>? offsets = null;
        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            if (t.QuantizationType != QuantizationType.PQ2_0) continue;
            long n = t.Shape.ElementCount;
            if (n % 64 != 0) continue;
            offsets ??= tensors.Select(x => x.DataOffset).Distinct().Order().ToList();
            int idx = offsets.BinarySearch(t.DataOffset);
            long end = idx + 1 < offsets.Count ? (long)offsets[idx + 1] : dataSectionLength;
            long span = end - (long)t.DataOffset;
            long pq = QuantizationType.PQ2_0.ComputeByteCount(n);
            long q2 = QuantizationType.Q2_0.ComputeByteCount(n);
            bool fitsPq = n % 128 == 0 && span >= pq && span < pq + alignment;
            bool fitsQ2 = span >= q2 && span < q2 + alignment;
            if (!fitsPq && fitsQ2)
                tensors[i] = t with { QuantizationType = QuantizationType.Q2_0 };
        }
    }

    /// <summary>
    /// Guards the GGUF type-id 42 collision. Upstream ggml's <c>Q2_0</c> is also type 42 but uses
    /// 64-element groups, whereas <see cref="QuantizationType.PQ2_0"/> uses 128-element groups
    /// (34 B/group); an upstream Q2_0 file would otherwise load and silently mis-decode. The tensor
    /// info carries no layout tag, so this checks what is observable: the element count must be a
    /// multiple of 128, and the on-disk extent (distance to the next tensor's offset, or to the end
    /// of the data section for the last one) must equal the PQ2_0 byte count plus at most
    /// <paramref name="alignment"/> bytes of padding. Pristine PrismML files use id 142 and always
    /// pass. All backend loaders (CPU/CUDA/Vulkan) consume tensors via <c>GgufFile.Open</c>, so this
    /// is the single choke point.
    /// </summary>
    /// <param name="tensors">Parsed tensor descriptors.</param>
    /// <param name="alignment">Data-section alignment (<c>general.alignment</c>).</param>
    /// <param name="dataSectionLength">Bytes from the data-section start to end of file.</param>
    /// <exception cref="NotSupportedException">A PQ2_0-typed tensor is not laid out as PQ2_0.</exception>
    public static void ValidatePq2_0Layout(IReadOnlyList<GgufTensorDescriptor> tensors, uint alignment, long dataSectionLength)
    {
        List<ulong>? offsets = null;
        foreach (var t in tensors)
        {
            if (t.QuantizationType != QuantizationType.PQ2_0) continue;

            long n = t.Shape.ElementCount;
            long pq = QuantizationType.PQ2_0.ComputeByteCount(n);
            string? why = null;
            if (n % 128 != 0)
            {
                why = $"element count {n} is not a multiple of 128";
            }
            else
            {
                offsets ??= tensors.Select(x => x.DataOffset).Distinct().Order().ToList();
                int idx = offsets.BinarySearch(t.DataOffset);
                long end = idx + 1 < offsets.Count ? (long)offsets[idx + 1] : dataSectionLength;
                long span = end - (long)t.DataOffset;
                if (span < pq || span >= pq + alignment)
                    why = $"on-disk extent is {span} bytes but PQ2_0 needs {pq} (+<{alignment} padding)";
            }

            if (why != null)
                throw new NotSupportedException(
                    $"Tensor '{t.Name}' has GGUF type id 42, which is ambiguous: dotLLM reads it as PQ2_0 " +
                    "(PrismML ternary, 128-element groups) but upstream ggml uses 42 for Q2_0 (64-element " +
                    $"groups), and this tensor does not fit the PQ2_0 layout ({why}). " +
                    "Upstream Q2_0 is not supported; PQ2_0 models should declare type 142.");
        }
    }

    /// <summary>
    /// Reads all tensor info entries from the current reader position.
    /// </summary>
    /// <param name="reader">A <see cref="BinaryReader"/> positioned after the metadata section.</param>
    /// <param name="header">The previously parsed header (provides version and tensor count).</param>
    /// <returns>List of tensor descriptors.</returns>
    /// <exception cref="NotSupportedException">Unrecognized quantization type.</exception>
    public static List<GgufTensorDescriptor> ReadTensorInfos(BinaryReader reader, GgufHeader header)
    {
        var tensors = new List<GgufTensorDescriptor>((int)Math.Min(header.TensorCount, int.MaxValue));

        for (ulong i = 0; i < header.TensorCount; i++)
        {
            string name = ReadGgufString(reader, header.Version);
            uint nDims = reader.ReadUInt32();

            var dims = new int[nDims];
            for (int d = 0; d < (int)nDims; d++)
            {
                ulong dim = reader.ReadUInt64();
                if (dim > int.MaxValue)
                    throw new InvalidDataException(
                        $"Tensor '{name}' dimension {d} is {dim}, which exceeds Int32.MaxValue.");
                dims[d] = (int)dim;
            }

            uint rawType = reader.ReadUInt32();
            QuantizationType quantType = MapGgufTensorType(rawType, name);
            ulong offset = reader.ReadUInt64();

            tensors.Add(new GgufTensorDescriptor(name, new TensorShape(dims), quantType, offset));
        }

        return tensors;
    }

    /// <summary>
    /// Reads a GGUF length-prefixed UTF-8 string. The length is uint64 in both supported
    /// versions (v2 and v3); only the obsolete v1 used a uint32 length.
    /// </summary>
    internal static string ReadGgufString(BinaryReader reader, uint version)
    {
        ulong length = reader.ReadUInt64();

        if (length == 0)
            return string.Empty;

        if (length > int.MaxValue)
            throw new InvalidDataException($"GGUF string length {length} exceeds Int32.MaxValue.");

        byte[] bytes = reader.ReadBytes((int)length);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    private static object ReadMetadataValue(BinaryReader reader, uint version, GgufValueType valueType)
    {
        return valueType switch
        {
            GgufValueType.UInt8 => reader.ReadByte(),
            GgufValueType.Int8 => reader.ReadSByte(),
            GgufValueType.UInt16 => reader.ReadUInt16(),
            GgufValueType.Int16 => reader.ReadInt16(),
            GgufValueType.UInt32 => reader.ReadUInt32(),
            GgufValueType.Int32 => reader.ReadInt32(),
            GgufValueType.Float32 => reader.ReadSingle(),
            GgufValueType.Bool => reader.ReadByte() != 0,
            GgufValueType.String => ReadGgufString(reader, version),
            GgufValueType.UInt64 => reader.ReadUInt64(),
            GgufValueType.Int64 => reader.ReadInt64(),
            GgufValueType.Float64 => reader.ReadDouble(),
            GgufValueType.Array => ReadArray(reader, version),
            _ => throw new InvalidDataException($"Unknown GGUF value type: {valueType}.")
        };
    }

    private static object ReadArray(BinaryReader reader, uint version)
    {
        var elementType = (GgufValueType)reader.ReadUInt32();
        // Array length is uint64 in both supported versions (v2 and v3); v1 used uint32.
        ulong count = reader.ReadUInt64();

        if (count > int.MaxValue)
            throw new InvalidDataException($"GGUF array length {count} exceeds Int32.MaxValue.");

        int len = (int)count;

        // Return strongly-typed arrays for common element types.
        return elementType switch
        {
            GgufValueType.UInt8 => ReadPrimitiveArray(reader, len, static r => r.ReadByte()),
            GgufValueType.Int8 => ReadPrimitiveArray(reader, len, static r => r.ReadSByte()),
            GgufValueType.UInt16 => ReadPrimitiveArray(reader, len, static r => r.ReadUInt16()),
            GgufValueType.Int16 => ReadPrimitiveArray(reader, len, static r => r.ReadInt16()),
            GgufValueType.UInt32 => ReadPrimitiveArray(reader, len, static r => r.ReadUInt32()),
            GgufValueType.Int32 => ReadPrimitiveArray(reader, len, static r => r.ReadInt32()),
            GgufValueType.Float32 => ReadPrimitiveArray(reader, len, static r => r.ReadSingle()),
            GgufValueType.Bool => ReadPrimitiveArray(reader, len, static r => r.ReadByte() != 0),
            GgufValueType.String => ReadStringArray(reader, len, version),
            GgufValueType.UInt64 => ReadPrimitiveArray(reader, len, static r => r.ReadUInt64()),
            GgufValueType.Int64 => ReadPrimitiveArray(reader, len, static r => r.ReadInt64()),
            GgufValueType.Float64 => ReadPrimitiveArray(reader, len, static r => r.ReadDouble()),
            _ => throw new InvalidDataException($"Unknown GGUF array element type: {elementType}.")
        };
    }

    private static T[] ReadPrimitiveArray<T>(BinaryReader reader, int count, Func<BinaryReader, T> readElement)
    {
        var array = new T[count];
        for (int i = 0; i < count; i++)
            array[i] = readElement(reader);
        return array;
    }

    private static string[] ReadStringArray(BinaryReader reader, int count, uint version)
    {
        var array = new string[count];
        for (int i = 0; i < count; i++)
            array[i] = ReadGgufString(reader, version);
        return array;
    }
}
