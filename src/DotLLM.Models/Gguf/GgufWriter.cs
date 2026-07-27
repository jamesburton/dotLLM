using System.Text;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Minimal GGUF v3 byte writer used by <see cref="SyntheticGemma4Gguf"/> to synthesize
/// fixtures. Mirrors the conventions of the test-side <c>GgufTestData</c> (32-byte data
/// alignment, header, ne0-inner dim order) so the output round-trips through
/// <see cref="GgufReader"/>/<see cref="GgufFile"/> identically to the real loader path.
/// </summary>
public sealed class GgufWriter
{
    private const uint Version = 3;

    private readonly List<Action<BinaryWriter>> _metadata = [];
    private readonly List<Action<BinaryWriter>> _tensorInfos = [];
    private readonly List<byte[]> _tensorData = [];

    /// <summary>Adds a metadata key with a caller-supplied value writer.</summary>
    public GgufWriter AddMetadata(string key, GgufValueType type, Action<BinaryWriter> writeValue)
    {
        _metadata.Add(w => { WriteString(w, key); w.Write((uint)type); writeValue(w); });
        return this;
    }

    /// <summary>Adds a string metadata entry.</summary>
    public GgufWriter AddString(string key, string value) =>
        AddMetadata(key, GgufValueType.String, w => WriteString(w, value));

    /// <summary>Adds a UInt32 metadata entry.</summary>
    public GgufWriter AddUInt32(string key, uint value) =>
        AddMetadata(key, GgufValueType.UInt32, w => w.Write(value));

    /// <summary>Adds a Float32 metadata entry.</summary>
    public GgufWriter AddFloat32(string key, float value) =>
        AddMetadata(key, GgufValueType.Float32, w => w.Write(value));

    /// <summary>Adds a Bool metadata entry.</summary>
    public GgufWriter AddBool(string key, bool value) =>
        AddMetadata(key, GgufValueType.Bool, w => w.Write((byte)(value ? 1 : 0)));

    /// <summary>Adds an Int32 array metadata entry.</summary>
    public GgufWriter AddInt32Array(string key, int[] values) =>
        AddMetadata(key, GgufValueType.Array, w =>
        {
            w.Write((uint)GgufValueType.Int32);
            w.Write((ulong)values.Length);
            foreach (int v in values) w.Write(v);
        });

    /// <summary>Adds a Float32 array metadata entry.</summary>
    public GgufWriter AddFloat32Array(string key, float[] values) =>
        AddMetadata(key, GgufValueType.Array, w =>
        {
            w.Write((uint)GgufValueType.Float32);
            w.Write((ulong)values.Length);
            foreach (float v in values) w.Write(v);
        });

    /// <summary>Adds a String array metadata entry.</summary>
    public GgufWriter AddStringArray(string key, string[] values) =>
        AddMetadata(key, GgufValueType.Array, w =>
        {
            w.Write((uint)GgufValueType.String);
            w.Write((ulong)values.Length);
            foreach (string s in values) WriteString(w, s);
        });

    /// <summary>Adds a tensor info entry (name, dims [ne0=K-inner], quant type) and its data blob.</summary>
    public GgufWriter AddTensor(string name, int[] dims, uint quantType, byte[] data)
    {
        ulong dataOffset = 0;
        foreach (byte[] blob in _tensorData) dataOffset += (ulong)blob.Length;

        _tensorInfos.Add(w =>
        {
            WriteString(w, name);
            w.Write((uint)dims.Length);
            foreach (int d in dims) w.Write((ulong)d);
            w.Write(quantType);
            w.Write(dataOffset);
        });
        _tensorData.Add(data);
        return this;
    }

    /// <summary>Serializes the complete GGUF byte array.</summary>
    public byte[] Build()
    {
        using var stream = new MemoryStream();
        using var w = new BinaryWriter(stream);

        w.Write(GgufReader.GgufMagic);
        w.Write(Version);
        w.Write((ulong)_tensorInfos.Count);
        w.Write((ulong)_metadata.Count);

        foreach (var m in _metadata) m(w);
        foreach (var t in _tensorInfos) t(w);
        w.Flush();

        long dataStart = AlignUp(stream.Position, 32);
        while (stream.Position < dataStart) w.Write((byte)0);

        foreach (byte[] blob in _tensorData) w.Write(blob);
        w.Flush();
        return stream.ToArray();
    }

    private static void WriteString(BinaryWriter w, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static long AlignUp(long value, long alignment) => (value + (alignment - 1)) & ~(alignment - 1);
}
