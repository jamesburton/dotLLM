using System.Diagnostics.CodeAnalysis;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Typed accessor over GGUF metadata key-value pairs. Provides strongly-typed getters
/// with clear error messages for missing keys and type mismatches.
/// </summary>
public sealed class GgufMetadata
{
    private readonly Dictionary<string, GgufMetadataValue> _entries;

    /// <summary>
    /// Initializes a new instance wrapping the given metadata dictionary.
    /// </summary>
    /// <param name="entries">The raw metadata entries parsed from a GGUF file.</param>
    public GgufMetadata(Dictionary<string, GgufMetadataValue> entries)
    {
        _entries = entries;
    }

    /// <summary>Number of metadata entries.</summary>
    public int Count => _entries.Count;

    /// <summary>All metadata keys.</summary>
    public IEnumerable<string> Keys => _entries.Keys;

    /// <summary>Checks whether a key exists.</summary>
    public bool ContainsKey(string key) => _entries.ContainsKey(key);

    /// <summary>Tries to get the raw metadata value for a key.</summary>
    public bool TryGetValue(string key, [MaybeNullWhen(false)] out GgufMetadataValue value) =>
        _entries.TryGetValue(key, out value);

    /// <summary>Gets a string value.</summary>
    /// <exception cref="KeyNotFoundException">Key not found.</exception>
    /// <exception cref="InvalidOperationException">Value is not a string.</exception>
    public string GetString(string key) => GetTyped<string>(key, GgufValueType.String);

    /// <summary>Gets a string value or a default.</summary>
    public string GetStringOrDefault(string key, string defaultValue = "") =>
        TryGetTyped<string>(key, GgufValueType.String, out var value) ? value : defaultValue;

    /// <summary>Gets a uint8 value.</summary>
    public byte GetUInt8(string key) => GetTyped<byte>(key, GgufValueType.UInt8);

    /// <summary>Gets a uint16 value.</summary>
    public ushort GetUInt16(string key) => GetTyped<ushort>(key, GgufValueType.UInt16);

    /// <summary>Gets a uint32 value.</summary>
    public uint GetUInt32(string key) => GetTyped<uint>(key, GgufValueType.UInt32);

    /// <summary>Gets a uint32 value or a default.</summary>
    public uint GetUInt32OrDefault(string key, uint defaultValue = 0) =>
        TryGetTyped<uint>(key, GgufValueType.UInt32, out var value) ? value : defaultValue;

    /// <summary>Gets an int32 value.</summary>
    public int GetInt32(string key) => GetTyped<int>(key, GgufValueType.Int32);

    /// <summary>Gets a uint64 value.</summary>
    public ulong GetUInt64(string key) => GetTyped<ulong>(key, GgufValueType.UInt64);

    /// <summary>Gets an int64 value.</summary>
    public long GetInt64(string key) => GetTyped<long>(key, GgufValueType.Int64);

    /// <summary>Gets a float32 value.</summary>
    public float GetFloat32(string key) => GetTyped<float>(key, GgufValueType.Float32);

    /// <summary>Gets a float32 value or a default.</summary>
    public float GetFloat32OrDefault(string key, float defaultValue = 0f) =>
        TryGetTyped<float>(key, GgufValueType.Float32, out var value) ? value : defaultValue;

    /// <summary>Gets a float64 value.</summary>
    public double GetFloat64(string key) => GetTyped<double>(key, GgufValueType.Float64);

    /// <summary>Gets a boolean value.</summary>
    public bool GetBool(string key) => GetTyped<bool>(key, GgufValueType.Bool);

    /// <summary>Gets a boolean value or a default.</summary>
    public bool GetBoolOrDefault(string key, bool defaultValue = false) =>
        TryGetTyped<bool>(key, GgufValueType.Bool, out var value) ? value : defaultValue;

    /// <summary>Gets a string array value.</summary>
    public string[] GetStringArray(string key) => GetTyped<string[]>(key, GgufValueType.Array);

    /// <summary>Gets a float32 array value.</summary>
    public float[] GetFloat32Array(string key) => GetTyped<float[]>(key, GgufValueType.Array);

    /// <summary>Gets an int32 array value.</summary>
    public int[] GetInt32Array(string key) => GetTyped<int[]>(key, GgufValueType.Array);

    /// <summary>Gets a uint32 array value.</summary>
    public uint[] GetUInt32Array(string key) => GetTyped<uint[]>(key, GgufValueType.Array);

    /// <summary>Gets a uint64 array value (exact 64-bit values; no float round-trip).</summary>
    public ulong[] GetUInt64Array(string key) => GetTyped<ulong[]>(key, GgufValueType.Array);

    /// <summary>Gets an int64 array value.</summary>
    public long[] GetInt64Array(string key) => GetTyped<long[]>(key, GgufValueType.Array);

    private T GetTyped<T>(string key, GgufValueType expectedType)
    {
        if (!_entries.TryGetValue(key, out var entry))
            throw new KeyNotFoundException($"GGUF metadata key '{key}' not found.");

        if (entry.Type != expectedType)
            throw new InvalidOperationException(
                $"GGUF metadata key '{key}' has type {entry.Type}, expected {expectedType}.");

        if (entry.Value is not T typed)
            throw new InvalidOperationException(
                $"GGUF metadata key '{key}' value is {entry.Value.GetType().Name}, expected {typeof(T).Name}.");

        return typed;
    }

    private bool TryGetTyped<T>(string key, GgufValueType expectedType, [MaybeNullWhen(false)] out T value)
    {
        if (_entries.TryGetValue(key, out var entry) && entry.Type == expectedType && entry.Value is T typed)
        {
            value = typed;
            return true;
        }

        value = default;
        return false;
    }
}
