using System.Text.Json;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// Loader for the HF-generated Qwen4-Exp reference fixtures under <c>Models/Qwen4Exp/Fixtures</c> (issue #816).
/// The generators (<c>Models/Qwen4Exp/Reference/gen_*.py</c>) run real <c>transformers</c> <c>qwen4_exp</c> code and
/// dump inputs, weights and intermediates as base64 little-endian tensors; CI has no torch, so the values are baked in.
/// </summary>
internal sealed class Qwen4ExpReferenceFixture
{
    private readonly Dictionary<string, (int[] Shape, string DType, byte[] Bytes)> _tensors = new();

    /// <summary>The generator's <c>meta</c> object (integer entries only are read by tests).</summary>
    public JsonElement Meta { get; }

    private Qwen4ExpReferenceFixture(JsonElement meta) => Meta = meta;

    /// <summary>Loads <c>Models/Qwen4Exp/Fixtures/{fileName}</c> from the test output directory.</summary>
    public static Qwen4ExpReferenceFixture Load(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Models", "Qwen4Exp", "Fixtures", fileName);
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = doc.RootElement;
        var fx = new Qwen4ExpReferenceFixture(root.GetProperty("meta").Clone());
        foreach (var p in root.GetProperty("tensors").EnumerateObject())
        {
            int[] shape = p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            fx._tensors[p.Name] = (shape, p.Value.GetProperty("dtype").GetString()!,
                                   Convert.FromBase64String(p.Value.GetProperty("b64").GetString()!));
        }
        return fx;
    }

    /// <summary>Integer meta entry.</summary>
    public int Int(string key) => Meta.GetProperty(key).GetInt32();

    /// <summary>Whether a tensor of this name exists.</summary>
    public bool Has(string name) => _tensors.ContainsKey(name);

    /// <summary>Shape of a tensor.</summary>
    public int[] Shape(string name) => _tensors[name].Shape;

    /// <summary>Reads an F32 tensor.</summary>
    public float[] F32(string name)
    {
        var (_, dtype, bytes) = _tensors[name];
        if (dtype != "f32") throw new InvalidOperationException($"{name} is {dtype}, not f32.");
        var r = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, r, 0, bytes.Length);
        return r;
    }

    /// <summary>Reads an int64 tensor.</summary>
    public long[] I64(string name)
    {
        var (_, dtype, bytes) = _tensors[name];
        if (dtype != "i64") throw new InvalidOperationException($"{name} is {dtype}, not i64.");
        var r = new long[bytes.Length / 8];
        Buffer.BlockCopy(bytes, 0, r, 0, bytes.Length);
        return r;
    }
}
