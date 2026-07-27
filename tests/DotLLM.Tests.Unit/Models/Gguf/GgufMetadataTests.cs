using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

public class GgufMetadataTests
{
    private static GgufMetadata BuildMetadata(Action<GgufTestData> configure)
    {
        var data = new GgufTestData(version: 3);
        configure(data);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var raw = GgufReader.ReadMetadata(reader, header);
        return new GgufMetadata(raw);
    }

    [Fact]
    public void GetString_ReturnsValue()
    {
        var metadata = BuildMetadata(d => d.AddString("name", "llama"));
        Assert.Equal("llama", metadata.GetString("name"));
    }

    [Fact]
    public void GetUInt32_ReturnsValue()
    {
        var metadata = BuildMetadata(d => d.AddUInt32("layers", 32));
        Assert.Equal(32u, metadata.GetUInt32("layers"));
    }

    [Fact]
    public void GetFloat32_ReturnsValue()
    {
        var metadata = BuildMetadata(d => d.AddFloat32("eps", 1e-5f));
        Assert.Equal(1e-5f, metadata.GetFloat32("eps"));
    }

    [Fact]
    public void GetBool_ReturnsValue()
    {
        var metadata = BuildMetadata(d => d.AddBool("flag", true));
        Assert.True(metadata.GetBool("flag"));
    }

    [Fact]
    public void GetStringArray_ReturnsValue()
    {
        string[] expected = ["a", "b", "c"];
        var metadata = BuildMetadata(d => d.AddStringArray("tokens", expected));
        Assert.Equal(expected, metadata.GetStringArray("tokens"));
    }

    [Fact]
    public void GetFloat32Array_ReturnsValue()
    {
        float[] expected = [1.0f, 2.0f, 3.0f];
        var metadata = BuildMetadata(d => d.AddFloat32Array("scores", expected));
        Assert.Equal(expected, metadata.GetFloat32Array("scores"));
    }

    [Fact]
    public void GetInt32Array_ReturnsValue()
    {
        int[] expected = [1, 2, 3];
        var metadata = BuildMetadata(d => d.AddInt32Array("types", expected));
        Assert.Equal(expected, metadata.GetInt32Array("types"));
    }

    [Fact]
    public void GetString_MissingKey_ThrowsKeyNotFound()
    {
        var metadata = BuildMetadata(_ => { });
        var ex = Assert.Throws<KeyNotFoundException>(() => metadata.GetString("nonexistent"));
        Assert.Contains("nonexistent", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void GetString_WrongType_ThrowsInvalidOperation()
    {
        var metadata = BuildMetadata(d => d.AddUInt32("number", 42));
        var ex = Assert.Throws<InvalidOperationException>(() => metadata.GetString("number"));
        Assert.Contains("UInt32", ex.Message, StringComparison.Ordinal);
        Assert.Contains("String", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void GetUInt32OrDefault_MissingKey_ReturnsDefault()
    {
        var metadata = BuildMetadata(_ => { });
        Assert.Equal(99u, metadata.GetUInt32OrDefault("missing", 99));
    }

    [Fact]
    public void GetStringOrDefault_MissingKey_ReturnsDefault()
    {
        var metadata = BuildMetadata(_ => { });
        Assert.Equal("fallback", metadata.GetStringOrDefault("missing", "fallback"));
    }

    [Fact]
    public void GetFloat32OrDefault_MissingKey_ReturnsDefault()
    {
        var metadata = BuildMetadata(_ => { });
        Assert.Equal(3.14f, metadata.GetFloat32OrDefault("missing", 3.14f));
    }

    [Fact]
    public void GetBoolOrDefault_MissingKey_ReturnsDefault()
    {
        var metadata = BuildMetadata(_ => { });
        Assert.True(metadata.GetBoolOrDefault("missing", true));
    }

    [Fact]
    public void ContainsKey_ExistingKey_ReturnsTrue()
    {
        var metadata = BuildMetadata(d => d.AddString("key", "val"));
        Assert.True(metadata.ContainsKey("key"));
    }

    [Fact]
    public void ContainsKey_MissingKey_ReturnsFalse()
    {
        var metadata = BuildMetadata(_ => { });
        Assert.False(metadata.ContainsKey("missing"));
    }

    [Fact]
    public void Count_ReturnsCorrectCount()
    {
        var metadata = BuildMetadata(d =>
        {
            d.AddString("a", "1");
            d.AddUInt32("b", 2);
        });
        Assert.Equal(2, metadata.Count);
    }

    [Fact]
    public void Keys_ReturnsAllKeys()
    {
        var metadata = BuildMetadata(d =>
        {
            d.AddString("alpha", "a");
            d.AddString("beta", "b");
        });
        Assert.Contains("alpha", metadata.Keys);
        Assert.Contains("beta", metadata.Keys);
    }
}
