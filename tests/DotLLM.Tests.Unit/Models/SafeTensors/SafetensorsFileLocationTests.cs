using System.Text.Json;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

public sealed class SafetensorsFileLocationTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"dotllm-file-ranges-{Guid.NewGuid():N}");

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OriginalRangesResolveExactBytesIncludingShardAndHeaderOffsets(bool sharded)
    {
        Directory.CreateDirectory(_directory);
        byte[] first = [0, 0, 128, 63, 0, 0, 0, 64];
        byte[] second = [0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64];
        string a = Path.Combine(_directory, "a.safetensors"), b = Path.Combine(_directory, "b.safetensors");
        var builder = new SafetensorsFixtureBuilder().AddRaw("first", "F32", [2], first);
        if (sharded)
            new SafetensorsFixtureBuilder().AddRaw("padding", "F32", [2], first)
                .AddRaw("second", "F32", [3], second).WriteTo(b);
        else builder.AddRaw("second", "F32", [3], second);
        builder.WriteTo(a);
        string index = Path.Combine(_directory, "model.safetensors.index.json");
        File.WriteAllText(index, JsonSerializer.Serialize(new { weight_map = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["first"] = "a.safetensors", ["second"] = sharded ? "b.safetensors" : "a.safetensors"
        }}));
        using ISafetensorsTensorSource source = sharded ? MultiShardSafetensorsFile.Open(index) : SafetensorsFile.Open(a);
        Check("first", first, a);
        Check("second", second, sharded ? b : a);
        Assert.Throws<KeyNotFoundException>(() => source.GetTensorFileLocation("missing"));
#pragma warning disable IDISP016 // Verify that the disposed source refuses direct file access.
        source.Dispose();
        Assert.Throws<ObjectDisposedException>(() => source.GetTensorFileLocation("first"));
#pragma warning restore IDISP016

        void Check(string name, byte[] expected, string path)
        {
            var location = source.GetTensorFileLocation(name);
            Assert.Equal(path, location.Path);
            using var stream = File.OpenRead(location.Path);
            stream.Position = location.FileOffset;
            byte[] bytes = new byte[expected.Length];
            stream.ReadExactly(bytes);
            Assert.Equal(expected, bytes);
        }
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }
}
