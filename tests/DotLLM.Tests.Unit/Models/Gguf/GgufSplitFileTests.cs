using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Multi-shard (<c>-0000N-of-0000M.gguf</c>) GGUF open — issue #756 (opened from #815).
/// Shapes mirror the real <c>unsloth/Qwen3.8-Flash-Next-GGUF</c> UD-Q4_K_XL set whose headers were
/// inspected: shard 1 is metadata-only (0 tensors) and shards 2..N carry just
/// <c>split.no</c>/<c>split.count</c>/<c>split.tensors.count</c> plus their own tensor table, with
/// per-shard-local data offsets.
/// </summary>
public sealed class GgufSplitFileTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-split-" + Guid.NewGuid().ToString("N"));

    public GgufSplitFileTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort cleanup */ }
    }

    private static byte[] F32Bytes(params float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private string ShardPath(int no, int count, string stem = "model") =>
        Path.Combine(_dir, $"{stem}-{no:D5}-of-{count:D5}.gguf");

    /// <summary>
    /// Writes a three-shard set: shard 1 = metadata only; shard 2 = {a, b}; shard 3 = {c}. Tensor data
    /// carries distinguishable values so a wrong-shard pointer is caught by value, not just by non-null.
    /// </summary>
    private string WriteThreeShardSet(int declaredTensors = 3)
    {
        var s1 = new GgufWriter()
            .AddString("general.architecture", "llama")
            .AddUInt32("llama.block_count", 7)
            .AddUInt16("split.no", 0).AddUInt16("split.count", 3).AddInt32("split.tensors.count", declaredTensors)
            .Build();
        var s2 = new GgufWriter()
            .AddUInt16("split.no", 1).AddUInt16("split.count", 3).AddInt32("split.tensors.count", declaredTensors)
            .AddTensor("a.weight", [4], (uint)QuantizationType.F32, F32Bytes(1, 2, 3, 4))
            .AddTensor("b.weight", [2], (uint)QuantizationType.F32, F32Bytes(10, 20))
            .Build();
        var s3 = new GgufWriter()
            .AddUInt16("split.no", 2).AddUInt16("split.count", 3).AddInt32("split.tensors.count", declaredTensors)
            .AddTensor("c.weight", [3], (uint)QuantizationType.F32, F32Bytes(100, 200, 300))
            .Build();

        var files = new[] { (ShardPath(1, 3), s1), (ShardPath(2, 3), s2), (ShardPath(3, 3), s3) };
        foreach (var (path, bytes) in files)
        {
            File.WriteAllBytes(path, bytes);
        }
        return ShardPath(1, 3);
    }

    [Fact]
    public void Open_SplitSet_UnifiesTensorTableAcrossShards()
    {
        string first = WriteThreeShardSet();

        using var file = GgufFile.Open(first);

        Assert.True(file.IsSplit);
        Assert.Equal(3, file.ShardCount);
        Assert.Equal(3, file.ShardPaths.Count);
        Assert.Equal(["a.weight", "b.weight", "c.weight"], file.Tensors.Select(t => t.Name).ToArray());
        Assert.Equal(3, file.TensorsByName.Count);
        // Metadata comes from shard 1 only.
        Assert.Equal("llama", file.Metadata.GetString("general.architecture"));
        Assert.Equal(7u, file.Metadata.GetUInt32("llama.block_count"));
        // Offsets stay shard-local: b follows a inside shard 2, c restarts at 0 in shard 3.
        Assert.Equal(0ul, file.TensorsByName["a.weight"].DataOffset);
        Assert.Equal(16ul, file.TensorsByName["b.weight"].DataOffset);
        Assert.Equal(0ul, file.TensorsByName["c.weight"].DataOffset);
        Assert.Equal(1, file.GetTensorShardIndex("a.weight"));
        Assert.Equal(2, file.GetTensorShardIndex("c.weight"));
    }

    [Fact]
    public void Open_SplitSet_TensorDataPointerResolvesIntoTheOwningShard()
    {
        string first = WriteThreeShardSet();
        using var file = GgufFile.Open(first);

        unsafe
        {
            Assert.Equal(1f, ((float*)file.TensorDataPointer("a.weight"))[0]);
            Assert.Equal(4f, ((float*)file.TensorDataPointer("a.weight"))[3]);
            Assert.Equal(20f, ((float*)file.TensorDataPointer("b.weight"))[1]);
            Assert.Equal(100f, ((float*)file.TensorDataPointer("c.weight"))[0]);
            Assert.Equal(300f, ((float*)file.TensorDataPointer(file.TensorsByName["c.weight"]))[2]);
        }
    }

    [Fact]
    public void Open_SplitSet_DataBasePointerThrowsInsteadOfAddressingTheWrongShard()
    {
        string first = WriteThreeShardSet();
        using var file = GgufFile.Open(first);

        var ex = Assert.Throws<InvalidOperationException>(() => file.DataBasePointer);
        Assert.Contains("TensorDataPointer", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_DataSectionLengthIsTheSumOfAllShards()
    {
        string first = WriteThreeShardSet();
        using var file = GgufFile.Open(first);

        // a(16) + b(8) in shard 2, c(12) in shard 3; shard 1 contributes nothing.
        Assert.Equal(16 + 8 + 12, file.DataSectionLength);
    }

    [Fact]
    public void Open_SingleFile_IsUnchangedByTheSplitMachinery()
    {
        var bytes = new GgufWriter()
            .AddString("general.architecture", "llama")
            .AddTensor("t.weight", [2], (uint)QuantizationType.F32, F32Bytes(5, 6))
            .Build();
        string path = Path.Combine(_dir, "single.gguf");
        File.WriteAllBytes(path, bytes);

        using var file = GgufFile.Open(path);

        Assert.False(file.IsSplit);
        Assert.Equal(1, file.ShardCount);
        Assert.Equal(8, file.DataSectionLength);
        Assert.NotEqual(nint.Zero, file.DataBasePointer);
        Assert.Equal(file.DataBasePointer, file.TensorDataPointer("t.weight"));
        Assert.Equal(0, file.GetTensorShardIndex("t.weight"));
    }

    [Fact]
    public void Open_SingleShardSplitCountOne_IsAnOrdinaryFile()
    {
        // llama-gguf-split can legally emit split.count == 1; it must not demand the -0000N-of- name.
        var bytes = new GgufWriter()
            .AddString("general.architecture", "llama")
            .AddUInt16("split.no", 0).AddUInt16("split.count", 1)
            .AddTensor("t.weight", [1], (uint)QuantizationType.F32, F32Bytes(9))
            .Build();
        string path = Path.Combine(_dir, "whatever.gguf");
        File.WriteAllBytes(path, bytes);

        using var file = GgufFile.Open(path);

        Assert.False(file.IsSplit);
        Assert.Single(file.Tensors);
    }

    [Fact]
    public void Open_SplitSet_MissingShard_ThrowsFileNotFoundNamingTheShard()
    {
        string first = WriteThreeShardSet();
        File.Delete(ShardPath(3, 3));

        var ex = Assert.Throws<FileNotFoundException>(() => GgufFile.Open(first));

        Assert.Equal(ShardPath(3, 3), ex.FileName);
        Assert.Contains("shard 3 is missing", ex.Message);
        Assert.Contains("00003-of-00003", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_NotTheFirstShard_ThrowsAndPointsAtShardOne()
    {
        WriteThreeShardSet();

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(ShardPath(2, 3)));

        Assert.Contains("open the first shard", ex.Message);
        Assert.Contains("model-00001-of-00003.gguf", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_SiblingWithWrongSplitNo_Throws()
    {
        // Shard 3's file is replaced by a copy of shard 2's bytes (e.g. a botched rename / mixed download).
        string first = WriteThreeShardSet();
        File.Copy(ShardPath(2, 3), ShardPath(3, 3), overwrite: true);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(first));

        Assert.Contains("split.no 1, expected 2", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_SiblingFromADifferentSplitCount_Throws()
    {
        string first = WriteThreeShardSet();
        var alien = new GgufWriter()
            .AddUInt16("split.no", 2).AddUInt16("split.count", 5).AddInt32("split.tensors.count", 3)
            .AddTensor("c.weight", [3], (uint)QuantizationType.F32, F32Bytes(1, 2, 3))
            .Build();
        File.WriteAllBytes(ShardPath(3, 3), alien);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(first));

        Assert.Contains("split.count 5, expected 3", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_TensorCountMismatch_Throws()
    {
        // Metadata promises 4 tensors; the shards hold 3 (a truncated / replaced shard).
        string first = WriteThreeShardSet(declaredTensors: 4);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(first));

        Assert.Contains("split.tensors.count=4", ex.Message);
        Assert.Contains("contain 3 tensors", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_DuplicateTensorAcrossShards_Throws()
    {
        string first = WriteThreeShardSet();
        var dup = new GgufWriter()
            .AddUInt16("split.no", 2).AddUInt16("split.count", 3).AddInt32("split.tensors.count", 3)
            .AddTensor("a.weight", [3], (uint)QuantizationType.F32, F32Bytes(1, 2, 3))
            .Build();
        File.WriteAllBytes(ShardPath(3, 3), dup);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(first));

        Assert.Contains("'a.weight' appears in both", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_TruncatedShard_BoundsCheckNamesTheShardFile()
    {
        string first = WriteThreeShardSet();
        string victim = ShardPath(2, 3);
        byte[] bytes = File.ReadAllBytes(victim);
        File.WriteAllBytes(victim, bytes.AsSpan(0, bytes.Length - 6).ToArray()); // cut into b.weight's data

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(first));

        Assert.Contains("beyond file boundary", ex.Message);
        Assert.Contains("b.weight", ex.Message);
        Assert.Contains("model-00002-of-00003.gguf", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_WithoutTheNamingConvention_Throws()
    {
        string first = WriteThreeShardSet();
        string renamed = Path.Combine(_dir, "renamed.gguf");
        File.Move(first, renamed);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(renamed));

        Assert.Contains("0000N-of-0000M", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_FileNameDisagreesWithSplitCount_Throws()
    {
        string first = WriteThreeShardSet();
        string renamed = ShardPath(1, 4);
        File.Move(first, renamed);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(renamed));

        Assert.Contains("one of 4 shards", ex.Message);
        Assert.Contains("split.count metadata says 3", ex.Message);
    }

    [Fact]
    public void Open_SplitSet_DisposeReleasesEveryShardFile()
    {
        string first = WriteThreeShardSet();
        var file = GgufFile.Open(first);
        file.Dispose();

        // Windows would refuse to delete a still-mapped shard.
        foreach (string path in file.ShardPaths)
            File.Delete(path);
        Assert.Empty(Directory.GetFiles(_dir));
    }
}
