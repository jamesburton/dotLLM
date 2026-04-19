using System.Buffers.Binary;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using DotLLM.Core.Tensors;
using DotLLM.HuggingFace;
using DotLLM.Models;
using DotLLM.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end verification that <see cref="ModelLoader.LoadFromSafetensors"/>
/// can ingest a multi-shard HuggingFace checkpoint
/// (<c>model.safetensors.index.json</c> + <c>model-0000N-of-0000M.safetensors</c>)
/// and run a forward pass producing finite vocab-sized logits.
/// </summary>
/// <remarks>
/// <para>
/// No tiny-random HF repo ships as multi-shard (they are all well under the
/// 5 GiB default shard cap), so we synthesise one. On first run, we
/// download the single-file <c>hf-internal-testing/tiny-random-LlamaForCausalLM</c>
/// checkpoint (4 MB F32), then re-emit its tensors across 2 shards +
/// <c>model.safetensors.index.json</c> in a scratch directory. The resharding
/// exercises the exact on-disk layout that a real 16 GB Llama-3-8B would
/// produce, minus the bandwidth cost.
/// </para>
/// <para>
/// The synthesis is idempotent: if the scratch directory already contains
/// <c>config.json</c> + the two shards + index.json, we skip it. Failures in
/// either the download or the reshard step skip the test gracefully.
/// </para>
/// </remarks>
public sealed class MultiShardSafetensorsLoadTests
{
    private const string BaseRepo = "hf-internal-testing/tiny-random-LlamaForCausalLM";
    private const int MaxAllowedBytes = 50 * 1024 * 1024;

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    private readonly ITestOutputHelper _output;

    public MultiShardSafetensorsLoadTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void LoadSynthesizedMultiShard_ProducesFiniteVocabLogits()
    {
        string? shardRoot = TryBuildMultiShardFixture(out string? skipReason);
        Skip.If(shardRoot is null, skipReason ?? "multi-shard fixture synthesis unavailable");

        _output.WriteLine($"Multi-shard fixture root: {shardRoot}");
        foreach (var f in Directory.GetFiles(shardRoot!).OrderBy(x => x))
            _output.WriteLine($"  {Path.GetFileName(f)}  {new FileInfo(f).Length} bytes");

        // Load via the directory path — ModelLoader auto-detects the index.
        using var loaded = LoadedModel.Open(shardRoot!);
        var (model, src, config) = (loaded.Model, loaded.Source, loaded.Config);

        Assert.IsType<MultiShardSafetensorsFile>(src);
        var msrc = (MultiShardSafetensorsFile)src;
        Assert.Equal(2, msrc.ShardCount);

        _output.WriteLine(
            $"Config: arch={config.Architecture} vocab={config.VocabSize} hidden={config.HiddenSize} "
          + $"layers={config.NumLayers} heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads} "
          + $"head_dim={config.HeadDim} intermediate={config.IntermediateSize} tied={config.TiedEmbeddings}");

        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];
        var sw = Stopwatch.StartNew();
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        sw.Stop();

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(config.VocabSize, logits.Shape[1]);

        var stats = ComputeStats(logits);
        _output.WriteLine(
            $"Forward: shape=[{logits.Shape[0]}, {logits.Shape[1]}] "
          + $"finite={stats.FiniteCount}/{stats.TotalCount} "
          + $"min={stats.Min:G4} max={stats.Max:G4} mean={stats.Mean:G4} stddev={stats.StdDev:G4} "
          + $"in {sw.Elapsed.TotalMilliseconds:F1} ms");

        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0, "Logits have zero variance — forward pass likely degenerate.");
    }

    /// <summary>
    /// Ensures the single-file tiny-random Llama is downloaded, then splits
    /// it into two shards + an index.json under a sibling directory. Returns
    /// the multi-shard directory path or null with a skip reason.
    /// </summary>
    private string? TryBuildMultiShardFixture(out string? skipReason)
    {
        // Ensure the single-file fixture is available.
        string singleDir = Path.Combine(
            CacheDir, BaseRepo.Replace('/', Path.DirectorySeparatorChar));
        string singleSafetensors = Path.Combine(singleDir, "model.safetensors");
        string singleConfig = Path.Combine(singleDir, "config.json");

        if (!File.Exists(singleSafetensors) || !File.Exists(singleConfig))
        {
            try
            {
                using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
                using var downloader = new HuggingFaceDownloader(http);

                string url = $"https://huggingface.co/{BaseRepo}/resolve/main/model.safetensors";
                using (var head = new HttpRequestMessage(HttpMethod.Head, url))
                using (var headResp = http.SendAsync(head, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult())
                {
                    if (!headResp.IsSuccessStatusCode)
                    {
                        skipReason = $"HEAD {url} returned {(int)headResp.StatusCode}";
                        return null;
                    }
                    if (headResp.Content.Headers.ContentLength is long t && t > MaxAllowedBytes)
                    {
                        skipReason = $"{BaseRepo}: model.safetensors is {t} bytes > cap {MaxAllowedBytes}";
                        return null;
                    }
                }

                _output.WriteLine($"{BaseRepo}: downloading single-file fixture to {singleDir}");
                downloader.DownloadFileAsync(BaseRepo, "model.safetensors", CacheDir, progress: null)
                    .GetAwaiter().GetResult();
                downloader.DownloadFileAsync(BaseRepo, "config.json", CacheDir, progress: null)
                    .GetAwaiter().GetResult();

                if (!File.Exists(singleSafetensors) || !File.Exists(singleConfig))
                {
                    skipReason = $"{BaseRepo}: download did not produce expected files";
                    return null;
                }
            }
            catch (Exception ex)
            {
                skipReason = $"{BaseRepo} download failed: {ex.GetType().Name}: {ex.Message}";
                return null;
            }
        }

        // Derive the sharded scratch directory alongside the single-file cache.
        string shardRoot = Path.Combine(
            CacheDir, BaseRepo.Replace('/', Path.DirectorySeparatorChar) + "--sharded-2");
        string shardA = Path.Combine(shardRoot, "model-00001-of-00002.safetensors");
        string shardB = Path.Combine(shardRoot, "model-00002-of-00002.safetensors");
        string indexPath = Path.Combine(shardRoot, "model.safetensors.index.json");
        string shardConfig = Path.Combine(shardRoot, "config.json");

        if (File.Exists(shardA) && File.Exists(shardB) && File.Exists(indexPath) && File.Exists(shardConfig))
        {
            skipReason = null;
            return shardRoot;
        }

        try
        {
            Directory.CreateDirectory(shardRoot);
            File.Copy(singleConfig, shardConfig, overwrite: true);
            ReshardSafetensorsByHalf(singleSafetensors, shardA, shardB, indexPath);
            skipReason = null;
            return shardRoot;
        }
        catch (Exception ex)
        {
            skipReason = $"reshard failed: {ex.GetType().Name}: {ex.Message}";
            return null;
        }
    }

    /// <summary>
    /// Reads the single-file safetensors at <paramref name="sourcePath"/>
    /// and writes the same tensors split ~half-and-half across two new
    /// safetensors files plus a matching <c>model.safetensors.index.json</c>.
    /// </summary>
    private static void ReshardSafetensorsByHalf(string sourcePath, string shardAPath, string shardBPath, string indexPath)
    {
        using var src = SafetensorsFile.Open(sourcePath);
        var tensors = src.Tensors;
        int midpoint = tensors.Count / 2;

        // Pass 1: group tensors in declaration order.
        var groupA = new List<SafetensorsTensorDescriptor>();
        var groupB = new List<SafetensorsTensorDescriptor>();
        for (int i = 0; i < tensors.Count; i++)
            (i < midpoint ? groupA : groupB).Add(tensors[i]);

        WriteShard(src, groupA, shardAPath);
        WriteShard(src, groupB, shardBPath);

        // Emit the index.
        var weightMap = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var t in groupA) weightMap[t.Name] = Path.GetFileName(shardAPath);
        foreach (var t in groupB) weightMap[t.Name] = Path.GetFileName(shardBPath);

        long totalSize = 0;
        foreach (var t in tensors) totalSize += t.ByteCount;

        string indexJson = JsonSerializer.Serialize(new
        {
            metadata = new { total_size = totalSize },
            weight_map = weightMap,
        });
        File.WriteAllText(indexPath, indexJson);
    }

    /// <summary>
    /// Writes <paramref name="tensors"/> from <paramref name="src"/> into a
    /// fresh safetensors file at <paramref name="destPath"/>. Data bytes are
    /// copied verbatim (no dtype conversion).
    /// </summary>
    private static void WriteShard(SafetensorsFile src, List<SafetensorsTensorDescriptor> tensors, string destPath)
    {
        // Build header JSON and precompute per-tensor offsets.
        using var headerStream = new MemoryStream();
        using (var w = new Utf8JsonWriter(headerStream, new JsonWriterOptions { Indented = false }))
        {
            w.WriteStartObject();
            long offset = 0;
            foreach (var t in tensors)
            {
                w.WriteStartObject(t.Name);
                w.WriteString("dtype", DTypeToToken(t.DType));
                w.WritePropertyName("shape");
                w.WriteStartArray();
                foreach (var d in t.Shape) w.WriteNumberValue(d);
                w.WriteEndArray();
                w.WritePropertyName("data_offsets");
                w.WriteStartArray();
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset + t.ByteCount);
                w.WriteEndArray();
                w.WriteEndObject();
                offset += t.ByteCount;
            }
            w.WriteEndObject();
        }
        byte[] headerJson = headerStream.ToArray();

        using var fs = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)headerJson.Length);
        fs.Write(prefix);
        fs.Write(headerJson);
        foreach (var t in tensors)
        {
            // Payload bytes: copy from the mmap view via GetTensorSpan.
            // All tiny-random fixtures are well under Int32.MaxValue per tensor.
            fs.Write(src.GetTensorSpan(t.Name));
        }
    }

    private static string DTypeToToken(SafetensorsDType dtype) => dtype switch
    {
        SafetensorsDType.F32 => "F32",
        SafetensorsDType.F16 => "F16",
        SafetensorsDType.BF16 => "BF16",
        SafetensorsDType.F64 => "F64",
        SafetensorsDType.I8 => "I8",
        SafetensorsDType.U8 => "U8",
        SafetensorsDType.I16 => "I16",
        SafetensorsDType.I32 => "I32",
        SafetensorsDType.I64 => "I64",
        SafetensorsDType.Bool => "BOOL",
        _ => throw new NotSupportedException($"cannot reshard tensor with dtype {dtype}"),
    };

    private static unsafe LogitStats ComputeStats(ITensor logits)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);

        int finite = 0;
        double sum = 0, sumSq = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (float v in span)
        {
            if (float.IsFinite(v))
            {
                finite++;
                sum += v;
                sumSq += (double)v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        double mean = finite > 0 ? sum / finite : 0.0;
        double variance = finite > 0 ? (sumSq / finite) - (mean * mean) : 0.0;
        double stddev = Math.Sqrt(Math.Max(0.0, variance));
        return new LogitStats(total, finite, (float)mean, (float)stddev, min, max);
    }

    private readonly record struct LogitStats(
        int TotalCount, int FiniteCount, float Mean, float StdDev, float Min, float Max);

    private sealed record LoadedModel(
        DotLLM.Core.Models.IModel Model,
        ISafetensorsTensorSource Source,
        DotLLM.Core.Models.ModelConfig Config) : IDisposable
    {
        public static LoadedModel Open(string path)
        {
            var (model, file, config) = ModelLoader.LoadFromSafetensors(path);
            return new LoadedModel(model, file, config);
        }
        public void Dispose()
        {
            Model.Dispose();
            Source.Dispose();
        }
    }
}
