using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaRecurrentPrefixSnapshotTests
{
    [SkippableTheory]
    [InlineData(KvCacheDType.F32, KvCacheDType.F32)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q4_0)]
    public void HostSnapshot_IsIndependentAndExactAcrossAlternatingHistories(KvCacheDType keys, KvCacheDType values)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = Path.Combine(AppContext.BaseDirectory, "ptx");
        Skip.IfNot(Directory.Exists(ptx), "PTX files missing");
        string path = Path.Combine(Path.GetTempPath(), $"prefix-snapshot-{Guid.NewGuid():N}.gguf");
        try
        {
            SyntheticQwen35HybridDenseMtpGguf.Write(path, withMtp: false, contextLength: 64, pq2_0Projections: true);
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, 0, ptx);
            model.ConfigureKvStorage(keys, values);
            var provider = Assert.IsAssignableFrom<IRecurrentPrefixSnapshotProvider>(model);
            using var a = model.CreateKvCache(64);
            using var b = model.CreateKvCache(64);
            model.Forward([1, 2, 3, 4], [0, 1, 2, 3], -1, a).Dispose();
            using var snapshotA = provider.CapturePrefixSnapshot(a, 4, RecurrentPrefixSnapshotStorage.Host);
            float[] expectedA = Next(model, a, 5, 4);
            model.ResetSequenceState();
            model.Forward([7, 6, 5], [0, 1, 2], -1, b).Dispose();
            using var snapshotB = provider.CapturePrefixSnapshot(b, 3, RecurrentPrefixSnapshotStorage.Host);
            float[] expectedB = Next(model, b, 4, 3);
            provider.RestorePrefixSnapshot(snapshotA, a);
            Assert.Equal(expectedA, Next(model, a, 5, 4));
            provider.RestorePrefixSnapshot(snapshotB, b);
            Assert.Equal(expectedB, Next(model, b, 4, 3));
            provider.RestorePrefixSnapshot(snapshotA, b);
            Assert.Equal(expectedA, Next(model, b, 5, 4));
            using var other = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, 0, ptx);
            using var otherKv = other.CreateKvCache(64);
            Assert.Throws<ArgumentException>(() => ((IRecurrentPrefixSnapshotProvider)other).RestorePrefixSnapshot(snapshotA, otherKv));
            Assert.Equal(0, otherKv.CurrentLength);
            snapshotA.Dispose();
            Assert.Throws<ObjectDisposedException>(() => provider.RestorePrefixSnapshot(snapshotA, a));
            Assert.Throws<ObjectDisposedException>(() => provider.TransferPrefixSnapshot(snapshotA, RecurrentPrefixSnapshotStorage.Device));
        }
        finally { File.Delete(path); }
    }

    private static unsafe float[] Next(CudaQwen3HybridDenseTransformerModel model, DotLLM.Core.Attention.IKvCache kv, int token, int position)
    {
        using var logits = model.Forward([token], [position], -1, kv);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount).ToArray();
    }
}


