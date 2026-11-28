using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #2030: speculative verification reads one logit row per drafted token. The dense CUDA host returned the last row only;
/// once widened, row i of one multi-token forward must be what decoding the same tokens one at a time gives at step i.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaAllRowLogitsTests(ITestOutputHelper output)
{
    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    private static float[] Row(ITensor logits, int row, int vocab)
    {
        unsafe { return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)row * vocab, vocab).ToArray(); }
    }

    private static int ArgMax(float[] row) => Array.IndexOf(row, row.Max());

    private static float MaxAbsDiff(float[] a, float[] b) => a.Zip(b, (x, y) => Math.Abs(x - y)).Max();

    [SkippableFact]
    public void WidenedForward_ReturnsTheRowEachSingleTokenDecodeWouldReturn()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string? path = TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_Q8_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir);
        int vocab = config.VocabSize;

        int[] prefix = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315];
        int[] batch = [18770, 374, 26194, 13];
        int[] positions = [10, 11, 12, 13];
        int[] prefixPositions = Enumerable.Range(0, prefix.Length).ToArray();

        // Reference: one token at a time on its own cache.
        var expected = new float[batch.Length][];
        using (var cache = model.CreateKvCache(64))
        {
            using (model.Forward(prefix, prefixPositions, -1, cache, lastTokenLogitsOnly: true)) { }
            for (int i = 0; i < batch.Length; i++)
            {
                using ITensor step = model.Forward([batch[i]], [positions[i]], -1, cache);
                expected[i] = Row(step, 0, vocab);
            }
        }

        // Not widened: one row, as before.
        Assert.Equal(1, model.MaxAllRowLogitsLength);
        using (var cache = model.CreateKvCache(64))
        {
            using (model.Forward(prefix, prefixPositions, -1, cache, lastTokenLogitsOnly: true)) { }
            using ITensor narrow = model.Forward(batch, positions, -1, cache);
            Assert.Equal(vocab, narrow.ElementCount);
        }

        Assert.True(model.TrySetAllRowLogitsLimit(batch.Length));
        Assert.Equal(batch.Length, model.MaxAllRowLogitsLength);
        using (var cache = model.CreateKvCache(64))
        {
            using (ITensor prefill = model.Forward(prefix, prefixPositions, -1, cache, lastTokenLogitsOnly: true))
                Assert.Equal(vocab, prefill.ElementCount);

            using ITensor wide = model.Forward(batch, positions, -1, cache);
            Assert.Equal(batch.Length * vocab, wide.ElementCount);
            for (int i = 0; i < batch.Length; i++)
            {
                float[] row = Row(wide, i, vocab);
                float own = MaxAbsDiff(row, expected[i]);
                // The control: the bound has to be far below the distance to any other row, or a repeated or shifted row would pass.
                float other = Enumerable.Range(0, batch.Length).Where(j => j != i).Min(j => MaxAbsDiff(row, expected[j]));
                output.WriteLine($"row {i}: max|diff| to its own decode step {own:F4}, to the nearest other step {other:F4}");
                Assert.Equal(ArgMax(expected[i]), ArgMax(row));
                Assert.True(own < other / 10, $"row {i} differs by {own} from its decode step but only {other} from another");
            }

            // lastTokenLogitsOnly stays one row inside the limit.
            cache.Rollback(prefix.Length);
            using ITensor last = model.Forward(batch, positions, -1, cache, lastTokenLogitsOnly: true);
            Assert.Equal(vocab, last.ElementCount);
            Assert.Equal(ArgMax(expected[^1]), ArgMax(Row(last, 0, vocab)));
        }

        // Capped: a longer forward than the cap returns one row.
        Assert.True(model.TrySetAllRowLogitsLimit(100));
        Assert.Equal(CudaTransformerModel.MaxLogitRows, model.MaxAllRowLogitsLength);
        using (var cache = model.CreateKvCache(64))
        {
            using ITensor longer = model.Forward(prefix.AsSpan(0, 9), prefixPositions.AsSpan(0, 9), -1, cache);
            Assert.Equal(vocab, longer.ElementCount);
        }
    }
}
