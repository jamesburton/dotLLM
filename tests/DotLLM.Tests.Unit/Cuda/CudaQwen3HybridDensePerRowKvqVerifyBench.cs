using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1068 measurement: the latency of one MTP verify (K+1 = 4 rows) over q8_0 KV at a given history depth, per-row
/// fused attention against dequant-to-scratch. Serving numbers mix this with the adaptive MTP gate (it backs off when
/// drafts stop paying), so this times the verify forward alone, from the same state, many times.
/// </summary>
[Trait("Category", "Benchmark")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDensePerRowKvqVerifyBench
{
    private const int Rows = 4;
    private const int Iterations = 30;

    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDensePerRowKvqVerifyBench(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(4096)]
    [InlineData(16384)]
    [InlineData(32768)]
    public void VerifyLatency_PerRowVsScratch(int depth)
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_BONSAI2_MTP_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "set DOTLLM_BONSAI2_MTP_GGUF");
        string? ptxDir = PtxDir();
        Skip.If(ptxDir is null, "No CUDA GPU or PTX");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);
        model.SetKvStorageForTest(KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        using var kv = model.CreateKvCache(depth + 64);

        int[] prompt = new int[depth];
        for (int i = 0; i < depth; i++) prompt[i] = 1 + (i * 7919) % 30000;
        using (model.Forward(prompt, Enumerable.Range(0, depth).ToArray(), deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        object? state = model.CheckpointRecurrentState();

        int[] rows = [1042, 77, 9001, 314];
        int[] positions = Enumerable.Range(depth, Rows).ToArray();
        foreach (bool perRow in new[] { true, false, true, false })
        {
            model.PerRowKvqForTest = perRow;
            for (int w = 0; w < 3; w++) Verify();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < Iterations; i++) Verify();
            sw.Stop();
            _out.WriteLine($"depth {depth}: {(perRow ? "per-row" : "scratch")} verify {sw.Elapsed.TotalMilliseconds / Iterations:F2} ms");
        }
        (state as IDisposable)?.Dispose();

        void Verify()
        {
            kv.Rollback(depth);
            model.RestoreRecurrentState(state);
            using ITensor _ = model.Forward(rows, positions, deviceId: -1, kv, lastTokenLogitsOnly: false);
        }
    }

    private static string? PtxDir()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return null;
        NativeLibrary.Free(h);
        if (!CudaDevice.IsAvailable()) return null;
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            string full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }
}
