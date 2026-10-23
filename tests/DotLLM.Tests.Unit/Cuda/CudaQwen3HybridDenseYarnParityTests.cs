using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1101: the CUDA Qwen3.5/3.8 hybrid host rotates exactly like the CPU oracle with <c>--rope-scaling yarn</c>,
/// at positions past the native window: trunk prefill and decode, the MTP head, and Bonsai-2's own RoPE at positions past
/// 262,144. See <see cref="HybridYarnParity"/> for why the fixture discriminates.
/// </summary>
/// <remarks>
/// Mutant runs (recorded in the PR): launching RoPE without the YaRN inverse frequencies (the pre-#1101 host) and with
/// <c>ropeMscale = 1</c> both fail these tests by orders of magnitude over the tolerance.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseYarnParityTests : IDisposable
{
    // F32 fixture: measured CPU-vs-GPU gap is ~1e-4 (cached, CUDA F16 KV rows) and ~1e-7 (cacheless). 1e-3 keeps a ~9x
    // margin over that while staying far below the ~2.5e-2 YaRN on/off difference (HybridYarnParity.SignalFloor).
    private const float AbsTol = HybridYarnParity.AbsTol;
    private const float RelTol = HybridYarnParity.RelTol;

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-qwen35-yarn-{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseYarnParityTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>600-token prefill (past the 512 native window), 4 decode steps and 3 MTP drafts match the CPU oracle.</summary>
    [SkippableFact]
    public void Cached_PastNativeWindow_MatchesCpu()
    {
        string ptxDir = RequireCuda();
        string path = HybridYarnParity.WriteDiscriminating(_scratch);
        const int Prompt = 600, Decode = 4, Mtp = 3;

        HybridYarnParity.Result cpu;
        using (var gguf = GgufFile.Open(path))
        {
            var config = HybridYarnParity.Config(gguf, yarn: true);
            using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            cpu = HybridYarnParity.RunCached(model, kv, Prompt, Decode, Mtp);
        }

        HybridYarnParity.Result gpu;
        using (var gguf = GgufFile.Open(path))
        {
            var config = HybridYarnParity.Config(gguf, yarn: true);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            Assert.Equal(config.MaxSequenceLength, model.NativeMaxSequenceLength);
            using var kv = model.CreateKvCache(config.MaxSequenceLength);
            gpu = HybridYarnParity.RunCached(model, kv, Prompt, Decode, Mtp);
        }

        float trunk = HybridYarnParity.AssertClose(cpu.Trunk, gpu.Trunk, AbsTol, RelTol, "trunk");
        float mtp = HybridYarnParity.AssertClose(cpu.Mtp, gpu.Mtp, AbsTol, RelTol, "mtp");
        _out.WriteLine($"max |cpu - cuda|: trunk {trunk:E3}, mtp {mtp:E3}");
    }

    /// <summary>Bonsai-2's own RoPE with YaRN (window 1,048,576), cacheless at positions past 262,144.</summary>
    [SkippableTheory]
    [InlineData(300_000, 16)]
    [InlineData(1_048_000, 16)]
    public void Uncached_BonsaiRope_PastNativeWindow_MatchesCpu(int start, int length)
    {
        string ptxDir = RequireCuda();
        string path = HybridYarnParity.WriteBonsaiLike(_scratch);

        List<float[]> cpu;
        using (var gguf = GgufFile.Open(path))
        {
            var config = HybridYarnParity.Config(gguf, yarn: true);
            Assert.Equal(1_048_576, config.MaxSequenceLength);
            using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
            cpu = HybridYarnParity.RunUncached(model, start, length);
        }

        List<float[]> gpu;
        using (var gguf = GgufFile.Open(path))
        {
            var config = HybridYarnParity.Config(gguf, yarn: true);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            gpu = HybridYarnParity.RunUncached(model, start, length);
        }

        float worst = HybridYarnParity.AssertClose(cpu, gpu, AbsTol, RelTol, $"positions {start}..{start + length - 1}");
        _out.WriteLine($"max |cpu - cuda| at {start}: {worst:E3}");
    }

    private static string RequireCuda()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        bool present = NativeLibrary.TryLoad(lib, out nint h);
        if (present) NativeLibrary.Free(h);
        Skip.IfNot(present && CudaDevice.IsAvailable(), "No CUDA GPU available");
        string? ptxDir = null;
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            string full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) { ptxDir = full; break; }
        }
        Skip.If(ptxDir is null, "PTX files not found");
        return ptxDir!;
    }
}
