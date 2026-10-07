using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Issue #790, CUDA half (runs on a CUDA host - this repo's Strix Halo box has none, so it skips there): every spelling of a CUDA request
/// (<c>gpu</c>, <c>cuda</c>, <c>cuda:0</c>) and <c>auto</c> really produce the CUDA model class through <see cref="DeviceModelLoader"/>, and
/// the next-token argmax matches the CPU model. Before #790 only a "gpu"-prefixed string reached CUDA; "cuda" silently loaded the CPU.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class DeviceSelectionCudaSmokeTests(ITestOutputHelper output)
{
    private static (GgufFile Gguf, ModelConfig Config, string Path, long Bytes) OpenSmall()
    {
        FixtureLocation fx = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("Llama-3.2-1B Q8_0"));
        var gguf = GgufFile.Open(fx.Path!);
        return (gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), fx.Path!, new FileInfo(fx.Path!).Length);
    }

    private static void SkipIfNoCuda() =>
        Skip.If(!DeviceEndpoint.Describe().Backends!.Any(b => b.Name == "cuda" && b.Servable), "No CUDA device on this machine.");

    private static int Argmax(DotLLM.Core.Tensors.ITensor logits)
    {
        int vocab = logits.Shape[logits.Shape.Rank - 1];
        int best = 0; float bv = float.NegativeInfinity;
        unsafe
        {
            var p = (float*)logits.DataPointer;
            p += (logits.Shape.ElementCount / vocab - 1) * vocab;
            for (int i = 0; i < vocab; i++) if (p[i] > bv) { bv = p[i]; best = i; }
        }
        return best;
    }

    [SkippableTheory]
    [InlineData("gpu")]
    [InlineData("cuda")]
    [InlineData("cuda:0")]
    [InlineData("auto")]
    public void CudaRequest_LoadsTheCudaModelClass_NotCpu_AndMatchesCpuArgmax(string requested)
    {
        SkipIfNoCuda();
        var (gguf, config, path, bytes) = OpenSmall();
        using (gguf)
        {
            var threading = new ThreadingConfig(0, 0);
            int[] ids = [1, 450, 7483, 310];
            int[] pos = [0, 1, 2, 3];

            var r = DeviceModelLoader.Load(gguf, config, requested, null, threading, bytes, Path.GetFileName(path), output.WriteLine, output.WriteLine);
            using var _ = r.Model;
            output.WriteLine($"{requested} -> {r.ResolvedDevice} ({r.Model.GetType().Name})");
            Assert.IsType<CudaTransformerModel>(r.Model);
            Assert.Equal("gpu:0", r.ResolvedDevice);
            Assert.Null(r.Warning);
            using var kv = r.KvCacheFactory!(16);
            using var cudaLogits = r.Model.Forward(ids, pos, deviceId: -1, kv);

            var cpu = DeviceModelLoader.Load(gguf, config, "cpu", null, threading, bytes, Path.GetFileName(path), output.WriteLine, output.WriteLine);
            using var _cpu = cpu.Model;
            Assert.IsNotType<CudaTransformerModel>(cpu.Model);   // control arm that could have disagreed
            using var cpuKv = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config), 16);
            using var cpuLogits = cpu.Model.Forward(ids, pos, deviceId: -1, cpuKv);
            Assert.Equal(Argmax(cpuLogits), Argmax(cudaLogits));
        }
    }

    [SkippableFact]
    public void BrokenCuda_ExplicitFailsLoudlyWithNumbers_AndAutoReachesCpuWithAWarning()
    {
        SkipIfNoCuda();
        var (gguf, config, path, bytes) = OpenSmall();
        using (gguf)
        {
            var threading = new ThreadingConfig(0, 0);
            var devices = DeviceEndpoint.Describe();
            string name = Path.GetFileName(path);
            (IModel, Func<int, DotLLM.Core.Attention.IKvCache>?) BreakGpu(string device, int? layers) =>
                device.StartsWith("gpu", StringComparison.Ordinal)
                    ? throw new InvalidOperationException("CUDA_ERROR_OUT_OF_MEMORY (simulated)")
                    : DeviceModelLoader.LoadExact(gguf, config, device, layers, threading, output.WriteLine, output.WriteLine);

            var ex = Assert.Throws<DeviceUnavailableException>(() =>
                DeviceModelLoader.LoadWith(gguf, config, "cuda", null, threading, bytes, name, output.WriteLine, output.WriteLine, devices, BreakGpu));
            Assert.Contains("CUDA_ERROR_OUT_OF_MEMORY", ex.Message);
            Assert.Contains("GiB device", ex.Message);     // real VRAM figure for a present device
            Assert.Contains("--device cpu", ex.Message);

            // Auto must try CUDA first on a CUDA host, fail it, optionally fall to Vulkan, and end on CPU with a warning naming the CUDA error.
            var auto = DeviceModelLoader.LoadWith(gguf, config, "auto", null, threading, bytes, name, output.WriteLine, output.WriteLine, devices,
                (d, l) => d == "vulkan" ? throw new InvalidOperationException("vulkan skipped in this test") : BreakGpu(d, l));
            using var _ = auto.Model;
            Assert.Equal("cpu", auto.ResolvedDevice);
            Assert.Contains("gpu:0", auto.Warning);
            Assert.Contains("CUDA_ERROR_OUT_OF_MEMORY", auto.Warning);
        }
    }
}
