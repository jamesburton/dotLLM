using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #790 smoke: <c>--device vulkan</c> (as <c>run</c>/<c>chat</c>/<c>serve</c> resolve it, through <see cref="DeviceModelLoader"/>) really
/// produces the Vulkan model class, and the opposite arms are sensitive: a CPU request produces a CPU model, a Vulkan path that cannot load
/// makes an explicit request fail and an <c>auto</c> request land on the CPU with a warning. Proven by the loaded model's concrete type
/// and a real forward pass on each arm - not by <c>IsSupported</c>.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class DeviceSelectionVulkanSmokeTests(ITestOutputHelper output)
{
    private static (GgufFile Gguf, ModelConfig Config, string Path, long Bytes) OpenSmall()
    {
        FixtureLocation fx = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM2_135M_INSTRUCT_Q8_GGUF", "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q8_0.gguf");
        Skip.If(!fx.Found, fx.SkipMessage("SmolLM2-135M-Instruct Q8_0"));
        var gguf = GgufFile.Open(fx.Path!);
        return (gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), fx.Path!, new FileInfo(fx.Path!).Length);
    }

    private static void SkipIfVulkanNotServable() =>
        Skip.If(!DeviceEndpoint.Describe().Backends!.Any(b => b.Name == "vulkan" && b.Servable), "Vulkan is not servable on this machine.");

    private static int Argmax(DotLLM.Core.Tensors.ITensor logits)
    {
        int vocab = logits.Shape[logits.Shape.Rank - 1];
        int best = 0; float bv = float.NegativeInfinity;
        unsafe
        {
            var p = (float*)logits.DataPointer;
            // last row only
            long rows = logits.Shape.ElementCount / vocab;
            p += (rows - 1) * vocab;
            for (int i = 0; i < vocab; i++) if (p[i] > bv) { bv = p[i]; best = i; }
        }
        return best;
    }

    [SkippableFact]
    public void ExplicitVulkan_LoadsTheVulkanModelClass_AndRunsAForwardOnIt_WhileCpuLoadsTheCpuClass()
    {
        SkipIfVulkanNotServable();
        var (gguf, config, path, bytes) = OpenSmall();
        using (gguf)
        {
            var threading = new ThreadingConfig(0, 0);
            int[] ids = [1, 450, 7483, 310];
            int[] pos = [0, 1, 2, 3];

            var vk = DeviceModelLoader.Load(gguf, config, "vulkan", null, threading, bytes, Path.GetFileName(path), output.WriteLine, output.WriteLine);
            using var _vk = vk.Model;
            Assert.IsType<VulkanTransformerModel>(vk.Model);
            Assert.True(vk.IsVulkan);
            Assert.Equal("vulkan", vk.ResolvedDevice);
            Assert.Null(vk.Warning);
            Assert.NotNull(vk.KvCacheFactory);
            using var vkKv = vk.KvCacheFactory!(16);
            Assert.IsType<VulkanKvCache>(vkKv);
            using var vkLogits = vk.Model.Forward(ids, pos, deviceId: -1, vkKv);

            var cpu = DeviceModelLoader.Load(gguf, config, "cpu", null, threading, bytes, Path.GetFileName(path), output.WriteLine, output.WriteLine);
            using var _cpu = cpu.Model;
            // Control arm that could have disagreed: a CPU request must NOT be the Vulkan class.
            Assert.IsNotType<VulkanTransformerModel>(cpu.Model);
            Assert.False(cpu.IsVulkan);
            Assert.Equal("cpu", cpu.ResolvedDevice);
            using var cpuKv = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config), 16);
            using var cpuLogits = cpu.Model.Forward(ids, pos, deviceId: -1, cpuKv);

            // Same model, two backends: the next-token argmax agrees (the logits themselves differ in the low digits).
            Assert.Equal(Argmax(cpuLogits), Argmax(vkLogits));
        }
    }

    [SkippableFact]
    public void Auto_OnThisMachine_ResolvesToAGpuAndNeverWarns_WhenAGpuIsServable()
    {
        SkipIfVulkanNotServable();
        var (gguf, config, path, bytes) = OpenSmall();
        using (gguf)
        {
            var r = DeviceModelLoader.Load(gguf, config, "auto", null, new ThreadingConfig(0, 0), bytes, Path.GetFileName(path), output.WriteLine, output.WriteLine);
            using var _ = r.Model;
            output.WriteLine($"auto resolved to {r.ResolvedDevice}");
            Assert.NotEqual("cpu", r.ResolvedDevice);
            Assert.Null(r.Warning);
        }
    }

    [SkippableFact]
    public void BrokenVulkanPath_ExplicitVulkanFailsLoudly_AndAutoReachesCpuWithAWarning()
    {
        SkipIfVulkanNotServable();
        var (gguf, config, path, bytes) = OpenSmall();
        using (gguf)
        {
            var threading = new ThreadingConfig(0, 0);
            var devices = DeviceEndpoint.Describe();
            string name = Path.GetFileName(path);

            // Perturbation: the real devices, but the Vulkan load itself breaks (what a missing SPIR-V directory / driver fault does).
            (IModel, Func<int, DotLLM.Core.Attention.IKvCache>?) BreakVulkan(string device, int? layers) =>
                device == "vulkan"
                    ? throw new InvalidOperationException("SPIR-V blobs not found (simulated)")
                    : DeviceModelLoader.LoadExact(gguf, config, device, layers, threading, output.WriteLine, output.WriteLine);

            var ex = Assert.Throws<DeviceUnavailableException>(() =>
                DeviceModelLoader.LoadWith(gguf, config, "vulkan", null, threading, bytes, name, output.WriteLine, output.WriteLine, devices, BreakVulkan));
            Assert.Contains("SPIR-V blobs not found", ex.Message);
            Assert.Contains("--device cpu", ex.Message);

            // Auto with CUDA absent (this box) falls Vulkan -> CPU, and must say so.
            Skip.If(devices.Backends!.Any(b => b.Name == "cuda" && b.Servable), "CUDA present: auto would try it first.");
            var auto = DeviceModelLoader.LoadWith(gguf, config, "auto", null, threading, bytes, name, output.WriteLine, output.WriteLine, devices, BreakVulkan);
            using var _ = auto.Model;
            Assert.Equal("cpu", auto.ResolvedDevice);
            Assert.False(auto.IsVulkan);
            Assert.NotNull(auto.Warning);
            Assert.Contains("SPIR-V blobs not found", auto.Warning);
        }
    }
}
