using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// gemma4-on-GPU GAP PROBE. Loads the portable synthetic <c>gemma4</c> fixture
/// (<see cref="SyntheticGemma4Gguf"/>) and ATTEMPTS to run it on the Vulkan backend (and
/// CUDA when an NVIDIA device exists), capturing exactly WHERE and WHY it fails. gemma4 is
/// NOT wired into either GPU backend yet — so this is a data-gathering diagnostic for
/// <c>docs/diffusiongemma/GEMMA4-GPU-GAPS.md</c>, not a pass/fail correctness gate.
/// </summary>
/// <remarks>
/// The probe ASSERTS that the GPU load/forward throws (the expected pre-gemma4-on-GPU
/// state) and records the throwing type + message via the test output, so a future PR that
/// lands gemma4 on a GPU backend will flip this probe to failing — a built-in reminder to
/// update the gap report and convert this into a real parity test.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class Gemma4GpuGapProbeTests
{
    private readonly ITestOutputHelper _output;

    public Gemma4GpuGapProbeTests(ITestOutputHelper output) => _output = output;

    private static string WriteFixture()
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_gemma4_gpuprobe_{Guid.NewGuid():N}.gguf");
        return SyntheticGemma4Gguf.WriteGemma4(path, SyntheticGemma4Gguf.Tiny);
    }

    [SkippableFact]
    public void Vulkan_Gemma4_Fixture_FailsWithCapturedGap()
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        string path = WriteFixture();
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

            // Sanity: the GGUF parses and config-extracts as gemma4 (the CPU-side metadata
            // mapping is wired). The gap is purely in the GPU forward path.
            Assert.Equal(DotLLM.Core.Configuration.Architecture.Gemma4, config.Architecture);
            _output.WriteLine(
                $"gemma4 config extracted: layers={config.NumLayers} hidden={config.HiddenSize} " +
                $"gemma4DualFfn={config.Gemma4DualFfn} moe={(config.Moe is not null)}");

            // Attempt to load + run gemma4 on Vulkan. Capture wherever it fails.
            var ex = Record.Exception(() =>
            {
                using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
                int[] ids = { 2, 7, 8, 9 }; // synthetic fixture BOS = 2
                int[] pos = { 0, 1, 2, 3 };
                using var _ = model.Forward(ids, pos, -1, kvCache: null);
            });

            Assert.NotNull(ex);
            _output.WriteLine($"VULKAN gemma4 GAP: {ex!.GetType().FullName}");
            _output.WriteLine($"VULKAN gemma4 GAP message: {ex.Message}");
            if (ex.StackTrace is not null)
                _output.WriteLine($"VULKAN gemma4 GAP frames:\n{DotLlmFrames(ex.StackTrace)}");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    [SkippableFact]
    public void Cuda_Gemma4_Fixture_FailsWithCapturedGap()
    {
        // CUDA cannot RUN on this host (no NVIDIA GPU); this probe only executes on a CUDA box
        // (e.g. the T5500). Build-verify of the managed CUDA path is covered by the solution build.
        Skip.IfNot(
            CudaDevice.IsAvailable(),
            "No CUDA device/driver on this host (CUDA is build-verify-only here).");

        string path = WriteFixture();
        try
        {
            var ex = Record.Exception(() =>
            {
                var (model, gguf, _) = CudaModelLoader.LoadFromGguf(path, deviceId: 0);
                using (gguf)
                using (model)
                {
                    int[] ids = { 2, 7, 8, 9 };
                    int[] pos = { 0, 1, 2, 3 };
                    using var _ = model.Forward(ids, pos, -1, kvCache: null);
                }
            });

            Assert.NotNull(ex);
            _output.WriteLine($"CUDA gemma4 GAP: {ex!.GetType().FullName}");
            _output.WriteLine($"CUDA gemma4 GAP message: {ex.Message}");
            if (ex.StackTrace is not null)
                _output.WriteLine($"CUDA gemma4 GAP frames:\n{DotLlmFrames(ex.StackTrace)}");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    // The throw site is often a Memmove deep in a copy; the meaningful gap signal is the
    // first few DotLLM frames (which Vulkan/CUDA routine assumed a weight that gemma4 stores
    // elsewhere). Print up to the first 6 DotLLM frames.
    private static string DotLlmFrames(string stackTrace)
    {
        var picked = new List<string>();
        foreach (var line in stackTrace.Split('\n'))
        {
            string t = line.Trim();
            if (t.StartsWith("at ", StringComparison.Ordinal) &&
                t.Contains("DotLLM.", StringComparison.Ordinal))
            {
                picked.Add("  " + t);
                if (picked.Count >= 6) break;
            }
        }
        return picked.Count > 0 ? string.Join('\n', picked) : "  (no DotLLM frames in stack)";
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
    }
}
