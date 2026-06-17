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

    /// <summary>
    /// CUDA gemma4 AR parity: load the synthetic gemma4 fixture on CUDA, run the
    /// forward, and compare against the CPU reference (EXACT argmax + logits within
    /// tolerance — CUDA's F32 reduction order differs from the CPU, so a checksum is
    /// NOT valid; tolerance is the cross-backend golden). Skips cleanly on this AMD
    /// dev box (no NVIDIA GPU) — runs on the T5500. Requires the regenerated
    /// gemma4_f32.ptx (native/build.{sh,ps1} with nvcc).
    /// </summary>
    [SkippableFact]
    public void Cuda_Gemma4_MatchesCpuReference()
    {
        Skip.IfNot(
            CudaDevice.IsAvailable(),
            "No CUDA device/driver on this host (CUDA is build-verify-only here).");

        string path = WriteFixture();
        int[] ids = { 2, 7, 8, 9 }; // synthetic fixture BOS = 2
        int[] pos = { 0, 1, 2, 3 };
        try
        {
            // CPU reference logits. The CPU IModel.Forward returns logits for ALL
            // seqLen positions ([seqLen, vocab]); the CUDA gemma4 forward (like every
            // CUDA forward) applies the final norm + LM head to the LAST token only
            // ([1, vocab]). Compare the LAST-position row of both so the two are
            // shape-compatible — an autoregressive next-token prediction.
            float[] cpuLogits;
            {
                var (cpuModel, cpuGguf, _) = DotLLM.Models.ModelLoader.LoadFromGguf(
                    path, DotLLM.Core.Configuration.ThreadingConfig.SingleThreaded);
                using (cpuGguf)
                using (cpuModel)
                using (var logits = cpuModel.Forward(ids, pos, -1, kvCache: null))
                {
                    cpuLogits = ToFloatArray(logits);
                }
            }

            // CUDA logits (last token only).
            float[] cudaLogits;
            {
                var (model, gguf, _) = CudaModelLoader.LoadFromGguf(path, deviceId: 0);
                using (gguf)
                using (model)
                using (var logits = model.Forward(ids, pos, -1, kvCache: null))
                {
                    cudaLogits = ToFloatArray(logits);
                }
            }

            // Slice the CPU logits to the last position's vocab row so both vectors
            // describe the same (final) token's distribution. vocab == CUDA length.
            int vocab = cudaLogits.Length;
            Assert.True(cpuLogits.Length % vocab == 0,
                $"CPU logit length {cpuLogits.Length} not a multiple of CUDA vocab {vocab}.");
            int lastRow = cpuLogits.Length - vocab;
            var cpuLast = new float[vocab];
            Array.Copy(cpuLogits, lastRow, cpuLast, 0, vocab);
            cpuLogits = cpuLast;

            int cpuArgmax = Argmax(cpuLogits);
            int cudaArgmax = Argmax(cudaLogits);

            _output.WriteLine($"CPU argmax={cpuArgmax} logit={cpuLogits[cpuArgmax]:F4}; "
                + $"CUDA argmax={cudaArgmax} logit={cudaLogits[cudaArgmax]:F4}");

            Assert.Equal(cpuArgmax, cudaArgmax);

            // Logits within tolerance (abs 6e-2 / rel 5e-3) — same envelope as the
            // Vulkan parity test. CUDA F32 reduction order differs from the CPU.
            int n = Math.Min(cpuLogits.Length, cudaLogits.Length);
            float maxAbs = 0f;
            for (int i = 0; i < n; i++)
            {
                float diff = MathF.Abs(cpuLogits[i] - cudaLogits[i]);
                float tol = 6e-2f + 5e-3f * MathF.Abs(cpuLogits[i]);
                maxAbs = MathF.Max(maxAbs, diff - tol);
                Assert.True(diff <= tol,
                    $"logit[{i}] diverged: cpu={cpuLogits[i]:F5} cuda={cudaLogits[i]:F5} diff={diff:F5} tol={tol:F5}");
            }
            float rawMax = 0f;
            for (int i = 0; i < n; i++) rawMax = MathF.Max(rawMax, MathF.Abs(cpuLogits[i] - cudaLogits[i]));
            _output.WriteLine($"max (diff - tol) over {n} logits = {maxAbs:E3} (<=0 ⇒ all within tolerance); raw max|diff|={rawMax:E3}");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    private static unsafe float[] ToFloatArray(DotLLM.Core.Tensors.ITensor t)
    {
        int n = (int)t.Shape.ElementCount;
        var arr = new float[n];
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) arr[i] = p[i];
        return arr;
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
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
