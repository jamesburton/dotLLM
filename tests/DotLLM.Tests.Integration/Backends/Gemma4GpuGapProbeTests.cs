using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models;
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

    private static string WriteFixture() => WriteFixture(SyntheticGemma4Gguf.Tiny);

    private static string WriteFixture(SyntheticGemma4Config cfg)
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_gemma4_gpuprobe_{Guid.NewGuid():N}.gguf");
        return SyntheticGemma4Gguf.WriteGemma4(path, cfg);
    }

    /// <summary>
    /// CPU↔Vulkan parity for the autoregressive gemma4 forward (cacheless single
    /// forward over the tiny synthetic fixture: dual head dim, V-from-K global
    /// layer, dual-FFN dense+MoE, custom router, GeGLU experts, layer scale, final
    /// softcap). Asserts last-token logits agree within the standard Vulkan
    /// reduction-order envelope. (The fixture's experts are Q4_K/Q5_1 — host-
    /// dequantised on the Vulkan side, in-kernel-dequantised on CPU — so the
    /// expert weights are bit-identical and only GPU-vs-CPU accumulation order
    /// consumes tolerance.)
    /// </summary>
    [SkippableFact]
    public void Vulkan_Gemma4_MatchesCpuReference()
        => RunVulkanParity(SyntheticGemma4Gguf.Tiny);

    /// <summary>
    /// CPU↔Vulkan parity on the <see cref="SyntheticGemma4Gguf.Real26BLike"/> fixture, which
    /// (unlike Tiny/Bench) sets <b>GlobalHeadDim = 512</b> and a full-head <c>rope.dimension_count</c>
    /// so the global (full-attention) layers run PARTIAL NeoX rope at the real 26B's head dim
    /// (rotate 128 of 512 dims, freq denominator = the full 512 head dim) AND the real dual schedule
    /// (sliding 256/8, global 512/2, V-from-K). This exercises the partial-rope
    /// frequency-denominator path the Tiny fixture (GlobalHeadDim 32) cannot — the bug that flipped
    /// the real-26B next token. The greedy argmax must agree (structural correctness); the per-logit
    /// envelope is WIDER than Tiny because the head_dim-512 dot products + dual-FFN + final softcap
    /// near zero accumulate more F32 reduction-order drift over the alternating-stride layers.
    /// </summary>
    [SkippableFact]
    public void Vulkan_Gemma4_Real26BLike_MatchesCpuReference()
        => RunVulkanParity(SyntheticGemma4Gguf.Real26BLike, absTol: 1.0f, relTol: 5.0e-3f);

    private void RunVulkanParity(SyntheticGemma4Config fixtureCfg, float absTol = 6.0e-2f, float relTol = 5.0e-3f)
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        string path = WriteFixture(fixtureCfg);
        try
        {
            int[] ids = { 2, 7, 8, 9, 5, 6 }; // synthetic fixture BOS = 2
            int[] pos = { 0, 1, 2, 3, 4, 5 };

            // ── CPU oracle ────────────────────────────────────────────
            float[] cpuLast;
            int vocab;
            {
                var (model, gguf, cfg) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
                using (gguf)
                using (model)
                {
                    vocab = cfg.VocabSize;
                    using ITensor logits = model.Forward(ids, pos, -1, null, null, AttentionMaskSpec.Causal);
                    cpuLast = LastRow(logits, vocab);
                }
            }

            // ── Vulkan under test ─────────────────────────────────────
            float[] vkLast;
            {
                using var gguf = GgufFile.Open(path);
                var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
                Assert.Equal(DotLLM.Core.Configuration.Architecture.Gemma4, cfg.Architecture);
                using var model = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spvDir);
                using ITensor logits = model.Forward(ids, pos, -1, kvCache: null);
                vkLast = LastRow(logits, vocab);
            }

            // Structural guard: the greedy next-token (argmax) must agree — this
            // is independent of the small-logit reduction-order drift below and
            // catches any real graph divergence (a broken op flips the argmax).
            int cpuArg = ArgMax(cpuLast), vkArg = ArgMax(vkLast);
            Assert.Equal(cpuArg, vkArg);

            // Per-logit envelope (absTol/relTol passed by the caller). gemma4's dual-FFN
            // + per-head norms + partial rope accumulate per-layer F32 reduction-order drift,
            // and the final softcap pushes logits near zero where relative error amplifies.
            // The argmax guard above is the exact structural check; this bar catches gross
            // per-op divergence without flagging benign reduction-order drift.
            int worst = -1; float worstDiff = 0;
            for (int c = 0; c < vocab; c++)
            {
                float diff = MathF.Abs(cpuLast[c] - vkLast[c]);
                if (diff > worstDiff) { worstDiff = diff; worst = c; }
            }
            for (int c = 0; c < vocab; c++)
            {
                float cpu = cpuLast[c], vk = vkLast[c];
                float bar = absTol + relTol * MathF.Abs(cpu);
                Assert.True(MathF.Abs(cpu - vk) <= bar,
                    $"col {c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={MathF.Abs(cpu - vk):E3} > {bar:E3}); "
                    + $"argmax cpu={cpuArg} vk={vkArg}; worst |diff|={worstDiff:E3} @ col {worst}");
            }
            _output.WriteLine($"gemma4 CPU↔Vulkan parity OK: argmax={cpuArg}, worst |diff|={worstDiff:E3} @ col {worst} over {vocab} logits.");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var all = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(all);
        // CPU returns [seqLen, vocab]; Vulkan returns [1, vocab]. Either way the
        // last `vocab` elements are the final token's logits (the sampler input).
        var row = new float[vocab];
        Array.Copy(all, total - vocab, row, 0, vocab);
        return row;
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
