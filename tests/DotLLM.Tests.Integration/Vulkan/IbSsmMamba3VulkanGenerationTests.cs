using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tokenizers;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Real-weight Vulkan-vs-CPU forward parity test for the Mamba-3 SSM
/// architecture on the <c>ib-ssm/mamba3-370M-10BT</c> checkpoint. Mirrors
/// <see cref="DotLLM.Tests.Integration.Models.Loaders.IbSsmMamba3GenerationTests"/>
/// but loads the same checkpoint twice — once into a CPU
/// <see cref="Mamba3TransformerModel"/> and once into a Vulkan
/// <see cref="VulkanMamba3TransformerModel"/> — and compares forward logits
/// per token position over a single prefill of a multi-token prompt.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a single prefill instead of growing-context reprefill.</b>
/// <see cref="VulkanMamba3TransformerModel"/> carries persistent SSM
/// recurrent state (<c>VulkanMamba3State</c>) across calls — there is no
/// public reset API today, and re-prefilling on a primed state would
/// silently accumulate the previous chunk's <c>ssm_state</c> into the new
/// pass. The CPU <see cref="Mamba3TransformerModel.Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int)"/>
/// overload (the one without an explicit
/// <see cref="Mamba3State"/>) creates a fresh state each call. To keep
/// both sides on equivalent semantics — fresh state, full sequence, no
/// chunk-boundary lookahead drift — we run a single prefill on each
/// backend over a fixed multi-token prompt and compare the per-position
/// logits.
/// </para>
/// <para>
/// <b>Coverage.</b> The single-prefill test exercises:
/// <list type="bullet">
///   <item>Embedding gather across a non-trivial vocab-id distribution.</item>
///   <item>Per-layer in_proj / out_proj matmuls (the only matmul-target
///     projections in Mamba-3) at prefill (GEMM) lengths.</item>
///   <item>Data-RoPE and SISO/MIMO scan kernels at <c>seqLen &gt; 1</c>.</item>
///   <item>RMSNorm + lm_head against the last-token reference. The Vulkan
///     Mamba-3 forward returns shape <c>[1, vocab]</c> (last token only —
///     it doesn't materialise per-position logits through lm_head), so
///     this test compares only the last row, which is the
///     generation-relevant position.</item>
/// </list>
/// Per-position prefill parity is covered by the Mamba-3 kernel-level unit
/// tests (CanonicalSsd / DataRope / ChunkBoundary). Chunk-boundary
/// state-resume is exercised by
/// <see cref="DotLLM.Tests.Unit.Vulkan.VulkanMamba3ChunkBoundaryF32KernelTests"/>.
/// Extending this integration test to multi-call decode would require a
/// public Reset API on <c>VulkanMamba3TransformerModel._recurrent</c> —
/// tracked as a follow-up.
/// </para>
/// <para>
/// <b>Tolerances.</b> Same numeric floors as
/// <see cref="VulkanTransformerModelTests"/> — 3.0 absolute on logits,
/// top-K=10 jaccard ≥ 0.5. Strict argmax-equality is required (single
/// last-token row, no decode trajectory). Mamba-3 carries an F32 forward
/// pass on both backends (the production loader does not attach a Q8_0
/// overlay), so drift here is pure F32 reduction-order noise — typically
/// tighter than the SmolLM-135M Q8_0 baseline this floor was calibrated on.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class IbSsmMamba3VulkanGenerationTests
{
    private const string CheckpointPathEnvVar = "DOTLLM_IBSSM_CHECKPOINT_PATH";
    private const string SafetensorsName = "model.safetensors";
    private const string ConventionalDir = "C:/temp/dotllm-ibssm";
    private const string UserProfileFallbackDir = "dotllm-ibssm-370m";

    // Tolerances mirrored from VulkanTransformerModelTests.
    private const float LogitsAbsTol = 3.0f;
    private const int TopKForJaccard = 10;
    private const float TopKJaccardFloor = 0.5f;

    private readonly ITestOutputHelper _output;

    public IbSsmMamba3VulkanGenerationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableFact]
    public void Mamba3_VulkanForward_MatchesCpuReference_OnPromptPrefill()
    {
        string? dir = ResolveCheckpointDir();
        if (dir is null)
        {
            _output.WriteLine(
                $"[SKIP] ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar} "
                + $"to the safetensors file or its directory, or place it at {ConventionalDir}/{SafetensorsName}.");
            return;
        }

        SkipIfVulkanUnavailable(out string spvDir);

        string weightsPath = Path.Combine(dir, SafetensorsName);
        _output.WriteLine($"Checkpoint dir: {dir}");

        ITokenizer? tokenizer = ModelLoader.LoadTokenizerFromHfDirectory(dir);
        Skip.If(tokenizer is null,
            $"No tokenizer.json found under {dir} — Mamba-3 parity test requires a usable tokenizer.");

        // ── CPU model ──────────────────────────────────────────────────
        var cpuLoadWatch = Stopwatch.StartNew();
        var (cpuModel, cpuFile, config) = ModelLoader.LoadFromSafetensors(weightsPath);
        cpuLoadWatch.Stop();
        Assert.Equal(Architecture.Mamba3, config.Architecture);
        _output.WriteLine(
            $"CPU load ({cpuLoadWatch.Elapsed.TotalSeconds:F1} s): "
            + $"vocab={config.VocabSize} hidden={config.HiddenSize} layers={config.NumLayers}");

        // ── Vulkan model ───────────────────────────────────────────────
        // Open a second safetensors source so the Vulkan loader has its own
        // mmap anchor. The CPU loader keeps the first source alive for its
        // weight views.
        var vkFile = OpenSafetensorsFile(weightsPath);
        VulkanMamba3TransformerModel? vkModel = null;
        try
        {
            var vkLoadWatch = Stopwatch.StartNew();
            vkModel = VulkanMamba3TransformerModel.LoadFromSafetensors(vkFile, config, spvDir);
            vkLoadWatch.Stop();
            _output.WriteLine($"Vulkan load ({vkLoadWatch.Elapsed.TotalSeconds:F1} s)");

            // A multi-position prompt — gives us several rows to compare.
            // Same prompt as the CPU test for consistency.
            int[] prompt = tokenizer!.Encode(
                "The quick brown fox jumps over the lazy dog and then runs to the capital of France");
            Assert.NotEmpty(prompt);
            foreach (int id in prompt) Assert.InRange(id, 0, config.VocabSize - 1);
            _output.WriteLine($"Prompt: {prompt.Length} tokens");

            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;

            // ── Forward pass on each backend ───────────────────────────
            var cpuFwdSw = Stopwatch.StartNew();
            using ITensor cpuLogits = cpuModel.Forward(prompt, positions, deviceId: -1);
            cpuFwdSw.Stop();
            _output.WriteLine($"CPU forward: {cpuFwdSw.Elapsed.TotalSeconds:F2} s");

            var vkFwdSw = Stopwatch.StartNew();
            using ITensor vkLogits = vkModel.Forward(prompt, positions, deviceId: -1);
            vkFwdSw.Stop();
            _output.WriteLine($"Vulkan forward: {vkFwdSw.Elapsed.TotalSeconds:F2} s");

            // Shape contract differs between the two backends: CPU returns
            // [seqLen, vocab] (per-position logits); Vulkan returns
            // [1, vocab] (last token only) because its forward downloads
            // only the LastTokenHidden row through the lm_head. We compare
            // the last token's logits — the load-bearing position for
            // generation. Per-position prefill parity is covered by the
            // Mamba-3 unit suite (CanonicalSsd / DataRope / ChunkBoundary
            // kernel tests) and the CPU
            // <see cref="IbSsmMamba3RealWeightsLoadTests"/> integration test.
            Assert.Equal(2, cpuLogits.Shape.Rank);
            Assert.Equal(2, vkLogits.Shape.Rank);
            Assert.Equal(prompt.Length, cpuLogits.Shape[0]);
            Assert.Equal(1, vkLogits.Shape[0]);
            Assert.Equal(config.VocabSize, cpuLogits.Shape[1]);
            Assert.Equal(config.VocabSize, vkLogits.Shape[1]);

            // Last-row comparison.
            float[] cpuLast = ExtractRow(cpuLogits, prompt.Length - 1, config.VocabSize);
            float[] vkLast = ExtractRow(vkLogits, 0, config.VocabSize);

            AssertLogitsMatch(cpuLast, vkLast, row: prompt.Length - 1);

            int cpuArg = Argmax(cpuLast);
            int vkArg = Argmax(vkLast);
            _output.WriteLine(
                $"  last row (pos {prompt.Length - 1}): cpu={cpuArg} vk={vkArg} "
                + (cpuArg == vkArg ? "[match]" : "[swap-in-topK]"));
            AssertStepArgmaxInOthersTopK(cpuLast, vkLast, row: prompt.Length - 1, k: 5);

            // For Mamba-3 we additionally require the last-token argmax to
            // match strictly — there is no "decode trajectory swap" noise
            // floor here because we run a single one-shot prefill. A
            // top-K-but-not-top-1 result on real weights with a clean F32
            // path would indicate a meaningful drift (e.g., scan kernel
            // accumulation order mismatch) and warrants investigation.
            Assert.True(cpuArg == vkArg,
                $"Last-token argmax differs: cpu={cpuArg} vk={vkArg}. Real-weight "
                + "Mamba-3 F32 forward should be argmax-equivalent on the last token.");
        }
        finally
        {
            vkModel?.Dispose();
            vkFile.Dispose();
            cpuModel.Dispose();
            cpuFile.Dispose();
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Helpers
    // ════════════════════════════════════════════════════════════════════

    private static unsafe float[] ExtractRow(ITensor logits, int row, int vocabSize)
    {
        var span = new ReadOnlySpan<float>(
            (void*)logits.DataPointer, logits.Shape[0] * vocabSize);
        float[] result = new float[vocabSize];
        span.Slice(row * vocabSize, vocabSize).CopyTo(result);
        return result;
    }

    private static ISafetensorsTensorSource OpenSafetensorsFile(string path)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) && File.Exists(path))
            return SafetensorsFile.Open(path);
        throw new FileNotFoundException(
            $"Expected a single-shard *.safetensors file for the Mamba-3 ib-ssm 370M checkpoint at {path}.",
            path);
    }

    private static string? ResolveCheckpointDir()
    {
        string? env = Environment.GetEnvironmentVariable(CheckpointPathEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (File.Exists(env))
            {
                string? d = Path.GetDirectoryName(env);
                if (d is not null && Directory.Exists(d)) return d;
            }
            if (Directory.Exists(env)) return env;
        }

        if (Directory.Exists(ConventionalDir) &&
            File.Exists(Path.Combine(ConventionalDir, SafetensorsName)))
            return ConventionalDir;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (!string.IsNullOrWhiteSpace(home))
        {
            string fallback = Path.Combine(home, UserProfileFallbackDir);
            if (Directory.Exists(fallback) &&
                File.Exists(Path.Combine(fallback, SafetensorsName)))
                return fallback;
        }
        return null;
    }

    // Tolerance helpers (mirrored from VulkanTransformerModelTests).

    private static int Argmax(float[] values)
    {
        int idx = 0;
        float best = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > best) { best = values[i]; idx = i; }
        }
        return idx;
    }

    private static int[] TopK(float[] values, int k)
    {
        int[] idx = new int[k];
        float[] vals = new float[k];
        for (int i = 0; i < k; i++) { idx[i] = -1; vals[i] = float.NegativeInfinity; }
        for (int i = 0; i < values.Length; i++)
        {
            float v = values[i];
            int insertAt = -1;
            for (int j = 0; j < k; j++)
            {
                if (v > vals[j]) { insertAt = j; break; }
            }
            if (insertAt < 0) continue;
            for (int j = k - 1; j > insertAt; j--) { vals[j] = vals[j - 1]; idx[j] = idx[j - 1]; }
            vals[insertAt] = v; idx[insertAt] = i;
        }
        return idx;
    }

    private static void AssertLogitsMatch(float[] expected, float[] actual, int row)
    {
        Assert.Equal(expected.Length, actual.Length);

        float maxAbs = 0;
        int worstIdx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
        }
        Assert.True(maxAbs <= LogitsAbsTol,
            $"row {row}: L∞ logit divergence {maxAbs:G6} > {LogitsAbsTol:G4} "
            + $"at idx {worstIdx} (cpu={expected[worstIdx]:G6} vk={actual[worstIdx]:G6})");

        int[] eTop = TopK(expected, TopKForJaccard);
        int[] aTop = TopK(actual, TopKForJaccard);
        var eSet = new HashSet<int>(eTop);
        int overlap = 0;
        foreach (int t in aTop) if (eSet.Contains(t)) overlap++;
        float jaccard = overlap / (float)TopKForJaccard;
        Assert.True(jaccard >= TopKJaccardFloor,
            $"row {row}: top-{TopKForJaccard} jaccard {jaccard:G3} < {TopKJaccardFloor:G3}. "
            + $"cpu={string.Join(',', eTop)} vk={string.Join(',', aTop)}");
    }

    private static void AssertStepArgmaxInOthersTopK(float[] cpu, float[] vk, int row, int k)
    {
        int cpuArg = Argmax(cpu);
        int vkArg = Argmax(vk);
        int[] cpuTopK = TopK(cpu, k);
        int[] vkTopK = TopK(vk, k);
        bool vkInCpu = Array.IndexOf(cpuTopK, vkArg) >= 0;
        bool cpuInVk = Array.IndexOf(vkTopK, cpuArg) >= 0;
        Assert.True(vkInCpu && cpuInVk,
            $"row {row}: argmax-in-top-{k} failed. "
            + $"cpu_arg={cpuArg} cpu_top={string.Join(',', cpuTopK)}; "
            + $"vk_arg={vkArg} vk_top={string.Join(',', vkTopK)}");
    }

    private static void SkipIfVulkanUnavailable(out string spvDir)
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        string? found = null;
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
            {
                found = full;
                break;
            }
        }
        Skip.If(found is null,
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
        spvDir = found!;
    }
}
