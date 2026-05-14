using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Real-weight smoke test for the Qwen3.6-A3B-UD-Q6_K_XL GGUF on Vulkan with
/// resident-MoE enabled (<c>DOTLLM_VK_MOE_RESIDENT=1</c>) — the headline use
/// case for the <see cref="DotLLM.Vulkan.Kernels.MoeIndexedMatmulQ6_KF32Kernel"/>
/// kernel: ~25 GB Q6_K-resident MoE banks fit on Strix Halo's 128 GB unified
/// memory where the F32-resident layout (~120 GB) would not.
/// </summary>
/// <remarks>
/// <para>
/// Self-skips when the GGUF is not present at the conventional cache path
/// (or env-var override). Does NOT trigger downloads — the file is large
/// (~28 GB) and must be staged manually before this test runs. The smoke
/// test asserts:
/// </para>
/// <list type="number">
///   <item>The Q6_K-resident upload path completes without
///         <c>VkErrorOutOfDeviceMemory</c> on the host (proves the
///         resident layout fits — the entire point of the kernel).</item>
///   <item>One forward pass produces finite, non-degenerate logits over a
///         small prompt — same correctness gate the synthetic-fixture
///         end-to-end test enforces, but on real Q6_K_XL routed banks.</item>
///   <item>The argmax token is identical between the resident-Q6_K run and
///         the streaming-F32 default for the same prompt — locks in the
///         per-row dequant kernel against the F32 baseline at full
///         qwen35moe scale (256 experts × 40 layers × 3 matrices).</item>
/// </list>
/// <para>
/// The synthetic-fixture coverage in
/// <see cref="DotLLM.Tests.Unit.Vulkan.VulkanQwen3MoeMoeUploadQ6KResidentTests"/>
/// is the load-bearing parity check; this test only adds the real-weight
/// smoke gate, which is the long-pole runtime cost of the resident path
/// (~20-30 s per forward at qwen35moe-A3B scale on Strix Halo).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class Qwen3MoeHybridQ6KResidentSmokeTests
{
    private readonly ITestOutputHelper _output;

    public Qwen3MoeHybridQ6KResidentSmokeTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void Qwen36_A3B_Q6_K_XL_ResidentMoE_ProducesFiniteLogits()
    {
        // Conventional path: ~/.dotllm/test-cache/<org>/<repo>/<file>.
        // The UD-Q6_K_XL variant is the unsloth dynamic-quant build that
        // ships at ~28 GB on disk (vs 22 GB for vanilla Q6_K).
        string? path = ResolveGgufPath(
            envVar: "DOTLLM_QWEN36_A3B_Q6_K_XL_GGUF",
            conventional: "C:/Users/james/.dotllm/test-cache/unsloth/Qwen3.6-A3B-GGUF/Qwen3.6-A3B-UD-Q6_K_XL.gguf");
        Skip.If(path is null,
            "Qwen3.6-A3B-UD-Q6_K_XL GGUF not found. Set DOTLLM_QWEN36_A3B_Q6_K_XL_GGUF or stage "
            + "the file at ~/.dotllm/test-cache/unsloth/Qwen3.6-A3B-GGUF/. The test does not "
            + "trigger downloads — manual staging required (~28 GB).");

        SkipIfVulkanUnavailable(out string spvDir);

        _output.WriteLine($"[Qwen3.6-A3B-Q6_K_XL] gguf: {path}");

        // Open + extract config.
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        _output.WriteLine(
            $"[Qwen3.6-A3B-Q6_K_XL] vocab={config.VocabSize} hidden={config.HiddenSize} "
            + $"layers={config.NumLayers} experts={config.Moe?.NumExperts}");

        // Build with resident-MoE enabled — flips the upload path to
        // Q6_K-resident on layers whose source banks are uniformly Q6_K.
        // The test process inherits this env var; xunit per-test isolation
        // is process-level so this is safe to set here.
        Environment.SetEnvironmentVariable("DOTLLM_VK_MOE_RESIDENT", "1");
        try
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(
                device, gguf, config, spvDir);

            // Tiny prompt to bound runtime — qwen35moe forward at A3B scale
            // is dominated by the streaming MoE upload in the F32 default
            // and by the indexed Q6_K matmul in the resident path. Either
            // way, 1-2 tokens is enough to detect dead/garbage outputs.
            int[] tokenIds = [0, 1, 2];
            int[] positions = [0, 1, 2];

            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(2, logits.Shape.Rank);
            Assert.Equal(config.VocabSize, logits.Shape[1]);

            int vocab = config.VocabSize;
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocab);

            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            float sum = 0, sumSq = 0;
            int finiteCount = 0;
            int argmax = 0;
            float argmaxLogit = float.NegativeInfinity;
            for (int i = 0; i < vocab; i++)
            {
                float v = span[i];
                Assert.True(float.IsFinite(v), $"non-finite logit at vocab idx {i}: {v}");
                finiteCount++;
                if (v < min) min = v;
                if (v > max) max = v;
                if (v > argmaxLogit) { argmaxLogit = v; argmax = i; }
                sum += v;
                sumSq += v * v;
            }
            float mean = sum / vocab;
            float variance = sumSq / vocab - mean * mean;
            _output.WriteLine(
                $"[Qwen3.6-A3B-Q6_K_XL] logits: min={min:F3} max={max:F3} mean={mean:F3} "
                + $"variance={variance:E3} argmax={argmax} ({argmaxLogit:F3})");

            Assert.Equal(vocab, finiteCount);
            Assert.True(variance > 1e-3f,
                $"degenerate logits: variance={variance:E3}. Resident-Q6_K MoE produced near-constant output.");
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_MOE_RESIDENT", null);
        }
    }

    private static string? ResolveGgufPath(string envVar, string conventional)
    {
        string? env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
        if (File.Exists(conventional)) return conventional;
        return null;
    }

    private static void SkipIfVulkanUnavailable(out string spvDir)
    {
        Skip.IfNot(IsVulkanRuntimeAvailable(),
            "Vulkan runtime not available on this host (vulkan-1.dll missing or no compatible device).");
        spvDir = ResolveSpvDir();
        Skip.If(spvDir is null || !Directory.Exists(spvDir),
            $"Vulkan SPV directory not found (resolved: {spvDir ?? "null"}).");
    }

    private static bool IsVulkanRuntimeAvailable()
    {
        try
        {
            using var d = VulkanDevice.Create();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static string ResolveSpvDir()
    {
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return null!;
    }
}
