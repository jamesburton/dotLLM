using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #801: on AMD the K-quant (Q4_K/Q5_K/Q6_K) dp4a MMVQ decode kernels are 28-37% slower end to end than the coalesced F32-in
/// GEMVs (Gemma-4-31B Q4_K_M 7.0 vs 9.8 tok/s), so the generic <see cref="VulkanTransformerModel"/> must NOT wire them by default
/// there. The policy table is pure and always runs; the wiring test builds a real Q4_K_M/Q6_K model and asserts the decision reached
/// the dispatch fields (<c>RecordMatmul</c> routes on exactly those), for both the default and the <c>=1</c> / <c>=0</c> overrides.
/// </summary>
[Collection("Q4KModel")]
[Trait("Category", "GPU")]
public sealed class VulkanKQuantDecodeKernelSelectionTests(Q4KModelFixture fixture)
{
    [Theory]
    [InlineData(0x1002u, null, false)]   // AMD default: F32-in GEMV
    [InlineData(0x10DEu, null, true)]    // NVIDIA default: unmeasured, keep MMVQ
    [InlineData(0x8086u, null, true)]    // Intel default: unmeasured, keep MMVQ
    [InlineData(0x1002u, "1", true)]     // forced on
    [InlineData(0x10DEu, "0", false)]    // forced off
    [InlineData(0x1002u, "", false)]     // empty = unset
    public void Policy_Table(uint vendor, string? env, bool expected) =>
        Assert.Equal(expected, VulkanTransformerModel.ResolveKQuantMmvq(vendor, env));

    [SkippableFact]
    public void Model_WiresKQuantMmvq_PerPolicy()
    {
        string spvDir = RequireVulkan(out uint vendor, out bool intDot);
        Skip.IfNot(intDot, "No integer-dot-product: MMVQ is never wired, so the default and override arms collapse.");

        bool defaultActive = BuildAndProbe(spvDir, env: null, out _);
        Assert.Equal(VulkanTransformerModel.ResolveKQuantMmvq(vendor, null), defaultActive);
        Assert.True(BuildAndProbe(spvDir, env: "1", out _), "=1 must wire the K-quant MMVQ kernels.");
        Assert.False(BuildAndProbe(spvDir, env: "0", out _), "=0 must leave only the F32-in GEMVs.");
    }

    [SkippableFact]
    public void F32InGemv_AndMmvq_AgreeOnLogits()
    {
        string spvDir = RequireVulkan(out _, out bool intDot);
        Skip.IfNot(intDot, "No integer-dot-product: nothing to compare against.");

        // Teacher-forced: both arms see the same token stream, so the logits are directly comparable.
        int[] tokens = [1, 450, 7483, 310, 3444, 338, 29892, 322, 1058];
        BuildAndProbe(spvDir, "0", out float[][] f32In, tokens);
        BuildAndProbe(spvDir, "1", out float[][] mmvq, tokens);

        for (int step = 0; step < f32In.Length; step++)
        {
            float scale = 0, maxDiff = 0;
            for (int i = 0; i < f32In[step].Length; i++)
            {
                scale = Math.Max(scale, Math.Abs(f32In[step][i]));
                maxDiff = Math.Max(maxDiff, Math.Abs(f32In[step][i] - mmvq[step][i]));
            }
            // MMVQ int8-quantises the activation, F32-in does not: they differ by quantisation noise, not by a wiring bug.
            Assert.True(maxDiff <= 0.05f * scale, $"step {step}: max |dlogit| {maxDiff:G4} vs logit scale {scale:G4}");
        }
    }

    private bool BuildAndProbe(string spvDir, string? env, out float[][] logits, int[]? tokens = null)
    {
        string? original = Environment.GetEnvironmentVariable(VulkanTransformerModel.KQuantMmvqEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(VulkanTransformerModel.KQuantMmvqEnvVar, env);
            using var gguf = GgufFile.Open(fixture.FilePath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
            bool active = model.KQuantMmvqDecodeActive;
            logits = [];
            if (tokens is not null)
            {
                using var cache = model.CreateKvCache(maxSeqLen: 64);
                var rows = new List<float[]>();
                for (int i = 0; i < tokens.Length; i++)
                    rows.Add(LastRow(model, [tokens[i]], [i], cache));
                logits = [.. rows];
            }
            return active;
        }
        finally
        {
            Environment.SetEnvironmentVariable(VulkanTransformerModel.KQuantMmvqEnvVar, original);
        }
    }

    private static unsafe float[] LastRow(VulkanTransformerModel model, int[] ids, int[] pos, VulkanKvCache cache)
    {
        using ITensor t = model.Forward(ids, pos, deviceId: -1, cache);
        int vocab = model.Config.VocabSize;
        var span = new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount);
        return span.Slice((t.Shape[0] - 1) * vocab, vocab).ToArray();
    }

    private static string RequireVulkan(out uint vendor, out bool intDot)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        using (var device = VulkanDevice.Create())
        {
            vendor = device.VendorId;
            intDot = device.HasIntegerDotProduct;
        }
        foreach (var c in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        })
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0) return full;
        }
        throw new SkipException("No compiled SPIR-V found.");
    }
}
