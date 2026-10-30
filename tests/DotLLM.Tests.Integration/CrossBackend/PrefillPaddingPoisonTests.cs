using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.CrossBackend;

/// <summary>
/// Issue #1290 (from azllm's <c>gemm_pad.hpp</c>): a prefill kernel that tiles the token dimension must never let scratch rows
/// past the real token count reach a real output. azllm hit it as a first token that changed between server starts. The audit
/// in #1290 found every CUDA / Vulkan prefill path guarded, clamped or zero-staged; this test keeps it that way.
/// </summary>
/// <remarks>
/// <para>
/// <b>Method.</b> The control loads the model and runs a <see cref="PromptTokens"/>-token prefill plus greedy decode. The probe
/// loads it again with every new device buffer filled with NaN bytes (forward scratch, KV cache, lazily allocated
/// buffers), first runs a longer <see cref="WarmTokens"/>-token prefill on a throwaway KV cache so the scratch grows past the
/// prompt, then the same prompt and decode on a fresh NaN-filled cache. Rows <c>[PromptTokens, WarmTokens)</c> of every scratch
/// buffer then hold another prompt's finite activations, rows past <see cref="WarmTokens"/> (CUDA's power-of-two capacity) and
/// every KV row past the position are NaN.
/// </para>
/// <para>
/// <b>Discriminates.</b> Any read of an unwritten row into a real output (a tile load without a token guard feeding a shared
/// activation scale, a NaN × 0 in an MMA P·V, an attention that reads KV past the sequence, an accumulate into an uncleared
/// output) turns the logits NaN or moves them; the comparison is bit-exact.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class PrefillPaddingPoisonTests(ITestOutputHelper output)
{
    private const int PromptTokens = 13;   // not a multiple of 8 / 16 / 32: every token tile has a ragged tail
    private const int WarmTokens = 29;
    private const int DecodeSteps = 4;

    public static TheoryData<string, string, string, string> Models => new()
    {
        { "DOTLLM_SMOLLM2_135M_Q8_0_GGUF", "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q8_0.gguf" },
        { "DOTLLM_SMOLLM_135M_Q4_K_M_GGUF", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q4_K_M.gguf" },
        { "DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf" },

        // PQ2_0: the token-tiled pq2_0 MMQ prefill kernels (dp4a / mma) with their dead-column staging.
        { "DOTLLM_BONSAI2_2B_GGUF", "prism-ml", "Ternary-Bonsai-2-2B-gguf", "Ternary-Bonsai-2-2B-PQ2_0.gguf" },
    };

    [SkippableTheory]
    [MemberData(nameof(Models))]
    public void Cuda_NaNPoisonedScratch_DoesNotChangeLogits(string env, string org, string repo, string file)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string path = Resolve(env, org, repo, file);
        List<float[]> clean = Run(path, poisoned: false, (gguf, cfg) => CudaModelLoader.CreateFromGguf(gguf, cfg, deviceId: 0),
            on => CudaDriverApi.TestPoisonAllocations = on, dispose: null);
        List<float[]> poisoned = Run(path, poisoned: true, (gguf, cfg) => CudaModelLoader.CreateFromGguf(gguf, cfg, deviceId: 0),
            on => CudaDriverApi.TestPoisonAllocations = on, dispose: null);
        AssertBitIdentical(clean, poisoned, $"CUDA {file}");
    }

    [SkippableTheory]
    [MemberData(nameof(Models))]
    public void Vulkan_NaNPoisonedScratch_DoesNotChangeLogits(string env, string org, string repo, string file)
    {
        Skip.IfNot(IsVulkanAvailable(), "No Vulkan device available.");
        string? spvDir = ResolveSpvDir();
        Skip.If(spvDir is null, "Vulkan SPV directory not found.");
        string path = Resolve(env, org, repo, file);
        VulkanDevice? device = null;
        (IModel, Func<int, IKvCache>) Load(GgufFile gguf, ModelConfig cfg)
        {
            device = VulkanDevice.Create();
            return VulkanModelLoader.CreateFromGguf(device, gguf, cfg, spvDir!);
        }
        void DisposeDevice() => device?.Dispose();

        List<float[]> clean = Run(path, poisoned: false, Load, on => VulkanDevice.TestPoisonNewBuffers = on, DisposeDevice);
        List<float[]> poisoned = Run(path, poisoned: true, Load, on => VulkanDevice.TestPoisonNewBuffers = on, DisposeDevice);
        AssertBitIdentical(clean, poisoned, $"Vulkan {file}");
    }

    /// <summary>Loads the model, optionally poisons new buffers, runs (warm-up then) prompt + greedy decode; last-row logits.</summary>
    private List<float[]> Run(string path, bool poisoned, Func<GgufFile, ModelConfig, (IModel, Func<int, IKvCache>)> load,
        Action<bool> setPoison, Action? dispose)
    {
        using var gguf = GgufFile.Open(path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int vocab = cfg.VocabSize;
        int[] warm = Enumerable.Range(0, WarmTokens).Select(i => (17 + i * 7919) % vocab).ToArray();
        int[] prompt = Enumerable.Range(0, PromptTokens).Select(i => (101 + i * 4099) % vocab).ToArray();
        var rows = new List<float[]>();
        var (model, kvFactory) = load(gguf, cfg);
        try
        {
            setPoison(poisoned);
            if (poisoned)
            {
                using IKvCache warmKv = kvFactory(64);
                using (model.Forward(warm, Enumerable.Range(0, WarmTokens).ToArray(), deviceId: -1, warmKv)) { }
                model.ResetSequenceState();   // recurrent (GDN) layers: the warm-up's state must not carry over
            }
            using IKvCache kv = kvFactory(64);
            int[] pos = Enumerable.Range(0, PromptTokens).ToArray();
            float[] last;
            using (ITensor p = model.Forward(prompt, pos, deviceId: -1, kv))
                rows.Add(last = LastRow(p, vocab));
            for (int s = 0; s < DecodeSteps; s++)
            {
                int next = Argmax(last);
                using ITensor d = model.Forward([next], [PromptTokens + s], deviceId: -1, kv);
                rows.Add(last = LastRow(d, vocab));
            }
        }
        finally
        {
            setPoison(false);
            model.Dispose();
            dispose?.Invoke();
        }
        output.WriteLine($"{(poisoned ? "poisoned" : "clean   ")} argmax: {string.Join(' ', rows.Select(Argmax))}");
        return rows;
    }

    private static void AssertBitIdentical(List<float[]> clean, List<float[]> poisoned, string label)
    {
        Assert.Equal(clean.Count, poisoned.Count);
        for (int r = 0; r < clean.Count; r++)
        {
            Assert.All(clean[r], v => Assert.True(float.IsFinite(v), $"{label}: the clean run is not finite"));
            for (int i = 0; i < clean[r].Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(clean[r][i]) == BitConverter.SingleToInt32Bits(poisoned[r][i]),
                    $"{label} {(r == 0 ? "prefill" : $"decode {r}")} logit {i}: clean {clean[r][i]:R} vs NaN-poisoned scratch {poisoned[r][i]:R}");
        }
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int rows = logits.Shape.Rank == 2 ? logits.Shape[0] : 1;
        var all = new ReadOnlySpan<float>((void*)logits.DataPointer, rows * vocab);
        return all.Slice((rows - 1) * vocab, vocab).ToArray();
    }

    private static int Argmax(float[] xs)
    {
        int best = 0;
        for (int i = 1; i < xs.Length; i++)
            if (xs[i] > xs[best]) best = i;
        return best;
    }

    private static string Resolve(string env, string org, string repo, string file)
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(env, org, repo, file);
        Skip.If(!fixture.Found, fixture.SkipMessage(file));
        return fixture.Path!;
    }

    private static bool IsVulkanAvailable()
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

    private static string? ResolveSpvDir()
    {
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return null;
    }
}
