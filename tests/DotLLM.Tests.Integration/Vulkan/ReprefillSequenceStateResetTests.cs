using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Guards the growing-context re-prefill driver used by <see cref="RealGgufVulkanParityTests"/>
/// against silent recurrent-state carryover (#328).
/// </summary>
/// <remarks>
/// <para><b>The hazard.</b> The parity driver re-forwards the whole token prefix at every step, so
/// every step is an independent sequence. The uncached
/// <c>IModel.Forward(tokens, positions, deviceId)</c> overload carries no state container and falls
/// back to the model-owned recurrent state (<c>_gdnCache</c> / <c>_ssmCache</c>), which persists
/// across calls. On a recurrent architecture — Qwen3MoeHybrid's Gated DeltaNet layers, Nemotron-H's
/// SSM layers — step <c>n</c> would therefore see steps <c>0..n-1</c>'s accumulated recurrence
/// stacked on top of re-reading the same tokens, and the two backends could diverge for reasons
/// that have nothing to do with kernel parity. That is precisely why #328 was filed rather than
/// fixed: rewiring the driver to the architecture dispatchers without settling this first would
/// produce a green or red number that means nothing. The identical hazard corrupted perplexity
/// scoring in #261.</para>
///
/// <para><b>The semantics, established.</b> <see cref="IModel.ResetSequenceState"/> is the hook: it
/// re-zeroes model-owned recurrent state, is a documented no-op on stateless models, and its
/// <see cref="IModel"/> default <em>throws</em> for a model that declares
/// <see cref="IModel.RequiresPerSequenceState"/> without overriding it — so a future architecture
/// cannot inherit a silent no-op. Both sides of the Qwen3.6 parity case implement it
/// (<c>Qwen3MoeHybridTransformerModel</c> and <c>VulkanQwen3MoeHybridTransformerModel</c>, each
/// <c>_gdnCache.Reset()</c>), as do the Nemotron-H and Qwen3HybridDense models on both backends.</para>
///
/// <para><b>What these tests prove.</b> The driver asserts a replay invariant: re-running step 0's
/// exact inputs after the decode loop must reproduce step 0's logits bit-for-bit, because a
/// correctly reset model is a pure function of (weights, tokens, positions). These tests establish
/// that the invariant actually <em>discriminates</em> on a recurrent architecture — that it is false
/// without the reset and true with it — so a green parity run is evidence rather than decoration.
/// They use the tiny seeded synthetic Qwen3.5-MoE fixture (layer 0 is a real Gated DeltaNet layer
/// with a real <c>GdnStateCache</c>), so they need no multi-GB download and no known-good reference:
/// only determinism.</para>
/// </remarks>
public sealed class ReprefillSequenceStateResetTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _scratch;
    private readonly string _ggufPath;
    private readonly List<GgufFile> _openFiles = [];

    // The synthetic fixture's context length is 8 and its vocab is 8, so the re-prefill loop must
    // stay inside both. 4 prompt tokens + 4 decode steps hits the ceiling exactly.
    private static readonly int[] Prompt = [1, 2, 3, 4];
    private const int DecodeSteps = 4;

    public ReprefillSequenceStateResetTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-reprefill-state-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
        _ggufPath = SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "qwen35moe-tiny.gguf"));
    }

    public void Dispose()
    {
        foreach (var f in _openFiles) f.Dispose();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// THE load-bearing test: with the reset the replay invariant holds, and WITHOUT it the very
    /// same invariant fails on the very same model. One test asserts both halves deliberately —
    /// the positive half alone would pass on a model with no recurrence at all, and would then
    /// have proved nothing about the architecture the parity driver actually needed it for.
    /// </summary>
    [Fact]
    public void ReprefillLoop_OnARecurrentModel_IsSequenceIndependentOnlyWhenStateIsReset()
    {
        using IModel withReset = LoadCpuModel();
        Assert.True(withReset.RequiresPerSequenceState,
            "fixture must be a recurrent architecture or this test proves nothing about #328");

        (float[] first, float[] replay) = RunReprefillLoop(withReset, resetBetweenSteps: true);
        AssertBitIdentical(first, replay, expectEqual: true);

        using IModel withoutReset = LoadCpuModel();
        (float[] leakedFirst, float[] leakedReplay) = RunReprefillLoop(withoutReset, resetBetweenSteps: false);
        AssertBitIdentical(leakedFirst, leakedReplay, expectEqual: false);
    }

    /// <summary>
    /// The same invariant on the Vulkan backend, through the architecture dispatcher the parity
    /// driver now uses. Recurrent state lives in device memory there, so a CPU-side pass says
    /// nothing about it.
    /// </summary>
    /// <remarks>
    /// <para><b>Statically skipped, and the reason is a real defect, not a missing fixture</b> —
    /// see #360. Running this fixture through <c>VulkanModelLoader.CreateFromGguf</c> on gfx1151
    /// kills the device with <c>VK_ERROR_DEVICE_LOST</c> in ~3 s. The skip is deliberate and
    /// static rather than a runtime <c>Skip.If</c> on the exception, for two reasons: a device
    /// loss invalidates the <c>VkDevice</c> for the remainder of the process, so letting this run
    /// would take unrelated GPU suites down with it; and a runtime skip would make an unfixed
    /// defect look like an absent fixture, which is the failure mode this whole change exists to
    /// remove. Delete the Skip once #360 is fixed — the test body is correct and ready.</para>
    /// <para>Consequence to be honest about: device-side recurrent-state reset is currently
    /// covered only indirectly, via the replay assertion the parity driver runs on real models.</para>
    /// </remarks>
    [Fact(Skip = "Blocked on #360: VulkanModelLoader device-losses (VK_ERROR_DEVICE_LOST) on the "
                 + "tiny synthetic qwen35moe fixture on gfx1151, and a device loss poisons the "
                 + "VkDevice for every later GPU test in the process. Not a fixture gap - a defect. "
                 + "The CPU sibling of this test covers the same invariant and passes.")]
    [Trait("Category", "GPU")]
    public void ReprefillLoop_OnARecurrentVulkanModel_IsSequenceIndependentOnlyWhenStateIsReset()
    {
        string spvDir = SkipIfVulkanUnavailable();   // unreachable while the Skip above is set (#360)

        using var device = VulkanDevice.Create();
        using GgufFile gguf = OpenFixture(out ModelConfig config);
        (IModel model, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
        using (model)
        {
            Assert.True(model.RequiresPerSequenceState,
                "fixture must be a recurrent architecture or this test proves nothing about #328");
            (float[] first, float[] replay) = RunReprefillLoop(model, resetBetweenSteps: true);
            AssertBitIdentical(first, replay, expectEqual: true);
        }

        using GgufFile gguf2 = OpenFixture(out ModelConfig config2);
        (IModel leaky, _) = VulkanModelLoader.CreateFromGguf(device, gguf2, config2, spvDir);
        using (leaky)
        {
            (float[] leakedFirst, float[] leakedReplay) = RunReprefillLoop(leaky, resetBetweenSteps: false);
            AssertBitIdentical(leakedFirst, leakedReplay, expectEqual: false);
        }
    }

    // ────────────────────────────────────────────────────────────────────

    /// <summary>Mirrors <see cref="RealGgufVulkanParityTests"/>'s resolution (same SPV layout).</summary>
    private static string SkipIfVulkanUnavailable()
    {
        bool runtime;
        try
        {
            using var probe = VulkanDevice.Create();
            runtime = true;
        }
        catch
        {
            runtime = false;
        }
        Skip.IfNot(runtime, "Vulkan runtime not available on this host.");

        string? dir = null;
        string? probeDir = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probeDir is not null; i++)
        {
            string candidate = Path.Combine(probeDir, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) { dir = candidate; break; }
            probeDir = Path.GetDirectoryName(probeDir);
        }
        Skip.If(dir is null, "Vulkan SPV directory not found.");
        return dir!;
    }

    private GgufFile OpenFixture(out ModelConfig config)
    {
        var gguf = GgufFile.Open(_ggufPath);
        config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        return gguf;
    }

    private IModel LoadCpuModel()
    {
        GgufFile gguf = OpenFixture(out ModelConfig config);
        _openFiles.Add(gguf);   // the CPU model reads straight out of the mapping
        return ModelLoader.CreateCpuModelFromGguf(gguf, config);
    }

    /// <summary>
    /// Runs the parity driver's growing-context re-prefill loop and then replays step 0's exact
    /// inputs. Returns (step-0 logits, replayed step-0 logits).
    /// </summary>
    private (float[] First, float[] Replay) RunReprefillLoop(IModel model, bool resetBetweenSteps)
    {
        var tokens = new List<int>(Prompt);
        float[]? first = null;
        int[] firstTokens = [];
        int[] firstPositions = [];

        for (int step = 0; step <= DecodeSteps; step++)
        {
            int[] tokenIds = tokens.ToArray();
            int[] positions = new int[tokenIds.Length];
            for (int i = 0; i < positions.Length; i++) positions[i] = i;

            if (resetBetweenSteps) model.ResetSequenceState();
            float[] logits = LastRow(model, tokenIds, positions);

            if (step == 0)
            {
                first = (float[])logits.Clone();
                firstTokens = tokenIds;
                firstPositions = positions;
            }

            tokens.Add(Argmax(logits));
        }

        if (resetBetweenSteps) model.ResetSequenceState();
        float[] replay = LastRow(model, firstTokens, firstPositions);
        _output.WriteLine($"reset={resetBetweenSteps}: step0[0]={first![0]:R} replay[0]={replay[0]:R}");
        return (first, replay);
    }

    private static unsafe float[] LastRow(IModel model, int[] tokenIds, int[] positions)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(2, logits.Shape.Rank);
        int seqLen = logits.Shape[0];
        int vocab = logits.Shape[1];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static void AssertBitIdentical(float[] first, float[] replay, bool expectEqual)
    {
        Assert.Equal(first.Length, replay.Length);
        bool identical = true;
        for (int i = 0; i < first.Length && identical; i++)
            identical = BitConverter.SingleToInt32Bits(first[i]) == BitConverter.SingleToInt32Bits(replay[i]);

        if (expectEqual)
        {
            Assert.True(identical,
                "with ResetSequenceState() between steps the re-prefill loop must be sequence-independent: "
                + "replaying step 0's inputs must reproduce step 0's logits bit-for-bit. It did not, so the "
                + "reset does not restore this architecture's model-owned recurrent state.");
        }
        else
        {
            Assert.False(identical,
                "WITHOUT ResetSequenceState() the replay was bit-identical anyway - so this fixture does not "
                + "actually carry recurrent state across Forward calls, and the reset assertion in "
                + "RealGgufVulkanParityTests would pass whether or not the reset worked. Fix the fixture, not "
                + "this assertion: a guard that cannot fail is the bug #328 exists to prevent.");
        }
    }
}
