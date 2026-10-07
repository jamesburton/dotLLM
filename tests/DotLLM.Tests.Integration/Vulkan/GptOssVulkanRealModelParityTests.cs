using System.Diagnostics;
using System.Text.Json;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #737 acceptance: real gpt-oss-20b (MXFP4) on Vulkan vs the CPU path. Greedy-token identity over several prompts
/// (one longer than the 128-token sliding window, so the windowed layers genuinely differ from dense ones), prefill-logit
/// parity, per-feature perturbation proof (each gpt-oss feature - sinks, YaRN, router bias, expert bias, OAI SwiGLU - must
/// move the logits when switched off, and its kernel dispatch counter must be non-zero), and Vulkan throughput.
/// The CPU reference is computed once and cached in <c>DOTLLM_GPTOSS_REF</c> (default: temp dir). Gated on
/// <c>DOTLLM_GPTOSS_GGUF</c> / the canonical HF-hub path; acquire the GPU lock before running.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class GptOssVulkanRealModelParityTests
{
    private const string DefaultHubGlob = "models--ggml-org--gpt-oss-20b-GGUF";
    private const int MaxNew = 24;

    private static readonly string LongProse = string.Concat(Enumerable.Repeat(
        "The history of the Roman Empire spans many centuries, during which the city grew from a small settlement on the "
        + "banks of the Tiber into the capital of a vast realm. Its legions marched across Europe, and its roads linked "
        + "provinces from Britain to Syria. ", 3));

    private static readonly string[] Prompts =
    [
        "The capital of France is",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
        LongProse + "In summary, the Roman Empire was",
    ];

    private readonly ITestOutputHelper _output;
    public GptOssVulkanRealModelParityTests(ITestOutputHelper output) => _output = output;

    private sealed record Ref(int[] PromptIds, int[] Generated, float[] PrefillLogits);

    private static string? ModelPath()
    {
        string? p = Environment.GetEnvironmentVariable("DOTLLM_GPTOSS_GGUF");
        if (!string.IsNullOrWhiteSpace(p) && File.Exists(p)) return p;
        string hub = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub", DefaultHubGlob, "snapshots");
        if (!Directory.Exists(hub)) return null;
        return Directory.EnumerateFiles(hub, "gpt-oss-20b-mxfp4.gguf", SearchOption.AllDirectories).FirstOrDefault();
    }

    private static int[] Encode(DotLLM.Tokenizers.ITokenizer tok, string text)
    {
        int[] enc = tok.Encode(text);
        if (tok.BosTokenId < 0) return enc;
        var ids = new int[enc.Length + 1];
        ids[0] = tok.BosTokenId;
        Array.Copy(enc, 0, ids, 1, enc.Length);
        return ids;
    }

    [SkippableFact]
    public void Vulkan_GptOss20b_GreedyTextAndLogits_MatchCpu_AndEveryFeatureIsLive()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = ModelPath();
        Skip.If(path is null, "Set DOTLLM_GPTOSS_GGUF to gpt-oss-20b-mxfp4.gguf.");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        string refFile = Environment.GetEnvironmentVariable("DOTLLM_GPTOSS_REF")
            ?? Path.Combine(Path.GetTempPath(), "dotllm-gptoss20b-cpu-ref.json");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var promptIds = Prompts.Select(p => Encode(tokenizer, p)).ToArray();
        Assert.True(promptIds[2].Length > 160, "long prompt must exceed the 128-token sliding window by a clear margin");

        Ref[] cpuRefs;
        if (File.Exists(refFile)
            && JsonSerializer.Deserialize<Ref[]>(File.ReadAllText(refFile)) is { } cached
            && cached.Length == Prompts.Length
            && cached.Select((r, i) => r.PromptIds.SequenceEqual(promptIds[i])).All(x => x))
        {
            cpuRefs = cached;
        }
        else
        {
            using var cpu = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, config);
            cpuRefs = promptIds.Select(ids => Generate(cpu, config,
                size => new SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config), size), ids)).ToArray();
            File.WriteAllText(refFile, JsonSerializer.Serialize(cpuRefs));
        }

        // ── Vulkan baseline ──
        float[] baselineLong;
        int totalSteps = 0, matchedSteps = 0;
        using (var vk = VulkanTransformerModel.LoadFromGguf(gguf, config, spv))
        {
            for (int p = 0; p < Prompts.Length; p++)
            {
                var got = Generate(vk, config, size => vk.CreateKvCache(size), promptIds[p]);
                double worst = 0;
                for (int i = 0; i < got.PrefillLogits.Length; i++)
                    worst = Math.Max(worst, Math.Abs(got.PrefillLogits[i] - cpuRefs[p].PrefillLogits[i]));
                int firstDiv = -1;
                for (int i = 0; i < MaxNew; i++)
                    if (got.Generated[i] != cpuRefs[p].Generated[i]) { firstDiv = i; break; }
                totalSteps += MaxNew;
                matchedSteps += firstDiv < 0 ? MaxNew : firstDiv;
                _output.WriteLine($"prompt {p} ({promptIds[p].Length} tok): prefill argmax cpu={ArgMax(cpuRefs[p].PrefillLogits)} "
                    + $"vk={ArgMax(got.PrefillLogits)} worst|dLogit|={worst:F4}; firstDivergence={firstDiv}");
                _output.WriteLine($"   cpu: '{tokenizer.Decode(cpuRefs[p].Generated)}'");
                _output.WriteLine($"   vk : '{tokenizer.Decode(got.Generated)}'");
                Assert.Equal(ArgMax(cpuRefs[p].PrefillLogits), ArgMax(got.PrefillLogits));
            }

            // Every gpt-oss kernel must have actually run (dispatch counters), not merely be selectable.
            var k = vk.GptOssKernels!;
            _output.WriteLine($"dispatches: sinks={k.SinkAttentionDispatches} swigluOai={k.SwiGluOaiDispatches} "
                + $"expertBias={k.ExpertBiasDispatches} rawTopK={k.RawTopKDispatches} mxfp4={k.Mxfp4MatmulDispatches} yarn={k.RopeYarnDispatches}");
            Assert.True(k.SinkAttentionDispatches > 0 && k.SwiGluOaiDispatches > 0 && k.ExpertBiasDispatches > 0
                && k.RawTopKDispatches > 0 && k.Mxfp4MatmulDispatches > 0 && k.RopeYarnDispatches > 0,
                "a gpt-oss kernel never dispatched");

            baselineLong = PrefillRow(vk, config.VocabSize, promptIds[2]);

            // Throughput (warm, same session).
            int[] bench = Enumerable.Range(0, 128).Select(i => 1000 + (i * 37) % 5000).ToArray();
            int[] benchPos = Enumerable.Range(0, 128).ToArray();
            using (var kv = vk.CreateKvCache(256))
                using (vk.Forward(bench.AsSpan(0, 8), benchPos.AsSpan(0, 8), -1, kv)) { }
            using (var kv = vk.CreateKvCache(256))
            {
                var sw = Stopwatch.StartNew();
                using (vk.Forward(bench, benchPos, -1, kv)) { }
                double pre = sw.Elapsed.TotalSeconds;
                sw.Restart();
                for (int i = 0; i < 32; i++)
                    using (vk.Forward([1234 + i], [128 + i], -1, kv)) { }
                double dec = sw.Elapsed.TotalSeconds;
                _output.WriteLine($"Vulkan throughput: prefill {128 / pre:F0} tok/s (128 tok), decode {32 / dec:F1} tok/s ({dec / 32 * 1000:F1} ms/tok)");
            }
        }

        Assert.True(matchedSteps >= totalSteps * 2 / 3,
            $"Greedy text diverged from CPU too early: matched {matchedSteps}/{totalSteps}.");

        // ── Perturbation: each feature switched off must move the long-prompt logits. ──
        foreach (string toggle in new[] { "sinks", "yarn_mscale", "yarn", "router_bias", "expert_bias", "swiglu_oai" })
        {
            Environment.SetEnvironmentVariable("DOTLLM_GPTOSS_DEBUG_OFF", toggle);
            try
            {
                using var vkOff = VulkanTransformerModel.LoadFromGguf(gguf, config, spv);
                float[] off = PrefillRow(vkOff, config.VocabSize, promptIds[2]);
                double worst = 0;
                for (int i = 0; i < off.Length; i++) worst = Math.Max(worst, Math.Abs(off[i] - baselineLong[i]));
                _output.WriteLine($"perturb {toggle}: worst|dLogit| vs baseline = {worst:F4}");
                Assert.True(worst > 0.05, $"feature '{toggle}' is inert: switching it off moved logits by only {worst}");
            }
            finally { Environment.SetEnvironmentVariable("DOTLLM_GPTOSS_DEBUG_OFF", null); }
        }
    }

    /// <summary>
    /// Issue #789: the expert-grouped coopmat MoE prefill (MXFP4 gate/up/down, per-expert bias in the shader epilogue, OAI SwiGLU in packed order) vs the scalar
    /// indexed path, IN-PROCESS on one loaded model (<see cref="VulkanTransformerModel.GroupedMoeEnabled"/>), plus both vs the CPU reference. The grouped path must be
    /// provably live (dispatch counter == MoE layers per prefill; zero when switched off) and must reproduce the scalar prefill logits to within the F16-operand noise.
    /// </summary>
    [SkippableFact]
    public void Vulkan_GptOss20b_GroupedMoePrefill_MatchesScalarPath_AndCpu()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = ModelPath();
        Skip.If(path is null, "Set DOTLLM_GPTOSS_GGUF to gpt-oss-20b-mxfp4.gguf.");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        string refFile = Environment.GetEnvironmentVariable("DOTLLM_GPTOSS_REF")
            ?? Path.Combine(Path.GetTempPath(), "dotllm-gptoss20b-cpu-ref.json");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var promptIds = Prompts.Select(pr => Encode(tokenizer, pr)).ToArray();
        Ref[]? cpuRefs = File.Exists(refFile) && JsonSerializer.Deserialize<Ref[]>(File.ReadAllText(refFile)) is { } cached
            && cached.Length == Prompts.Length && cached.Select((r, i) => r.PromptIds.SequenceEqual(promptIds[i])).All(x => x) ? cached : null;

        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, config, spv);
        Skip.IfNot(vk.GroupedMoeKernelsBuilt, "device lacks wave64 coopmat (or DOTLLM_VK_MOE_GROUPED=0): grouped path not built.");
        for (int p = 0; p < Prompts.Length; p++)
        {
            vk.GroupedMoeEnabled = true;
            int before = vk.GroupedMoeDispatchCount, beforeMx = vk.GroupedMoeMxfp4DispatchCount;
            float[] grouped = PrefillRow(vk, config.VocabSize, promptIds[p]);
            int groupedLayers = vk.GroupedMoeDispatchCount - before;
            // every gpt-oss layer is MoE; prompts shorter than the 16-token gate (prompt 0) keep the scalar path by design.
            int expectLayers = promptIds[p].Length >= 16 ? config.NumLayers : 0;
            Assert.Equal(expectLayers, groupedLayers);
            Assert.Equal(groupedLayers, vk.GroupedMoeMxfp4DispatchCount - beforeMx);

            vk.GroupedMoeEnabled = false;
            before = vk.GroupedMoeDispatchCount;
            float[] scalar = PrefillRow(vk, config.VocabSize, promptIds[p]);
            Assert.Equal(before, vk.GroupedMoeDispatchCount);                    // off really is off
            vk.GroupedMoeEnabled = true;

            double rmsGS = Rms(grouped, scalar);
            string cpuNote = "";
            if (cpuRefs is not null)
                cpuNote = $" | rms(grouped-cpu)={Rms(grouped, cpuRefs[p].PrefillLogits):F4} rms(scalar-cpu)={Rms(scalar, cpuRefs[p].PrefillLogits):F4}";
            _output.WriteLine($"prompt {p} ({promptIds[p].Length} tok): grouped layers={groupedLayers}; rms(grouped-scalar)={rmsGS:F4} (logit rms {Rms(scalar, new float[scalar.Length]):F3}); "
                + $"argmax grouped={ArgMax(grouped)} scalar={ArgMax(scalar)}{cpuNote}");
            Assert.Equal(ArgMax(scalar), ArgMax(grouped));
            if (expectLayers == 0) Assert.Equal(0.0, rmsGS);                    // scalar path both times: bit-identical
            Assert.True(rmsGS < 0.05 * Rms(scalar, new float[scalar.Length]) + 0.05, $"grouped vs scalar logit rms {rmsGS} exceeds the F16-operand envelope");
            if (cpuRefs is not null)
                Assert.True(Rms(grouped, cpuRefs[p].PrefillLogits) < Rms(scalar, cpuRefs[p].PrefillLogits) * 1.5 + 0.05, "grouped path is further from CPU than the scalar path by more than noise");
        }

        // Greedy grouped-vs-scalar over a full generation (prefill via the grouped path, decode unchanged). F16 operands perturb the logits by ~1% of
        // their rms, so exact text identity is only guaranteed where the top-1 margin exceeds that noise: a divergence is accepted ONLY at a step where the
        // scalar path's own top-1/top-2 margin is below 0.5 logits (a near-tie), and the step is reported.
        vk.GroupedMoeEnabled = true;
        var (gTok, _) = GreedyWithMargins(vk, config, promptIds[2]);
        vk.GroupedMoeEnabled = false;
        var (sTok, sMargin) = GreedyWithMargins(vk, config, promptIds[2]);
        vk.GroupedMoeEnabled = true;
        _output.WriteLine($"greedy grouped: '{tokenizer.Decode(gTok)}' | greedy scalar : '{tokenizer.Decode(sTok)}'");
        int firstDiv = -1;
        for (int i = 0; i < MaxNew; i++) if (gTok[i] != sTok[i]) { firstDiv = i; break; }
        _output.WriteLine($"greedy first divergence step: {firstDiv}" + (firstDiv >= 0 ? $" (scalar top1-top2 margin there = {sMargin[firstDiv]:F3} logits)" : ""));
        Assert.True(firstDiv < 0 || sMargin[firstDiv] < 0.5f, $"greedy text diverged at step {firstDiv} where the scalar margin is {(firstDiv >= 0 ? sMargin[firstDiv] : 0f)} (not a near-tie)");
    }

    private static unsafe (int[] Tokens, float[] Margins) GreedyWithMargins(VulkanTransformerModel vk, DotLLM.Core.Models.ModelConfig config, int[] promptIds)
    {
        int vocab = config.VocabSize;
        var toks = new List<int>(); var margins = new List<float>();
        using var kv = vk.CreateKvCache(promptIds.Length + MaxNew + 2);
        for (int step = 0; step < MaxNew; step++)
        {
            int[] ids = step == 0 ? promptIds : [toks[^1]];
            int[] pos = step == 0 ? Enumerable.Range(0, promptIds.Length).ToArray() : [promptIds.Length + step - 1];
            using ITensor logits = vk.Forward(ids, pos, -1, kv);
            var row = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab);
            int best = 0;
            for (int i = 1; i < vocab; i++) if (row[i] > row[best]) best = i;
            float second = float.NegativeInfinity;
            for (int i = 0; i < vocab; i++) if (i != best && row[i] > second) second = row[i];
            toks.Add(best); margins.Add(row[best] - second);
        }
        return (toks.ToArray(), margins.ToArray());
    }

    private static double Rms(float[] a, float[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; s += d * d; }
        return Math.Sqrt(s / a.Length);
    }

    /// <summary>
    /// Depth sweep: truncate the model to N layers on BOTH backends and compare last-position logits. A real bug in one
    /// op shows up as a jump at the first layer that exercises it (layer 0 = sliding window, layer 1 = dense; the long
    /// prompt crosses the 128-token window); backend noise instead grows smoothly with depth.
    /// </summary>
    [SkippableFact]
    public void Vulkan_GptOss20b_DepthSweep_RelativeErrorGrowsSmoothly()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_GPTOSS_DEPTH_SWEEP") != "1", "Set DOTLLM_GPTOSS_DEPTH_SWEEP=1 (diagnostic).");
        string? path = ModelPath();
        Skip.If(path is null, "Set DOTLLM_GPTOSS_GGUF to gpt-oss-20b-mxfp4.gguf.");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        using var gguf = GgufFile.Open(path!);
        var fullConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        foreach (int promptIdx in new[] { 0, 2 })
        {
            int[] ids = Encode(tokenizer, Prompts[promptIdx]);
            foreach (int depth in new[] { 1, 4, 24 })
            {
                var config = fullConfig with { NumLayers = depth };
                float[] c, v, vF32;
                using (var cpu = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, config))
                    c = CpuPrefillRow(cpu, config, ids);
                using (var vk = VulkanTransformerModel.LoadFromGguf(gguf, config, spv))
                    v = PrefillRow(vk, config.VocabSize, ids);
                // Control arm: same Vulkan model with the integer-dot (Q8_1 activation-quantised) matmul paths off, i.e.
                // Q8_0 weights x F32 activations. |v - vF32| is the Vulkan activation-quantisation noise floor.
                Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_MMVQ", "1");
                Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_MMQ", "1");
                try
                {
                    using var vk32 = VulkanTransformerModel.LoadFromGguf(gguf, config, spv);
                    vF32 = PrefillRow(vk32, config.VocabSize, ids);
                }
                finally
                {
                    Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_MMVQ", null);
                    Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_MMQ", null);
                }
                _output.WriteLine($"prompt {promptIdx} depth {depth,2}: cpu~vk {Rel(c, v)} | cpu~vkF32act {Rel(c, vF32)} | vk~vkF32act {Rel(v, vF32)}");
            }
        }
    }

    private static string Rel(float[] a, float[] b)
    {
        double num = 0, den = 0, dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            num += d * d; den += (double)a[i] * a[i];
            dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
        }
        return $"relL2={Math.Sqrt(num / den):E2} cos={dot / Math.Sqrt(na * nb):F6}";
    }

    private static unsafe float[] CpuPrefillRow(DotLLM.Core.Models.IModel m, DotLLM.Core.Models.ModelConfig config, int[] ids)
    {
        using var kv = new SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config), ids.Length + 2);
        using ITensor logits = m.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, kv);
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * config.VocabSize, config.VocabSize).ToArray();
    }

    private static unsafe float[] PrefillRow(VulkanTransformerModel vk, int vocab, int[] ids)
    {
        using var kv = vk.CreateKvCache(ids.Length + 2);
        using ITensor logits = vk.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, kv);
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab).ToArray();
    }

    private static unsafe Ref Generate(DotLLM.Core.Models.IModel model, DotLLM.Core.Models.ModelConfig config,
        Func<int, DotLLM.Core.Attention.IKvCache> makeCache, int[] promptIds)
    {
        int vocab = config.VocabSize;
        var generated = new List<int>();
        float[] prefill;
        using var kv = makeCache(promptIds.Length + MaxNew + 2);
        int[] pos = Enumerable.Range(0, promptIds.Length).ToArray();
        using (ITensor logits = model.Forward(promptIds, pos, -1, kv))
            prefill = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab).ToArray();
        generated.Add(ArgMax(prefill));
        for (int step = 1; step < MaxNew; step++)
        {
            using ITensor logits = model.Forward([generated[^1]], [promptIds.Length + step - 1], -1, kv);
            var row = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab);
            int best = 0;
            for (int i = 1; i < vocab; i++) if (row[i] > row[best]) best = i;
            generated.Add(best);
        }
        return new Ref(promptIds, generated.ToArray(), prefill);
    }

    private static int ArgMax(float[] v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }
}
