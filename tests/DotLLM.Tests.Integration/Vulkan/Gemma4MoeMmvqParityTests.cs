using DotLLM.Core.Configuration;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Decode-path parity for the indexed MoE MMVQ (dp4a) expert kernels (issue #137):
/// <c>moe_indexed_matmul_q4_k_mmvq</c> (gate/up), <c>moe_indexed_matmul_q5_1_mmvq</c>
/// and <c>moe_indexed_matmul_q8_0_mmvq</c> (down), fed by the row-wise Q8_1
/// activation quantizer. Each supported down-bank quant type (Q5_1 verbatim,
/// Q5_0 → bit-exact Q5_1 repack, Q8_0 with the per-expert scale pre-folded) runs
/// a KV-cached S==1 decode on the Vulkan backend and must match the CPU
/// cacheless full-sequence oracle: argmax-exact plus a per-logit envelope (the
/// mmvq path int8-quantizes the activations, so it is NOT bit-exact vs F32).
/// </summary>
/// <remarks>
/// <para>DISCRIMINATING FIXTURE (repo rule): shapes are chosen so no two routing
/// quantities coincide — 6 experts ≠ top-3 ≠ 9 down-blocks-per-row ≠ 2 gate/up
/// super-blocks — so an expert-index/stride mix-up or a q/qh nibble-plane swap
/// cannot cancel out. ExpertFeedForward = 288 (9 Q5_1/Q8_0 blocks) exercises the
/// shader's second 8-block window INCLUDING the ragged tail lane-idle path;
/// Hidden = 512 exercises multi-super-block Q4_K rows. The test also asserts via
/// <see cref="VulkanTransformerModel.MoeMmvqDispatchCount"/> that the fast path
/// actually executed (a silently-skipped path would trivially "pass").</para>
/// <para>The multi-token prefill stays on the scalar indexed kernels by design
/// (byte-identical to pre-#137), so this test isolates the new kernels to the
/// single decode step it checks.</para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class Gemma4MoeMmvqParityTests
{
    private readonly ITestOutputHelper _output;

    public Gemma4MoeMmvqParityTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(QuantizationType.Q5_1)]
    [InlineData(QuantizationType.Q5_0)]
    [InlineData(QuantizationType.Q8_0)]
    public unsafe void Vulkan_Gemma4_MoeMmvqDecode_MatchesCpuOracle(QuantizationType downQuant)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");
        string spvDir = ResolveSpvDir();

        // Non-degenerate routing shapes (see remarks): experts=6, topK=3,
        // Ie=288 (9 32-blocks), hidden=512 (2 Q4_K super-blocks).
        var cfg = SyntheticGemma4Gguf.Tiny with
        {
            HiddenSize = 512,
            ExpertFeedForward = 288,
            ExpertCount = 6,
            ExpertUsedCount = 3,
            ExpertDownQuant = downQuant,
        };
        Assert.NotEqual(cfg.ExpertCount, cfg.ExpertUsedCount);          // routing-degeneracy guard
        Assert.NotEqual(cfg.ExpertFeedForward / 32, cfg.ExpertCount);   // blocks/row != experts
        Assert.True(cfg.ExpertFeedForward / 32 > 8,
            "Ie must exceed one 8-block mmvq window so the window loop + ragged tail execute.");

        string path = Path.Combine(Path.GetTempPath(), $"syn_gemma4_moemmvq_{downQuant}_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, cfg);

        try
        {
            // Stay within the sliding window (8) so cacheless vs prefill+decode
            // see identical attention windows (isolates the MoE kernels).
            int[] ids = { 2, 7, 8, 9, 5, 6, 3 };
            int[] pos = { 0, 1, 2, 3, 4, 5, 6 };
            int last = ids.Length - 1;

            // ── Oracle: CPU cacheless forward over the whole sequence ──
            float[] cpuLast;
            int vocab;
            {
                var (cpuModel, cpuGguf, cpuCfg) = ModelLoader.LoadFromGguf(path);
                using var _g = cpuGguf;
                using var _m = cpuModel;
                vocab = cpuCfg.VocabSize;
                using ITensor logits = cpuModel.Forward(ids, pos, deviceId: -1,
                    kvCache: null, adapter: null, AttentionMaskSpec.Causal);
                cpuLast = LastRow(logits, vocab);
            }

            // ── Baseline: Vulkan SCALAR decode (mmvq opted out at load time).
            // Anchors the CPU↔Vulkan drift of the pre-#137 path on this exact
            // fixture, so the mmvq assertion below measures ONLY the int8
            // activation-quant delta added by the new kernels. ──
            float[] vkScalarLast = RunVulkanDecode(path, spvDir, ids, pos, last, vocab,
                disableMoeMmvq: true, out long scalarDispatches);
            Assert.Equal(0L, scalarDispatches);

            // ── Under test: Vulkan mmvq decode ──
            float[] vkLast = RunVulkanDecode(path, spvDir, ids, pos, last, vocab,
                disableMoeMmvq: false, out long mmvqDispatches);

            // The fast path must actually have executed on the decode step —
            // one dispatch per MoE layer of the 6-layer fixture.
            Assert.True(mmvqDispatches > 0,
                "MoeMmvqDispatchCount == 0 — the indexed MMVQ decode path was silently skipped, "
                + "so this test validated nothing. Check kernel creation / scratch gating.");
            _output.WriteLine($"[{downQuant}] mmvq dispatches on decode step: {mmvqDispatches}");

            // 1) Structural gate: greedy argmax must agree across all three
            //    paths. (Absolute CPU↔GPU envelopes on a random tiny fixture
            //    are router-tie sensitive: on the Q5_0 seed the PRE-EXISTING
            //    scalar path already drifts 2.3e-1 vs CPU from reduction-order
            //    divergence cascading through a marginal top-K pick — so the
            //    CPU comparison is argmax-level, and the tight envelope below
            //    isolates exactly what #137 adds.)
            int cpuArg = ArgMax(cpuLast);
            Assert.Equal(cpuArg, ArgMax(vkScalarLast));
            Assert.Equal(cpuArg, ArgMax(vkLast));

            // 2) Kernel-isolation envelope: mmvq vs the SAME-GPU scalar decode.
            //    The two paths share the router / top-K / broadcast / GeGLU /
            //    scatter dispatches bit-for-bit — they differ ONLY in the three
            //    expert matmuls (dp4a over int8-quantized activations vs F32-in
            //    scalar), so this bound is what the new kernels may add. A real
            //    addressing / nibble-plane / scale bug produces O(1) errors and
            //    argmax flips; the int8 activation-quant drift stays ~1e-1.
            AssertEnvelope(vkScalarLast, vkLast, vocab, absTol: 1.5e-1f, relTol: 1.0e-2f,
                label: $"[{downQuant}] mmvq-vs-scalar", out float mmvqWorst);
            float cpuWorst = 0;
            for (int c = 0; c < vocab; c++)
                cpuWorst = MathF.Max(cpuWorst, MathF.Abs(cpuLast[c] - vkLast[c]));

            _output.WriteLine(
                $"[{downQuant}] MoE mmvq decode parity OK: argmax={cpuArg}, "
                + $"mmvq-vs-scalar worst |diff|={mmvqWorst:E3}, mmvq-vs-cpu worst |diff|={cpuWorst:E3} over {vocab} logits.");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    private const string MoeMmvqEnvVar = "DOTLLM_VULKAN_DISABLE_MOE_MMVQ";

    /// <summary>
    /// Loads a fresh Vulkan model (optionally with the MoE-mmvq opt-out set for
    /// the duration of the load — the gate is evaluated at kernel creation),
    /// prefills <c>ids[0, last)</c> and decodes the last token through the KV
    /// cache, returning the final logits row and the mmvq dispatch count.
    /// </summary>
    private static float[] RunVulkanDecode(
        string path, string spvDir, int[] ids, int[] pos, int last, int vocab,
        bool disableMoeMmvq, out long mmvqDispatches)
    {
        string? prev = Environment.GetEnvironmentVariable(MoeMmvqEnvVar);
        if (disableMoeMmvq)
            Environment.SetEnvironmentVariable(MoeMmvqEnvVar, "1");
        try
        {
            using var gguf = GgufFile.Open(path);
            var modelCfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(Architecture.Gemma4, modelCfg.Architecture);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, modelCfg, spvDir);
            using var kv = model.CreateKvCache(maxSeqLen: 16);

            using (var _ = model.Forward(ids.AsSpan(0, last), pos.AsSpan(0, last), -1, kv)) { }
            Assert.Equal(0L, model.MoeMmvqDispatchCount); // prefill (seqLen>1) must stay scalar

            using ITensor logits = model.Forward(ids.AsSpan(last, 1), pos.AsSpan(last, 1), -1, kv);
            mmvqDispatches = model.MoeMmvqDispatchCount;
            return LastRow(logits, vocab);
        }
        finally
        {
            if (disableMoeMmvq)
                Environment.SetEnvironmentVariable(MoeMmvqEnvVar, prev);
        }
    }

    private static void AssertEnvelope(
        float[] expected, float[] actual, int vocab, float absTol, float relTol,
        string label, out float worstDiff)
    {
        worstDiff = 0;
        int worst = -1;
        for (int c = 0; c < vocab; c++)
        {
            float diff = MathF.Abs(expected[c] - actual[c]);
            if (diff > worstDiff) { worstDiff = diff; worst = c; }
        }
        for (int c = 0; c < vocab; c++)
        {
            float diff = MathF.Abs(expected[c] - actual[c]);
            float bar = absTol + relTol * MathF.Abs(expected[c]);
            Assert.True(diff <= bar,
                $"{label} col {c}: expected={expected[c]:F6} vs actual={actual[c]:F6} "
                + $"(|diff|={diff:E3} > {bar:E3}); worst |diff|={worstDiff:E3} @ col {worst}");
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
        var row = new float[vocab];
        Array.Copy(all, total - vocab, row, 0, vocab);
        return row;
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
