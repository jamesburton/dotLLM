using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Tests.Unit.Models.Qwen4Exp;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #873: why the Vulkan qwen4exp logits sit ~0.02-0.03 KL from the CPU oracle on real text - and why that is not a bug. The oracle (like
/// llama.cpp) quantizes the activations of every quantized GEMM (Q8_K / Q8_0); the Vulkan 1..8-row MMVQ paths do the same (Q8_1), but the
/// &gt;= 9-row prefill GEMMs read the activations unquantized. On this model (4-stream gated residual, 512-expert top-10 router, sparse
/// attention indexer) the activation-quantization noise is digitized into a macroscopic, entropy-proportional logit shift, so the two
/// activation precisions form two clusters ~0.02-0.05 KL apart on real text, with each cluster internally ~0.01.
/// The test pins the mechanism on a small K-quant checkpoint, using the oracle's own F32-activation control
/// (<see cref="MoeSwiGluMlp.DiagF32Activations"/> + <see cref="MoeSwiGluMlp.DiagF32ActivationsDense"/>) as the discriminator: a Vulkan prefill must be
/// markedly CLOSER to the unquantized-activation oracle than to the quantized-activation one. If someone makes the prefill GEMMs quantize their
/// activations (or breaks the control) the inequality flips. (The 1-row decode numbers are printed, not asserted: on this tiny model the decode
/// kernels are not in the quantized cluster the way they are on the real 176B file, where decode sits 0.016 from the quantized-activation oracle
/// and 0.046 from the F32 one.)
/// </summary>
[Trait("Category", "GPU")]
[Collection(Qwen4ExpDiagnosticSwitchCollection.Name)]
public sealed class VulkanQwen4ExpActivationPrecisionTests(ITestOutputHelper output)
{
    private static double Kl(float[] p, float[] q)
    {
        static double[] Lsm(float[] x)
        {
            double max = x.Max(), sum = 0;
            foreach (float v in x) sum += Math.Exp(v - max);
            double lse = max + Math.Log(sum);
            return x.Select(v => v - lse).ToArray();
        }
        double[] lp = Lsm(p), lq = Lsm(q);
        double kl = 0;
        for (int i = 0; i < p.Length; i++) kl += Math.Exp(lp[i]) * (lp[i] - lq[i]);
        return kl;
    }

    [SkippableFact]
    public void VulkanPrefillTracksUnquantizedActivations_DecodeTracksQuantizedActivations()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(4), spvDir);   // kq256-kquant: Q8_0 projections + Q4_K experts
        bool savedMoe = MoeSwiGluMlp.DiagF32Activations, savedDense = MoeSwiGluMlp.DiagF32ActivationsDense;
        const int prompts = 8, T = 40;
        double prefillVsQ = 0, prefillVsF = 0, decodeVsQ = 0, decodeVsF = 0;
        try
        {
            for (int s = 0; s < prompts; s++)
            {
                var ids = VulkanQwen4ExpParityTests.Ids(T, rig.Config.VocabSize, seed: 100 + s);
                var pos = Enumerable.Range(0, T).ToArray();

                MoeSwiGluMlp.DiagF32Activations = false; MoeSwiGluMlp.DiagF32ActivationsDense = false;
                var cpuQ = Q4eRig.Row(rig.Cpu.Forward(ids, pos, -1), T - 1);
                MoeSwiGluMlp.DiagF32Activations = true; MoeSwiGluMlp.DiagF32ActivationsDense = true;
                var cpuF = Q4eRig.Row(rig.Cpu.Forward(ids, pos, -1), T - 1);
                MoeSwiGluMlp.DiagF32Activations = savedMoe; MoeSwiGluMlp.DiagF32ActivationsDense = savedDense;

                // Vulkan single-shot prefill: a 40-row forward takes the unquantized-activation GEMMs.
                var vkPrefill = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);
                prefillVsQ += Kl(cpuQ, vkPrefill); prefillVsF += Kl(cpuF, vkPrefill);

                // Vulkan decode: 39-row prefill, then the last token as a 1-row forward (MMVQ, Q8_1 activations).
                using var st = rig.Vk.CreateState();
                using (var w = rig.Vk.Forward(ids.AsSpan(0, T - 1), pos.AsSpan(0, T - 1), -1, st)) { }
                var vkDecode = Q4eRig.Row(rig.Vk.Forward(ids.AsSpan(T - 1, 1), pos.AsSpan(T - 1, 1), -1, st), 0);
                decodeVsQ += Kl(cpuQ, vkDecode); decodeVsF += Kl(cpuF, vkDecode);
            }
        }
        finally { MoeSwiGluMlp.DiagF32Activations = savedMoe; MoeSwiGluMlp.DiagF32ActivationsDense = savedDense; }

        output.WriteLine($"mean KL over {prompts} prompts: prefill vs quantized-act oracle {prefillVsQ / prompts:E3}, vs F32-act oracle {prefillVsF / prompts:E3}; " +
                         $"decode vs quantized-act {decodeVsQ / prompts:E3}, vs F32-act {decodeVsF / prompts:E3}");
        Assert.True(prefillVsQ / prompts > 1e-5, "the quantized-activation oracle must be measurably away from the prefill (otherwise the comparison is two noise values)");
        Assert.True(prefillVsF < 0.5 * prefillVsQ,
            $"a >= 9-row Vulkan prefill must be markedly closer to the unquantized-activation oracle ({prefillVsF:E3}) than to the quantized one ({prefillVsQ:E3})");
    }
}
