using Xunit;

namespace DotLLM.Tests.Unit.Models.MiMoV2;

/// <summary>
/// The REAP50 Q2_K file's quantisation mix on a tiny random MiMo-V2 (hidden 256 / expert ff 64 so every K is legal for the
/// K-quants): routed gate/up banks Q2_K, down bank MXFP4, attention + dense FFN + embeddings Q8_0, lm_head Q6_K. The GGUF is
/// produced by <c>llama-quantize --tensor-type ffn_gate_exps=q2_k --tensor-type ffn_up_exps=q2_k --tensor-type ffn_down_exps=mxfp4
/// --output-tensor-type q6_k --token-embedding-type q8_0 f32.gguf mix.gguf Q8_0</c> from the converter's F32 file
/// (<c>tools/mimo2-tiny-oracle</c>, <c>make_tiny.py ... k256</c>), so llama.cpp and dotLLM read the SAME bytes. Fixture dir:
/// <c>~/.dotllm/test-cache/mimo2-tiny-k256/</c> (override <c>DOTLLM_MIMO2_TINY_K256_DIR</c>); skipped when absent.
/// </summary>
public sealed class MiMoV2QuantMixParityTests
{
    private static string? Dir()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_MIMO2_TINY_K256_DIR");
        string d = !string.IsNullOrEmpty(env) ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "mimo2-tiny-k256");
        return File.Exists(Path.Combine(d, "tiny_k256_mix.gguf")) ? d : null;
    }

    [SkippableFact]
    public void F32_Matches_Hf_On_The_K256_Geometry()
    {
        string? dir = Dir();
        Skip.If(dir is null, "mimo2-tiny-k256 fixture not present (see class remarks)");
        var (ours, vocab, _) = MiMoV2TinyOracleParityTests.RunDotLlm(dir!, file: "tiny_k256_f32.gguf");
        var vsHf = MiMoV2TinyOracleParityTests.Compare(ours, MiMoV2TinyOracleParityTests.ReadF32(Path.Combine(dir!, "hf_logits.f32")), vocab);
        Console.WriteLine($"[mimo2 k256 f32] dotLLM-vs-HF maxAbs={vsHf.MaxAbs:E2} maxKL={vsHf.MaxKl:E2} top1={vsHf.Top1:P0}");
        Assert.True(vsHf.MaxAbs < 5e-3f, $"maxAbs {vsHf.MaxAbs}");
        Assert.Equal(1.0, vsHf.Top1);
    }

    /// <summary>
    /// Random Q2_K weights make this model chaotic (a perturbed router flips experts), so a direct dotLLM-vs-llama.cpp comparison of
    /// the quantised file mixes two things: weight-decoding correctness and activation-quantisation noise amplified by routing.
    /// The F32-decoded control separates them: <c>mixdeq</c> is the same file with every tensor dequantised to F32 by gguf-py.
    /// </summary>
    [SkippableFact]
    public void Q2K_Mxfp4_Q80_Mix_Decodes_Like_LlamaCpp_And_Noise_Is_Of_Equal_Order()
    {
        string? dir = Dir();
        Skip.If(dir is null, "mimo2-tiny-k256 fixture not present (see class remarks)");
        Skip.IfNot(File.Exists(Path.Combine(dir!, "tiny_k256_mixdeq.gguf")), "tiny_k256_mixdeq.gguf (deq.py output) missing");
        var (mix, vocab, _) = MiMoV2TinyOracleParityTests.RunDotLlm(dir!, file: "tiny_k256_mix.gguf");
        var (deq, _, _) = MiMoV2TinyOracleParityTests.RunDotLlm(dir!, file: "tiny_k256_mixdeq.gguf");
        float[] llMix = MiMoV2TinyOracleParityTests.ReadF32(Path.Combine(dir!, "ll_mix.f32"));
        float[] llDeq = MiMoV2TinyOracleParityTests.ReadF32(Path.Combine(dir!, "ll_mixdeq.f32"));
        float[] llF32 = MiMoV2TinyOracleParityTests.ReadF32(Path.Combine(dir!, "ll_f32.f32"));
        // C: identical exact weights on both engines => F32 agreement (proves the dequant-to-F32 control file is faithful and the MoE path is exact)
        var c = MiMoV2TinyOracleParityTests.Compare(deq, llDeq, vocab);
        // A: dotLLM reading the packed bytes vs the exact decoded weights; B: llama.cpp likewise (activation quantisation + packed kernels)
        var a = MiMoV2TinyOracleParityTests.Compare(mix, llDeq, vocab);
        var b = MiMoV2TinyOracleParityTests.Compare(llMix, llDeq, vocab);
        var d = MiMoV2TinyOracleParityTests.Compare(mix, llMix, vocab);
        var quant = MiMoV2TinyOracleParityTests.Compare(llDeq, llF32, vocab);   // what Q2_K etc. did to the model
        Console.WriteLine($"[mimo2 k256 mix] C dotLLM(deq)-vs-llama(deq) maxAbs={c.MaxAbs:E2} maxKL={c.MaxKl:E2} | " +
                          $"A dotLLM(packed)-vs-exact maxKL={a.MaxKl:E2} top1={a.Top1:P0} | B llama(packed)-vs-exact maxKL={b.MaxKl:E2} top1={b.Top1:P0} | " +
                          $"D dotLLM(packed)-vs-llama(packed) maxKL={d.MaxKl:E2} top1={d.Top1:P0} | quantisation damage vs F32 original maxKL={quant.MaxKl:E2}");
        Assert.True(c.MaxAbs < 5e-3f, $"F32 control: dotLLM vs llama.cpp on identical exact weights maxAbs {c.MaxAbs}");
        Assert.True(quant.MaxKl > 100 * c.MaxKl, "control arm: Q2_K must visibly damage this random model, else the check is vacuous");
        // dotLLM's packed-weight noise must be of the same order as llama.cpp's own (not 10x worse): both are activation-quantisation noise
        Assert.True(a.MaxKl < 10 * Math.Max(b.MaxKl, 1e-3), $"dotLLM packed-vs-exact maxKL {a.MaxKl} vs llama.cpp's own {b.MaxKl}");
    }
}
