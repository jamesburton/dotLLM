using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>Serialises the tests that flip the process-wide qwen4exp diagnostic switches (they would corrupt a concurrently running model).</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class Qwen4ExpDiagnosticSwitchCollection
{
    public const string Name = "Qwen4ExpDiagnosticSwitches";
}

/// <summary>
/// #873: the two oracle diagnostics that separated "Vulkan drifts from the CPU oracle" from "the model is chaotic under any perturbation":
/// <see cref="Qwen4ExpTransformerModel.DiagMoeInputNoise"/> (a calibrated relative perturbation of every MoE input) and
/// <see cref="MoeSwiGluMlp.DiagF32Activations"/> / <see cref="MoeSwiGluMlp.DiagF32ActivationsDense"/> (the F32-decoded control: dequantized weights
/// against UNQUANTIZED activations). A control that cannot disagree proves nothing, so each test pins that the switch is (a) inert when off,
/// (b) engaged when on, and (c) graded: a bigger perturbation moves the logits more.
/// </summary>
[Collection(Qwen4ExpDiagnosticSwitchCollection.Name)]
public sealed unsafe class Qwen4ExpNoiseFloorDiagnosticsTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-diag-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _d = [];

    public void Dispose()
    {
        foreach (var x in _d) x.Dispose();
        try { Directory.Delete(_dir, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private Qwen4ExpTransformerModel Load(Q4eQuant quant)
    {
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".gguf");
        File.WriteAllBytes(path, Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.KQuant256, quant));
        var (m, g, _) = ModelLoader.LoadFromGguf(path);
        _d.Add(g); _d.Add(m);
        return (Qwen4ExpTransformerModel)m;
    }

    private static float[] Logits(Qwen4ExpTransformerModel m)
    {
        var ids = new int[40];
        for (int i = 0; i < ids.Length; i++) ids[i] = 4 + (i * 7 + 3) % 50;
        using var st = m.CreateState();
        using var t = m.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, st, lastTokenLogitsOnly: false);
        return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    private static double MaxAbsDiff(float[] a, float[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    [Fact]
    public void MoeInputNoise_IsInertWhenOff_AndGradedWhenOn()
    {
        var m = Load(Q4eQuant.KQuant);
        float saved = Qwen4ExpTransformerModel.DiagMoeInputNoise;
        try
        {
            Qwen4ExpTransformerModel.DiagMoeInputNoise = 0f;
            var baseline = Logits(m);
            Assert.Equal(baseline, Logits(m));   // deterministic, so any later difference is the switch

            double[] diffs = new double[3];
            float[] eps = [1e-5f, 1e-3f, 1e-1f];
            for (int i = 0; i < eps.Length; i++)
            {
                Qwen4ExpTransformerModel.DiagMoeInputNoise = eps[i];
                diffs[i] = MaxAbsDiff(baseline, Logits(m));
                output.WriteLine($"eps {eps[i]:E0}: max|dlogit| {diffs[i]:E3}");
            }
            Assert.True(diffs[0] > 0, "even 1e-5 relative noise must move some logit (the switch is engaged)");
            Assert.True(diffs[1] > diffs[0] && diffs[2] > diffs[1], "a larger perturbation must move the logits more (the control is graded, not a no-op)");
        }
        finally { Qwen4ExpTransformerModel.DiagMoeInputNoise = saved; }
    }

    [Fact]
    public void F32ActivationControl_ChangesQuantizedModel_AndIsNoOpOnF32Weights()
    {
        bool savedMoe = MoeSwiGluMlp.DiagF32Activations, savedDense = MoeSwiGluMlp.DiagF32ActivationsDense;
        try
        {
            var q = Load(Q4eQuant.KQuant);
            var f = Load(Q4eQuant.F32);
            MoeSwiGluMlp.DiagF32Activations = false; MoeSwiGluMlp.DiagF32ActivationsDense = false;
            var qOff = Logits(q); var fOff = Logits(f);

            MoeSwiGluMlp.DiagF32Activations = true; MoeSwiGluMlp.DiagF32ActivationsDense = true;
            var qOn = Logits(q); var fOn = Logits(f);

            double dq = MaxAbsDiff(qOff, qOn), df = MaxAbsDiff(fOff, fOn);
            output.WriteLine($"quantized weights: control moves logits by {dq:E3}; F32 weights: {df:E3}");
            Assert.True(dq > 0, "the F32-activation control must differ from the quantized-activation path on a quantized model");
            Assert.Equal(0.0, df);   // nothing to un-quantize: both paths are the same F32 GEMM

            // And it is the activation quantization it removes, not a gross change: same model, small logit shift.
            double scale = qOff.Max(Math.Abs);
            Assert.True(dq < 0.25 * scale, $"control shift {dq} should be a small fraction of the logit scale {scale}");
        }
        finally { MoeSwiGluMlp.DiagF32Activations = savedMoe; MoeSwiGluMlp.DiagF32ActivationsDense = savedDense; }
    }
}
