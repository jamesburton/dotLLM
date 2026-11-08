using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1613 model-level A/B on the real GLM-4.5-Air REAP IQ4_XS file (opt-in: <c>DOTLLM_GLM4MOE_GGUF</c>; ~44 GiB resident, needs the Strix:GPU
/// and Strix:MEMORY locks). One load; the Q5_1 shared-expert down projections (24 layers) run once on the #1613 fast arms (MMQ prefill,
/// MMVQ decode) and once on the #1600 F32-in pair (<see cref="VulkanTransformerModel.Q5_1FastEnabled"/> off). The two differ only by the
/// Q8_1 activation rounding of 24 projections, so the logits must agree to a KL far below the activation-precision floor (~0.012, #873) and
/// the greedy token must be identical; the fast dispatch counter proves the fast arms ran (perturbation: switching them off changes it to 0).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Glm4MoeQ5_1FastPathRealFileTests(ITestOutputHelper output)
{
    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    private static double Kl(float[] p, float[] q)
    {
        static double[] LogSoftmax(float[] x)
        {
            double mx = x.Max(), s = 0;
            foreach (float f in x) s += Math.Exp(f - mx);
            double ls = Math.Log(s);
            return x.Select(f => f - mx - ls).ToArray();
        }
        var lp = LogSoftmax(p); var lq = LogSoftmax(q);
        double kl = 0;
        for (int i = 0; i < lp.Length; i++) kl += Math.Exp(lp[i]) * (lp[i] - lq[i]);
        return kl;
    }

    private static float[] LastRow(float[] logits, int rows) => logits.AsSpan((rows - 1) * (logits.Length / rows), logits.Length / rows).ToArray();

    [SkippableFact]
    public void FastArms_MatchF32InPair_AndRan()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_GLM4MOE_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanTransformerModel.LoadFromGguf(device, gguf, cfg, spvDir);
        output.WriteLine("post-load: " + device.MemorySnapshot());

        int[] ids = [12516, 5193, 264, 882, 11, 304, 264, 2613, 14117, 553, 279, 9392, 11, 1052, 12157, 264, 7640, 4753, 6941, 3757, 13, 3757, 572, 641];
        const int prefix = 20;
        int[] pos = Enumerable.Range(0, prefix).ToArray();

        (float[] prefillLast, float[] decode, long fast, long total) Run(bool fast)
        {
            model.Q5_1FastEnabled = fast;
            long f0 = model.Q5_1FastDispatchCount, t0 = model.Q5_1DenseDispatchCount;
            using var cache = model.CreateKvCache(256);
            using ITensor p = model.Forward(ids.AsSpan(0, prefix), pos, deviceId: -1, cache);
            float[] pl = Copy(p);
            float[] last = p.Shape[0] > 1 ? LastRow(pl, p.Shape[0]) : pl;
            using ITensor d = model.Forward(ids.AsSpan(prefix, 1), [prefix], deviceId: -1, cache);
            return (last, Copy(d), model.Q5_1FastDispatchCount - f0, model.Q5_1DenseDispatchCount - t0);
        }

        // ONE indicative A/B (the box is build-contended; the lead replays it in a quiet window): pp512 then 8 decode steps, slow/fast/slow/fast.
        double Time(bool fast, int rows)
        {
            model.Q5_1FastEnabled = fast;
            int[] p512 = Enumerable.Range(0, rows).Select(i => ids[i % ids.Length]).ToArray();
            int[] pos512 = Enumerable.Range(0, rows).ToArray();
            using var cache = model.CreateKvCache(rows + 16);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using (ITensor p = model.Forward(p512, pos512, deviceId: -1, cache)) { }
            double pp = sw.Elapsed.TotalMilliseconds;
            sw.Restart();
            for (int s = 0; s < 8; s++) using (ITensor d = model.Forward([ids[s]], [rows + s], deviceId: -1, cache)) { }
            double tg = sw.Elapsed.TotalMilliseconds / 8;
            output.WriteLine($"indicative fast={fast}: pp{rows} {pp:F0} ms ({rows * 1000 / pp:F0} tok/s), decode {tg:F1} ms/token");
            return pp;
        }
        foreach (bool f in new[] { false, true, false, true }) Time(f, 512);

        var slow = Run(fast: false);
        var quick = Run(fast: true);
        var slow2 = Run(fast: false);   // the F32-in pair is deterministic: a repeat pins the noise floor of this comparison at 0

        double klP = Kl(slow.prefillLast, quick.prefillLast), klD = Kl(slow.decode, quick.decode);
        double klRepeat = Math.Max(Kl(slow.prefillLast, slow2.prefillLast), Kl(slow.decode, slow2.decode));
        output.WriteLine($"Q5_1 dispatches: slow fast={slow.fast}/{slow.total}, fast fast={quick.fast}/{quick.total}; "
            + $"KL(slow||fast) prefill-last={klP:E3} decode={klD:E3}; slow-repeat KL={klRepeat:E3}; "
            + $"argmax prefill {Argmax(slow.prefillLast)}/{Argmax(quick.prefillLast)} decode {Argmax(slow.decode)}/{Argmax(quick.decode)}");
        Assert.Equal(0, slow.fast);
        Assert.True(quick.fast > 0 && quick.fast == quick.total, $"the fast arms ran {quick.fast} of {quick.total} Q5_1 dispatches");
        Assert.True(klP < 2e-3 && klD < 2e-3, $"KL(slow||fast) prefill {klP:E3} decode {klD:E3} exceeds 2e-3");
        Assert.Equal(Argmax(slow.prefillLast), Argmax(quick.prefillLast));
        Assert.Equal(Argmax(slow.decode), Argmax(quick.decode));
    }
}
