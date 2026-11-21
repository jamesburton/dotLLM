using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in A/B harness (issue #1102) for split-KV decode attention at Bonsai 2's full-attention shape (24 query heads over
/// 4 KV heads, head_dim 256): the per-query-head pass against the GQA-grouped pass, each over an F32 and an F16 cache.
/// </summary>
/// <remarks>
/// <para>One layer's decode attention per launch. Every arm runs in one process, interleaved, with the order reversed on
/// alternate rounds; the metric is the minimum wall time per arm (submit and wait included). The effective bandwidth column
/// counts each cached K and V byte once (the unique bytes a decode step must read).</para>
/// <para>Each arm's output is first compared with the F32 per-query-head arm on the same rows (F16 arms on F16-rounded rows),
/// so a fast arm that computes something else fails instead of reporting.</para>
/// <para>Enable with <c>DOTLLM_SPLITKV_DECODE_AB=1</c>; <c>DOTLLM_SPLITKV_DECODE_AB_KV</c> overrides the KV lengths
/// (comma-separated, default 8192,32768,131072).</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanSplitKvDecodeBench
{
    private const int Rounds = 9;

    // DOTLLM_SPLITKV_DECODE_AB_SHAPE=heads,kvHeads,headDim overrides the Bonsai 2 shape (Flint: 32,4,128, issue #1161).
    private static readonly int[] Shape = (Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB_SHAPE") ?? "24,4,256")
        .Split(',').Select(int.Parse).ToArray();
    private static readonly int NumHeads = Shape[0], NumKvHeads = Shape[1], HeadDim = Shape[2];
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the bench with the xUnit output sink.</summary>
    /// <param name="output">Sink for the results table.</param>
    public VulkanSplitKvDecodeBench(ITestOutputHelper output) => _output = output;

    /// <summary>Times the four arms at each KV length.</summary>
    [SkippableFact]
    public void AB_SplitKvDecode_Bonsai2Shape()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB") == "1", "DOTLLM_SPLITKV_DECODE_AB=1 to enable this A/B.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        int[] lengths = (Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB_KV") ?? "8192,32768,131072")
            .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();

        using var device = VulkanDevice.Create();
        using var k32 = VulkanSplitKvAttentionKernel.Create(device, spvDir, VulkanKvStorage.F32);
        using var k16 = VulkanSplitKvAttentionKernel.Create(device, spvDir, VulkanKvStorage.F16);
        _output.WriteLine($"device {device.DeviceName}");
        _output.WriteLine("seqKv\tarm\tmin_ms\tunique_GB/s\tmax|diff| vs f32-classic");

        foreach (int seqKv in lengths)
        {
            int kvElems = seqKv * NumKvHeads * HeadDim;
            var rng = new Random(seqKv);
            float[] q = new float[NumHeads * HeadDim];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);

            // Rows already rounded to F16, so every arm computes on the same values.
            float[] kv32 = new float[kvElems];
            ushort[] kv16 = new ushort[kvElems];
            for (int i = 0; i < kvElems; i++)
            {
                Half h = (Half)(rng.NextDouble() * 2 - 1);
                kv16[i] = BitConverter.HalfToUInt16Bits(h);
                kv32[i] = (float)h;
            }

            using var bq = device.Allocate((long)q.Length * sizeof(float));
            // The cache lives in device-local memory in the model (#1161: host-visible buffers read at a fraction of the bandwidth on
            // UMA and made every arm look latency-bound).
            using var bk32 = DeviceLocal(device, MemoryMarshal.AsBytes(kv32.AsSpan()));
            using var bv32 = DeviceLocal(device, MemoryMarshal.AsBytes(kv32.AsSpan()));
            using var bk16 = DeviceLocal(device, MemoryMarshal.AsBytes(kv16.AsSpan()));
            using var bv16 = DeviceLocal(device, MemoryMarshal.AsBytes(kv16.AsSpan()));
            using var bo = device.Allocate((long)q.Length * sizeof(float));
            device.Upload(q.AsSpan(), bq);
            kv32 = null!;
            kv16 = null!;

            var arms = new (string Name, VulkanSplitKvAttentionKernel Kernel, bool Gqa, bool V2, VulkanDevice.Buffer K, VulkanDevice.Buffer V, int Bytes)[]
            {
                ("f32-classic", k32, false, false, bk32, bv32, 4),
                ("f32-gqa", k32, true, false, bk32, bv32, 4),
                ("f16-classic", k16, false, false, bk16, bv16, 2),
                ("f16-gqa", k16, true, false, bk16, bv16, 2),
                ("f32-gqa2", k32, true, true, bk32, bv32, 4),
                ("f16-gqa2", k16, true, true, bk16, bv16, 2),
            };
            k16.PreferGqa = true;
            k16.PreferGqa2 = true;
            if (!k16.UsesGqa2(NumHeads, NumKvHeads, HeadDim)) arms = arms[..4];

            float[]? reference = null;
            var outputs = new float[arms.Length][];
            for (int a = 0; a < arms.Length; a++)
            {
                outputs[a] = new float[q.Length];
                Run(arms[a]);
                device.Download(bo, outputs[a]);
                reference ??= outputs[a];
            }

            var best = Enumerable.Repeat(double.MaxValue, arms.Length).ToArray();
            for (int round = 0; round < Rounds; round++)
            {
                for (int i = 0; i < arms.Length; i++)
                {
                    int a = (round & 1) == 0 ? i : arms.Length - 1 - i;
                    var sw = Stopwatch.StartNew();
                    Run(arms[a]);
                    best[a] = Math.Min(best[a], sw.Elapsed.TotalMilliseconds);
                }
            }

            for (int a = 0; a < arms.Length; a++)
            {
                float maxDiff = 0;
                for (int i = 0; i < q.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(outputs[a][i] - reference![i]));
                double gbs = 2.0 * kvElems * arms[a].Bytes / (best[a] / 1000) / 1e9;
                _output.WriteLine($"{seqKv}\t{arms[a].Name}\t{best[a]:F3}\t{gbs:F1}\t{maxDiff:E2}");
                Assert.True(maxDiff <= 1e-3f, $"{arms[a].Name} at seqKv {seqKv} disagrees with f32-classic by {maxDiff}");
            }

            void Run((string Name, VulkanSplitKvAttentionKernel Kernel, bool Gqa, bool V2, VulkanDevice.Buffer K, VulkanDevice.Buffer V, int Bytes) arm)
            {
                arm.Kernel.PreferGqa = arm.Gqa;
                arm.Kernel.PreferGqa2 = arm.V2;
                Assert.Equal(arm.Gqa, arm.Kernel.UsesGqa(NumHeads, NumKvHeads, HeadDim));
                arm.Kernel.Launch(bq, arm.K, arm.V, bo, 1, seqKv, NumHeads, NumKvHeads, HeadDim, seqKv - 1);
            }
        }
    }

    /// <summary>
    /// #1926 kernel microbench: the v2 grouped pass over F16, Q8_0, Q4_0, tq4 and tq3 caches, each as the retained pre-change body
    /// (<c>_pre1926</c>), the current body at GROUP = 8 (E1a / E3 loaders only) and the current body specialised on the shape's group
    /// (E1b). One layer's decode attention (split + merge) per dispatch; <see cref="QuantRepsPerSubmit"/> dispatches are recorded into
    /// one submit so the submit / fence cost is amortised; minimum and median per-dispatch time over <see cref="Rounds"/> interleaved,
    /// order-reversed rounds after one discarded warm-up round.
    /// </summary>
    /// <remarks>
    /// Before timing, every Q8_0 / Q4_0 pre-change arm is checked against the F32 v2 pass on the C#-dequantised rows (same operands,
    /// a few ULP), and every current arm against its own pre-change arm (1e-6 of the amplitude). tq arms are checked against their
    /// pre-change arm only (that body is held to the scratch oracle by <c>VulkanTurboQuantInPlaceKernelTests</c>). The rotations
    /// <c>tq_rotate_q</c> / <c>tq_rotate_o</c> are not in the timed region. <c>DOTLLM_SPLITKV_DECODE_AB_ARMS</c> narrows the KV formats
    /// (default <c>f16,q8_0,q4_0,tq4,tq3</c>); <c>DOTLLM_SPLITKV_DECODE_AB_KV</c> the lengths (default 8192,32768).
    /// </remarks>
    [SkippableFact]
    public void AB_SplitKvDecode_QuantArms()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB") == "1", "DOTLLM_SPLITKV_DECODE_AB=1 to enable this A/B.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "attention_f32_splitkv_gqa_v2_kvq4_pre1926.spv")), "pre-change control SPIR-V absent");
        int[] lengths = (Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB_KV") ?? "8192,32768")
            .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        string[] only = (Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB_ARMS") ?? "f16,q8_0,q4_0,tq4,tq3")
            .Split(',', StringSplitOptions.RemoveEmptyEntries);

        using var device = VulkanDevice.Create();
        _output.WriteLine($"device {device.DeviceName}; compute units {device.ComputeUnitCount?.ToString() ?? "unknown"}; shape {NumHeads}/{NumKvHeads}/{HeadDim}; {QuantRepsPerSubmit} dispatches per submit");
        _output.WriteLine("seqKv\tarm\tmin_ms\tmedian_ms\tunique_GB/s\tG_elem/s\tcheck");
        using var f32 = VulkanSplitKvAttentionKernel.Create(device, spvDir, VulkanKvStorage.F32);
        f32.PreferGqa = true;

        foreach (int seqKv in lengths)
        {
            int stride = NumKvHeads * HeadDim, kvElems = seqKv * stride;
            var rng = new Random(0x1926 + seqKv);
            float[] q = new float[NumHeads * HeadDim];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
            using var bq = device.Allocate((long)q.Length * sizeof(float));
            using var bo = device.Allocate((long)q.Length * sizeof(float));
            device.Upload(q.AsSpan(), bq);
            float[] kf = VulkanQ8KvAttentionKernelTests.Rand(rng, kvElems, 0.5f), vf = VulkanQ8KvAttentionKernelTests.Rand(rng, kvElems, 0.5f);

            var arms = new List<QuantArm>();
            var owned = new List<IDisposable>();
            try
            {
                foreach (string arm in only)
                {
                    VulkanKvStorage st = arm switch { "f16" => VulkanKvStorage.F16, "q8_0" => VulkanKvStorage.Q8_0, "q4_0" => VulkanKvStorage.Q4_0, _ => VulkanKvStorage.Tq };
                    DotLLM.Core.Attention.TurboQuantLayout? l = st == VulkanKvStorage.Tq
                        ? new DotLLM.Core.Attention.TurboQuantLayout(HeadDim, NumKvHeads, arm == "tq3" ? 3 : 4, DotLLM.Core.Attention.TurboQuantLayout.DefaultNorm) : null;
                    var pre = VulkanSplitKvAttentionKernel.Create(device, spvDir, st, l, "_pre1926");
                    var now = VulkanSplitKvAttentionKernel.Create(device, spvDir, st, l, "");
                    owned.Add(pre); owned.Add(now);
                    pre.PreferGqa = now.PreferGqa = true;
                    VulkanDevice.Buffer bk, bv;
                    TqKvOperands? ops = null;
                    double bpe;
                    if (l is { } tl)
                    {
                        var fx = TqFixture.Create(device, rng, tl, seqKv, deviceLocal: true);
                        owned.Add(fx);
                        bk = fx.K; bv = fx.V; ops = fx.Operands;
                        bpe = (tl.CodeUintsPerVec * 4.0 + 2.0) / HeadDim;
                    }
                    else
                    {
                        byte[] kb = SplitKvV2LoaderBitIdentityTests.Encode(st.KeyFormat(), kf), vb = SplitKvV2LoaderBitIdentityTests.Encode(st.ValueFormat(), vf);
                        bk = DeviceLocal(device, kb); bv = DeviceLocal(device, vb);
                        owned.Add(bk); owned.Add(bv);
                        bpe = (double)kb.Length / kvElems;
                        if (st is VulkanKvStorage.Q8_0 or VulkanKvStorage.Q4_0)
                        {
                            // Oracle for the pre-change arm: the F32 v2 pass on the dequantised rows.
                            float[] kd = st == VulkanKvStorage.Q8_0 ? VulkanQ8KvAttentionKernelTests.DequantizeRef(kb) : VulkanDenseQ8KvKernelTests.DequantizeQ4Ref(kb);
                            float[] vd = st == VulkanKvStorage.Q8_0 ? VulkanQ8KvAttentionKernelTests.DequantizeRef(vb) : VulkanDenseQ8KvKernelTests.DequantizeQ4Ref(vb);
                            using var bkd = DeviceLocal(device, MemoryMarshal.AsBytes(kd.AsSpan()));
                            using var bvd = DeviceLocal(device, MemoryMarshal.AsBytes(vd.AsSpan()));
                            float[] want = new float[q.Length], got = new float[q.Length];
                            f32.Launch(bq, bkd, bvd, bo, 1, seqKv, NumHeads, NumKvHeads, HeadDim, seqKv - 1);
                            device.Download(bo, want);
                            pre.Launch(bq, bk, bv, bo, 1, seqKv, NumHeads, NumKvHeads, HeadDim, seqKv - 1);
                            device.Download(bo, got);
                            float amp = want.Max(MathF.Abs), worst = 0;
                            for (int i = 0; i < q.Length; i++) worst = MathF.Max(worst, MathF.Abs(want[i] - got[i]));
                            Assert.True(worst <= 1e-5f * amp, $"{arm}-pre vs F32 on dequantised rows: {worst:G3} (amplitude {amp:G3})");
                            _output.WriteLine($"{seqKv}\t{arm}-pre vs f32-dequant oracle: max |diff| {worst / amp:E2} of amplitude");
                        }
                    }
                    int refIdx = arms.Count;
                    arms.Add(new QuantArm($"{arm}-pre", pre, false, bk, bv, ops, bpe, -1));
                    arms.Add(new QuantArm($"{arm}-g8", now, false, bk, bv, ops, bpe, refIdx));
                    arms.Add(new QuantArm($"{arm}-gspec", now, true, bk, bv, ops, bpe, refIdx));
                    arms.Add(new QuantArm($"{arm}-gspec-f16p", now, true, bk, bv, ops, bpe, refIdx, PartF16: true));
                    // #1926 P3: the same pipeline on the pre-change grid (the shared 256-workgroup target) vs the CU-sized default.
                    arms.Add(new QuantArm($"{arm}-gspec-wg256", now, true, bk, bv, ops, bpe, refIdx, TargetWg: 256));
                    // DOTLLM_SPLITKV_DECODE_AB_EXTRA=<suffix>: one more arm from attention_f32_splitkv_gqa_v2{twin}<suffix>.spv (dropped in
                    // the test bin's spv dir by hand), e.g. the previous commit's module, at the specialised GROUP.
                    string? extra = Environment.GetEnvironmentVariable("DOTLLM_SPLITKV_DECODE_AB_EXTRA");
                    if (!string.IsNullOrEmpty(extra) && File.Exists(Path.Combine(spvDir, st.ShaderName("attention_f32_splitkv_gqa_v2") + extra + ".spv")))
                    {
                        var x = VulkanSplitKvAttentionKernel.Create(device, spvDir, st, l, extra);
                        owned.Add(x);
                        x.PreferGqa = true;
                        arms.Add(new QuantArm($"{arm}-x{extra}-gspec", x, true, bk, bv, ops, bpe, refIdx));
                    }
                }
                kf = null!; vf = null!;

                var outputs = new float[arms.Count][];
                for (int a = 0; a < arms.Count; a++)
                {
                    outputs[a] = new float[q.Length];
                    Submit(arms[a], 1);
                    device.Download(bo, outputs[a]);
                }
                var check = new string[arms.Count];
                for (int a = 0; a < arms.Count; a++)
                {
                    if (arms[a].Ref < 0) { check[a] = "ref"; continue; }
                    float[] r = outputs[arms[a].Ref];
                    float amp = r.Max(MathF.Abs), worst = 0;
                    for (int i = 0; i < q.Length; i++) worst = MathF.Max(worst, MathF.Abs(outputs[a][i] - r[i]));
                    // F16 partials (E4): the stated 1e-3-of-amplitude tolerance (SplitKvPartialF16Tests); every other arm 1e-6.
                    float tol = arms[a].PartF16 ? 1e-3f : 1e-6f;
                    Assert.True(amp > 1e-4f && worst <= tol * amp, $"{arms[a].Name} vs {arms[arms[a].Ref].Name}: {worst:G3} (amplitude {amp:G3})");
                    check[a] = worst == 0 ? "bit-identical" : $"{worst / amp:E1}";
                }

                var times = new List<double>[arms.Count];
                for (int a = 0; a < arms.Count; a++) times[a] = new List<double>();
                for (int a = 0; a < arms.Count; a++) Submit(arms[a], QuantRepsPerSubmit);   // warm-up round, discarded
                for (int round = 0; round < Rounds; round++)
                {
                    for (int i = 0; i < arms.Count; i++)
                    {
                        int a = (round & 1) == 0 ? i : arms.Count - 1 - i;
                        var sw = Stopwatch.StartNew();
                        Submit(arms[a], QuantRepsPerSubmit);
                        times[a].Add(sw.Elapsed.TotalMilliseconds / QuantRepsPerSubmit);
                    }
                }
                for (int a = 0; a < arms.Count; a++)
                {
                    double min = times[a].Min(), med = times[a].Order().ElementAt(times[a].Count / 2), max = times[a].Max();
                    double elems = 2.0 * kvElems;
                    _output.WriteLine($"{seqKv}\t{arms[a].Name}\t{min:F4}\t{med:F4} [max {max:F4}]\t{elems * arms[a].Bpe / (min / 1000) / 1e9:F1}\t{elems / (min / 1000) / 1e9:F1}\t{check[a]}");
                }
            }
            finally
            {
                foreach (var d in owned) d.Dispose();
            }

            void Submit(QuantArm arm, int reps)
            {
                arm.Kernel.PreferGroupSpec = arm.Spec;
                arm.Kernel.PreferPartF16 = arm.PartF16;
                arm.Kernel.Gqa2TargetWorkgroups = arm.TargetWg > 0 ? arm.TargetWg : DefaultTarget(arm.Kernel);
                Assert.True(arm.Kernel.UsesGqa2(NumHeads, NumKvHeads, HeadDim), $"{arm.Name}: v2 pass not taken");
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                for (int r = 0; r < reps; r++)
                    arm.Kernel.Record(ctx.CommandBuffer, bq, arm.K, arm.V, bo, 1, seqKv, NumHeads, NumKvHeads, HeadDim, positionOffset: seqKv - 1, tq: arm.Tq);
                ctx.SubmitAndWait();
            }
        }
    }

    private static readonly Dictionary<VulkanSplitKvAttentionKernel, int> s_defaultTarget = new();

    private static int DefaultTarget(VulkanSplitKvAttentionKernel k)
    {
        if (!s_defaultTarget.TryGetValue(k, out int t)) s_defaultTarget[k] = t = k.Gqa2TargetWorkgroups;
        return t;
    }

    private sealed record QuantArm(string Name, VulkanSplitKvAttentionKernel Kernel, bool Spec, VulkanDevice.Buffer K, VulkanDevice.Buffer V,
        TqKvOperands? Tq, double Bpe, int Ref, bool PartF16 = false, int TargetWg = 0);

    // Dispatches per timed submit in the quantized-arm bench (each is one layer's split + merge).
    private const int QuantRepsPerSubmit = 20;

    private static VulkanDevice.Buffer DeviceLocal(VulkanDevice device, ReadOnlySpan<byte> data)
    {
        using var staging = device.Allocate(data.Length);
        device.Upload(data, staging);
        var dst = device.AllocateDeviceLocal(data.Length);
        device.CopyBufferSynchronous(staging, dst, (ulong)data.Length);
        return dst;
    }
}
