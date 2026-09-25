using System.Diagnostics;
using System.Globalization;
using System.Text;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for issue #533 — measures each candidate FIX for the coopmat
/// flash-attention odd-KV parity defect, on correctness first and prefill cost
/// second.
/// </summary>
/// <remarks>
/// <para>
/// #533 established (and NVIDIA cross-checked) that
/// <c>attention_flash_f32_coopmat.comp</c> returns a different value for the
/// LAST causally visible query row whenever the effective KV length is odd, on
/// AMD only. The shader already zero-pads its KV tile to the full
/// <c>BC = 64</c> columns, so no partially-filled coopmat tile is ever fed to
/// the matrix op — the only thing an odd KV length changes before the coopmat
/// stage is the PARITY OF THE NON-ZERO COLUMN COUNT in the staged tile.
/// </para>
/// <para>
/// Each candidate is a separate <c>.comp</c> so baseline and candidate are held
/// open in ONE process and measured same-session, order-reversed (this box's
/// cold-vs-warm launches have produced 2-3x phantom deltas):
/// </para>
/// <list type="bullet">
///   <item><c>_v1scalarqk</c> — QK^T on a scalar loop, P·V still coopmat (stage isolation).</item>
///   <item><c>_v2scalarpv</c> — P·V on a scalar loop, QK^T still coopmat (stage isolation).</item>
///   <item><c>_v3duppad</c> — both matmuls stay on the matrix cores; the last real KV
///         column is duplicated into the first pad slot when <c>tileLen</c> is odd, so the
///         non-zero column count is always even. Masking already NEG_INFs <c>c &gt;= tileLen</c>,
///         so the softmax is untouched.</item>
/// </list>
/// <para>Enable with <c>DOTLLM_533_FIX_PROBE=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe533FixCandidateTests
{
    private const int NumHeads = 32, NumKvHeads = 8, HeadDim = 64;
    private const int QRow = NumHeads * HeadDim;
    private const int KvRow = NumKvHeads * HeadDim;

    private static readonly string[] Candidates =
    [
        "attention_flash_f32_coopmat",              // baseline (production)
        "attention_flash_f32_coopmat_v1scalarqk",
        "attention_flash_f32_coopmat_v2scalarpv",
        "attention_flash_f32_coopmat_v3duppad",
    ];

    private readonly ITestOutputHelper _out;
    public Probe533FixCandidateTests(ITestOutputHelper output) => _out = output;

    private static bool Enabled =>
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_533_FIX_PROBE"), "1", StringComparison.Ordinal);

    // ─────────────────────────────────────────────────────────────
    // Correctness
    // ─────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Candidates_Invariance()
    {
        Skip.IfNot(Enabled, "DOTLLM_533_FIX_PROBE=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(VulkanFlashAttentionCoopmatKernel.SupportsDevice(device), "No 16x16x16 f16->f32 subgroup coopmat tile.");

        const int refN = 800;   // > SeqKvThreshold (640) so the hd64 path is exercised too
        var rng = new Random(5340);
        float[] q = RandomFloats(rng, refN * QRow);
        float[] kk = RandomFloats(rng, refN * KvRow);
        float[] vv = RandomFloats(rng, refN * KvRow);

        using var bufQ = device.Allocate((long)refN * QRow * sizeof(float));
        using var bufK = device.Allocate((long)refN * KvRow * sizeof(float));
        using var bufV = device.Allocate((long)refN * KvRow * sizeof(float));
        using var bufO = device.Allocate((long)refN * QRow * sizeof(float));
        device.Upload(q, bufQ);
        device.Upload(kk, bufK);
        device.Upload(vv, bufV);

        var sb = new StringBuilder();
        sb.AppendLine($"Device: {device.DeviceName} (VendorId 0x{device.VendorId:X4}) SubgroupSize {device.SubgroupSize}");

        // The f16-class control: a candidate that silently routed to the scalar
        // F32 kernel would show 0 here and would trivially "pass" invariance.
        using var scalar = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir)
            ?? throw new InvalidOperationException("scalar FA kernel unavailable");

        foreach (string name in Candidates)
        {
            VulkanFlashAttentionCoopmatKernel kernel;
            try
            {
                kernel = VulkanFlashAttentionCoopmatKernel.Create(
                    device, spvDir, FlashAttentionCoopmatVariant.Default, name);
            }
            catch (Exception ex)
            {
                sb.AppendLine($"### {name}: UNAVAILABLE ({ex.GetType().Name}: {ex.Message})");
                continue;
            }

            using (kernel)
            {
                bool hd64 = File.Exists(Path.Combine(spvDir, name + "_hd64.spv"));
                sb.AppendLine();
                sb.AppendLine($"### {name}  (hd64 spv present: {hd64})");

                float[] Run(int seqQ, int seqKv, int posOff)
                {
                    kernel.Launch(bufQ, bufK, bufV, bufO, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOff);
                    var all = new float[(long)refN * QRow];
                    device.Download(bufO, all);
                    return all;
                }

                // --- f16-class perturbation proof: coopmat vs scalar FA at a fixed shape ---
                {
                    float[] got = Run(64, 64, 0);
                    scalar.Launch(bufQ, bufK, bufV, bufO, 64, 64, NumHeads, NumKvHeads, HeadDim);
                    var sref = new float[(long)refN * QRow];
                    device.Download(bufO, sref);
                    long d = 0; float mx = 0;
                    for (int i = 0; i < 64 * QRow; i++)
                    {
                        if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(sref[i])) d++;
                        mx = MathF.Max(mx, MathF.Abs(got[i] - sref[i]));
                    }
                    sb.AppendLine($"  vs scalar FA @64: differing={d}/{64 * QRow} maxAbs={mx:E3}  (must be f16-class ~1E-03)");
                }

                // --- A. square prefill, short (base shader): rows of L vs the L=160 reference ---
                {
                    float[] reference = Run(160, 160, 0);
                    sb.AppendLine("  A. square prefill (base shader), reference L=160:");
                    foreach (int L in new[] { 61, 62, 63, 64, 65, 66, 67, 127, 128, 129 })
                        sb.AppendLine("     " + CompareRows(reference, Run(L, L, 0), L, L));
                }

                // --- B. square prefill, long (hd64 path when its spv exists): vs L=800 ---
                {
                    float[] reference = Run(refN, refN, 0);
                    sb.AppendLine($"  B. square prefill (seqKv>={VulkanFlashAttentionCoopmatKernel.SeqKvThreshold} -> hd64 gate), reference L={refN}:");
                    foreach (int L in new[] { 641, 642, 703, 704, 767, 768 })
                        sb.AppendLine("     " + CompareRows(reference, Run(L, L, 0), L, L));
                }

                // --- C. odd positionOffset (the chunk-continuation shape) ---
                //     seqQ=32 at offset P; all seqKv >= P+32 must give identical rows.
                {
                    sb.AppendLine("  C. continuation: seqQ=32 @ posOff, seqKv swept (all must match seqKv=760):");
                    foreach (int P in new[] { 0, 1, 16, 17, 100, 101 })
                    {
                        float[] reference = Run(32, 760, P);
                        float[] got = Run(32, P + 32, P);
                        sb.AppendLine($"     posOff={P,3} ({(P % 2 == 0 ? "even" : "odd ")}) " + CompareRows(reference, got, 32, P + 32));
                    }
                }
            }
        }

        _out.WriteLine(sb.ToString());
    }

    // ─────────────────────────────────────────────────────────────
    // Prefill cost — same-session, order-reversed A/B vs the baseline shader
    // ─────────────────────────────────────────────────────────────

    private const int WarmupPasses = 2;
    private const int Passes = 9;
    private const int Batch = 4;

    private static readonly (string Tag, int SeqQ, int SeqKv, int NumHeads, int NumKvHeads, int HeadDim)[] Shapes =
    [
        ("p128_even  (base)", 128, 128, 9, 3, 64),
        ("p127_ODD   (base)", 127, 127, 9, 3, 64),
        ("p512_even  (base)", 512, 512, 32, 4, 64),
        ("p511_ODD   (base)", 511, 511, 32, 4, 64),
        ("p2048_even (hd64 gate)", 2048, 2048, 8, 2, 64),
        ("p2047_ODD  (hd64 gate)", 2047, 2047, 8, 2, 64),
        ("p512_hd128_even", 512, 512, 8, 8, 128),
        ("p511_hd128_ODD", 511, 511, 8, 8, 128),
    ];

    [SkippableFact]
    public void Candidates_PrefillCost()
    {
        Skip.IfNot(Enabled, "DOTLLM_533_FIX_PROBE=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(VulkanFlashAttentionCoopmatKernel.SupportsDevice(device), "No coopmat tile.");

        using var baseline = VulkanFlashAttentionCoopmatKernel.Create(
            device, spvDir, FlashAttentionCoopmatVariant.Default, Candidates[0]);

        var sb = new StringBuilder();
        sb.AppendLine($"Device: {device.DeviceName} SubgroupSize {device.SubgroupSize}");
        sb.AppendLine($"Passes={Passes} (median, min-max reported) Batch={Batch} dispatches/pass, order reversed per pass.");

        foreach (string name in Candidates.Skip(1))
        {
            VulkanFlashAttentionCoopmatKernel cand;
            try
            {
                cand = VulkanFlashAttentionCoopmatKernel.Create(
                    device, spvDir, FlashAttentionCoopmatVariant.Default, name);
            }
            catch (Exception ex)
            {
                sb.AppendLine($"### {name}: UNAVAILABLE ({ex.GetType().Name})");
                continue;
            }

            using (cand)
            {
                sb.AppendLine();
                sb.AppendLine($"### baseline -> {name}");
                sb.AppendLine("| shape | baseline us (min-max) | candidate us (min-max) | speedup (median) |");
                sb.AppendLine("|---|---:|---:|---:|");
                var rng = new Random(0x533);
                foreach (var (tag, seqQ, seqKv, nh, nkv, hd) in Shapes)
                {
                    long qBytes = (long)seqQ * nh * hd * sizeof(float);
                    long kvBytes = (long)seqKv * nkv * hd * sizeof(float);
                    using var bufQ = device.Allocate(qBytes);
                    using var bufK = device.Allocate(kvBytes);
                    using var bufV = device.Allocate(kvBytes);
                    using var bufO = device.Allocate(qBytes);
                    device.Upload(RandomFloats(rng, (int)(qBytes / sizeof(float))), bufQ);
                    device.Upload(RandomFloats(rng, (int)(kvBytes / sizeof(float))), bufK);
                    device.Upload(RandomFloats(rng, (int)(kvBytes / sizeof(float))), bufV);

                    var (r, c) = MeasurePaired(device,
                        (cb, b) => { for (int i = 0; i < b; i++) baseline.Record(cb, bufQ, bufK, bufV, bufO, seqQ, seqKv, nh, nkv, hd); },
                        (cb, b) => { for (int i = 0; i < b; i++) cand.Record(cb, bufQ, bufK, bufV, bufO, seqQ, seqKv, nh, nkv, hd); });

                    double ratio = c.Median > 0 ? r.Median / c.Median : 0;
                    sb.AppendLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {tag} | {r.Median:F2} ({r.Min:F2}-{r.Max:F2}) | {c.Median:F2} ({c.Min:F2}-{c.Max:F2}) | {ratio:F3}x |"));
                }
            }
        }

        _out.WriteLine(sb.ToString());
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    private static string CompareRows(float[] reference, float[] got, int rows, int label)
    {
        long diff = 0; float maxAbs = 0; int firstRow = -1, lastRow = -1;
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < QRow; i++)
            {
                int idx = r * QRow + i;
                if (BitConverter.SingleToInt32Bits(reference[idx]) != BitConverter.SingleToInt32Bits(got[idx]))
                {
                    diff++;
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[idx] - got[idx]));
                    if (firstRow < 0) firstRow = r;
                    lastRow = r;
                }
            }
        return string.Create(CultureInfo.InvariantCulture,
            $"L={label,4} ({(label % 2 == 0 ? "even" : "odd ")}) differing={diff,7}/{(long)rows * QRow,-8} maxAbs={maxAbs:E3} rows[{firstRow}..{lastRow}]");
    }

    private readonly record struct Stat(double Median, double Min, double Max);

    private static (Stat Reference, Stat Challenger) MeasurePaired(
        VulkanDevice device, Action<nint, int> recordReference, Action<nint, int> recordChallenger)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunPass(device, recordReference, Batch);
            RunPass(device, recordChallenger, Batch);
        }
        var refUs = new double[Passes];
        var challUs = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            if ((p & 1) == 0)
            {
                refUs[p] = RunPass(device, recordReference, Batch);
                challUs[p] = RunPass(device, recordChallenger, Batch);
            }
            else
            {
                challUs[p] = RunPass(device, recordChallenger, Batch);
                refUs[p] = RunPass(device, recordReference, Batch);
            }
        }
        Array.Sort(refUs); Array.Sort(challUs);
        return (new Stat(refUs[Passes / 2], refUs[0], refUs[Passes - 1]),
                new Stat(challUs[Passes / 2], challUs[0], challUs[Passes - 1]));
    }

    private static double RunPass(VulkanDevice device, Action<nint, int> record, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        record(ctx.CommandBuffer, batch);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
