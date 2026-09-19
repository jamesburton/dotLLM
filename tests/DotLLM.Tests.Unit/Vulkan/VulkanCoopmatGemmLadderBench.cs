using System.Diagnostics;
using System.Globalization;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #439's discriminating experiment: does the PQ2_0 coopmat GEMM's <b>arithmetic
/// intensity</b> explain the 6.5x prefill gap against the PrismML llama.cpp fork on gfx1151?
/// </summary>
/// <remarks>
/// <para>
/// <b>What is being decided.</b> <c>.docs/COOPMAT_GEMM_DIAGNOSIS.md</c> measured occupancy and
/// register spill and killed both (the shipping kernel runs at ~9 waves/SIMD with zero spill).
/// The one hypothesis left standing is that every coopmat GEMM in the tree emits a 16x16 tile
/// from one subgroup — <b>4.0 MAC/byte staged</b> — against llama.cpp's applicable
/// <c>VK_KHR_cooperative_matrix</c> tile (<c>l_warptile_mmq</c>: 128x128, BK=32) at <b>32.0</b>.
/// It has a scar: <c>matmul_i2_s_f32_gemm_coopmat_wt.comp</c> already took 4.0 -&gt; 8.0 and
/// bought nothing (#229, #384-#386).
/// </para>
/// <para>
/// So this is a <b>three-point ladder at constant everything-else</b>, not an optimization.
/// (a) 16x16/1 subgroup = 4.0, (b) 64x64/4 = 16.0, (c) 128x128/4 = 32.0, all on one shader
/// template at BK=32, all unpinned wave64, all with the same LDS padding and the same staging
/// code. (a) is the control: it differs from the shipping kernel in BK, wave width and padding,
/// so only <b>(a) -&gt; (b) -&gt; (c)</b> is a clean intensity measurement. The shipping kernel
/// is reported as the absolute bar, and the (a)/shipping ratio tells you what the control itself
/// costs.
/// </para>
/// <para>
/// <b>A flat curve is the valuable result.</b> If (c) is no faster than (a), the last standing
/// hypothesis is dead and nobody needs to write the shared blocked-GEMM template.
/// </para>
/// <para>
/// <b>Methodology</b> is <see cref="VulkanPQ2_0GemmBench"/>'s, for the reasons documented there:
/// batched submissions behind one fence, interleaved passes with the order reversed every pass,
/// and the <b>median of per-pass ratios</b>. On this UMA part absolute throughput swings ~40%
/// with CPU memory-bandwidth contention, so only the same-session ratio is evidence; the
/// microsecond columns are context.
/// </para>
/// <para>
/// <b><c>lm_head</c> is the honest row.</b> At 248320x5120 it is 337 MB packed — the only shape
/// too large for gfx1151's 32 MB MALL. A per-layer projection is included for contrast and is
/// labelled; it sits in cache and will flatter every variant equally.
/// </para>
/// <para>
/// Enable with <c>DOTLLM_COOPMAT_LADDER_BENCH=1</c>. Tokens default to 256 (a multiple of the
/// widest tile, so no variant is penalised by a half-empty N tile) via
/// <c>DOTLLM_COOPMAT_LADDER_TOKENS</c>; repetitions per submission via
/// <c>DOTLLM_COOPMAT_LADDER_BATCH</c> (default 4, chosen so a full run fits inside a 30-minute
/// GPU lock).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanCoopmatGemmLadderBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 9;
    private const int GroupSize = 128;
    private const int GroupBytes = 34;

    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the benchmark with the xUnit output sink.</summary>
    /// <param name="output">Sink for the result table.</param>
    public VulkanCoopmatGemmLadderBench(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// <c>lm_head</c> first because it is the row that decides this, then one per-layer
    /// projection for contrast. Dimensions are the real Bonsai-2 27B tensor headers.
    /// </summary>
    private static readonly (string Tag, int M, int K)[] Shapes =
    [
        ("lm_head    (248320x5120) HONEST: 337 MB, exceeds 32 MB MALL", 248320, 5120),
        ("ffn_gate/up (17408x5120) CACHE-RESIDENT contrast",             17408, 5120),
    ];

    /// <summary>
    /// Runs the ladder: shipping kernel against each of the three points, then the two
    /// intensity steps that actually discriminate, (a) -&gt; (b) and (a) -&gt; (c).
    /// </summary>
    [SkippableFact]
    public void Bench_CoopmatGemmIntensityLadder()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_COOPMAT_LADDER_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_COOPMAT_LADDER_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int batch = EnvInt("DOTLLM_COOPMAT_LADDER_BATCH", 4);
        int tokens = EnvInt("DOTLLM_COOPMAT_LADDER_TOKENS", 256);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "VK_KHR_cooperative_matrix absent.");

        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}");
        _output.WriteLine($"spvDir: {spvDir}");
        _output.WriteLine($"tokens={tokens}  batch={batch}  {WarmupPasses} warmup + {Passes} interleaved order-reversed passes (median of per-pass RATIOS)");
        _output.WriteLine("");
        _output.WriteLine("Intensity: shipping/(a)/(d) = 4.0 MAC/byte, (b) = 16.0, (c)/(e) = 32.0.");
        _output.WriteLine("(d) and (e) are (a) and (c) with a cheaper PQ2_0 unpack and nothing else changed.");
        _output.WriteLine("Only (a)->(b)->(c) and (d)->(e) isolate intensity; only (a)->(d) and (c)->(e)");
        _output.WriteLine("isolate unpack cost. The shipping rows also carry BK, wave width and LDS padding.");
        _output.WriteLine("Two hypotheses in one session: #439 tile vs #440 RGP counters (WMMA 0.46% of issue");
        _output.WriteLine("slots, VALU 43.9%, ~95 VALU per WMMA). See .docs/COOPMAT_GEMM_DIAGNOSIS.md.");

        var pairs = new (string Label, PQ2_0GemmVariant Reference, PQ2_0GemmVariant Challenger)[]
        {
            ("shipping coopmat32 -> (a) 16x16/1sg   [control: BK+wave+pad, NOT intensity]",
                PQ2_0GemmVariant.Coopmat32, PQ2_0GemmVariant.Ladder16x16x1),
            ("(a) 4.0 -> (b) 16.0 MAC/byte          [INTENSITY, 4x]",
                PQ2_0GemmVariant.Ladder16x16x1, PQ2_0GemmVariant.Ladder64x64x4),
            ("(a) 4.0 -> (c) 32.0 MAC/byte          [INTENSITY, 8x — THE ANSWER]",
                PQ2_0GemmVariant.Ladder16x16x1, PQ2_0GemmVariant.Ladder128x128x4),
            ("(b) 16.0 -> (c) 32.0 MAC/byte         [INTENSITY, 2x]",
                PQ2_0GemmVariant.Ladder64x64x4, PQ2_0GemmVariant.Ladder128x128x4),
            ("shipping coopmat32 -> (c) 128x128/4sg [END TO END vs what ships]",
                PQ2_0GemmVariant.Coopmat32, PQ2_0GemmVariant.Ladder128x128x4),

            // --- #440's competing hypothesis: the dequant, not the tile. ---
            ("(a) -> (d) cheap unpack, SAME 16x16 tile   [UNPACK, tile held constant]",
                PQ2_0GemmVariant.Ladder16x16x1, PQ2_0GemmVariant.Ladder16x16x1FastUnpack),
            ("(c) -> (e) cheap unpack, SAME 128x128 tile [UNPACK at the big tile]",
                PQ2_0GemmVariant.Ladder128x128x4, PQ2_0GemmVariant.Ladder128x128x4FastUnpack),
            ("(d) -> (e) big tile, unpack held cheap     [INTENSITY, unpack controlled]",
                PQ2_0GemmVariant.Ladder16x16x1FastUnpack, PQ2_0GemmVariant.Ladder128x128x4FastUnpack),

            // --- Decomposes the (a)-vs-shipping bundle: Coopmat is 64-thread wave64 at BK=128,
            //     so this holds wave width constant and varies BK + padding + B coalescing. ---
            ("coopmat (wave64, BK=128) -> (a) (wave64, BK=32) [BK+pad+coalescing, wave held]",
                PQ2_0GemmVariant.Coopmat, PQ2_0GemmVariant.Ladder16x16x1),
        };

        foreach (var (label, refVariant, challVariant) in pairs)
        {
            if (!refVariant.IsSupportedOn(device) || !challVariant.IsSupportedOn(device))
            {
                _output.WriteLine($"### {label}: unsupported on this device");
                continue;
            }

            using var refKernel = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, refVariant);
            using var challKernel = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, challVariant);

            _output.WriteLine("");
            _output.WriteLine($"### {label}");
            _output.WriteLine("| shape | reference µs | challenger µs | speedup | ref GFLOP/s | chall GFLOP/s |");
            _output.WriteLine("|---|---:|---:|---:|---:|---:|");

            var rng = new Random(0x4_39);
            foreach (var (tag, m, k) in Shapes)
            {
                long rowBytes = (long)(k / GroupSize) * GroupBytes;
                long wBytes = m * rowBytes;
                using var bufW = device.Allocate((wBytes + 3) & ~3L);
                using var bufB = device.Allocate((long)tokens * k * sizeof(float));
                using var bufC = device.Allocate((long)tokens * m * sizeof(float));

                // Buffers are re-allocated per shape while the kernels persist, so the
                // handle-keyed descriptor cache can hand back a set bound to freed memory —
                // the failure mode is correct-then-zeros output, which mimics a truncated
                // kernel. Drop the cache whenever the buffers change.
                refKernel.InvalidateDescriptorCache();
                challKernel.InvalidateDescriptorCache();

                byte[] w = new byte[wBytes];
                rng.NextBytes(w);              // random packed codes; timing is data-independent
                float[] b = new float[(long)tokens * k];
                for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
                device.Upload(new ReadOnlySpan<byte>(w), bufW);
                device.Upload(b, bufB);

                (double refUs, double challUs, double ratio) =
                    MeasurePaired(device, refKernel, challKernel, bufW, bufB, bufC, m, k, tokens, batch);

                double flop = 2.0 * m * (double)k * tokens;
                _output.WriteLine($"| {tag} | {refUs:F1} | {challUs:F1} | {ratio:F2}x | "
                    + $"{flop / (refUs * 1e-6) / 1e9:F1} | {flop / (challUs * 1e-6) / 1e9:F1} |");
            }
        }
    }

    private static (double reference, double challenger, double ratio) MeasurePaired(
        VulkanDevice device, MatMulPQ2_0GemmF32Kernel reference, MatMulPQ2_0GemmF32Kernel challenger,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunPass(device, reference, w, b, c, m, k, n, batch);
            RunPass(device, challenger, w, b, c, m, k, n, batch);
        }

        var refUs = new double[Passes];
        var challUs = new double[Passes];
        var ratios = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            double tr, tc;
            if ((p & 1) == 0)   // alternate order so first-vs-second bias cancels
            {
                tr = RunPass(device, reference, w, b, c, m, k, n, batch);
                tc = RunPass(device, challenger, w, b, c, m, k, n, batch);
            }
            else
            {
                tc = RunPass(device, challenger, w, b, c, m, k, n, batch);
                tr = RunPass(device, reference, w, b, c, m, k, n, batch);
            }

            refUs[p] = tr;
            challUs[p] = tc;
            ratios[p] = tc > 0 ? tr / tc : 0;   // >1 means the challenger is faster
        }

        Array.Sort(refUs); Array.Sort(challUs); Array.Sort(ratios);
        return (refUs[Passes / 2], challUs[Passes / 2], ratios[Passes / 2]);
    }

    private static double RunPass(
        VulkanDevice device, MatMulPQ2_0GemmF32Kernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++)
            kernel.Record(ctx.CommandBuffer, w, b, c, m, k, n);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : fallback;
}
