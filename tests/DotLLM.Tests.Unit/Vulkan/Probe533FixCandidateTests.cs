using System.Diagnostics;
using System.Linq;
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
/// the matrix op. The MEASURED mechanism (v1/v2/v3 below) is narrower than the
/// "non-zero column count parity" reading this file first carried — v3 refuted
/// that: P holds exact zeros in exactly the columns where a LONGER dispatch of
/// the same prompt holds real V data, and AMD's coopmat P·V does not return the
/// same value for those columns' <c>0 * v</c> contributions. QK^T is innocent
/// (v1 changes nothing); P·V alone carries it (v2 goes to 0 everywhere).
/// </para>
/// <para>
/// Each candidate is a separate <c>.comp</c> so baseline and candidate are held
/// open in ONE process and measured same-session, order-reversed (this box's
/// cold-vs-warm launches have produced 2-3x phantom deltas):
/// </para>
/// <list type="bullet">
///   <item><c>_v1scalarqk</c> — QK^T on a scalar loop, P·V still coopmat (stage isolation).</item>
///   <item><c>_v2scalarpv</c> — P·V on a scalar loop, QK^T still coopmat (stage isolation).</item>
///   <item><c>_pre543</c> — the RETAINED CONTROL for #543: this shader as it stood
///         before it, with the #533 scalar P·V tail reading the f16 LDS staging tiles.
///         Production now reads f32 there (P from <c>sTile</c>, V staged through the idle
///         <c>oStage</c>), which measured 1.45E-04 against the scalar f32 FA kernel where
///         this control measures 3.30E-04 — and which is what makes a 1-token chunk write
///         bit-identical KV to a single-pass prefill. The control exists because a bound
///         nothing in the tree can violate is not a gate; the attribution arms that got
///         there (f32-P-only, f32-V-only, global-V, hdCeil-packed stage, mask-pass-only)
///         were deleted once they had answered, and their numbers are in
///         <c>.docs/ISSUE_543_MEASUREMENTS.md</c>.</item>
///   <item><c>_v3duppad</c> — both matmuls stay on the matrix cores; the last real KV
///         column is duplicated into the first pad slot when <c>tileLen</c> is odd, so the
///         non-zero column count is always even. REFUTED: no effect, which is what
///         moved the diagnosis from column-count parity to the <c>0 * v</c> reading.</item>
///   <item><c>[gate-off]</c> — the SHIPPED SPIR-V with <c>requireInvariantPv = 0</c>, which is
///         the pre-fix all-coopmat path and what the vendor policy runs on NVIDIA. Measured
///         bitwise identical to the retired <c>_pre533</c> copy on every arm, so it is the RED
///         control: being the same module, it cannot drift out of date the way a second copy of
///         the shader could.</item>
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
        "attention_flash_f32_coopmat",              // PRODUCTION (carries the #533 fix)
        GateOffCandidate,                           // SAME spv, requireInvariantPv=0 — the RED control
        "attention_flash_f32_coopmat_v1scalarqk",
        "attention_flash_f32_coopmat_v2scalarpv",
        "attention_flash_f32_coopmat_v3duppad",
        Pre543Candidate,
        // OPTIONAL, normally absent — the shader as it stood before the fix, whose
        // .comp/.spv were retired once [gate-off] was measured bitwise identical to
        // it. The name is kept so the pre-fix module can be dropped back into the
        // spv dir (`git show <old-rev>:native/vulkan/spv/<name>.spv`) to answer one
        // question the in-module control cannot: whether adding the push constant
        // and the gate branch changed the cost of the all-coopmat path itself, which
        // is what the NVIDIA exemption depends on. Reports UNAVAILABLE when absent.
        "attention_flash_f32_coopmat_pre533",
    ];

    /// <summary>
    /// The production shader with the #533 safe-tile gate forced OFF via the
    /// <c>requireInvariantPv</c> push constant — i.e. exactly what a device the
    /// vendor policy exempts (NVIDIA) executes. Not a separate SPIR-V: it is the
    /// shipped module with one push-constant word changed, which is what makes
    /// it a control the shader cannot drift away from.
    /// </summary>
    private const string GateOffCandidate = "attention_flash_f32_coopmat[gate-off]";

    /// <summary>
    /// #543's retained pre-fix control — see the class remarks. Unlike
    /// <c>_pre533</c> this one is a real committed shader, because the change it
    /// controls for is in the shader BODY and cannot be reproduced by flipping a
    /// specialization constant.
    /// </summary>
    private const string Pre543Candidate = "attention_flash_f32_coopmat_pre543";

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
        var differingByCandidate = new Dictionary<string, long>(StringComparer.Ordinal);
        var scalarErrorByCandidate = new Dictionary<string, float>(StringComparer.Ordinal);
        var gateByCandidate = new Dictionary<string, uint>(StringComparer.Ordinal);
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
                bool gateOff = name == GateOffCandidate;
                kernel = VulkanFlashAttentionCoopmatKernel.Create(
                    device, spvDir, FlashAttentionCoopmatVariant.Default,
                    gateOff ? "attention_flash_f32_coopmat" : name,
                    requireInvariantPv: gateOff ? 0u : null);
            }
            catch (Exception ex)
            {
                sb.AppendLine($"### {name}: UNAVAILABLE ({ex.GetType().Name}: {ex.Message})");
                continue;
            }

            using (kernel)
            {
                bool hd64 = File.Exists(Path.Combine(
                    spvDir, (name == GateOffCandidate ? "attention_flash_f32_coopmat" : name) + "_hd64.spv"));
                sb.AppendLine();
                sb.AppendLine($"### {name}  (hd64 spv present: {hd64}, REQUIRE_INVARIANT_PV={kernel.RequireInvariantPv})");

                long totalDiff = 0;

                float[] Run(int seqQ, int seqKv, int posOff, int slidingWindow = 0)
                {
                    kernel.Launch(bufQ, bufK, bufV, bufO, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOff, slidingWindow);
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
                    scalarErrorByCandidate[name] = mx;
                    gateByCandidate[name] = kernel.RequireInvariantPv;
                }

                // --- #543 coverage decay: the same error, bucketed by query-row
                //     block. A change that only touches KV-length-DEPENDENT tiles
                //     (the #533 scalar tail) has a coverage ceiling that falls
                //     with position: a query row at position p spans
                //     ceil((p+1)/64) KV tiles of which at most ~2 are unsafe, so
                //     at L=64 every tile is unsafe (the @64 line above is a BEST
                //     case) and by row 448 it is 1 in 8. Without this the @64
                //     number reads as a whole-prefill claim, which it is not.
                {
                    const int L = 512;
                    float[] got = Run(L, L, 0);
                    scalar.Launch(bufQ, bufK, bufV, bufO, L, L, NumHeads, NumKvHeads, HeadDim);
                    var sref = new float[(long)refN * QRow];
                    device.Download(bufO, sref);
                    var parts = new List<string>();
                    for (int blk = 0; blk < L / 64; blk++)
                    {
                        float mx = 0;
                        for (int r = blk * 64; r < (blk + 1) * 64; r++)
                            for (int c = 0; c < QRow; c++)
                                mx = MathF.Max(mx, MathF.Abs(got[(long)r * QRow + c] - sref[(long)r * QRow + c]));
                        parts.Add(string.Create(CultureInfo.InvariantCulture, $"rows{blk * 64,4}+: {mx:E3}"));
                    }
                    sb.AppendLine($"  vs scalar FA @{L}, maxAbs per 64-row block (coverage decay):");
                    sb.AppendLine("     " + string.Join("  ", parts));
                }

                // --- #543's own reachability shape: a short continuation chunk
                //     deep into the KV cache, where almost every tile is safe.
                {
                    float[] got = Run(32, 512, 480);
                    scalar.Launch(bufQ, bufK, bufV, bufO, 32, 512, NumHeads, NumKvHeads, HeadDim, 480);
                    var sref = new float[(long)refN * QRow];
                    device.Download(bufO, sref);
                    float mx = 0;
                    for (int i = 0; i < 32 * QRow; i++) mx = MathF.Max(mx, MathF.Abs(got[i] - sref[i]));
                    sb.AppendLine($"  vs scalar FA, continuation seqQ=32 @posOff=480 seqKv=512: maxAbs={mx:E3}");
                }

                // --- A. square prefill, short (base shader): rows of L vs the L=160 reference ---
                {
                    float[] reference = Run(160, 160, 0);
                    sb.AppendLine("  A. square prefill (base shader), reference L=160:");
                    foreach (int L in new[] { 61, 62, 63, 64, 65, 66, 67, 127, 128, 129 })
                    {
                        (string line, long d) = CompareRows(reference, Run(L, L, 0), L, L);
                        totalDiff += d;
                        sb.AppendLine("     " + line);
                    }
                }

                // --- B. square prefill, long (hd64 path when its spv exists): vs L=800 ---
                {
                    float[] reference = Run(refN, refN, 0);
                    sb.AppendLine($"  B. square prefill (seqKv>={VulkanFlashAttentionCoopmatKernel.SeqKvThreshold} -> hd64 gate), reference L={refN}:");
                    foreach (int L in new[] { 641, 642, 703, 704, 767, 768 })
                    {
                        (string line, long d) = CompareRows(reference, Run(L, L, 0), L, L);
                        totalDiff += d;
                        sb.AppendLine("     " + line);
                    }
                }

                // --- C. odd positionOffset (the chunk-continuation shape) ---
                //     seqQ=32 at offset P; all seqKv >= P+32 must give identical rows.
                {
                    sb.AppendLine("  C. continuation: seqQ=32 @ posOff, seqKv swept (all must match seqKv=760):");
                    foreach (int P in new[] { 0, 1, 16, 17, 100, 101 })
                    {
                        float[] reference = Run(32, 760, P);
                        float[] got = Run(32, P + 32, P);
                        (string line, long d) = CompareRows(reference, got, 32, P + 32);
                        totalDiff += d;
                        sb.AppendLine($"     posOff={P,3} ({(P % 2 == 0 ? "even" : "odd ")}) " + line);
                    }
                }

                // --- D. sliding window (Gemma-3 local layers / Mistral family) ---
                //     The fix deliberately does NOT force the scalar path on a
                //     sliding window; this arm is what makes that claim measured.
                {
                    float[] reference = Run(160, 160, 0, 40);
                    sb.AppendLine("  D. square prefill, slidingWindow=40, reference L=160:");
                    foreach (int L in new[] { 61, 62, 63, 64, 65, 66, 67 })
                    {
                        (string line, long d) = CompareRows(reference, Run(L, L, 0, 40), L, L);
                        totalDiff += d;
                        sb.AppendLine("     " + line);
                    }
                }

                differingByCandidate[name] = totalDiff;
            }
        }

        _out.WriteLine(sb.ToString());

        // REGRESSION GATE. The production shader must be exactly row-count
        // invariant; the retained pre-fix control must NOT be, or this test is
        // not discriminating anything (it passed for months while REPORTING
        // the defect — that is the landmine this assert removes).
        Assert.Equal(0L, differingByCandidate["attention_flash_f32_coopmat"]);

        // #543 ACCURACY GATE. Absolute, not a ratio: which of two kernels is the
        // weaker one is per-change, and a ratio gate is satisfied whenever both
        // sides regress together (the rule #549 was written down for). The bound
        // is the measured 1.446E-04 with headroom, and the retained pre-#543
        // control must exceed it, or the gate is guarding nothing.
        //
        // seqQ = seqKv = 64 causal is the discriminating shape BECAUSE every tile
        // there is KV-length-dependent and so takes the scalar tail; the same
        // error at rows 448+ of a p512 prefill is ~3.5E-05 either way, since a
        // row that deep spans 8 KV tiles of which at most ~2 are tail ones. The
        // per-64-row-block line above is the record of that decay — read it
        // before claiming this number describes a whole prefill.
        //
        // The bound applies only where the tail EXISTS. On a vendor #533 exempts
        // (NVIDIA) the production kernel is created with requireInvariantPv = 0,
        // the specialization constant dead-strips the tail before the backend
        // compiler runs, and every tile goes to the all-coopmat path — which is
        // the 3.3E-04 this gate asserts production stays under. Asserting it
        // there would fail on exactly the box the NVIDIA cross-check runs on, for
        // a path that is correct by design.
        const float Pre543Bound = 2.0E-04f;
        if (scalarErrorByCandidate.TryGetValue("attention_flash_f32_coopmat", out float prodErr))
        {
            uint prodGate = gateByCandidate["attention_flash_f32_coopmat"];
            if (prodGate == 0u)
            {
                sb.AppendLine();
                sb.AppendLine($"#543 accuracy bound NOT APPLIED: this device (VendorId " +
                              $"0x{device.VendorId:X4}) resolves requireInvariantPv to 0, so the f32 " +
                              $"scalar tail is compiled out and every tile is all-coopmat " +
                              $"({prodErr:E3}). The bound describes the gated path only.");
                _out.WriteLine(sb.ToString());
            }
            else
            {
                Assert.True(prodErr < Pre543Bound,
                    $"Production coopmat FA is {prodErr:E3} from the scalar f32 FA kernel at " +
                    $"seqQ=seqKv=64, above #543's bound of {Pre543Bound:E3}. Either the f32 scalar " +
                    "tail stopped running (check the #533 safe-tile gate and the sTile writeback) " +
                    "or its operands went back to the f16 staging tiles.");
            }
        }
        if (scalarErrorByCandidate.TryGetValue(Pre543Candidate, out float preErr))
        {
            // The control has to be skipped on the SAME condition as production, and
            // for a sharper reason than symmetry. Where requireInvariantPv resolves to
            // 0, BOTH arms compile their tail out and run the identical all-coopmat
            // path, so the control's 3.3E-04 clears the bound because it is measuring
            // the same thing production is — not because it is the pre-fix numerics.
            // Asserting it there passes while discriminating nothing, which reads as
            // "control validated" when the control was never exercised.
            if (gateByCandidate[Pre543Candidate] == 0u)
            {
                sb.AppendLine($"#543 control NOT EXERCISED: pre543 also resolves " +
                              $"requireInvariantPv to 0 here, so it is the same all-coopmat path as " +
                              $"production ({preErr:E3}). Neither half of the bound discriminates on " +
                              $"this vendor; run it on one the #533 gate applies to.");
                _out.WriteLine(sb.ToString());
            }
            else
            {
                Assert.True(preErr > Pre543Bound,
                    $"The retained pre-#543 control measured {preErr:E3}, inside the bound this gate " +
                    $"asserts production stays under ({Pre543Bound:E3}). The gate is no longer " +
                    "discriminating: re-derive the control (it must keep the f16-staged tail).");
            }
        }

        // The control must fire — but ONLY on hardware that actually has the
        // defect. #533 is an AMD implementation property: the same committed
        // SPIR-V is exactly 0-differing at every length on an RTX 3060, which is
        // how the issue was decided in the first place. So a clean control on a
        // non-AMD device is the EXPECTED cross-vendor result, not a broken
        // control, and failing there would make every NVIDIA/Intel run red for a
        // bug that vendor does not have. On AMD a clean control is exactly the
        // landmine this assert exists to catch (the sweep passed for months while
        // merely REPORTING the defect), so there it still fails.
        const uint VendorAmd = 0x1002;
        if (differingByCandidate.TryGetValue(GateOffCandidate, out long pre) && pre == 0)
        {
            Skip.IfNot(device.VendorId == VendorAmd,
                $"#533 gate-off control is invariant on this device (VendorId 0x{device.VendorId:X4}), " +
                "which is the expected non-AMD result — the fix arm above is clean but this run " +
                "cannot demonstrate that the test discriminates. Run it on AMD for that.");
            Assert.Fail(
                "The gate-off control came back invariant on an AMD device — either the driver " +
                "changed or requireInvariantPv is no longer reaching the shader. This test no " +
                "longer discriminates the fix; re-derive the control before trusting it.");
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Prefill cost — same-session, order-reversed A/B vs the baseline shader
    // ─────────────────────────────────────────────────────────────

    private const int WarmupPasses = 2;
    private const int Batch = 4;

    /// <summary>
    /// Pass count. The default 9 sees a large effect and cannot settle a small
    /// one; raise it with <c>DOTLLM_533_BENCH_PASSES</c> on a quiet box when the
    /// question is a few percent (#545 established this the hard way).
    /// </summary>
    private static int Passes =>
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_533_BENCH_PASSES"), out int p) && p > 0 ? p : 9;

    /// <summary>
    /// Baseline arm. Default is the SHIPPED module with the gate off, i.e. the
    /// pre-#533 all-coopmat path — the right reference for "what did the fix
    /// cost". Set <c>DOTLLM_533_COST_BASELINE=shipped</c> to make the baseline
    /// the production kernel at its AMD default instead, which is the reference
    /// for "what does this candidate cost against what we ship today" — the
    /// question #540 and #543 actually ask. Comparing two candidates through
    /// their ratios against a third arm does not answer it: the baseline column
    /// itself moved 28.1-38.9 us across arms in one session.
    /// </summary>
    private static bool BaselineIsShipped =>
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_533_COST_BASELINE"), "shipped", StringComparison.OrdinalIgnoreCase);

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

        // Reference arm = the pre-fix all-coopmat path, which is now the SHIPPED
        // module with requireInvariantPv = 0 rather than a second copy of the
        // shader. Challenger arm = everything else, including the production
        // kernel at its vendor default.
        using var baseline = VulkanFlashAttentionCoopmatKernel.Create(
            device, spvDir, FlashAttentionCoopmatVariant.Default,
            "attention_flash_f32_coopmat", requireInvariantPv: BaselineIsShipped ? null : 0u);

        var sb = new StringBuilder();
        sb.AppendLine($"Device: {device.DeviceName} SubgroupSize {device.SubgroupSize}");
        sb.AppendLine($"Baseline: attention_flash_f32_coopmat REQUIRE_INVARIANT_PV={baseline.RequireInvariantPv} " +
                      $"({(BaselineIsShipped ? "SHIPPED default — 'what does the candidate cost vs today'" : "gate off = pre-#533 all-coopmat")})");
        sb.AppendLine($"Passes={Passes} (median, min-max reported) Batch={Batch} dispatches/pass, order reversed per pass.");

        foreach (string name in Candidates.Where(n => !string.Equals(n, GateOffCandidate, StringComparison.Ordinal)))
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

        // Candidate 4 (gate coopmat FA off on AMD) costs exactly "the scalar FA
        // kernel instead of the coopmat one", at EVERY shape — measure it the
        // same way.
        using (var scalarFa = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir))
        {
            if (scalarFa is not null)
            {
                sb.AppendLine();
                sb.AppendLine("### baseline -> scalar FA kernel (candidate 4: coopmat gated off)");
                sb.AppendLine("| shape | baseline us (min-max) | scalar FA us (min-max) | speedup (median) |");
                sb.AppendLine("|---|---:|---:|---:|");
                var rng2 = new Random(0x534);
                foreach (var (tag, seqQ, seqKv, nh, nkv, hd) in Shapes)
                {
                    long qBytes = (long)seqQ * nh * hd * sizeof(float);
                    long kvBytes = (long)seqKv * nkv * hd * sizeof(float);
                    using var bufQ = device.Allocate(qBytes);
                    using var bufK = device.Allocate(kvBytes);
                    using var bufV = device.Allocate(kvBytes);
                    using var bufO = device.Allocate(qBytes);
                    device.Upload(RandomFloats(rng2, (int)(qBytes / sizeof(float))), bufQ);
                    device.Upload(RandomFloats(rng2, (int)(kvBytes / sizeof(float))), bufK);
                    device.Upload(RandomFloats(rng2, (int)(kvBytes / sizeof(float))), bufV);

                    var (r, c) = MeasurePaired(device,
                        (cb, b) => { for (int i = 0; i < b; i++) baseline.Record(cb, bufQ, bufK, bufV, bufO, seqQ, seqKv, nh, nkv, hd); },
                        (cb, b) => { for (int i = 0; i < b; i++) scalarFa.Record(cb, bufQ, bufK, bufV, bufO, seqQ, seqKv, nh, nkv, hd); });
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

    private static (string Line, long Differing) CompareRows(float[] reference, float[] got, int rows, int label)
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
        return (string.Create(CultureInfo.InvariantCulture,
            $"L={label,4} ({(label % 2 == 0 ? "even" : "odd ")}) differing={diff,7}/{(long)rows * QRow,-8} maxAbs={maxAbs:E3} rows[{firstRow}..{lastRow}]"), diff);
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
