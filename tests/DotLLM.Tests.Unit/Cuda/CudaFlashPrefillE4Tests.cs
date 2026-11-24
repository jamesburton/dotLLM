using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1943 (prefill plan E4): opt-in tools that compare several builds of <c>attention_flash_mma_prefill_kvq.ptx</c> in ONE
/// process, so a candidate and the pre-E4 kernel share a GPU state and a session. Each build gets its own <see cref="CudaKernels"/>
/// over a copy of the test PTX directory with only that file replaced.
/// <list type="bullet">
/// <item><c>DOTLLM_E4_VARIANTS</c> = <c>name=path;name=path...</c> (the first is the reference build; mutants are named <c>m_*</c>).</item>
/// <item><see cref="FlashPrefillE4_VariantSweep"/>: per build and shape, the same oracles as the regular tests: the F16 entry against
/// scalar <c>attention_f16</c> and each quantized entry against scalar <c>attention_kvq</c> (normalised error, E1 bound), each
/// quantized entry bit-identical to its own F16 entry over the dequantized scratch (#1923), plus the ulp difference to the reference
/// build. Non-mutant builds must pass; a mutant is reported KILLED when any of those checks fails.</item>
/// <item><see cref="FlashPrefillE4_AbMicrobench"/> (<c>DOTLLM_E4_BENCH=1</c>): ms per launch, interleaved by round
/// (<c>DOTLLM_E4_ROUNDS</c>, default 3), Qwen3-4B 32/8/128 at 15,360 and 31,744 cached keys + 1,024 rows (f16, q8_0, q4_0), and
/// regression guards 28/4/128 (group 7: Qwen2.5-7B) and 14/2/64 (group 7, hd64: Qwen2.5-0.5B) on f16.</item>
/// </list>
/// </summary>
public sealed unsafe partial class CudaAttentionKvqTests
{
    private sealed record E4Build(string Name, CudaKernels Kernels, string Dir) : IDisposable
    {
        public bool IsMutant => Name.StartsWith("m_", StringComparison.Ordinal);
        public void Dispose()
        {
            Kernels.Dispose();
            try { Directory.Delete(Dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    private List<E4Build> LoadE4Builds()
    {
        string? spec = Environment.GetEnvironmentVariable("DOTLLM_E4_VARIANTS");
        Skip.If(string.IsNullOrWhiteSpace(spec), "DOTLLM_E4_VARIANTS not set");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        string src = _kernels!.PtxDirectory;
        var builds = new List<E4Build>();
        foreach (string item in spec!.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int eq = item.IndexOf('=');
            string name = item[..eq], path = item[(eq + 1)..];
            Assert.True(File.Exists(path), $"variant {name}: {path} not found");
            string dir = Path.Combine(Path.GetTempPath(), $"dotllm-e4-{Environment.ProcessId}-{name}");
            Directory.CreateDirectory(dir);
            foreach (string f in Directory.GetFiles(src, "*.ptx")) File.Copy(f, Path.Combine(dir, Path.GetFileName(f)), overwrite: true);
            File.Copy(path, Path.Combine(dir, "attention_flash_mma_prefill_kvq.ptx"), overwrite: true);
            var k = new CudaKernels(dir);
            Assert.True(k.HasAttentionFlashPrefillKvq(128, null, null), $"variant {name}: kvq module did not load");
            builds.Add(new E4Build(name, k, dir));
            _out.WriteLine($"build {name}: {path} (hd128 f16 static smem {k.AttentionFlashPrefillKvqStaticSmemBytes(128, null, null)} B)");
        }
        return builds;
    }

    [SkippableFact]
    public void FlashPrefillE4_VariantSweep()
    {
        var builds = LoadE4Builds();
        // #1943 review I1: judge every build on its E4 body. Without this the occupancy choice hands some shapes (hd64 f16 group 7 on
        // sm_86) to the _legacy twin and every mutant is inert there. Builds without twins (the pre-E4 base) are unaffected.
        bool? savedOverride = CudaKernels.FlashPrefillKvqLegacyBodyOverride;
        CudaKernels.FlashPrefillKvqLegacyBodyOverride = false;
        try
        {
            var failures = builds.ToDictionary(b => b.Name, _ => new List<string>());
            var maxUlpVsRef = builds.ToDictionary(b => b.Name, _ => 0);
            var worstScalar = builds.ToDictionary(b => b.Name, _ => 0.0);
            // F16 entry cases (#1923 E1) and the quantized FlashCases (#1850 / #1923 E2).
            var cases = F16FlashCases().Select(c => ((KvCacheDType?)null, (KvCacheDType?)null, (int)c[0], (int)c[1], (int)c[2], (int)c[3], (int)c[4]))
                .Concat(FlashCases().Select(c => ((KvCacheDType?)(KvCacheDType)c[0], (KvCacheDType?)(KvCacheDType)c[1],
                    (int)c[2], (int)c[3], (int)c[4], (int)c[5], (int)c[6])))
                .ToList();
            foreach (var (kFmt, vFmt, hd, nh, nkv, seqQ, offset) in cases)
            {
                int seqKv = offset + seqQ;
                using var f = new Fixture(this, kFmt ?? KvCacheDType.Q8_0, vFmt ?? KvCacheDType.Q8_0, hd, nh, nkv, seqQ, seqKv, seqKv,
                    window: 0, sinks: false);
                // Scalar oracle (base kernels).
                if (kFmt is null)
                    _kernels!.LaunchAttention(f.Q, f.KScratch, f.VScratch, f.OutScalar, seqQ, seqKv, nh, nkv, hd, offset, 0, _stream!.Handle);
                else
                    _kernels!.LaunchAttentionKvq(f.Q, f.KQuant, f.VQuant, 0, 0, f.OutScalar, seqQ, seqKv, nh, nkv, hd, offset, 0,
                        _stream!.Handle, 0, kFmt.Value, vFmt!.Value, seqKv, 0);
                _stream.Synchronize();
                ushort[] scalar = f.Read(f.OutScalar);
                ushort[]? reference = null;
                string tag = $"{kFmt?.ToString() ?? "F16"}/{vFmt?.ToString() ?? "F16"} hd{hd} {nh}/{nkv} q{seqQ} off{offset}";
                foreach (var b in builds)
                {
                    b.Kernels.LaunchAttentionFlashPrefillKvq(f.Q, f.KScratch, f.VScratch, null, null, f.OutRef, seqQ, seqKv, nh, nkv, hd,
                        offset, _stream.Handle);
                    if (kFmt is not null)
                        b.Kernels.LaunchAttentionFlashPrefillKvq(f.Q, f.KQuant, f.VQuant, kFmt, vFmt, f.OutNew, seqQ, seqKv, nh, nkv, hd,
                            offset, _stream.Handle);
                    _stream.Synchronize();
                    ushort[] f16 = f.Read(f.OutRef);
                    ushort[] got = kFmt is null ? f16 : f.Read(f.OutNew);
                    var fail = failures[b.Name];
                    if (got.Any(h => !float.IsFinite(HalfToFloat(h)))) fail.Add($"{tag}: non-finite output");
                    if (kFmt is not null && FirstDiff(f16, got) is int d and >= 0)
                        fail.Add($"{tag}: quantized entry != its F16 entry at {d}");
                    double worst = 0;
                    for (int i = 0; i < scalar.Length; i++)
                    {
                        double a = HalfToFloat(got[i]), s = HalfToFloat(scalar[i]);
                        worst = Math.Max(worst, Math.Abs(a - s) / (3e-3 + 3e-3 * Math.Abs(s)));
                    }
                    worstScalar[b.Name] = Math.Max(worstScalar[b.Name], worst);
                    if (!(worst <= 1.0)) fail.Add($"{tag}: normalised error vs scalar {worst:F2}");
                    reference ??= got;
                    int ulp = 0, ndiff = 0;
                    for (int i = 0; i < got.Length; i++)
                    {
                        int u = Math.Abs(OrderedHalf(got[i]) - OrderedHalf(reference[i]));
                        if (u != 0) ndiff++;
                        ulp = Math.Max(ulp, u);
                    }
                    maxUlpVsRef[b.Name] = Math.Max(maxUlpVsRef[b.Name], ulp);
                    _out.WriteLine($"  {b.Name,-10} {tag,-34} vs scalar {worst:F3}  vs ref max {ulp} ulp, {ndiff}/{got.Length} differ");
                }
            }
            foreach (var b in builds)
            {
                var fail = failures[b.Name];
                string verdict = b.IsMutant ? (fail.Count > 0 ? "KILLED" : "SURVIVED") : (fail.Count == 0 ? "PASS" : "FAIL");
                _out.WriteLine($"E4SWEEP {b.Name}: {verdict}; worst vs scalar {worstScalar[b.Name]:F3}, max {maxUlpVsRef[b.Name]} ulp vs " +
                               $"{builds[0].Name}; {fail.Count} failed checks{(fail.Count > 0 ? ": " + string.Join(" | ", fail.Take(4)) : "")}");
            }
            var bad = builds.Where(b => !b.IsMutant && failures[b.Name].Count > 0).Select(b => b.Name).ToList();
            Assert.True(bad.Count == 0, $"builds failing the oracles: {string.Join(", ", bad)}");
        }
        finally
        {
            CudaKernels.FlashPrefillKvqLegacyBodyOverride = savedOverride;
            foreach (var b in builds) b.Dispose();
        }
    }

    // Monotone integer image of an F16 bit pattern (so |a - b| counts representable values between them).
    private static int OrderedHalf(ushort h) => (h & 0x8000) != 0 ? -(h & 0x7FFF) : h;

    [SkippableFact]
    public void FlashPrefillE4_AbMicrobench()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_E4_BENCH") == "1", "DOTLLM_E4_BENCH=1 not set");
        var builds = LoadE4Builds().Where(b => !b.IsMutant).ToList();
        int rounds = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_E4_ROUNDS"), out int rr) ? rr : 3;
        const int seqQ = 1024, reps = 5;
        var arms = new List<(string Tag, int Hd, int Nh, int Nkv, int Depth, KvCacheDType? Fmt)>
        {
            ("32/8/128 d15360 f16", 128, 32, 8, 15360, null), ("32/8/128 d15360 q8_0", 128, 32, 8, 15360, KvCacheDType.Q8_0),
            ("32/8/128 d15360 q4_0", 128, 32, 8, 15360, KvCacheDType.Q4_0),
            ("32/8/128 d31744 f16", 128, 32, 8, 31744, null), ("32/8/128 d31744 q8_0", 128, 32, 8, 31744, KvCacheDType.Q8_0),
            ("32/8/128 d31744 q4_0", 128, 32, 8, 31744, KvCacheDType.Q4_0),
            ("28/4/128 d15360 f16", 128, 28, 4, 15360, null), ("14/2/64 d15360 f16", 64, 14, 2, 15360, null),
        };
        var ms = new Dictionary<(string, string), List<double>>();
        try
        {
            foreach (var grp in arms.GroupBy(a => (a.Hd, a.Nh, a.Nkv, a.Depth, IsQ4: a.Fmt == KvCacheDType.Q4_0)))
            {
                var (hd, nh, nkv, depth, isQ4) = grp.Key;
                int seqKv = depth + seqQ;
                var fmt = isQ4 ? KvCacheDType.Q4_0 : KvCacheDType.Q8_0;
                using var f = new Fixture(this, fmt, fmt, hd, nh, nkv, seqQ, seqKv, seqKv, window: 0, sinks: false);
                for (int r = 0; r < rounds; r++)
                    foreach (var arm in grp)
                        foreach (var b in builds)
                        {
                            nint k = arm.Fmt is null ? f.KScratch : f.KQuant, v = arm.Fmt is null ? f.VScratch : f.VQuant;
                            void Launch() => b.Kernels.LaunchAttentionFlashPrefillKvq(f.Q, k, v, arm.Fmt, arm.Fmt, f.OutNew, seqQ, seqKv,
                                nh, nkv, hd, depth, _stream!.Handle);
                            Launch(); Launch();
                            _stream!.Synchronize();
                            var sw = System.Diagnostics.Stopwatch.StartNew();
                            for (int i = 0; i < reps; i++) Launch();
                            _stream.Synchronize();
                            double t = sw.Elapsed.TotalMilliseconds / reps;
                            (ms.TryGetValue((arm.Tag, b.Name), out var l) ? l : ms[(arm.Tag, b.Name)] = []).Add(t);
                        }
            }
            foreach (var arm in arms)
            {
                var baseMed = Median(ms[(arm.Tag, builds[0].Name)]);
                foreach (var b in builds)
                {
                    var l = ms[(arm.Tag, b.Name)];
                    double med = Median(l);
                    _out.WriteLine($"E4BENCH {arm.Tag,-22} {b.Name,-10} median {med:F3} ms [{l.Min():F3}-{l.Max():F3}] " +
                                   $"speedup vs {builds[0].Name} {baseMed / med:F3} (rounds: {string.Join(" ", l.Select(x => x.ToString("F3")))})");
                }
            }
        }
        finally
        {
            foreach (var b in builds) b.Dispose();
        }
    }

    private static double Median(List<double> a)
    {
        var s = a.OrderBy(x => x).ToArray();
        return s.Length % 2 == 1 ? s[s.Length / 2] : 0.5 * (s[s.Length / 2 - 1] + s[s.Length / 2]);
    }
}

/// <summary>
/// #1943, regular (not opt-in): the shipped E4 entries and their <c>_legacy</c> (pre-E4 body) twins in the same PTX. Both are 16-key
/// tiles with the same arithmetic in the same order, so they must agree BIT FOR BIT on every shape and format (Q in registers and
/// the single-buffer schedule change no rounding); and the occupancy choice must take the twin exactly where the E4 entry fits fewer
/// blocks per SM (hd128 group 7 on sm_86) and the E4 entry at Qwen3-4B's group 4.
/// </summary>
public sealed unsafe partial class CudaAttentionKvqTests
{
    public static IEnumerable<object[]> E4BodyCases() =>
        F16FlashCases().Select(c => new object[] { "F16", c[0], c[1], c[2], c[3], c[4] })
            .Concat(FlashCases().Select(c => new object[] { $"{c[0]}/{c[1]}", c[2], c[3], c[4], c[5], c[6] }))
            .Append(new object[] { "F16", 128, 28, 4, 1000, 291 })          // group 7 hd128 (Qwen2.5-7B shape)
            .Append(new object[] { "Q8_0/Q4_0", 128, 28, 4, 33, 1024 })
            // #1943 review I1: group 1 (MHA; E4 on every GPU, and one warp owns every staged row group) and hd64 at the groups that
            // run E4 in production on sm_86 (4-5), plus hd64 group 8.
            .Append(new object[] { "F16", 128, 32, 32, 33, 291 })
            .Append(new object[] { "Q8_0/Q8_0", 128, 8, 8, 40, 4057 })
            .Append(new object[] { "Q4_0/Q8_0", 64, 4, 4, 47, 2002 })
            .Append(new object[] { "F16", 64, 16, 4, 1000, 291 })
            .Append(new object[] { "Q8_0/Q8_0", 64, 20, 4, 45, 70 })          // hd64 group 5
            .Append(new object[] { "F16", 64, 32, 4, 33, 0 });                 // hd64 group 8

    [SkippableTheory]
    [MemberData(nameof(E4BodyCases))]
    public void FlashPrefillKvq_E4AndLegacyBodies_BitIdentical(string fmts, int headDim, int numHeads, int numKvHeads, int seqQ, int offset)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        KvCacheDType? kFmt = fmts == "F16" ? null : Enum.Parse<KvCacheDType>(fmts.Split('/')[0]);
        KvCacheDType? vFmt = fmts == "F16" ? null : Enum.Parse<KvCacheDType>(fmts.Split('/')[1]);
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillKvq(headDim, kFmt, vFmt), "flash prefill kvq kernel not loaded");
        Skip.If(_kernels.AttentionFlashPrefillKvqStaticSmemBytes(headDim, kFmt, vFmt) == 0, "pre-E4 PTX (no E4 entry)");
        int seqKv = offset + seqQ;
        using var f = new Fixture(this, kFmt ?? KvCacheDType.Q8_0, vFmt ?? KvCacheDType.Q8_0, headDim, numHeads, numKvHeads, seqQ, seqKv,
            seqKv, window: 0, sinks: false);
        nint k = kFmt is null ? f.KScratch : f.KQuant, v = kFmt is null ? f.VScratch : f.VQuant;
        bool? saved = CudaKernels.FlashPrefillKvqLegacyBodyOverride;
        long legacy0 = CudaKernels.AttentionFlashPrefillKvqLegacyBodyLaunches;
        try
        {
            CudaKernels.FlashPrefillKvqLegacyBodyOverride = false;
            _kernels.LaunchAttentionFlashPrefillKvq(f.Q, k, v, kFmt, vFmt, f.OutNew, seqQ, seqKv, numHeads, numKvHeads, headDim, offset,
                _stream!.Handle);
            CudaKernels.FlashPrefillKvqLegacyBodyOverride = true;
            _kernels.LaunchAttentionFlashPrefillKvq(f.Q, k, v, kFmt, vFmt, f.OutRef, seqQ, seqKv, numHeads, numKvHeads, headDim, offset,
                _stream.Handle);
        }
        finally
        {
            CudaKernels.FlashPrefillKvqLegacyBodyOverride = saved;
        }
        _stream.Synchronize();
        Assert.Equal(legacy0 + 1, CudaKernels.AttentionFlashPrefillKvqLegacyBodyLaunches);   // the twin really ran once
        ushort[] e4 = f.Read(f.OutNew), old = f.Read(f.OutRef);
        AssertFinite(e4);
        int diff = FirstDiff(old, e4);
        Assert.True(diff < 0, $"E4 body differs from the legacy body at element {diff}: " +
            (diff >= 0 ? $"{HalfToFloat(e4[diff])} vs {HalfToFloat(old[diff])}" : ""));
    }

    [SkippableFact]
    public void FlashPrefillKvq_OccupancyChoice_TakesLegacyOnlyWhereItFitsMoreBlocks()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null || !_kernels.HasAttentionFlashPrefillKvq(128, null, null), "flash prefill kvq kernel not loaded");
        Skip.If(_kernels!.AttentionFlashPrefillKvqStaticSmemBytes(128, null, null) == 0, "pre-E4 PTX (no E4 entry)");
        Skip.If(CudaKernels.FlashPrefillKvqLegacyBodyOverride is not null, "body override set");
        var dev = CudaDevice.GetDevice(0);
        Skip.IfNot(dev.ComputeCapabilityMajor == 8 && dev.ComputeCapabilityMinor == 6, "the expected choices are for sm_86 (RTX 30xx)");
        // hd128 F16 entry index = 1 * 5 + 4.
        Assert.False(_kernels.UseFlashPrefillKvqLegacy(9, 128, 4), "group 4 (Qwen3-4B) must run the E4 body");
        Assert.True(_kernels.UseFlashPrefillKvqLegacy(9, 128, 7), "group 7 hd128 fits 2 legacy blocks vs 1 E4 block per SM");
        // The tie (1 vs 1 block per SM) must go to E4: pins `>` rather than `>=` (review M2).
        Assert.False(_kernels.UseFlashPrefillKvqLegacy(9, 128, 8), "group 8 hd128 ties (1/1), so it must run the E4 body");
    }
}
