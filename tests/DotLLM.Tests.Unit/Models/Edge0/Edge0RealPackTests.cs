using DotLLM.Core.Configuration;
using DotLLM.Models.Edge0;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>
/// Checks against the real <c>Edge0/Edge0-145B-q3k</c> pack, read in place from the Hugging Face hub cache (or
/// <c>DOTLLM_EDGE0_PACK_DIR</c>). Silently passes when the pack (or the files it needs) is absent: these are measurements on the real weights,
/// not unit tests, and the download is 105 GB.
/// </summary>
public sealed class Edge0RealPackTests(ITestOutputHelper output)
{
    internal static string? FindPackDir()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_PACK_DIR");
        if (!string.IsNullOrEmpty(env)) return Directory.Exists(env) ? env : null;
        string hub = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
            ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME") ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");
        string snaps = Path.Combine(hub, "models--Edge0--Edge0-145B-q3k", "snapshots");
        if (!Directory.Exists(snaps)) return null;
        return Directory.GetDirectories(snaps).OrderByDescending(d => Directory.GetLastWriteTimeUtc(d)).FirstOrDefault();
    }

    private static long LinkedLength(string path)
    {
        try
        {
            string p = File.ResolveLinkTarget(path, true)?.FullName ?? path;
            return File.Exists(p) ? new FileInfo(p).Length : -1;
        }
        catch (IOException) { return -1; }   // absent, or a link whose blob is still downloading
    }

    /// <summary>The files this test class needs, complete (a partial HF download leaves <c>.incomplete</c> blobs, never a snapshot link).</summary>
    private static string? CompletePack()
    {
        string? dir = FindPackDir();
        if (dir is null) return null;
        string[] need = ["native_experts.txt", "experts.bin", "labels.bin", "latent_dense_q8.bin", "index.txt", "dense.bin", "scales/l0.f32", "scales/l47.f32"];
        return need.All(f => LinkedLength(Path.Combine(dir, f)) > 0) ? dir : null;
    }

    private static double Rms(float[] v) => Math.Sqrt(v.Sum(x => (double)x * x) / v.Length);

    [Fact]
    public void RealPack_GeometryLabelsAndWeightsAreSane()
    {
        string? dir = CompletePack();
        if (dir is null) { output.WriteLine("Edge0 pack not complete locally: skipped."); return; }
        using var pack = Edge0Pack.Open(dir);
        Assert.Equal(48, pack.Layers);
        Assert.Equal(512, pack.Experts);
        Assert.Equal(1792, pack.Latent);
        Assert.Equal(8, pack.Groups);
        Assert.Equal(1_630_720L, pack.ExpertRecordBytes);
        Assert.True(pack.HasColumnScale);
        Assert.Equal(QuantizationType.Q8_0, pack.LatentDenseType);

        foreach (int l in new[] { 0, 23, 47 })
        {
            var hist = new int[8];
            foreach (int g in pack.Labels(l)) hist[g]++;
            output.WriteLine($"layer {l} experts per group: [{string.Join(", ", hist)}]");
            Assert.Equal(512, hist.Sum());
            var inv = pack.InverseColumnScale(l);
            output.WriteLine($"layer {l} 1/s: min {inv.ToArray().Min():G4} max {inv.ToArray().Max():G4}");
            Assert.All(inv.ToArray(), v => Assert.True(float.IsFinite(v) && v > 0));
        }

        foreach (var (l, e) in new[] { (0, 0), (0, 511), (23, 300), (47, 511) })
        {
            var gate = Edge0LatentMoeReference.Dequant(pack.GateBank(l) + (nint)(e * pack.ExpertRecordBytes), 640L * 1792, pack.GateUpType);
            var up = Edge0LatentMoeReference.Dequant(pack.UpBank(l) + (nint)(e * pack.ExpertRecordBytes), 640L * 1792, pack.GateUpType);
            var down = Edge0LatentMoeReference.Dequant(pack.DownBank(l) + (nint)(e * pack.ExpertRecordBytes), 1792L * 640, pack.DownType);
            output.WriteLine($"expert ({l},{e}): rms gate {Rms(gate):G3} up {Rms(up):G3} down {Rms(down):G3}");
            Assert.All(new[] { gate, up, down }, w => Assert.True(w.All(float.IsFinite) && Rms(w) > 1e-4 && Rms(w) < 10));
        }
        // Fingerprint of the column scale: the decoded Q3_K gate/up columns have rms proportional to 1/s (rms * s is constant to ~2%, measured
        // on layers 0 and 5). It pins "scales/lN.f32 belongs to THIS layer of experts.bin"; it does not by itself prove which way the engine
        // multiplies (that is the strata source, docs/EDGE0_PACK_FORMAT.md section 1).
        foreach (int l in new[] { 0, 5, 40 })
        {
            var colSq = new double[1792];
            for (int e = 0; e < 16; e++)
                foreach (nint bank in new[] { pack.GateBank(l), pack.UpBank(l) })
                {
                    var w = Edge0LatentMoeReference.Dequant(bank + (nint)(e * pack.ExpertRecordBytes), 640L * 1792, pack.GateUpType);
                    for (int r = 0; r < 640; r++)
                        for (int c = 0; c < 1792; c++) colSq[c] += (double)w[r * 1792 + c] * w[r * 1792 + c];
                }
            var inv = pack.InverseColumnScale(l);
            double[] a = colSq.Select(v => 0.5 * Math.Log(v)).ToArray(), b = inv.ToArray().Select(v => (double)Math.Log(v)).ToArray();
            double ma = a.Average(), mb = b.Average();
            double corr = a.Zip(b, (x, y) => (x - ma) * (y - mb)).Sum() / Math.Sqrt(a.Sum(x => (x - ma) * (x - ma)) * b.Sum(y => (y - mb) * (y - mb)));
            output.WriteLine($"layer {l}: corr(log column rms of gate/up, log 1/s) = {corr:F3}");
            Assert.True(corr > 0.9, $"layer {l}: the column scale does not match the decoded gate/up columns (corr {corr:F3})");
        }
        foreach (var (l, g) in new[] { (0, 0), (47, 7) })
        {
            var D = Edge0LatentMoeReference.Dequant(pack.D(l, g), 1792L * 2560, pack.LatentDenseType);
            var U = Edge0LatentMoeReference.Dequant(pack.U(l, g), 2560L * 1792, pack.LatentDenseType);
            output.WriteLine($"D({l},{g}) rms {Rms(D):G3}  U({l},{g}) rms {Rms(U):G3}");
            Assert.True(Rms(D) > 1e-3 && Rms(D) < 0.1 && Rms(U) > 1e-3 && Rms(U) < 0.1);   // NVFP4 pack of the same model: ~0.011 / ~0.0099
        }
    }

    [Fact]
    public void RealPack_RoutedLatentMoe_MatchesOracle_AndRealDataDiscriminatesTheMutants()
    {
        string? dir = CompletePack();
        if (dir is null) { output.WriteLine("Edge0 pack not complete locally: skipped."); return; }
        using var pack = Edge0Pack.Open(dir);
        var rng = new Random(3);
        var x = new float[2560];
        for (int i = 0; i < x.Length; i++)   // unit-RMS Gaussian, as in strata_formats.py `check --moe`
        {
            double u1 = 1 - rng.NextDouble(), u2 = rng.NextDouble();
            x[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        foreach (int layer in new[] { 0, 31 })
        {
            float[] router = ReadBf16(dir, $"blk.{layer}.ffn_gate_inp.weight", 512 * 2560);
            int[] lab = pack.Labels(layer).ToArray();
            float[] s = pack.InverseColumnScale(layer).ToArray().Select(v => 1f / v).ToArray();   // back to the stored s (all > 0 on the real pack)

            var want = Edge0LatentMoeReference.Routed(pack, layer, router, 10, x, lab, s, out int[] ids, out double[] w);
            var got = new float[2560];
            Edge0LatentMoe.Forward(pack, layer, router, 10, true, null, x, got, 1);
            double err = RelL2(got, want);
            var groups = ids.Select(e => lab[e]).Distinct().Count();
            output.WriteLine($"layer {layer}: experts [{string.Join(",", ids)}] groups {groups}  |routed| rms {Rms(want):G3}  rel L2 vs oracle {err:E3}");
            Assert.True(err < 0.1, $"layer {layer}: rel L2 {err:E3}");

            // The real data must be able to tell the mutants apart (substitution check): oracle with shifted labels / no colscale / s instead of 1/s.
            var shifted = Edge0LatentMoeReference.Routed(pack, layer, router, 10, x, lab.Select(g => (g + 1) % 8).ToArray(), s, out _, out _);
            var noScale = Edge0LatentMoeReference.Routed(pack, layer, router, 10, x, lab, null, out _, out _);
            var wrongDir = Edge0LatentMoeReference.Routed(pack, layer, router, 10, x, lab, s.Select(v => 1f / v).ToArray(), out _, out _);
            output.WriteLine($"  oracle distance to: shifted labels {RelL2(shifted, want):G3}, no colscale {RelL2(noScale, want):G3}, colscale as s {RelL2(wrongDir, want):G3}");
            Assert.True(RelL2(shifted, want) > 5 * err, "shifted-label mutant is not distinguishable on this real slice");
            Assert.True(RelL2(noScale, want) > 5 * err, "dropped-colscale mutant is not distinguishable on this real slice");
            Assert.True(RelL2(wrongDir, want) > 5 * err, "wrong-direction colscale mutant is not distinguishable on this real slice");
        }
    }

    private static double RelL2(float[] got, float[] want)
    {
        double num = 0, den = 0;
        for (int i = 0; i < got.Length; i++) { double d = got[i] - want[i]; num += d * d; den += (double)want[i] * want[i]; }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    /// <summary>Reads a BF16 tensor of <c>dense.bin</c> through <c>index.txt</c> (kind 4) as F32.</summary>
    private static float[] ReadBf16(string dir, string name, int count)
    {
        foreach (string line in File.ReadLines(Path.Combine(dir, "index.txt")))
        {
            var p = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (p.Length != 19 || p[0] != name) continue;
            Assert.Equal("4", p[2]);                     // BF16 raw
            long off = long.Parse(p[3], System.Globalization.CultureInfo.InvariantCulture), bytes = long.Parse(p[4], System.Globalization.CultureInfo.InvariantCulture);
            Assert.Equal(count * 2L, bytes);
            string dense = File.ResolveLinkTarget(Path.Combine(dir, "dense.bin"), true)?.FullName ?? Path.Combine(dir, "dense.bin");
            using var fs = new FileStream(dense, FileMode.Open, FileAccess.Read, FileShare.Read);
            fs.Seek(off, SeekOrigin.Begin);
            var raw = new byte[bytes];
            fs.ReadExactly(raw);
            var f = new float[count];
            for (int i = 0; i < count; i++) f[i] = BitConverter.Int32BitsToSingle((raw[2 * i] | raw[2 * i + 1] << 8) << 16);
            return f;
        }
        throw new InvalidDataException($"{name} not in index.txt");
    }
}
