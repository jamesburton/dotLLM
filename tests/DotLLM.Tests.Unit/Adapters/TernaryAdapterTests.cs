using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.ChunkReuse;
using System.Runtime.Intrinsics.X86;
using Xunit;

namespace DotLLM.Tests.Unit.Adapters;

/// <summary>
/// Ternary residual adapters (#1616) against the NeuroSymbolicAI reference (<c>Ternary.Core</c> + <c>Expert.Model</c>):
/// golden SHA-256s of reference outputs, an independent managed port of the reference pipeline, the real .nsadapter format,
/// gating/alpha/scales semantics, and engine wiring.
/// </summary>
public sealed unsafe class TernaryAdapterTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("nsad").FullName;
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    private static string Fx(string name) => Path.Combine(AppContext.BaseDirectory, "Adapters", "Fixtures", name + ".nsadapter");

    // ------------------------------------------------------------------ deterministic generators (same as the reference harness) ----

    private sealed class XorShift
    {
        private ulong _s;
        public XorShift(ulong seed) => _s = seed;
        public ulong Next() { _s ^= _s << 13; _s ^= _s >> 7; _s ^= _s << 17; return _s; }
        public float F() => ((Next() >> 40) / 8388608f) - 1f;
    }

    private static float[] Row(ulong seed, int d)
    {
        var g = new XorShift(seed);
        var h = new float[d];
        for (int i = 0; i < d; i++) h[i] = g.F() * (i % 97 == 0 ? 8f : 1f);
        return h;
    }

    private static NsAdapterPackage Synth(int rank, int d, ulong seed, float alpha, float thr, bool scales, int layer = 3)
    {
        var g = new XorShift(seed);
        sbyte W() => (sbyte)((int)(g.Next() % 3) - 1);
        var down = new PackedTernaryTensor(rank, d);
        var up = new PackedTernaryTensor(d, rank);
        for (int r = 0; r < rank; r++) for (int c = 0; c < d; c++) down[r, c] = W();
        for (int r = 0; r < d; r++) for (int c = 0; c < rank; c++) up[r, c] = W();
        if (scales)
        {
            float[] pool = [0.25f, 0.5f, 0.75f, 1f, 1.25f, 1.5f, 2f, 3f];
            for (int r = 0; r < rank; r++) down.Scales[r] = pool[(int)(g.Next() % 8)];
            for (int r = 0; r < d; r++) up.Scales[r] = pool[(int)(g.Next() % 8)];
        }
        var m = new NsAdapterManifest { TargetModel = "synthetic", TargetLayer = layer, InputDimension = d, Rank = rank, Alpha = alpha, ActivationThreshold = thr, Domain = "synthetic" };
        return new NsAdapterPackage(m, down, up);
    }

    private static string Hash(float[] a)
    {
        var b = new byte[a.Length * 4];
        Buffer.BlockCopy(a, 0, b, 0, b.Length);
        return Convert.ToHexStringLower(SHA256.HashData(b));
    }

    private static void AssertBitEqual(float[] expected, float[] actual, string ctx)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"{ctx}: element {i}: expected {expected[i]:R} got {actual[i]:R}");
    }

    private static IEnumerable<TernaryKernelIsa> Isas()
    {
        yield return TernaryKernelIsa.Scalar;
        if (Avx2.IsSupported) yield return TernaryKernelIsa.Avx2;
    }

    // ------------------------------------------------------------------ independent managed port of the reference ----

    /// <summary>Straight port of SimulatedTransformerSpine (absmax int8) + TernaryExpert.Forward over PackedTernaryTensor; the only addition is the documented ActivationThreshold gate.</summary>
    private static float[] RefApply(IReadOnlyList<NsAdapterPackage> pkgs, float[] hIn)
    {
        int d = hIn.Length;
        var h = (float[])hIn.Clone();
        float maxAbsH = 0f;
        for (int i = 0; i < d; i++) { float a = Math.Abs(h[i]); if (a > maxAbsH) maxAbsH = a; }
        float qs = maxAbsH > 1e-6f ? 127.0f / maxAbsH : 0.0f;
        var x = new sbyte[d];
        for (int i = 0; i < d; i++) x[i] = (sbyte)Math.Clamp((int)Math.Round(h[i] * qs), -128, 127);
        var deltaBuffer = new float[d];
        foreach (var p in pkgs)
        {
            var temp = RefExpert(p, x);
            for (int i = 0; i < d; i++) deltaBuffer[i] += temp[i];
        }
        for (int i = 0; i < d; i++) h[i] += deltaBuffer[i];
        return h;
    }

    private static float[] RefExpert(NsAdapterPackage p, sbyte[] x)
    {
        int d = p.Manifest.InputDimension, k = p.Manifest.Rank;
        var delta = new float[d];
        float gate = p.Manifest.Alpha;
        if (gate <= 0.0f) return delta;
        var inter = new float[k];
        for (int j = 0; j < k; j++)
        {
            long sum = 0;
            for (int c = 0; c < d; c++) { sbyte w = p.Down[j, c]; if (w == 1) sum += x[c]; else if (w == -1) sum -= x[c]; }
            inter[j] = sum * p.Down.Scales[j];
        }
        float maxAbs = 0f;
        for (int j = 0; j < k; j++) { float v = Math.Max(0f, inter[j]); inter[j] = v; if (v > maxAbs) maxAbs = v; }
        if (!(maxAbs > Math.Max(p.Manifest.ActivationThreshold, 1e-6f))) return delta;
        float quantScale = 127.0f / maxAbs;
        var q = new sbyte[k];
        for (int j = 0; j < k; j++) q[j] = (sbyte)Math.Clamp((int)Math.Round(inter[j] * quantScale), -128, 127);
        float dq = (maxAbs / 127.0f) * gate;
        for (int r = 0; r < d; r++)
        {
            long sum = 0;
            for (int j = 0; j < k; j++) { sbyte w = p.Up[r, j]; if (w == 1) sum += q[j]; else if (w == -1) sum -= q[j]; }
            delta[r] = sum * p.Up.Scales[r];
            delta[r] *= dq;
        }
        return delta;
    }

    private static float[] RunHook(NsAdapterPackage[] pkgs, float[] h, int layer, TernaryKernelIsa isa)
    {
        using var hook = new TernaryResidualHook(pkgs[0].Manifest.InputDimension) { Isa = isa };
        foreach (var p in pkgs) hook.Add(p);
        var row = (float[])h.Clone();
        hook.ModifyResidual(row, layer, 1);
        return row;
    }

    // ------------------------------------------------------------------ goldens from the reference implementation ----

    // SHA-256 of the output rows (4 rows each, Row(100+r)) produced by the REAL reference (Ternary.Core + Expert.Model, built
    // from neuro-symbolic-ai 15104eb): Apply = SimulatedTransformerSpine int8 step + TernaryExpert(Gate = manifest Alpha).Forward + h += delta.
    private static readonly Dictionary<string, string[]> Golden = new()
    {
        ["bracket-parity"] = ["4361063c738d350fe28c24e474a63e34c969f945285a99a1b0f2da106b91c8b4", "2f55f880e65e9569c6bb5300adcc7fffacacd38b9731101e407c33293f5a6ca9", "a61e53641e64e5575ea46fb15e53f604f69c4d4befc3ad3a0f65bdbe5d13b521", "393245cabd0206f49af7ee302dba130fcbaed3949d367f49abe9fc033d62d2f3"],
        ["csharp14-syntax-attractor"] = ["70805522d488a54ed0abfe6169f29bdcd380b75ff3cedc80c5620760efec0f96", "906368f593694b4ca8c6eaf1b221cba109d7dd3f77c081e10a8d30d907d7f14f", "0ebf1b2edd632ffdc6f13bb78e43160a99df465801f2327bcfde56b9bf1b7a1f", "7887ec147ebc7803c20304871f75f6a92bdd4edb087c840fe8ebf3edfb6dc50b"],
        ["modulo97-counter"] = ["032cef7216c280af19e4a0539c5ebb5494a0f029fedf3791805e51c1c0fbfc75", "d8347e4fbe3fd0fbb32c3f487a0e6285b876e6f305bfe9bc3fe23ead387e8cda", "d8e6b2d52a578be11a5ab965081c17e47d64ac84601551fb1f4dbc3227f3387c", "eac8dd3ee9db2174dff24f5e96c0426090cba4a32c92cd88f951620544a7361a"],
        ["synth-r4-scaled-a0.5"] = ["ed259c901de0699e6413e5fca08199b81458554e34736f7f6589b626dc717b2d", "049835e30d49cd81baa66b3e9915cec0f2ea5f276056f7425efbedec3d77d7e9", "08d5ff4b16459ff9576bdf8812669d6783302d8c68d03b8ae1a3f5e1c7b6f03a", "eaac898ea11a304ea5b6d7e6f3ec9d3cd54966477ee2ddbb751d4429b193cf9a"],
        ["synth-r32-scaled-a1"] = ["3fcbe9980b9b9a5694d057b77d883bac3c4d7262edfd66838cf8b3e84a762b3e", "278461b7446670d87d432ddfacb4c74e8917956397820df9644fd9c3a02f3ef8", "a88ea213deee1dd2d543137126fe29bd8a1e0277fb64d6453b1061d2430b5418", "fd1a8e6272cbeeed443a68554f4749e887c6bff107dd6b9ade56f86a4ba9d338"],
        ["synth-r7-d100-unitscale"] = ["effdc238af123562b6c6a4ac0d44e783f73684e69ab7e2a1b6caaffd4a7af33c", "0230120ba3e73cea9ccb78c86ab6943667259dc83302763ec2372c7d1213b14d", "c00f78c27df50fe9677049625d328a6df65aeccace065ef39df9ab0b60f9a40b", "f99eaba46dd31c5e236512c6a4d39a7ce2eafe3b8072311a050359015ac0ba50"],
        ["synth-r3-d1000-a0"] = ["c68fecbf59d0aa31541f81eee38a2323463b8794625967f559eb679bc5c84f3c", "af45c64ff9d90b8ad7834dbeb8ad67ecc9e0f595c162103fc7a7e2612c9d1e48", "da02ed217c271e19b710eaca0973a11a1ddd93944339df5e20e7ec1de0071ec3", "e431aea89597bbd49da3e963a20119831c22ce6425a7ea64226f0134dc53add6"],
    };

    private static NsAdapterPackage GoldenPackage(string name) => name switch
    {
        "synth-r32-scaled-a1" => Synth(32, 4096, 12, 1f, 0f, true),
        "synth-r3-d1000-a0" => Synth(3, 1000, 14, 0f, 0f, true),
        _ => NsAdapterPackage.Load(Fx(name)),
    };

    [Theory]
    [InlineData("bracket-parity")]
    [InlineData("csharp14-syntax-attractor")]
    [InlineData("modulo97-counter")]
    [InlineData("synth-r4-scaled-a0.5")]
    [InlineData("synth-r32-scaled-a1")]
    [InlineData("synth-r7-d100-unitscale")]
    [InlineData("synth-r3-d1000-a0")]
    public void Output_IsBitIdentical_ToReferenceGoldens_OnEveryIsa(string name)
    {
        var pkg = GoldenPackage(name);
        int d = pkg.Manifest.InputDimension;
        foreach (var isa in Isas())
            for (int r = 1; r <= 4; r++)
            {
                float[] outRow = RunHook([pkg], Row((ulong)(100 + r), d), pkg.Manifest.TargetLayer, isa);
                Assert.Equal(Golden[name][r - 1], Hash(outRow));
            }
    }

    [Fact]
    public void SampleAdapters_HaveTheDocumentedShapes()
    {
        var c = NsAdapterPackage.Load(Fx("csharp14-syntax-attractor"));
        Assert.Equal((14, 2, 4096), (c.Manifest.TargetLayer, c.Manifest.Rank, c.Manifest.InputDimension));
        var m = NsAdapterPackage.Load(Fx("modulo97-counter"));
        Assert.Equal((20, 1, 4096), (m.Manifest.TargetLayer, m.Manifest.Rank, m.Manifest.InputDimension));
        var b = NsAdapterPackage.Load(Fx("bracket-parity"));
        Assert.Equal((16, 1, 4096), (b.Manifest.TargetLayer, b.Manifest.Rank, b.Manifest.InputDimension));
    }

    // ------------------------------------------------------------------ vs independent managed port, many shapes ----

    [Fact]
    public void ManyShapes_AllIsas_MatchManagedReferencePort_Bitwise()
    {
        int cases = 0;
        foreach (int d in new[] { 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 100, 129, 257, 513 })
            foreach (int rank in new[] { 1, 2, 3, 8, 17, 40 })
                foreach (float alpha in new[] { 1f, 0.37f, 2.5f })
                {
                    var pkg = Synth(rank, d, (ulong)(d * 1000 + rank), alpha, 0f, scales: (cases & 1) == 0);
                    for (ulong r = 1; r <= 3; r++)
                    {
                        float[] h = Row(r, d);
                        float[] expected = RefApply([pkg], h);
                        foreach (var isa in Isas())
                            AssertBitEqual(expected, RunHook([pkg], h, 3, isa), $"d={d} r={rank} a={alpha} isa={isa}");
                    }
                    cases++;
                }
        Assert.True(cases >= 250);
    }

    [Fact]
    public void TwoAdaptersAtOneLayer_SumInRegistrationOrder_LikeReferenceDeltaBuffer()
    {
        int d = 130;
        var a = Synth(5, d, 1, 0.5f, 0f, true);
        var b = Synth(9, d, 2, 1.5f, 0f, true);
        for (ulong r = 1; r <= 8; r++)
        {
            float[] h = Row(r, d);
            foreach (var isa in Isas())
            {
                AssertBitEqual(RefApply([a, b], h), RunHook([a, b], h, 3, isa), $"a,b isa={isa}");
                AssertBitEqual(RefApply([b, a], h), RunHook([b, a], h, 3, isa), $"b,a isa={isa}");
            }
        }
        using var hook = new TernaryResidualHook(d);
        hook.Add(a); hook.Add(b);
        Assert.Equal(14, hook.RankAt(3));
        Assert.Equal(2, hook.CircuitCount);
    }

    // ------------------------------------------------------------------ gating / alpha ----

    [Fact]
    public void Alpha_IsAppliedExactly_AndNonPositiveAlphaIsIdentity()
    {
        int d = 96;
        var half = Synth(6, d, 4, 0.5f, 0f, true);
        var one = Synth(6, d, 4, 1f, 0f, true);
        float[] h = Row(9, d);
        float[] o1 = RunHook([one], h, 3, TernaryKernelIsa.Auto);
        float[] oh = RunHook([half], h, 3, TernaryKernelIsa.Auto);
        bool differs = false;
        for (int i = 0; i < d; i++)
        {
            Assert.Equal(h[i] + (o1[i] - h[i]) * 0.5f, oh[i], 3);   // alpha scales the delta (float tolerance: (a+b)-a rounding)
            if (o1[i] != h[i]) differs = true;
        }
        Assert.True(differs, "adapter must change the row");
        foreach (float a in new[] { 0f, -1f })
            AssertBitEqual(h, RunHook([Synth(6, d, 4, a, 0f, true)], h, 3, TernaryKernelIsa.Auto), $"alpha={a}");
    }

    [Fact]
    public void ActivationThreshold_GatesOnLargestReluDownValue()
    {
        int d = 200;
        float[] h = Row(5, d);
        var open = Synth(8, d, 7, 1f, 0f, true);
        float[] changed = RunHook([open], h, 3, TernaryKernelIsa.Auto);
        Assert.NotEqual(Hash(h), Hash(changed));
        // Largest ReLU'd down value m for this row, from the managed port; a threshold just below keeps it open, just above closes it.
        float m = MaxRelu(open, h);
        Assert.True(m > 1f);
        AssertBitEqual(changed, RunHook([Synth(8, d, 7, 1f, m * 0.99f, true)], h, 3, TernaryKernelIsa.Auto), "threshold below m");
        AssertBitEqual(h, RunHook([Synth(8, d, 7, 1f, m * 1.01f, true)], h, 3, TernaryKernelIsa.Auto), "threshold above m");
        foreach (var isa in Isas())
            AssertBitEqual(h, RunHook([Synth(8, d, 7, 1f, m * 1.01f, true)], h, 3, isa), $"closed isa={isa}");
    }

    private static float MaxRelu(NsAdapterPackage p, float[] h)
    {
        float maxH = h.Max(Math.Abs);
        float qs = 127f / maxH;
        var x = h.Select(v => (sbyte)Math.Clamp((int)Math.Round(v * qs), -128, 127)).ToArray();
        float m = 0f;
        for (int j = 0; j < p.Manifest.Rank; j++)
        {
            long s = 0;
            for (int c = 0; c < p.Manifest.InputDimension; c++) s += p.Down[j, c] * x[c];
            m = Math.Max(m, Math.Max(0f, s * p.Down.Scales[j]));
        }
        return m;
    }

    // ------------------------------------------------------------------ format ----

    [Fact]
    public void Format_RoundTrip_PreservesEverythingAndReferenceBytes()
    {
        foreach (string n in new[] { "csharp14-syntax-attractor", "synth-r4-scaled-a0.5", "synth-r7-d100-unitscale" })
        {
            var p = NsAdapterPackage.Load(Fx(n));
            string path = Path.Combine(_dir, n + ".nsadapter");
            p.Save(path);
            // Optional cross-implementation check: export what we wrote so the reference loader can be run on it (see docs).
            if (Environment.GetEnvironmentVariable("DOTLLM_NSADAPTER_EXPORT_DIR") is { Length: > 0 } export)
            {
                Directory.CreateDirectory(export);
                File.Copy(path, Path.Combine(export, n + ".nsadapter"), true);
            }
            var q = NsAdapterPackage.Load(path);
            Assert.Equal(p.Down.Data, q.Down.Data); Assert.Equal(p.Up.Data, q.Up.Data);
            Assert.Equal(p.Down.Scales, q.Down.Scales); Assert.Equal(p.Up.Scales, q.Up.Scales);
            Assert.Equal(p.Manifest.DownChecksum, q.Manifest.DownChecksum);
            Assert.Equal((p.Manifest.TargetLayer, p.Manifest.Rank, p.Manifest.InputDimension, p.Manifest.Alpha, p.Manifest.ActivationThreshold, p.Manifest.Domain),
                         (q.Manifest.TargetLayer, q.Manifest.Rank, q.Manifest.InputDimension, q.Manifest.Alpha, q.Manifest.ActivationThreshold, q.Manifest.Domain));
            // Our tensor serialisation is byte-identical to the reference's for the same tensor (reference-written down.bin re-serialises to itself).
            using var za = ZipFile.OpenRead(Fx(n));
            using var ms = new MemoryStream();
            p.Down.WriteTo(ms);
            using var raw = new MemoryStream();
            za.GetEntry("down.bin")!.Open().CopyTo(raw);
            Assert.Equal(raw.ToArray(), ms.ToArray());
        }
    }

    [Fact]
    public void Format_PackedCodes_AreBitNetLayout_00_01_11()
    {
        var t = new PackedTernaryTensor(1, 8);
        t[0, 0] = 1; t[0, 1] = -1; t[0, 2] = 0; t[0, 3] = 1; t[0, 4] = -1;
        Assert.Equal(0b01_00_11_01, t.Data[0]);      // col0 at bits 0-1: +1=01, col1: -1=11, col2: 0, col3: +1
        Assert.Equal(0b11, t.Data[1]);               // col4 -1
        t.Data[1] |= 0b10 << 2;                      // reserved code 10 at col5 reads as 0
        Assert.Equal((sbyte)0, t[0, 5]);
    }

    private byte[] MakeZip(Action<ZipArchive> build)
    {
        using var ms = new MemoryStream();
        using (var z = new ZipArchive(ms, ZipArchiveMode.Create, true)) build(z);
        return ms.ToArray();
    }

    private static void Put(ZipArchive z, string name, byte[] data)
    {
        using var s = z.CreateEntry(name).Open();
        s.Write(data);
    }

    private static byte[] Tensor(PackedTernaryTensor t) { using var ms = new MemoryStream(); t.WriteTo(ms); return ms.ToArray(); }

    private static byte[] ManifestJson(NsAdapterManifest m, string? extra = null)
    {
        string j = JsonSerializer.Serialize(m, NsAdapterJsonContextForTests.Options);
        if (extra is not null) j = j.TrimEnd('}').TrimEnd() + "," + extra + "}";
        return Encoding.UTF8.GetBytes(j);
    }

    private static (NsAdapterManifest m, PackedTernaryTensor down, PackedTernaryTensor up) Parts()
    {
        var p = Synth(3, 100, 5, 1f, 0f, true);
        p.Manifest.DownChecksum = p.Down.DataChecksum();
        p.Manifest.UpChecksum = p.Up.DataChecksum();
        return (p.Manifest, p.Down, p.Up);
    }

    [Fact]
    public void Load_ToleratesUnknownManifestFields_AndLegacyEntryNames()
    {
        var (m, down, up) = Parts();
        byte[] zip = MakeZip(z =>
        {
            Put(z, "manifest.json", ManifestJson(m, "\"SomeFutureField\": {\"x\": [1,2,3]}"));
            Put(z, "weights_down.bin", Tensor(down));
            Put(z, "weights_up.bin", Tensor(up));
        });
        var p = NsAdapterPackage.Load(new MemoryStream(zip));
        Assert.Equal(3, p.Manifest.Rank);
    }

    [Fact]
    public void Load_AcceptsHeaderlessRawTensors_LikeTheReference()
    {
        var (m, down, up) = Parts();
        m.DownChecksum = Convert.ToHexStringLower(SHA256.HashData(down.Data));
        m.UpChecksum = Convert.ToHexStringLower(SHA256.HashData(up.Data));
        byte[] zip = MakeZip(z => { Put(z, "manifest.json", ManifestJson(m)); Put(z, "down.bin", down.Data); Put(z, "up.bin", up.Data); });
        var p = NsAdapterPackage.Load(new MemoryStream(zip));
        Assert.All(p.Down.Scales, s => Assert.Equal(1f, s));
        Assert.Equal(down.Data, p.Down.Data);
    }

    [Fact]
    public void Load_RejectsCorruptMismatchedAndOversizedPackages()
    {
        var (m, down, up) = Parts();
        byte[] Good(NsAdapterManifest mm, byte[]? downBytes = null, byte[]? upBytes = null) =>
            MakeZip(z => { Put(z, "manifest.json", ManifestJson(mm)); Put(z, "down.bin", downBytes ?? Tensor(down)); Put(z, "up.bin", upBytes ?? Tensor(up)); });
        NsAdapterPackage.Load(new MemoryStream(Good(m)));   // control

        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream([1, 2, 3, 4, 5])));
        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(MakeZip(z => Put(z, "manifest.json", ManifestJson(m))))));   // no tensors

        var badSum = Clone(m); badSum.DownChecksum = new string('0', 64);
        Assert.Contains("checksum", Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(Good(badSum)))).Message);
        var noSum = Clone(m); noSum.UpChecksum = "";
        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(Good(noSum))));

        var flipped = Tensor(down); flipped[^1] ^= 0x04;
        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(Good(m, flipped))));

        var rankLie = Clone(m); rankLie.Rank = 4;
        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(Good(rankLie))));
        var dLie = Clone(m); dLie.InputDimension = 99;
        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(Good(dLie))));
        var negLayer = Clone(m); negLayer.TargetLayer = -2;
        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(Good(negLayer))));
        var zeroRank = Clone(m); zeroRank.Rank = 0;
        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(Good(zeroRank))));

        // up tensor with the wrong orientation (rank x D instead of D x rank): dimensions are cross-checked against the manifest
        var wrongUp = new PackedTernaryTensor(3, 100);
        var wu = Clone(m); wu.UpChecksum = wrongUp.DataChecksum();
        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(Good(wu, upBytes: Tensor(wrongUp)))));

        // entry larger than the manifest allows is refused before it is read
        Assert.Throws<InvalidDataException>(() => NsAdapterPackage.Load(new MemoryStream(Good(m, upBytes: new byte[Tensor(up).Length + 50_000]))));
    }

    private static NsAdapterManifest Clone(NsAdapterManifest m) => new()
    {
        TargetModel = m.TargetModel, TargetLayer = m.TargetLayer, InputDimension = m.InputDimension, Rank = m.Rank, Alpha = m.Alpha,
        ActivationThreshold = m.ActivationThreshold, Domain = m.Domain, Description = m.Description, CreatedAt = m.CreatedAt,
        DownChecksum = m.DownChecksum, UpChecksum = m.UpChecksum,
    };

    [Fact]
    public void Hook_ValidatesAgainstTheModel_AndRankCap()
    {
        using var hook = new TernaryResidualHook(100);
        hook.Add(Synth(4, 100, 1, 1f, 0f, false, layer: 5));
        hook.ValidateFor(100, 6);
        Assert.Throws<InvalidOperationException>(() => hook.ValidateFor(101, 6));
        Assert.Throws<InvalidOperationException>(() => hook.ValidateFor(100, 5));
        Assert.Throws<InvalidOperationException>(() => hook.Add(Synth(4, 64, 1, 1f, 0f, false)));
        Assert.True(hook.AppliesTo(5)); Assert.False(hook.AppliesTo(4)); Assert.False(hook.AppliesTo(6));
        hook.Add(Synth(250, 100, 2, 1f, 0f, false, layer: 7));
        Assert.Throws<NotSupportedException>(() => hook.Add(Synth(10, 100, 3, 1f, 0f, false, layer: 7)));
    }

    [Fact]
    public void ManifestDefaults_SurviveSourceGeneratedDeserialization()
    {
        var parsed = JsonSerializer.Deserialize("{}", NsAdapterJsonContextForTests.Context);
        var direct = new NsAdapterManifest();
        Assert.Equal((direct.Alpha, direct.ActivationThreshold, direct.Domain, direct.TargetModel), (parsed!.Alpha, parsed.ActivationThreshold, parsed.Domain, parsed.TargetModel));
        Assert.Equal(1.0f, parsed.Alpha);
    }

    [Fact]
    public void GgufMetadata_UsesTheReferenceKeySet()
    {
        var p = NsAdapterPackage.Load(Fx("csharp14-syntax-attractor"));
        var md = p.ToGgufMetadata();
        string[] keys = ["domain", "target_model", "layer", "target_layer", "input_dimension", "rank", "alpha", "activation_threshold", "description", "created_at", "down_checksum", "up_checksum"];
        Assert.Equal(keys.Select(k => "neuro_symbolic." + k).Order(StringComparer.Ordinal), md.Keys.Order(StringComparer.Ordinal));
        Assert.Equal("14", md["neuro_symbolic.layer"]); Assert.Equal("2", md["neuro_symbolic.rank"]); Assert.Equal("4096", md["neuro_symbolic.input_dimension"]);
    }

    // ------------------------------------------------------------------ engine wiring ----

    private sealed class Tiny : IDisposable
    {
        private readonly GgufFile _gguf;
        public ModelConfig Config { get; }
        public TransformerModel Model { get; }
        public Tiny(string dir)
        {
            string path = Path.Combine(dir, "tiny.gguf");
            File.WriteAllBytes(path, SyntheticDenseGguf.Build("llama"));
            _gguf = GgufFile.Open(path);
            Config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
            Model = TransformerModel.LoadFromGguf(_gguf, Config, ThreadingConfig.SingleThreaded);
        }
        public float[] Logits(int[] ids)
        {
            using ITensor t = Model.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1);
            return new ReadOnlySpan<float>((float*)t.DataPointer, (int)t.Shape.ElementCount).ToArray();
        }
        public void Dispose() { Model.Dispose(); _gguf.Dispose(); }
    }

    private static readonly int[] Ids = [3, 7, 11, 2, 40, 5];

    private TernaryResidualHook MakeHook(Tiny t, int layer, float alpha, ulong seed = 21)
    {
        var hook = new TernaryResidualHook(t.Config.HiddenSize);
        hook.Add(Synth(8, t.Config.HiddenSize, seed, alpha, 0f, true, layer));
        return hook;
    }

    [Fact]
    public void Engine_NoAdapter_Unchanged_AdapterChangesOutput_DetachRestores_AlphaZeroIdentity()
    {
        using var tiny = new Tiny(_dir);
        Assert.Null(tiny.Model.ResidualModifier);
        float[] baseline = tiny.Logits(Ids);
        AssertBitEqual(baseline, tiny.Logits(Ids), "determinism");
        using var hook = MakeHook(tiny, 1, 3f);
        hook.ValidateFor(tiny.Config.HiddenSize, tiny.Config.NumLayers);
        tiny.Model.ResidualModifier = hook;
        float[] steered = tiny.Logits(Ids);
        double maxDiff = 0;
        for (int i = 0; i < steered.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(steered[i] - baseline[i]));
        Assert.True(maxDiff > 1e-4, $"adapter did not change logits (max diff {maxDiff})");
        tiny.Model.ResidualModifier = null;
        AssertBitEqual(baseline, tiny.Logits(Ids), "after detach");
        using var zero = MakeHook(tiny, 0, 0f);
        tiny.Model.ResidualModifier = zero;
        AssertBitEqual(baseline, tiny.Logits(Ids), "alpha=0");
    }

    private sealed class Recorder : IResidualModifier
    {
        public readonly List<(int layer, int tokens)> Calls = new();
        public bool AppliesTo(int layerIndex) => true;
        public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount) => Calls.Add((layerIndex, tokenCount));
    }

    [Fact]
    public void Engine_ModifierRunsAfterEachLayer_InOrder_WithAllTokenRows()
    {
        using var tiny = new Tiny(_dir);
        var rec = new Recorder();
        tiny.Model.ResidualModifier = rec;
        _ = tiny.Logits(Ids);
        Assert.Equal(Enumerable.Range(0, tiny.Config.NumLayers).Select(l => (l, Ids.Length)).ToArray(), rec.Calls.ToArray());
    }

    [Fact]
    public void Engine_AdapterAppliedAtItsLayer_NotAnother()
    {
        using var tiny = new Tiny(_dir);
        using var a0 = MakeHook(tiny, 0, 3f, seed: 33);
        using var a1 = MakeHook(tiny, 1, 3f, seed: 33);
        tiny.Model.ResidualModifier = a0;
        float[] l0 = tiny.Logits(Ids);
        tiny.Model.ResidualModifier = a1;
        float[] l1 = tiny.Logits(Ids);
        Assert.NotEqual(l0, l1);
        var only = new OnlyAt(a0, 0);
        tiny.Model.ResidualModifier = only;
        AssertBitEqual(l0, tiny.Logits(Ids), "manual layer-0 injection == hook at layer 0");
        Assert.Equal(1, only.Fired);
    }

    private sealed class OnlyAt(TernaryResidualHook inner, int layer) : IResidualModifier
    {
        public int Fired;
        public bool AppliesTo(int layerIndex) => layerIndex == layer;
        public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount) { Fired++; inner.ModifyResidual(residual, layerIndex, tokenCount); }
    }
}

/// <summary>JSON helpers for tests: the context is internal to Models, so tests use their own reflection-free options mirroring it.</summary>
internal static class NsAdapterJsonContextForTests
{
    public static readonly JsonSerializerOptions Options = new() { WriteIndented = true, PropertyNameCaseInsensitive = true };
    public static readonly System.Text.Json.Serialization.Metadata.JsonTypeInfo<NsAdapterManifest> Context =
        (System.Text.Json.Serialization.Metadata.JsonTypeInfo<NsAdapterManifest>)typeof(NsAdapterPackage).Assembly
            .GetType("DotLLM.Models.Adapters.Ternary.NsAdapterJsonContext")!
            .GetProperty("Default")!.GetValue(null)!.GetType().GetProperty("NsAdapterManifest")!
            .GetValue(typeof(NsAdapterPackage).Assembly.GetType("DotLLM.Models.Adapters.Ternary.NsAdapterJsonContext")!.GetProperty("Default")!.GetValue(null))!;
}
