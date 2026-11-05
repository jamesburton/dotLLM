using System.Globalization;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Tests.Unit.ChunkReuse;
using Xunit;

namespace DotLLM.Tests.Unit.Adapters;

/// <summary>
/// GGUF form of ternary adapters (#1619): package -> GGUF -> package is bit-identical (codes, scales, alpha/threshold bits, every manifest field),
/// the reference's <c>neuro_symbolic.*</c> keys are emitted verbatim, corrupt/mismatched files are rejected, and the server wiring attaches the hook.
/// </summary>
public sealed unsafe class NsAdapterGgufTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("nsgguf").FullName;
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    private static string Fx(string name) => Path.Combine(AppContext.BaseDirectory, "Adapters", "Fixtures", name + ".nsadapter");
    private string P(string n) => Path.Combine(_dir, n);

    private static NsAdapterPackage Synth(int rank, int d, ulong seed, float alpha, float thr, int layer, string domain)
    {
        ulong s = seed;
        ulong Next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
        var down = new PackedTernaryTensor(rank, d); var up = new PackedTernaryTensor(d, rank);
        for (int r = 0; r < rank; r++) for (int c = 0; c < d; c++) down[r, c] = (sbyte)((int)(Next() % 3) - 1);
        for (int r = 0; r < d; r++) for (int c = 0; c < rank; c++) up[r, c] = (sbyte)((int)(Next() % 3) - 1);
        for (int r = 0; r < rank; r++) down.Scales[r] = 0.01f + (Next() % 1000) / 777f;
        for (int r = 0; r < d; r++) up.Scales[r] = 0.01f + (Next() % 1000) / 913f;
        var m = new NsAdapterManifest
        {
            TargetModel = "synthetic-llama", TargetLayer = layer, InputDimension = d, Rank = rank, Alpha = alpha, ActivationThreshold = thr,
            Domain = domain, Description = "round trip é中 test", CreatedAt = new DateTimeOffset(2026, 11, 4, 12, 34, 56, 789, TimeSpan.FromHours(1)),
        };
        return new NsAdapterPackage(m, down, up);
    }

    private static void AssertSame(NsAdapterPackage a, NsAdapterPackage b)
    {
        Assert.Equal(a.Down.Rows, b.Down.Rows); Assert.Equal(a.Down.Columns, b.Down.Columns);
        Assert.Equal(a.Down.Data, b.Down.Data); Assert.Equal(a.Up.Data, b.Up.Data);
        Assert.Equal(a.Down.Scales.Select(BitConverter.SingleToInt32Bits), b.Down.Scales.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(a.Up.Scales.Select(BitConverter.SingleToInt32Bits), b.Up.Scales.Select(BitConverter.SingleToInt32Bits));
        var x = a.Manifest; var y = b.Manifest;
        Assert.Equal(x.TargetModel, y.TargetModel); Assert.Equal(x.TargetLayer, y.TargetLayer); Assert.Equal(x.InputDimension, y.InputDimension);
        Assert.Equal(x.Rank, y.Rank); Assert.Equal(x.Domain, y.Domain); Assert.Equal(x.Description, y.Description);
        Assert.Equal(BitConverter.SingleToInt32Bits(x.Alpha), BitConverter.SingleToInt32Bits(y.Alpha));
        Assert.Equal(BitConverter.SingleToInt32Bits(x.ActivationThreshold), BitConverter.SingleToInt32Bits(y.ActivationThreshold));
        Assert.Equal(x.CreatedAt, y.CreatedAt); Assert.Equal(x.CreatedAt.Offset, y.CreatedAt.Offset);
    }

    [Theory]
    [InlineData("bracket-parity")]
    [InlineData("csharp14-syntax-attractor")]
    [InlineData("modulo97-counter")]
    public void ReferenceSampleAdapters_RoundTripBitIdentical_ThroughGguf(string name)
    {
        var pkg = NsAdapterPackage.Load(Fx(name));
        byte[] gguf = NsAdapterGguf.Write([pkg]);
        File.WriteAllBytes(P("a.gguf"), gguf);
        Assert.True(NsAdapterGguf.LooksLikeGguf(P("a.gguf")));
        var back = Assert.Single(NsAdapterGguf.Read(P("a.gguf")));
        AssertSame(pkg, back);
        Assert.Equal(pkg.Down.DataChecksum(), back.Manifest.DownChecksum);
        // and back to a ZIP package, which the reference-format loader accepts
        back.Save(P("b.nsadapter"));
        AssertSame(pkg, NsAdapterPackage.Load(P("b.nsadapter")));
    }

    [Theory]
    [InlineData(5, 203, 0.123456789f, 0.3f)]     // D not a multiple of 4 or 8; alpha/threshold not exactly representable in G7
    [InlineData(1, 7, 1f, 0f)]
    [InlineData(32, 512, 2.5f, 1e-3f)]
    [InlineData(3, 64, 0.1f + 0.2f, 12345.678f)]
    public void SyntheticPackages_RoundTripBitIdentical_IncludingOddDimsAndExactFloats(int rank, int d, float alpha, float thr)
    {
        var pkg = Synth(rank, d, (ulong)(rank * 977 + d), alpha, thr, 3, "dom\"ain");
        var back = Assert.Single(NsAdapterGguf.Read(WriteFile([pkg])));
        AssertSame(pkg, back);
        // The string form uses the reference's G7: it does NOT round-trip this alpha, so the exact binary32 key is what makes it bit-exact.
        if (alpha == 0.123456789f)
            Assert.NotEqual(BitConverter.SingleToInt32Bits(alpha), BitConverter.SingleToInt32Bits(float.Parse(pkg.ToGgufMetadata()["neuro_symbolic.alpha"], CultureInfo.InvariantCulture)));
    }

    private string WriteFile(IReadOnlyList<NsAdapterPackage> pkgs, string name = "x.gguf")
    {
        NsAdapterGguf.Write(P(name), pkgs);
        return P(name);
    }

    [Fact]
    public void FirstAdapter_UsesTheReferenceKeySetVerbatim()
    {
        var pkg = Synth(4, 64, 9, 1f, 0f, 2, "d");
        using var g = GgufFile.Open(WriteFile([pkg]));
        var expected = pkg.ToGgufMetadata();
        expected["neuro_symbolic.down_checksum"] = pkg.Down.DataChecksum();
        expected["neuro_symbolic.up_checksum"] = pkg.Up.DataChecksum();
        foreach (var (k, v) in expected) Assert.Equal(v, g.Metadata.GetString(k));
        Assert.Equal("nsadapter", g.Metadata.GetString("general.architecture"));
        foreach (string t in new[] { "down", "up", "down_scale", "up_scale" }) Assert.True(g.TensorsByName.ContainsKey($"nsadapter.0.{t}"), t);
        Assert.Equal(QuantizationType.I32, g.TensorsByName["nsadapter.0.down"].QuantizationType);
        Assert.Equal(QuantizationType.F32, g.TensorsByName["nsadapter.0.up_scale"].QuantizationType);
    }

    [Fact]
    public void MultipleAdapters_KeepOrderLayersAndEachOthersTensors()
    {
        var a = Synth(4, 128, 1, 1f, 0f, 1, "a"); var b = Synth(6, 128, 2, 0.5f, 0.1f, 4, "b"); var c = Synth(2, 128, 3, 2f, 0f, 1, "c");
        var back = NsAdapterGguf.Read(WriteFile([a, b, c]));
        Assert.Equal(3, back.Count);
        AssertSame(a, back[0]); AssertSame(b, back[1]); AssertSame(c, back[2]);
        using var g = GgufFile.Open(P("x.gguf"));
        Assert.Equal("b", g.Metadata.GetString("neuro_symbolic.1.domain"));
        Assert.Equal(3u, g.Metadata.GetUInt32("nsadapter.count"));
    }

    [Fact]
    public void HookLoadedFromGgufAndZip_ProducesBitIdenticalOutput_AndMixesBothFormats()
    {
        var a = Synth(8, 96, 5, 1.5f, 0f, 2, "a"); var b = Synth(8, 96, 6, 1f, 0f, 2, "b");
        a.Save(P("a.nsadapter")); NsAdapterGguf.Write(P("b.gguf"), [b]); NsAdapterGguf.Write(P("both.gguf"), [a, b]);
        var row = new float[96]; for (int i = 0; i < 96; i++) row[i] = MathF.Sin(i * 0.37f) * (i % 11 == 0 ? 6 : 1);
        float[] Run(params string[] paths)
        {
            using var hook = TernaryResidualHook.Load(paths);
            var r = (float[])row.Clone();
            hook.ModifyResidual(r, 2, 1);
            return r;
        }
        float[] zips = Run(Path.Combine(_dir, "a.nsadapter"), SaveZip(b, "b.nsadapter"));
        float[] mixed = Run(P("a.nsadapter"), P("b.gguf"));
        float[] ggufOnly = Run(P("both.gguf"));
        Assert.Equal(zips.Select(BitConverter.SingleToInt32Bits), mixed.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(zips.Select(BitConverter.SingleToInt32Bits), ggufOnly.Select(BitConverter.SingleToInt32Bits));
        Assert.NotEqual(row, zips);   // the adapters actually did something
    }

    private string SaveZip(NsAdapterPackage p, string name) { p.Save(P(name)); return P(name); }

    // ------------------------------------------------------------------------------------------------ rejection ----

    private static byte[] Tensor32(float[] v) { var b = new byte[v.Length * 4]; Buffer.BlockCopy(v, 0, b, 0, b.Length); return b; }

    /// <summary>A hand-built adapter GGUF with a controllable defect, to prove each validation can fire.</summary>
    private byte[] Handmade(NsAdapterPackage pkg, Action<Dictionary<string, object>> tweak)
    {
        // Same layout as NsAdapterGguf.Write, assembled by hand so individual fields can be broken.
        var t = new Dictionary<string, object>
        {
            ["downType"] = (uint)QuantizationType.I32, ["upType"] = (uint)QuantizationType.I32, ["scaleType"] = (uint)QuantizationType.F32,
            ["downData"] = Pad(pkg.Down.Data), ["upData"] = Pad(pkg.Up.Data), ["dsData"] = Tensor32(pkg.Down.Scales), ["usData"] = Tensor32(pkg.Up.Scales),
            ["downElems"] = Pad(pkg.Down.Data).Length / 4, ["dsElems"] = pkg.Down.Scales.Length, ["skipDownScale"] = false,
            ["downChecksum"] = pkg.Down.DataChecksum(), ["upChecksum"] = pkg.Up.DataChecksum(), ["count"] = 1u,
        };
        tweak(t);
        var w = new GgufWriter().AddString("general.architecture", "nsadapter").AddUInt32("nsadapter.count", (uint)(uint)t["count"]);
        foreach (var (k, v) in pkg.ToGgufMetadata())
            w.AddString(k, k.EndsWith("down_checksum", StringComparison.Ordinal) ? (string)t["downChecksum"] : k.EndsWith("up_checksum", StringComparison.Ordinal) ? (string)t["upChecksum"] : v);
        w.AddTensor("nsadapter.0.down", [(int)t["downElems"]], (uint)t["downType"], (byte[])t["downData"]);
        w.AddTensor("nsadapter.0.up", [Pad(pkg.Up.Data).Length / 4], (uint)t["upType"], (byte[])t["upData"]);
        if (!(bool)t["skipDownScale"]) w.AddTensor("nsadapter.0.down_scale", [(int)t["dsElems"]], (uint)t["scaleType"], (byte[])t["dsData"]);
        w.AddTensor("nsadapter.0.up_scale", [pkg.Up.Scales.Length], (uint)t["scaleType"], (byte[])t["usData"]);
        return w.Build();
    }

    private static byte[] Pad(byte[] d) { var r = new byte[(d.Length + 3) & ~3]; d.CopyTo(r, 0); return r; }

    private InvalidDataException Fails(NsAdapterPackage pkg, Action<Dictionary<string, object>> tweak, string name)
    {
        File.WriteAllBytes(P(name), Handmade(pkg, tweak));
        return Assert.Throws<InvalidDataException>(() => NsAdapterGguf.Read(P(name)));
    }

    [Fact]
    public void HandmadeValidFile_Loads_SoTheRejectionTestsBelowAreMeaningful()
    {
        var pkg = Synth(5, 203, 77, 1f, 0f, 1, "ok");
        File.WriteAllBytes(P("ok.gguf"), Handmade(pkg, _ => { }));
        AssertSame(pkg, Assert.Single(NsAdapterGguf.Read(P("ok.gguf"))));
    }

    [Fact]
    public void Read_RejectsCorruptMismatchedAndMalformedFiles()
    {
        var pkg = Synth(5, 203, 77, 1f, 0f, 1, "bad");
        Assert.Contains("checksum", Fails(pkg, t => { var d = (byte[])((byte[])t["downData"]).Clone(); d[3] ^= 0x01; t["downData"] = d; }, "c1.gguf").Message);
        Assert.Contains("checksum", Fails(pkg, t => t["upChecksum"] = new string('0', 64), "c2.gguf").Message);
        Assert.Contains("I32", Fails(pkg, t => t["downType"] = (uint)QuantizationType.F32, "c3.gguf").Message);
        Assert.Contains("elements", Fails(pkg, t => { t["downData"] = new byte[((byte[])t["downData"]).Length + 4]; t["downElems"] = (int)t["downElems"] + 1; }, "c4.gguf").Message);
        Assert.Contains("missing", Fails(pkg, t => t["skipDownScale"] = true, "c5.gguf").Message);
        Assert.Contains("F32", Fails(pkg, t => t["scaleType"] = (uint)QuantizationType.I32, "c6.gguf").Message);
        Assert.Contains("elements", Fails(pkg, t => { t["dsData"] = Tensor32(new float[3]); t["dsElems"] = 3; }, "c7.gguf").Message);
        Assert.Contains("count", Fails(pkg, t => t["count"] = 0u, "c8.gguf").Message);
        // non-zero padding: 5 rows x ceil(203/4)=51 bytes = 255 -> one padding byte
        Assert.Contains("padding", Fails(pkg, t => { var d = (byte[])((byte[])t["downData"]).Clone(); d[^1] = 7; t["downData"] = d; }, "c9.gguf").Message);
        // not a GGUF at all / a model GGUF without adapter tensors
        File.WriteAllText(P("junk.gguf"), "this is not a gguf");
        Assert.Throws<InvalidDataException>(() => NsAdapterGguf.Read(P("junk.gguf")));
        File.WriteAllBytes(P("model.gguf"), SyntheticDenseGguf.Build("llama"));
        Assert.Contains("no adapter", Assert.Throws<InvalidDataException>(() => NsAdapterGguf.Read(P("model.gguf"))).Message);
    }

    [Fact]
    public void Read_RejectsOutOfRangeManifestAndMissingKeys()
    {
        var pkg = Synth(4, 64, 3, 1f, 0f, 1, "k");
        // missing key: drop created_at by writing the metadata by hand
        var w = new GgufWriter().AddString("general.architecture", "nsadapter");
        foreach (var (k, v) in pkg.ToGgufMetadata()) if (!k.EndsWith("created_at", StringComparison.Ordinal)) w.AddString(k, v);
        w.AddTensor("nsadapter.0.down", [Pad(pkg.Down.Data).Length / 4], (uint)QuantizationType.I32, Pad(pkg.Down.Data));
        File.WriteAllBytes(P("m.gguf"), w.Build());
        Assert.Contains("created_at", Assert.Throws<InvalidDataException>(() => NsAdapterGguf.Read(P("m.gguf"))).Message);

        // absurd rank: must be rejected before any tensor is sized from it
        var w2 = new GgufWriter().AddString("general.architecture", "nsadapter");
        foreach (var (k, v) in pkg.ToGgufMetadata()) w2.AddString(k, k.EndsWith(".rank", StringComparison.Ordinal) ? "999999" : v);
        w2.AddTensor("nsadapter.0.down", [1], (uint)QuantizationType.I32, new byte[4]);
        File.WriteAllBytes(P("r.gguf"), w2.Build());
        Assert.Contains("out of range", Assert.Throws<InvalidDataException>(() => NsAdapterGguf.Read(P("r.gguf"))).Message);
    }

    // ------------------------------------------------------------------------------------------------ server wiring ----

    private sealed class Tiny : IDisposable
    {
        public readonly GgufFile Gguf; public readonly ModelConfig Config; public readonly TransformerModel Model;
        public Tiny(string dir)
        {
            string path = Path.Combine(dir, "tiny.gguf");
            File.WriteAllBytes(path, SyntheticDenseGguf.Build("llama"));
            Gguf = GgufFile.Open(path);
            Config = GgufModelConfigExtractor.Extract(Gguf.Metadata);
            Model = TransformerModel.LoadFromGguf(Gguf, Config, ThreadingConfig.SingleThreaded);
        }
        public float[] Logits(int[] ids)
        {
            using ITensor t = Model.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1);
            return new ReadOnlySpan<float>((float*)t.DataPointer, (int)t.Shape.ElementCount).ToArray();
        }
        public void Dispose() { Model.Dispose(); Gguf.Dispose(); }
    }

    [Fact]
    public void ServerWiring_AttachesTheGgufAdapter_AndItChangesLogitsLikeTheDirectHook()
    {
        using var tiny = new Tiny(_dir);
        int d = tiny.Config.HiddenSize;
        var pkg = Synth(8, d, 21, 3f, 0f, 1, "srv");
        string path = WriteFile([pkg], "srv.gguf");
        int[] ids = [3, 7, 11, 2, 40, 5];
        float[] baseline = tiny.Logits(ids);

        ServerStartup.AttachTernaryAdapters(tiny.Model, tiny.Config, tiny.Gguf, new ServerOptions { Model = "tiny", TernaryAdapterPaths = [path] });
        Assert.IsType<TernaryResidualHook>(tiny.Model.ResidualModifier);
        float[] served = tiny.Logits(ids);
        Assert.NotEqual(baseline, served);

        var hook = (TernaryResidualHook)tiny.Model.ResidualModifier!;
        tiny.Model.ResidualModifier = null;
        using var direct = new TernaryResidualHook(d);
        direct.Add(pkg);
        tiny.Model.ResidualModifier = direct;
        Assert.Equal(served.Select(BitConverter.SingleToInt32Bits), tiny.Logits(ids).Select(BitConverter.SingleToInt32Bits));
        tiny.Model.ResidualModifier = null;
        hook.Dispose();
    }

    [Fact]
    public void ServerWiring_NoPaths_IsANoOp_AndAMismatchedAdapterFailsTheLoad()
    {
        using (var tiny = new Tiny(_dir))
        {
            ServerStartup.AttachTernaryAdapters(tiny.Model, tiny.Config, tiny.Gguf, new ServerOptions { Model = "tiny" });
            Assert.Null(tiny.Model.ResidualModifier);
        }
        var tiny2 = new Tiny(_dir);
        string bad = WriteFile([Synth(4, tiny2.Config.HiddenSize + 8, 5, 1f, 0f, 0, "wrongD")], "bad.gguf");
        var ex = Assert.Throws<InvalidServerOptionException>(() =>
            ServerStartup.AttachTernaryAdapters(tiny2.Model, tiny2.Config, tiny2.Gguf, new ServerOptions { Model = "tiny", TernaryAdapterPaths = [bad] }));
        Assert.Contains("--ternary-adapter", ex.Message);   // the helper disposed model and GGUF on failure, as LoadModelCore expects
    }
}
