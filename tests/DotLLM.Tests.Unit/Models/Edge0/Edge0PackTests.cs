using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Edge0;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>Edge0 pack reader (#1967, G1): geometry derivation, in-place pointers, and refusal of inconsistent folders.</summary>
public sealed unsafe class Edge0PackTests
{
    [Fact]
    public void Open_DerivesGeometry_AndMapsRecordsInPlace()
    {
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        Assert.Equal(SyntheticEdge0Pack.Layers, pack.Layers);
        Assert.Equal(SyntheticEdge0Pack.Experts, pack.Experts);
        Assert.Equal(SyntheticEdge0Pack.Latent, pack.Latent);
        Assert.Equal(SyntheticEdge0Pack.Groups, pack.Groups);
        Assert.True(pack.HasColumnScale);
        Assert.Equal(QuantizationType.Q8_0, pack.LatentDenseType);

        // record = 2 x (64 rows x 110 B) + 256 rows x 2 blocks x 18 B
        Assert.Equal(2L * 64 * 110 + 256L * 2 * 18, pack.ExpertRecordBytes);
        Assert.Equal(64L * 110, pack.GateUpBytes);

        // The pointers address the file's own bytes (spot-check the first byte of several regions against the file).
        byte[] file = File.ReadAllBytes(Path.Combine(syn.Dir, "experts.bin"));
        foreach (var (l, e) in new[] { (0, 0), (0, 15), (1, 3), (1, 15) })
        {
            long off = ((long)l * SyntheticEdge0Pack.Experts + e) * pack.ExpertRecordBytes;
            Assert.Equal(off, (long)(pack.ExpertRecord(l, e) - pack.ExpertRecord(0, 0)));
            Assert.True(new ReadOnlySpan<byte>((void*)pack.ExpertRecord(l, e), (int)pack.ExpertRecordBytes)
                .SequenceEqual(file.AsSpan((int)off, (int)pack.ExpertRecordBytes)), $"record ({l},{e}) differs from the file");
        }
        // gate | up | down bank bases are the record offsets 0, GateUpBytes, 2*GateUpBytes
        Assert.Equal(pack.GateUpBytes, pack.UpBank(1) - pack.GateBank(1));
        Assert.Equal(pack.GateUpBytes, pack.DownBank(1) - pack.UpBank(1));
    }

    [Fact]
    public void Labels_And_InverseColumnScale_MatchTheFiles()
    {
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        for (int l = 0; l < pack.Layers; l++)
        {
            Assert.True(pack.Labels(l).SequenceEqual(syn.Labels.AsSpan(l * pack.Experts, pack.Experts)));
            var inv = pack.InverseColumnScale(l);
            Assert.Equal(pack.Latent, inv.Length);
            for (int i = 0; i < inv.Length; i++) Assert.Equal(1f / syn.Scales[l][i], inv[i]);
        }
    }

    [Fact]
    public void NonPositiveScale_MapsToIdentity()
    {
        using var syn = new SyntheticEdge0Pack();
        var s = new float[SyntheticEdge0Pack.Latent];
        Array.Fill(s, 2f); s[5] = 0f; s[9] = -3f;
        var b = new byte[s.Length * 4];
        Buffer.BlockCopy(s, 0, b, 0, b.Length);
        File.WriteAllBytes(Path.Combine(syn.Dir, "scales", "l0.f32"), b);
        using var pack = syn.Open();
        Assert.Equal(0.5f, pack.InverseColumnScale(0)[0]);
        Assert.Equal(1f, pack.InverseColumnScale(0)[5]);   // strata: v > 0 ? 1/v : 1
        Assert.Equal(1f, pack.InverseColumnScale(0)[9]);
    }

    [Fact]
    public void DUPointers_AddressGroupBlocks()
    {
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        byte[] file = File.ReadAllBytes(Path.Combine(syn.Dir, "latent_dense_q8.bin"));
        long dRow = 64 / 32 * 34, uRow = 256 / 32 * 34;
        long layerBytes = (long)3 * 256 * dRow + (long)3 * 64 * uRow;
        Assert.Equal(layerBytes, pack.DLayerBytes + pack.ULayerBytes);
        foreach (var (l, g) in new[] { (0, 0), (0, 2), (1, 1) })
        {
            long dOff = l * layerBytes + (long)g * 256 * dRow, uOff = l * layerBytes + 3L * 256 * dRow + (long)g * 64 * uRow;
            Assert.True(new ReadOnlySpan<byte>((void*)pack.D(l, g), 256 * (int)dRow).SequenceEqual(file.AsSpan((int)dOff, 256 * (int)dRow)));
            Assert.True(new ReadOnlySpan<byte>((void*)pack.U(l, g), 64 * (int)uRow).SequenceEqual(file.AsSpan((int)uOff, 64 * (int)uRow)));
        }
    }

    [Fact]
    public void Q4Latent_IsRecognisedBySize()
    {
        using var syn = new SyntheticEdge0Pack(q4Latent: true);
        using var pack = syn.Open();
        Assert.Equal(QuantizationType.Q4_0, pack.LatentDenseType);
        Assert.Equal(SyntheticEdge0Pack.Groups, pack.Groups);   // derived from the file size, not assumed
    }

    [Fact]
    public void NoScalesFolder_DerivesLatentFromRecordSize()
    {
        using var syn = new SyntheticEdge0Pack(withScales: false);
        using var pack = syn.Open();
        Assert.False(pack.HasColumnScale);
        Assert.Equal(SyntheticEdge0Pack.Latent, pack.Latent);
        Assert.True(pack.InverseColumnScale(0).IsEmpty);
    }

    [Fact]
    public void Open_RefusesTruncatedExperts()
    {
        using var syn = new SyntheticEdge0Pack();
        string p = Path.Combine(syn.Dir, "experts.bin");
        using (var fs = new FileStream(p, FileMode.Open)) fs.SetLength(fs.Length - 1);
        Assert.Throws<InvalidDataException>(() => syn.Open());
    }

    [Fact]
    public void Open_RefusesOutOfRangeLabel()
    {
        using var syn = new SyntheticEdge0Pack();
        string p = Path.Combine(syn.Dir, "labels.bin");
        var b = File.ReadAllBytes(p);
        b[0] = 3;   // group 3 of 3 (valid 0..2)
        File.WriteAllBytes(p, b);
        Assert.Throws<InvalidDataException>(() => syn.Open());
    }

    [Fact]
    public void Open_RefusesNvfp4Manifest()
    {
        using var syn = new SyntheticEdge0Pack();
        string p = Path.Combine(syn.Dir, "native_experts.txt");
        File.WriteAllText(p, File.ReadAllText(p).Replace(" 11 20 ", " 40 40 "));
        Assert.Throws<NotSupportedException>(() => syn.Open());
    }

    [Fact]
    public void Open_RefusesShortScaleVector()
    {
        using var syn = new SyntheticEdge0Pack();
        File.WriteAllBytes(Path.Combine(syn.Dir, "scales", "l1.f32"), new byte[16]);
        Assert.Throws<InvalidDataException>(() => syn.Open());
    }
}
