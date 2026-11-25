using System.Buffers.Binary;
using System.Globalization;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Edge0;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>
/// Writes a tiny, structurally faithful Edge0 Q3_K pack folder (same files, same record layout, small geometry) for unit tests.
/// Geometry: latent 256 (one Q3_K block per gate/up row), expert FFN 64, hidden 64.
/// </summary>
internal sealed class SyntheticEdge0Pack : IDisposable
{
    public const int Layers = 2, Experts = 16, Hidden = 64, Latent = 256, Ff = 64, Groups = 3, TopK = 4;

    public string Dir { get; }
    public int[] Labels { get; }
    public float[][] Scales { get; }
    public bool Q4Latent { get; }

    public SyntheticEdge0Pack(int seed = 7, bool withScales = true, bool q4Latent = false, bool skewedLabels = true)
    {
        Q4Latent = q4Latent;
        Dir = Path.Combine(Path.GetTempPath(), "dotllm-edge0-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Dir);
        var rng = new Random(seed);

        long guRow = Dequantize.RowByteSize(Latent, QuantizationType.Q3_K), dnRow = Dequantize.RowByteSize(Ff, QuantizationType.IQ4_NL);
        long record = 2 * guRow * Ff + dnRow * Latent;

        // native_experts.txt
        var manifest = new List<string> { "# strata native experts v6: latent A-only, Q3_K gu + shared colscale (synthetic)" };
        for (int l = 0; l < Layers; l++)
            manifest.Add(string.Create(CultureInfo.InvariantCulture, $"{l} 11 20 {(long)l * Experts * record} {record}"));
        File.WriteAllLines(Path.Combine(Dir, "native_experts.txt"), manifest);

        // experts.bin: gate | up | down per record, valid fp16 scales, random payload
        using (var f = File.Create(Path.Combine(Dir, "experts.bin")))
        {
            for (int l = 0; l < Layers; l++)
                for (int e = 0; e < Experts; e++)
                {
                    var rec = new byte[record];
                    long o = 0;
                    for (int part = 0; part < 2; part++)       // gate, up: Ff rows of Q3_K
                        for (int r = 0; r < Ff; r++, o += guRow) FillQ3K(rec, o, Latent / 256, rng);
                    for (int r = 0; r < Latent; r++, o += dnRow) FillIq4Nl(rec, o, Ff / 32, rng);
                    f.Write(rec);
                }
        }

        // labels.bin: int32 [layers][experts]; skewed so one group is empty-ish and the groups are uneven
        Labels = new int[Layers * Experts];
        for (int i = 0; i < Labels.Length; i++)
            Labels[i] = skewedLabels ? (rng.Next(10) < 6 ? 0 : 1 + rng.Next(Groups - 1)) : i % Groups;
        var lb = new byte[Labels.Length * 4];
        for (int i = 0; i < Labels.Length; i++) BinaryPrimitives.WriteInt32LittleEndian(lb.AsSpan(i * 4), Labels[i]);
        File.WriteAllBytes(Path.Combine(Dir, "labels.bin"), lb);

        // scales/lN.f32: s in [0.4, 2.5] (inverse differs materially from s and from 1)
        Scales = new float[Layers][];
        if (withScales)
        {
            Directory.CreateDirectory(Path.Combine(Dir, "scales"));
            for (int l = 0; l < Layers; l++)
            {
                Scales[l] = new float[Latent];
                for (int i = 0; i < Latent; i++) Scales[l][i] = 0.4f + 2.1f * (float)rng.NextDouble();
                var sb = new byte[Latent * 4];
                Buffer.BlockCopy(Scales[l], 0, sb, 0, sb.Length);
                File.WriteAllBytes(Path.Combine(Dir, "scales", $"l{l}.f32"), sb);
            }
        }

        // latent_dense_q8.bin / q4.bin: per layer, D = Groups*Latent rows of Hidden, then U = Groups*Hidden rows of Latent
        var lt = q4Latent ? QuantizationType.Q4_0 : QuantizationType.Q8_0;
        long dRow = Dequantize.RowByteSize(Hidden, lt), uRow = Dequantize.RowByteSize(Latent, lt);
        using (var f = File.Create(Path.Combine(Dir, q4Latent ? "latent_dense_q4.bin" : "latent_dense_q8.bin")))
        {
            for (int l = 0; l < Layers; l++)
            {
                var d = new byte[Groups * Latent * dRow];
                for (long r = 0; r < (long)Groups * Latent; r++) FillLegacy(d, r * dRow, Hidden / 32, q4Latent, rng);
                f.Write(d);
                var u = new byte[Groups * Hidden * uRow];
                for (long r = 0; r < (long)Groups * Hidden; r++) FillLegacy(u, r * uRow, Latent / 32, q4Latent, rng);
                f.Write(u);
            }
        }
    }

    private static void PutHalf(byte[] b, long at, float v) => BinaryPrimitives.WriteHalfLittleEndian(b.AsSpan((int)at), (Half)v);

    private static void FillQ3K(byte[] b, long at, int blocks, Random rng)
    {
        for (int i = 0; i < blocks; i++)
        {
            long o = at + i * 110L;
            rng.NextBytes(b.AsSpan((int)o, 108));
            PutHalf(b, o + 108, 0.004f + 0.004f * (float)rng.NextDouble());   // d at the end of block_q3_K
        }
    }

    private static void FillIq4Nl(byte[] b, long at, int blocks, Random rng)
    {
        for (int i = 0; i < blocks; i++)
        {
            long o = at + i * 18L;
            PutHalf(b, o, 0.01f + 0.01f * (float)rng.NextDouble());           // d first
            rng.NextBytes(b.AsSpan((int)o + 2, 16));
        }
    }

    private static void FillLegacy(byte[] b, long at, int blocks, bool q4, Random rng)
    {
        int bytes = q4 ? 18 : 34;
        for (int i = 0; i < blocks; i++)
        {
            long o = at + i * (long)bytes;
            PutHalf(b, o, 0.01f + 0.02f * (float)rng.NextDouble());
            if (q4) rng.NextBytes(b.AsSpan((int)o + 2, 16));
            else for (int j = 0; j < 32; j++) b[o + 2 + j] = (byte)(sbyte)rng.Next(-40, 41);
        }
    }

    public Edge0Pack Open() => Edge0Pack.Open(Dir, Hidden, Ff, preferLatentQ4: Q4Latent);

    public void Dispose()
    {
        try { Directory.Delete(Dir, recursive: true); } catch (IOException) { /* a still-mapped file on Windows; temp dir, best effort */ }
    }
}
