using System.Text;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Byte-level GGUF splitter (the moral equivalent of llama.cpp's <c>llama-gguf-split --split</c>), used to turn a
/// real single-file fixture into a <c>name-0000N-of-0000M.gguf</c> set for the split-load end-to-end tests (#756).
/// Tensor bytes and the original metadata KV region are copied verbatim, so any difference between the merged
/// and the split model is attributable to the loader, never to the splitter. Shard 1 carries the metadata plus
/// <c>split.*</c> keys; shards 2..N carry only <c>split.*</c> keys and their own tensors with shard-local offsets.
/// </summary>
public static class GgufSplitter
{
    private sealed record TensorInfo(string Name, ulong[] Dims, uint Type, ulong Offset, long Size);

    /// <summary>
    /// Splits <paramref name="source"/> into <paramref name="shardCount"/> shards in <paramref name="outDir"/> and returns the
    /// path of shard 1. Tensors are partitioned contiguously (file order) into roughly equal byte ranges.
    /// </summary>
    /// <param name="metadataOnlyFirst">When true shard 1 holds no tensors (like the unsloth sets); otherwise it takes its share (like llama-gguf-split).</param>
    public static string Split(string source, string outDir, string stem, int shardCount, bool metadataOnlyFirst)
    {
        Directory.CreateDirectory(outDir);
        using var fs = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);

        if (br.ReadUInt32() != 0x46554747u) throw new InvalidDataException("not a GGUF file");
        uint version = br.ReadUInt32();
        ulong tensorCount = br.ReadUInt64();
        ulong kvCount = br.ReadUInt64();
        long kvStart = fs.Position;
        uint alignment = 32;
        for (ulong i = 0; i < kvCount; i++)
        {
            string key = ReadString(br);
            uint type = br.ReadUInt32();
            if (key == "general.alignment" && type == 4) { long p = fs.Position; alignment = br.ReadUInt32(); fs.Position = p; }
            SkipValue(br, type);
        }
        long kvEnd = fs.Position;

        var infos = new List<TensorInfo>((int)tensorCount);
        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = ReadString(br);
            uint nd = br.ReadUInt32();
            var dims = new ulong[nd];
            for (int d = 0; d < nd; d++) dims[d] = br.ReadUInt64();
            uint type = br.ReadUInt32();
            ulong off = br.ReadUInt64();
            infos.Add(new TensorInfo(name, dims, type, off, 0));
        }
        long dataStart = AlignUp(fs.Position, alignment);
        long dataLen = fs.Length - dataStart;

        // Tensor byte extents: distance to the next tensor's offset (offsets are ascending in practice).
        var sorted = infos.Select((t, idx) => (t, idx)).OrderBy(x => x.t.Offset).ToList();
        var sizes = new long[infos.Count];
        for (int k = 0; k < sorted.Count; k++)
        {
            long end = k + 1 < sorted.Count ? (long)sorted[k + 1].t.Offset : dataLen;
            sizes[sorted[k].idx] = end - (long)sorted[k].t.Offset;
        }
        for (int i = 0; i < infos.Count; i++) infos[i] = infos[i] with { Size = sizes[i] };

        // Partition in file order into shard groups of roughly equal bytes.
        int dataShards = metadataOnlyFirst ? shardCount - 1 : shardCount;
        long total = infos.Sum(t => t.Size);
        var groups = Enumerable.Range(0, shardCount).Select(_ => new List<TensorInfo>()).ToList();
        int gi = metadataOnlyFirst ? 1 : 0;
        long acc = 0;
        foreach (var t in infos)
        {
            int rel = gi - (metadataOnlyFirst ? 1 : 0);
            if (rel < dataShards - 1 && acc >= total * (rel + 1) / dataShards) gi++;
            groups[gi].Add(t);
            acc += t.Size;
        }

        byte[] kvRaw = new byte[kvEnd - kvStart];
        fs.Position = kvStart;
        fs.ReadExactly(kvRaw);

        string firstPath = ShardPath(outDir, stem, 1, shardCount);
        for (int s = 0; s < shardCount; s++)
        {
            using var o = new FileStream(ShardPath(outDir, stem, s + 1, shardCount), FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(o, Encoding.UTF8, leaveOpen: true);
            bw.Write(0x46554747u);
            bw.Write(version);
            bw.Write((ulong)groups[s].Count);
            bw.Write(s == 0 ? kvCount + 3 : 3UL);
            if (s == 0) bw.Write(kvRaw);
            WriteKv16(bw, "split.no", (ushort)s);
            WriteKv16(bw, "split.count", (ushort)shardCount);
            bw.Write(WriteString("split.tensors.count")); bw.Write(5u); bw.Write((int)infos.Count);

            ulong cursor = 0;
            var local = new List<(TensorInfo T, ulong Off)>();
            foreach (var t in groups[s])
            {
                local.Add((t, cursor));
                cursor = (ulong)AlignUp((long)cursor + t.Size, alignment);
            }
            foreach (var (t, off) in local)
            {
                bw.Write(WriteString(t.Name));
                bw.Write((uint)t.Dims.Length);
                foreach (ulong d in t.Dims) bw.Write(d);
                bw.Write(t.Type);
                bw.Write(off);
            }
            bw.Flush();
            PadTo(o, alignment);
            long shardDataStart = o.Position;
            foreach (var (t, off) in local)
            {
                PadTo(o, 1, shardDataStart + (long)off);
                var buf = new byte[t.Size];
                fs.Position = dataStart + (long)t.Offset;
                fs.ReadExactly(buf);
                o.Write(buf);
            }
            PadTo(o, 1, shardDataStart + (long)cursor);
        }
        return firstPath;
    }

    /// <summary>Path of shard <paramref name="no"/> (1-based) following the llama.cpp naming convention.</summary>
    public static string ShardPath(string dir, string stem, int no, int count) =>
        Path.Combine(dir, $"{stem}-{no:D5}-of-{count:D5}.gguf");

    private static long AlignUp(long v, uint a) => (v + a - 1) / a * a;

    private static void PadTo(FileStream o, uint alignment) => PadTo(o, 1, AlignUp(o.Position, alignment));

    private static void PadTo(FileStream o, int _, long position)
    {
        long pad = position - o.Position;
        if (pad < 0) throw new InvalidOperationException("overlapping tensor ranges");
        if (pad > 0) o.Write(new byte[pad]);
    }

    private static string ReadString(BinaryReader br)
    {
        ulong len = br.ReadUInt64();
        return Encoding.UTF8.GetString(br.ReadBytes((int)len));
    }

    private static byte[] WriteString(string s)
    {
        byte[] b = Encoding.UTF8.GetBytes(s);
        var r = new byte[8 + b.Length];
        BitConverter.TryWriteBytes(r.AsSpan(0, 8), (ulong)b.Length);
        b.CopyTo(r, 8);
        return r;
    }

    private static void WriteKv16(BinaryWriter bw, string key, ushort value)
    {
        bw.Write(WriteString(key)); bw.Write(2u); bw.Write(value);
    }

    private static void SkipValue(BinaryReader br, uint type)
    {
        switch (type)
        {
            case 0: case 1: case 7: br.BaseStream.Position += 1; break;
            case 2: case 3: br.BaseStream.Position += 2; break;
            case 4: case 5: case 6: br.BaseStream.Position += 4; break;
            case 10: case 11: case 12: br.BaseStream.Position += 8; break;
            case 8: { ulong n = br.ReadUInt64(); br.BaseStream.Position += (long)n; break; }
            case 9:
            {
                uint et = br.ReadUInt32();
                ulong n = br.ReadUInt64();
                for (ulong i = 0; i < n; i++) SkipValue(br, et);
                break;
            }
            default: throw new InvalidDataException($"unknown GGUF value type {type}");
        }
    }
}
