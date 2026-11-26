using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Edge0;

/// <summary>
/// Reader for the Edge0 MTP draft head (G6, #1990): the <c>mtp/</c> folder of an Edge0 model (<c>dense.txt</c> + <c>dense.bin</c>, <c>experts.bin</c>, <c>draft_vocab.bin</c>),
/// written by edge0-strata <c>tools/mtp_rt.py</c>. Everything is mapped in place; nothing is copied or converted except the two 7 MB halves of <c>eh_proj</c>, which the head
/// loaders expect as one matrix.
/// </summary>
/// <remarks>
/// <para><b>dense.txt</b> rows are <c>name kind rows cols offset bytes</c> (<c>kind</c> = q8_0 | bf16 | f32; a GGUF tensor is <c>[cols, rows]</c>). Norm weights are F32 with
/// the Gemma "+1" already applied (the same folded convention the qwen4exp GGUF uses); the large projections are Q8_0, the router and hyper-connection mixers BF16.</para>
/// <para><b>experts.bin</b> is 512 blobs of <see cref="ExpertBlobBytes"/> bytes in Strata's PLANAR Q2_0 layout (a lossless relayout of ggml Q2_0 blocks, 18 B per 64 elements =
/// fp16 d + 16 code bytes): per expert <c>[gate/up codes | down codes | gate/up scales | down scales]</c>, the gate/up rows interleaved (row <c>2r</c> = gate row <c>r</c>,
/// <c>2r+1</c> = up row <c>r</c>). The Vulkan path reads that layout in place through the planar kernel; <see cref="RelayoutToGgmlBanks"/> builds ordinary ggml banks (the CPU oracle).</para>
/// <para><b>draft_vocab.bin</b> is a raw little-endian int32 array: the token ids of the draft head's vocabulary subset (the draft logits are computed over these rows of the
/// output matrix only). Since the verify window decides every emitted token this only affects draft quality.</para>
/// </remarks>
public sealed unsafe class Edge0MtpPack : IDisposable
{
    /// <summary>Hidden width of the head (2560).</summary>
    public const int Hidden = 2560;

    /// <summary>Routed-expert FFN width (640).</summary>
    public const int ExpertFf = 640;

    /// <summary>Routed experts (512).</summary>
    public const int Experts = 512;

    /// <summary>Elements per Q2_0 block.</summary>
    public const int Q2Block = 64;

    /// <summary>Bytes of one Q2_0 block (fp16 d + 16 code bytes).</summary>
    public const int Q2BlockBytes = 18;

    /// <summary>Gate/up rows of one expert (gate and up interleaved).</summary>
    public const int GuRows = 2 * ExpertFf;

    /// <summary>Q2_0 blocks per gate/up row (K = 2560).</summary>
    public const int GuBlocksPerRow = Hidden / Q2Block;

    /// <summary>Q2_0 blocks per down row (K = 640).</summary>
    public const int DownBlocksPerRow = ExpertFf / Q2Block;

    /// <summary>Byte offset of the gate/up code plane inside an expert blob.</summary>
    public const long GuCodeOffset = 0;

    /// <summary>Byte offset of the down code plane.</summary>
    public const long DownCodeOffset = (long)GuRows * GuBlocksPerRow * 16;

    /// <summary>Byte offset of the gate/up scale (fp16) plane.</summary>
    public const long GuScaleOffset = DownCodeOffset + (long)Hidden * DownBlocksPerRow * 16;

    /// <summary>Byte offset of the down scale plane.</summary>
    public const long DownScaleOffset = GuScaleOffset + (long)GuRows * GuBlocksPerRow * 2;

    /// <summary>Bytes of one expert blob (3 * 640 * 2560 / 64 * 18 = 1,382,400).</summary>
    public const long ExpertBlobBytes = DownScaleOffset + (long)Hidden * DownBlocksPerRow * 2;

    /// <summary>One <c>dense.txt</c> entry resolved to an address.</summary>
    public readonly record struct DenseTensor(string Name, string Kind, int Rows, int Cols, nint Pointer, long Bytes);

    private readonly List<(MemoryMappedFile Mmf, MemoryMappedViewAccessor View)> _maps = [];
    private readonly Dictionary<string, DenseTensor> _dense = new(StringComparer.Ordinal);
    private readonly List<nint> _owned = [];
    private bool _disposed;

    /// <summary>The <c>mtp</c> folder.</summary>
    public string Directory { get; }

    /// <summary>First byte of <c>experts.bin</c> (512 blobs of <see cref="ExpertBlobBytes"/>).</summary>
    public nint ExpertsPointer { get; }

    /// <summary>True when <c>experts.bin</c> is a shared writable view a UMA driver can import in place.</summary>
    public bool ImportableViews { get; }

    /// <summary>Token ids of the draft vocabulary, in file order: unique but NOT sorted (edge0-strata draft_vocab.py keeps the base subset's order and appends added ids).</summary>
    public int[] DraftVocab { get; }

    /// <summary>The dense tensors by their Strata name (<c>self_attn.q_proj.weight</c>, ...).</summary>
    public IReadOnlyDictionary<string, DenseTensor> Dense => _dense;

    private Edge0MtpPack(string dir, nint experts, bool importable, int[] vocab) { Directory = dir; ExpertsPointer = experts; ImportableViews = importable; DraftVocab = vocab; }

    /// <summary>The <c>mtp</c> folder beside a model folder (or <paramref name="modelDir"/> itself when it is the folder), or null when there is none.</summary>
    public static string? Find(string modelDir)
    {
        string d = Path.Combine(modelDir, "mtp");
        return File.Exists(Path.Combine(d, "dense.txt")) && File.Exists(Path.Combine(d, "experts.bin")) ? d
            : File.Exists(Path.Combine(modelDir, "dense.txt")) && File.Exists(Path.Combine(modelDir, "experts.bin")) && File.Exists(Path.Combine(modelDir, "draft_vocab.bin")) ? modelDir : null;
    }

    /// <summary>Opens and validates the head folder (index offsets and sizes against <c>dense.bin</c>, <c>experts.bin</c> against 512 blobs, the vocabulary against the vocabulary size).</summary>
    /// <param name="mtpDir">The <c>mtp</c> folder.</param>
    /// <param name="vocabSize">The model's vocabulary size (every draft id must be below it).</param>
    /// <param name="importableViews">Map <c>experts.bin</c> as a shared writable view so a UMA GPU driver can import it (see <see cref="Edge0Pack.Open"/>).</param>
    public static Edge0MtpPack Open(string mtpDir, int vocabSize, bool importableViews = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(mtpDir);
        string idx = Path.Combine(mtpDir, "dense.txt"), denseBin = Path.Combine(mtpDir, "dense.bin"), expBin = Path.Combine(mtpDir, "experts.bin"), vocabBin = Path.Combine(mtpDir, "draft_vocab.bin");
        foreach (string f in new[] { idx, denseBin, expBin }) if (!File.Exists(f)) throw new InvalidDataException($"Edge0 MTP file missing: {f}");
        long expLen = LengthOf(expBin);
        if (expLen != Experts * ExpertBlobBytes) throw new InvalidDataException($"experts.bin is {expLen} B, expected {Experts} x {ExpertBlobBytes} = {Experts * ExpertBlobBytes} B (Q2_0 gate/up/down blobs).");
        int[] vocab = [];
        if (File.Exists(vocabBin))
        {
            byte[] raw = File.ReadAllBytes(vocabBin);
            if (raw.Length % 4 != 0) throw new InvalidDataException($"draft_vocab.bin is {raw.Length} B, not a whole number of int32 ids.");
            vocab = new int[raw.Length / 4];
            Buffer.BlockCopy(raw, 0, vocab, 0, raw.Length);
            foreach (int t in vocab) if ((uint)t >= (uint)vocabSize) throw new InvalidDataException($"draft_vocab.bin names token {t} outside the vocabulary (0..{vocabSize - 1}).");
        }

        var maps = new List<(MemoryMappedFile, MemoryMappedViewAccessor)>();
        try
        {
            bool writable = importableViews;
            nint expPtr = Map(expBin, maps, ref writable);
            bool unused = false;
            nint densePtr = Map(denseBin, maps, ref unused);
            long denseLen = LengthOf(denseBin);
            var pack = new Edge0MtpPack(mtpDir, expPtr, writable, vocab);
            pack._maps.AddRange(maps);
            foreach (string line in File.ReadAllLines(idx))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] f = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (f.Length != 6) throw new InvalidDataException($"dense.txt: malformed line '{line}'.");
                int rows = int.Parse(f[2], CultureInfo.InvariantCulture), cols = int.Parse(f[3], CultureInfo.InvariantCulture);
                long off = long.Parse(f[4], CultureInfo.InvariantCulture), bytes = long.Parse(f[5], CultureInfo.InvariantCulture);
                long want = f[1] switch
                {
                    "q8_0" => (long)rows * (cols / 32) * 34,
                    "bf16" => (long)rows * cols * 2,
                    "f32" => (long)rows * cols * 4,
                    _ => throw new InvalidDataException($"dense.txt: unknown kind '{f[1]}' for {f[0]}."),
                };
                if (want != bytes || off < 0 || off + bytes > denseLen) throw new InvalidDataException($"dense.txt: {f[0]} ({f[1]} {rows}x{cols}) has {bytes} B at {off}; expected {want} B inside the {denseLen} B file.");
                pack._dense[f[0]] = new DenseTensor(f[0], f[1], rows, cols, densePtr + (nint)off, bytes);
            }
            return pack;
        }
        catch
        {
            foreach (var (m, v) in maps) { v.SafeMemoryMappedViewHandle.ReleasePointer(); v.Dispose(); m.Dispose(); }
            throw;
        }
    }

    private static long LengthOf(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        return fs.Length;
    }

    private static nint Map(string path, List<(MemoryMappedFile, MemoryMappedViewAccessor)> maps, ref bool writable)
    {
        MemoryMappedFile? mmf = null;
        if (writable)
        {
            try
            {
                var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite);
                mmf = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, leaveOpen: false);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { writable = false; }
        }
        mmf ??= MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var view = mmf.CreateViewAccessor(0, 0, writable ? MemoryMappedFileAccess.ReadWrite : MemoryMappedFileAccess.Read);
        byte* p = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        maps.Add((mmf, view));
        return (nint)p + (nint)view.PointerOffset;
    }

    /// <summary>Pointer to expert <paramref name="expert"/>'s blob.</summary>
    public nint ExpertBlob(int expert)
    {
        if ((uint)expert >= (uint)Experts) throw new ArgumentOutOfRangeException(nameof(expert));
        return ExpertsPointer + (nint)(expert * ExpertBlobBytes);
    }

    /// <summary>
    /// Builds ordinary contiguous ggml Q2_0 banks (<c>[experts][rows][blocks x (fp16 d | 16 codes)]</c>) from the planar blobs: gate (640 rows of 2560), up, down (2560 rows of 640).
    /// A host-memory copy of all 708 MB, for the CPU oracle and tests; the Vulkan path reads the planar blobs in place. The three allocations are owned by this pack.
    /// </summary>
    public (nint Gate, nint Up, nint Down) RelayoutToGgmlBanks()
    {
        long gateBytes = (long)Experts * ExpertFf * GuBlocksPerRow * Q2BlockBytes, downBytes = (long)Experts * Hidden * DownBlocksPerRow * Q2BlockBytes;
        nint gate = Alloc(gateBytes), up = Alloc(gateBytes), down = Alloc(downBytes);
        Parallel.For(0, Experts, e =>
        {
            byte* blob = (byte*)ExpertBlob(e);
            for (int r = 0; r < ExpertFf; r++)
            {
                CopyRow(blob, 2 * r, GuBlocksPerRow, GuCodeOffset, GuScaleOffset, (byte*)gate + ((long)e * ExpertFf + r) * GuBlocksPerRow * Q2BlockBytes);
                CopyRow(blob, 2 * r + 1, GuBlocksPerRow, GuCodeOffset, GuScaleOffset, (byte*)up + ((long)e * ExpertFf + r) * GuBlocksPerRow * Q2BlockBytes);
            }
            for (int r = 0; r < Hidden; r++)
                CopyRow(blob, r, DownBlocksPerRow, DownCodeOffset, DownScaleOffset, (byte*)down + ((long)e * Hidden + r) * DownBlocksPerRow * Q2BlockBytes);
        });
        return (gate, up, down);
    }

    private static void CopyRow(byte* blob, int row, int blocksPerRow, long codeOff, long scaleOff, byte* dst)
    {
        for (int b = 0; b < blocksPerRow; b++)
        {
            long blockIdx = (long)row * blocksPerRow + b;
            dst[b * Q2BlockBytes] = blob[scaleOff + blockIdx * 2];
            dst[b * Q2BlockBytes + 1] = blob[scaleOff + blockIdx * 2 + 1];
            Buffer.MemoryCopy(blob + codeOff + blockIdx * 16, dst + b * Q2BlockBytes + 2, 16, 16);
        }
    }

    private nint Alloc(long bytes)
    {
        nint p = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        _owned.Add(p);
        return p;
    }

    /// <summary>
    /// A GGUF-shaped view of the head under the qwen4exp tensor names (<c>blk.{n}.*</c>, <c>blk.{n}.nextn.*</c>) so the existing head loaders read it unchanged. Dense tensors point into
    /// the mapped <c>dense.bin</c>; <c>eh_proj</c> is the two Q8_0 halves concatenated per output row (a 14 MB copy, owned by this pack). With <paramref name="ggmlExpertBanks"/> the three
    /// routed-expert tensors are included, relaid out as ordinary Q2_0 banks (see <see cref="RelayoutToGgmlBanks"/>); without them the view is the head minus its experts (the
    /// Vulkan loader reads <see cref="ExpertsPointer"/> directly). The view does not own the pack.
    /// </summary>
    /// <param name="config">The trunk configuration (layer count, hidden width, head geometry).</param>
    /// <param name="ggmlExpertBanks">Include contiguous ggml expert banks (the CPU oracle).</param>
    /// <param name="viewOwnsPack">Dispose this pack when the returned view is disposed.</param>
    /// <param name="trunk">The trunk (skeleton) file: its <c>token_embd</c> / <c>output</c> are registered in the view because the tensor contract lists them; the head loaders never read them.</param>
    public GgufFile BuildHeadGguf(ModelConfig config, bool ggmlExpertBanks, GgufFile? trunk = null, bool viewOwnsPack = false)
    {
        int il = config.NumLayers, H = config.HiddenSize;
        if (H != Hidden) throw new InvalidDataException($"The Edge0 MTP head is built for hidden {Hidden}, the model has {H}.");
        var q4 = config.Qwen4Exp ?? throw new ArgumentException("A qwen4exp configuration is required.", nameof(config));
        int hcDim = q4.HyperConnectionCount * H;
        string b = $"blk.{il}.", n = b + "nextn.";
        var list = new List<(string, TensorShape, QuantizationType, nint)>();

        DenseTensor Need(string name) => _dense.TryGetValue(name, out var t) ? t : throw new InvalidDataException($"The Edge0 MTP head has no tensor '{name}'.");
        QuantizationType Qt(DenseTensor t) => t.Kind switch { "q8_0" => QuantizationType.Q8_0, "bf16" => QuantizationType.BF16, _ => QuantizationType.F32 };
        void Add(string gguf, string strata, params int[] dims)
        {
            var t = Need(strata);
            long elems = 1; foreach (int d in dims) elems *= d;
            if (elems != (long)t.Rows * t.Cols) throw new InvalidDataException($"{strata}: {t.Rows}x{t.Cols} does not fit {gguf} {string.Join('x', dims)}.");
            list.Add((gguf, new TensorShape(dims), Qt(t), t.Pointer));
        }

        // eh_proj = [fc_embedding | fc_hidden] per output row.
        var fe = Need("fc_embedding.weight"); var fh = Need("fc_hidden.weight");
        if (fe.Kind != "q8_0" || fh.Kind != "q8_0" || fe.Rows != H || fe.Cols != H || fh.Rows != H || fh.Cols != H) throw new InvalidDataException("fc_embedding / fc_hidden must be Q8_0 2560x2560.");
        long half = (long)(H / 32) * 34;
        nint eh = Alloc(2 * half * H);
        for (int o = 0; o < H; o++)
        {
            Buffer.MemoryCopy((void*)(fe.Pointer + (nint)(o * half)), (void*)(eh + (nint)(o * 2 * half)), half, half);
            Buffer.MemoryCopy((void*)(fh.Pointer + (nint)(o * half)), (void*)(eh + (nint)(o * 2 * half + half)), half, half);
        }
        list.Add((n + "eh_proj.weight", new TensorShape(2 * H, H), QuantizationType.Q8_0, eh));
        Add(n + "enorm.weight", "pre_fc_norm_embedding.weight", H);
        Add(n + "hnorm.weight", "pre_fc_norm_hidden.weight", hcDim);
        int lr = q4.HyperConnectionLowRank;
        Add(n + "hc_head_norm.weight", "hyper_connection_mixer.hc_norm.weight", hcDim);
        Add(n + "hc_head_down.weight", "hyper_connection_mixer.input_mix_weight_down.weight", hcDim, lr);
        Add(n + "hc_head_up.weight", "hyper_connection_mixer.input_mix_weight_up.weight", lr, hcDim);
        foreach (var (g, s) in new[] { ("hc_attn", "attn_hyper_connection"), ("hc_ffn", "mlp_hyper_connection") })
        {
            Add(b + g + "_norm.weight", s + ".hc_norm.weight", hcDim);
            Add(b + g + "_down.weight", s + ".input_mix_weight_down.weight", hcDim, lr);
            Add(b + g + "_up.weight", s + ".input_mix_weight_up.weight", lr, hcDim);
            Add(b + g + "_inject.weight", s + ".block_inject_weight.weight", hcDim, q4.HyperConnectionCount);
        }
        int heads = config.NumAttentionHeads, kv = config.NumKvHeads, hd = config.HeadDim;
        Add(b + "attn_q.weight", "self_attn.q_proj.weight", H, 2 * hd * heads);
        Add(b + "attn_k.weight", "self_attn.k_proj.weight", H, hd * kv);
        Add(b + "attn_v.weight", "self_attn.v_proj.weight", H, hd * kv);
        Add(b + "attn_output.weight", "self_attn.o_proj.weight", hd * heads, H);
        Add(b + "attn_q_norm.weight", "self_attn.q_norm.weight", hd);
        Add(b + "attn_k_norm.weight", "self_attn.k_norm.weight", hd);
        // The indexer's q and k projections are one 640-row matrix (4 query heads then the key): contiguous row slices, no copy.
        var iqk = Need("self_attn.indexer.index_qk_proj.weight");
        int idxQ = q4.IndexerHeadCount * q4.IndexerKeyLength, idxK = q4.IndexerKeyLength;
        if (iqk.Kind != "q8_0" || iqk.Rows != idxQ + idxK || iqk.Cols != H) throw new InvalidDataException($"index_qk_proj is {iqk.Kind} {iqk.Rows}x{iqk.Cols}; expected q8_0 {idxQ + idxK}x{H}.");
        list.Add((b + "indexer.q_proj.weight", new TensorShape(H, idxQ), QuantizationType.Q8_0, iqk.Pointer));
        list.Add((b + "indexer.k_proj.weight", new TensorShape(H, idxK), QuantizationType.Q8_0, iqk.Pointer + (nint)((long)idxQ * (H / 32) * 34)));
        Add(b + "indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight", q4.IndexerKeyLength);
        Add(b + "indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight", q4.IndexerKeyLength);
        int ffShared = config.Moe!.SharedExpertIntermediateSize ?? config.Moe.MoeIntermediateSize;
        Add(b + "ffn_gate_inp.weight", "mlp.gate.weight", H, Experts);
        Add(b + "ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight", H, ffShared);
        Add(b + "ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight", H, ffShared);
        Add(b + "ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight", ffShared, H);
        Add(b + "ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight", H);
        if (trunk is not null)
            foreach (string name in new[] { Architectures.Qwen4ExpTensors.TokenEmbd, Architectures.Qwen4ExpTensors.Output })
                if (trunk.TensorsByName.TryGetValue(name, out var td)) list.Add((name, td.Shape, td.QuantizationType, trunk.TensorDataPointer(td)));
        if (ggmlExpertBanks)
        {
            var (g, u, d) = RelayoutToGgmlBanks();
            list.Add((b + "ffn_gate_exps.weight", new TensorShape(H, ExpertFf, Experts), QuantizationType.Q2_0, g));
            list.Add((b + "ffn_up_exps.weight", new TensorShape(H, ExpertFf, Experts), QuantizationType.Q2_0, u));
            list.Add((b + "ffn_down_exps.weight", new TensorShape(ExpertFf, H, Experts), QuantizationType.Q2_0, d));
        }
        return GgufFile.CreateExternal(list, owner: viewOwnsPack ? this : null);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var (m, v) in _maps) { v.SafeMemoryMappedViewHandle.ReleasePointer(); v.Dispose(); m.Dispose(); }
        foreach (nint p in _owned) NativeMemory.AlignedFree((void*)p);
    }
}
