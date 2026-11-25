using System.Globalization;
using System.IO.MemoryMappedFiles;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Edge0;

/// <summary>Which factor <see cref="Edge0Pack.InverseColumnScale"/> returns (a diagnostic; production uses <see cref="Inverse"/>).</summary>
public enum Edge0ColumnScaleMode
{
    /// <summary>The D output is multiplied by <c>1/s</c> (edge0-strata's convention; the default).</summary>
    Inverse = 0,

    /// <summary>Diagnostic: multiplied by <c>s</c> itself (the opposite direction).</summary>
    Direct = 1,

    /// <summary>Diagnostic: not scaled at all.</summary>
    None = 2,
}

/// <summary>
/// Reader for the Edge0 "strata pack v3" model folder (HF <c>Edge0/Edge0-145B-q3k</c> and siblings): the latent-MoE part of an
/// Edge0 model that is NOT in the skeleton GGUF. Everything is mapped in place (read-only mmap); no weight is widened or copied.
/// The byte layouts are documented in <c>docs/EDGE0_PACK_FORMAT.md</c>.
/// </summary>
/// <remarks>
/// <para>Files consumed: <c>native_experts.txt</c> (per-layer expert blob geometry), <c>experts.bin</c> (fixed-size expert records),
/// <c>labels.bin</c> (expert to D/U group), <c>latent_dense_q8.bin</c> or <c>latent_dense_q4.bin</c> (D and U), and the optional
/// <c>scales/l{N}.f32</c> (gate/up column scales of the Q3_K build). The pack index (<c>index.txt</c>/<c>dense.bin</c>) and the
/// skeleton GGUF are read elsewhere.</para>
/// </remarks>
public sealed unsafe class Edge0Pack : IDisposable
{
    /// <summary>GGML type id of Q3_K in <c>native_experts.txt</c>.</summary>
    public const int GgmlQ3K = 11;

    /// <summary>GGML type id of IQ4_NL in <c>native_experts.txt</c>.</summary>
    public const int GgmlIq4Nl = 20;

    private readonly List<(MemoryMappedFile Mmf, MemoryMappedViewAccessor View)> _maps = [];
    private readonly int[] _labels;
    private readonly float[][]? _inverseColScale;
    private readonly float[][]? _colScale;
    private readonly nint _experts, _latent;
    private bool _disposed;

    /// <summary>Pack directory.</summary>
    public string Directory { get; }

    /// <summary>Number of MoE layers.</summary>
    public int Layers { get; }

    /// <summary>Routed experts per layer (512).</summary>
    public int Experts { get; }

    /// <summary>Model hidden width (2560): the router, D input and U output width.</summary>
    public int Hidden { get; }

    /// <summary>Latent width (1792 for the 145B, 1280 for the 120B): the expert gate/up input and down output width.</summary>
    public int Latent { get; }

    /// <summary>Expert FFN width (640).</summary>
    public int ExpertFf { get; }

    /// <summary>Number of D/U routing groups per layer (8).</summary>
    public int Groups { get; }

    /// <summary>Storage type of the expert gate and up projections (Q3_K).</summary>
    public QuantizationType GateUpType { get; }

    /// <summary>Storage type of the expert down projection (IQ4_NL).</summary>
    public QuantizationType DownType { get; }

    /// <summary>Storage type of D and U (<see cref="QuantizationType.Q8_0"/> or <see cref="QuantizationType.Q4_0"/>).</summary>
    public QuantizationType LatentDenseType { get; }

    /// <summary>Bytes of one gate or up projection (<see cref="ExpertFf"/> rows of <see cref="Latent"/> inputs).</summary>
    public long GateUpBytes { get; }

    /// <summary>Bytes of one down projection (<see cref="Latent"/> rows of <see cref="ExpertFf"/> inputs).</summary>
    public long DownBytes { get; }

    /// <summary>Bytes of one expert record (gate | up | down); also the byte stride between consecutive experts.</summary>
    public long ExpertRecordBytes { get; }

    /// <summary>
    /// Diagnostic switch (default <see cref="Edge0ColumnScaleMode.Inverse"/>, the semantics of edge0-strata): which factor
    /// <see cref="InverseColumnScale"/> returns. Exists so the direction of the scale can be tested end to end on real weights
    /// (teacher-forced NLL under each arm); production never sets it.
    /// </summary>
    public Edge0ColumnScaleMode ColumnScaleMode { get; set; } = Edge0ColumnScaleMode.Inverse;

    /// <summary>True when <c>scales/l{N}.f32</c> exist (the Q3_K build): gate/up rows are stored as <c>A*diag(s)</c>.</summary>
    public bool HasColumnScale => _inverseColScale is not null;

    /// <summary>Bytes of one D row (<see cref="Hidden"/> inputs) in <see cref="LatentDenseType"/>.</summary>
    public long DRowBytes { get; }

    /// <summary>Bytes of one U row (<see cref="Latent"/> inputs) in <see cref="LatentDenseType"/>.</summary>
    public long URowBytes { get; }

    /// <summary>Bytes of one layer's D block (<c>Groups * Latent</c> rows of <see cref="Hidden"/>).</summary>
    public long DLayerBytes => (long)Groups * Latent * DRowBytes;

    /// <summary>Bytes of one layer's U block (<c>Groups * Hidden</c> rows of <see cref="Latent"/>).</summary>
    public long ULayerBytes => (long)Groups * Hidden * URowBytes;

    private Edge0Pack(string dir, int layers, int experts, int hidden, int latent, int expertFf, int groups,
        QuantizationType guType, QuantizationType dType, QuantizationType latentType, int[] labels, float[][]? invScale,
        nint experts0, nint latent0, float[][]? colScale)
    {
        Directory = dir; Layers = layers; Experts = experts; Hidden = hidden; Latent = latent; ExpertFf = expertFf; Groups = groups;
        GateUpType = guType; DownType = dType; LatentDenseType = latentType;
        _labels = labels; _inverseColScale = invScale; _colScale = colScale; _experts = experts0; _latent = latent0;
        GateUpBytes = Dequantize.RowByteSize((long)latent, guType) * expertFf;
        DownBytes = Dequantize.RowByteSize(expertFf, dType) * latent;
        ExpertRecordBytes = 2 * GateUpBytes + DownBytes;
        DRowBytes = Dequantize.RowByteSize(hidden, latentType);
        URowBytes = Dequantize.RowByteSize(latent, latentType);
    }

    /// <summary>
    /// Opens a pack folder and validates every file size against the geometry (file sizes, per-layer offsets and record sizes in
    /// <c>native_experts.txt</c>, label range, scale vector length). Does not touch the expert bytes.
    /// </summary>
    /// <param name="directory">The model folder (an HF snapshot directory works; the files are read in place).</param>
    /// <param name="hidden">Model hidden width (the router width; <c>ModelConfig.HiddenSize</c>).</param>
    /// <param name="expertFf">Expert FFN width (<c>moe_intermediate_size</c>, 640).</param>
    /// <param name="preferLatentQ4">Open <c>latent_dense_q4.bin</c> instead of <c>latent_dense_q8.bin</c> when both exist.</param>
    /// <exception cref="InvalidDataException">A file is missing or inconsistent with the geometry.</exception>
    /// <exception cref="NotSupportedException">The pack stores an expert type this reader does not handle (NVFP4 packs carry a 16-byte tail).</exception>
    public static Edge0Pack Open(string directory, int hidden = 2560, int expertFf = 640, bool preferLatentQ4 = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        string manifest = Path.Combine(directory, "native_experts.txt");
        string expertsPath = Path.Combine(directory, "experts.bin");
        string labelsPath = Path.Combine(directory, "labels.bin");
        foreach (string f in new[] { manifest, expertsPath, labelsPath })
            if (!File.Exists(f)) throw new InvalidDataException($"Edge0 pack file missing: {f}");

        // ── native_experts.txt: "layer gu_type d_type offset blob_bytes" per layer ──
        var rows = new List<(int Layer, int Gu, int D, long Offset, long Bytes)>();
        bool manifestNamesColscale = false;
        foreach (string raw in File.ReadLines(manifest))
        {
            string line = raw.Trim();
            if (line.Length == 0) continue;
            if (line[0] == '#')
            {
                // The header ("... latent A-only, Q3_K gu + shared colscale ...") says the gate/up rows were stored as A*diag(s).
                manifestNamesColscale |= line.Contains("colscale", StringComparison.OrdinalIgnoreCase);
                continue;
            }
            string[] p = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (p.Length < 5) throw new InvalidDataException($"native_experts.txt: malformed row '{line}'");
            rows.Add((int.Parse(p[0], CultureInfo.InvariantCulture), int.Parse(p[1], CultureInfo.InvariantCulture),
                int.Parse(p[2], CultureInfo.InvariantCulture), long.Parse(p[3], CultureInfo.InvariantCulture),
                long.Parse(p[4], CultureInfo.InvariantCulture)));
        }
        if (rows.Count == 0) throw new InvalidDataException("native_experts.txt has no layer rows.");
        int layers = rows.Count;
        var (_, gu, d, _, recBytes) = rows[0];
        if (gu != GgmlQ3K || d != GgmlIq4Nl)
            throw new NotSupportedException(
                $"native_experts.txt: expert types (gate/up {gu}, down {d}) are not Q3_K/IQ4_NL ({GgmlQ3K}/{GgmlIq4Nl}); " +
                "only the Q3_K build is supported (the NVFP4 build adds a 16-byte per-expert scale tail: follow-up).");

        // ── latent width: scales vector length, else derived from the record size ──
        string scalesDir = Path.Combine(directory, "scales");
        bool hasScales = File.Exists(Path.Combine(scalesDir, "l0.f32"));
        // Decided by the manifest, not by what happens to be on disk: a pack whose manifest names the column scale but whose scales/ is
        // absent (a partial download, a skipped folder) would otherwise load as an unscaled pack and run silently wrong (#1972).
        if (manifestNamesColscale && !hasScales)
            throw new InvalidDataException(
                $"native_experts.txt names a shared column scale (gate/up rows are stored as A*diag(s)) but {Path.Combine(scalesDir, "l0.f32")} is missing: " +
                "download the scales/ folder of the pack (a pack without it would run unscaled and produce wrong output).");
        int latent;
        if (hasScales) latent = checked((int)(LengthOf(Path.Combine(scalesDir, "l0.f32")) / sizeof(float)));
        else latent = SolveLatent(recBytes, expertFf);
        var guType = QuantizationType.Q3_K; var dType = QuantizationType.IQ4_NL;
        long guBytes = Dequantize.RowByteSize(latent, guType) * expertFf;
        long dnBytes = Dequantize.RowByteSize(expertFf, dType) * latent;
        long record = 2 * guBytes + dnBytes;
        if (record != recBytes)
            throw new InvalidDataException($"native_experts.txt: record bytes {recBytes} != geometry ({latent} latent, {expertFf} ff) {record}.");
        for (int l = 0; l < layers; l++)
        {
            if (rows[l].Layer != l || rows[l].Gu != gu || rows[l].D != d || rows[l].Bytes != record)
                throw new InvalidDataException($"native_experts.txt: row {l} is inconsistent ({rows[l]}).");
        }

        // ── labels ──
        long labelBytes = LengthOf(labelsPath);
        if (labelBytes % (4L * layers) != 0)
            throw new InvalidDataException($"labels.bin: {labelBytes} B is not int32 x {layers} layers x experts.");
        int experts = checked((int)(labelBytes / 4 / layers));
        for (int l = 0; l < layers; l++)
            if (rows[l].Offset != (long)l * experts * record)
                throw new InvalidDataException($"native_experts.txt: layer {l} offset {rows[l].Offset} != {(long)l * experts * record}.");
        long expertsLen = LengthOf(expertsPath);
        if (expertsLen != (long)layers * experts * record)
            throw new InvalidDataException($"experts.bin: {expertsLen} B != {layers} x {experts} x {record} = {(long)layers * experts * record}.");
        int[] labels = new int[layers * experts];
        using (var fs = File.OpenRead(labelsPath))
        {
            var bytes = new byte[labelBytes];
            fs.ReadExactly(bytes);
            Buffer.BlockCopy(bytes, 0, labels, 0, bytes.Length);   // little-endian on every supported host
        }

        // ── column scales: store 1/s (s <= 0 -> 1), the factor the D output is multiplied by ──
        float[][]? inv = null, rawScales = null;
        if (hasScales)
        {
            inv = new float[layers][];
            rawScales = new float[layers][];
            for (int l = 0; l < layers; l++)
            {
                string p = Path.Combine(scalesDir, $"l{l}.f32");
                if (!File.Exists(p)) throw new InvalidDataException($"scales: {p} is missing (all-or-none).");
                byte[] sb = File.ReadAllBytes(p);
                if (sb.Length != latent * sizeof(float))
                    throw new InvalidDataException($"scales: {p} has {sb.Length} B, expected {latent * sizeof(float)}.");
                var s = new float[latent];
                Buffer.BlockCopy(sb, 0, s, 0, sb.Length);
                var iv = new float[latent];
                for (int i = 0; i < latent; i++) iv[i] = s[i] > 0f ? 1f / s[i] : 1f;
                inv[l] = iv;
                rawScales[l] = s;
            }
        }

        // ── latent dense: size decides the group count and cross-checks the format ──
        string q8 = Path.Combine(directory, "latent_dense_q8.bin"), q4 = Path.Combine(directory, "latent_dense_q4.bin");
        string latentPath = preferLatentQ4 && File.Exists(q4) ? q4 : File.Exists(q8) ? q8 : File.Exists(q4) ? q4 : "";
        if (latentPath.Length == 0) throw new InvalidDataException("latent_dense_q8.bin / latent_dense_q4.bin missing.");
        var latentType = latentPath == q8 ? QuantizationType.Q8_0 : QuantizationType.Q4_0;
        long perGroup = ((long)latent * Dequantize.RowByteSize(hidden, latentType)) + ((long)hidden * Dequantize.RowByteSize(latent, latentType));
        long latentLen = LengthOf(latentPath);
        if (latentLen % ((long)layers * perGroup) != 0)
            throw new InvalidDataException($"{Path.GetFileName(latentPath)}: {latentLen} B is not {layers} layers x groups x {perGroup} B.");
        int groups = checked((int)(latentLen / layers / perGroup));
        for (int i = 0; i < labels.Length; i++)
            if ((uint)labels[i] >= (uint)groups)
                throw new InvalidDataException($"labels.bin: entry {i} = {labels[i]} is outside 0..{groups - 1}.");

        var maps = new List<(MemoryMappedFile, MemoryMappedViewAccessor)>();
        try
        {
            nint e0 = Map(expertsPath, maps), l0 = Map(latentPath, maps);
            var pack = new Edge0Pack(directory, layers, experts, hidden, latent, expertFf, groups, guType, dType, latentType,
                labels, inv, e0, l0, rawScales);
            pack._maps.AddRange(maps);
            return pack;
        }
        catch
        {
            foreach (var (m, v) in maps) { v.SafeMemoryMappedViewHandle.ReleasePointer(); v.Dispose(); m.Dispose(); }
            throw;
        }
    }

    /// <summary>File length through an open handle: correct for a symlinked file (an HF snapshot entry), where <c>FileInfo.Length</c> may describe the link.</summary>
    private static long LengthOf(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        return fs.Length;
    }

    private static int SolveLatent(long recordBytes, int expertFf)
    {
        // record = 2 * ff * rowQ3K(l) + l * rowIq4Nl(ff); with l = 256 n a Q3_K row is 110 n bytes, so the record is linear in n.
        long iq4Row = Dequantize.RowByteSize(expertFf, QuantizationType.IQ4_NL);
        long denom = 2L * expertFf * 110 + 256L * iq4Row;
        if (recordBytes % denom != 0) throw new InvalidDataException($"cannot derive the latent width from record bytes {recordBytes}.");
        return checked((int)(recordBytes / denom * 256));
    }

    private static nint Map(string path, List<(MemoryMappedFile, MemoryMappedViewAccessor)> maps)
    {
        var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* p = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        maps.Add((mmf, view));
        return (nint)p + (nint)view.PointerOffset;
    }

    // ───────────────────────────── experts ─────────────────────────────

    /// <summary>Pointer to the first byte of expert <paramref name="expert"/>'s record (its gate projection).</summary>
    public nint ExpertRecord(int layer, int expert)
    {
        CheckLayer(layer);
        if ((uint)expert >= (uint)Experts) throw new ArgumentOutOfRangeException(nameof(expert));
        return _experts + (nint)(((long)layer * Experts + expert) * ExpertRecordBytes);
    }

    /// <summary>Base of the gate bank of a layer: expert <c>e</c>'s gate is at <c>GateBank + e * ExpertRecordBytes</c> (stride, not packed).</summary>
    public nint GateBank(int layer) => ExpertRecord(layer, 0);

    /// <summary>Base of the up bank of a layer (see <see cref="GateBank"/>).</summary>
    public nint UpBank(int layer) => ExpertRecord(layer, 0) + (nint)GateUpBytes;

    /// <summary>Base of the down bank of a layer (see <see cref="GateBank"/>).</summary>
    public nint DownBank(int layer) => ExpertRecord(layer, 0) + (nint)(2 * GateUpBytes);

    /// <summary>D/U routing group of every expert of <paramref name="layer"/>.</summary>
    public ReadOnlySpan<int> Labels(int layer)
    {
        CheckLayer(layer);
        return _labels.AsSpan(layer * Experts, Experts);
    }

    /// <summary>D/U group of one expert.</summary>
    public int GroupOf(int layer, int expert) => Labels(layer)[expert];

    /// <summary>
    /// <c>1/s</c> per latent column (<c>s &lt;= 0</c> maps to 1): the factor the D output is multiplied by before the gate/up projections.
    /// Empty when the pack has no <c>scales/</c> folder. Under the diagnostic <see cref="ColumnScaleMode"/> it returns <c>s</c> itself
    /// (<see cref="Edge0ColumnScaleMode.Direct"/>) or nothing (<see cref="Edge0ColumnScaleMode.None"/>).
    /// </summary>
    public ReadOnlySpan<float> InverseColumnScale(int layer)
    {
        CheckLayer(layer);
        if (_inverseColScale is null) return default;
        return ColumnScaleMode switch
        {
            Edge0ColumnScaleMode.Direct => _colScale![layer],
            Edge0ColumnScaleMode.None => default,
            _ => _inverseColScale[layer],
        };
    }

    // ───────────────────────────── D / U ─────────────────────────────

    /// <summary>All eight D matrices of a layer stacked: <c>Groups * Latent</c> rows of <see cref="Hidden"/> inputs; group g is rows <c>[g*Latent, (g+1)*Latent)</c>.</summary>
    public nint DLayer(int layer)
    {
        CheckLayer(layer);
        return _latent + (nint)(layer * (DLayerBytes + ULayerBytes));
    }

    /// <summary>Group <paramref name="group"/>'s D: <see cref="Latent"/> rows of <see cref="Hidden"/> inputs.</summary>
    public nint D(int layer, int group)
    {
        CheckGroup(group);
        return DLayer(layer) + (nint)((long)group * Latent * DRowBytes);
    }

    /// <summary>Group <paramref name="group"/>'s U: <see cref="Hidden"/> rows of <see cref="Latent"/> inputs.</summary>
    public nint U(int layer, int group)
    {
        CheckGroup(group);
        return DLayer(layer) + (nint)DLayerBytes + (nint)((long)group * Hidden * URowBytes);
    }

    private void CheckLayer(int layer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);   // the views are unmapped: a pointer would dangle
        if ((uint)layer >= (uint)Layers) throw new ArgumentOutOfRangeException(nameof(layer));
    }

    private void CheckGroup(int group)
    {
        if ((uint)group >= (uint)Groups) throw new ArgumentOutOfRangeException(nameof(group));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var (m, v) in _maps) { v.SafeMemoryMappedViewHandle.ReleasePointer(); v.Dispose(); m.Dispose(); }
        _maps.Clear();
    }
}
