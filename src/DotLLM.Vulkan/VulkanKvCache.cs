using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan-side KV cache. Per-layer device-local buffers of shape
/// <c>[maxSeqLen, numKvHeads * headDim]</c>, FP32 or (issue #1171) FP16 (<see cref="Storage"/>). The host never touches
/// cached K/V — updates are recorded as <c>vkCmdCopyBuffer</c> from the
/// host-visible activation buffers to the device-local cache, either
/// synchronously (legacy <see cref="UpdateDevice"/>) or appended to a caller-
/// supplied command buffer (<see cref="RecordUpdate"/>, used by the
/// fence-pipelined forward pass).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <c>DotLLM.Engine.KvCache.SimpleKvCache</c> semantics: <c>Update</c>
/// appends new K/V rows at the supplied position indices. The attention kernel
/// reads straight from the device buffers via <see cref="GetKeysBuffer"/> /
/// <see cref="GetValuesBuffer"/>; no staging copies are required.
/// </para>
/// <para>
/// Implements <see cref="IKvCache"/> so code that already knows about the CPU
/// cache semantics (text-generation loop, tests) can swap the backing store
/// transparently. The <see cref="IKvCache.Update(ITensor, ITensor, ReadOnlySpan{int}, int)"/>
/// and <see cref="IKvCache.Update(TensorRef, TensorRef, ReadOnlySpan{int}, int)"/>
/// overloads expect CPU-resident tensor pointers (the caller is responsible
/// for uploading); we only use the device-side path from
/// <see cref="VulkanTransformerModel"/>, but the IKvCache contract lets this
/// object satisfy the same API.
/// </para>
/// </remarks>
public sealed class VulkanKvCache : IKvCache, IPerLayerKvCache, IHostStagedKvCache, IPositionedHostKvCache, IWindowedKvCache, IPrefixReusableKvCache, IKvPrefixSeedable,
    IDeviceResidentKvCache
{
    private readonly VulkanDevice _device;
    private readonly VulkanDevice.Buffer[] _keys;
    private readonly VulkanDevice.Buffer[] _values;
    private readonly int _numLayers;
    private int _maxSeqLen;   // raised by TryGrowTo (issue #1781)
    // Per-layer row stride (numKvHeads × headDim, FP32 elements). Uniform across
    // layers for every dense/GQA/MoE model; PER-LAYER for Gemma-4, whose sliding
    // and global layers carry different KV-head counts AND head dims (e.g. sliding
    // 8×256 vs global 2×512) — so each layer's cached K/V row is a different width.
    private readonly int[] _kvStride;
    private readonly VulkanKvStorage _storage;
    private readonly int _elemBytes;

    // Compact sliding-window rows (#1176): null unless the geometry marks windowed slots. Grow-only, so a resize (which swaps
    // buffer handles and invalidates descriptor sets) happens only in EnsureWindowCapacity, before recording.
    private readonly SlidingWindowRows? _rows;

    // TILE_KV of the reference sink kernel (attention_sinks_f32.glsl), which tiles the KV range from row 0. Ring bases stay multiples of it,
    // so the kernel sees the same tiles as on a full-length cache and the ring is bit-identical to it (the decode / small-row kernels
    // split relative to the window start and do not care). The GQA flash sink twins (#1200) tile in 16 / 32 rows, which divide it.
    private const int KvTileRows = 256;
    private readonly VulkanDevice.Buffer? _moveScratch; // overlapping compaction moves stage through here (K then V half)

    // TurboQuant storage (issue #1892): _keys / _values hold the packed codes (CodeUintsPerVec words per head vector plus one guard
    // word per layer); these hold the rest. Null / default for every other storage. Nothing wider than codes + norms is ever stored.
    private readonly VulkanDevice.Buffer[]? _keyNorms;
    private readonly VulkanDevice.Buffer[]? _valueNorms;
    private readonly VulkanDevice.Buffer? _tqCentroids;   // 2^mseBits F32, shared by K and V
    private readonly VulkanDevice.Buffer? _tqSignsK;      // headDim F32 +-1 (K rotation)
    private readonly VulkanDevice.Buffer? _tqSignsV;      // headDim F32 +-1 (V rotation; a different seed)
    private readonly TurboQuantLayout _tqLayout;
    private readonly float _tqInvSqrtD;
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>
    /// Device bytes this cache holds now (<see cref="IDeviceResidentKvCache"/>): every layer's K and V buffer plus the compaction
    /// scratch. Being device-resident is what lets a prefix hit whose growth is refused evict the other retained entry and retry
    /// (issue #1843, as on CUDA) instead of re-prefilling the conversation.
    /// </summary>
    public long DeviceBytes
    {
        get
        {
            long bytes = _moveScratch?.Size ?? 0;
            for (int i = 0; i < _numLayers; i++) bytes += _keys[i].Size + _values[i].Size;
            if (_keyNorms is not null)   // TurboQuant: norms and the constant buffers count too (#1892)
            {
                for (int i = 0; i < _numLayers; i++) bytes += _keyNorms[i].Size + _valueNorms![i].Size;
                bytes += _tqCentroids!.Size + _tqSignsK!.Size + _tqSignsV!.Size;
            }
            return bytes;
        }
    }

    /// <summary>
    /// Element format of the cached rows. <see cref="VulkanKvStorage.F16"/> rows are written by the F16 convert kernel and read in place by the
    /// <c>_kvf16</c> attention twins; only models that create those twins (Flint, issue #1171) accept such a cache.
    /// </summary>
    public VulkanKvStorage Storage => _storage;

    /// <summary>Device bytes held by the K and V buffers of every layer.</summary>
    public long AllocatedBytes
    {
        get
        {
            if (_storage == VulkanKvStorage.Tq) return DeviceBytes;
            if (_rows is not null) return _rows.AllocatedBytes(i => _storage.KeyRowBytes(_kvStride[i]) + _storage.ValueRowBytes(_kvStride[i]));
            long total = 0;
            for (int i = 0; i < _numLayers; i++) total += _maxSeqLen * (_storage.KeyRowBytes(_kvStride[i]) + _storage.ValueRowBytes(_kvStride[i]));
            return total;
        }
    }

    /// <summary>
    /// Creates the per-layer K/V buffers with a UNIFORM <c>numKvHeads × headDim</c>
    /// row stride (every layer the same width — the dense / GQA / MoE case).
    /// Memory is not zeroed — the forward pass only reads positions it has written.
    /// </summary>
    public VulkanKvCache(VulkanDevice device, int numLayers, int numKvHeads, int headDim, int maxSeqLen)
        : this(device, BuildUniformStrides(numLayers, numKvHeads, headDim), maxSeqLen)
    {
    }

    /// <summary>
    /// Creates the per-layer K/V buffers from a Core <see cref="KvGeometry"/> descriptor.
    /// Byte-identical to the uniform constructor for every dense/GQA/MoE model and supplies
    /// distinct per-layer strides for Gemma-4. This is the constructor
    /// <see cref="VulkanTransformerModel.CreateKvCache(int)"/> uses (via
    /// <see cref="KvGeometry.FromConfig(DotLLM.Core.Models.ModelConfig)"/>), so the per-layer-stride derivation lives in one
    /// place (Core) rather than being rebuilt inline.
    /// </summary>
    public VulkanKvCache(VulkanDevice device, KvGeometry geometry, int maxSeqLen)
        : this(device, geometry, maxSeqLen, VulkanKvStorage.F32)
    {
    }

    /// <summary>
    /// As <see cref="VulkanKvCache(VulkanDevice, KvGeometry, int)"/> with an explicit element format (issue #1171). A geometry with
    /// sliding windows (<see cref="KvGeometry.HasSlidingWindows"/>) keeps only the recent rows of those slots (issue #1176): see
    /// <see cref="IWindowedKvCache"/> and <see cref="EnsureWindowCapacity"/>.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="geometry">Per-layer KV geometry.</param>
    /// <param name="maxSeqLen">Positions per layer.</param>
    /// <param name="storage">Element format of the cached rows.</param>
    public VulkanKvCache(VulkanDevice device, KvGeometry geometry, int maxSeqLen, VulkanKvStorage storage)
        : this(device, ExtractStrides(geometry), maxSeqLen, storage,
            geometry.HasSlidingWindows ? new SlidingWindowRows(geometry, maxSeqLen, growOnly: true, baseAlignment: KvTileRows) : null)
    {
    }

    private static int[] ExtractStrides(KvGeometry geometry)
    {
        var strides = new int[geometry.LayerCount];
        for (int l = 0; l < strides.Length; l++)
            strides[l] = geometry.KvStrideOf(l);
        return strides;
    }

    /// <summary>
    /// Creates the per-layer K/V buffers from an explicit PER-LAYER row stride
    /// (FP32 elements per token position, i.e. <c>numKvHeads × headDim</c> for that
    /// layer). Used by Gemma-4, whose sliding and global layers have different
    /// KV-head counts and head dims, so each layer's cached row is a different width.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="kvStridePerLayer">Per-layer row width in elements.</param>
    /// <param name="maxSeqLen">Positions per layer.</param>
    /// <param name="storage">Element format of the cached rows (F16 needs an even row width: two halves per word).</param>
    public VulkanKvCache(VulkanDevice device, int[] kvStridePerLayer, int maxSeqLen, VulkanKvStorage storage = VulkanKvStorage.F32)
        : this(device, kvStridePerLayer, maxSeqLen, storage, rows: null)
    {
    }

    /// <summary>
    /// Creates an in-place TurboQuant cache (issue #1892): per layer, the packed codes and the norms of K and V in
    /// <paramref name="layout"/>, plus the shared centroids and the two sign vectors. Rows are written by
    /// <see cref="Kernels.TurboQuantEncodeF32Kernel"/> and read in place, in code space, by the <c>_kvtq</c> attention twins. The codec
    /// constants come from the caller (the Vulkan project does not depend on the Engine codec). Uniform geometry; full-length rows on
    /// every layer (no compact sliding-window ring). Bytes per cached position and layer:
    /// <c>2 x numKvHeads x (4 x CodeUintsPerVec + NormBytes)</c>.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="numLayers">Layers.</param>
    /// <param name="maxSeqLen">Positions per layer.</param>
    /// <param name="layout">Code and norm layout (head dim a power of two at most 256; F16 norms unless a reference run).</param>
    /// <param name="centroids">The <c>2^mseBits</c> centroids, already scaled by 1/sqrt(d).</param>
    /// <param name="signsK">K rotation signs (+-1), <c>headDim</c> entries.</param>
    /// <param name="signsV">V rotation signs, from a different seed than <paramref name="signsK"/>.</param>
    /// <param name="invSqrtD">The codec's exact 1/sqrt(headDim).</param>
    public VulkanKvCache(VulkanDevice device, int numLayers, int maxSeqLen, TurboQuantLayout layout,
        ReadOnlySpan<float> centroids, ReadOnlySpan<float> signsK, ReadOnlySpan<float> signsV, float invSqrtD)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (layout.NumKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(layout));
        layout = Kernels.TqKvTwin.Require(VulkanKvStorage.Tq, layout)!.Value;
        if (centroids.Length != 1 << layout.MseBits)
            throw new ArgumentException($"centroids must have 2^mseBits = {1 << layout.MseBits} entries.", nameof(centroids));
        if (signsK.Length != layout.HeadDim || signsV.Length != layout.HeadDim)
            throw new ArgumentException("signs must have headDim entries.", nameof(signsK));

        _device = device;
        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _kvStride = BuildUniformStrides(numLayers, layout.NumKvHeads, layout.HeadDim);
        _storage = VulkanKvStorage.Tq;
        _elemBytes = 0;   // no element size: the host-staged element paths refuse TurboQuant
        _tqLayout = layout;
        _tqInvSqrtD = invSqrtD;
        _keys = new VulkanDevice.Buffer[numLayers];
        _values = new VulkanDevice.Buffer[numLayers];
        _keyNorms = new VulkanDevice.Buffer[numLayers];
        _valueNorms = new VulkanDevice.Buffer[numLayers];
        long codeBytes = layout.CodeBytesPerLayer(maxSeqLen);   // includes the guard word of the straddle read
        long normBytes = Kernels.TqKvTwin.NormBufferBytes((long)maxSeqLen * layout.NumKvHeads, layout.Norm);
        try
        {
            for (int i = 0; i < numLayers; i++)
            {
                _keys[i] = device.AllocateDeviceLocal(codeBytes);
                _values[i] = device.AllocateDeviceLocal(codeBytes);
                _keyNorms[i] = device.AllocateDeviceLocal(normBytes);
                _valueNorms[i] = device.AllocateDeviceLocal(normBytes);
            }
            _tqCentroids = device.Allocate((long)centroids.Length * sizeof(float));
            _tqSignsK = device.Allocate((long)layout.HeadDim * sizeof(float));
            _tqSignsV = device.Allocate((long)layout.HeadDim * sizeof(float));
            device.Upload(centroids, _tqCentroids);
            device.Upload(signsK, _tqSignsK);
            device.Upload(signsV, _tqSignsV);
        }
        catch
        {
            // A Dispose that itself throws must not mask the allocation failure that got us here.
            try { Dispose(); } catch { /* best effort: the original exception is the one the caller needs */ }
            throw;
        }
    }

    /// <summary>The TurboQuant layout of a <see cref="VulkanKvStorage.Tq"/> cache (issue #1892); null for every other storage.</summary>
    public TurboQuantLayout? TurboQuantLayout => _storage == VulkanKvStorage.Tq ? _tqLayout : null;

    /// <summary>The norm and centroid buffers a <c>_kvtq</c> attention twin binds for <paramref name="layerIndex"/>.</summary>
    internal Kernels.TqKvOperands TqOperands(int layerIndex)
    {
        RequireTq();
        return new(_keyNorms![layerIndex], _valueNorms![layerIndex], _tqCentroids!);
    }

    /// <summary>K rotation signs: the query is rotated with these before a twin scores it (<c>tq_rotate_q</c>).</summary>
    internal VulkanDevice.Buffer TqSignsK { get { RequireTq(); return _tqSignsK!; } }

    /// <summary>V rotation signs: the attention output is rotated back with these (<c>tq_rotate_o</c>).</summary>
    internal VulkanDevice.Buffer TqSignsV { get { RequireTq(); return _tqSignsV!; } }

    private void RequireTq()
    {
        if (_storage != VulkanKvStorage.Tq) throw new InvalidOperationException($"This KV cache stores {_storage}, not TurboQuant.");
    }

    /// <summary>Test hook: the stored code words of a layer side, <c>CodeUintsPerVec</c> per head vector.</summary>
    internal void DownloadTqCodes(int layerIndex, bool values, Span<uint> destination)
    {
        RequireTq();
        _device.Download((values ? _values : _keys)[layerIndex], System.Runtime.InteropServices.MemoryMarshal.Cast<uint, float>(destination));
    }

    /// <summary>Test hook: the stored norm bytes of a layer side, in the layout's norm format.</summary>
    internal void DownloadTqNormBytes(int layerIndex, bool values, Span<byte> destination)
    {
        RequireTq();
        var tmp = new float[(destination.Length + 3) / 4];
        _device.Download((values ? _valueNorms! : _keyNorms!)[layerIndex], tmp.AsSpan());
        MemoryMarshal.AsBytes(tmp.AsSpan())[..destination.Length].CopyTo(destination);
    }

    private VulkanKvCache(VulkanDevice device, int[] kvStridePerLayer, int maxSeqLen, VulkanKvStorage storage, SlidingWindowRows? rows)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(kvStridePerLayer);
        if (kvStridePerLayer.Length == 0) throw new ArgumentOutOfRangeException(nameof(kvStridePerLayer));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (storage == VulkanKvStorage.Tq)
            throw new ArgumentException("A TurboQuant KV cache needs its layout and codec constants; use that constructor (#1892).", nameof(storage));

        _device = device;
        _numLayers = kvStridePerLayer.Length;
        _maxSeqLen = maxSeqLen;
        _kvStride = (int[])kvStridePerLayer.Clone();
        _storage = storage;
        _elemBytes = storage.IsBlockQuantized() ? 0 : storage.ElementBytes();   // Q8_0 rows are sized by RowBytes; the host-staged element paths refuse Q8_0
        _rows = rows;

        _keys = new VulkanDevice.Buffer[_numLayers];
        _values = new VulkanDevice.Buffer[_numLayers];

        for (int i = 0; i < _numLayers; i++)
        {
            if (_kvStride[i] <= 0)
                throw new ArgumentOutOfRangeException(nameof(kvStridePerLayer),
                    $"Per-layer KV stride must be positive; layer {i} = {_kvStride[i]}.");
            if (storage == VulkanKvStorage.F16 && (_kvStride[i] & 1) != 0)
                throw new ArgumentOutOfRangeException(nameof(kvStridePerLayer), $"An F16 KV row must hold an even element count; layer {i} = {_kvStride[i]}.");
            if (storage.IsBlockQuantized() && (_kvStride[i] & 31) != 0)
                throw new ArgumentOutOfRangeException(nameof(kvStridePerLayer), $"A {storage} KV row must be a multiple of 32 elements; layer {i} = {_kvStride[i]}.");
            long rowsPerLayer = rows?.CapacityOf(i) ?? maxSeqLen;   // K and V sized per side: mixed block formats differ (#1884)
            _keys[i] = device.AllocateDeviceLocal(rowsPerLayer * storage.KeyRowBytes(_kvStride[i]));
            _values[i] = device.AllocateDeviceLocal(rowsPerLayer * storage.ValueRowBytes(_kvStride[i]));
        }

        if (rows is not null)
        {
            // K and V of the largest possible compaction stage side by side.
            long scratch = 0;
            for (int i = 0; i < _numLayers; i++)
                scratch = Math.Max(scratch, rows.MaxKeptRows(i) * (storage.KeyRowBytes(_kvStride[i]) + storage.ValueRowBytes(_kvStride[i])));
            _moveScratch = device.AllocateDeviceLocal(scratch);
        }
    }

    private static int[] BuildUniformStrides(int numLayers, int numKvHeads, int headDim)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        var strides = new int[numLayers];
        Array.Fill(strides, numKvHeads * headDim);
        return strides;
    }

    /// <summary>Returns the device buffer holding cached keys for the given layer.</summary>
    internal VulkanDevice.Buffer GetKeysBuffer(int layerIndex) => _keys[layerIndex];

    /// <summary>Returns the device buffer holding cached values for the given layer.</summary>
    internal VulkanDevice.Buffer GetValuesBuffer(int layerIndex) => _values[layerIndex];

    /// <summary>
    /// Copies new <paramref name="kDev"/> / <paramref name="vDev"/> rows into
    /// the device-local cached K/V buffers at the given positions. Source
    /// buffers are the current forward pass's host-visible K/V activations;
    /// the cache destination is device-local (VRAM on a dGPU, driver-tiled on
    /// UMA). Issues a synchronous <c>vkCmdCopyBuffer</c> + fence wait.
    /// Prefer <see cref="RecordUpdate"/> from the fence-pipelined forward pass.
    /// </summary>
    internal void UpdateDevice(
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));
        RequireF32("UpdateDevice");
        ThrowIfWindowed(layerIndex, "UpdateDevice");

        int rowBytes = _kvStride[layerIndex] * sizeof(float);

        // Single contiguous range if positions are consecutive — one copy call
        // covers the whole seqLen. Otherwise fall back to per-row copies.
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        bool contiguous = IsContiguousAscending(positions);

        if (contiguous)
        {
            int startPos = positions[0];
            ulong totalBytes = (ulong)rowBytes * (ulong)seqLen;
            _device.CopyBufferRangeSynchronous(kDev, _keys[layerIndex],
                srcOffset: 0, dstOffset: (ulong)((long)startPos * rowBytes), size: totalBytes);
            _device.CopyBufferRangeSynchronous(vDev, _values[layerIndex],
                srcOffset: 0, dstOffset: (ulong)((long)startPos * rowBytes), size: totalBytes);
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                _device.CopyBufferRangeSynchronous(kDev, _keys[layerIndex],
                    srcOffset: (ulong)((long)i * rowBytes),
                    dstOffset: (ulong)((long)pos * rowBytes),
                    size: (ulong)rowBytes);
                _device.CopyBufferRangeSynchronous(vDev, _values[layerIndex],
                    srcOffset: (ulong)((long)i * rowBytes),
                    dstOffset: (ulong)((long)pos * rowBytes),
                    size: (ulong)rowBytes);
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <summary>
    /// Appends K/V copy commands onto the supplied <paramref name="cmdBuf"/>.
    /// The caller is responsible for the <c>TRANSFER → COMPUTE_SHADER</c>
    /// barrier that follows (so the attention kernel reads the freshly
    /// written cache rows), and for advancing <see cref="CurrentLength"/>
    /// after the batch commits.
    /// </summary>
    /// <remarks>An F16 cache is written by <paramref name="f16Writer"/> (a compute dispatch: the caller's following barrier is compute to compute).</remarks>
    internal unsafe void RecordUpdate(
        nint cmdBuf,
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex,
        Kernels.IVulkanKvF16Writer? f16Writer = null,
        VulkanDevice.Buffer? positionsDev = null,
        Kernels.IVulkanKvQ8Writer? q8Writer = null,
        Kernels.IVulkanKvQ8Writer? q8ValueWriter = null,   // the V side's writer when K and V differ (mixed block formats, #1884)
        Kernels.TurboQuantEncodeF32Kernel? tqWriter = null)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        if (_storage == VulkanKvStorage.Tq)
        {
            // Encoded on the device, one workgroup per head vector, K and V with their own sign sets (#1892). Attention then reads these
            // rows, the ones just written included, from the codes: there is no original-space copy to read instead.
            if (tqWriter is null)
                throw new InvalidOperationException("A TurboQuant KV cache is written by the encode kernel; pass the writer.");
            int tqMax = ValidateAndFindMaxPos(positions, seqLen);
            if (!IsContiguousAscending(positions))
                throw new NotSupportedException(
                    "A TurboQuant KV cache is written at contiguous positions only; the row-scatter writer (selective recompute, chunk-KV reuse) is step 4 of #1887.");
            var l = _tqLayout;
            tqWriter.Record(cmdBuf, kDev, _tqCentroids!, _tqSignsK!, _keys[layerIndex], _keyNorms![layerIndex],
                seqLen, l.HeadDim, l.NumKvHeads, l.MseBits, l.CodeUintsPerVec, positions[0], _tqInvSqrtD, l.Norm);
            tqWriter.Record(cmdBuf, vDev, _tqCentroids!, _tqSignsV!, _values[layerIndex], _valueNorms![layerIndex],
                seqLen, l.HeadDim, l.NumKvHeads, l.MseBits, l.CodeUintsPerVec, positions[0], _tqInvSqrtD, l.Norm);
            if (tqMax + 1 > _currentLength) _currentLength = tqMax + 1;
            return;
        }

        int rowBytes = _kvStride[layerIndex] * sizeof(float);
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        bool contiguous = IsContiguousAscending(positions);

        // Compact sliding-window slot (#1176): compact on the device first, then write at row = position - base.
        int rowBase = 0;
        if (_rows is not null && _rows.IsWindowed(layerIndex))
        {
            if (!contiguous)
                throw new InvalidOperationException("A compact sliding-window KV slot is written from contiguous positions only.");
            RecordWindowMove(cmdBuf, layerIndex, positions[0], maxPos);
            rowBase = _rows.BaseOf(layerIndex);
        }

        if (_storage.IsBlockQuantized())
        {
            // Quantised on the device, one invocation per 32-element block; rows are a multiple of 32 elements, so every append is whole blocks.
            if (q8Writer is null)
                throw new InvalidOperationException($"A {_storage} KV cache is written by the quantise kernel; pass the writer.");
            var vWriter = q8ValueWriter ?? q8Writer;
            if (q8Writer.Storage != _storage.KeyFormat() || vWriter.Storage != _storage.ValueFormat())
                throw new InvalidOperationException($"The {q8Writer.Storage} / {vWriter.Storage} writers cannot write a {_storage} KV cache (#1884).");
            if (!contiguous)
            {
                // Selective recompute / chunk-KV reuse (#1884): rows at arbitrary positions, one scatter dispatch per tensor.
                if (positionsDev is null)
                    throw new InvalidOperationException($"Writing a {_storage} KV cache at non-contiguous positions needs the device positions buffer.");
                q8Writer.RecordScatterRows(cmdBuf, kDev, _keys[layerIndex], positionsDev, _kvStride[layerIndex], seqLen, rowBase);
                vWriter.RecordScatterRows(cmdBuf, vDev, _values[layerIndex], positionsDev, _kvStride[layerIndex], seqLen, rowBase);
            }
            else
            {
                long elemOffset = (long)(positions[0] - rowBase) * _kvStride[layerIndex], count = (long)seqLen * _kvStride[layerIndex];
                q8Writer.RecordF32ToQ8(cmdBuf, kDev, _keys[layerIndex], elemOffset, count);
                vWriter.RecordF32ToQ8(cmdBuf, vDev, _values[layerIndex], elemOffset, count);
            }
            if (maxPos + 1 > _currentLength) _currentLength = maxPos + 1;
            return;
        }

        if (_storage == VulkanKvStorage.F16)
        {
            // The F32 rows the projections produced are converted on the device; there is no copy-engine path for F16.
            if (f16Writer is null)
                throw new InvalidOperationException("An F16 KV cache is written by the convert kernel; pass the writer.");
            if (!contiguous)
            {
                // Selective recompute (#1151): rows at arbitrary positions, one scatter dispatch per tensor (positions are a device buffer).
                if (positionsDev is null)
                    throw new InvalidOperationException("Writing an F16 KV cache at non-contiguous positions needs the device positions buffer.");
                f16Writer.RecordScatterRows(cmdBuf, kDev, _keys[layerIndex], positionsDev, _kvStride[layerIndex], seqLen, rowBase);
                f16Writer.RecordScatterRows(cmdBuf, vDev, _values[layerIndex], positionsDev, _kvStride[layerIndex], seqLen, rowBase);
            }
            else
            {
                long elemOffset = (long)(positions[0] - rowBase) * _kvStride[layerIndex], count = (long)seqLen * _kvStride[layerIndex];
                f16Writer.RecordF32ToF16(cmdBuf, kDev, _keys[layerIndex], elemOffset, count);
                f16Writer.RecordF32ToF16(cmdBuf, vDev, _values[layerIndex], elemOffset, count);
            }
            if (maxPos + 1 > _currentLength) _currentLength = maxPos + 1;
            return;
        }

        // Hazard-scoped barriers (issue #144): declare the append's access set
        // (reads the fresh K/V activations, writes the per-layer cache rows)
        // so the tracker emits the RoPE→copy barrier and the later
        // copy→attention barrier only when actually pending. One declaration
        // per buffer pair covers both the contiguous and per-row forms.
        _device.ActiveHazards?.OnTransfer(kDev.Handle, _keys[layerIndex].Handle);
        _device.ActiveHazards?.OnTransfer(vDev.Handle, _values[layerIndex].Handle);
        // The copies below read what RoPE (a compute shader) just wrote; a COMPUTE->COMPUTE barrier does not order a TRANSFER read (races on Intel Arc, #1249).
        if (_device.ActiveHazards is null) Kernels.KernelSupport.ComputeToTransferBarrier(cmdBuf);

        if (contiguous)
        {
            int startPos = positions[0] - rowBase;
            var region = new VkBufferCopy
            {
                srcOffset = 0,
                dstOffset = (ulong)((long)startPos * rowBytes),
                size = (ulong)rowBytes * (ulong)seqLen,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[layerIndex].Handle, 1, region);
            VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[layerIndex].Handle, 1, region);
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                var region = new VkBufferCopy
                {
                    srcOffset = (ulong)((long)i * rowBytes),
                    dstOffset = (ulong)((long)pos * rowBytes),
                    size = (ulong)rowBytes,
                };
                VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[layerIndex].Handle, 1, region);
                VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[layerIndex].Handle, 1, region);
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <summary>
    /// Validates <paramref name="positions"/> and advances <see cref="CurrentLength"/> as if
    /// <see cref="RecordUpdate"/> had run, WITHOUT recording any copy commands. Used by the
    /// fused RoPE+KV-write path (<c>RopeKvWriteF32Kernel</c>, issue #380), which writes cache
    /// rows directly via its own compute dispatch instead of going through <see cref="RecordUpdate"/>'s
    /// <c>vkCmdCopyBuffer</c> calls, but still needs the same length-tracking side effect.
    /// </summary>
    internal void AdvanceCurrentLength(ReadOnlySpan<int> positions, int seqLen)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));
        if (_rows is not null)
            throw new InvalidOperationException("The fused RoPE+KV write addresses rows by absolute position; a compact sliding-window cache cannot take it.");

        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    private int ValidateAndFindMaxPos(ReadOnlySpan<int> positions, int seqLen)
    {
        int maxPos = -1;
        for (int i = 0; i < seqLen; i++)
        {
            int pos = positions[i];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            if (pos > maxPos) maxPos = pos;
        }
        return maxPos;
    }

    private static bool IsContiguousAscending(ReadOnlySpan<int> positions)
    {
        for (int i = 1; i < positions.Length; i++)
        {
            if (positions[i] != positions[i - 1] + 1)
                return false;
        }
        return true;
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanKvCache is updated via UpdateDevice from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanKvCache is updated via UpdateDevice from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache exposes device buffers via GetKeysBuffer, not TensorRef.");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache exposes device buffers via GetValuesBuffer, not TensorRef.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache does not materialise cached keys as CPU tensors.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    /// <remarks>A compact sliding-window cache refuses a cut past its 64-position margin (<see cref="CanTruncateTo"/>).</remarks>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        Truncate(length);
    }

    /// <inheritdoc/>
    public bool CanRollbackAfterAppend(int minPos, int maxPos) =>
        _rows is null || _rows.CanTruncateToAfterAppend(_currentLength, minPos, maxPos);

    /// <summary>Resets the visible length. Used when starting a new sequence.</summary>
    public void Reset() => Truncate(0);

    /// <inheritdoc/>
    public int RowBaseOf(int layerIndex) => _rows?.BaseOf(layerIndex) ?? 0;

    /// <summary>True when some layers keep only their sliding-window rows (issue #1176).</summary>
    public bool HasCompactWindows => _rows is not null;

    /// <summary>
    /// True when the cache can be cut back to <paramref name="length"/>: always for a full cache; for compact sliding-window
    /// layers only while they still hold the rows a query at that position needs.
    /// </summary>
    /// <param name="length">The length to cut back to.</param>
    /// <returns>Whether <see cref="Rollback"/> / <see cref="SetCurrentLength"/> to that length succeeds.</returns>
    public bool CanTruncateTo(int length) => _rows is null || _rows.CanTruncateTo(length);

    /// <summary>
    /// Test hook (issue #1781): replaces the free-device-memory admission of <see cref="TryGrowTo"/>. Called with the device and the
    /// peak extra bytes the growth needs; returns whether it fits.
    /// </summary>
    internal static Func<VulkanDevice, long, bool>? GrowthAdmissionOverride { get; set; }

    /// <summary>Number of growths <see cref="TryGrowTo"/> carried out, process-wide (test hook).</summary>
    internal static long GrowthsForTest => Interlocked.Read(ref s_growths);
    private static long s_growths;

    /// <summary>
    /// Raises <see cref="MaxLength"/> to <paramref name="maxLength"/> keeping the <see cref="CurrentLength"/> committed rows (issue #1781),
    /// so a retained prefix-cache entry serves a next turn that outgrew it instead of re-prefilling the whole conversation. Layer by layer,
    /// larger K and V buffers are allocated, the live rows copied device to device in the cache's own format (no widening), and the old
    /// buffers freed: the transient cost is one old layer's K and V. The growth is admitted against free device memory before anything is
    /// allocated (never spill: on an integrated GPU an allocation past the OS limit succeeds and the next submit loses the device), and an
    /// out-of-memory error is caught as well; either refusal returns <see langword="false"/> with the cache unchanged. Must not run while
    /// a forward is recording. The freed handles are evicted from every descriptor cache by the device's destroy log (#467).
    /// A cache with compact sliding-window layers does not grow (its ring capacities are planned for the length it was made with).
    /// </summary>
    /// <param name="maxLength">Positions the cache must hold.</param>
    /// <returns><see langword="true"/> when <see cref="MaxLength"/> is now at least <paramref name="maxLength"/>.</returns>
    public bool TryGrowTo(int maxLength)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (maxLength <= _maxSeqLen) return true;
        if (_rows is not null) return false;
        if (_storage == VulkanKvStorage.Tq) return false;   // growth of codes + norms is step 4 of #1887; a refusal is a prefix miss, never a wider copy

        long growth = 0, transit = 0;
        for (int i = 0; i < _numLayers; i++)
        {
            long rowBytes = _storage.KeyRowBytes(_kvStride[i]) + _storage.ValueRowBytes(_kvStride[i]);   // K + V
            growth += (maxLength - _maxSeqLen) * rowBytes;
            transit = Math.Max(transit, _maxSeqLen * rowBytes);
        }
        if (!AdmitGrowth(_device, growth + transit))
            return false;

        int layer = 0;
        try
        {
            for (; layer < _numLayers; layer++)
            {
                long kRow = _storage.KeyRowBytes(_kvStride[layer]), vRow = _storage.ValueRowBytes(_kvStride[layer]);
                ulong keepK = (ulong)((long)_currentLength * kRow), keepV = (ulong)((long)_currentLength * vRow);
                var nk = _device.AllocateDeviceLocal((long)maxLength * kRow);
                VulkanDevice.Buffer nv;
                try { nv = _device.AllocateDeviceLocal((long)maxLength * vRow); }
                catch { nk.Dispose(); throw; }
                if (keepK > 0)
                {
                    try
                    {
                        _device.CopyBufferRangeSynchronous(_keys[layer], nk, 0, 0, keepK);
                        _device.CopyBufferRangeSynchronous(_values[layer], nv, 0, 0, keepV);
                    }
                    catch { nk.Dispose(); nv.Dispose(); throw; }
                }
                _keys[layer].Dispose();
                _values[layer].Dispose();
                _keys[layer] = nk;
                _values[layer] = nv;
            }
        }
        catch (VulkanException e) when (e.ErrorCode is VulkanNemotronHKvCache.VulkanOutOfDeviceMemory or VulkanNemotronHKvCache.VulkanOutOfHostMemory)
        {
            // Layers below `layer` already hold maxLength positions; the rest keep the old length, so the cache stays at the old one.
            return false;
        }

        _maxSeqLen = maxLength;
        RecordGrowth();
        return true;
    }

    /// <summary>Counts one growth for <see cref="GrowthsForTest"/> (shared with <see cref="VulkanQuantizedKvCache"/>, #1843).</summary>
    internal static void RecordGrowth() => Interlocked.Increment(ref s_growths);

    /// <summary>
    /// Admits a KV growth of <paramref name="bytes"/> peak extra device bytes on <paramref name="device"/> (shared with
    /// <see cref="VulkanQuantizedKvCache"/>, #1843); <see cref="GrowthAdmissionOverride"/> replaces it in tests.
    /// </summary>
    internal static bool AdmitGrowth(VulkanDevice device, long bytes)
    {
        if (GrowthAdmissionOverride is { } admit) return admit(device, bytes);
        if (string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT"), "1", StringComparison.Ordinal)) return true;
        try
        {
            // As the MTP-head attach (#880): what the device can hold resident (UMA: capped at the OS limit), minus what this process and
            // other processes already hold there.
            long usable = VulkanMemoryCapacity.UsableCapacityBytes(device.ResidentCapacityBytes(),
                GC.GetGCMemoryInfo().TotalAvailableMemoryBytes, device.PhysicalDeviceTypeValue);
            long others = device.ReadOtherProcessPressure()?.OtherBytes ?? 0;
            return bytes <= usable - device.ResidentLiveBytes() - others;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return false;   // no reading, no growth: the caller falls back to a miss
        }
    }

    /// <summary>
    /// Grows any compact sliding-window slot too small for an append of <paramref name="positions"/> (a prefill chunk longer
    /// than the ring's free half). Must run before the forward starts recording: a grown slot gets new buffer handles, so the
    /// caller drops its cached descriptor sets when this returns true. The kept rows move with a device-side copy.
    /// </summary>
    /// <param name="positions">Positions the next forward appends.</param>
    /// <returns>True when any buffer was replaced.</returns>
    public bool EnsureWindowCapacity(ReadOnlySpan<int> positions)
    {
        if (_rows is null || positions.Length == 0) return false;
        int maxPos = ValidateAndFindMaxPos(positions, positions.Length);
        int minPos = int.MaxValue;
        foreach (int p in positions) minPos = Math.Min(minPos, p);
        bool resized = false;
        for (int l = 0; l < _numLayers; l++)
        {
            if (!_rows.TryPlan(l, minPos, maxPos, _currentLength, out var m) || !m.Resize) continue;
            _keys[l] = Regrow(_keys[l], m, _storage.KeyRowBytes(_kvStride[l]));
            _values[l] = Regrow(_values[l], m, _storage.ValueRowBytes(_kvStride[l]));
            _rows.Commit(l, m);
            resized = true;
        }
        return resized;

        VulkanDevice.Buffer Regrow(VulkanDevice.Buffer old, WindowMove m, long rowBytes)
        {
            var grown = _device.AllocateDeviceLocal(m.Capacity * rowBytes);
            if (m.Keep > 0)
                _device.CopyBufferRangeSynchronous(old, grown, (ulong)(m.Shift * rowBytes), 0, (ulong)(m.Keep * rowBytes));
            old.Dispose();
            return grown;
        }
    }

    /// <summary>
    /// Records the compaction of a windowed slot when the append <c>[minPos, maxPos]</c> would overflow its ring: the kept rows
    /// move to row 0 on the device, directly when source and destination do not overlap, else staged through the move scratch
    /// (<c>vkCmdCopyBuffer</c> forbids overlapping regions).
    /// </summary>
    private void RecordWindowMove(nint cmdBuf, int layer, int minPos, int maxPos)
    {
        if (!_rows!.TryPlan(layer, minPos, maxPos, _currentLength, out var m)) return;
        if (m.Resize)
            throw new InvalidOperationException(
                $"Layer {layer}'s sliding-window ring is too small for positions {minPos}..{maxPos}; call {nameof(EnsureWindowCapacity)} before recording.");
        if (m.Shift > 0 && m.Keep > 0)
        {
            ulong kRow = (ulong)_storage.KeyRowBytes(_kvStride[layer]), vRow = (ulong)_storage.ValueRowBytes(_kvStride[layer]);
            ulong srcK = (ulong)m.Shift * kRow, bytesK = (ulong)m.Keep * kRow, srcV = (ulong)m.Shift * vRow, bytesV = (ulong)m.Keep * vRow;
            nint k = _keys[layer].Handle, v = _values[layer].Handle;
            if (m.Shift >= m.Keep)
            {
                CopyRange(cmdBuf, k, k, srcK, 0, bytesK, first: true);
                CopyRange(cmdBuf, v, v, srcV, 0, bytesV, first: false);
            }
            else
            {
                nint s = _moveScratch!.Handle;
                CopyRange(cmdBuf, k, s, srcK, 0, bytesK, first: true);   // K then V in the scratch
                CopyRange(cmdBuf, v, s, srcV, bytesK, bytesV, first: false);
                LegacyBarrier(cmdBuf);
                CopyRange(cmdBuf, s, k, 0, 0, bytesK, first: false);
                CopyRange(cmdBuf, s, v, bytesK, 0, bytesV, first: false);
            }

            LegacyBarrier(cmdBuf);
        }

        _rows.Commit(layer, m);
    }

    /// <summary>One <c>vkCmdCopyBuffer</c> region, declared to the hazard tracker (or preceded by a blanket barrier when untracked and <paramref name="first"/>).</summary>
    private void CopyRange(nint cmdBuf, nint src, nint dst, ulong srcOffset, ulong dstOffset, ulong size, bool first)
    {
        if (_device.ActiveHazards is { } hz) hz.OnTransfer(src, dst);
        else if (first) Kernels.KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src, dst, 1, region);
    }

    /// <summary>Blanket compute/transfer barrier on the untracked path; the hazard tracker inserts its own.</summary>
    private void LegacyBarrier(nint cmdBuf)
    {
        if (_device.ActiveHazards is null) Kernels.KernelSupport.ComputeTransferFullBarrier(cmdBuf);
    }

    private void Truncate(int length)
    {
        if (_rows is not null)
        {
            if (!_rows.CanTruncateTo(length))
                throw new InvalidOperationException(
                    $"Cannot cut the KV-cache back to {length}: a sliding-window layer already dropped rows that length needs "
                    + $"(margin {SlidingWindowRows.Margin} positions).");
            if (length == 0) _rows.Reset();
        }

        _currentLength = length;
    }

    private void ThrowIfWindowed(int layerIndex, string operation)
    {
        if (_rows is not null && _rows.IsWindowed(layerIndex))
            throw new NotSupportedException(
                $"VulkanKvCache.{operation}: layer {layerIndex} keeps only its sliding-window rows; build the cache with "
                + "KvGeometry.FromConfig(config) for full-history consumers.");
    }

    /// <summary>
    /// Number of layers in this cache.
    /// </summary>
    public int LayerCount => _numLayers;

    /// <summary>
    /// Per-row stride (<c>numKvHeads * headDim</c>, FP32 elements) of layer 0.
    /// For uniform caches this is the model-wide stride; for Gemma-4 (per-layer
    /// strides) use <see cref="KvStrideOf"/> to get a specific layer's width.
    /// </summary>
    public int KvStride => _kvStride[0];

    /// <summary>Per-row stride (FP32 elements) for the given layer.</summary>
    public int KvStrideOf(int layerIndex) => _kvStride[layerIndex];

    /// <summary>
    /// Ingests host-resident K/V rows (FP32, layout <c>[length, kvStride]</c>)
    /// for the given <paramref name="layerIndex"/> at positions <c>[0, length)</c>.
    /// Used by the hybrid CPU-prefill / iGPU-decode handoff: after CPU prefill
    /// has populated a <c>SimpleKvCache</c>, each layer's contiguous host buffer
    /// is uploaded into the device-local Vulkan KV cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Performs one host→staging map+copy plus one
    /// <c>vkCmdCopyBuffer</c> (staging→device) per call. The staging buffer is
    /// allocated and freed inside this method — handoff happens once per
    /// generation, never on the decode hot path. On a UMA APU (Strix Halo)
    /// the physical bytes never leave system DRAM; the driver does swizzle the
    /// layout in the device-local heap.
    /// </para>
    /// <para>
    /// Advances <see cref="CurrentLength"/> to <c>max(CurrentLength, length)</c>
    /// so the subsequent device decode sees positions <c>[0, length)</c> as
    /// already-cached. Both <c>keys</c> and <c>values</c> must cover exactly
    /// <c>length × KvStride</c> FP32 elements.
    /// </para>
    /// </remarks>
    public unsafe void IngestFromHost(int layerIndex, int length,
        ReadOnlySpan<float> keys, ReadOnlySpan<float> values)
    {
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        RequireNotTq("IngestFromHost");
        ThrowIfWindowed(layerIndex, "IngestFromHost");
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "length must be positive.");
        if (length > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length),
                $"length {length} exceeds cache MaxLength {_maxSeqLen}.");
        long expectedFloats = (long)length * _kvStride[layerIndex];
        if (keys.Length != expectedFloats || values.Length != expectedFloats)
            throw new ArgumentException(
                $"keys/values must contain exactly length × kvStride = {expectedFloats} floats; "
                + $"got keys={keys.Length}, values={values.Length}.");

        if (_storage == VulkanKvStorage.F16)
        {
            // F16 cache (#1151): the host rows stay F32 on the wire; round to nearest-even on write (the WriteRows staging path).
            WriteRows(layerIndex, 0, length, keys, values);
            return;
        }

        long bytes = expectedFloats * sizeof(float);

        // Upload keys then values. Use the device's host-visible Allocate as a
        // one-shot staging buffer per call, then synchronous copy → device-local
        // KV destination. The synchronous variant is fine here: handoff is
        // off the per-token hot path.
        using (var stagingK = _device.Allocate(bytes))
        {
            MapAndCopy(stagingK, keys);
            _device.CopyBufferRangeSynchronous(stagingK, _keys[layerIndex],
                srcOffset: 0, dstOffset: 0, size: (ulong)bytes);
        }

        using (var stagingV = _device.Allocate(bytes))
        {
            MapAndCopy(stagingV, values);
            _device.CopyBufferRangeSynchronous(stagingV, _values[layerIndex],
                srcOffset: 0, dstOffset: 0, size: (ulong)bytes);
        }

        if (length > _currentLength)
            _currentLength = length;
    }

    /// <summary>
    /// Sets the visible length without changing buffer contents. Used after
    /// <see cref="IngestFromHost"/> calls for every layer to advance the
    /// observed length atomically across layers (the per-layer call already
    /// advances individually; this is a no-op for single-layer ingest but
    /// makes the multi-layer code path explicit at the call site).
    /// </summary>
    public void SetCurrentLength(int length)
    {
        if ((uint)length > (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length));
        Truncate(length);
    }

    // ── IHostStagedKvCache: device↔host staging for cross-device KV handoff ──
    // The two halves of a device→host→device transfer: DownloadLayer reads this (prefill-device) cache's
    // K/V to host; the decode-device cache's UploadLayer (== IngestFromHost) writes it back to its device.
    // This is the production transport for a two-GPU prefill→decode handoff and is exercised by the
    // DisaggregatedScheduler's StagedKvHandoffTransfer.

    /// <inheritdoc/>
    public int StagedLayerElementCount(int layerIndex)
    {
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _currentLength * _kvStride[layerIndex];
    }

    /// <inheritdoc/>
    public void DownloadLayer(int layerIndex, Span<float> keys, Span<float> values)
    {
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        RequireNotTq("DownloadLayer");
        ThrowIfWindowed(layerIndex, "DownloadLayer");
        int count = _currentLength * _kvStride[layerIndex];
        if (count == 0) return;
        if (keys.Length < count || values.Length < count)
            throw new ArgumentException($"keys/values must hold at least currentLength × kvStride = {count} floats.");
        if (_storage == VulkanKvStorage.F16)
        {
            ReadRows(layerIndex, 0, _currentLength, keys[..count], values[..count]);   // widened to F32 on the host (#1151)
            return;
        }
        // Device-local K/V → host. On a dGPU this is a real VRAM read-back; on UMA the bytes never leave DRAM.
        _device.Download(_keys[layerIndex], keys[..count]);
        _device.Download(_values[layerIndex], values[..count]);
    }

    /// <inheritdoc/>
    public void UploadLayer(int layerIndex, int length, ReadOnlySpan<float> keys, ReadOnlySpan<float> values)
        => IngestFromHost(layerIndex, length, keys, values);

    // shortcut: the host-staged F32 transport (handoff, ingest) has no F16 arm yet; add one when an F16 model needs cross-device handoff.
    private void RequireNotQ8(string operation)
    {
        if (_storage.IsBlockQuantized())
            throw new NotSupportedException($"VulkanKvCache {operation} has no {_storage} arm (prefix snapshots, handoff and host row access need F16 or F32 KV); the Q8_0 cache is read in place by the attention kernels only.");
    }

    // TurboQuant rows are codes + norms in code space: no host element transport, handoff or row copy exists for them yet (#1887 step 4).
    private void RequireNotTq(string operation)
    {
        if (_storage == VulkanKvStorage.Tq)
            throw new NotSupportedException($"VulkanKvCache.{operation} has no TurboQuant arm: the cache holds codes and norms, read in place by the attention kernels only.");
    }

    // The synchronous test/diagnostic UpdateDevice moves F32 rows only; the host-staged transports (handoff, ingest) convert for F16 (#1151).
    private void RequireF32(string operation)
    {
        if (_storage != VulkanKvStorage.F32)
            throw new NotSupportedException($"VulkanKvCache.{operation} moves F32 rows; this cache stores {_storage}.");
    }

    private unsafe void MapAndCopy(VulkanDevice.Buffer staging, ReadOnlySpan<float> source)
    {
        // 64-bit: source.Length is an element count, so `length * 4` wraps int at
        // 512 M floats (2 GiB) — the same element-count × element-size overflow
        // fixed on the CPU side in #429. A per-layer KV slab of that size is only
        // out of reach on drivers that cap maxMemoryAllocationSize at 2 GiB.
        long byteLen = (long)source.Length * sizeof(float);
        nint mapped = _device.MapMemoryWithRetry(staging.Memory, 0, (ulong)byteLen, "vkMapMemory IngestFromHost staging");
        try
        {
            fixed (float* src = source)
            {
                System.Buffer.MemoryCopy(src, (void*)mapped, byteLen, byteLen);
            }
        }
        finally
        {
            Interop.VulkanApi.vkUnmapMemory(_device.Handle, staging.Memory);
        }
    }

    // ── IPositionedHostKvCache: row-range host IO + device-to-device copy (chunk-level KV reuse, #1531) ──

    /// <inheritdoc/>
    public void WriteRows(int layerIndex, int startPosition, int count, ReadOnlySpan<float> keys, ReadOnlySpan<float> values)
    {
        long elems = CheckRowRange(layerIndex, startPosition, count, "WriteRows");
        if (keys.Length != elems || values.Length != elems)
            throw new ArgumentException($"keys/values must contain exactly count × kvStride = {elems} floats; got keys={keys.Length}, values={values.Length}.");
        long bytes = elems * _elemBytes;
        long dstOffset = (long)startPosition * _kvStride[layerIndex] * _elemBytes;
        StageUpload(keys, _keys[layerIndex], bytes, dstOffset);
        StageUpload(values, _values[layerIndex], bytes, dstOffset);
        if (startPosition + count > _currentLength)
            _currentLength = startPosition + count;
    }

    /// <inheritdoc/>
    public void ReadRows(int layerIndex, int startPosition, int count, Span<float> keys, Span<float> values)
    {
        long elems = CheckRowRange(layerIndex, startPosition, count, "ReadRows");
        if (startPosition + count > _currentLength)
            throw new ArgumentOutOfRangeException(nameof(count), $"rows [{startPosition}, {startPosition + count}) exceed CurrentLength {_currentLength}.");
        if (keys.Length < elems || values.Length < elems)
            throw new ArgumentException($"keys/values must hold at least count × kvStride = {elems} floats.");
        long bytes = elems * _elemBytes;
        long srcOffset = (long)startPosition * _kvStride[layerIndex] * _elemBytes;
        StageDownload(_keys[layerIndex], srcOffset, bytes, keys[..(int)elems]);
        StageDownload(_values[layerIndex], srcOffset, bytes, values[..(int)elems]);
    }

    /// <summary>
    /// Device-to-device copy of <paramref name="count"/> rows of layer <paramref name="layerIndex"/> from
    /// <paramref name="source"/> positions <c>[srcStart, srcStart+count)</c> to this cache's
    /// <c>[dstStart, dstStart+count)</c>, with no host hop. Both caches must share the device, element format and layer stride.
    /// Advances <see cref="CurrentLength"/> to cover the written rows. Bytes are copied verbatim, so K keeps the
    /// RoPE phase of its source position (re-rotate afterwards if <c>dstStart != srcStart</c>).
    /// </summary>
    public void CopyRowsFrom(VulkanKvCache source, int layerIndex, int srcStart, int dstStart, int count)
    {
        ArgumentNullException.ThrowIfNull(source);
        RequireNotTq("CopyRowsFrom");
        if (!ReferenceEquals(source._device, _device)) throw new ArgumentException("Caches must share one VulkanDevice.", nameof(source));
        if (source._storage != _storage) throw new ArgumentException("Caches must share the element format.", nameof(source));
        long elems = CheckRowRange(layerIndex, dstStart, count, "CopyRowsFrom");
        source.CheckRowRange(layerIndex, srcStart, count, "CopyRowsFrom");
        if (source._kvStride[layerIndex] != _kvStride[layerIndex]) throw new ArgumentException("Layer strides differ.", nameof(source));
        if (srcStart + count > source._currentLength)
            throw new ArgumentOutOfRangeException(nameof(count), $"source rows [{srcStart}, {srcStart + count}) exceed its CurrentLength {source._currentLength}.");
        ulong bytes = (ulong)(elems * _elemBytes);
        long rowBytes = (long)_kvStride[layerIndex] * _elemBytes;
        _device.CopyBufferRangeSynchronous(source._keys[layerIndex], _keys[layerIndex], (ulong)(srcStart * rowBytes), (ulong)(dstStart * rowBytes), bytes);
        _device.CopyBufferRangeSynchronous(source._values[layerIndex], _values[layerIndex], (ulong)(srcStart * rowBytes), (ulong)(dstStart * rowBytes), bytes);
        if (dstStart + count > _currentLength)
            _currentLength = dstStart + count;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Issue #1780. Needs another full-history <see cref="VulkanKvCache"/> on the same device with the same element format and per-layer row
    /// widths. Rows are copied device to device in that format (F16 stays F16, Q8_0 blocks stay blocks): the cost is
    /// <c>rows x (K + V row bytes)</c> summed over layers, with no extra allocation. Must not run while a forward is recording.
    /// </remarks>
    public bool TrySeedPrefixFrom(IKvCache source, int rows)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source is not VulkanKvCache src || ReferenceEquals(src, this) || src._disposed) return false;
        if (!ReferenceEquals(src._device, _device) || src._storage != _storage || src._numLayers != _numLayers) return false;
        if (_rows is not null || src._rows is not null) return false;
        if (_storage == VulkanKvStorage.Tq) return false;   // seeding needs the seed / centroid fingerprint check: step 4 of #1887
        if (rows <= 0 || rows > src._currentLength || rows > _maxSeqLen) return false;
        for (int i = 0; i < _numLayers; i++)
            if (src._kvStride[i] != _kvStride[i]) return false;
        // No allocation: the rows land in this cache's own buffers (the miss's allocation, admitted when it was made). A failed submit
        // leaves CurrentLength as it was, so no partly copied row is ever claimed.
        try
        {
            for (int i = 0; i < _numLayers; i++)
            {
                // Row bytes, not _elemBytes (block rows have no element size), per side (mixed formats, #1884).
                _device.CopyBufferRangeSynchronous(src._keys[i], _keys[i], 0, 0, (ulong)((long)rows * _storage.KeyRowBytes(_kvStride[i])));
                _device.CopyBufferRangeSynchronous(src._values[i], _values[i], 0, 0, (ulong)((long)rows * _storage.ValueRowBytes(_kvStride[i])));
            }
        }
        catch (VulkanException e) when (e.ErrorCode is VulkanNemotronHKvCache.VulkanOutOfDeviceMemory or VulkanNemotronHKvCache.VulkanOutOfHostMemory)
        {
            return false;
        }
        _currentLength = rows;
        return true;
    }

    private long CheckRowRange(int layerIndex, int startPosition, int count, string operation)
    {
        if ((uint)layerIndex >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        RequireNotTq(operation);
        ThrowIfWindowed(layerIndex, operation);
        if (startPosition < 0) throw new ArgumentOutOfRangeException(nameof(startPosition));
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count), "count must be positive.");
        if ((long)startPosition + count > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(count), $"rows [{startPosition}, {startPosition + count}) exceed cache MaxLength {_maxSeqLen}.");
        return (long)count * _kvStride[layerIndex];
    }

    // Host F32 → (F32 | F16) staging → device range. One-shot staging per call: chunk placement is off the per-token hot path.
    private unsafe void StageUpload(ReadOnlySpan<float> source, VulkanDevice.Buffer dst, long bytes, long dstOffset)
    {
        RequireNotQ8("host row upload");
        using var staging = _device.Allocate(bytes);
        nint mapped = _device.MapMemoryWithRetry(staging.Memory, 0, (ulong)bytes, "vkMapMemory WriteRows staging");
        try
        {
            if (_storage == VulkanKvStorage.F16)
            {
                // Saturate at the F16 range exactly as the device writers do (convert_f32_to_f16_*): an infinity stored in a K/V row would poison a whole softmax row.
                var halves = new Span<Half>((void*)mapped, source.Length);
                TensorPrimitives.ConvertToHalf(source, halves);
                for (int i = 0; i < halves.Length; i++)
                    if (Half.IsInfinity(halves[i]) && !float.IsInfinity(source[i]))
                        halves[i] = source[i] > 0 ? Half.MaxValue : Half.MinValue;
            }
            else
                source.CopyTo(new Span<float>((void*)mapped, source.Length));
        }
        finally
        {
            Interop.VulkanApi.vkUnmapMemory(_device.Handle, staging.Memory);
        }
        _device.CopyBufferRangeSynchronous(staging, dst, 0, (ulong)dstOffset, (ulong)bytes);
    }

    private unsafe void StageDownload(VulkanDevice.Buffer src, long srcOffset, long bytes, Span<float> destination)
    {
        RequireNotQ8("host row download");
        // Host-cached memory: the default host-visible type is write-combined (uncached) on AMD, and CPU reads from it ran at ~130 MiB/s (#1551).
        using var staging = _device.AllocateHostReadback(bytes);
        _device.CopyBufferRangeSynchronous(src, staging, (ulong)srcOffset, 0, (ulong)bytes);
        nint mapped = _device.MapMemoryWithRetry(staging.Memory, 0, (ulong)bytes, "vkMapMemory ReadRows staging");
        try
        {
            if (_storage == VulkanKvStorage.F16)
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>((void*)mapped, destination.Length), destination);
            else
                new ReadOnlySpan<float>((void*)mapped, destination.Length).CopyTo(destination);
        }
        finally
        {
            Interop.VulkanApi.vkUnmapMemory(_device.Handle, staging.Memory);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            _keys[i]?.Dispose();
            _values[i]?.Dispose();
            _keyNorms?[i]?.Dispose();
            _valueNorms?[i]?.Dispose();
        }

        _tqCentroids?.Dispose();
        _tqSignsK?.Dispose();
        _tqSignsV?.Dispose();
        _moveScratch?.Dispose();
    }
}
