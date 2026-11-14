using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>Exclusive F16/Q8_0 KV pages allocated as a sequence grows, with device pointer tables.</summary>
public sealed unsafe class HipPagedKvCache : IKvCache
{
    /// <summary>Set to exactly <c>1</c> to acknowledge experimental Q8 KV precision loss.</summary>
    public const string Q8OptInEnvironmentVariable = "DOTLLM_HIP_ALLOW_Q8_KV";

    /// <summary>Returns the Q8 acknowledgement error, or null for F16 or an explicit opt-in.</summary>
    public static string? ExperimentalQ8Error(KvCacheDType keyType, KvCacheDType valueType)
        => (keyType == KvCacheDType.Q8_0 || valueType == KvCacheDType.Q8_0)
            && Environment.GetEnvironmentVariable(Q8OptInEnvironmentVariable) != "1"
            ? $"HIP Q8_0 KV is experimental: measured precision loss failed the original 1K–8K logit quality gates. "
                + $"Set {Q8OptInEnvironmentVariable}=1 to acknowledge this loss, or use the default F16 KV. See docs/PRECISION.md."
            : null;

    private readonly HipContext _context;
    private readonly List<HipDeviceBuffer> _pages = [];
    private readonly int _layers;
    private readonly long _keyPlaneBytes, _layerPageBytes;
    private readonly int _contextLimit;
    private HipDeviceBuffer? _tables;
    private int _tableCapacity;
    private bool _disposed;
    internal object SyncRoot { get; } = new();
    internal HipContext Owner => _context;

    /// <inheritdoc/>
    public int CurrentLength { get; private set; }
    /// <inheritdoc/>
    public int MaxLength { get; private set; }
    /// <summary>Tokens held in each physical page.</summary>
    public int PageTokens { get; }
    /// <summary>Key storage format; F32 denotes unquantized F16 device storage.</summary>
    public KvCacheDType KeyType { get; }
    /// <summary>Value storage format; F32 denotes unquantized F16 device storage.</summary>
    public KvCacheDType ValueType { get; }
    /// <summary>Device containing these pages.</summary>
    public int DeviceId => _context.DeviceId;
    /// <summary>Live allocation bytes, including the pointer tables.</summary>
    public long AllocatedBytes { get { lock (SyncRoot) return (_tables?.ByteLength ?? 0) + _pages.Sum(b => b.ByteLength); } }
    /// <summary>Number of physically allocated pages per layer.</summary>
    public int AllocatedPages => _pages.Count;

    internal HipPagedKvCache(HipContext context, int layers, int kvDimension, int maxLength, int contextLimit,
        int pageTokens = 16, KvCacheDType keyType = KvCacheDType.F32, KvCacheDType valueType = KvCacheDType.F32)
    {
        if (ExperimentalQ8Error(keyType, valueType) is { } error) throw new NotSupportedException(error);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDimension);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(pageTokens);
        if (maxLength > contextLimit) throw new ArgumentOutOfRangeException(nameof(maxLength));
        _context = context;
        _layers = layers;
        _contextLimit = contextLimit;
        MaxLength = maxLength;
        PageTokens = pageTokens;
        KeyType = keyType; ValueType = valueType;
        _keyPlaneBytes = checked((long)pageTokens * RowBytes(keyType));
        _layerPageBytes = checked(_keyPlaneBytes + (long)pageTokens * RowBytes(valueType));

        long RowBytes(KvCacheDType type) => type switch
        {
            KvCacheDType.F32 => checked((long)kvDimension * sizeof(ushort)),
            KvCacheDType.Q8_0 when kvDimension % 32 == 0 => checked((long)kvDimension / 32 * 34),
            _ => throw new NotSupportedException("HIP KV supports F16 and Q8_0 with 32-element blocks."),
        };
    }

    internal nint KeyTable(int layer) => _tables!.Pointer + checked((nint)((long)layer * 2 * _tableCapacity * sizeof(nint)));
    internal nint ValueTable(int layer) => KeyTable(layer) + checked(_tableCapacity * sizeof(nint));

    internal void EnsureNextPosition(int position, HipStream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (position != CurrentLength || (uint)position >= (uint)MaxLength)
            throw new ArgumentException($"HIP KV positions must append at {CurrentLength}, below capacity {MaxLength}.", nameof(position));
        int pageIndex = position / PageTokens;
        if (pageIndex < AllocatedPages) return;

        stream.Synchronize();
        HipDeviceBuffer? added = null;
        HipDeviceBuffer? newTables = null;
        int capacity = _tableCapacity == 0 ? 1 : _tableCapacity;
        if (pageIndex >= capacity) capacity = checked(capacity * 2);
        try
        {
            _pages.EnsureCapacity(pageIndex + 1);
            // One allocation covers every layer, so page growth needs only one payload allocation.
            added = new HipDeviceBuffer(_context, checked(_layerPageBytes * _layers));
            if (capacity != _tableCapacity)
            {
                // Pointer metadata only; tensor data remains on the GPU.
                var pointers = new nint[checked(2 * _layers * capacity)];
                for (int layer = 0; layer < _layers; layer++)
                    for (int page = 0; page <= pageIndex; page++)
                    {
                        nint pointer = (page == pageIndex ? added.Pointer : _pages[page].Pointer)
                            + checked((nint)(layer * _layerPageBytes));
                        pointers[layer * 2 * capacity + page] = pointer;
                        pointers[(layer * 2 + 1) * capacity + page] = pointer + checked((nint)_keyPlaneBytes);
                    }
                newTables = new HipDeviceBuffer(_context, checked((long)pointers.Length * sizeof(nint)));
                fixed (nint* p = pointers)
                    HipDriverApi.hipMemcpyHtoD(newTables.Pointer, (nint)p, (nuint)newTables.ByteLength).ThrowOnError();
            }
            else
            {
                for (int layer = 0; layer < _layers; layer++)
                {
                    nint key = added.Pointer + checked((nint)(layer * _layerPageBytes));
                    nint value = key + checked((nint)_keyPlaneBytes);
                    HipDriverApi.hipMemcpyHtoD(KeyTable(layer) + pageIndex * sizeof(nint), (nint)(&key), (nuint)sizeof(nint)).ThrowOnError();
                    HipDriverApi.hipMemcpyHtoD(ValueTable(layer) + pageIndex * sizeof(nint), (nint)(&value), (nuint)sizeof(nint)).ThrowOnError();
                }
            }
            _pages.Add(added);
            added = null;
            if (newTables is not null)
            {
                var oldTables = _tables;
                _tables = newTables;
                newTables = null;
                _tableCapacity = capacity;
                oldTables?.Dispose();
            }
        }
        finally
        {
            newTables?.Dispose();
            added?.Dispose();
        }
    }

    internal void CommitPosition() => CurrentLength++;

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        lock (SyncRoot)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (length < 0 || length > CurrentLength) throw new ArgumentOutOfRangeException(nameof(length));
            _context.MakeCurrent();
            HipDriverApi.hipDeviceSynchronize().ThrowOnError();
            int keep = length == 0 ? 0 : (length - 1) / PageTokens + 1;
            while (_pages.Count > keep)
            {
                _pages[^1].Dispose();
                _pages.RemoveAt(_pages.Count - 1);
            }
            if (keep == 0)
            {
                _tables?.Dispose();
                _tables = null;
                _tableCapacity = 0;
            }
            CurrentLength = length;
        }
    }

    /// <inheritdoc/>
    public bool TryGrowTo(int maxLength)
    {
        lock (SyncRoot)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (maxLength <= 0 || maxLength > _contextLimit) return false;
            MaxLength = Math.Max(MaxLength, maxLength);
            return true;
        }
    }

    private static NotSupportedException DeviceOnly() => new("HIP paged KV is device-resident. Update it through HipTransformerModel; contiguous host tensor access is unsupported.");
    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) => throw DeviceOnly();
    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) => throw DeviceOnly();
    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) => throw DeviceOnly();
    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) => throw DeviceOnly();
    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) => throw DeviceOnly();
    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) => throw DeviceOnly();

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (SyncRoot)
        {
            if (_disposed) return;
            HipDriverApi.hipSetDevice(DeviceId).ThrowOnError();
            HipDriverApi.hipDeviceSynchronize().ThrowOnError();
            foreach (var page in _pages) page.Dispose();
            _pages.Clear();
            _tables?.Dispose();
            _tables = null;
            CurrentLength = 0;
            _disposed = true;
        }
    }
}
