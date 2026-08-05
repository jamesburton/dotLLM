using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident pool of fixed-size KV blocks (issue #252). Mirrors
/// <see cref="DotLLM.Engine.KvCache.KvBlockPool"/>'s shape — per-layer storage of
/// <c>[totalBlocks, blockSize, kvStride]</c>, a free-list allocator, and per-block reference
/// counts for copy-on-write / prefix sharing — but block storage lives in device memory
/// (FP16, matching <see cref="CudaKvCache"/>'s on-device element type) instead of host
/// <c>NativeMemory</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Deliberate divergence from the CPU pool</b>: this pool is built around <see cref="KvGeometry"/>
/// (per-layer KV row width) rather than CPU's single scalar <c>numKvHeads * headDim</c>, matching
/// how <see cref="CudaKvCache"/> and <see cref="CudaQuantizedKvCache"/> already generalize for
/// Gemma-4's non-uniform sliding/global layer shapes. This costs nothing for the uniform case
/// (every non-Gemma-4 architecture) — <see cref="KvGeometry.IsUniform"/> collapses to the same
/// scalar-stride arithmetic — and avoids blocking a future Gemma-4 CUDA decode path on a
/// paged-cache rewrite. Block IDs remain layer-independent (a block ID indexes the same logical
/// block across every layer); only the per-layer byte offset within that block's storage differs.
/// </para>
/// <para>
/// <b>Allocation bookkeeping stays host-side</b>, exactly like the CPU pool (a C# <c>lock</c> +
/// <see cref="Interlocked"/> refcounts over plain host arrays) — block allocate/free/addref happens
/// at most once per <see cref="BlockSize"/> decode tokens, not on the per-token hot path, so there
/// is no reason to pay a GPU round-trip for it. Only the block <em>storage</em> and copy-on-write
/// duplication (<see cref="CopyBlock"/>) touch the device.
/// </para>
/// </remarks>
public sealed class CudaKvBlockPool : IDisposable
{
    private readonly int _blockSize;
    private readonly int _numLayers;
    private readonly KvGeometry _geom;
    private readonly int _totalBlocks;
    private readonly long[] _blockBytesPerLayer; // per-layer bytes for one block (blockSize * strideOf(layer) * sizeof(ushort))

    // Per-layer contiguous device storage. Block i for layer j starts at
    // _keyBuffers[j] + i * _blockBytesPerLayer[j].
    private readonly nint[] _keyBuffers;
    private readonly nint[] _valueBuffers;

    // Free list (stack-based) — identical shape/semantics to KvBlockPool (CPU reference).
    private readonly int[] _freeStack;
    private int _freeCount;
    private readonly object _lock = new();

    // Reference counting per block (CoW / prefix sharing).
    private readonly int[] _refCounts;

    private readonly CudaContext? _context;
    private bool _disposed;

    /// <summary>
    /// The CUDA context this pool's device memory was allocated under (issue #268) — exposed so
    /// dependent objects allocated later against the same device (e.g. <see cref="CudaPagedKvCache"/>)
    /// can call <see cref="CudaContext.MakeCurrent"/> before their own allocation calls, rather than
    /// silently relying on whatever context happens to already be current on the calling thread.
    /// </summary>
    internal CudaContext? Context => _context;

    /// <summary>Number of tokens per block.</summary>
    public int BlockSize => _blockSize;

    /// <summary>Total number of blocks in the pool.</summary>
    public int TotalBlocks => _totalBlocks;

    /// <summary>Number of currently free blocks.</summary>
    public int FreeBlocks
    {
        get { lock (_lock) return _freeCount; }
    }

    /// <summary>Number of transformer layers.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Per-layer KV row width (FP16 elements) for <paramref name="layerIndex"/>.</summary>
    public int KvStrideOf(int layerIndex) => _geom.KvStrideOf(layerIndex);

    /// <summary>Total device bytes reserved by this pool across key and value storage, all layers.</summary>
    public long AllocatedBytes
    {
        get
        {
            long total = 0;
            for (int i = 0; i < _numLayers; i++)
                total += 2L * _totalBlocks * _blockBytesPerLayer[i];
            return total;
        }
    }

    /// <summary>
    /// Creates a new GPU-resident block pool with a single uniform <c>numKvHeads * headDim</c>
    /// stride. Byte-identical to the per-layer constructor with <see cref="KvGeometry.Uniform"/>.
    /// </summary>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numKvHeads">Number of KV attention heads per layer.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="blockSize">Number of tokens per block (default: 16, matches the CPU pool).</param>
    /// <param name="totalBlocks">Total number of blocks in the pool.</param>
    /// <param name="context">Optional CUDA context for explicit multi-device placement; null uses the current context.</param>
    public CudaKvBlockPool(int numLayers, int numKvHeads, int headDim,
                            int blockSize = 16, int totalBlocks = 4096, CudaContext? context = null)
        : this(KvGeometry.Uniform(numLayers, numKvHeads, headDim), blockSize, totalBlocks, context)
    {
    }

    /// <summary>
    /// Creates a new GPU-resident block pool from a Core <see cref="KvGeometry"/> descriptor.
    /// Uniform for every dense/GQA/MoE model; per-layer for Gemma-4 (see remarks on the type).
    /// </summary>
    public CudaKvBlockPool(KvGeometry geometry, int blockSize = 16, int totalBlocks = 4096, CudaContext? context = null)
    {
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (totalBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(totalBlocks));

        _blockSize = blockSize;
        _numLayers = geometry.LayerCount;
        _geom = geometry;
        _totalBlocks = totalBlocks;
        _context = context;

        _keyBuffers = new nint[_numLayers];
        _valueBuffers = new nint[_numLayers];
        _blockBytesPerLayer = new long[_numLayers];

        _context?.MakeCurrent();

        for (int i = 0; i < _numLayers; i++)
        {
            long blockBytes = (long)blockSize * geometry.KvStrideOf(i) * sizeof(ushort); // FP16
            _blockBytesPerLayer[i] = blockBytes;
            nuint layerBytes = (nuint)(blockBytes * totalBlocks);
            CudaDriverApi.cuMemAlloc_v2(out _keyBuffers[i], layerBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _valueBuffers[i], layerBytes).ThrowOnError();
        }

        // Initialize free stack (all blocks free, LIFO order) — identical to the CPU pool.
        _freeStack = new int[totalBlocks];
        for (int i = 0; i < totalBlocks; i++)
            _freeStack[i] = totalBlocks - 1 - i; // top of stack = block 0
        _freeCount = totalBlocks;

        _refCounts = new int[totalBlocks];
    }

    /// <summary>
    /// Allocates a block from the free pool. Sets ref count to 1.
    /// </summary>
    /// <returns>Block ID.</returns>
    /// <exception cref="InvalidOperationException">No free blocks available.</exception>
    public int Allocate()
    {
        lock (_lock)
        {
            if (_freeCount == 0)
                throw new InvalidOperationException("CudaKvBlockPool exhausted: no free blocks available.");

            int blockId = _freeStack[--_freeCount];
            _refCounts[blockId] = 1;
            return blockId;
        }
    }

    /// <summary>
    /// Increments the reference count for a block (used for shared prefix / beam search).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddRef(int blockId)
    {
        ValidateBlockId(blockId);
        int newCount = Interlocked.Increment(ref _refCounts[blockId]);
        if (newCount <= 1)
        {
            Interlocked.Decrement(ref _refCounts[blockId]);
            throw new InvalidOperationException($"Cannot add a reference to free block {blockId}.");
        }
    }

    /// <summary>
    /// Gets the current reference count for a block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int RefCount(int blockId)
    {
        ValidateBlockId(blockId);
        return Volatile.Read(ref _refCounts[blockId]);
    }

    /// <summary>
    /// Decrements the reference count. When it reaches 0, returns the block to the free pool.
    /// </summary>
    public void Release(int blockId)
    {
        ValidateBlockId(blockId);
        int newCount = Interlocked.Decrement(ref _refCounts[blockId]);
        if (newCount < 0)
        {
            Interlocked.Increment(ref _refCounts[blockId]);
            throw new InvalidOperationException($"Cannot release free block {blockId}.");
        }
        if (newCount == 0)
        {
            lock (_lock)
            {
                _freeStack[_freeCount++] = blockId;
            }
        }
    }

    /// <summary>
    /// Allocates a new block and copies all layer data from the source block (copy-on-write),
    /// via device-to-device async copies on <paramref name="stream"/>.
    /// </summary>
    /// <returns>New block ID with independent data.</returns>
    public int CopyBlock(int sourceBlockId, nint stream)
    {
        ValidateBlockId(sourceBlockId);
        if (RefCount(sourceBlockId) <= 0)
            throw new InvalidOperationException($"Cannot copy free block {sourceBlockId}.");

        int newBlockId = Allocate();

        for (int layer = 0; layer < _numLayers; layer++)
        {
            long blockBytes = _blockBytesPerLayer[layer];
            nint kSrc = _keyBuffers[layer] + (nint)(sourceBlockId * blockBytes);
            nint kDst = _keyBuffers[layer] + (nint)(newBlockId * blockBytes);
            nint vSrc = _valueBuffers[layer] + (nint)(sourceBlockId * blockBytes);
            nint vDst = _valueBuffers[layer] + (nint)(newBlockId * blockBytes);

            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)blockBytes, stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)blockBytes, stream).ThrowOnError();
        }

        return newBlockId;
    }

    /// <summary>
    /// Gets the device pointer to the start of key data for a specific block and layer.
    /// Points to <c>blockSize * KvStrideOf(layerIndex)</c> FP16 elements.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetKeyPtr(int blockId, int layerIndex)
    {
        ValidateBlockId(blockId);
        ValidateLayerIndex(layerIndex);
        return _keyBuffers[layerIndex] + (nint)(blockId * _blockBytesPerLayer[layerIndex]);
    }

    /// <summary>
    /// Gets the device pointer to the start of value data for a specific block and layer.
    /// Points to <c>blockSize * KvStrideOf(layerIndex)</c> FP16 elements.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetValuePtr(int blockId, int layerIndex)
    {
        ValidateBlockId(blockId);
        ValidateLayerIndex(layerIndex);
        return _valueBuffers[layerIndex] + (nint)(blockId * _blockBytesPerLayer[layerIndex]);
    }

    private void ValidateBlockId(int blockId)
    {
        if ((uint)blockId >= (uint)_totalBlocks)
            throw new ArgumentOutOfRangeException(nameof(blockId));
    }

    private void ValidateLayerIndex(int layerIndex)
    {
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _context?.MakeCurrent();
        for (int i = 0; i < _numLayers; i++)
        {
            if (_keyBuffers[i] != 0) { CudaDriverApi.cuMemFree_v2(_keyBuffers[i]); _keyBuffers[i] = 0; }
            if (_valueBuffers[i] != 0) { CudaDriverApi.cuMemFree_v2(_valueBuffers[i]); _valueBuffers[i] = 0; }
        }
    }
}
