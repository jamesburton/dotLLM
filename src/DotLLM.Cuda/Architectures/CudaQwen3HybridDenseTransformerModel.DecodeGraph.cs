using DotLLM.Core.Attention;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Single-token decode through a CUDA graph (issue #1222). The 64-layer trunk is ~2,000 kernel launches per token; on
/// WDDM each eager <c>cuLaunchKernel</c> costs more host time than the average kernel runs, so plain decode on the
/// RTX 3060 was launch-bound (GPU busy ~27 ms of ~50 ms per token).
/// </summary>
/// <remarks>
/// <para>
/// <b>Capture, don't patch.</b> The layer loop is captured into a graph and pushed into an executable graph with
/// <c>cuGraphExecUpdate</c> (or re-instantiated when the topology changes, e.g. a different attention tier at a new
/// depth), then the whole trunk is one <c>cuGraphLaunch</c>. Every per-step argument — KV row pointers, <c>seqKv</c>,
/// scratch that grew — is simply captured fresh, so no kernel needs a device-side position. The launch sequence is the
/// eager one, so output is bit-identical.
/// </para>
/// <para>
/// <b>Capture ahead.</b> Capturing ~2,000 launches still costs ~4-5 ms of host time, plus ~1.5 ms to update. So after
/// step <c>t</c>'s graph is launched, step <c>t+1</c> (position + 1) is captured into the OTHER of two executable graphs
/// while the GPU runs step <c>t</c>. The next call launches it directly if nothing it baked in changed: same position,
/// KV cache, KV length, and no device allocation or free since (<see cref="DeviceAllocEpoch"/>). The only host state the
/// trunk mutates is the KV length; the pre-capture saves it, restores it, and the hit replays the captured value.
/// </para>
/// <para>
/// Nothing may allocate or free inside a capture. The known growable buffers are grown before it (state, KV); any
/// other allocation trips <see cref="ThrowIfCapturing"/>, the capture is abandoned before anything ran, and the step
/// runs eagerly (which performs the growth). <c>DOTLLM_CUDA_HYBRID_GRAPH=0</c> disables the path.
/// </para>
/// </remarks>
public sealed unsafe partial class CudaQwen3HybridDenseTransformerModel
{
    /// <summary>Environment switch for the decode graph; <c>0</c> keeps eager launches.</summary>
    internal const string DecodeGraphEnvVar = "DOTLLM_CUDA_HYBRID_GRAPH";

    private static readonly bool s_decodeGraphEnabled = Environment.GetEnvironmentVariable(DecodeGraphEnvVar) != "0";

    /// <summary>
    /// Bumped by every device allocation or free of this host and its forward state, so a graph captured ahead of time
    /// can tell that a pointer it baked in may be gone. Process-wide on purpose: a false bump only costs a re-capture.
    /// </summary>
    internal static int DeviceAllocEpoch;

    // Set while this thread is capturing the decode trunk; checked by AllocDevice / FreeIfNonZero.
    [ThreadStatic]
    private static bool t_capturingDecode;

    /// <summary>Widest forward (token rows) run through a graph: decode and MTP / speculative verifies.</summary>
    internal const int MaxGraphRows = 8;

    // [0], [1]: one-token steps (double-buffered for the capture-ahead); [rows]: a rows-wide verify (2..8). Separate
    // executables per width, so alternating decode and verify forwards update in place instead of re-instantiating.
    private readonly nint[] _decodeGraphExec = new nint[MaxGraphRows + 1];
    private int _decodeGraphSlot;          // the exec launched last
    private bool _decodeGraphBroken;       // a capture failed for a reason other than growth: stay eager

    // The step captured ahead into exec[1 - _decodeGraphSlot], valid while the key below still matches.
    private bool _aheadValid;
    private int _aheadPosition, _aheadLenBefore, _aheadLenAfter, _aheadEpoch;
    private IKvCache? _aheadKvCache;

    private int _decodeGraphLaunches, _decodeGraphAheadHits, _decodeGraphInstantiations, _decodeGraphFallbacks;

    /// <summary>In-process override of <see cref="DecodeGraphEnvVar"/> (tests only).</summary>
    internal static bool? DecodeGraphOverride { get; set; }

    /// <summary>Test hook: graph launches, how many of them were captured ahead, instantiations, and eager fallbacks.</summary>
    internal (int Launches, int AheadHits, int Instantiations, int EagerFallbacks) DecodeGraphCounts =>
        (_decodeGraphLaunches, _decodeGraphAheadHits, _decodeGraphInstantiations, _decodeGraphFallbacks);

    private sealed class CaptureAllocationException : Exception
    {
    }

    /// <summary>Counts a device allocation or free, and aborts a decode-graph capture before it happens inside one.</summary>
    private static void ThrowIfCapturing()
    {
        DeviceAllocEpoch++;
        if (t_capturingDecode)
            throw new CaptureAllocationException();
    }

    private bool DecodeGraphEligible =>
        (DecodeGraphOverride ?? s_decodeGraphEnabled) && !_decodeGraphBroken
        && !ProfileTrace && !DebugTrace && !DotLLM.Models.Architectures.TensorDump.Enabled;

    /// <summary>
    /// Runs the layer loop of a one-token forward as one graph launch. Returns <see langword="false"/> — with nothing
    /// enqueued — when the graph is off, ineligible, or the capture could not complete, so the caller runs it eagerly.
    /// </summary>
    private bool TryForwardDecodeGraph(ReadOnlySpan<int> positions, int hiddenSize, int numHeads, int numKvHeads,
        int headDim, float eps, IKvCache? kvCache)
    {
        if (!DecodeGraphEligible)
        {
            _aheadValid = false;
            return false;
        }

        // Grow what the attention layers would otherwise grow mid-capture.
        if (kvCache is not null && _attentionLayerCount > 0)
            EnsureF16KvCache(kvCache.MaxLength, numKvHeads, headDim);

        int rows = positions.Length;
        int slot;
        if (rows > 1)
        {
            slot = rows;
            if (!CaptureTrunkInto(slot, positions, hiddenSize, numHeads, numKvHeads, headDim, eps, kvCache))
            {
                _aheadValid = false;
                _decodeGraphFallbacks++;
                return false;
            }
        }
        else if (_aheadValid && _aheadPosition == positions[0] && ReferenceEquals(_aheadKvCache, kvCache)
            && ReferenceEquals(_lastForwardKvCache, kvCache) && _aheadLenBefore == _f16CacheCurrentLength
            && _aheadEpoch == DeviceAllocEpoch)
        {
            slot = 1 - _decodeGraphSlot;
            _f16CacheCurrentLength = _aheadLenAfter;   // the trunk's one host-side effect, replayed
            _decodeGraphAheadHits++;
        }
        else
        {
            slot = _decodeGraphSlot;
            if (!CaptureTrunkInto(slot, positions, hiddenSize, numHeads, numKvHeads, headDim, eps, kvCache))
            {
                _aheadValid = false;
                _decodeGraphFallbacks++;
                return false;
            }
        }

        _aheadValid = false;
        CudaDriverApi.cuGraphLaunch(_decodeGraphExec[slot], _stream.Handle).ThrowOnError();
        if (rows == 1) _decodeGraphSlot = slot;
        _decodeGraphLaunches++;
        return true;
    }

    /// <summary>
    /// Captures the step at <c>position + 1</c> into the idle executable graph while the GPU runs the current one. Host
    /// state the capture touches is restored; the next <see cref="TryForwardDecodeGraph"/> launches it if still valid.
    /// </summary>
    private void CaptureNextDecodeStepAhead(int position, int hiddenSize, int numHeads, int numKvHeads, int headDim,
        float eps, IKvCache? kvCache)
    {
        _aheadValid = false;
        if (!DecodeGraphEligible || (kvCache is not null && position + 1 >= kvCache.MaxLength))
            return;

        int lenBefore = _f16CacheCurrentLength;
        int epoch = DeviceAllocEpoch;
        Span<int> next = [position + 1];
        bool ok = CaptureTrunkInto(1 - _decodeGraphSlot, next, hiddenSize, numHeads, numKvHeads, headDim, eps, kvCache);
        int lenAfter = _f16CacheCurrentLength;
        _f16CacheCurrentLength = lenBefore;
        if (!ok || epoch != DeviceAllocEpoch)
            return;

        _aheadValid = true;
        _aheadPosition = position + 1;
        _aheadLenBefore = lenBefore;
        _aheadLenAfter = lenAfter;
        _aheadEpoch = epoch;
        _aheadKvCache = kvCache;
    }

    /// <summary>Captures the 64-layer trunk for <c>positions.Length</c> tokens and commits it to executable graph <paramref name="slot"/>.</summary>
    private bool CaptureTrunkInto(int slot, ReadOnlySpan<int> positions, int hiddenSize, int numHeads, int numKvHeads,
        int headDim, float eps, IKvCache? kvCache)
    {
        nint s = _stream.Handle;
        nint graph = 0;
        bool captured = false;
        CudaDriverApi.cuStreamBeginCapture_v2(s, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED).ThrowOnError();
        t_capturingDecode = true;
        try
        {
            for (int layer = 0; layer < _layers.Length; layer++)
                RunSingleLayerBody(layer, positions.Length, positions, hiddenSize, numHeads, numKvHeads, headDim, eps, kvCache);
            captured = true;
        }
        catch (CaptureAllocationException)
        {
            // Growth needed: the caller runs this step eagerly, which performs it.
        }
        catch (CudaException ex)
        {
            _decodeGraphBroken = true;
            Console.Error.WriteLine($"[dotLLM.Cuda] Hybrid decode graph disabled after a capture error: {ex.Message}");
        }
        finally
        {
            t_capturingDecode = false;
            if (CudaDriverApi.cuStreamEndCapture(s, out graph) != 0) captured = false;
        }

        try
        {
            if (!captured) return false;

            ref nint exec = ref _decodeGraphExec[slot];
            if (exec != 0)
            {
                byte* info = stackalloc byte[32];   // CUgraphExecUpdateResultInfo
                if (CudaDriverApi.cuGraphExecUpdate(exec, graph, (nint)info) != 0)
                {
                    CudaDriverApi.cuGraphExecDestroy(exec);
                    exec = 0;
                }
            }
            if (exec == 0)
            {
                CudaDriverApi.cuGraphInstantiateWithFlags(out exec, graph, 0).ThrowOnError();
                _decodeGraphInstantiations++;
            }
            return true;
        }
        finally
        {
            if (graph != 0) CudaDriverApi.cuGraphDestroy(graph);
        }
    }

    private void DisposeDecodeGraph()
    {
        for (int i = 0; i < _decodeGraphExec.Length; i++)
        {
            if (_decodeGraphExec[i] != 0)
            {
                CudaDriverApi.cuGraphExecDestroy(_decodeGraphExec[i]);
                _decodeGraphExec[i] = 0;
            }
        }
    }
}
