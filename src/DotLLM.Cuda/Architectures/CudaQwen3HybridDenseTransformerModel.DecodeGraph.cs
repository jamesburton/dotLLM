using DotLLM.Core.Attention;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Single-token decode through a CUDA graph (issue #1222). The 64-layer trunk is ~2,000 kernel launches per token; on
/// WDDM each eager <c>cuLaunchKernel</c> costs more host time than the average kernel runs, so plain decode on the
/// RTX 3060 was launch-bound (GPU busy ~28 ms of ~50 ms per token).
/// </summary>
/// <remarks>
/// <para>
/// Every token the layer loop is <b>re-captured</b> into a graph (capture records launches without submitting them, which
/// is far cheaper than eager submission) and pushed into one executable graph with <c>cuGraphExecUpdate</c>; the whole
/// trunk is then a single <c>cuGraphLaunch</c>. Re-capturing is what keeps this correct without device-side positions:
/// every argument that changes per step (KV row pointers, <c>seqKv</c>, the attention tier picked for the depth, scratch
/// that grew) is simply captured fresh. When the topology changes (a different attention tier) the update fails and the
/// graph is re-instantiated. The math is the eager launch sequence exactly, so output is bit-identical.
/// </para>
/// <para>
/// Nothing may allocate or free inside a capture. The known growable buffers are grown before it (state, logits, KV);
/// any other allocation trips <see cref="ThrowIfCapturing"/>, the capture is abandoned before anything ran, and the step
/// runs eagerly (which performs the growth). <c>DOTLLM_CUDA_HYBRID_GRAPH=0</c> disables the path.
/// </para>
/// </remarks>
public sealed unsafe partial class CudaQwen3HybridDenseTransformerModel
{
    /// <summary>Environment switch for the decode graph; <c>0</c> keeps eager launches.</summary>
    internal const string DecodeGraphEnvVar = "DOTLLM_CUDA_HYBRID_GRAPH";

    private static readonly bool s_decodeGraphEnabled = Environment.GetEnvironmentVariable(DecodeGraphEnvVar) != "0";

    // Set while this thread is capturing the decode trunk; checked by AllocDevice / FreeIfNonZero.
    [ThreadStatic]
    private static bool t_capturingDecode;

    private nint _decodeGraphExec;
    private bool _decodeGraphBroken;   // a capture failed for a reason other than growth: stay eager

    /// <summary>In-process override of <see cref="DecodeGraphEnvVar"/> (tests only).</summary>
    internal static bool? DecodeGraphOverride { get; set; }

    /// <summary>Test hook: decode steps run through the graph, and how many of them (re)instantiated it.</summary>
    internal (int Launches, int Instantiations, int EagerFallbacks) DecodeGraphCounts =>
        (_decodeGraphLaunches, _decodeGraphInstantiations, _decodeGraphFallbacks);

    private int _decodeGraphLaunches, _decodeGraphInstantiations, _decodeGraphFallbacks;

    private sealed class CaptureAllocationException : Exception
    {
    }

    /// <summary>Aborts a decode-graph capture before a device allocation or free happens inside it.</summary>
    private static void ThrowIfCapturing()
    {
        if (t_capturingDecode)
            throw new CaptureAllocationException();
    }

    /// <summary>
    /// Runs the layer loop of a one-token forward as one graph launch. Returns <see langword="false"/> — with nothing
    /// enqueued — when the graph is off, ineligible, or the capture could not complete, so the caller runs the loop eagerly.
    /// </summary>
    private bool TryForwardDecodeGraph(ReadOnlySpan<int> positions, int hiddenSize, int numHeads, int numKvHeads,
        int headDim, float eps, IKvCache? kvCache)
    {
        if (!(DecodeGraphOverride ?? s_decodeGraphEnabled) || _decodeGraphBroken
            || ProfileTrace || DebugTrace || DotLLM.Models.Architectures.TensorDump.Enabled
            || _rowSnapshotRequestRows > 0)
            return false;

        // Grow what the attention layers would otherwise grow mid-capture.
        if (kvCache is not null && _attentionLayerCount > 0)
            EnsureF16KvCache(kvCache.MaxLength, numKvHeads, headDim);

        nint s = _stream.Handle;
        nint graph = 0;
        bool captured = false;
        CudaDriverApi.cuStreamBeginCapture_v2(s, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED).ThrowOnError();
        t_capturingDecode = true;
        try
        {
            for (int layer = 0; layer < _layers.Length; layer++)
                RunSingleLayerBody(layer, 1, positions, hiddenSize, numHeads, numKvHeads, headDim, eps, kvCache);
            captured = true;
        }
        catch (CaptureAllocationException)
        {
            // Growth needed this step: fall back to eager below, which performs it. Next step captures again.
        }
        catch (CudaException ex)
        {
            _decodeGraphBroken = true;
            Console.Error.WriteLine($"[dotLLM.Cuda] Hybrid decode graph disabled after a capture error: {ex.Message}");
        }
        finally
        {
            t_capturingDecode = false;
            int end = CudaDriverApi.cuStreamEndCapture(s, out graph);
            if (end != 0) captured = false;
        }

        try
        {
            if (!captured)
            {
                _decodeGraphFallbacks++;
                return false;
            }

            if (_decodeGraphExec != 0)
            {
                byte* info = stackalloc byte[32];
                if (CudaDriverApi.cuGraphExecUpdate(_decodeGraphExec, graph, (nint)info) != 0)
                {
                    CudaDriverApi.cuGraphExecDestroy(_decodeGraphExec);
                    _decodeGraphExec = 0;
                }
            }
            if (_decodeGraphExec == 0)
            {
                CudaDriverApi.cuGraphInstantiateWithFlags(out _decodeGraphExec, graph, 0).ThrowOnError();
                _decodeGraphInstantiations++;
            }
            CudaDriverApi.cuGraphLaunch(_decodeGraphExec, s).ThrowOnError();
            _decodeGraphLaunches++;
            return true;
        }
        finally
        {
            if (graph != 0) CudaDriverApi.cuGraphDestroy(graph);
        }
    }

    private void DisposeDecodeGraph()
    {
        if (_decodeGraphExec != 0)
        {
            CudaDriverApi.cuGraphExecDestroy(_decodeGraphExec);
            _decodeGraphExec = 0;
        }
    }
}
