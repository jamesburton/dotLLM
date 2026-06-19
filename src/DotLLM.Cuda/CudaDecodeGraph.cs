using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Steady-state status of the single-token decode CUDA graph, for diagnostics/CLI visibility.
/// Reflects the most recent decode step.
/// </summary>
public enum CudaDecodeGraphState
{
    /// <summary>No decode step has run yet (e.g. only prefill so far).</summary>
    None,
    /// <summary>Graph capture disabled via <c>DOTLLM_CUDA_GRAPH=0</c>; raw kernel launches used.</summary>
    Off,
    /// <summary>Graph enabled but this configuration is ineligible (non-BitNet, multi-token, debug flags, or unsupported KV-cache).</summary>
    Ineligible,
    /// <summary>Graph enabled and eligible, but capture failed; fell back to raw kernel launches.</summary>
    Fallback,
    /// <summary>The graph was (re)captured on this step.</summary>
    Captured,
    /// <summary>A captured graph was replayed (the steady-state fast path).</summary>
    Replayed,
}

internal sealed class CudaDecodeGraph : IDisposable
{
    private nint _graphExec;
    private bool _capturing;

    internal bool IsCaptured => _graphExec != 0;

    internal void Begin(nint stream)
    {
        CudaDriverApi.cuStreamBeginCapture_v2(stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
        _capturing = true;
    }

    internal bool TryEnd(nint stream)
    {
        nint graph = 0;
        int result = CudaDriverApi.cuStreamEndCapture(stream, out graph);
        _capturing = false;
        if (result != 0)
            return false;

        try
        {
            result = CudaDriverApi.cuGraphInstantiateWithFlags(out _graphExec, graph, 0);
            return result == 0;
        }
        finally
        {
            if (graph != 0)
                CudaDriverApi.cuGraphDestroy(graph);
        }
    }

    internal void Abort(nint stream)
    {
        if (!_capturing)
            return;

        _capturing = false;
        CudaDriverApi.cuStreamEndCapture(stream, out nint graph);
        if (graph != 0)
            CudaDriverApi.cuGraphDestroy(graph);
    }

    internal void Launch(nint stream)
        => CudaDriverApi.cuGraphLaunch(_graphExec, stream).ThrowOnError();

    public void Dispose()
    {
        nint graphExec = _graphExec;
        _graphExec = 0;
        if (graphExec != 0)
            CudaDriverApi.cuGraphExecDestroy(graphExec);
    }
}
