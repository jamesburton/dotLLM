using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

internal sealed class CudaDecodeGraph : IDisposable
{
    private const int StreamCaptureModeThreadLocal = 1;

    private nint _graphExec;
    private bool _capturing;

    internal bool IsCaptured => _graphExec != 0;

    internal void Begin(nint stream)
    {
        CudaDriverApi.cuStreamBeginCapture_v2(stream, StreamCaptureModeThreadLocal).ThrowOnError();
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
