using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Decode at depth through a CUDA graph for every dense KV type (issue #1922 E2, ticket #1918).
/// </summary>
/// <remarks>
/// <para>
/// <b>Capture, don't patch</b> (the mechanism the hybrid model ships, #1222). The eager <see cref="ForwardInternal"/> body is
/// captured whole for this one step, pushed into the executable graph with <c>cuGraphExecUpdate</c> (re-instantiated only when
/// the topology changes, e.g. a different attention kernel at a new depth) and launched once. Every per-step argument —
/// <c>seqKv</c>, kvSplit, KV row offsets — is captured fresh, so no kernel needs a device-side position and the launch sequence
/// is the eager one: output is bit-identical to eager by construction.
/// </para>
/// <para>
/// <b>Where it runs.</b> A single-token dense decode the older graphs do not take: an F16 <see cref="CudaKvCache"/> at
/// <c>position &gt;= GraphCaptureMaxDepth</c> (the shallow graph's <c>attention_f16_dyn</c> is slower there), a pure-quantized
/// <see cref="CudaQuantizedKvCache"/> (no FP16 window) at any depth, and the in-place TurboQuant cache at any depth.
/// MLA, MoE (own graph), LoRA, paged KV, profiling, debug captures and host-gathered embeddings stay eager.
/// </para>
/// <para>
/// <b>Why re-capture is safe.</b> The eager decode body allocates nothing on these caches (KV growth is explicit,
/// <c>TryGrowTo</c>, outside the forward; the forward state is grown before the capture), and its only host-side effect is the
/// KV length, which it <i>sets</i> from the position rather than increments. So a capture that fails part-way leaves nothing
/// to undo: the step simply runs eagerly and sets the same length. That is why the windowed quantized cache (whose update
/// advances eviction bookkeeping) is not admitted. A failed capture, instantiate or launch runs that step eagerly and disables
/// the graph for that configuration only. <c>DOTLLM_CUDA_STEP_GRAPH=0</c> turns the path off.
/// </para>
/// </remarks>
public sealed unsafe partial class CudaTransformerModel
{
    /// <summary>Kill switch for the step graph (default on). Mutable so tests can A/B in one process.</summary>
    internal static bool StepGraphEnabled = Environment.GetEnvironmentVariable("DOTLLM_CUDA_STEP_GRAPH") != "0";

    private bool _stepGraphCapturing;
    private nint _stepGraphExec;

    // A failed capture / instantiate / launch disables the step graph only for the configuration it failed on (#1922 review M4):
    // the same KV cache, the same cache buffers (generation) and the same attention tier (dense v2 engaged or not). The next
    // change of any of those retries it, so one bad request cannot cost a long-lived server the graph forever.
    private (IKvCache Cache, int Generation, bool DenseTier)? _stepGraphFailedOn;

    /// <summary>Test hook (#1922 review I2): 1 = fail the next instantiate, 2 = fail the next launch, as if the driver had.</summary>
    internal static int StepGraphInjectFailureForTest;

    /// <summary>Step-graph launches, (re)instantiations and capture failures (tests and benches read these).</summary>
    internal int StepGraphLaunches { get; private set; }
    internal int StepGraphInstantiations { get; private set; }
    internal int StepGraphFallbacks { get; private set; }

    // Host time spent capturing + updating/instantiating (not launching) the step graph; printed at dispose when
    // DOTLLM_CUDA_STEP_GRAPH_STATS=1, so a bench can tell whether the serial re-capture is the remaining cost.
    private long _stepGraphHostTicks;
    private static readonly bool s_stepGraphStats = Environment.GetEnvironmentVariable("DOTLLM_CUDA_STEP_GRAPH_STATS") == "1";

    private bool StepGraphEligible(int rows, bool suppressGraphCapture, nint deferredHostDest, bool isMla, bool isMoe, IKvCache? kvCache,
        int position)
    {
        if (!StepGraphEnabled || !UseGraphCapture || suppressGraphCapture || deferredHostDest != 0
            || rows != 1 || isMla || isMoe || ProfilingEnabled || _currentAdapter is not null || DebugCaptureAttnLayer >= 0
            || ResidualModifier is not null   // #1919: a graph replay never reaches the per-layer PostLayer residual-modifier call
            || _weights.TokenEmbedHost != 0)
            return false;
        bool admitted = kvCache switch
        {
            CudaKvCache => true,   // reached only when the shallow f16 graph declined (depth cap, NoPE layers)
            CudaQuantizedKvCache q => StepGraphAdmitsQuantized(q.ReferenceDequantScratch, q.WindowCapacity),
            CudaTurboQuantInPlaceKvCache => true,
            _ => false,
        };
        return admitted && !(_stepGraphFailedOn is { } f && f.Equals(StepGraphKey(kvCache!, position)));
    }

    /// <summary>
    /// Which quantized caches the step graph may capture (#1922 review I1): pure quantized only. The windowed cache's eager update
    /// advances host-side eviction bookkeeping (<c>_layerQuantizedLength</c>) while it is being captured, so a capture that failed
    /// part-way would make the eager rerun skip an eviction and corrupt a row; the test-only dequant-scratch oracle stays eager.
    /// Pure so it is testable without a GPU.
    /// </summary>
    internal static bool StepGraphAdmitsQuantized(bool referenceDequantScratch, int windowCapacity)
        => !referenceDequantScratch && windowCapacity == 0;

    private static (IKvCache, int, bool) StepGraphKey(IKvCache cache, int position)
    {
        int gen = cache switch { CudaKvCache c => c.Generation, CudaQuantizedKvCache q => q.Generation, _ => 0 };
        return (cache, gen, CudaDenseDecodeV2.Enabled && position + 1 >= CudaDenseDecodeV2.MinSeqKv);
    }

    private ITensor? StepGraphFallback(IKvCache kvCache, int position, string what)
    {
        StepGraphFallbacks++;
        bool first = _stepGraphFailedOn is null;
        _stepGraphFailedOn = StepGraphKey(kvCache, position);
        if (first)
            Console.Error.WriteLine($"[dotLLM.Cuda] Step decode graph {what}; this step and this configuration run eagerly.");
        return null;
    }

    /// <summary>
    /// Captures this decode step's eager body, updates or instantiates the step graph and launches it. Returns the logits, or
    /// null (nothing enqueued) when the capture failed, in which case the caller runs the eager body.
    /// </summary>
    private ITensor? ForwardStepGraph(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache kvCache)
    {
        _context.MakeCurrent();
        _state.EnsureCapacity(1);
        nint s = _stream.Handle;

        // The eager body uploads these from pageable host memory, which a capture cannot record: upload them now.
        int tok = tokenIds[0], pos = positions[0];
        CudaDriverApi.cuMemcpyHtoD_v2(_state.TokenIdsDevice, (nint)(&tok), sizeof(int)).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)(&pos), sizeof(int)).ThrowOnError();

        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        nint graph = 0;
        Exception? error = null;
        CudaDriverApi.cuStreamBeginCapture_v2(s, CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
        _stepGraphCapturing = true;
        try
        {
            ForwardInternal(tokenIds, positions, deviceId, kvCache, suppressGraphCapture: false, deferredHostDest: 0);
        }
        catch (Exception e)
        {
            error = e;
        }
        finally
        {
            _stepGraphCapturing = false;
        }
        int rc = CudaDriverApi.cuStreamEndCapture(s, out graph);
        try
        {
            if (error is not null || rc != 0)
                return StepGraphFallback(kvCache, pos, $"capture failed ({error?.Message ?? $"CUresult {rc}"})");

            if (_stepGraphExec != 0)
            {
                byte* info = stackalloc byte[32];   // CUgraphExecUpdateResultInfo
                if (CudaDriverApi.cuGraphExecUpdate(_stepGraphExec, graph, (nint)info) != 0)
                {
                    CudaDriverApi.cuGraphExecDestroy(_stepGraphExec);
                    _stepGraphExec = 0;
                }
            }
            if (_stepGraphExec == 0)
            {
                // #1922 review I2: an instantiate failure (e.g. out of memory at a topology change) is a fallback, not an
                // exception out of Forward. The capture's host-side effects are idempotent for every admitted cache.
                int irc = StepGraphInjectFailureForTest == 1 ? 2 /* CUDA_ERROR_OUT_OF_MEMORY */
                    : CudaDriverApi.cuGraphInstantiateWithFlags(out _stepGraphExec, graph, 0);
                if (StepGraphInjectFailureForTest == 1) StepGraphInjectFailureForTest = 0;
                if (irc != 0)
                {
                    _stepGraphExec = 0;
                    return StepGraphFallback(kvCache, pos, $"instantiate failed (CUresult {irc})");
                }
                StepGraphInstantiations++;
            }
        }
        finally
        {
            if (graph != 0) CudaDriverApi.cuGraphDestroy(graph);
        }

        _stepGraphHostTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
        int lrc = StepGraphInjectFailureForTest == 2 ? 2 : CudaDriverApi.cuGraphLaunch(_stepGraphExec, s);
        if (StepGraphInjectFailureForTest == 2) StepGraphInjectFailureForTest = 0;
        if (lrc != 0)
        {
            CudaDriverApi.cuGraphExecDestroy(_stepGraphExec);
            _stepGraphExec = 0;
            return StepGraphFallback(kvCache, pos, $"launch failed (CUresult {lrc})");
        }
        StepGraphLaunches++;
        GraphReplayCount++;
        DecodeGraphState = CudaDecodeGraphState.Replayed;
        _stream.Synchronize();

        var result = UnmanagedTensor.Allocate(new TensorShape(1, Config.VocabSize), DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.LogitsF32, (nuint)(Config.VocabSize * sizeof(float))).ThrowOnError();
        return result;
    }

    private void DisposeStepGraph()
    {
        if (s_stepGraphStats && StepGraphLaunches > 0)
            Console.Error.WriteLine($"[dotLLM.Cuda] step graph: launches={StepGraphLaunches} instantiations={StepGraphInstantiations} " +
                $"fallbacks={StepGraphFallbacks} capture+update={_stepGraphHostTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency / StepGraphLaunches:F3} ms/step");
        if (_stepGraphExec != 0) { CudaDriverApi.cuGraphExecDestroy(_stepGraphExec); _stepGraphExec = 0; }
    }
}
