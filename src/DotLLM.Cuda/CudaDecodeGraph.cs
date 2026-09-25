using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

// ═══════════════════════════════════════════════════════════════════════════
// Qwen3HybridDense (Bonsai-27B, PQ2_0) decode-graph investigation (#161
// candidate #6, 2026-07-23) — PHASE 1 STOPPED HERE, NO-GO for implementation
// this round. This class (and CudaTransformerModel's ForwardDecodeGraph /
// CaptureDecodeGraph) is the PROVEN, WORKING pattern for the dense F16 path.
// This note records why the SAME pattern does NOT carry over to
// CudaQwen3HybridDenseTransformerModel (Architectures/) without substantial
// new work, so a future attempt doesn't have to re-derive this from scratch.
//
// ── What "device-position-aware" means in the WORKING F16 path (verified) ──
// CudaTransformerModel.ForwardDecodeGraph's own doc comment (and the SASS-
// level reasoning behind it) confirms the mechanism this investigation's
// task brief predicted: TWO device-resident int scalars, `_decodePosDevice`
// and `_decodeSeqKvDevice`, are bumped via a tiny `cuMemcpyHtoD_v2` BEFORE
// each `cuGraphLaunch` (outside the graph). `LaunchAttentionDyn`
// (`attention_f16_dyn`) reads `seq_kv`/`position_offset` from those buffers
// INSIDE the kernel rather than taking them as `cuLaunchKernel` args, so the
// captured graph node's baked-in pointer VALUES stay constant across replays
// while the VALUES THEY POINT TO change every step. `CudaKvCache.
// UpdateDeviceSingleDevicePos` does the equivalent for the KV-cache write: a
// `kv_write_one_f16` kernel computes `dst = base + posPtr[0]*stride`
// device-side, instead of the eager path's host-computed `cuMemcpyDtoDAsync`
// destination pointer (which WOULD be baked stale into a graph node).
//
// ── Why Qwen3HybridDense's decode step does NOT already fit this pattern ──
// Read in full for this investigation: CudaQwen3HybridDenseTransformerModel.
// Forward → RunSingleLayerBody → ForwardGdnBody / ForwardFullAttnBody /
// ForwardDenseFfnBody (src/DotLLM.Cuda/Architectures/). Per-decode-step
// kernel-launch census, classified per the task's (a)/(b)/(c) buckets:
//
//   (a) Pure function of current-token activations, safe as-is: dense FFN
//       body (RmsNorm/copy-RmsNorm, GateUp GEMV or TryFusedPQ2_0Gemm2,
//       SwiGLU, Down GEMV — all read/write the FIXED per-token scratch
//       buffers in CudaQwen3HybridDenseForwardState, which are allocated
//       ONCE and never reallocated across seqLen==1 decode calls, since
//       EnsureCapacity(1) is a no-op after the first call). Also the GDN
//       projections (Qkv/Gate/Alpha/Beta) and de-interleave/L2-norm steps.
//       PQ2_0 decode GEMVs are F32-native as of this issue's own candidate
//       #1 (LaunchPQ2_0GemvF32Native / LaunchPQ2_0Gemv2F32Native) — no
//       _activF16InScratch/_activF16OutScratch round-trip on this path at
//       all for a PQ2_0 model, so that historical growable-scratch hazard
//       (relevant to the I2_S/other-quant Gemm branches) does not apply
//       here. Host-side per-token embedding lookup (CPU mmap dequant of
//       the ONE new token's row, H2D into the stable HiddenState buffer)
//       is structurally identical to the working path's "host inputs
//       uploaded before the graph launch, landing in stable graph-baked
//       buffers" discipline — fine to keep outside the graph as-is.
//
//   (c) Persistent cross-step state, GDN recurrence — SAFE, already matches
//       the correct pattern: CudaGdnStateCache allocates its conv_state/
//       gdn_state buffers ONCE at construction (fixed size, no grow/
//       realloc path — see CudaGdnStateCache.cs), and GetConvStatePtr/
//       GetGdnStatePtr are pure pointer arithmetic into that stable
//       allocation. LaunchGdnScanStepF32 reads-then-writes the state
//       buffer in place; its other kernel args (nVHead/nKHead/dState) are
//       model-config constants, invariant across decode steps. This is
//       the ONE piece of the decode step that would graph-replay correctly
//       today without any new kernel work.
//
//   (b)/(c) Persistent cross-step state, KV-cache — THREE CONFIRMED HAZARDS,
//       all in ForwardFullAttnBody / WriteF16KvRows (the 16 full-attention
//       layers only; does not affect the 48 GDN layers):
//
//       1. [HIGH, immediate, deterministic] WriteF16KvRows computes the
//          cache-row destination pointer on the HOST from `positions[0]`
//          (`kDst = kBase + positions[0]*rowBytes`) and passes it straight
//          to `cuMemcpyDtoDAsync_v2`. This is EXACTLY the eager-vs-graph gap
//          CudaTransformerModel's own doc comment warns about for the F16
//          path (which is why that path was rebuilt around
//          UpdateDeviceSingleDevicePos / kv_write_one_f16). Naively
//          captured, every replay after the first would silently
//          overwrite the SAME cache row — the KV cache would stop
//          advancing at step 2 and every later step attends over a
//          corrupted/frozen history. No device-position-aware KV-write
//          kernel exists for this model's private F16 K/V cache
//          (`_f16KCache`/`_f16VCache`) today.
//
//       2. [HIGH, immediate, deterministic] `LaunchAttentionF32` (attention_
//          f32.ptx) takes `seqKv` and `positionOffset` as plain host `int`
//          kernel-launch arguments (see CudaKernels.LaunchAttentionF32) —
//          there is no `attention_f32_dyn` counterpart to `attention_f16_
//          dyn` that reads them from a device buffer. `seqKv` comes from
//          the host field `_f16CacheCurrentLength`, which grows by 1 every
//          decode step. Captured naively, every replay would attend over
//          the SAME fixed window/offset baked in at capture time —
//          plausibly still producing FINITE, locally-coherent-looking
//          logits for a few tokens (this is the specific trap the task
//          brief warns about: "finite" is not "correct"), diverging from
//          eager output as soon as the frozen window falls behind the
//          real position.
//
//       3. [MEDIUM, DELAYED ONSET — the most dangerous of the three from a
//          testing standpoint] `ForwardFullAttnBody` re-dequantizes the
//          ENTIRE cached K/V history from F16 to F32 EVERY decode step
//          (`kvLiveElems = seqKv * kvElems`, growing every step) into
//          `_f32KvReadStagingK`/`_f32KvReadStagingV`, grown via
//          `EnsureF32KvReadStaging` in power-of-two increments keyed off
//          `seqKv`. Because `seqKv` increases monotonically across an
//          entire generation (unlike the per-call `seqLen`, which is
//          always 1 for decode), these buffers WILL be freed and
//          reallocated to new addresses whenever `seqKv` crosses a
//          power-of-two boundary (e.g. 257, 513, 1025 tokens into a
//          generation) — invalidating any previously captured graph's
//          baked-in pointers with no invalidation check in place. A short
//          smoke test (a handful to a few dozen decode steps, as Phase 2's
//          own validation plan would naturally start with) is very likely
//          to NOT cross such a boundary and would report clean, bit-exact
//          results while a longer real generation silently corrupts later.
//          This is structurally deeper than (1)/(2): even a hypothetical
//          device-position-aware `attention_f32_dyn` variant would not fix
//          it, because the CONVERT kernels' grid/block dimensions
//          (`LaunchConvertF16ToF32(..., kvLiveElems, streamH)`) are also
//          sized from the growing host `seqKv` value and baked into the
//          graph's launch topology at capture time — a captured graph's
//          launch configuration, not just its kernel-arg values, would
//          need to stay constant across replays, which the current
//          "re-dequant the whole growing history every step" design
//          cannot satisfy without itself being redesigned around a FIXED
//          max-capacity buffer (sized once to MaxSequenceLength) with an
//          internal, device-read runtime bound — i.e. the same shape as
//          CudaKvCache/CudaQuantizedKvCache already use for the dense F16
//          path, but re-implemented for this model's private F32 K/V
//          staging path (it does not route through IKvCache the same way).
//
//   Additional completeness gaps (lower severity — these break capture
//   loudly via `cuStreamEndCapture` failure or an explicit guard, not
//   silently, but still needed for a real implementation):
//     - DebugTrace/ProfileTrace/DumpDevice2D all call `_stream.Synchronize()`
//       mid-forward when their respective env vars are set; a graph-eligible
//       fast path would need an `!ProfilingEnabled`-equivalent guard exactly
//       like CudaTransformerModel.Forward's eligibility check.
//     - None of CudaDecodeGraph's scaffolding (UseGraphCapture toggle,
//       DisableGraphCapture env override, _decodeGraphKvCache/
//       _decodeGraphLayerCount invalidation, WasRolledBack guard for
//       speculative-decode rollback) exists yet for this model class — it
//       would all need to be ported/adapted, not just the kernel variants.
//
// ── Go/no-go call ──
// NO-GO for implementation in this round. Two of the three KV-cache hazards
// (1 and 2) are near-certain to reproduce the exact "produces garbage"
// failure mode this codebase already has one documented incident of
// (BitNet/I2_S vs. the generic capture body, see CudaTransformerModel.
// Forward's `!isBitNet` guard) — this time from genuinely missing
// device-position-aware kernel variants rather than a missing-op coverage
// gap. Hazard 3 is worse in a different way: it would likely PASS a short
// bit-exact multi-step smoke test (the task brief's own Phase-2 validation
// plan) and only manifest hundreds of tokens into a real generation,
// which is precisely the kind of false-green-light this investigation's
// standing discipline (native/kernels/pq2_0_gemv.cu's header) exists to
// guard against. Closing this out also costs less than it looks: issue
// #161's own candidate list says to "re-evaluate [graph capture] AFTER #1
// ships" (F32-native GEMV, eliminating ~704 of ~1,088 convert-launch
// plumbing calls per token) — #1 shipped in this round (13cd0c0), so a
// large fraction of the original motivation (per-launch CPU dispatch
// overhead) is already gone, which lowers the expected payoff of graph
// capture at the same time this investigation raises its cost/risk.
//
// Recommendation: do not fold this into #161. If pursued, scope it as its
// own dedicated issue whose FIRST deliverable is porting the CudaKvCache-
// style fixed-capacity, device-position-aware incremental KV-cache/
// attention design (proven for the F16 dense path) onto this hybrid
// model's private F32 K/V cache — that redesign is the actual prerequisite,
// not "add graph capture around the existing code." GDN state handling
// needs no rework (already correct). Validate with the same bit-exact,
// MULTIPLE-consecutive-step methodology the task brief specifies, but
// extend the real-GGUF validation run well past the first power-of-two
// KV-cache-staging-buffer growth boundary (>256 decode steps at minimum)
// given hazard 3 above — a short run is not sufficient evidence here.
// ═══════════════════════════════════════════════════════════════════════════

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
