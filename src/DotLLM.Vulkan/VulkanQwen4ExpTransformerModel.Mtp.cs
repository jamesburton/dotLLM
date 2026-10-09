using DotLLM.Core.Models;
using DotLLM.Vulkan.Kernels;
using System.Numerics.Tensors;

namespace DotLLM.Vulkan;

public sealed unsafe partial class VulkanQwen4ExpTransformerModel
{
    /// <summary>
    /// Go/no-go diagnostic for the Vulkan MTP head (#820): wall ms of one draft-step proxy. The head block is a QSA + 512-expert MoE block
    /// on the gated residual plus a head mixer and the trunk's LM head; the trunk's FIRST QSA layer has exactly that geometry, so one
    /// 1-row recording of <c>GR-read, QSA, GR-write, GR-read, MoE, GR-write, head mixer, LM head</c> (one command buffer, one submit, logits
    /// downloaded and arg-maxed on the host) measures the draft step without the head weights. Omitted: <c>eh_proj</c> (a ~52 MB Q8_0 GEMV)
    /// and the head's Q8_0 expert banks (trunk banks are Q4_K).
    /// </summary>
    /// <param name="state">A state that has just run a forward (its last residual row seeds the proxy; its KV is read, not advanced).</param>
    /// <param name="reps">Steps to time.</param>
    /// <returns>Per-step wall ms.</returns>
    public double[] ProbeDraftStepMs(VulkanQwen4ExpSequenceState state, int reps)
    {
        int fq = -1;
        for (int il = 0; il < Config.NumLayers; il++)
            if (Config.HybridLayout!.LayerKind[il] != HybridLayerKind.GatedDeltaNet) { fq = il; break; }
        if (fq < 0) throw new InvalidOperationException("No QSA layer.");
        var kv = state.OwnKv;
        var st = _core.Q4State;
        int S = _streams, H = _hidden;
        int pos = state.Length;
        Span<int> positions = [pos];
        var submit = _core.Q4Submit;
        var w0 = _core.Q4Weights;
        var times = new double[reps];
        float[] logits = new float[_vocab];
        for (int r = 0; r < reps; r++)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            if (kv.CurrentLength > pos) kv.Rollback(pos);
            _core.Q4UploadPositions(positions);
            submit.Begin();
            nint cmd = submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmd);
            void Barrier() => KernelSupport.ComputeTransferFullBarrier(cmd);
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _res, _headRes, (ulong)(((long)Math.Min(pos, _scratchCapacity) - 1) * S * H * 4), 0, (ulong)((long)S * H * 4));
            Barrier();
            RecordGrRead(cmd, _attnGr[fq], _headRes, st.NormOutput, 1, inject: true);
            _core.Q4RecordAttention(cmd, fq, 1, positions, kv);
            Barrier();
            _gr.RecordWrite(cmd, _headRes, st.NormOutput, _gains, 1, S, H);
            Barrier();
            RecordGrRead(cmd, _ffnGr[fq], _headRes, st.NormOutput, 1, inject: true);
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, st.NormOutput, st.MoeSharedInput, 0, 0, (ulong)H * 4);
            Barrier();
            _core.Q4RecordMoe(cmd, _moe[fq], 1);
            Barrier();
            _gr.RecordWrite(cmd, _headRes, st.NormOutput, _gains, 1, S, H);
            Barrier();
            RecordGrRead(cmd, _headGr, _headRes, st.NormOutput, 1, inject: false);
            _core.Q4RecordMatmul(cmd, w0.OutputWeight, w0.OutputDeviceQuantType, st.NormOutput, st.Logits,
                outputDim: w0.OutputOutputDim, inputDim: w0.OutputInputDim, seqLen: 1);
            KernelSupport.ComputeToHostBarrier(cmd);
            submit.SubmitAndWait();
            _device.Download(st.Logits, logits.AsSpan());
            int top = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)logits);
            if (top < 0) throw new InvalidOperationException();
            times[r] = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }
        if (kv.CurrentLength > pos) kv.Rollback(pos);
        return times;
    }
}
