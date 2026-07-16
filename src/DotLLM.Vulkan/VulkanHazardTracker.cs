using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Hazard-scoped pipeline-barrier recorder (issue #144). Replaces the legacy
/// blanket barrier-after-every-dispatch scheme on the tracked forward path:
/// each recorded op (compute dispatch or transfer copy) declares the buffers
/// it reads and writes, and a single batched <c>vkCmdPipelineBarrier</c> is
/// emitted only when the new op actually conflicts with work recorded since
/// the last barrier (read-after-write, write-after-read, write-after-write).
/// Independent ops — e.g. the Q/K/V GEMVs, gate/up projections, per-expert
/// MoE matmuls — record back-to-back with no barrier and overlap on the GPU,
/// which is exactly llama.cpp's ggml-vulkan submission model.
/// </summary>
/// <remarks>
/// <para>
/// <b>Hazard model</b> (the "per-buffer last-write epoch" scheme): ops are
/// numbered 1..n in recording order; per buffer we track the epoch of its
/// last write and last read. A barrier synchronizes EVERYTHING recorded
/// before it (it is a global memory barrier, like the legacy one), so a
/// single watermark <c>_lastBarrier</c> captures its effect: op <c>i</c>
/// needs a barrier iff it reads a buffer whose last write is newer than the
/// watermark, or writes a buffer whose last write OR last read is newer.
/// This collapses to llama.cpp's semantics and is easy to prove sound:
/// barriers are only ever ADDED relative to the minimal hazard set, never
/// skipped for a true dependency, as long as every recorded command declares
/// a superset of its true access set.
/// </para>
/// <para>
/// <b>Access declarations</b> come from two sources, both conservative:
/// compute dispatches declare their descriptor buffer list at
/// <see cref="Kernels.DescriptorSetCache.GetOrCreate"/> time with a
/// per-binding writes mask reflected from the shader's own SPIR-V
/// <c>NonWritable</c> decorations (<see cref="SpirvReflection"/>) — an
/// unqualified binding counts as written; transfer copies declare src=read,
/// dst=write at the call site. Whole buffers, not ranges: sub-range aliasing
/// (KV-cache appends, scratch reuse) over-synchronizes, never under.
/// </para>
/// <para>
/// <b>Stages/access</b>: the emitted barrier covers
/// <c>COMPUTE|TRANSFER → COMPUTE|TRANSFER</c> with
/// <c>SHADER_WRITE|TRANSFER_WRITE → SHADER_READ|SHADER_WRITE|TRANSFER_READ|TRANSFER_WRITE</c>,
/// i.e. a superset of every legacy non-host barrier shape, so one emission
/// path replaces all of them. Host barriers (HOST→COMPUTE upload,
/// COMPUTE→HOST readback) are NOT tracked — they stay unconditional at the
/// forward's boundaries.
/// </para>
/// <para>
/// <b>Untracked commands are safe by direction</b>: a barrier the tracker
/// does not know about (e.g. the split-KV kernel's internal partial→merge
/// barrier) only adds synchronization; a WRITE the tracker does not know
/// about would be unsafe, which is why every dispatch is funneled through
/// <c>DescriptorSetCache</c> (the only descriptor-set path in the backend)
/// and every forward-path copy site carries an explicit
/// <see cref="OnTransfer"/> guard.
/// </para>
/// <para>
/// Kill-switch: <c>DOTLLM_VULKAN_LEGACY_BARRIERS=1</c> disables tracking
/// entirely (the model never arms the tracker and the legacy blanket
/// barriers are emitted unchanged). Debug aid:
/// <c>DOTLLM_VULKAN_HAZARD_VALIDATE=1</c> keeps the tracker armed but forces
/// a barrier at every guard — bit-identical to legacy ordering while still
/// exercising the tracked code path.
/// </para>
/// </remarks>
internal sealed class VulkanHazardTracker
{
    private struct Access
    {
        public long Write;
        public long Read;
    }

    private readonly Dictionary<nint, Access> _access = new(512);
    private readonly bool _alwaysBarrier;
    private long _op;
    private long _lastBarrier;
    private nint _cmdBuf;

    internal VulkanHazardTracker(bool alwaysBarrier = false)
        => _alwaysBarrier = alwaysBarrier;

    /// <summary>
    /// Arms the tracker for a fresh recording. Epochs and per-buffer state
    /// reset — the previous forward's fence wait made all prior writes
    /// visible, so carrying state across forwards would only add barriers.
    /// </summary>
    internal void Begin(nint cmdBuf)
    {
        _cmdBuf = cmdBuf;
        _op = 0;
        _lastBarrier = 0;
        _access.Clear();
    }

    /// <summary>
    /// Redirects barrier emission to a new command buffer after a mid-forward
    /// chunk submit (<see cref="VulkanDevice.SubmitContext.SplitSubmit"/>).
    /// Epoch state is kept: pipeline barriers synchronize across command
    /// buffers in same-queue submission order, so pending hazards from the
    /// previous chunk are still resolved by a barrier in the new one.
    /// </summary>
    internal void OnCommandBufferSwitch(nint cmdBuf) => _cmdBuf = cmdBuf;

    /// <summary>
    /// Declares one compute dispatch over the descriptor buffer list
    /// <paramref name="buffers"/> (index i = binding i); bit i of
    /// <paramref name="writesMask"/> set means the shader may write binding i,
    /// clear means it only reads. Emits one batched barrier first when the
    /// dispatch conflicts with anything recorded since the last barrier.
    /// </summary>
    internal void OnDispatch(ReadOnlySpan<nint> buffers, uint writesMask)
        => OnAccess(buffers, writesMask);

    /// <summary>
    /// Declares one transfer copy <paramref name="src"/> → <paramref name="dst"/>
    /// (whole-buffer granularity). Call immediately before recording the
    /// <c>vkCmdCopyBuffer</c>.
    /// </summary>
    internal void OnTransfer(nint src, nint dst)
    {
        Span<nint> buffers = stackalloc nint[2] { src, dst };
        OnAccess(buffers, 0b10u);
    }

    [SkipLocalsInit]
    private void OnAccess(ReadOnlySpan<nint> buffers, uint writesMask)
    {
        bool hazard = _alwaysBarrier;
        if (!hazard)
        {
            for (int i = 0; i < buffers.Length; i++)
            {
                if (!_access.TryGetValue(buffers[i], out var a))
                    continue; // Never touched (e.g. weights): no hazard possible.
                bool writes = (writesMask & (1u << i)) != 0;
                if (a.Write > _lastBarrier || (writes && a.Read > _lastBarrier))
                {
                    hazard = true;
                    break;
                }
            }
        }

        if (hazard)
        {
            EmitBarrier();
            _lastBarrier = _op; // Every op recorded so far is before this barrier.
        }

        long op = ++_op;
        for (int i = 0; i < buffers.Length; i++)
        {
            ref var a = ref CollectionsMarshal.GetValueRefOrAddDefault(_access, buffers[i], out _);
            if ((writesMask & (1u << i)) != 0)
                a.Write = op;
            else if (a.Read < op)
                a.Read = op;
        }
    }

    /// <summary>
    /// One global memory barrier covering compute+transfer both ways — a
    /// superset of every legacy non-host barrier shape, batched over all
    /// hazards accumulated since the previous barrier.
    /// </summary>
    private unsafe void EmitBarrier()
    {
        var barrier = new VkMemoryBarrier
        {
            sType = VkStructureType.MemoryBarrier,
            srcAccessMask = VkAccessFlags.ShaderWrite | VkAccessFlags.TransferWrite,
            dstAccessMask = VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite
                | VkAccessFlags.TransferRead | VkAccessFlags.TransferWrite,
        };
        ProfileCounters.Barriers++;
        ProfileCounters.HazardBarriers++;
        VulkanApi.vkCmdPipelineBarrier(
            _cmdBuf,
            srcStageMask: VkPipelineStageFlags.ComputeShader | VkPipelineStageFlags.Transfer,
            dstStageMask: VkPipelineStageFlags.ComputeShader | VkPipelineStageFlags.Transfer,
            dependencyFlags: 0,
            memoryBarrierCount: 1, pMemoryBarriers: barrier,
            bufferMemoryBarrierCount: 0, pBufferMemoryBarriers: 0,
            imageMemoryBarrierCount: 0, pImageMemoryBarriers: 0);
    }
}
