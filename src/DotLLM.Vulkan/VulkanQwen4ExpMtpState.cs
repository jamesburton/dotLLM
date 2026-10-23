using DotLLM.Core.Models;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-sequence state of the Vulkan qwen4exp MTP draft head (#820): the head's own dense K/V cells (one attention layer, device resident),
/// the head's input residual for the next draft step (<see cref="Pending"/>, a 4-stream residual row, BEFORE the head mixer), the trunk
/// residual of the last absorbed position (<see cref="Carry"/>, the <c>R_{p-1}</c> the next absorb pairs its first token with, #469) and
/// the trunk residual rows of the last forward (<see cref="Captured"/>) that <see cref="SeedFromCapturedRow"/> picks from.
/// </summary>
/// <remarks>
/// Everything stays on the device: the residual rows are 4 x hidden floats each and never visit the host. Seeding is lazy (a row index,
/// resolved by the next command buffer that reads the pending/carry rows) so a round does not pay a submit for it.
/// </remarks>
public sealed class VulkanQwen4ExpMtpState : IMtpState
{
    /// <summary>Trunk residual rows kept after a forward (a verify batch is at most this long; a longer prefill keeps its last rows).</summary>
    internal const int CaptureRows = 32;

    private readonly VulkanDevice _device;
    private readonly int _rowFloats;
    private readonly int _maxLength;
    private int _covered;
    private bool _disposed;

    internal VulkanNemotronHKvCache Kv { get; }
    internal VulkanDevice.Buffer Pending { get; }
    internal VulkanDevice.Buffer Carry { get; }
    internal VulkanDevice.Buffer Captured { get; }

    /// <summary>First absolute row index (within the last forward) held in <see cref="Captured"/>.</summary>
    internal int CapturedFirst { get; set; }

    /// <summary>Row count of the last forward (rows <c>[CapturedFirst, CapturedTotal)</c> are held).</summary>
    internal int CapturedTotal { get; set; }

    /// <summary>Row of <see cref="Captured"/> (absolute index) to copy into pending and carry before the next use; -1 = none.</summary>
    internal int PendingSeedRow { get; set; } = -1;

    /// <summary>Tokens fed to the draft steps of the current round (n-gram prefetch of the coming verify rows, #822).</summary>
    internal List<int> Run { get; } = new();
    internal int RunStart, RunNextPosition;

    internal VulkanQwen4ExpMtpState(VulkanDevice device, VulkanNemotronHKvCache kv, int rowFloats, int maxLength)
    {
        _device = device; Kv = kv; _rowFloats = rowFloats; _maxLength = maxLength;
        long rowBytes = (long)rowFloats * sizeof(float);
        Pending = device.AllocateDeviceLocal(rowBytes);
        Carry = device.AllocateDeviceLocal(rowBytes);
        Captured = device.AllocateDeviceLocal(rowBytes * CaptureRows);
        // Start defined, not garbage: a first draft step on a sequence whose prefill never ran (tests) reads these.
        var zeros = new float[rowFloats];
        device.Upload(zeros, Pending);
        device.Upload(zeros, Carry);
    }

    /// <inheritdoc/>
    public int CurrentLength => _covered;

    /// <summary>Longest sequence, in positions, the head can hold.</summary>
    public int MaxLength => _maxLength;

    /// <inheritdoc/>
    /// <remarks>The captured rows live on the device; use <see cref="DownloadCapturedRow"/> to read one.</remarks>
    public ReadOnlySpan<float> CapturedHiddenRows => ReadOnlySpan<float>.Empty;

    /// <inheritdoc/>
    public int CapturedRowCount => CapturedTotal;

    /// <inheritdoc/>
    public int HiddenSize => _rowFloats;

    /// <summary>Head K/V cells currently held.</summary>
    public int CellCount => Kv.CurrentLength;

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        ThrowIfDisposed();
        if (length < 0 || length > _covered) throw new ArgumentOutOfRangeException(nameof(length));
        _covered = length;
        int cells = Math.Max(length - 1, 0);   // token 0 owns no cell
        if (Kv.CurrentLength > cells) Kv.Rollback(cells);
    }

    /// <inheritdoc/>
    public void SeedFromCapturedRow(int rowIndex)
    {
        ThrowIfDisposed();
        if (rowIndex < CapturedFirst || rowIndex >= CapturedTotal)
            throw new ArgumentOutOfRangeException(nameof(rowIndex),
                $"rowIndex {rowIndex} out of range [{CapturedFirst}, {CapturedTotal}): the last trunk forward did not carry this state, or the row was not kept.");
        PendingSeedRow = rowIndex;
    }

    /// <summary>Records the lazy seed (captured row to pending and carry) into <paramref name="cmd"/>; no-op when none is queued.</summary>
    internal void RecordSeed(nint cmd)
    {
        if (PendingSeedRow < 0) return;
        ulong rowBytes = (ulong)_rowFloats * sizeof(float);
        ulong src = (ulong)(PendingSeedRow - CapturedFirst) * rowBytes;
        KernelSupport.ComputeTransferFullBarrier(cmd);
        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, Captured, Pending, src, 0, rowBytes);
        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, Captured, Carry, src, 0, rowBytes);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        PendingSeedRow = -1;
    }

    /// <summary>Test/diagnostic: the captured trunk residual row <paramref name="rowIndex"/> (absolute index within the last forward).</summary>
    public float[] DownloadCapturedRow(int rowIndex)
    {
        if (rowIndex < CapturedFirst || rowIndex >= CapturedTotal) throw new ArgumentOutOfRangeException(nameof(rowIndex));
        var all = new float[CaptureRows * _rowFloats];
        _device.Download(Captured, all);
        return all.AsSpan((rowIndex - CapturedFirst) * _rowFloats, _rowFloats).ToArray();
    }

    internal void MarkCovered(int position) => _covered = position + 1;
    internal void SetCovered(int length) => _covered = length;

    internal void BeginAbsorb(int firstPosition, int count)
    {
        ThrowIfDisposed();
        if (_covered > firstPosition) Rollback(firstPosition);
        else if (_covered < firstPosition)
            throw new InvalidOperationException(
                $"MTP absorb at position {firstPosition} but the head only covers {_covered} positions. Every trunk forward of the " +
                "sequence must pass the MTP state so the head absorbs it (prefill included).");
        if ((long)firstPosition + count > _maxLength)
            throw new InvalidOperationException(
                $"VulkanQwen4ExpMtpState covers {_maxLength} positions; absorbing [{firstPosition}, {firstPosition + count}) exceeds it. " +
                "Create the state with CreateMtpState(maxSequenceLength) covering the whole sequence.");
    }

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(nameof(VulkanQwen4ExpMtpState)); }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Kv.Dispose(); Pending.Dispose(); Carry.Dispose(); Captured.Dispose();
    }
}
