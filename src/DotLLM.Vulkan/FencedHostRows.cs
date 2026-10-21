using System.Diagnostics;

namespace DotLLM.Vulkan;

/// <summary>
/// The host side of one device submission whose results the device writes straight into host-visible memory (issue #1083, the
/// iGPU expert service's <c>Begin</c> / <c>End</c>): which rows are in flight, and the rule that the host observes the submission's
/// fence <b>before</b> it reads a single result byte. Until the fence signals, the host-visible output still holds whatever the
/// previous submission (or nothing) left there, and nothing else in the protocol orders the host read after the device write.
/// </summary>
/// <remarks>
/// Kept apart from the Vulkan objects so the ordering is testable without a device: the fence wait is a delegate, and a test's
/// fake "device" writes its results only when that delegate runs (the latest moment a real device may finish).
/// Not thread-safe: one submission at a time, from one thread.
/// </remarks>
internal sealed unsafe class FencedHostRows
{
    private readonly Action _waitFence;
    private int _rows, _rowFloats;
    private long _beginTimestamp;

    /// <summary>Creates the tracker over <paramref name="waitFence"/>, which blocks until the submitted work has finished.</summary>
    /// <param name="waitFence">Blocks until the device signals the submission's fence (throws when the device is lost).</param>
    public FencedHostRows(Action waitFence)
    {
        ArgumentNullException.ThrowIfNull(waitFence);
        _waitFence = waitFence;
    }

    /// <summary>True between <see cref="Submitted"/> and <see cref="Collect"/>.</summary>
    public bool InFlight => _rows > 0;

    /// <summary>Microseconds from the submission's start until <see cref="Collect"/> observed its fence.</summary>
    public double LastComputeUs { get; private set; }

    /// <summary>Microseconds the last <see cref="Collect"/> blocked on the fence.</summary>
    public double LastWaitUs { get; private set; }

    /// <summary>Records that a submission of <paramref name="rows"/> rows of <paramref name="rowFloats"/> floats each is in flight.</summary>
    /// <param name="rows">Rows the device will write (at least 1).</param>
    /// <param name="rowFloats">Floats per row (at least 1).</param>
    /// <param name="beginTimestamp"><see cref="Stopwatch.GetTimestamp"/> when the submission's host work began.</param>
    /// <exception cref="InvalidOperationException">A submission is already in flight.</exception>
    public void Submitted(int rows, int rowFloats, long beginTimestamp)
    {
        ThrowIfInFlight();
        ArgumentOutOfRangeException.ThrowIfLessThan(rows, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(rowFloats, 1);
        _rows = rows;
        _rowFloats = rowFloats;
        _beginTimestamp = beginTimestamp;
    }

    /// <summary>Throws when a submission is in flight (a second one would overwrite the first's input and output).</summary>
    /// <exception cref="InvalidOperationException">A submission is in flight.</exception>
    public void ThrowIfInFlight()
    {
        if (InFlight) throw new InvalidOperationException("An iGPU expert submission is already in flight; call End first.");
    }

    /// <summary>
    /// Waits on the fence, then copies the in-flight rows, <c>[rows, rowFloats]</c>, from the device's host-visible output
    /// <paramref name="deviceRows"/> to <paramref name="outRows"/>. A failed wait still ends the submission (the next one must not
    /// see it in flight) and copies nothing.
    /// </summary>
    /// <param name="deviceRows">The host-visible memory the device writes.</param>
    /// <param name="outRows">Destination, at least <c>rows · rowFloats</c> floats.</param>
    /// <returns>Rows copied.</returns>
    /// <exception cref="InvalidOperationException">Nothing is in flight.</exception>
    public int Collect(float* deviceRows, float* outRows)
    {
        if (!InFlight) throw new InvalidOperationException("No iGPU expert submission is in flight.");
        int rows = _rows;
        long w = Stopwatch.GetTimestamp();
        try
        {
            // The ordering this type exists for: no result byte is read before the fence has signalled.
            _waitFence();
        }
        finally
        {
            _rows = 0;
            LastWaitUs = Stopwatch.GetElapsedTime(w).TotalMicroseconds;
            LastComputeUs = Stopwatch.GetElapsedTime(_beginTimestamp).TotalMicroseconds;
        }
        long bytes = (long)rows * _rowFloats * sizeof(float);
        Buffer.MemoryCopy(deviceRows, outRows, bytes, bytes);
        return rows;
    }

    /// <summary>
    /// Waits for a submission still in flight at teardown, so its memory is not freed under the device; a lost device (the
    /// delegate throws <typeparamref name="TLost"/>) leaves nothing to wait for.
    /// </summary>
    /// <typeparam name="TLost">The exception type a lost device raises.</typeparam>
    public void DrainForDispose<TLost>() where TLost : Exception
    {
        if (!InFlight) return;
        try { _waitFence(); }
        catch (TLost) { /* device lost: nothing left to wait for */ }
        _rows = 0;
    }
}
