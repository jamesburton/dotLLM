namespace DotLLM.Core.Configuration;

/// <summary>
/// Thrown when a device buffer that grows with the context does not fit in free VRAM (issue #1068). Derives from
/// <see cref="InvalidOperationException"/> so existing callers that treat a failed growth as a failed request keep working.
/// </summary>
public sealed class DeviceMemoryAdmissionException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="message">The refusal from <see cref="DeviceMemoryBudget.DescribeRefusal"/>.</param>
    public DeviceMemoryAdmissionException(string message) : base(message) { }

    /// <summary>
    /// Creates the exception with the numbers behind it (issue #1848), so a caller can tell whether freeing memory it holds (retained
    /// prefix-cache entries) could make the allocation fit before it gives that memory up.
    /// </summary>
    /// <param name="message">The refusal from <see cref="DeviceMemoryBudget.DescribeRefusal"/>.</param>
    /// <param name="requiredBytes">Bytes the refused allocation needs.</param>
    /// <param name="freeBytes">Free device bytes it was admitted against.</param>
    /// <param name="reserveBytes">Bytes kept free.</param>
    public DeviceMemoryAdmissionException(string message, long requiredBytes, long freeBytes, long reserveBytes) : base(message)
    {
        RequiredBytes = requiredBytes;
        FreeBytes = freeBytes;
        ReserveBytes = reserveBytes;
    }

    /// <summary>Bytes the refused allocation needs, when the thrower knows them.</summary>
    public long? RequiredBytes { get; }

    /// <summary>Free device bytes the allocation was admitted against, when known.</summary>
    public long? FreeBytes { get; }

    /// <summary>Bytes the admission keeps free, when known.</summary>
    public long? ReserveBytes { get; }

    /// <summary>
    /// Whether releasing <paramref name="reclaimableBytes"/> of device memory could make the refused allocation fit (#1848). True
    /// when the numbers are unknown (a thrower that only has a message, e.g. a Vulkan out-of-device-memory mapping): the caller
    /// cannot rule it out.
    /// </summary>
    /// <param name="reclaimableBytes">Device bytes the caller could free.</param>
    /// <returns>Whether freeing them can help.</returns>
    public bool CouldFitAfterReleasing(long reclaimableBytes)
        => RequiredBytes is not long required || FreeBytes is not long free
           || DeviceMemoryBudget.Fits(free + Math.Max(0, reclaimableBytes), required, ReserveBytes ?? 0);
}
