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
}
