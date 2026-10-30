using DotLLM.Vulkan;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Lowers <see cref="IqF16PrefillMatmul"/>'s minimum batch (read when a model is built) so a short test prompt takes the
/// dequantise-to-F16 + coopmat prefill (#1060), and resets its path counters. Restores the variable on dispose. Tests using it run in
/// the serial <c>VulkanKernels</c> collection, so the process-wide variable is not raced.
/// </summary>
internal sealed class F16PrefillTestScope : IDisposable
{
    private readonly string? _saved;

    /// <summary>Sets the minimum batch to <paramref name="minTokens"/>.</summary>
    /// <param name="minTokens">Smallest prompt routed through the F16 path while the scope is open.</param>
    public F16PrefillTestScope(int minTokens = 2)
    {
        _saved = Environment.GetEnvironmentVariable(IqF16PrefillMatmul.MinTokensEnvVar);
        Environment.SetEnvironmentVariable(IqF16PrefillMatmul.MinTokensEnvVar, minTokens.ToString(System.Globalization.CultureInfo.InvariantCulture));
        IqF16PrefillMatmul.Counters.Reset();
    }

    /// <inheritdoc/>
    public void Dispose() => Environment.SetEnvironmentVariable(IqF16PrefillMatmul.MinTokensEnvVar, _saved);
}
