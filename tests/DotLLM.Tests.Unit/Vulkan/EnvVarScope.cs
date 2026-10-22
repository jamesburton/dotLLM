namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Sets one process environment variable for the scope's lifetime and restores the previous value on dispose. Tests using it run in
/// the serial <c>VulkanKernels</c> collection, so the process-wide variable is not raced.
/// </summary>
internal sealed class EnvVarScope : IDisposable
{
    private readonly string _name;
    private readonly string? _saved;

    /// <summary>Sets <paramref name="name"/> to <paramref name="value"/>.</summary>
    /// <param name="name">Variable name.</param>
    /// <param name="value">Value while the scope is open.</param>
    public EnvVarScope(string name, string value)
    {
        _name = name;
        _saved = Environment.GetEnvironmentVariable(name);
        Environment.SetEnvironmentVariable(name, value);
    }

    /// <inheritdoc/>
    public void Dispose() => Environment.SetEnvironmentVariable(_name, _saved);
}
