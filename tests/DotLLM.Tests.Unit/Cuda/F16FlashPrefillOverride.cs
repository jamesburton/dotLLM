using DotLLM.Cuda;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Sets <see cref="CudaTransformerModel.EnableF16FlashPrefill"/> for a scope and restores the previous value on dispose (#1923): the
/// flag is process-wide, so a test or harness that pins it must not leak the change into later tests in the same testhost.
/// </summary>
internal sealed class F16FlashPrefillOverride : IDisposable
{
    private readonly bool _previous;

    private F16FlashPrefillOverride(bool value)
    {
        _previous = CudaTransformerModel.EnableF16FlashPrefill;
        CudaTransformerModel.EnableF16FlashPrefill = value;
    }

    /// <summary>Sets the flag to <paramref name="value"/> until the returned scope is disposed.</summary>
    public static F16FlashPrefillOverride Set(bool value) => new(value);

    /// <inheritdoc/>
    public void Dispose() => CudaTransformerModel.EnableF16FlashPrefill = _previous;
}
