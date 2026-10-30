using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Load-time refusal of a weight format a backend path has no kernel for. Refusing before any upload is the
/// alternative to widening the weights (CLAUDE.md rule 2) or running them through a kernel for another format.
/// </summary>
public static class QuantSupport
{
    /// <summary>Refuses the file when it holds any tensor of <paramref name="format"/>.</summary>
    /// <param name="tensors">The GGUF tensor table.</param>
    /// <param name="format">The unsupported format.</param>
    /// <param name="backend">The backend's name, for the message.</param>
    /// <param name="issue">The tracking issue, for the message.</param>
    /// <exception cref="NotSupportedException">A <paramref name="format"/> tensor is present.</exception>
    public static void RejectAny(
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, QuantizationType format, string backend, int issue)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        int count = 0;
        string? first = null;
        foreach (var (name, desc) in tensors)
        {
            if (desc.QuantizationType != format)
                continue;
            count++;
            if (first is null || string.CompareOrdinal(name, first) < 0)
                first = name;
        }

        if (count == 0)
            return;

        throw new NotSupportedException(
            $"The {backend} backend has no {format} (GGUF type {(int)format}) path for this model, which has {count} {format} " +
            $"tensor(s) (first: '{first}'). Use --device cpu (#{issue}).");
    }
}
