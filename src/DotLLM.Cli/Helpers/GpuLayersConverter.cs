using System.ComponentModel;
using System.Globalization;
using DotLLM.Core.Configuration;

namespace DotLLM.Cli.Helpers;

/// <summary>
/// Binds <c>--gpu-layers</c> (issue #1851): <c>auto</c> (<see cref="GpuOffloadPlanner.AutoGpuLayers"/>), <c>all</c> / <c>-1</c>, or a
/// non-negative layer count, onto an <c>int?</c> setting.
/// </summary>
public sealed class GpuLayersConverter : TypeConverter
{
    /// <summary>The option help shared by <c>run</c>, <c>chat</c> and <c>serve</c>.</summary>
    public const string OptionHelp =
        "CUDA layer placement: 'auto' (the largest number of leading layers whose weights, KV for the context and scratch fit in free " +
        "VRAM; the rest run on the CPU), or N leading layers on the GPU and the rest on the CPU (0 = CPU only, 'all' or -1 = every " +
        "layer). Omit for the device default: every layer with --device gpu, and with --device auto an automatic split when the model " +
        "does not fit a CUDA device whole. Ignored on vulkan (whole model is device-resident).";

    /// <inheritdoc/>
    public override bool CanConvertFrom(ITypeDescriptorContext? context, Type sourceType)
        => sourceType == typeof(string) || base.CanConvertFrom(context, sourceType);

    /// <inheritdoc/>
    public override object? ConvertFrom(ITypeDescriptorContext? context, CultureInfo? culture, object value)
    {
        if (value is not string s) return base.ConvertFrom(context, culture, value);
        return GpuOffloadPlanner.TryParseGpuLayers(s, out int? layers, out string? error)
            ? layers
            : throw new FormatException(error);
    }
}
