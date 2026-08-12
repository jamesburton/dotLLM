using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan;

/// <summary>One tensor's on-device residency outcome.</summary>
public readonly record struct VulkanTensorResidency(
    string Name, QuantizationType Source, QuantizationType Device,
    long PackedBytes, long UploadedBytes)
{
    /// <summary>True when the tensor could not be kept packed and was widened on upload.</summary>
    public bool Expanded => Device != Source;
}

/// <summary>
/// Accounting for what Vulkan kept packed and what it widened to F32.
/// <para>
/// This exists because <c>VulkanWeights.DeviceQuantTypeFor</c> ends in an unconditional
/// <c>return QuantizationType.F32</c>: a type with no Vulkan kernel expands 5-7x on upload
/// with no diagnostic at all. It is also how a new kernel is proven to be genuinely routed
/// to — a capability flag says a path exists, not that it ran.
/// </para>
/// </summary>
public sealed class VulkanResidencyReport
{
    private readonly List<VulkanTensorResidency> _entries = new();

    /// <summary>All recorded tensors, in upload order.</summary>
    public IReadOnlyList<VulkanTensorResidency> Entries => _entries;

    /// <summary>Total bytes of the source (packed) representation across all recorded tensors.</summary>
    public long PackedBytes { get; private set; }

    /// <summary>Total bytes actually uploaded to the device across all recorded tensors.</summary>
    public long UploadedBytes { get; private set; }

    /// <summary>Count of tensors whose device type differs from their source type.</summary>
    public int ExpandedTensorCount { get; private set; }

    /// <summary>Records one tensor's upload outcome.</summary>
    public void Add(string name, QuantizationType source, QuantizationType device,
        long packedBytes, long uploadedBytes)
    {
        var entry = new VulkanTensorResidency(name, source, device, packedBytes, uploadedBytes);
        _entries.Add(entry);
        PackedBytes += packedBytes;
        UploadedBytes += uploadedBytes;
        if (entry.Expanded) ExpandedTensorCount++;
    }

    /// <summary>Human-readable summary; lists only the tensors that were widened.</summary>
    public string Describe()
    {
        if (ExpandedTensorCount == 0)
            return $"Vulkan residency: all {_entries.Count} tensors kept packed ({PackedBytes:N0} bytes).";

        var sb = new System.Text.StringBuilder();
        sb.AppendLine(
            $"Vulkan residency: {ExpandedTensorCount} of {_entries.Count} tensors widened on upload "
            + $"({PackedBytes:N0} packed -> {UploadedBytes:N0} uploaded bytes, "
            + $"{(double)UploadedBytes / Math.Max(PackedBytes, 1):F1}x).");
        foreach (var group in _entries.Where(e => e.Expanded)
                     .GroupBy(e => (e.Source, e.Device))
                     .OrderByDescending(g => g.Sum(e => e.UploadedBytes - e.PackedBytes)))
        {
            long extra = group.Sum(e => e.UploadedBytes - e.PackedBytes);
            sb.AppendLine(
                $"  {group.Key.Source} -> {group.Key.Device}: {group.Count()} tensor(s), "
                + $"+{extra:N0} bytes. First: {group.First().Name}");
        }
        return sb.ToString().TrimEnd();
    }
}
