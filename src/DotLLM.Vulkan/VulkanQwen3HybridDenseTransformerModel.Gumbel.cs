using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-side Gumbel-max sampling of the verify rows (#1382): the LM head's logits stay where the head wrote them, a compute shader
/// draws <c>argmax(logits * invTemp + Gumbel(seed, n, i))</c> per row, and only one token id per row crosses to the host.
/// </summary>
public sealed partial class VulkanQwen3HybridDenseTransformerModel
{
    /// <summary>Env switch: <c>DOTLLM_VULKAN_DEVICE_GUMBEL=0</c> turns the device draw off (the host path then reads the logits back).</summary>
    public const string DeviceGumbelEnvVar = "DOTLLM_VULKAN_DEVICE_GUMBEL";

    private readonly record struct GumbelRequest(ulong Seed, int FirstTokenIndex, float InvTemperature);

    private GumbelRequest? _pendingGumbel;
    private GumbelArgMaxKernel? _gumbelKernel;
    private VulkanDevice.Buffer? _gumbelIds;
    private int[] _gumbelHost = [];
    private int _gumbelHostRows;

    /// <summary>Verify forwards whose rows were drawn on the device (a perturbation counter for tests and the A/B harness).</summary>
    internal int DeviceGumbelForwards { get; private set; }

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelSampling
        => _spvDir.Length > 0 && GumbelArgMaxKernel.IsAvailable(_spvDir)
           && Environment.GetEnvironmentVariable(DeviceGumbelEnvVar) != "0";

    /// <inheritdoc/>
    public int DeviceGumbelMaxRows => SupportsDeviceGumbelSampling ? Math.Min(_allRowLogitsLimit, 32) : 0;

    /// <inheritdoc/>
    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, Span<int> sampledIds)
    {
        int rows = tokenIds.Length;
        if (!SupportsDeviceGumbelSampling || rows > DeviceGumbelMaxRows)
            throw new NotSupportedException($"{nameof(VulkanQwen3HybridDenseTransformerModel)}: no device Gumbel sampling for {rows} rows.");
        if (sampledIds.Length < rows) throw new ArgumentException("sampledIds is shorter than the row count.", nameof(sampledIds));

        // Every allocation happens BEFORE recording: a new buffer invalidates the handle-keyed descriptor caches.
        bool created = false;
        _gumbelKernel ??= GumbelArgMaxKernel.Create(_device, _spvDir);
        if (_gumbelIds is null)
        {
            _gumbelIds = _device.AllocateHostReadback(32 * sizeof(uint));
            created = true;
        }
        if (created) InvalidateDescriptorCaches();

        _gumbelHostRows = 0;
        _pendingGumbel = new GumbelRequest(seed, firstTokenIndex, invTemperature);
        try
        {
            using var placeholder = recurrentRowSnapshots
                ? ForwardWithRecurrentSnapshots(tokenIds, positions, deviceId, kvCache, mtpState)
                : Forward(tokenIds, positions, deviceId, kvCache, adapter: null, mtpState);
        }
        finally
        {
            _pendingGumbel = null;
        }
        if (_gumbelHostRows != rows)
            throw new InvalidOperationException($"The device Gumbel draw covered {_gumbelHostRows} of {rows} rows (the forward took a path without the hook).");
        _gumbelHost.AsSpan(0, rows).CopyTo(sampledIds);
        DeviceGumbelForwards++;
    }
}
