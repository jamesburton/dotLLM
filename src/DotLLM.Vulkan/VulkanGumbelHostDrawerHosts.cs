using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

// #1435: device Gumbel draw for the Vulkan hosts without an in-place LM-head hook. Each runs its own forward and lets
// GumbelHostLogitsDrawer upload the logits rows and draw them on the GPU (bit-identical to GumbelNoise.Sample). One copy per host so the
// IModel members stay on the concrete type; the logic lives in GumbelHostDrawerHolder.

public sealed partial class VulkanQwen3MoeHybridTransformerModel
{
    private GumbelHostDrawerHolder? _gumbelHostDrawer;
    private GumbelHostDrawerHolder GumbelHostDrawer => _gumbelHostDrawer ??= new GumbelHostDrawerHolder(_device, this);

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelSampling => GumbelHostDrawer.Available;

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelFilter => GumbelHostDrawer.FilterAvailable;

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelBatch => GumbelHostDrawer.Available;

    /// <inheritdoc/>
    public int DeviceGumbelMaxRows => GumbelHostDrawer.Available ? GumbelHostLogitsDrawer.MaxRows : 0;

    /// <inheritdoc/>
    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, Span<int> sampledIds)
        => GumbelHostDrawer.ForwardGumbelSampled(tokenIds, positions, kvCache, mtpState, recurrentRowSnapshots, seed, firstTokenIndex,
            invTemperature, default, sampledIds);

    /// <inheritdoc/>
    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, GumbelFilter filter,
        Span<int> sampledIds)
        => GumbelHostDrawer.ForwardGumbelSampled(tokenIds, positions, kvCache, mtpState, recurrentRowSnapshots, seed, firstTokenIndex,
            invTemperature, filter, sampledIds);

    /// <inheritdoc/>
    public void ForwardBatchGumbelSampled(IReadOnlyList<SequenceForwardRequest> requests, int deviceId,
        ReadOnlySpan<GumbelDraw> draws, Span<int> sampledIds)
        => GumbelHostDrawer.ForwardBatchGumbelSampled(requests, deviceId, draws, sampledIds);
}

public sealed partial class VulkanQwen4ExpTransformerModel
{
    private GumbelHostDrawerHolder? _gumbelHostDrawer;
    private GumbelHostDrawerHolder GumbelHostDrawer => _gumbelHostDrawer ??= new GumbelHostDrawerHolder(_device, this);

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelSampling => GumbelHostDrawer.Available;

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelFilter => GumbelHostDrawer.FilterAvailable;

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelBatch => GumbelHostDrawer.Available;

    /// <inheritdoc/>
    public int DeviceGumbelMaxRows => GumbelHostDrawer.Available ? GumbelHostLogitsDrawer.MaxRows : 0;

    /// <inheritdoc/>
    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, Span<int> sampledIds)
        => GumbelHostDrawer.ForwardGumbelSampled(tokenIds, positions, kvCache, mtpState, recurrentRowSnapshots, seed, firstTokenIndex,
            invTemperature, default, sampledIds);

    /// <inheritdoc/>
    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, GumbelFilter filter,
        Span<int> sampledIds)
        => GumbelHostDrawer.ForwardGumbelSampled(tokenIds, positions, kvCache, mtpState, recurrentRowSnapshots, seed, firstTokenIndex,
            invTemperature, filter, sampledIds);

    /// <inheritdoc/>
    public void ForwardBatchGumbelSampled(IReadOnlyList<SequenceForwardRequest> requests, int deviceId,
        ReadOnlySpan<GumbelDraw> draws, Span<int> sampledIds)
        => GumbelHostDrawer.ForwardBatchGumbelSampled(requests, deviceId, draws, sampledIds);
}

public sealed partial class VulkanTransformerModel
{
    private GumbelHostDrawerHolder? _gumbelHostDrawer;
    private GumbelHostDrawerHolder GumbelHostDrawer => _gumbelHostDrawer ??= new GumbelHostDrawerHolder(_device, this);

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelSampling => GumbelHostDrawer.Available;

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelFilter => GumbelHostDrawer.FilterAvailable;

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelBatch => GumbelHostDrawer.Available;

    /// <inheritdoc/>
    public int DeviceGumbelMaxRows => GumbelHostDrawer.Available ? GumbelHostLogitsDrawer.MaxRows : 0;

    /// <inheritdoc/>
    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, Span<int> sampledIds)
        => GumbelHostDrawer.ForwardGumbelSampled(tokenIds, positions, kvCache, mtpState, recurrentRowSnapshots, seed, firstTokenIndex,
            invTemperature, default, sampledIds);

    /// <inheritdoc/>
    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, GumbelFilter filter,
        Span<int> sampledIds)
        => GumbelHostDrawer.ForwardGumbelSampled(tokenIds, positions, kvCache, mtpState, recurrentRowSnapshots, seed, firstTokenIndex,
            invTemperature, filter, sampledIds);

    /// <inheritdoc/>
    public void ForwardBatchGumbelSampled(IReadOnlyList<SequenceForwardRequest> requests, int deviceId,
        ReadOnlySpan<GumbelDraw> draws, Span<int> sampledIds)
        => GumbelHostDrawer.ForwardBatchGumbelSampled(requests, deviceId, draws, sampledIds);
}

public sealed partial class VulkanNemotronHTransformerModel
{
    private GumbelHostDrawerHolder? _gumbelHostDrawer;
    private GumbelHostDrawerHolder GumbelHostDrawer => _gumbelHostDrawer ??= new GumbelHostDrawerHolder(_device, this);

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelSampling => GumbelHostDrawer.Available;

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelFilter => GumbelHostDrawer.FilterAvailable;

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelBatch => GumbelHostDrawer.Available;

    /// <inheritdoc/>
    public int DeviceGumbelMaxRows => GumbelHostDrawer.Available ? GumbelHostLogitsDrawer.MaxRows : 0;

    /// <inheritdoc/>
    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, Span<int> sampledIds)
        => GumbelHostDrawer.ForwardGumbelSampled(tokenIds, positions, kvCache, mtpState, recurrentRowSnapshots, seed, firstTokenIndex,
            invTemperature, default, sampledIds);

    /// <inheritdoc/>
    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, GumbelFilter filter,
        Span<int> sampledIds)
        => GumbelHostDrawer.ForwardGumbelSampled(tokenIds, positions, kvCache, mtpState, recurrentRowSnapshots, seed, firstTokenIndex,
            invTemperature, filter, sampledIds);

    /// <inheritdoc/>
    public void ForwardBatchGumbelSampled(IReadOnlyList<SequenceForwardRequest> requests, int deviceId,
        ReadOnlySpan<GumbelDraw> draws, Span<int> sampledIds)
        => GumbelHostDrawer.ForwardBatchGumbelSampled(requests, deviceId, draws, sampledIds);
}
