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

    // Batched per-row draw (#1434): one id and one (seed, index, invTemp) record per sequence of a continuous-batch decode step.
    private GumbelArgMaxRowsKernel? _gumbelRowsKernel;
    private VulkanDevice.Buffer? _gumbelRowIds;
    private VulkanDevice.Buffer? _gumbelRowParams;
    private int _gumbelRowCapacity;
    private GumbelDraw[]? _pendingGumbelBatch;
    private int[] _gumbelBatchHost = [];
    private int _gumbelBatchHostRows;

    /// <summary>Batched decode steps whose rows were drawn by the fused per-row kernel (a perturbation counter for tests).</summary>
    internal int DeviceGumbelFusedBatches { get; private set; }

    /// <summary>Batched decode steps that drew on the device through the serial per-sequence loop (below the fuse threshold).</summary>
    internal int DeviceGumbelSerialBatches { get; private set; }

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

    /// <inheritdoc/>
    public bool SupportsDeviceGumbelBatch => SupportsDeviceGumbelSampling && GumbelArgMaxRowsKernel.IsAvailable(_spvDir);

    /// <inheritdoc/>
    /// <remarks>
    /// At or above the fuse threshold (<c>BatchFuseMinRows</c>) the sequences run as one stacked forward and the LM-head logits are drawn by
    /// <see cref="GumbelArgMaxRowsKernel"/> (per-row seed / index / temperature); below it each sequence runs the serial forward and the
    /// single-row <see cref="GumbelArgMaxKernel"/> draws its row. Either way the draw is the CPU <c>GumbelNoise.Sample</c> over the same
    /// logits the plain <see cref="ForwardBatch"/> would return.
    /// </remarks>
    public void ForwardBatchGumbelSampled(IReadOnlyList<SequenceForwardRequest> requests, int deviceId,
        ReadOnlySpan<GumbelDraw> draws, Span<int> sampledIds)
    {
        ArgumentNullException.ThrowIfNull(requests);
        int n = requests.Count;
        if (!SupportsDeviceGumbelBatch)
            throw new NotSupportedException($"{nameof(VulkanQwen3HybridDenseTransformerModel)}: no batched device Gumbel sampling.");
        if (draws.Length != n) throw new ArgumentException("One draw per request is required.", nameof(draws));
        if (sampledIds.Length < n) throw new ArgumentException("sampledIds is shorter than the request count.", nameof(sampledIds));
        if (n == 0) return;
        for (int i = 0; i < n; i++)
            if (requests[i].TokenIds.Length != 1)
                throw new ArgumentException($"Request {i}: batched device Gumbel sampling takes single-token decode requests.", nameof(requests));
        ValidateBatchRequests(requests);

        // Every allocation happens BEFORE recording: a new buffer invalidates the handle-keyed descriptor caches.
        bool created = false;
        _gumbelKernel ??= GumbelArgMaxKernel.Create(_device, _spvDir);
        _gumbelRowsKernel ??= GumbelArgMaxRowsKernel.Create(_device, _spvDir);
        if (_gumbelIds is null) { _gumbelIds = _device.AllocateHostReadback(32 * sizeof(uint)); created = true; }
        if (_gumbelRowIds is null || _gumbelRowCapacity < n)
        {
            _gumbelRowIds?.Dispose();
            _gumbelRowParams?.Dispose();
            _gumbelRowCapacity = Math.Max(n, 32);
            _gumbelRowIds = _device.AllocateHostReadback((long)_gumbelRowCapacity * sizeof(uint));
            _gumbelRowParams = _device.Allocate((long)_gumbelRowCapacity * GumbelArgMaxRowsKernel.RowParamBytes);
            created = true;
        }
        if (created) InvalidateDescriptorCaches();

        if (n >= 2 && BatchFuseEnabled && _prof is null && TotalRows(requests) >= BatchFuseMinRows)
        {
            _gumbelBatchHostRows = 0;
            _pendingGumbelBatch = draws.ToArray();
            try
            {
                _ = ForwardBatchFused(requests);
            }
            finally
            {
                _pendingGumbelBatch = null;
            }
            if (_gumbelBatchHostRows != n)
                throw new InvalidOperationException($"The batched device Gumbel draw covered {_gumbelBatchHostRows} of {n} rows.");
            _gumbelBatchHost.AsSpan(0, n).CopyTo(sampledIds);
            DeviceGumbelFusedBatches++;
            return;
        }

        for (int i = 0; i < n; i++)
        {
            var r = requests[i];
            _gumbelHostRows = 0;
            _pendingGumbel = new GumbelRequest(draws[i].Seed, draws[i].TokenIndex, draws[i].InvTemperature);
            try
            {
                using var placeholder = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.GdnState);
            }
            finally
            {
                _pendingGumbel = null;
            }
            if (_gumbelHostRows != 1)
                throw new InvalidOperationException("The device Gumbel draw did not cover the row (the forward took a path without the hook).");
            sampledIds[i] = _gumbelHost[0];
        }
        DeviceGumbelSerialBatches++;
    }
}
