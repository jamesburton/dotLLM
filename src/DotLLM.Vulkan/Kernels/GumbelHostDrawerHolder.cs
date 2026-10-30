using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Per-model owner of a lazily created <see cref="GumbelHostLogitsDrawer"/>, plus the <c>IModel</c> Gumbel members implemented once for the
/// Vulkan hosts without an in-place LM-head hook (Gemma4, Qwen4Exp, Qwen3-MoE-hybrid, Nemotron-H; #1435).
/// </summary>
internal sealed class GumbelHostDrawerHolder : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly IModel _model;
    private GumbelHostLogitsDrawer? _drawer;
    private bool _probed;
    private string? _spvDir;

    public GumbelHostDrawerHolder(VulkanDevice device, IModel model)
    {
        _device = device;
        _model = model;
    }

    /// <summary>True when the per-row shader is compiled and the draw is not switched off (<c>DOTLLM_VULKAN_DEVICE_GUMBEL=0</c>).</summary>
    public bool Available
    {
        get
        {
            if (!_probed)
            {
                _probed = true;
                try { _spvDir = VulkanModelLoader.ResolveSpvDir(); } catch (Exception) { _spvDir = null; }
            }
            return _spvDir is not null && GumbelHostLogitsDrawer.IsAvailable(_spvDir);
        }
    }

    /// <summary>True when the draw can also apply top-k (the index tie search covers vocabularies up to 2^20).</summary>
    public bool FilterAvailable => Available && _model.Config.VocabSize <= GumbelArgMaxRowsKernel.MaxFilteredVocab;

    /// <summary>The drawer, created on first use.</summary>
    internal GumbelHostLogitsDrawer Drawer => _drawer ??= new GumbelHostLogitsDrawer(_device, _spvDir ?? throw new InvalidOperationException("Gumbel shaders unavailable."));

    public void ForwardGumbelSampled(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, IKvCache? kvCache, IMtpState? mtpState,
        bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, GumbelFilter filter, Span<int> sampledIds)
    {
        if (!Available) throw new NotSupportedException($"{_model.GetType().Name}: device Gumbel sampling unavailable.");
        if (!filter.IsNone && !FilterAvailable) throw new NotSupportedException($"{_model.GetType().Name}: filtered device Gumbel sampling unavailable.");
        if (sampledIds.Length < tokenIds.Length) throw new ArgumentException("sampledIds is shorter than the row count.", nameof(sampledIds));
        Drawer.ForwardGumbelSampled(_model, tokenIds, positions, kvCache, mtpState, recurrentRowSnapshots, seed, firstTokenIndex,
            invTemperature, filter, sampledIds);
    }

    public void ForwardBatchGumbelSampled(IReadOnlyList<SequenceForwardRequest> requests, int deviceId, ReadOnlySpan<GumbelDraw> draws,
        Span<int> sampledIds)
    {
        if (!Available) throw new NotSupportedException($"{_model.GetType().Name}: device Gumbel sampling unavailable.");
        int n = requests.Count;
        if (draws.Length != n) throw new ArgumentException("One draw per request is required.", nameof(draws));
        if (sampledIds.Length < n) throw new ArgumentException("sampledIds is shorter than the request count.", nameof(sampledIds));
        for (int i = 0; i < n; i++)
        {
            if (requests[i].TokenIds.Length != 1)
                throw new ArgumentException($"Request {i}: batched device Gumbel sampling takes single-token decode requests.", nameof(requests));
            if (draws[i].TopK != 0 || draws[i].MinP > 0f)
                if (!FilterAvailable) throw new NotSupportedException($"{_model.GetType().Name}: filtered device Gumbel sampling unavailable.");
        }
        if (n == 0) return;
        var logits = _model.ForwardBatch(requests, deviceId);
        try
        {
            Drawer.DrawBatch(logits, _model.Config.VocabSize, draws, sampledIds);
        }
        finally
        {
            foreach (ITensor t in logits) t.Dispose();
        }
    }

    public void Dispose() => _drawer?.Dispose();
}
