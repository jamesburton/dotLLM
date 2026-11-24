using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.ChunkReuse;
using Xunit;

namespace DotLLM.Tests.Unit.Diagnostics;

/// <summary>CPU model of one synthetic family for the #1920 tap suite (attention-only Llama, GDN-hybrid dense, GDN-hybrid MoE).</summary>
internal sealed class CpuHiddenStateTapFixture : HiddenStateTapFixture
{
    private readonly string _dir;
    private readonly GgufFile _gguf;
    private readonly ModelConfig _config;
    private readonly string _family;

    public CpuHiddenStateTapFixture(string family, bool withMtp = false)
    {
        _family = family;
        _dir = Path.Combine(Path.GetTempPath(), $"dotllm-taps-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, "m.gguf");
        switch (family)
        {
            case HiddenStateTapBackendSuite.Llama:
                File.WriteAllBytes(path, SyntheticDenseGguf.Build("llama"));
                Prompt = [3, 9, 14, 5, 21, 8, 30, 2];
                break;
            case HiddenStateTapBackendSuite.HybridDense:
                SyntheticQwen35HybridDenseMtpGguf.Write(path, withMtp: withMtp, contextLength: 256);
                Prompt = [3, 4, 5, 6, 3, 7, 5, 4];
                break;
            case HiddenStateTapBackendSuite.MoeHybrid:
                SyntheticQwen35MoeGguf.Write(path, contextLength: 256);
                Prompt = [3, 4, 5, 6, 3, 7, 5, 4];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(family));
        }
        _gguf = GgufFile.Open(path);
        _config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        Model = ModelLoader.CreateCpuModelFromGguf(_gguf, _config);
    }

    public override IModel Model { get; }
    public override int NumLayers => _config.NumLayers;
    public override int Hidden => _config.HiddenSize;
    public override int Vocab => _config.VocabSize;
    public override int[] Prompt { get; }
    public override bool ExpectsPackedBatchFallback => _family == HiddenStateTapBackendSuite.Llama;

    /// <summary>Steady-state managed bytes of 16 untapped decode forwards on this thread (minimum of three windows after a long warm-up).
    /// <b>MoE-hybrid has NO absolute baseline on CPU:</b> its allocation (router scratch, parallel-region closures) is 21632 B in a clean process but
    /// 26240 B (+288 B per decode) once <c>SolutionScannerTests</c> (or a similar heavy test) has run earlier in the same process, deterministically in
    /// both cases, so an absolute figure fails depending on test order, independent of the tap code. The unattached path is the same registry read
    /// as on the other families, whose absolute baselines (1664 B) hold in every order and die on an allocation mutant; the MoE unattached path
    /// is still covered by the no-dispatch/no-counter, perturbation and abort tests.</summary>
    public override long? BaselineAllocBytesPer16Decodes => _family == HiddenStateTapBackendSuite.MoeHybrid ? null : 1664;

    public override IKvCache NewKv(int maxLen = 96)
        => new SimpleKvCache(KvGeometry.FromConfig(_config, Model.SupportsCompactSlidingKv), maxLen);

    public override IKvCache? NewWindowedKv(int maxLen)
        => _family != HiddenStateTapBackendSuite.Llama ? null
            : new SimpleKvCache(KvGeometry.FromConfig(_config with { SlidingWindowSize = 8, SlidingWindowPattern = 2 }, compactSlidingWindows: true), maxLen);

    private readonly List<IModel> _extra = [];

    public override float[] OutputNormWeight => ReadF32Row(_gguf, "output_norm.weight", 0, _config.HiddenSize);
    public override float NormEps => _config.NormEpsilon;
    public override float[] EmbeddingRow(int token) => ReadF32Row(_gguf, "token_embd.weight", token, _config.HiddenSize);

    public override DotLLM.Core.Diagnostics.IHiddenStateTapHost? NewSlidingWindowConfigHost()
    {
        if (_family != HiddenStateTapBackendSuite.Llama) return null;
        var m = ModelLoader.CreateCpuModelFromGguf(_gguf, _config with { SlidingWindowSize = 8, SlidingWindowPattern = 2 });
        _extra.Add(m);
        return (DotLLM.Core.Diagnostics.IHiddenStateTapHost)m;
    }

    public override float[] KvBytes(IKvCache kv)
    {
        var cache = (SimpleKvCache)kv;
        var all = new List<float>();
        for (int layer = 0; ; layer++)
        {
            try
            {
                all.AddRange(cache.KeysSpan(layer).ToArray());
                all.AddRange(cache.ValuesSpan(layer).ToArray());
            }
            catch (ArgumentOutOfRangeException) { break; }
        }
        return [.. all];
    }

    public override void Dispose()
    {
        foreach (var m in _extra) m.Dispose();
        Model.Dispose();
        _gguf.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }
}

/// <summary>The #1920 tap suite on the CPU backend (all three families).</summary>
[Collection("HiddenStateTaps")]
public sealed class HiddenStateTapCpuTests : HiddenStateTapBackendSuite
{
    protected override HiddenStateTapFixture Open(string family) => new CpuHiddenStateTapFixture(family);
    protected override HiddenStateTapFixture? OpenHybridDenseWithMtp() => new CpuHiddenStateTapFixture(HybridDense, withMtp: true);
}
