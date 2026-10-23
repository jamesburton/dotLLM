using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;

namespace DotLLM.Tests.Unit.Models.Lfm2Moe;

internal sealed class Lfm2TestFixture : IDisposable
{
    private readonly string _path;
    public string Path => _path;
    public ModelConfig Config { get; private set; } = null!;
    private readonly GgufFile _gguf;
    public Lfm2MoeTransformerModel Model { get; }
    public Lfm2TestFixture(int hiddenSize = 8)
    {
        int headDim = hiddenSize / 2;
        using var data = new GgufTestData();
        var random = new Random(1028);
        void Add(string name, int[] shape, bool norm = false)
        {
            int n = shape.Aggregate(1, (a, b) => a * b);
            var values = Enumerable.Range(0, n).Select(_ => norm ? 1f : (float)(random.NextDouble() - .5) * .4f).ToArray();
            data.AddTensor(name, shape, 0, MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
        }
        Add("token_embd.weight", [hiddenSize, 8]); Add("token_embd_norm.weight", [hiddenSize], true);
        for (int i = 0; i < 3; i++)
        {
            string p = $"blk.{i}";
            Add($"{p}.attn_norm.weight", [hiddenSize], true); Add($"{p}.ffn_norm.weight", [hiddenSize], true);
            if (i != 1)
            {
                Add($"{p}.shortconv.in_proj.weight", [hiddenSize, 3 * hiddenSize]); Add($"{p}.shortconv.out_proj.weight", [hiddenSize, hiddenSize]);
                Add($"{p}.shortconv.conv.weight", [3, hiddenSize]);
            }
            else
            {
                Add($"{p}.attn_q.weight", [hiddenSize, hiddenSize]); Add($"{p}.attn_k.weight", [hiddenSize, headDim]);
                Add($"{p}.attn_v.weight", [hiddenSize, headDim]); Add($"{p}.attn_output.weight", [hiddenSize, hiddenSize]);
                Add($"{p}.attn_q_norm.weight", [headDim], true); Add($"{p}.attn_k_norm.weight", [headDim], true);
            }
            Add($"{p}.ffn_gate.weight", [hiddenSize, 12]); Add($"{p}.ffn_up.weight", [hiddenSize, 12]); Add($"{p}.ffn_down.weight", [12, hiddenSize]);
        }
        _path = data.WriteToTempFile(); _gguf = GgufFile.Open(_path);
        Config = new ModelConfig
        {
            Architecture = DotLLM.Core.Configuration.Architecture.Lfm2Moe, VocabSize = 8, HiddenSize = hiddenSize, IntermediateSize = 12,
            NumLayers = 3, NumAttentionHeads = 2, NumKvHeads = 1, HeadDim = headDim, MaxSequenceLength = 16,
            ShortConvCacheLength = 3,
            HybridLayout = new HybridLayerLayout { LayerKind = [HybridLayerKind.ShortConv, HybridLayerKind.Attention, HybridLayerKind.ShortConv], HeadCountKv = [0, 1, 0], FeedForwardLength = [12, 12, 12] },
            Moe = new MoeConfig { NumExperts = 2, NumExpertsPerTok = 1, MoeIntermediateSize = 12, MlpOnlyLayers = [0, 1, 2] }
        };
        Model = Lfm2MoeTransformerModel.LoadFromGguf(_gguf, Config);
    }
    public void Dispose() { Model.Dispose(); _gguf.Dispose(); File.Delete(_path); }
}
