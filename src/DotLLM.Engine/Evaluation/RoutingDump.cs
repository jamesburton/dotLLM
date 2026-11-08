using System.Buffers;
using System.Text.Json;
using DotLLM.Cpu;

namespace DotLLM.Engine.Evaluation;

/// <summary>
/// Writes the MoE router's decisions per document (<c>dotllm perplexity --dump-routing</c>, #1706): for each document, each
/// MoE layer and every input position, the selected top-k expert ids and the gate weights the model multiplied them by.
/// </summary>
/// <remarks>
/// <para>Output, one JSON object per document:
/// <c>{"doc":0,"n_tokens":N,"top_k":K,"layers":[{"layer":L,"experts":[[e0..eK-1],...N rows],"weights":[[w0..wK-1],...N rows]},...]}</c>.
/// Rows cover ALL <c>N</c> fed positions (the last position's logits are unscored but it still routes). Slot order is
/// descending router score. Weights are the final gate weights (after softmax, optional top-k renormalisation and scale).
/// Only layers that ran an MoE FFN appear (dense layers of a mixed model are absent).</para>
/// <para>Armed per document via <see cref="MoeRoutingProbe"/>; a layer that does not report exactly <c>N</c> positions
/// (an MoE path that bypasses the shared router, e.g. gpt-oss quantised experts) is an error, never a silent gap.</para>
/// </remarks>
public sealed class RoutingDump : IDocumentObserver, MoeRoutingProbe.ISink, IDisposable
{
    private readonly TextWriter _out;
    private readonly int _expectedLayers;
    private readonly SortedDictionary<int, LayerRows> _layers = [];
    private int _tokens;
    private int _topK;

    private sealed class LayerRows
    {
        public readonly List<int> Experts = [];
        public readonly List<float> Weights = [];
        public int Positions;
    }

    /// <summary>Creates a dump writing to <paramref name="output"/>.</summary>
    /// <param name="output">Destination for the JSONL lines.</param>
    /// <param name="expectedMoeLayers">Number of MoE layers the model should report per document (checked at document end); 0 = do not check the count.</param>
    public RoutingDump(TextWriter output, int expectedMoeLayers = 0)
    {
        _out = output;
        _expectedLayers = expectedMoeLayers;
    }

    /// <inheritdoc/>
    public void BeginDocument(int doc, ReadOnlySpan<int> tokens)
    {
        _layers.Clear();
        _tokens = tokens.Length;
        MoeRoutingProbe.Sink = this;
    }

    /// <inheritdoc/>
    public void Record(int layer, int seqLen, int topK, ReadOnlySpan<int> experts, ReadOnlySpan<float> weights)
    {
        if (layer < 0)
            throw new InvalidOperationException("The MoE router reported no layer index; this architecture does not support --dump-routing.");
        if (!_layers.TryGetValue(layer, out LayerRows? rows)) _layers[layer] = rows = new LayerRows();
        _topK = topK;
        for (int i = 0; i < experts.Length; i++) rows.Experts.Add(experts[i]);
        for (int i = 0; i < weights.Length; i++) rows.Weights.Add(weights[i]);
        rows.Positions += seqLen;
    }

    /// <inheritdoc/>
    public void EndDocument(int doc)
    {
        MoeRoutingProbe.Sink = null;
        if (_tokens < 2) { WriteDoc(doc); return; }   // no forward ran (a document needs >= 2 tokens to score)
        if (_layers.Count == 0)
            throw new InvalidOperationException(
                "No MoE routing was captured: the model has no MoE layers, or its MoE path bypasses the shared router (--dump-routing is for MoE models on the CPU transformer).");
        if (_expectedLayers > 0 && _layers.Count != _expectedLayers)
            throw new InvalidOperationException($"Document {doc}: {_layers.Count} MoE layers reported routing, expected {_expectedLayers}.");
        foreach (var (layer, rows) in _layers)
            if (rows.Positions != _tokens)
                throw new InvalidOperationException($"Document {doc}, layer {layer}: routing covers {rows.Positions} positions, expected {_tokens}.");
        WriteDoc(doc);
    }

    private void WriteDoc(int doc)
    {
        var buf = new ArrayBufferWriter<byte>(1 << 16);
        using (var w = new Utf8JsonWriter(buf))
        {
            w.WriteStartObject();
            w.WriteNumber("doc", doc);
            w.WriteNumber("n_tokens", _tokens);
            w.WriteNumber("top_k", _topK);
            w.WriteStartArray("layers");
            foreach (var (layer, rows) in _layers)
            {
                w.WriteStartObject();
                w.WriteNumber("layer", layer);
                w.WriteStartArray("experts");
                for (int t = 0; t < rows.Positions; t++)
                {
                    w.WriteStartArray();
                    for (int k = 0; k < _topK; k++) w.WriteNumberValue(rows.Experts[t * _topK + k]);
                    w.WriteEndArray();
                }
                w.WriteEndArray();
                w.WriteStartArray("weights");
                for (int t = 0; t < rows.Positions; t++)
                {
                    w.WriteStartArray();
                    for (int k = 0; k < _topK; k++) w.WriteNumberValue(rows.Weights[t * _topK + k]);
                    w.WriteEndArray();
                }
                w.WriteEndArray();
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteEndObject();
        }
        _out.WriteLine(System.Text.Encoding.UTF8.GetString(buf.WrittenSpan));
        _out.Flush();
    }

    /// <inheritdoc/>
    public void Dispose() => MoeRoutingProbe.Sink = null;
}
