namespace DotLLM.Core.Models;

/// <summary>
/// Sequence-pooling strategy used to reduce a per-token hidden-state matrix
/// <c>[seqLen, hiddenSize]</c> to a single embedding vector <c>[hiddenSize]</c>.
/// </summary>
/// <remarks>
/// <para>Numeric values are deliberately identical to llama.cpp's
/// <c>enum llama_pooling_type</c> (<c>include/llama.h</c>), because the GGUF key
/// <c>{arch}.pooling_type</c> stores that enum's raw value:</para>
/// <list type="bullet">
///   <item><description><c>LLAMA_POOLING_TYPE_NONE = 0</c></description></item>
///   <item><description><c>LLAMA_POOLING_TYPE_MEAN = 1</c></description></item>
///   <item><description><c>LLAMA_POOLING_TYPE_CLS  = 2</c></description></item>
///   <item><description><c>LLAMA_POOLING_TYPE_LAST = 3</c></description></item>
///   <item><description><c>LLAMA_POOLING_TYPE_RANK = 4</c> (reranking head — not supported here)</description></item>
/// </list>
/// <para>Per llama.cpp (<c>src/llama-graph.cpp</c>, <c>llm_graph_context::build_pooling</c>)
/// pooling is applied to <c>result_norm</c> — the hidden state <b>after</b> the final
/// output norm and <b>before</b> the LM head. dotLLM matches that placement exactly.</para>
/// </remarks>
public enum PoolingType
{
    /// <summary>No pooling: one vector per token. Not representable in the OpenAI embeddings response.</summary>
    None = 0,

    /// <summary>Arithmetic mean over all token hidden states (uniform <c>1/n</c> weights, as llama.cpp's <c>build_inp_mean</c>).</summary>
    Mean = 1,

    /// <summary>The hidden state of the first token in the sequence (BERT-style <c>[CLS]</c>).</summary>
    Cls = 2,

    /// <summary>The hidden state of the last token in the sequence. The natural choice for causal decoders.</summary>
    Last = 3,

    /// <summary>Reranking classification head. Not implemented.</summary>
    Rank = 4,
}
