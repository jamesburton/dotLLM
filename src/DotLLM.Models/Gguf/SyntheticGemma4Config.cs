using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Dimensions + per-tensor-class quant types for a <see cref="SyntheticGemma4Gguf"/> fixture.
/// Every field has a TINY default (see <see cref="SyntheticGemma4Gguf.Tiny"/>); the
/// <see cref="SyntheticGemma4Gguf.Bench"/> preset scales the same shape up. All quantized-row
/// K dimensions are kept block-valid for their format (Q4_K = 256, Q5_1/Q5_0/Q8_0 = 32).
/// </summary>
public sealed record SyntheticGemma4Config
{
    /// <summary>Number of transformer layers. Layers matching <see cref="GlobalLayerStride"/> are global/V-less.</summary>
    public int BlockCount { get; init; } = 6;

    /// <summary>Model hidden size. Multiple of 256 so Q4_K gate_up rows (K = hidden) are block-valid.</summary>
    public int HiddenSize { get; init; } = 256;

    /// <summary>Query head count (all layers).</summary>
    public int HeadCount { get; init; } = 4;

    /// <summary>Sliding (SWA) per-head dim. <c>HeadCount*SlidingHeadDim</c> and <c>SlidingKvHeads*SlidingHeadDim</c> must be %32.</summary>
    public int SlidingHeadDim { get; init; } = 16;

    /// <summary>Global (full-attention) per-head dim — the dual head dim. %32 with the head counts.</summary>
    public int GlobalHeadDim { get; init; } = 32;

    /// <summary>KV head count on sliding layers.</summary>
    public int SlidingKvHeads { get; init; } = 2;

    /// <summary>KV head count on global layers (the smaller, V-less count).</summary>
    public int GlobalKvHeads { get; init; } = 1;

    /// <summary>Dense ("shared expert") FFN width. %32 for Q8_0 dense rows.</summary>
    public int DenseFeedForward { get; init; } = 64;

    /// <summary>Per-expert FFN width (Ie). %32 so Q5_1 down rows (K = Ie) are block-valid.</summary>
    public int ExpertFeedForward { get; init; } = 32;

    /// <summary>Number of routed experts.</summary>
    public int ExpertCount { get; init; } = 8;

    /// <summary>Experts selected per token (top-k).</summary>
    public int ExpertUsedCount { get; init; } = 2;

    /// <summary>Vocabulary size (token_embd M dim). %32 keeps a Q8_0 token_embd row block-valid.</summary>
    public int VocabSize { get; init; } = 256;

    /// <summary>Max sequence / context length metadata.</summary>
    public int ContextLength { get; init; } = 512;

    /// <summary>Sliding-window span (positions).</summary>
    public int SlidingWindow { get; init; } = 8;

    /// <summary>RMSNorm epsilon.</summary>
    public float NormEpsilon { get; init; } = 1e-6f;

    /// <summary>Global-layer RoPE base.</summary>
    public float GlobalRopeFreqBase { get; init; } = 1_000_000f;

    /// <summary>Sliding-layer RoPE base.</summary>
    public float SlidingRopeFreqBase { get; init; } = 10_000f;

    /// <summary>Final-logit soft-cap (Gemma4 = 30).</summary>
    public float FinalLogitSoftcap { get; init; } = 30f;

    /// <summary>Diffusion canvas length.</summary>
    public int CanvasLength { get; init; } = 8;

    /// <summary>
    /// Every Nth layer (counting from the end pattern) is a global/V-less layer. Default 6 ⇒
    /// the LAST layer of the tiny 6-layer model is global (layers 0-4 sliding, layer 5 global),
    /// matching the design's <c>sliding_window_pattern [1,1,1,1,1,0]</c>.
    /// </summary>
    public int GlobalLayerStride { get; init; } = 6;

    // ── Special token ids ──
    /// <summary>Beginning-of-sequence token id (Gemma uses 2).</summary>
    public int BosTokenId { get; init; } = 2;
    /// <summary>End-of-sequence token id.</summary>
    public int EosTokenId { get; init; } = 1;
    /// <summary>Unknown token id.</summary>
    public int UnknownTokenId { get; init; } = 0;
    /// <summary>Diffusion mask token id (canvas absorbing state).</summary>
    public int MaskTokenId { get; init; } = 4;

    // ── Per-tensor-class quant types (default = real-model mix) ──
    /// <summary>Attention q/k/v/o projections. Default Q8_0.</summary>
    public QuantizationType AttnQuant { get; init; } = QuantizationType.Q8_0;

    /// <summary>Dense FFN gate/up/down. Default Q8_0.</summary>
    public QuantizationType DenseFfnQuant { get; init; } = QuantizationType.Q8_0;

    /// <summary>Fused gate_up experts. Default Q4_K.</summary>
    public QuantizationType ExpertGateUpQuant { get; init; } = QuantizationType.Q4_K;

    /// <summary>Down experts. Default Q5_1.</summary>
    public QuantizationType ExpertDownQuant { get; init; } = QuantizationType.Q5_1;

    /// <summary>Token embedding (tied LM head). Default Q8_0.</summary>
    public QuantizationType TokenEmbdQuant { get; init; } = QuantizationType.Q8_0;

    /// <summary>Diffusion self-conditioning projections. Default Q8_0.</summary>
    public QuantizationType SelfCondQuant { get; init; } = QuantizationType.Q8_0;

    /// <summary>
    /// An all-F32 preset of this config (pure-correctness golden; no quant error). Keeps the
    /// same dims, swaps every weight class to F32.
    /// </summary>
    public SyntheticGemma4Config AllF32() => this with
    {
        AttnQuant = QuantizationType.F32,
        DenseFfnQuant = QuantizationType.F32,
        ExpertGateUpQuant = QuantizationType.F32,
        ExpertDownQuant = QuantizationType.F32,
        TokenEmbdQuant = QuantizationType.F32,
        SelfCondQuant = QuantizationType.F32,
    };

    /// <summary>True when layer <paramref name="i"/> is a global (full-attention, V-less) layer.</summary>
    public bool IsGlobalLayer(int i) => ((i + 1) % GlobalLayerStride) == 0;

    /// <summary>Validates the dims satisfy each quant format's block-size constraint.</summary>
    public void Validate()
    {
        void Req(bool ok, string msg) { if (!ok) throw new InvalidOperationException($"SyntheticGemma4Config: {msg}"); }
        static int Block(QuantizationType qt) => qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K or QuantizationType.Q2_K or QuantizationType.Q3_K ? 256
            : qt is QuantizationType.F32 or QuantizationType.F16 ? 1 : 32;

        Req(HeadCount > 0 && HiddenSize > 0 && BlockCount > 0, "core dims must be positive.");
        Req(ExpertUsedCount <= ExpertCount, "ExpertUsedCount must be <= ExpertCount.");

        // Attention K dims (Q8_0 default → %32). q/k/v rows have K = hidden; o has K = HeadCount*hd.
        Req(HiddenSize % Block(AttnQuant) == 0, $"HiddenSize {HiddenSize} not block-valid for AttnQuant {AttnQuant}.");
        Req((HeadCount * SlidingHeadDim) % Block(AttnQuant) == 0, "sliding attn_output K not block-valid.");
        Req((HeadCount * GlobalHeadDim) % Block(AttnQuant) == 0, "global attn_output K not block-valid.");

        // Dense FFN: gate/up K = hidden, down K = ff.
        Req(HiddenSize % Block(DenseFfnQuant) == 0, "HiddenSize not block-valid for DenseFfnQuant.");
        Req(DenseFeedForward % Block(DenseFfnQuant) == 0, "DenseFeedForward not block-valid for DenseFfnQuant.");

        // Experts: gate_up K = hidden (Q4_K → %256); down K = Ie (Q5_1 → %32).
        Req(HiddenSize % Block(ExpertGateUpQuant) == 0, $"HiddenSize {HiddenSize} not block-valid for ExpertGateUpQuant {ExpertGateUpQuant} (needs %{Block(ExpertGateUpQuant)}).");
        Req(ExpertFeedForward % Block(ExpertDownQuant) == 0, $"ExpertFeedForward {ExpertFeedForward} not block-valid for ExpertDownQuant {ExpertDownQuant} (needs %{Block(ExpertDownQuant)}).");

        // token_embd row K = hidden.
        Req(HiddenSize % Block(TokenEmbdQuant) == 0, "HiddenSize not block-valid for TokenEmbdQuant.");
        Req(VocabSize > MaskTokenId && VocabSize > BosTokenId && VocabSize > EosTokenId, "special token ids must be < VocabSize.");
    }
}
