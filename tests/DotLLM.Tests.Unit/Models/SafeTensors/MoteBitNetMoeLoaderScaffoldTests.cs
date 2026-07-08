using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// SCAFFOLD (skipped) for the remaining dotLLM-side work needed to LOAD + FORWARD a
/// trained identity-MoTE BitNet exported by trackM-mote's <c>scripts/lora/mote_export.py</c>.
///
/// The exported checkpoint is a self-contained HF-style directory (config.json +
/// model.safetensors, bf16 master weights). The CONFIG already loads with no changes
/// (see <see cref="HfConfigExtractorMoteTests"/>). The WEIGHT-load + forward path does
/// not yet exist because a BitNet-MoE layer differs from dotLLM's SwiGLU MoE in three
/// numerically load-bearing ways. Each skipped test names one build item and the exact
/// seam to touch. Un-skip + implement TDD, keeping BitNet/MoE/I2S/Safetensors green.
///
/// Design note: .planning/2026-07-08-mote-dotllm-export-design.md
/// I2_S MoE expert kernel already exists on branch issue/i2s-moe-kernel
/// (<c>src/DotLLM.Cpu/Kernels/MatMul.I2S.cs</c> → <c>MoeIndexedMatmulI2_S</c>) — merge/rebase it in.
/// </summary>
public sealed class MoteBitNetMoeLoaderScaffoldTests
{
    // BUILD ITEM 1 — a BitNet-MoE layer loader.
    // src/DotLLM.Models/Architectures/TransformerWeightsSafetensors.cs:
    //   LoadBitNetLayer currently passes moe:null and always loads dense
    //   mlp.{gate,up,down}_proj. When config.Moe?.IsMoeLayer(i) is true, dispatch to a
    //   new LoadBitNetMoeLayer that resolves, per expert e in [0..num_experts):
    //     model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight  (ResolveLinearAsI2S)
    //     model.layers.{i}.mlp.experts.{e}.ffn_sub_norm.weight         (ResolveNorm, [intermediate])
    //   and the router:
    //     model.layers.{i}.mlp.gate.weight  ([num_experts, hidden], F32)
    //     model.layers.{i}.mlp.gate.bias    ([num_experts], F32)  <-- NEW tensor
    [Fact(Skip = "Build item 1: LoadBitNetMoeLayer (per-expert I2_S FFN + ffn_sub_norm + router weight/bias).")]
    public void BitNetMoeLayer_LoadsPerExpertTernaryFfnAndRouter() { }

    // BUILD ITEM 2 — router bias in MoeLayerWeights + Route().
    // src/DotLLM.Models/Architectures/TransformerWeights.cs (MoeLayerWeights): add
    //   GateBias (float[]?, [num_experts]).
    // src/DotLLM.Cpu/Kernels/MoeSwiGluMlp.cs (Route): add bias to the router GEMV logits
    //   before softmax/top-k. With top_k=1 + NormTopKProb the gate weight is 1.0, so the
    //   bias only shifts the argmax expert SELECTION — but it is REQUIRED for correct
    //   selection (cannot be folded into the weight).
    [Fact(Skip = "Build item 2: additive router bias in MoE Route() (shifts top-1 argmax).")]
    public void MoeRoute_AppliesRouterBias() { }

    // BUILD ITEM 3 — BitNet expert FFN semantics (relu2 + per-expert ffn_sub_norm).
    // src/DotLLM.Cpu/Kernels/MoeSwiGluMlp.cs assumes SwiGLU: silu(gate)*up -> down, no
    //   sub-norm. A BitNet expert is: down( ffn_sub_norm( relu2(gate(x)) * up(x) ) ) with
    //   per-BitLinear activation+weight quant. Add a BitNet-MoE expert path that (a) uses
    //   config.ActivationFunction (ReluSquared) and (b) applies the per-expert ffn_sub_norm
    //   RMSNorm before down_proj, dispatching experts through the I2_S kernel
    //   (MoeIndexedMatmulI2_S). The skip expert (index 0) has an all-zero down_proj that
    //   packs to I2_S zeros and outputs exactly 0 — no special-casing needed.
    [Fact(Skip = "Build item 3: BitNet-MoE expert forward (relu2 + per-expert ffn_sub_norm + I2_S experts).")]
    public void BitNetMoeExpert_UsesRelu2AndFfnSubNorm() { }

    // BUILD ITEM 4 — end-to-end parity gate.
    // Load a tiny exported identity-MoTE (produced by mote_export.py --self-test style
    // fixture) on CPU and assert logits match the PyTorch reference within tol. This is
    // the acceptance test that closes the bridge; it depends on items 1-3.
    [Fact(Skip = "Build item 4: end-to-end CPU logit parity vs the PyTorch identity-MoTE reference.")]
    public void ExportedIdentityMote_ForwardMatchesReferenceLogits() { }
}
