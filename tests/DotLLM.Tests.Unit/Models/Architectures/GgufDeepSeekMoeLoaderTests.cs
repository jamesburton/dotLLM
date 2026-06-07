using NativeMemory = System.Runtime.InteropServices.NativeMemory;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Tests the DeepSeek-V2/V3 GGUF MoE 3D-stacked-expert tensor loader
/// (<see cref="TransformerWeights.LoadDeepSeekMoeLayer"/>). Focused
/// white-box test: builds a synthetic 3D fused-experts tensor in memory,
/// invokes the loader directly with a fake <see cref="GgufTensorDescriptor"/>
/// dictionary, and asserts each per-expert F32 output buffer carries the
/// expected slice of the source tensor.
/// </summary>
/// <remarks>
/// This validates the slicing math (offset / stride per expert) and the
/// per-expert F32 dequant path. End-to-end "real GGUF loads" coverage
/// arrives via the smoke test against the cached
/// <c>DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf</c> when the download
/// completes — that's task #11 follow-up.
/// </remarks>
public sealed class GgufDeepSeekMoeLoaderTests
{
    [Fact]
    public unsafe void LoadDeepSeekMoeLayer_F32_PerExpertSlicesMatchSource()
    {
        // Tiny shapes that exercise the slicing math:
        //   numExperts = 3, hidden = 4, intermediate = 5, no shared experts.
        const int numExperts = 3;
        const int hidden = 4;
        const int intermediate = 5;
        const int hiddenSize = hidden;
        const int numAttentionHeads = 1;     // unused by MoE path but required by ModelConfig
        const int numLayers = 1;

        // Build a contiguous F32 buffer for the fused gate-experts tensor:
        //   shape on disk = [hidden, intermediate, num_experts]  (innermost = hidden)
        // Each expert's slice covers `intermediate × hidden` consecutive floats.
        // We fill with a deterministic ramp so per-expert content is distinguishable.
        int gateElems = numExperts * intermediate * hidden;
        float[] gateSrc = new float[gateElems];
        for (int i = 0; i < gateElems; i++)
            gateSrc[i] = 0.01f * (i + 1);

        int upElems = gateElems;
        float[] upSrc = new float[upElems];
        for (int i = 0; i < upElems; i++)
            upSrc[i] = 1.0f + 0.01f * i;

        // ffn_down_exps: shape on disk = [intermediate, hidden, num_experts]
        // Per-expert slice = hidden × intermediate floats (M=hidden, K=intermediate).
        int downElems = numExperts * hidden * intermediate;
        float[] downSrc = new float[downElems];
        for (int i = 0; i < downElems; i++)
            downSrc[i] = -0.5f + 0.01f * i;

        // Router gate: [hidden, num_experts] (innermost = hidden).
        float[] routerSrc = new float[numExperts * hidden];
        for (int i = 0; i < routerSrc.Length; i++)
            routerSrc[i] = 0.05f * (i + 1);

        // Pin all source arrays for the lifetime of the test (the loader
        // dereferences nint pointers into them via Dequantize.ToFloat32).
        fixed (float* gatePtr = gateSrc)
        fixed (float* upPtr = upSrc)
        fixed (float* downPtr = downSrc)
        fixed (float* routerPtr = routerSrc)
        {
            // Construct fake GGUF descriptors. DataOffset is from `dataBase`,
            // and we'll pass each tensor's pointer as the dataBase + 0 offset
            // by lying with a per-tensor `dataBase` that's just the pointer.
            // (LoadDeepSeekMoeLayer reads `dataBase + DataOffset` — so we can
            // either centralise into a single concatenated blob or hand each
            // tensor its own dataBase. The latter is cheaper for a unit test.)
            var tensors = new Dictionary<string, GgufTensorDescriptor>
            {
                ["blk.0.ffn_gate_inp.weight"] = new GgufTensorDescriptor(
                    "blk.0.ffn_gate_inp.weight",
                    new TensorShape(hidden, numExperts), QuantizationType.F32, 0),
                ["blk.0.ffn_gate_exps.weight"] = new GgufTensorDescriptor(
                    "blk.0.ffn_gate_exps.weight",
                    new TensorShape(hidden, intermediate, numExperts), QuantizationType.F32, 0),
                ["blk.0.ffn_up_exps.weight"] = new GgufTensorDescriptor(
                    "blk.0.ffn_up_exps.weight",
                    new TensorShape(hidden, intermediate, numExperts), QuantizationType.F32, 0),
                ["blk.0.ffn_down_exps.weight"] = new GgufTensorDescriptor(
                    "blk.0.ffn_down_exps.weight",
                    new TensorShape(intermediate, hidden, numExperts), QuantizationType.F32, 0),
            };

            // We need a stable dataBase per tensor. Trick: pass per-tensor
            // dataBases by reconstructing the dictionary entries with the
            // appropriate base baked in as DataOffset = (long)(per-tensor-ptr - shared-base).
            // For unit testing simplicity, build a single dataBase pointing at
            // a heap-allocated combined buffer.
            int combinedFloats = routerSrc.Length + gateElems + upElems + downElems;
            float[] combined = new float[combinedFloats];
            int o = 0;
            Array.Copy(routerSrc, 0, combined, o, routerSrc.Length);
            ulong routerOffset = (ulong)(o * sizeof(float)); o += routerSrc.Length;
            Array.Copy(gateSrc, 0, combined, o, gateElems);
            ulong gateOffset = (ulong)(o * sizeof(float)); o += gateElems;
            Array.Copy(upSrc, 0, combined, o, upElems);
            ulong upOffset = (ulong)(o * sizeof(float)); o += upElems;
            Array.Copy(downSrc, 0, combined, o, downElems);
            ulong downOffset = (ulong)(o * sizeof(float));

            tensors["blk.0.ffn_gate_inp.weight"] = tensors["blk.0.ffn_gate_inp.weight"] with { DataOffset = routerOffset };
            tensors["blk.0.ffn_gate_exps.weight"] = tensors["blk.0.ffn_gate_exps.weight"] with { DataOffset = gateOffset };
            tensors["blk.0.ffn_up_exps.weight"] = tensors["blk.0.ffn_up_exps.weight"] with { DataOffset = upOffset };
            tensors["blk.0.ffn_down_exps.weight"] = tensors["blk.0.ffn_down_exps.weight"] with { DataOffset = downOffset };

            fixed (float* basePtr = combined)
            {
                nint dataBase = (nint)basePtr;

                // Build a minimal ModelConfig — only Moe / hidden / heads used by
                // the loader. The other fields are required by the record.
                var moe = new MoeConfig
                {
                    NumExperts = numExperts,
                    NumExpertsPerTok = 2,
                    MoeIntermediateSize = intermediate,
                    NormTopKProb = true,
                };
                var config = new ModelConfig
                {
                    Architecture = Architecture.DeepSeekV2,
                    VocabSize = 32,
                    HiddenSize = hiddenSize,
                    IntermediateSize = intermediate,
                    NumLayers = numLayers,
                    NumAttentionHeads = numAttentionHeads,
                    NumKvHeads = numAttentionHeads,
                    HeadDim = hiddenSize / numAttentionHeads,
                    MaxSequenceLength = 16,
                    Moe = moe,
                };

                var owned = new List<nint>();
                try
                {
                    var bundle = TransformerWeights.LoadDeepSeekMoeLayer(
                        layerIdx: 0, dataBase: dataBase,
                        tensors: tensors, config: config, owned: owned);

                    Assert.Equal(numExperts, bundle.NumExperts);
                    Assert.Equal(numExperts, bundle.W1.Length);
                    Assert.Equal(numExperts, bundle.W2.Length);
                    Assert.Equal(numExperts, bundle.W3.Length);

                    // Verify each expert's slice is the contiguous run of the
                    // source array starting at e * (intermediate * hidden).
                    for (int e = 0; e < numExperts; e++)
                    {
                        int slot = e * intermediate * hidden;
                        var expectedGate = new ReadOnlySpan<float>(gateSrc, slot, intermediate * hidden);
                        var actualGate = new ReadOnlySpan<float>((void*)bundle.W1[e], intermediate * hidden);
                        for (int i = 0; i < expectedGate.Length; i++)
                            Assert.Equal(expectedGate[i], actualGate[i]);

                        var expectedUp = new ReadOnlySpan<float>(upSrc, slot, intermediate * hidden);
                        var actualUp = new ReadOnlySpan<float>((void*)bundle.W3[e], intermediate * hidden);
                        for (int i = 0; i < expectedUp.Length; i++)
                            Assert.Equal(expectedUp[i], actualUp[i]);

                        // ffn_down_exps stride uses [intermediate, hidden, num_experts]
                        // — per-expert slice = hidden × intermediate.
                        int downSlot = e * hidden * intermediate;
                        var expectedDown = new ReadOnlySpan<float>(downSrc, downSlot, hidden * intermediate);
                        var actualDown = new ReadOnlySpan<float>((void*)bundle.W2[e], hidden * intermediate);
                        for (int i = 0; i < expectedDown.Length; i++)
                            Assert.Equal(expectedDown[i], actualDown[i]);
                    }

                    // Router landed in MoeLayerWeights.Gate — direct F32 copy.
                    Assert.Equal(numExperts * hidden, bundle.Gate.Length);
                    for (int i = 0; i < bundle.Gate.Length; i++)
                        Assert.Equal(routerSrc[i], bundle.Gate[i]);

                    // No shared experts in this fixture.
                    Assert.Equal(0, bundle.NumSharedExperts);
                    Assert.False(bundle.HasSharedExpert);
                }
                finally
                {
                    foreach (var ptr in owned)
                        if (ptr != 0) NativeMemory.AlignedFree((void*)ptr);
                }
            }
        }
    }

    [Fact]
    public unsafe void LoadDeepSeekMoeLayer_RejectsWrongShape()
    {
        // 2D tensor where 3D was expected — should throw with a clear message.
        var tensors = new Dictionary<string, GgufTensorDescriptor>
        {
            ["blk.0.ffn_gate_inp.weight"] = new GgufTensorDescriptor(
                "blk.0.ffn_gate_inp.weight",
                new TensorShape(4, 2), QuantizationType.F32, 0),
            ["blk.0.ffn_gate_exps.weight"] = new GgufTensorDescriptor(
                "blk.0.ffn_gate_exps.weight",
                new TensorShape(4, 5), QuantizationType.F32, 32),  // wrong: should be 3D
            ["blk.0.ffn_up_exps.weight"] = new GgufTensorDescriptor(
                "blk.0.ffn_up_exps.weight",
                new TensorShape(4, 5, 2), QuantizationType.F32, 0),
            ["blk.0.ffn_down_exps.weight"] = new GgufTensorDescriptor(
                "blk.0.ffn_down_exps.weight",
                new TensorShape(5, 4, 2), QuantizationType.F32, 0),
        };

        var moe = new MoeConfig
        {
            NumExperts = 2, NumExpertsPerTok = 1, MoeIntermediateSize = 5, NormTopKProb = true,
        };
        var config = new ModelConfig
        {
            Architecture = Architecture.DeepSeekV2,
            VocabSize = 32, HiddenSize = 4, IntermediateSize = 5,
            NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1,
            HeadDim = 4, MaxSequenceLength = 16,
            Moe = moe,
        };
        float[] dummy = new float[256];
        nint dataBase;
        fixed (float* p = dummy) { dataBase = (nint)p; }
        // The buffer is heap-allocated by the GC and may move, but we never
        // dereference dataBase here — the loader throws on shape validation
        // before any read.

        var owned = new List<nint>();
        try
        {
            Assert.Throws<InvalidDataException>(() =>
                TransformerWeights.LoadDeepSeekMoeLayer(
                    layerIdx: 0, dataBase: dataBase,
                    tensors: tensors, config: config, owned: owned));
        }
        finally
        {
            foreach (var ptr in owned)
                if (ptr != 0) NativeMemory.AlignedFree((void*)ptr);
        }
    }
}
