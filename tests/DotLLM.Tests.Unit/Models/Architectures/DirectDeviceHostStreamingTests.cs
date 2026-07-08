using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// GPU-free tests for the direct-to-device weight-streaming free contract added to
/// the GPU loaders. The actual host→device upload loop lives in
/// <c>CudaWeights.LoadFromGguf</c> and needs a CUDA device to run end-to-end
/// (covered by the existing CUDA safetensors / BitNet load tests, which skip when no
/// GPU is present). What is testable WITHOUT a GPU — and what these tests pin — is the
/// invariant the loop relies on: <see cref="TransformerWeights.TryReleaseOwnedHostAllocation"/>
/// frees each loader-owned host scratch buffer exactly once, never frees a memory-mapped
/// zero-copy view, is idempotent (no double-free), and leaves <see cref="TransformerWeights.Dispose"/>
/// safe to run afterwards.
///
/// <para>
/// Each test drives the release callback with the SAME per-layer slot visitation the CUDA
/// loop performs for a plain dense / BitNet layer (Q/K/V/O then Gate/Up/Down), so it
/// discriminates the streamed-free path from the legacy batch-free path: owned buffers are
/// released early, mmap views are not, and the count matches the number of owned allocations.
/// </para>
/// </summary>
public sealed class DirectDeviceHostStreamingTests : IDisposable
{
    private readonly string _scratch;

    public DirectDeviceHostStreamingTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-ddhs-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ── Llama-shaped dims (small; every dim a multiple of nothing special) ──
    private const int Hidden = 64;
    private const int NumHeads = 4;
    private const int HeadDim = 16;
    private const int Intermediate = 128;
    private const int Vocab = 32;

    // ── BitNet-shaped dims (hidden=128 so every linear's element count is a
    //    multiple of 128, as I2_S ternary packing requires) ──
    private const int BnHidden = 128;
    private const int BnHeads = 4;
    private const int BnKvHeads = 2;
    private const int BnHeadDim = 32;
    private const int BnIntermediate = 256;
    private const int BnVocab = 64;

    /// <summary>
    /// The exact per-layer slot visitation the CUDA upload loop performs for a plain
    /// dense / BitNet layer: Q/K/V/O (attention block) then Gate/Up/Down (dense FFN
    /// block). Returns the number of buffers actually freed (owned) and asserts no slot
    /// pointer is released more than once (which would betray aliasing / double-free).
    /// </summary>
    private static int SimulateStreamedUpload(TransformerWeights w)
    {
        int freed = 0;
        var releasedOnce = new HashSet<nint>();
        foreach (ref readonly var lw in w.Layers.AsSpan())
        {
            Span<nint> slots =
            [
                lw.QWeight, lw.KWeight, lw.VWeight, lw.OWeight,
                lw.GateWeight, lw.UpWeight, lw.DownWeight,
            ];
            foreach (nint p in slots)
            {
                if (w.TryReleaseOwnedHostAllocation(p))
                {
                    freed++;
                    Assert.True(p == nint.Zero || releasedOnce.Add(p),
                        $"Owned host pointer 0x{p:X} was stream-freed more than once.");
                }
            }
        }
        return freed;
    }

    [Fact]
    public void MixedLlama_StreamsOnlyOwnedUpcasts_MmapViewsUntouched()
    {
        // gate_proj is BF16 (→ owned F32 upcast); everything else is F32 (→ mmap view).
        var w = LoadLlamaWeights(numLayers: 3, gateAsBf16: true);
        try
        {
            // Exactly one owned allocation per layer (the gate upcast). q/k/v/o/up/down
            // are zero-copy mmap views and are NOT owned.
            Assert.Equal(3, w.LiveOwnedAllocationCount);

            // Releasing an mmap view is a no-op and never decrements the owned count.
            ref readonly var l0 = ref w.Layers[0];
            Assert.False(w.TryReleaseOwnedHostAllocation(l0.QWeight));   // F32 mmap
            Assert.False(w.TryReleaseOwnedHostAllocation(l0.DownWeight)); // F32 mmap
            Assert.False(w.TryReleaseOwnedHostAllocation(w.TokenEmbedWeight)); // F32 mmap embed
            Assert.Equal(3, w.LiveOwnedAllocationCount);

            // Simulated upload frees exactly the 3 owned gate upcasts, nothing else.
            int freed = SimulateStreamedUpload(w);
            Assert.Equal(3, freed);
            Assert.Equal(0, w.LiveOwnedAllocationCount);
        }
        finally
        {
            // Dispose after streamed release must not double-free.
            w.Dispose();
            w.Dispose(); // second dispose is safe
        }
    }

    [Fact]
    public void OwnedRelease_IsIdempotent_NoDoubleFree()
    {
        var w = LoadLlamaWeights(numLayers: 1, gateAsBf16: true);
        try
        {
            nint gate = w.Layers[0].GateWeight;
            Assert.True(w.TryReleaseOwnedHostAllocation(gate));  // first free
            Assert.False(w.TryReleaseOwnedHostAllocation(gate)); // idempotent — already gone
            Assert.False(w.TryReleaseOwnedHostAllocation(gate)); // still safe
            Assert.Equal(0, w.LiveOwnedAllocationCount);
        }
        finally
        {
            w.Dispose();
        }
    }

    [Fact]
    public void PureF32Llama_HasNoOwnedAllocations_StreamingIsNoOp()
    {
        // All-F32 fixture: every projection is a zero-copy mmap view. There is nothing
        // to stream-free, so the streamed path frees zero and disposal still works.
        var w = LoadLlamaWeights(numLayers: 2, gateAsBf16: false);
        try
        {
            Assert.Equal(0, w.LiveOwnedAllocationCount);
            Assert.Equal(0, SimulateStreamedUpload(w));
            Assert.Equal(0, w.LiveOwnedAllocationCount);
        }
        finally
        {
            w.Dispose();
        }
    }

    [Fact]
    public void BitNet_AllLinearsAreOwnedI2S_StreamsEachExactlyOnce()
    {
        const int numLayers = 2;
        var w = LoadBitNetWeights(numLayers);
        try
        {
            // Every linear projection (7 per layer) is quantized to an owned I2_S buffer;
            // the embedding stays a mmap F32 view. So owned == 7 * layers, nothing more.
            Assert.Equal(7 * numLayers, w.LiveOwnedAllocationCount);

            // Embedding is a mmap view — never freed by streaming.
            Assert.False(w.TryReleaseOwnedHostAllocation(w.TokenEmbedWeight));

            int freed = SimulateStreamedUpload(w);
            Assert.Equal(7 * numLayers, freed);
            Assert.Equal(0, w.LiveOwnedAllocationCount);
        }
        finally
        {
            // No double-free of the streamed I2_S buffers at disposal.
            w.Dispose();
        }
    }

    [Fact]
    public void DisposeWithoutStreaming_StillFreesAllOwned()
    {
        // Legacy batch behavior (callback null): nothing streamed, Dispose frees everything.
        var w = LoadBitNetWeights(numLayers: 1);
        Assert.Equal(7, w.LiveOwnedAllocationCount);
        w.Dispose();
        Assert.Equal(0, w.LiveOwnedAllocationCount);
    }

    // ────────────────────────── fixtures ──────────────────────────

    private TransformerWeights LoadLlamaWeights(int numLayers, bool gateAsBf16)
    {
        var rng = new Random(42);
        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [Vocab, Hidden], Rand(rng, Vocab * Hidden));
        b.AddFloat32("model.norm.weight", [Hidden], Ones(Hidden));
        b.AddFloat32("lm_head.weight", [Vocab, Hidden], Rand(rng, Vocab * Hidden));
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [Hidden], Ones(Hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [Hidden], Ones(Hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight", [NumHeads * HeadDim, Hidden], Rand(rng, NumHeads * HeadDim * Hidden));
            b.AddFloat32($"{p}.self_attn.k_proj.weight", [NumHeads * HeadDim, Hidden], Rand(rng, NumHeads * HeadDim * Hidden));
            b.AddFloat32($"{p}.self_attn.v_proj.weight", [NumHeads * HeadDim, Hidden], Rand(rng, NumHeads * HeadDim * Hidden));
            b.AddFloat32($"{p}.self_attn.o_proj.weight", [Hidden, NumHeads * HeadDim], Rand(rng, Hidden * NumHeads * HeadDim));
            if (gateAsBf16)
                AddBf16($"{p}.mlp.gate_proj.weight", [Intermediate, Hidden], b, rng);
            else
                b.AddFloat32($"{p}.mlp.gate_proj.weight", [Intermediate, Hidden], Rand(rng, Intermediate * Hidden));
            b.AddFloat32($"{p}.mlp.up_proj.weight", [Intermediate, Hidden], Rand(rng, Intermediate * Hidden));
            b.AddFloat32($"{p}.mlp.down_proj.weight", [Hidden, Intermediate], Rand(rng, Hidden * Intermediate));
        }
        string path = Path.Combine(_scratch, $"llama-{numLayers}-{gateAsBf16}.safetensors");
        b.WriteTo(path);

        var file = SafetensorsFile.Open(path);
        var config = new ModelConfig
        {
            Architecture = Architecture.Llama,
            VocabSize = Vocab,
            HiddenSize = Hidden,
            IntermediateSize = Intermediate,
            NumLayers = numLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm),
        };
        // The loader copies BF16 into owned scratch and wires F32 as mmap views; the file
        // must outlive the weights (mmap anchor). We dispose it at test end via Dispose of
        // the scratch dir; keep it alive by not disposing here (fine for a unit test's lifetime).
        return TransformerWeightsSafetensorsLoader.Load(file, config);
    }

    private TransformerWeights LoadBitNetWeights(int numLayers)
    {
        var rng = new Random(58);
        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [BnVocab, BnHidden], Rand(rng, BnVocab * BnHidden));
        b.AddFloat32("model.norm.weight", [BnHidden], Ones(BnHidden));
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [BnHidden], Ones(BnHidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [BnHidden], Ones(BnHidden));
            AddBf16($"{p}.self_attn.q_proj.weight", [BnHeads * BnHeadDim, BnHidden], b, rng);
            AddBf16($"{p}.self_attn.k_proj.weight", [BnKvHeads * BnHeadDim, BnHidden], b, rng);
            AddBf16($"{p}.self_attn.v_proj.weight", [BnKvHeads * BnHeadDim, BnHidden], b, rng);
            AddBf16($"{p}.self_attn.o_proj.weight", [BnHidden, BnHeads * BnHeadDim], b, rng);
            b.AddFloat32($"{p}.self_attn.attn_sub_norm.weight", [BnHidden], Ones(BnHidden));
            AddBf16($"{p}.mlp.gate_proj.weight", [BnIntermediate, BnHidden], b, rng);
            AddBf16($"{p}.mlp.up_proj.weight", [BnIntermediate, BnHidden], b, rng);
            AddBf16($"{p}.mlp.down_proj.weight", [BnHidden, BnIntermediate], b, rng);
            b.AddFloat32($"{p}.mlp.ffn_sub_norm.weight", [BnIntermediate], Ones(BnIntermediate));
        }
        string path = Path.Combine(_scratch, $"bitnet-{numLayers}.safetensors");
        b.WriteTo(path);

        var file = SafetensorsFile.Open(path);
        var config = new ModelConfig
        {
            Architecture = Architecture.BitNet,
            ActivationFunction = ActivationFunction.ReluSquared,
            VocabSize = BnVocab,
            HiddenSize = BnHidden,
            IntermediateSize = BnIntermediate,
            NumLayers = numLayers,
            NumAttentionHeads = BnHeads,
            NumKvHeads = BnKvHeads,
            HeadDim = BnHeadDim,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = true,
            RoPEConfig = new RoPEConfig(Theta: 500000.0f, DimensionCount: BnHeadDim, Type: RoPEType.Norm),
        };
        return TransformerWeightsSafetensorsLoader.Load(file, config);
    }

    private static void AddBf16(string name, int[] shape, SafetensorsFixtureBuilder b, Random rng)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        var bytes = new byte[n * 2];
        for (long i = 0; i < n; i++)
        {
            float v = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.05);
            uint bits = Unsafe.As<float, uint>(ref v);
            ushort bf16 = (ushort)(bits >> 16);
            bytes[i * 2] = (byte)(bf16 & 0xFF);
            bytes[i * 2 + 1] = (byte)(bf16 >> 8);
        }
        b.AddRaw(name, "BF16", shape, bytes);
    }

    private static float[] Rand(Random rng, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.05);
        return v;
    }

    private static float[] Ones(int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = 1.0f;
        return v;
    }
}
