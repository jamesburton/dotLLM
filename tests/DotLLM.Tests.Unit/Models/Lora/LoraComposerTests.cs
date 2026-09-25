using System.Runtime.InteropServices;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Lora;

public sealed unsafe class LoraComposerTests
{
    // Minimal single-layer config with hidden==qOut so q_proj dims are square-ish.
    private static ModelConfig Cfg(int hidden, int heads, int headDim) => new()
    {
        Architecture = Architecture.Llama,
        NumLayers = 1, HiddenSize = hidden, NumAttentionHeads = heads, NumKvHeads = heads,
        HeadDim = headDim, IntermediateSize = hidden, VocabSize = 32, MaxSequenceLength = 8,
    };

    // Build a 1-layer F32 adapter targeting q_proj with deterministic small weights.
    private static LoraAdapter MakeAdapter(string name, int rank, float alpha, int inDim, int outDim, int seed)
    {
        var a = new LoraAdapter(name, rank, alpha, new[] { "q_proj" });
        nint b = LoraAdapter.AllocAligned((long)rank * inDim);   // [rank, inDim]
        nint au = LoraAdapter.AllocAligned((long)outDim * rank); // [outDim, rank]
        var rng = new Random(seed);
        float* bp = (float*)b; for (long i = 0; i < (long)rank * inDim; i++) bp[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
        float* ap = (float*)au; for (long i = 0; i < (long)outDim * rank; i++) ap[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
        a.AddLayerWeights(0, "q_proj", new LoraLayerWeights(au, b, inDim, outDim));
        return a;
    }

    // Apply one adapter's q_proj delta into y via the CPU kernel.
    private static void ApplyOne(ILoraAdapter a, float* x, float* y, int seqLen, int inDim, int outDim)
    {
        var w = a.GetLayerWeights(0, "q_proj")!.Value;
        LoraDelta.Apply(x, (float*)w.BHandle, (float*)w.AHandle, y, seqLen, inDim, outDim,
                        a.Rank, a.Alpha / a.Rank);
    }

    [Fact]
    public void Composite_Delta_Equals_Sum_Of_Singles()
    {
        const int inDim = 16, outDim = 16, seqLen = 2;
        var cfg = Cfg(inDim, 1, outDim);
        using var a1 = MakeAdapter("a1", 4, 8f, inDim, outDim, 1);
        using var a2 = MakeAdapter("a2", 8, 16f, inDim, outDim, 2);

        var x = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * inDim * sizeof(float)), 64);
        var ySum = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * outDim * sizeof(float)), 64);
        var yComp = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * outDim * sizeof(float)), 64);
        try
        {
            var rng = new Random(7);
            for (int i = 0; i < seqLen * inDim; i++) x[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < seqLen * outDim; i++) { ySum[i] = 0; yComp[i] = 0; }

            // Reference: sum of each adapter's delta (weights 1.0 and 0.5).
            ApplyOne(a1, x, ySum, seqLen, inDim, outDim);
            // weight 0.5 on a2: scale becomes 0.5 * alpha/rank — apply into a temp then add.
            var yTmp = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * outDim * sizeof(float)), 64);
            for (int i = 0; i < seqLen * outDim; i++) yTmp[i] = 0;
            var w2 = a2.GetLayerWeights(0, "q_proj")!.Value;
            LoraDelta.Apply(x, (float*)w2.BHandle, (float*)w2.AHandle, yTmp, seqLen, inDim, outDim,
                            a2.Rank, 0.5f * a2.Alpha / a2.Rank);
            for (int i = 0; i < seqLen * outDim; i++) ySum[i] += yTmp[i];
            NativeMemory.AlignedFree(yTmp);

            // Composite via the composer (a1 weight 1.0, a2 weight 0.5).
            using var comp = LoraComposer.Compose(new (ILoraAdapter, float)[] { (a1, 1f), (a2, 0.5f) }, cfg);
            Assert.Equal(a1.Rank + a2.Rank, comp.Rank);
            ApplyOne(comp, x, yComp, seqLen, inDim, outDim); // comp scale = Alpha/Rank = 1.0

            for (int i = 0; i < seqLen * outDim; i++)
                Assert.True(MathF.Abs(ySum[i] - yComp[i]) < 1e-4f, $"idx {i}: {ySum[i]} vs {yComp[i]}");
        }
        finally { NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(ySum); NativeMemory.AlignedFree(yComp); }
    }

    [Fact]
    public void Single_Element_Stack_Matches_Original_Adapter()
    {
        const int inDim = 16, outDim = 16, seqLen = 1;
        var cfg = Cfg(inDim, 1, outDim);
        using var a = MakeAdapter("a", 4, 8f, inDim, outDim, 3);
        var x = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * inDim * sizeof(float)), 64);
        var y1 = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * outDim * sizeof(float)), 64);
        var yc = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * outDim * sizeof(float)), 64);
        try
        {
            var rng = new Random(9);
            for (int i = 0; i < seqLen * inDim; i++) x[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < seqLen * outDim; i++) { y1[i] = 0; yc[i] = 0; }
            ApplyOne(a, x, y1, seqLen, inDim, outDim);
            using var comp = LoraComposer.Compose(new (ILoraAdapter, float)[] { (a, 1f) }, cfg);
            ApplyOne(comp, x, yc, seqLen, inDim, outDim);
            for (int i = 0; i < seqLen * outDim; i++) Assert.True(MathF.Abs(y1[i] - yc[i]) < 1e-5f);
        }
        finally { NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(y1); NativeMemory.AlignedFree(yc); }
    }

    [Fact]
    public void Rank_Cap_Exceeded_Throws()
    {
        var cfg = Cfg(16, 1, 16);
        using var a = MakeAdapter("a", 200, 200f, 16, 16, 1);
        using var b = MakeAdapter("b", 200, 200f, 16, 16, 2);
        Assert.Throws<NotSupportedException>(() =>
        {
            LoraComposer.Compose(new (ILoraAdapter, float)[] { (a, 1f), (b, 1f) }, cfg, maxRank: 256);
        });
    }

    [Fact]
    public void Non_Uniform_Coverage_Throws()
    {
        var cfg = Cfg(16, 1, 16) with { NumLayers = 2 };
        using var a = MakeAdapter("a", 4, 8f, 16, 16, 1);            // layer 0 only
        var b = new LoraAdapter("b", 4, 8f, new[] { "q_proj" });
        b.AddLayerWeights(1, "q_proj", new LoraLayerWeights(         // layer 1 only — mismatched coverage
            LoraAdapter.AllocAligned(4 * 16), LoraAdapter.AllocAligned(16 * 4), 16, 16));
        using var _ = b;
        Assert.Throws<InvalidOperationException>(() =>
        {
            LoraComposer.Compose(new (ILoraAdapter, float)[] { (a, 1f), (b, 1f) }, cfg);
        });
    }

    [Fact]
    public void Composite_Is_Order_Independent()
    {
        // compose [(a1,1),(a2,1)] and [(a2,1),(a1,1)] — rank-concat order differs
        // but the additive sum must produce the same delta for any input.
        const int inDim = 16, outDim = 16, seqLen = 2;
        var cfg = Cfg(inDim, 1, outDim);
        using var a1 = MakeAdapter("a1", 4, 8f, inDim, outDim, 11);
        using var a2 = MakeAdapter("a2", 8, 8f, inDim, outDim, 22);

        var x = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * inDim * sizeof(float)), 64);
        var y12 = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * outDim * sizeof(float)), 64);
        var y21 = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * outDim * sizeof(float)), 64);
        try
        {
            var rng = new Random(42);
            for (int i = 0; i < seqLen * inDim; i++) x[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < seqLen * outDim; i++) { y12[i] = 0; y21[i] = 0; }

            using var comp12 = LoraComposer.Compose(new (ILoraAdapter, float)[] { (a1, 1f), (a2, 1f) }, cfg);
            using var comp21 = LoraComposer.Compose(new (ILoraAdapter, float)[] { (a2, 1f), (a1, 1f) }, cfg);

            ApplyOne(comp12, x, y12, seqLen, inDim, outDim);
            ApplyOne(comp21, x, y21, seqLen, inDim, outDim);

            for (int i = 0; i < seqLen * outDim; i++)
                Assert.True(MathF.Abs(y12[i] - y21[i]) < 1e-5f, $"idx {i}: {y12[i]} vs {y21[i]}");
        }
        finally { NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(y12); NativeMemory.AlignedFree(y21); }
    }
}
