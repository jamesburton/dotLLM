using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Autoregressive Gemma-4 KV-cache decode equivalence on the CPU backend — the CPU
/// mirror of <see cref="Gemma4VulkanKvCacheTests"/> and the KV Phase 0 acceptance
/// gate. Proves the per-layer-strided <see cref="SimpleKvCache"/> wired into the CPU
/// <c>RunGemma4Layer</c> attention produces the same next-token logits via
/// prefill-then-decode as a single cacheless forward over the whole sequence (the
/// established oracle for the gemma4 CPU forward).
/// </summary>
/// <remarks>
/// DISCRIMINATING FIXTURE: the synthetic config is deliberately chosen so the sliding
/// layers (KV stride <c>SlidingKvHeads × SlidingHeadDim = 2 × 16 = 32</c>) and the
/// global layer (<c>GlobalKvHeads × GlobalHeadDim = 2 × 32 = 64</c>) have DIFFERENT
/// cached-row widths. The pre-Phase-0 CPU forward rejected this outright via
/// <c>GuardKvCacheHeadDim</c>; a cache that used a single uniform stride would
/// mis-address one of the two layer classes and flip the argmax. The test asserts the
/// strides differ before relying on the parity check, so a degenerate equal-stride
/// fixture cannot mask the bug. This test FAILS before Phase 0 (the guard throws) and
/// passes only after the per-layer CPU cache path lands.
/// </remarks>
public sealed class Gemma4CpuKvCacheTests
{
    private readonly ITestOutputHelper _output;

    public Gemma4CpuKvCacheTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Cpu_Gemma4_KvCacheDecode_MatchesCachelessForward()
    {
        // Differing per-layer KV strides: sliding 2×16=32, global 2×32=64.
        var cfg = SyntheticGemma4Gguf.Tiny with { GlobalKvHeads = 2 };
        string path = Path.Combine(Path.GetTempPath(), $"syn_gemma4_cpukv_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, cfg);

        try
        {
            // Kept within the sliding window (8) so each token's attention window is
            // identical whether computed in one shot or incrementally — isolating the
            // test to cache addressing, not window-clip semantics.
            int[] ids = { 2, 7, 8, 9, 5, 6, 3 }; // synthetic fixture BOS = 2
            int[] pos = { 0, 1, 2, 3, 4, 5, 6 };
            int last = ids.Length - 1;

            using var gguf = GgufFile.Open(path);
            var modelCfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(Architecture.Gemma4, modelCfg.Architecture);
            int vocab = modelCfg.VocabSize;

            using var model = TransformerModel.LoadFromGguf(gguf, modelCfg);

            // Guard: the geometry really does exercise differing per-layer strides
            // (else the test is degenerate and cannot catch a uniform-stride bug).
            var geom = KvGeometry.FromConfig(modelCfg);
            int slidingStride = geom.KvStrideOf(0);                       // layer 0 = sliding
            int globalStride = geom.KvStrideOf(modelCfg.NumLayers - 1);   // last = global
            Assert.NotEqual(slidingStride, globalStride);
            _output.WriteLine($"per-layer KV strides differ: sliding={slidingStride}, global={globalStride}");

            // ── Oracle: single cacheless forward over the whole sequence ──
            float[] cachelessLast;
            using (ITensor logits = model.Forward(ids, pos, deviceId: -1, kvCache: null))
                cachelessLast = LastRow(logits, vocab);

            // ── Under test: prefill [0,last) then decode the last token ──
            float[] decodeLast;
            using (var kv = new SimpleKvCache(KvGeometry.FromConfig(modelCfg), maxSeqLen: 16))
            {
                using (var _ = model.Forward(ids.AsSpan(0, last), pos.AsSpan(0, last), deviceId: -1, kvCache: kv)) { }
                Assert.Equal(last, kv.CurrentLength);

                using ITensor logits = model.Forward(ids.AsSpan(last, 1), pos.AsSpan(last, 1), deviceId: -1, kvCache: kv);
                Assert.Equal(ids.Length, kv.CurrentLength);
                decodeLast = LastRow(logits, vocab);
            }

            // Structural guard — the greedy next token must agree exactly. A
            // mis-strided cache flips the argmax; reduction-order drift does not.
            int cachelessArg = ArgMax(cachelessLast), decodeArg = ArgMax(decodeLast);
            Assert.Equal(cachelessArg, decodeArg);

            // Per-logit envelope: prefill (GEMM) vs decode (GEMV) take different
            // reduction orders, so allow the same drift the Vulkan parity test uses.
            const float absTol = 6.0e-2f, relTol = 5.0e-3f;
            int worst = -1; float worstDiff = 0;
            for (int c = 0; c < vocab; c++)
            {
                float diff = MathF.Abs(cachelessLast[c] - decodeLast[c]);
                if (diff > worstDiff) { worstDiff = diff; worst = c; }
            }
            for (int c = 0; c < vocab; c++)
            {
                float a = cachelessLast[c], b = decodeLast[c];
                float bar = absTol + relTol * MathF.Abs(a);
                Assert.True(MathF.Abs(a - b) <= bar,
                    $"col {c}: cacheless={a:F6} vs decode={b:F6} (|diff|={MathF.Abs(a - b):E3} > {bar:E3}); "
                    + $"argmax cacheless={cachelessArg} decode={decodeArg}; worst |diff|={worstDiff:E3} @ col {worst}");
            }
            _output.WriteLine($"gemma4 CPU KV-cache decode parity OK: argmax={cachelessArg}, worst |diff|={worstDiff:E3} @ col {worst} over {vocab} logits.");
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var all = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(all);
        var row = new float[vocab];
        Array.Copy(all, total - vocab, row, 0, vocab);
        return row;
    }
}
