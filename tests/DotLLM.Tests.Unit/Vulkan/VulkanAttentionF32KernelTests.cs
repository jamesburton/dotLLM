using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Core.PositionEncoding;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Vulkan FP32 attention kernel.
/// </summary>
/// <remarks>
/// Compared against <see cref="Attention.ExecuteScalar"/> — the scalar CPU
/// reference that does not use the tiled online-softmax code path. The GPU
/// uses flash-attention-style online softmax, so reduction order differs;
/// tolerance follows the mandate (rel 1e-3 / abs 1e-4).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanAttentionF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableFact]
    public void Launch_SingleHead_SmallDecode()
    {
        // Sanity: decode-like shape, 1 query position attending to 8 KV positions,
        // head_dim = 64, single head on both sides.
        RunOne(seqQ: 1, seqKv: 8, numHeads: 1, numKvHeads: 1, headDim: 64, positionOffset: 7);
    }

    [SkippableFact]
    public void Launch_SingleHead_FourQueries()
    {
        // Prefill: 4 queries against 4 keys, single head.
        RunOne(seqQ: 4, seqKv: 4, numHeads: 1, numKvHeads: 1, headDim: 64, positionOffset: 0);
    }

    [SkippableFact]
    public void Launch_SmolLm_Decode()
    {
        // SmolLM-135M decode shape: nh=9, nkv=3, head_dim=64, seq_q=1, seq_kv=128.
        // Position offset = 127 so the single query attends to all 128 cached keys.
        RunOne(seqQ: 1, seqKv: 128, numHeads: 9, numKvHeads: 3, headDim: 64, positionOffset: 127);
    }

    [SkippableFact]
    public void Launch_SmolLm_Prefill()
    {
        // Prefill-ish shape: 64 queries, 64 keys, SmolLM head config.
        RunOne(seqQ: 64, seqKv: 64, numHeads: 9, numKvHeads: 3, headDim: 64, positionOffset: 0);
    }

    [SkippableFact]
    public void Launch_Llama_HeadDim128_Decode()
    {
        // Llama-style head dim 128 through the fixed MAX_HEAD_DIM shader path.
        RunOne(seqQ: 1, seqKv: 64, numHeads: 8, numKvHeads: 8, headDim: 128, positionOffset: 63);
    }

    [SkippableFact]
    public void Launch_MultiTile_TripleTile()
    {
        // Exercises the online-softmax tile loop: seq_kv > TILE_KV (256).
        // Two-tile boundary: seq_kv = 400.
        RunOne(seqQ: 1, seqKv: 400, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 399);
    }

    [SkippableFact]
    public void Launch_Alibi_MatchesCpuReference()
    {
        RunOne(seqQ: 4, seqKv: 8, numHeads: 6, numKvHeads: 2, headDim: 64, positionOffset: 0, useAlibi: true);
    }

    // Discriminates against the legacy `posQ - tkv > slidingWindow` off-by-one
    // (which kept one extra oldest token visible). With slidingWindow=4 and
    // posQ=7, the correct mask hides tkv∈{0,1,2,3}; the buggy mask hides only
    // tkv∈{0,1,2}, leaking tkv=3 into the softmax.
    [SkippableFact]
    public void Launch_SlidingWindow_DiscriminatesOffByOne()
    {
        RunOne(seqQ: 1, seqKv: 8, numHeads: 1, numKvHeads: 1, headDim: 64, positionOffset: 7,
            slidingWindow: 4);
    }

    [SkippableFact]
    public void Launch_SlidingWindow_MultiTile()
    {
        RunOne(seqQ: 1, seqKv: 400, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 399,
            slidingWindow: 128);
    }

    // ── Non-causal mask modes (issue #41 item 2) ─────────────────
    //
    // These DISCRIMINATE against the causal path: a query at an early position
    // must attend to FUTURE keys, which the causal mask (tkv > posQ) would zero
    // out. The CPU reference is invoked in the same mask mode, so parity holds
    // only if the Vulkan shader honours maskMode/prefixLen identically.

    [SkippableFact]
    public void Launch_Bidirectional_AttendsFutureKeys()
    {
        // 4 queries × 4 keys, positionOffset=0. Under Causal, query 0 sees only
        // key 0; under Bidirectional it sees all 4 keys. The CPU reference runs
        // the same Bidirectional mode, so a shader that ignored maskMode (still
        // causal) would diverge for the early query rows.
        RunOne(seqQ: 4, seqKv: 4, numHeads: 2, numKvHeads: 1, headDim: 64, positionOffset: 0,
            maskMode: AttentionMaskMode.Bidirectional);
    }

    [SkippableFact]
    public void Launch_Bidirectional_Gqa_Decode()
    {
        // Single bidirectional query over a 128-key canvas at positionOffset=0:
        // it must attend to ALL 128 keys (a causal mask would leave only key 0).
        RunOne(seqQ: 1, seqKv: 128, numHeads: 9, numKvHeads: 3, headDim: 64, positionOffset: 0,
            maskMode: AttentionMaskMode.Bidirectional);
    }

    [SkippableFact]
    public void Launch_Bidirectional_MultiTile()
    {
        // seqKv > TILE_KV (256): exercises the online-softmax tile loop with no
        // causal upper bound. positionOffset=0 so every key is in-window.
        RunOne(seqQ: 1, seqKv: 400, numHeads: 4, numKvHeads: 2, headDim: 64, positionOffset: 0,
            maskMode: AttentionMaskMode.Bidirectional);
    }

    [SkippableFact]
    public void Launch_Hybrid_PrefixCausal_CanvasBidirectional()
    {
        // prefixLen=3: query positions 0..2 stay causal (mask future keys),
        // positions 3..7 are bidirectional (attend to the full 8-key range).
        // Distinct from both pure-causal and pure-bidirectional outputs.
        RunOne(seqQ: 8, seqKv: 8, numHeads: 2, numKvHeads: 1, headDim: 64, positionOffset: 0,
            maskMode: AttentionMaskMode.Hybrid, prefixLen: 3);
    }

    [SkippableFact]
    public void Launch_Hybrid_PrefixLenZero_EqualsBidirectional()
    {
        // prefixLen=0 means no causal prefix → every query is a canvas query →
        // identical to Bidirectional. Validates the boundary of the hybrid rule.
        RunOne(seqQ: 4, seqKv: 4, numHeads: 2, numKvHeads: 2, headDim: 64, positionOffset: 0,
            maskMode: AttentionMaskMode.Hybrid, prefixLen: 0);
    }

    // ─────────────────────────────────────────────────────────────

    private static void RunOne(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset,
        bool useAlibi = false, int slidingWindow = 0,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x511E + seqQ * 41 + seqKv * 17 + numHeads * 7 + headDim);
        float[] qh = RandomFloats(rng, seqQ * numHeads * headDim);
        float[] kh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] vh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] expected = new float[seqQ * numHeads * headDim];

        int? sw = slidingWindow > 0 ? slidingWindow : null;
        float scale = 1.0f / MathF.Sqrt(headDim);
        ReadOnlySpan<float> slopes = useAlibi
            ? AlibiPositionEncoding.CreateSlopes(numHeads)
            : default;
        Attention.ExecuteScalar(qh, kh, vh, expected,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            scale, slopes, sw, softCap: 0f, maskMode, prefixLen);

        // GPU path.
        using var device = VulkanDevice.Create();
        using var kernel = AttentionF32Kernel.Create(device, spvDir);

        using var bufQ   = device.Allocate((long)qh.Length * sizeof(float));
        using var bufK   = device.Allocate((long)kh.Length * sizeof(float));
        using var bufV   = device.Allocate((long)vh.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(qh.AsSpan(), bufQ);
        device.Upload(kh.AsSpan(), bufK);
        device.Upload(vh.AsSpan(), bufV);

        kernel.Launch(bufQ, bufK, bufV, bufOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            slidingWindow: slidingWindow, useAlibi: useAlibi,
            maskMode: maskMode, prefixLen: prefixLen);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        AssertClose(expected, actual, seqQ, seqKv, numHeads, numKvHeads, headDim);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0); // [-1, 1]
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"Attention drift exceeded tolerance " +
            $"(seqQ={seqQ},seqKv={seqKv},nh={numHeads},nkv={numKvHeads},hd={headDim}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
