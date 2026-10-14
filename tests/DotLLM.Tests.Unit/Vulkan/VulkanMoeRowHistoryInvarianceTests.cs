using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1016: a 2-row Vulkan forward must not depend on whether a larger forward ran earlier in the same process.
/// </summary>
/// <remarks>
/// <para>
/// <b>Defect.</b> On gpt-oss-20b and Gemma-4-26B-A4B the logits of a 2-row forward differed (mean KL 2.6e-3 / 0.35) after a 3- or 4-row forward had run,
/// bit-identically for 3 and 4 rows, while the very same forward in a fresh process was bit-stable run to run. Resetting the forward scratch to its
/// minimal size restored the fresh result exactly, and over-allocating ONLY the multi-row Q8_1 activation scratch (<c>Q8_1XqRows</c>) reproduced the
/// divergence: the dp4a MMQ / small-row Q8 paths were gated on "<c>[n, K]</c> fits the scratch BYTES", and that scratch is
/// <c>capacity x maxK</c> with <c>maxK = max(hidden, intermediate, moeIntermediate)</c> - which leaves out the attention output projection's input
/// width <c>numHeads x headDim</c>. For every model where that width exceeds <c>maxK</c> (MoE models, whose dense <c>intermediate</c> is small) the
/// o_proj fit check was <c>n * qDim &lt;= capacity * maxK</c>: false right after a forward of exactly <c>n</c> rows, true once an earlier forward had grown the capacity
/// to <c>1.42 n</c>. Same op, same input, two different kernels (MMQ vs the fallback GEMM), selected by history. Dense models never tripped it because
/// their intermediate is always wider than <c>numHeads x headDim</c>.
/// </para>
/// <para>
/// <b>Why this fixture.</b> The condition is a ratio of widths, so the tiny model has Q8_0 attention weights (the quantization whose MMQ path is gated on
/// the scratch) and <c>numHeads x headDim</c> = 128 against <c>hidden = moeIntermediate = 64</c>: after a 4-row forward a 2-row o_proj fits
/// (2 x 128 &lt;= 4 x 64), on a fresh state it does not (2 x 128 &gt; 2 x 64). Run on the pre-fix code this test fails; see the PR for the mutant evidence.
/// The 1-row case is excluded on purpose: the single-row Q8_1 scratch is capacity independent.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeRowHistoryInvarianceTests : IDisposable
{
    private const int Hidden = 64, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 32, Vocab = 64, Experts = 8, TopK = 2, MoeI = 64, Window = 3, Ctx = 128;
    private const int QDim = Heads * HeadDim;   // 128 > Hidden = MoeI = 64: the o_proj input is wider than the scratch's maxK

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-1016-" + Guid.NewGuid().ToString("N"));

    public VulkanMoeRowHistoryInvarianceTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { /* best effort */ }
    }

    /// <summary>Rows run before the 2-row pass: 3 and 4 both grow the capacity past the o_proj fit threshold; 8 is the scheduler / prefill-chunk shaped case.</summary>
    [SkippableTheory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(8)]
    public void TwoRowForward_IsIndependentOfAnEarlierLargerForward(int earlierRows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteGptOssQ8Fixture();
        int[][] seqs = Sequences();

        float[] fresh = RunPasses(path, spvDir, seqs, [2]);
        float[] afterLarger = RunPasses(path, spvDir, seqs, [earlierRows, 2]);

        Assert.Equal(fresh.Length, afterLarger.Length);
        int differing = 0; float maxDiff = 0;
        for (int i = 0; i < fresh.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(fresh[i]) != BitConverter.SingleToInt32Bits(afterLarger[i])) differing++;
            maxDiff = MathF.Max(maxDiff, MathF.Abs(fresh[i] - afterLarger[i]));
        }
        Assert.True(differing == 0,
            $"a {earlierRows}-row forward changed the next 2-row forward's logits: {differing}/{fresh.Length} values differ, max |d| = {maxDiff:E3} (#1016: " +
            "the o_proj Q8 MMQ path was selected from the scratch's byte size, i.e. from history).");
    }

    /// <summary>The same property one level down: the fit decision is a function of the row width and row count alone, never of how large the scratch has grown.</summary>
    [SkippableFact]
    public void Q8RowsFit_DoesNotDependOnPriorCapacityGrowth()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteGptOssQ8Fixture();
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();

        var fits = new List<bool>();
        foreach (int[] history in new[] { new[] { 2 }, new[] { 4, 2 }, new[] { 8, 2 } })
        {
            var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, cfg, spvDir, 0);
            using var _ = (IDisposable)model;
            var vk = (VulkanTransformerModel)model;
            foreach (int rows in history)
            {
                using IKvCache kv = kvFactory(Ctx);
                int[] ids = Enumerable.Range(0, rows).Select(i => (i * 7 + 3) % Vocab).ToArray();
                using var l = vk.Forward(ids, Enumerable.Range(0, rows).ToArray(), -1, kv);
            }
            fits.Add(vk.Q8RowsFitForTest(seqLen: 2, inputDim: QDim));
        }
        Assert.All(fits, f => Assert.Equal(fits[0], f));
        Assert.False(fits[0], "the o_proj width exceeds the scratch's per-row K, so it must never claim to fit");
    }

    private static int[][] Sequences()
    {
        var rng = new Random(1016);
        return Enumerable.Range(0, 3).Select(s => Enumerable.Range(0, 11 + s).Select(_ => rng.Next(Vocab)).ToArray()).ToArray();
    }

    /// <summary>Loads the model fresh, then teacher-forces every sequence in chunks of each size in <paramref name="chunks"/> in turn; returns the logits of the LAST chunk size's pass.</summary>
    private static unsafe float[] RunPasses(string path, string spvDir, int[][] seqs, int[] chunks)
    {
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, cfg, spvDir, 0);
        using var _ = (IDisposable)model;
        Assert.True(model.TrySetAllRowLogitsLimit(8));

        var last = new List<float>();
        foreach (int chunk in chunks)
        {
            last.Clear();
            foreach (int[] ids in seqs)
            {
                using IKvCache kv = kvFactory(Ctx);
                for (int i = 0; i < ids.Length; i += chunk)
                {
                    int n = Math.Min(chunk, ids.Length - i);
                    using ITensor t = model.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), -1, kv);
                    int count = t.Shape[0] * t.Shape[1];
                    var row = new float[count];
                    new ReadOnlySpan<float>((void*)t.DataPointer, count).CopyTo(row);
                    last.AddRange(row);
                }
            }
        }
        return last.ToArray();
    }

    // ───────────────────────── fixture: synthetic gpt-oss, Q8_0 attention, MXFP4 experts ─────────────────────────

    private string WriteGptOssQ8Fixture()
    {
        string path = Path.Combine(_dir, "gptoss-q8.gguf");
        var rng = new Random(7);
        var w = new GgufWriter();
        const string arch = "gpt-oss";
        w.AddString("general.architecture", arch);
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.context_length", Ctx);
        w.AddUInt32($"{arch}.embedding_length", Hidden);
        w.AddUInt32($"{arch}.block_count", Layers);
        w.AddUInt32($"{arch}.feed_forward_length", MoeI);
        w.AddUInt32($"{arch}.attention.head_count", Heads);
        w.AddUInt32($"{arch}.attention.head_count_kv", KvHeads);
        w.AddUInt32($"{arch}.attention.key_length", HeadDim);
        w.AddUInt32($"{arch}.attention.value_length", HeadDim);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        w.AddUInt32($"{arch}.attention.sliding_window", Window);
        w.AddUInt32($"{arch}.expert_count", Experts);
        w.AddUInt32($"{arch}.expert_used_count", TopK);
        w.AddUInt32($"{arch}.expert_feed_forward_length", MoeI);
        w.AddFloat32($"{arch}.rope.freq_base", 10000.0f);
        w.AddString($"{arch}.rope.scaling.type", "yarn");
        w.AddFloat32($"{arch}.rope.scaling.factor", 2.0f);
        w.AddUInt32($"{arch}.rope.scaling.original_context_length", 64);
        w.AddUInt32($"{arch}.vocab_size", Vocab);

        AddF32(w, rng, "token_embd.weight", [Hidden, Vocab], 0.3f);
        for (int i = 0; i < Layers; i++)
        {
            string p = $"blk.{i}";
            AddNorm(w, rng, $"{p}.attn_norm.weight", Hidden);
            AddQ8(w, rng, $"{p}.attn_q.weight", inK: Hidden, outM: QDim, 0.3f);
            AddQ8(w, rng, $"{p}.attn_k.weight", inK: Hidden, outM: KvHeads * HeadDim, 0.3f);
            AddQ8(w, rng, $"{p}.attn_v.weight", inK: Hidden, outM: KvHeads * HeadDim, 0.3f);
            AddQ8(w, rng, $"{p}.attn_output.weight", inK: QDim, outM: Hidden, 0.3f);
            AddF32(w, rng, $"{p}.attn_q.bias", [QDim], 0.02f);
            AddF32(w, rng, $"{p}.attn_k.bias", [KvHeads * HeadDim], 0.02f);
            AddF32(w, rng, $"{p}.attn_v.bias", [KvHeads * HeadDim], 0.02f);
            AddF32(w, rng, $"{p}.attn_output.bias", [Hidden], 0.02f);
            var sinks = new float[Heads];
            for (int h = 0; h < Heads; h++) sinks[h] = 1.0f + 0.5f * h;
            w.AddTensor($"{p}.attn_sinks.weight", [Heads], (uint)QuantizationType.F32, MemoryMarshal.AsBytes(sinks.AsSpan()).ToArray());
            AddNorm(w, rng, $"{p}.post_attention_norm.weight", Hidden);
            AddF32(w, rng, $"{p}.ffn_gate_inp.weight", [Hidden, Experts], 0.5f);
            AddF32(w, rng, $"{p}.ffn_gate_inp.bias", [Experts], 0.05f);
            AddMxfp4(w, rng, $"{p}.ffn_gate_exps.weight", inK: Hidden, midOut: MoeI);
            AddMxfp4(w, rng, $"{p}.ffn_up_exps.weight", inK: Hidden, midOut: MoeI);
            AddMxfp4(w, rng, $"{p}.ffn_down_exps.weight", inK: MoeI, midOut: Hidden);
            AddF32(w, rng, $"{p}.ffn_gate_exps.bias", [Experts * MoeI], 0.05f);
            AddF32(w, rng, $"{p}.ffn_up_exps.bias", [Experts * MoeI], 0.05f);
            AddF32(w, rng, $"{p}.ffn_down_exps.bias", [Experts * Hidden], 0.05f);
        }
        AddNorm(w, rng, "output_norm.weight", Hidden);
        AddF32(w, rng, "output.weight", [Hidden, Vocab], 0.3f);
        File.WriteAllBytes(path, w.Build());
        return path;
    }

    private static float[] Random(Random rng, long n, float amp)
    {
        var f = new float[n];
        for (long i = 0; i < n; i++) f[i] = (float)(rng.NextDouble() * 2 - 1) * amp;
        return f;
    }

    private static void AddF32(GgufWriter w, Random rng, string name, int[] dims, float amp)
    {
        long n = 1; foreach (int d in dims) n *= d;
        w.AddTensor(name, dims, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(Random(rng, n, amp).AsSpan()).ToArray());
    }

    private static void AddNorm(GgufWriter w, Random rng, string name, int n)
    {
        var f = Random(rng, n, 0.05f);
        for (int i = 0; i < n; i++) f[i] += 1.0f;
        w.AddTensor(name, [n], (uint)QuantizationType.F32, MemoryMarshal.AsBytes(f.AsSpan()).ToArray());
    }

    /// <summary>Q8_0 matrix [ne0 = K, ne1 = M]: M rows of K/32 blocks, quantized with the production CPU quantizer.</summary>
    private static unsafe void AddQ8(GgufWriter w, Random rng, string name, int inK, int outM, float amp)
    {
        float[] src = Random(rng, (long)inK * outM, amp);
        long rowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(inK, QuantizationType.Q8_0);
        var bytes = new byte[outM * rowBytes];
        fixed (float* s = src)
        fixed (byte* d = bytes)
            for (int r = 0; r < outM; r++)
                DotLLM.Cpu.Kernels.MatMul.QuantizeF32ToQ8_0(s + (long)r * inK, d + r * rowBytes, inK);
        w.AddTensor(name, [inK, outM], (uint)QuantizationType.Q8_0, bytes);
    }

    /// <summary>MXFP4 expert bank [K, midOut, experts]: well-formed random 17-byte blocks (E8M0 scale near unit magnitude + 16 nibble bytes).</summary>
    private static void AddMxfp4(GgufWriter w, Random rng, string name, int inK, int midOut)
    {
        long rows = (long)midOut * Experts, blocks = inK / 32;
        var data = new byte[rows * blocks * 17];
        long o = 0;
        for (long r = 0; r < rows * blocks; r++)
        {
            data[o++] = (byte)rng.Next(122, 133);
            for (int j = 0; j < 16; j++) data[o++] = (byte)rng.Next(256);
        }
        w.AddTensor(name, [inK, midOut, Experts], (uint)QuantizationType.MXFP4, data);
    }
}
