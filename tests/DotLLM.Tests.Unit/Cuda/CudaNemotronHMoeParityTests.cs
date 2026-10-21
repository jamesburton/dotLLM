using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #403: <c>nemotron_h_moe</c> on CUDA, end to end through a synthetic GGUF, the real config extractor, the CPU
/// loader and <see cref="CudaModelLoader.CreateFromGguf"/>, held to the CPU forward.
/// <list type="bullet">
///   <item><b>Bolt shape</b> (LatentMoE, hidden 512, latent 256, scale 5): two Mamba2 layers and four MoE layers that differ in
///   expert count (6 / 8 / 5 / 7), top-k (2 / 3 / 1 / 4) and width (512 / 256 / 256 / 512), with Q4_K / Q5_K / Q6_K / Q8_0 /
///   IQ2_XXS banks.</item>
///   <item><b>Lightning shape</b>: the same model with no latent, so the experts read and write the hidden size.</item>
///   <item>Prefill of 7 tokens, then one decode token with the SSM state carried (each model runs prefill then decode on
///   the same instance; the CPU model's recurrent state carries across cacheless calls).</item>
/// </list>
/// Two CPU oracles: the CPU model on the same quantized file (whose K-quant dots quantize their activations to Q8_K, so it
/// sits measurably off), and the CPU model on a twin file whose banks are the dequantized F32 values (exact F32 maths; every
/// dense weight is F32 in both files, so CUDA's only quantized reads are the banks under test). The selection-bias control
/// (Vulkan Bolt parity test) proves the comparison can see a routing error.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaNemotronHMoeParityTests : IDisposable
{
    private const string Arch = "nemotron_h_moe";
    private const int Vocab = 16, Hidden = 512, LatentSize = 256, SharedInter = 64;
    private const int DInner = 32, DState = 8, NHead = 4, DConv = 4;
    private const float Eps = 1e-5f, WScale = 5.0f;

    /// <summary>Layer plan: null = Mamba2 SSM layer, otherwise an MoE layer.</summary>
    private static readonly MoeSpec?[] Layers =
    [
        null,
        new(Experts: 6, TopK: 2, Inter: 512, UpQt: QuantizationType.Q4_K, DownQt: QuantizationType.Q5_K),
        new(Experts: 8, TopK: 3, Inter: 256, UpQt: QuantizationType.Q6_K, DownQt: QuantizationType.Q6_K),
        null,
        new(Experts: 5, TopK: 1, Inter: 256, UpQt: QuantizationType.Q5_K, DownQt: QuantizationType.Q8_0),
        new(Experts: 7, TopK: 4, Inter: 512, UpQt: QuantizationType.IQ2_XXS, DownQt: QuantizationType.IQ2_XXS),
    ];

    private static readonly int[] Prompt = [3, 9, 1, 14, 6, 0, 11];
    private const int DecodeToken = 5;

    private sealed record MoeSpec(int Experts, int TopK, int Inter, QuantizationType UpQt, QuantizationType DownQt);

    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public CudaNemotronHMoeParityTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-nemoe-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <param name="latent">true = Bolt (LatentMoE); false = Lightning (experts on the hidden size).</param>
    /// <param name="blockTokens">CUDA MoE token block; 3 splits the 7-token prefill into three blocks.</param>
    [SkippableTheory]
    [InlineData(true, 512)]
    [InlineData(true, 3)]
    [InlineData(false, 512)]
    public void PrefillAndDecode_MatchCpu(bool latent, int blockTokens)
    {
        Skip.IfNot(CudaNemotronMoeTestSupport.IsCudaAvailable(), "No CUDA GPU available");
        string? ptxDir = CudaNemotronMoeTestSupport.FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string quantPath = WriteFixture(_scratch, latent, zeroBias: false, f32Banks: false);
        string noBiasPath = WriteFixture(_scratch, latent, zeroBias: true, f32Banks: false);
        string exactPath = WriteFixture(_scratch, latent, zeroBias: false, f32Banks: true);

        float[] cpu = RunCpu(quantPath);
        float[] cpuNoBias = RunCpu(noBiasPath);
        float[] exact = RunCpu(exactPath);

        // Control: the fixture must be sensitive to the selection bias, or "matches CPU" proves nothing about the routing.
        float control = MaxDiff(cpu, cpuNoBias);
        Assert.True(control > 0.05f, $"fixture insensitive to the selection bias (max logit delta {control:E3})");
        float cpuVsExact = MaxDiff(exact, cpu);
        Assert.True(cpuVsExact * 4 < control, $"CPU oracle vs F32-bank oracle {cpuVsExact:E3} is not small against the bias effect {control:E3}");

        int savedBlock = CudaNemotronHTransformerModel.MoeBlockTokens;
        CudaNemotronHTransformerModel.MoeBlockTokens = blockTokens;
        CudaNemotronHTransformerModel.MoePathCounters.Reset();
        float[] cuda;
        try { cuda = RunCuda(quantPath, ptxDir!); }
        finally { CudaNemotronHTransformerModel.MoeBlockTokens = savedBlock; }

        // Path counters: every MoE layer ran on the expert-grouped kernels, once per token block, for every bank type.
        int passes = (Prompt.Length + blockTokens - 1) / blockTokens + 1; // prefill blocks + the decode step
        int moeLayers = Layers.Count(l => l is not null);
        Assert.Equal(moeLayers * passes, Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.LayerCalls));
        Assert.Equal(1 * passes, Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.Q4_K));
        Assert.Equal(2 * passes, Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.Q5_K));
        Assert.Equal(2 * passes, Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.Q6_K));
        Assert.Equal(1 * passes, Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.Q8_0));
        Assert.Equal(2 * passes, Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.IQ2_XXS));

        float vsExact = MaxDiff(exact, cuda), vsCpu = MaxDiff(cpu, cuda);
        float prefillVsExact = MaxDiff(exact[..(Prompt.Length * Vocab)], cuda[..(Prompt.Length * Vocab)]);
        float decodeVsExact = MaxDiff(exact[(Prompt.Length * Vocab)..], cuda[(Prompt.Length * Vocab)..]);
        _output.WriteLine($"latent={latent} block={blockTokens}: |cuda-exact| prefill {prefillVsExact:E3} decode {decodeVsExact:E3}; "
            + $"|cuda-cpu|={vsCpu:E3} |cpu-exact|={cpuVsExact:E3} biasControl={control:E3}");

        // CUDA reads the same quantized banks as the exact oracle's dequantized F32 values with FP32 activations, and runs every
        // dense weight (F32 here) through FP32 cuBLAS, so it is exact up to F32 summation order.
        const float tol = 5e-3f;
        Assert.True(tol * 10 < control, $"tolerance {tol} is not small against the bias effect {control:E3}");
        Assert.True(vsExact <= tol, $"max |exact - cuda| = {vsExact:E3} > {tol} (prefill {prefillVsExact:E3}, decode {decodeVsExact:E3})");
        Assert.True(vsCpu * 4 < control, $"max |cpu - cuda| = {vsCpu:E3} is not small against the bias effect {control:E3}");
        for (int row = 0; row < cuda.Length / Vocab; row++)
            Assert.Equal(ArgMax(exact, row), ArgMax(cuda, row));
    }

    // ───────────────────────── runs ─────────────────────────

    /// <summary>Prompt prefill then one decode token on one CPU model instance; rows = prompt rows + the decode row.</summary>
    private static unsafe float[] RunCpu(string path)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.NemotronHMoe, config.Architecture);
        using var model = NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        return PrefillThenDecode(model);
    }

    private static float[] RunCuda(string path, string ptxDir)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (model, _) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir);
        using (model)
        {
            Assert.IsType<CudaNemotronHTransformerModel>(model);
            return PrefillThenDecode(model);
        }
    }

    private static unsafe float[] PrefillThenDecode(IModel model)
    {
        var rows = new List<float>();
        using (var logits = model.Forward(Prompt, [.. Enumerable.Range(0, Prompt.Length)], deviceId: 0))
            rows.AddRange(new ReadOnlySpan<float>((void*)logits.DataPointer, Prompt.Length * Vocab).ToArray());
        using (var logits = model.Forward([DecodeToken], [Prompt.Length], deviceId: 0))
            rows.AddRange(new ReadOnlySpan<float>((void*)logits.DataPointer, Vocab).ToArray());
        return [.. rows];
    }

    // ───────────────────────── fixture ─────────────────────────

    /// <summary>
    /// Writes the fixture GGUF. The RNG stream is identical across the three variants: <paramref name="zeroBias"/> only zeroes
    /// the (deterministic) selection bias, and <paramref name="f32Banks"/> writes the dequantized F32 values of the very same
    /// quantized banks. Shared with the expert-offload tests (#1070), which hold the offloaded model to the resident one.
    /// </summary>
    internal static string WriteFixture(string scratch, bool latent, bool zeroBias, bool f32Banks)
    {
        var rng = new Random(403);
        var g = new GgufWriter();
        int n = Layers.Length;
        g.AddString("general.architecture", Arch);
        g.AddUInt32("general.alignment", 32);
        g.AddUInt32($"{Arch}.context_length", 64);
        g.AddUInt32($"{Arch}.embedding_length", Hidden);
        g.AddUInt32($"{Arch}.block_count", (uint)n);
        g.AddUInt32($"{Arch}.attention.head_count", 2);
        g.AddInt32Array($"{Arch}.attention.head_count_kv", new int[n]);
        g.AddInt32Array($"{Arch}.feed_forward_length", [.. Layers.Select(l => l?.Inter ?? 0)]);
        g.AddUInt32($"{Arch}.attention.key_length", 4);
        g.AddUInt32($"{Arch}.vocab_size", Vocab);
        g.AddFloat32($"{Arch}.attention.layer_norm_rms_epsilon", Eps);
        g.AddUInt32($"{Arch}.ssm.conv_kernel", DConv);
        g.AddUInt32($"{Arch}.ssm.state_size", DState);
        g.AddUInt32($"{Arch}.ssm.group_count", 1);
        g.AddUInt32($"{Arch}.ssm.time_step_rank", NHead);
        g.AddUInt32($"{Arch}.ssm.inner_size", DInner);
        g.AddUInt32($"{Arch}.expert_count", (uint)Layers.Max(l => l?.Experts ?? 0));
        g.AddInt32Array($"{Arch}.expert_used_count", [.. Layers.Select(l => l?.TopK ?? 0)]);
        g.AddInt32Array($"{Arch}.expert_feed_forward_length", [.. Layers.Select(l => l?.Inter ?? 0)]);
        g.AddUInt32($"{Arch}.expert_shared_feed_forward_length", SharedInter);
        g.AddUInt32($"{Arch}.expert_shared_count", 1);
        g.AddUInt32($"{Arch}.expert_gating_func", 2);
        g.AddBool($"{Arch}.expert_weights_norm", true);
        g.AddFloat32($"{Arch}.expert_weights_scale", WScale);
        g.AddUInt32($"{Arch}.expert_group_count", 1);
        g.AddUInt32($"{Arch}.expert_group_used_count", 1);
        if (latent) g.AddUInt32($"{Arch}.moe_latent_size", LatentSize);

        int convDim = DInner + 2 * DState;
        int inProj = 2 * DInner + 2 * DState + NHead;
        int expertDim = latent ? LatentSize : Hidden;

        F32(g, "token_embd.weight", [Hidden, Vocab], Rand(rng, Hidden * Vocab, 0.5f));
        for (int l = 0; l < n; l++)
        {
            string p = $"blk.{l}";
            F32(g, $"{p}.attn_norm.weight", [Hidden], RandPos(rng, Hidden));
            if (Layers[l] is not { } spec)
            {
                F32(g, $"{p}.ssm_in.weight", [Hidden, inProj], Rand(rng, Hidden * inProj, 0.05f));
                F32(g, $"{p}.ssm_conv1d.weight", [DConv, convDim], Rand(rng, DConv * convDim, 0.5f));
                F32(g, $"{p}.ssm_conv1d.bias", [convDim], Rand(rng, convDim, 0.1f));
                F32(g, $"{p}.ssm_a", [1, NHead], [.. Enumerable.Range(0, NHead).Select(_ => -(float)(rng.NextDouble() * 0.5 + 0.2))]);
                F32(g, $"{p}.ssm_d", [1, NHead], Rand(rng, NHead, 0.5f));
                F32(g, $"{p}.ssm_dt.bias", [NHead], Rand(rng, NHead, 0.5f));
                F32(g, $"{p}.ssm_norm.weight", [DInner, 1], RandPos(rng, DInner));
                F32(g, $"{p}.ssm_out.weight", [DInner, Hidden], Rand(rng, DInner * Hidden, 0.1f));
                continue;
            }

            int e = spec.Experts, ff = spec.Inter;
            F32(g, $"{p}.ffn_gate_inp.weight", [Hidden, e], Rand(rng, e * Hidden, 0.12f));
            // Bias of the order of the sigmoid probabilities' spread, so it flips top-k for a good share of tokens.
            float[] bias = new float[e];
            for (int i = 0; i < e; i++) bias[i] = zeroBias ? 0f : (i % 2 == 0 ? 1 : -1) * (0.05f + 0.15f * i);
            F32(g, $"{p}.exp_probs_b.bias", [e], bias);
            Bank(g, rng, $"{p}.ffn_up_exps.weight", spec.UpQt, expertDim, ff, e, 0.12f, f32Banks);
            Bank(g, rng, $"{p}.ffn_down_exps.weight", spec.DownQt, ff, expertDim, e, 0.06f, f32Banks);
            F32(g, $"{p}.ffn_up_shexp.weight", [Hidden, SharedInter], Rand(rng, SharedInter * Hidden, 0.08f));
            F32(g, $"{p}.ffn_down_shexp.weight", [SharedInter, Hidden], Rand(rng, Hidden * SharedInter, 0.2f));
            if (latent)
            {
                F32(g, $"{p}.ffn_latent_down.weight", [Hidden, LatentSize], Rand(rng, LatentSize * Hidden, 0.1f));
                F32(g, $"{p}.ffn_latent_up.weight", [LatentSize, Hidden], Rand(rng, Hidden * LatentSize, 0.1f));
            }
        }
        F32(g, "output_norm.weight", [Hidden], RandPos(rng, Hidden));
        F32(g, "output.weight", [Hidden, Vocab], Rand(rng, Vocab * Hidden, 0.08f));

        string path = Path.Combine(scratch, $"nemoe-{latent}-{zeroBias}-{f32Banks}.gguf");
        File.WriteAllBytes(path, g.Build());
        return path;
    }

    /// <summary>A routed bank [ne0 = inK, ne1 = outM, ne2 = experts] stored as <paramref name="qt"/>, or as its dequantized F32 values.</summary>
    private static void Bank(GgufWriter g, Random rng, string name, QuantizationType qt, int inK, int outM, int experts,
        float amp, bool asF32)
    {
        int rows = outM * experts;
        byte[] q = qt == QuantizationType.IQ2_XXS
            ? RandomIq2Xxs(rng, rows * inK)
            : QuantRows(Rand(rng, rows * inK, amp), rows, inK, qt);
        if (!asF32)
        {
            g.AddTensor(name, [inK, outM, experts], (uint)qt, q);
            return;
        }
        float[] f = new float[(long)rows * inK];
        long rowBytes = Dequantize.RowByteSize(inK, qt);
        unsafe
        {
            fixed (byte* p = q)
                for (int r = 0; r < rows; r++)
                    Dequantize.ToFloat32((nint)(p + r * rowBytes), inK, qt, f.AsSpan(r * inK, inK));
        }
        F32(g, name, [inK, outM, experts], f);
    }

    private static byte[] QuantRows(float[] values, int rows, int cols, QuantizationType qt) => qt switch
    {
        QuantizationType.Q4_K => Q4KFixture.QuantizeRows(values, rows, cols),
        QuantizationType.Q5_K => Q5KFixture.QuantizeRows(values, rows, cols),
        QuantizationType.Q6_K => Q6KFixture.QuantizeRows(values, rows, cols),
        QuantizationType.Q8_0 => QuantQ8_0(values, rows, cols),
        _ => throw new ArgumentOutOfRangeException(nameof(qt)),
    };

    private static byte[] QuantQ8_0(float[] values, int rows, int cols)
    {
        long rowBytes = Dequantize.RowByteSize(cols, QuantizationType.Q8_0);
        byte[] bytes = new byte[rowBytes * rows];
        for (int r = 0; r < rows; r++)
            Quantize.FromFloat32(values.AsSpan(r * cols, cols), cols, QuantizationType.Q8_0,
                bytes.AsSpan((int)(r * rowBytes), (int)rowBytes));
        return bytes;
    }

    /// <summary>Random IQ2_XXS blocks (every byte pattern is valid) with d in [0.001, 0.002]: weights of order 0.05-0.1.</summary>
    private static byte[] RandomIq2Xxs(Random rng, int elements)
    {
        byte[] b = new byte[elements / 256 * 66];
        rng.NextBytes(b);
        for (int blk = 0; blk < b.Length; blk += 66)
        {
            ushort h = BitConverter.HalfToUInt16Bits((Half)(0.001f + (float)rng.NextDouble() * 0.001f));
            b[blk] = (byte)(h & 0xFF);
            b[blk + 1] = (byte)(h >> 8);
        }
        return b;
    }

    // ───────────────────────── helpers ─────────────────────────

    private static float MaxDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float worst = 0;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst;
    }

    private static int ArgMax(float[] logits, int row)
    {
        int best = 0;
        for (int i = 1; i < Vocab; i++) if (logits[row * Vocab + i] > logits[row * Vocab + best]) best = i;
        return best;
    }

    private static float[] Rand(Random rng, int n, float amp)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * amp;
        return a;
    }

    private static float[] RandPos(Random rng, int n)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 1.0f + (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return a;
    }

    private static void F32(GgufWriter g, string name, int[] dims, float[] data)
        => g.AddTensor(name, dims, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(data.AsSpan()).ToArray());
}
