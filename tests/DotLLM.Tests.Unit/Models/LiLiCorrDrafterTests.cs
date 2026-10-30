using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Architectures.DFlash;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// LiLiCorr drafting on a Nemotron-H MoE (Bolt-format) trunk (issue #1237), CPU host: the drafter loads (NVFP4 scales, BF16,
/// sinks, the lattice head), the recurrent checkpoint round-trips, and speculative decoding is token-identical to plain greedy.
/// The end-to-end gate is shared with the Vulkan host (<c>VulkanNemotronHDFlashTests</c>).
/// </summary>
public sealed class LiLiCorrDrafterTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-lilicorr-{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _out;

    public LiLiCorrDrafterTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>A target backend for the gates: the model and a KV-cache for it.</summary>
    internal delegate (IModel Model, IKvCache Kv) BackendFactory(GgufFile gguf, ModelConfig config);

    internal static (IModel, IKvCache) Cpu(GgufFile gguf, ModelConfig config)
    {
        var m = NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        return (m, new SimpleKvCache(m.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength));
    }

    [Fact]
    public void Drafter_Loads_LiLiCorrHead_SinksAndScales()
    {
        string path = SyntheticBoltLiLiCorrGguf.WriteDrafter(Path.Combine(_dir, "drafter.gguf"), seed: 1);
        using var drafter = DFlashDrafter.Open(path, SyntheticBoltLiLiCorrGguf.Hidden, SyntheticBoltLiLiCorrGguf.Layers,
            SyntheticBoltLiLiCorrGguf.Vocab, new DFlashCpuBackend(null));
        Assert.True(drafter.IsLiLiCorr);
        Assert.False(drafter.IsDFlash2);
        Assert.Equal(SyntheticBoltLiLiCorrGguf.Block, drafter.BlockSize);
        Assert.Equal(SyntheticBoltLiLiCorrGguf.Block + 1, drafter.AnchorRows);
        // The four sink vectors and the scaled NVFP4 matrices are bound (sinks join the device vectors).
        Assert.Equal(SyntheticBoltLiLiCorrGguf.DLayers, drafter.DeviceVectors.Count(v => v.Length == SyntheticBoltLiLiCorrGguf.DHeads));
        Assert.Contains(drafter.DeviceMatrices, m => m.Qt == QuantizationType.NVFP4 && m.Scale != 1f);
        Assert.Contains(drafter.DeviceMatrices, m => m.Qt == QuantizationType.BF16);
    }

    /// <summary>The sink only enlarges the softmax denominator: with a huge negative sink the attention equals the sink-less one.</summary>
    [Fact]
    public unsafe void CpuAttend_Sinks_AddOneLogitToTheDenominator()
    {
        string path = SyntheticBoltLiLiCorrGguf.WriteDrafter(Path.Combine(_dir, "drafter.gguf"), seed: 2);
        var backend = new DFlashCpuBackend(null);
        using var drafter = DFlashDrafter.Open(path, SyntheticBoltLiLiCorrGguf.Hidden, SyntheticBoltLiLiCorrGguf.Layers,
            SyntheticBoltLiLiCorrGguf.Vocab, backend);
        const int B = 3;
        int q = SyntheticBoltLiLiCorrGguf.DHeads * SyntheticBoltLiLiCorrGguf.DHeadDim, kv = SyntheticBoltLiLiCorrGguf.DKvHeads * SyntheticBoltLiLiCorrGguf.DHeadDim;
        var rng = new Random(3);
        float[] Q = Enumerable.Range(0, B * q).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        float[] K = Enumerable.Range(0, B * kv).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        float[] V = Enumerable.Range(0, B * kv).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        float[] Run(float[]? sinks)
        {
            float[] o = new float[B * q];
            fixed (float* qp = Q, kp = K, vp = V, op = o)
                backend.Attend(0, 0, 1, 0, 0, B, (nint)qp, (nint)kp, (nint)vp, (nint)op, sinks);
            return o;
        }
        float[] none = Run(null);
        float[] far = Run(Enumerable.Repeat(-1e4f, SyntheticBoltLiLiCorrGguf.DHeads).ToArray());
        float[] zero = Run(new float[SyntheticBoltLiLiCorrGguf.DHeads]);
        Assert.Equal(none, far);
        // A sink at logit 0 shrinks every output by sum / (sum + 1): never equal, never larger in magnitude.
        for (int i = 0; i < none.Length; i++)
            Assert.True(Math.Abs(zero[i]) < Math.Abs(none[i]) || none[i] == 0f, $"elem {i}: {zero[i]} vs {none[i]}");
    }

    /// <summary>Checkpoint, advance the recurrent state, restore: the next forward equals the one before the detour.</summary>
    [Fact]
    public unsafe void Checkpoint_Restore_RewindsTheMambaState()
    {
        string trunk = SyntheticBoltLiLiCorrGguf.WriteTarget(Path.Combine(_dir, "trunk.gguf"));
        using var gguf = GgufFile.Open(trunk);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (model, kv) = Cpu(gguf, config);
        using var _m = (IDisposable)model;
        using var _k = kv;
        Assert.True(model.SupportsRecurrentStateCheckpoint);
        using (model.Forward([1, 2, 3], [0, 1, 2], -1, kv)) { }
        object? ckpt = model.CheckpointRecurrentState();
        float[] first;
        using (var l = model.Forward([4, 5], [3, 4], -1, kv)) first = new ReadOnlySpan<float>((void*)l.DataPointer, 2 * config.VocabSize).ToArray();
        model.RestoreRecurrentState(ckpt);
        kv.Rollback(3);
        float[] again;
        using (var l = model.Forward([4, 5], [3, 4], -1, kv)) again = new ReadOnlySpan<float>((void*)l.DataPointer, 2 * config.VocabSize).ToArray();
        (ckpt as IDisposable)?.Dispose();
        Assert.Equal(first, again);
    }

    /// <summary>
    /// End-to-end gate: MTP-decoder rounds drafted by a LiLiCorr drafter on the Bolt-format trunk emit exactly the plain greedy
    /// sequence, and some drafts are accepted and some rejected (the replay rollback ran).
    /// </summary>
    [Fact]
    public void SpeculativeDecode_WithLiLiCorr_IsTokenIdenticalToGreedy()
        => AssertTokenIdentical(_dir, Cpu, _out);

    /// <summary>The gate on any backend: the greedy reference is the same backend's plain decode.</summary>
    internal static void AssertTokenIdentical(string dir, BackendFactory backend, ITestOutputHelper output, int seeds = 24)
    {
        string trunk = SyntheticBoltLiLiCorrGguf.WriteTarget(Path.Combine(dir, "trunk.gguf"));
        int[] prompt = [1, 5, 3, 7, 2, 9, 4];
        const int NewTokens = 40;
        const int Block = SyntheticBoltLiLiCorrGguf.Block;

        List<int> greedy;
        using (var g = GgufFile.Open(trunk))
        {
            var cfg = GgufModelConfigExtractor.Extract(g.Metadata);
            var (plain, kv) = backend(g, cfg);
            using (plain as IDisposable) using (kv)
            {
                greedy = [.. prompt];
                using (var l = plain.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, kv))
                    greedy.Add(DFlashDrafterTests.ArgMaxLast(l, cfg.VocabSize));
                while (greedy.Count < prompt.Length + NewTokens + Block)
                    using (var l = plain.Forward([greedy[^1]], [greedy.Count - 1], -1, kv))
                        greedy.Add(DFlashDrafterTests.ArgMaxLast(l, cfg.VocabSize));
            }
        }
        output.WriteLine($"greedy: [{string.Join(",", greedy.Skip(prompt.Length))}]");

        bool accepted = false, rejected = false;
        for (int seed = 0; seed < seeds && !(accepted && rejected); seed++)
        {
            string drafterPath = SyntheticBoltLiLiCorrGguf.WriteDrafter(Path.Combine(dir, $"drafter-{seed}.gguf"), seed);
            var perRound = RunDrafted(trunk, drafterPath, backend, prompt, NewTokens, greedy);
            accepted |= perRound.Any(n => n > 1);
            rejected |= perRound.Any(n => n < Block);
            output.WriteLine($"seed {seed}: emitted/round [{string.Join(",", perRound)}]");
        }
        Assert.True(accepted, "no seed got a draft accepted: the accept path never ran");
        Assert.True(rejected, "no round rejected a draft: the rollback path never ran");
    }

    private static List<int> RunDrafted(string trunk, string drafterPath, BackendFactory backend, int[] prompt, int newTokens, List<int> greedy)
    {
        const int Block = SyntheticBoltLiLiCorrGguf.Block;
        using var gguf = GgufFile.Open(trunk);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (model, kvCache) = backend(gguf, config);
        using var _m = model as IDisposable;
        using var _k = kvCache;
        Assert.False(model.SupportsMtp);
        ((IDFlashDraftAttachable)model).AttachDFlashDrafter(drafterPath);
        Assert.True(model.SupportsMtp && model.SupportsMtpArgMax);
        Assert.Equal(Block - 1, model.MtpPreferredDraftTokens);

        using var mtp = model.CreateMtpState()!;
        var ids = new List<int>(prompt);
        using (var l = model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, kvCache, adapter: null, mtp))
            ids.Add(DFlashDrafterTests.ArgMaxLast(l, config.VocabSize));
        Assert.Equal(prompt.Length, mtp.CurrentLength);

        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var outBuf = new int[Block];
        var perRound = new List<int>();
        while (ids.Count < prompt.Length + newTokens)
        {
            int position = ids.Count - 1;
            var r = decoder.DraftAndVerify(model, kvCache, mtp, pipeline, ids, null, position, config.VocabSize, Block - 1, outBuf);
            Assert.True(r.AcceptedCount >= 1);
            for (int i = 0; i < r.AcceptedCount; i++) ids.Add(outBuf[i]);
            perRound.Add(r.AcceptedCount);
            Assert.Equal(ids.Count - 1, mtp.CurrentLength);
        }
        Assert.Equal(greedy.Take(ids.Count), ids);
        return perRound;
    }
}
