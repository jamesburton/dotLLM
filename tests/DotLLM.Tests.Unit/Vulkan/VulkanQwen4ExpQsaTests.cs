using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Tests.Unit.Models.Qwen4Exp;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Vulkan QSA sparse attention (issue #819): kernel-level parity (pool / score / exact top-k select) against the CPU oracle
/// <see cref="Qwen4ExpQsa"/> / <see cref="Qwen4ExpIndexerCache"/>, and whole-model parity beyond the dense limit with a small indexer budget
/// so the sparse regime is reached with a few dozen tokens.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpQsaTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpQsaTests(ITestOutputHelper output) => _out = output;

    private static float[] Rand(Random rng, long n, float scale = 1f)
    {
        var a = new float[n];
        for (long i = 0; i < n; i++) a[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>Oracle top-k: (score desc, index asc), ascending ids.</summary>
    private static int[] OracleSelect(float[] scores, int nb, int budget)
    {
        var order = Enumerable.Range(0, nb).ToArray();
        Array.Sort(order, (a, b) => { int c = scores[b].CompareTo(scores[a]); return c != 0 ? c : a.CompareTo(b); });
        var sel = order.Take(budget).ToArray();
        Array.Sort(sel);
        return sel;
    }

    private static void Barrier(nint cmd) => KernelSupport.ComputeToComputeBarrier(cmd);

    [SkippableTheory]
    [InlineData("random", 600, 512, 1)]       // just above the budget
    [InlineData("random", 4000, 512, 1)]
    [InlineData("random", 65536, 512, 2)]     // the 262K-token regime: 65536 blocks
    [InlineData("ties", 3000, 512, 3)]        // scores quantised to a handful of values: the tie rule (lower index first) decides the set
    [InlineData("allequal", 2000, 512, 4)]
    [InlineData("zeros", 1500, 512, 5)]       // relu can zero a lot of blocks
    [InlineData("random", 100, 16, 6)]        // small budget
    [InlineData("ties", 700, 32, 7)]
    public void Select_MatchesOracle_ExactSetIncludingTies(string kind, int nb, int budget, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Qwen4ExpQsaKernels.Create(device, spvDir);
        const int Block = 4;
        var rng = new Random(seed);
        // Three queries with different visible block counts: nb, nb-1 (if > budget) and a dense one (<= budget, must be skipped).
        int[] nbs = [nb, Math.Max(nb - 7, budget + 1), budget];
        int Q = nbs.Length, nbCap = nb + 3;
        var scores = new float[(long)Q * nbCap];
        for (int q = 0; q < Q; q++)
            for (int b = 0; b < nbs[q]; b++)
            {
                float s = kind switch
                {
                    "random" => (float)rng.NextDouble() * 3f,
                    "ties" => rng.Next(0, 6) * 0.5f,
                    "allequal" => 1.25f,
                    "zeros" => rng.Next(0, 5) == 0 ? (float)rng.NextDouble() : 0f,
                    _ => 0f,
                };
                scores[(long)q * nbCap + b] = s;
            }
        using var bSel = device.AllocateDeviceLocal((long)budget * 4);
        var init = new int[budget];
        Array.Fill(init, -7);
        var raw = new float[budget];
        for (int q = 0; q < Q; q++)
        {
            device.Upload(MemoryMarshal.AsBytes<int>(init), bSel);
            // Re-pack this query's scores into row 0 of a dedicated score buffer.
            var one = new float[nbCap];
            Array.Copy(scores, (long)q * nbCap, one, 0, nbCap);
            using var bOne = device.AllocateDeviceLocal(nbCap * 4L);
            device.Upload(one, bOne);
            int firstPos = nbs[q] * Block - 1;
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
            k.RecordSelect(ctx.CommandBuffer, bOne, bSel, qBase: 0, qCount: 1, firstPos: firstPos, Block, budget, nbCap);
            KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
            ctx.SubmitAndWait();
            device.Download(bSel, raw);
            var sel = MemoryMarshal.Cast<float, int>(raw).Slice(0, budget).ToArray();
            if (nbs[q] <= budget)
            {
                Assert.All(sel, v => Assert.Equal(-7, v));   // dense queries are left alone
                continue;
            }
            var want = OracleSelect(one, nbs[q], budget);
            Assert.Equal(want, sel);
        }
        _out.WriteLine($"{kind} nb={nb} budget={budget}: exact set + ascending order for {Q} queries");
    }

    [SkippableTheory]
    [InlineData(4, 128, 64, 5000)]
    [InlineData(3, 16, 8, 700)]
    public void PoolAndScore_MatchOracle(int heads, int dim, int ropeDim, int tokens)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Qwen4ExpQsaKernels.Create(device, spvDir);
        const int Block = 4;
        float theta = 1.0e7f, eps = 1e-6f;
        var rng = new Random(11);
        var raw = Rand(rng, (long)tokens * dim);
        var gamma = Rand(rng, dim, 0.5f);
        for (int i = 0; i < dim; i++) gamma[i] += 1f;

        // Oracle pooled keys through the CPU cache.
        int maxPos = tokens + 8;
        var cos = new float[(long)maxPos * (ropeDim / 2)]; var sin = new float[cos.Length];
        RoPE.PrecomputeFrequencyTable(maxPos, ropeDim, theta, cos, sin);
        using var cache = new Qwen4ExpIndexerCache(dim, Block);
        cache.Append(raw, tokens, gamma, eps, cos, sin, ropeDim);
        int blocks = tokens / Block;
        var oracle = cache.Pooled.Slice(0, blocks * dim).ToArray();

        using var bRaw = device.AllocateDeviceLocal(raw.Length * 4L);
        using var bGamma = device.AllocateDeviceLocal(dim * 4L);
        using var bPooled = device.AllocateDeviceLocal((blocks + 1L) * dim * 4);
        device.Upload(raw, bRaw); device.Upload(gamma, bGamma);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
            k.RecordPool(ctx.CommandBuffer, bRaw, bGamma, bPooled, 0, blocks, dim, ropeDim, Block, eps, theta);
            KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
            ctx.SubmitAndWait();
        }
        var pooled = new float[blocks * dim];
        device.Download(bPooled, pooled);
        double num = 0, den = 0, maxAbs = 0;
        for (int i = 0; i < pooled.Length; i++)
        {
            double d = pooled[i] - oracle[i]; num += d * d; den += (double)oracle[i] * oracle[i]; maxAbs = Math.Max(maxAbs, Math.Abs(d));
        }
        _out.WriteLine($"pool dim={dim} blocks={blocks}: relL2 {Math.Sqrt(num / den):E3}, max abs {maxAbs:E3}");
        Assert.True(Math.Sqrt(num / den) < 2e-4, "pooled keys diverge from the oracle");

        // Scoring: queries at the tail of the range, compared with the scalar oracle on the oracle's pooled keys.
        int Q = 6, first = tokens - Q;
        var iq = Rand(rng, (long)Q * heads * dim);
        int budget = 8;                                    // tiny budget so these queries are all sparse
        int nbCap = blocks + 1;
        using var bIq = device.AllocateDeviceLocal(iq.Length * 4L);
        using var bScores = device.AllocateDeviceLocal((long)Q * nbCap * 4);
        device.Upload(iq, bIq);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
            k.RecordScore(ctx.CommandBuffer, bIq, bPooled, bScores, 0, Q, first, heads, dim, Block, budget, nbCap, blocks);
            KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
            ctx.SubmitAndWait();
        }
        var scores = new float[Q * nbCap];
        device.Download(bScores, scores);
        for (int q = 0; q < Q; q++)
        {
            int nb = (first + q + 1) / Block;
            var want = new float[nb];
            Qwen4ExpQsa.ScoreBlocksScalar(iq.AsSpan(q * heads * dim, heads * dim), heads, dim, oracle, nb, want);
            double n2 = 0, d2 = 0;
            for (int b = 0; b < nb; b++) { double d = scores[q * nbCap + b] - want[b]; n2 += d * d; d2 += (double)want[b] * want[b]; }
            Assert.True(Math.Sqrt(n2 / Math.Max(d2, 1e-30)) < 2e-4, $"query {q}: block scores diverge ({Math.Sqrt(n2 / d2):E3})");
        }
    }

    // ───────────────────────── whole model, sparse regime ─────────────────────────

    /// <summary>Tiny geometry with a 4-block (16-token) indexer budget: dense up to 19 tokens, sparse from the 20th.</summary>
    private static byte[] SmallBudgetModel(int budgetTokens = 16, Q4eGeometry? g = null)
        => Qwen4ExpRandomGguf.Build((g ?? Qwen4ExpRandomGguf.Tiny) with { Budget = budgetTokens, Context = 512 }, Q4eQuant.F32);

    private static (double RelL2, double Kl, bool Top1) Cmp(float[] a, float[] b) => VulkanQwen4ExpParityTests.Compare(a, b);

    private static double[] RowErrors(float[] a, float[] b, int rows)
    {
        int len = a.Length / rows;
        var r = new double[rows];
        for (int t = 0; t < rows; t++)
        {
            double n2 = 0, d2 = 0;
            for (int i = 0; i < len; i++) { double d = a[t * len + i] - b[t * len + i]; n2 += d * d; d2 += (double)a[t * len + i] * a[t * len + i]; }
            r[t] = Math.Sqrt(n2 / Math.Max(d2, 1e-30));
        }
        return r;
    }

    [SkippableFact]
    public void BlockSelectionFlipRate_IsRare()
    {
        // Only the top-k boundary can make the GPU and the oracle disagree (float summation order). Characterise how often over several prompts.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
        int T = 100, total = 0, flipped = 0;
        var cpuTrace = new Dictionary<string, float[]>();
        rig.Cpu.Trace = (n, d, r, c) => cpuTrace[n] = d.ToArray();
        var vkTrace = new Dictionary<string, float[]>();
        rig.Vk.Trace = (n, d, r, c) => vkTrace[n] = d;
        var pos = Enumerable.Range(0, T).ToArray();
        for (int seed = 1; seed <= 8; seed++)
        {
            var ids = VulkanQwen4ExpParityTests.Ids(T, rig.Config.VocabSize, seed);
            rig.Cpu.Forward(ids, pos, -1);
            rig.Vk.Forward(ids, pos, -1);
            var rows = RowErrors(cpuTrace["blk.3.l_out"], vkTrace["blk.3.l_out"], T);
            int bad = rows.Count(r => r > 1e-3);
            total += T - 19; flipped += bad;
            _out.WriteLine($"seed {seed}: {bad} rows above 1e-3, worst {rows.Max():E2}");
        }
        _out.WriteLine($"flipped rows: {flipped} of {total} sparse rows");
        Assert.True(flipped <= total / 25, $"{flipped} of {total} rows differ materially");
    }

    [SkippableFact]
    public void AMaterialRowMismatch_IsATopKBoundaryNearTie()
    {
        // The only way the GPU and the oracle can disagree materially is a different block at the top-k boundary. For every row that does disagree,
        // the GPU's own scores must show rank-`budget` and rank-`budget+1` within float noise of each other (a genuine near-tie), otherwise the
        // mismatch is a real bug in selection or in the gather attention. (Tiny geometry: layer 3 is the only QSA layer, 4-block budget.)
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
        const int T = 100, Budget = 4;
        var cpuTrace = new Dictionary<string, float[]>();
        rig.Cpu.Trace = (n, d, r, c) => cpuTrace[n] = d.ToArray();
        var vkTrace = new Dictionary<string, float[]>();
        rig.Vk.Trace = (n, d, r, c) => vkTrace[n] = d;
        var pos = Enumerable.Range(0, T).ToArray();
        int examined = 0;
        foreach (int seed in new[] { 7, 3, 5 })
        {
            var ids = VulkanQwen4ExpParityTests.Ids(T, rig.Config.VocabSize, seed);
            rig.Cpu.Forward(ids, pos, -1);
            rig.Vk.Forward(ids, pos, -1);
            var rows = RowErrors(cpuTrace["blk.3.l_out"], vkTrace["blk.3.l_out"], T);
            var scoresBuf = rig.Vk.Qsa.ScoresBuffer!;
            int nbCap = rig.Vk.Qsa.NbCap;
            var scores = new float[scoresBuf.Size / 4];
            rig.Device.Download(scoresBuf, scores);
            for (int r = 0; r < T; r++)
            {
                if (rows[r] <= 1e-3) continue;
                int nb = (r + 1) / 4;
                var s = scores.AsSpan(r * nbCap, nb).ToArray().OrderByDescending(x => x).ToArray();
                double gap = (s[Budget - 1] - s[Budget]) / Math.Max(Math.Abs(s[Budget - 1]), 1e-30);
                _out.WriteLine($"seed {seed} row {r}: relL2 {rows[r]:E2}; GPU rank-{Budget} {s[Budget - 1]:G9} vs rank-{Budget + 1} {s[Budget]:G9}, relative gap {gap:E2}");
                Assert.True(gap < 5e-4, $"row {r} disagrees with the oracle but its top-k boundary is not a near-tie (gap {gap:E2}): selection or gather is wrong");
                examined++;
            }
        }
        _out.WriteLine($"{examined} mismatching rows examined");
    }

    [SkippableFact]
    public void Prefill_BeyondDenseLimit_MatchesOracle_AndOracleSparseDiffersFromDense()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
        Assert.Equal(19, rig.Vk.DenseContextLimit);
        int T = 100;
        var ids = VulkanQwen4ExpParityTests.Ids(T, rig.Config.VocabSize);
        var pos = Enumerable.Range(0, T).ToArray();

        var cpuTrace = new Dictionary<string, float[]>();
        rig.Cpu.Trace = (n, d, r, c) => cpuTrace[n] = d.ToArray();
        var vkTrace = new Dictionary<string, float[]>();
        rig.Vk.Trace = (n, d, r, c) => vkTrace[n] = d;
        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, pos, -1), T - 1);
        var vk = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);
        for (int il = 0; il < rig.Config.NumLayers; il++)
        {
            var rows = RowErrors(cpuTrace[$"blk.{il}.l_out"], vkTrace[$"blk.{il}.l_out"], T);
            int bad = rows.Count(r => r > 1e-3);
            _out.WriteLine($"layer {il}: median row relL2 {rows.OrderBy(r => r).ElementAt(T / 2):E2}, rows above 1e-3: {bad}, worst {rows.Max():E2} (row {Array.IndexOf(rows, rows.Max())})");
            // A near-tie at the top-k boundary can legitimately flip one block between the GPU's and the oracle's float summation order.
            Assert.True(bad <= T / 25, $"layer {il}: {bad} of {T} rows diverge");
        }
        var (rl2, kl, top1) = Cmp(cpu, vk);
        _out.WriteLine($"sparse prefill vs oracle: relL2 {rl2:E3} KL {kl:E3} top1 {top1}");
        Assert.True(rl2 < 3e-3 && kl < 1e-4);
        Assert.True(rig.Vk.Qsa.SparseLayerRecordings > 0, "the sparse path never ran");

        // Sensitive control: the oracle's own dense attention must differ from its sparse attention at this length, otherwise the
        // comparison above could not tell sparse from dense.
        rig.Cpu.SetForceDenseAttention(true);
        var cpuDense = Q4eRig.Row(rig.Cpu.Forward(ids, pos, -1), T - 1);
        rig.Cpu.SetForceDenseAttention(false);
        var (dl2, dkl, _) = Cmp(cpu, cpuDense);
        _out.WriteLine($"oracle sparse vs oracle dense: relL2 {dl2:E3} KL {dkl:E3}");
        Assert.True(dl2 > 20 * rl2 && dl2 > 1e-2, $"control is insensitive: dense-vs-sparse {dl2:E3} vs Vulkan-vs-oracle {rl2:E3}");
        var (vl2, _, _) = Cmp(cpuDense, vk);
        Assert.True(vl2 > 20 * rl2, "the Vulkan result is closer to the dense oracle than expected: it may not be sparse");
    }

    [SkippableFact]
    public void TokenByToken_CrossesTheDenseLimit_AndGreedyDecodeFollowsOracle()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
        int V = rig.Config.VocabSize;
        var ids = VulkanQwen4ExpParityTests.Ids(70, V, seed: 21);
        float[]? cpu = null, vk = null;
        double worst = 0;
        for (int i = 0; i < ids.Length; i++)
        {
            cpu = Q4eRig.Row(rig.Cpu.Forward([ids[i]], [i], -1), 0);
            vk = Q4eRig.Row(rig.Vk.Forward([ids[i]], [i], -1), 0);
            if (i >= 15) { var (r, _, _) = Cmp(cpu, vk); worst = Math.Max(worst, r); }
        }
        _out.WriteLine($"token-by-token decode through the dense limit (19): worst relL2 {worst:E3}");
        Assert.True(worst < 3e-3, $"worst relL2 {worst:E3}");
        for (int s = 0; s < 12; s++)
        {
            int next = Array.IndexOf(cpu!, cpu!.Max());
            Assert.Equal(next, Array.IndexOf(vk!, vk!.Max()));
            int p = ids.Length + s;
            cpu = Q4eRig.Row(rig.Cpu.Forward([next], [p], -1), 0);
            vk = Q4eRig.Row(rig.Vk.Forward([next], [p], -1), 0);
            var (r, _, _) = Cmp(cpu, vk);
            Assert.True(r < 3e-3, $"greedy step {s}: relL2 {r:E3}");
        }
    }

    [SkippableFact]
    public void ChunkedPrefill_EqualsSingleShot()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
        int V = rig.Config.VocabSize, T = 90;
        var ids = VulkanQwen4ExpParityTests.Ids(T, V, seed: 33);
        var pos = Enumerable.Range(0, T).ToArray();
        var single = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);

        using var st = rig.Vk.CreateState();
        float[]? chunked = null;
        foreach (var (a, b) in new[] { (0, 13), (13, 51), (51, 52), (52, 90) })   // straddles the limit, a 1-row chunk, a block boundary
            chunked = Q4eRig.Row(rig.Vk.Forward(ids.AsSpan(a, b - a), pos.AsSpan(a, b - a), -1, st), 0);
        var (r1, _, _) = Cmp(single, chunked!);
        _out.WriteLine($"chunked vs single-shot: relL2 {r1:E3}");
        Assert.True(r1 < 1e-3, $"chunked prefill diverges from single-shot: {r1:E3}");
    }

    [SkippableFact]
    public void ForwardLongerThanThePlannedRows_IsChunked_AndKeepsAllRowLogits()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const string Env = "DOTLLM_VK_PLANNED_ROWS";
        string? old = Environment.GetEnvironmentVariable(Env);
        Environment.SetEnvironmentVariable(Env, "16");
        try
        {
            using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
            Assert.Equal(16, rig.Vk.MaxRowsPerForward);
            int T = 60;
            var ids = VulkanQwen4ExpParityTests.Ids(T, rig.Config.VocabSize, seed: 8);
            var pos = Enumerable.Range(0, T).ToArray();
            Assert.True(rig.Vk.TrySetAllRowLogitsLimit(T));
            using var cpu = rig.Cpu.Forward(ids, pos, -1);
            using var vk = rig.Vk.Forward(ids, pos, -1);
            Assert.Equal(T, vk.Shape[0]);
            double worst = 0;
            for (int r = 0; r < T; r++) worst = Math.Max(worst, VulkanQwen4ExpParityTests.Compare(Q4eRig.Row(cpu, r), Q4eRig.Row(vk, r)).RelL2);
            _out.WriteLine($"60 rows in 16-row chunks (sparse from row 19): worst row relL2 {worst:E3}");
            Assert.True(worst < 3e-3);
            // Without the all-rows opt-in only the last row comes back, still through chunks.
            using var rig2 = new Q4eRig(SmallBudgetModel(), spvDir);
            using var last = rig2.Vk.Forward(ids, pos, -1, null, lastTokenLogitsOnly: true);
            Assert.Equal(1, last.Shape[0]);
            Assert.True(VulkanQwen4ExpParityTests.Compare(Q4eRig.Row(cpu, T - 1), Q4eRig.Row(last, 0)).RelL2 < 3e-3);
        }
        finally { Environment.SetEnvironmentVariable(Env, old); }
    }

    [SkippableFact]
    public void FullHeadDim256Geometry_BeyondDenseLimit_MatchesOracle()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        // The released attention geometry (head_dim 256, 64 rotary dims, 4 query heads over 2 KV heads) and a 64-wide indexer.
        using var rig = new Q4eRig(SmallBudgetModel(32, Qwen4ExpRandomGguf.Hd256), spvDir);
        int T = 150;
        var ids = VulkanQwen4ExpParityTests.Ids(T, rig.Config.VocabSize, seed: 4);
        var pos = Enumerable.Range(0, T).ToArray();
        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, pos, -1), T - 1);
        var vk = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);
        var (rl2, kl, _) = Cmp(cpu, vk);
        _out.WriteLine($"hd256 sparse prefill: relL2 {rl2:E3} KL {kl:E3}");
        Assert.True(rl2 < 3e-3 && kl < 1e-4);
    }
}
