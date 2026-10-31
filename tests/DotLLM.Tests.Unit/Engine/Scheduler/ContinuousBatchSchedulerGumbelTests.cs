using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.Scheduler;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// #1434: the continuous-batch scheduler's batched decode in Gumbel mode. The host path (<c>SamplerPipeline.Sample</c>, keyed on
/// <c>GeneratedTokens.Count</c>) and the device path (<c>IModel.ForwardBatchGumbelSampled</c> with per-row seed / index / 1/T) must emit
/// the same tokens, because Gumbel-max is a pure function of (logits, seed, generation index).
/// </summary>
public sealed partial class ContinuousBatchSchedulerTests
{
    private const int GumbelMaxTokens = 14;

    // Flat-ish, position- and token-dependent logits: the noise (not a single peak) decides the token. Token 0 (EOS) is suppressed.
    private static float GumbelLogit(int lastToken, int position, int v)
    {
        if (v == EosTokenId) return -50f;
        uint h = (uint)(lastToken * 73856093) ^ (uint)(position * 19349663) ^ (uint)(v * 83492791);
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        return (h % 2000) / 1000f - 1f;   // [-1, 1)
    }

    private static InferenceRequest GumbelRequest(int promptLen, int seed, float temperature = 0.9f, int topK = 0)
    {
        var tokens = new int[promptLen];
        for (int i = 0; i < promptLen; i++) tokens[i] = i + 1;
        return new InferenceRequest
        {
            TokenIds = tokens,
            Options = new InferenceOptions
            {
                Temperature = temperature, TopK = topK, Seed = seed, GumbelSampling = true, MaxTokens = GumbelMaxTokens,
            },
        };
    }

    private static async Task<int[][]> RunGumbel(bool device, int[] seeds, float temperature = 0.9f, int topK = 0,
        Action<TestFixture>? inspect = null, bool useDeviceFlag = true)
    {
        using var fix = new TestFixture(requiresPerSequenceState: false);
        fix.Model.LogitFn = GumbelLogit;
        fix.Model.DeviceGumbelBatch = device;
        fix.Scheduler.UseDeviceGumbel = useDeviceFlag;
        var handles = seeds.Select((s, i) => fix.Scheduler.Submit(GumbelRequest(3 + i, s, temperature, topK))).ToArray();
        DriveUntilIdle(fix.Scheduler);
        var outs = new int[seeds.Length][];
        for (int i = 0; i < handles.Length; i++) outs[i] = (await handles[i].Completion).GeneratedTokenIds;
        inspect?.Invoke(fix);
        return outs;
    }

    [Fact]
    public async Task DeviceGumbelBatch_EmitsTheSameTokensAsTheHostPath()
    {
        int[] seeds = [11, 12, 13, 14];
        long hostSteps = -1, devSteps = -1; int hostCalls = -1, devCalls = -1;
        var host = await RunGumbel(device: false, seeds, inspect: f => { hostSteps = f.Scheduler.DeviceGumbelDecodeSteps; hostCalls = f.Model.DeviceGumbelBatchCalls; });
        var dev = await RunGumbel(device: true, seeds, inspect: f => { devSteps = f.Scheduler.DeviceGumbelDecodeSteps; devCalls = f.Model.DeviceGumbelBatchCalls; });

        Assert.Equal(0, hostSteps);                       // the model did not advertise it: logits path
        Assert.Equal(0, hostCalls);
        Assert.True(devSteps > 0 && devCalls == devSteps, $"device path never ran ({devSteps} steps, {devCalls} calls)");
        for (int i = 0; i < seeds.Length; i++)
        {
            Assert.Equal(GumbelMaxTokens, host[i].Length);
            Assert.Equal(host[i], dev[i]);
        }
        // Discriminating: distinct seeds give distinct text, so equality above is not a degenerate constant stream.
        Assert.NotEqual(host[0], host[1]);
    }

    [Fact]
    public async Task DeviceGumbelBatch_PassesTheRealSeedIndexAndTemperature()
    {
        List<GumbelDraw>? last = null;
        using var fix = new TestFixture(requiresPerSequenceState: false);
        fix.Model.LogitFn = GumbelLogit;
        fix.Model.DeviceGumbelBatch = true;
        var opts = new InferenceOptions { Temperature = 0.5f, Seed = 77, GumbelSampling = true, MaxTokens = 4 };
        var pipeline = new DotLLM.Engine.Samplers.SamplerPipeline(opts);
        var a = fix.Scheduler.Submit(new InferenceRequest { TokenIds = [1, 2, 3], Options = opts });
        var b = fix.Scheduler.Submit(new InferenceRequest { TokenIds = [1, 2, 3, 4], Options = opts });
        // Step until the first batched decode happens, then inspect its draws.
        for (int i = 0; i < 50 && fix.Model.DeviceGumbelBatchCalls == 0; i++) fix.Scheduler.Step();
        last = fix.Model.LastDraws.ToList();
        DriveUntilIdle(fix.Scheduler);
        await a.Completion; await b.Completion;

        Assert.Equal(2, last.Count);
        Assert.All(last, d =>
        {
            Assert.Equal(pipeline.GumbelSeed, d.Seed);          // same options => same derived seed
            Assert.Equal(1, d.TokenIndex);                       // first decode step: one token (from prefill) generated so far
            Assert.Equal(1f / 0.5f, d.InvTemperature);
        });
    }

    [Fact]
    public async Task DeviceGumbelBatch_IsSeededReproducible()
    {
        int[] seeds = [5, 6, 7];
        var a = await RunGumbel(device: true, seeds);
        var b = await RunGumbel(device: true, seeds);
        for (int i = 0; i < seeds.Length; i++) Assert.Equal(a[i], b[i]);
    }

    [Fact]
    public async Task DeviceGumbelBatch_KeepsTheHostPath_ForIneligiblePipelines_AndWhenDisabled()
    {
        int[] seeds = [21, 22, 23];
        long steps = -1;
        var topK = await RunGumbel(device: true, seeds, topK: 5, inspect: f => steps = f.Scheduler.DeviceGumbelDecodeSteps);
        Assert.Equal(0, steps);                                  // top-k needs the host: whole batch keeps the logits path
        var hostTopK = await RunGumbel(device: false, seeds, topK: 5);
        for (int i = 0; i < seeds.Length; i++) Assert.Equal(hostTopK[i], topK[i]);

        steps = -1;
        var off = await RunGumbel(device: true, seeds, useDeviceFlag: false, inspect: f => steps = f.Scheduler.DeviceGumbelDecodeSteps);
        Assert.Equal(0, steps);
        var on = await RunGumbel(device: true, seeds);
        for (int i = 0; i < seeds.Length; i++) Assert.Equal(on[i], off[i]);   // switching it off never changes the text
    }
}
