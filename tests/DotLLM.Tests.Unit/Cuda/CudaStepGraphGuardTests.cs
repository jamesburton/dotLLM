using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1920 / #1919: the dense CUDA step graph (#1922 E2) captures a whole decode step and replays it; a replay never reaches the per-layer
/// residual-modifier call or the per-layer tap sites, so each is guarded out of the step graph. Written WITHOUT a GPU: never run here.
/// <para>
/// MUTANTS (apply on a CUDA box, rebuild, run this class; each must make exactly the named test fail; revert afterwards):
/// </para>
/// <list type="number">
/// <item>Modifier guard. File <c>src/DotLLM.Cuda/CudaTransformerModel.StepGraph.cs</c>, in <c>StepGraphEligible</c>:
///   old <c>            || ResidualModifier is not null   // #1919: a graph replay never reaches the per-layer PostLayer residual-modifier call</c>
///   new (line deleted).
///   <see cref="StepGraph_IsNotTaken_WhenAResidualModifierIsAttached"/> must fail: the step graph launches (StepGraphLaunches advances
///   and/or the modifier call count is short of steps x layers), because a replay never calls the modifier.</item>
/// <item>Tap guard. File <c>src/DotLLM.Cuda/CudaTransformerModel.cs</c>, the #1922 E2 call site:
///   old <c>if (ts is null &amp;&amp; !_stepGraphCapturing &amp;&amp; !_moeGraphCapturing</c>
///   new <c>if (!_stepGraphCapturing &amp;&amp; !_moeGraphCapturing</c>.
///   <see cref="StepGraph_IsNotTaken_WhenAHiddenStateTapIsAttached"/> must fail: StepGraphLaunches advances (counter) and/or the
///   PostLayer callback is missed on replayed steps (samples &lt; steps x layers).</item>
/// </list>
/// Each test first proves the same configuration DOES take the step graph with nothing attached (a failure, not a skip, when it cannot
/// be engaged), so a guard cannot pass by the graph never being reachable.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaStepGraphGuardTests
{
    private readonly ITestOutputHelper _out;
    public CudaStepGraphGuardTests(ITestOutputHelper output) => _out = output;

    private const int Steps = 4;
    private static readonly TimeSpan Watchdog = TimeSpan.FromSeconds(60);

    private sealed record Run(List<float[]> Rows, int[] Tokens, int StepGraphLaunches, int Calls, int NumLayers);

    private sealed class CountingModifier : ICudaResidualModifier
    {
        public int Calls;
        public bool AppliesTo(int layerIndex) => true;
        public void Apply(nint residualDevicePtr, CudaResidualDType dtype, int layerIndex, int tokenCount, int hiddenSize, nint stream) => Calls++;
    }

    private enum Attach { None, Modifier, Tap }

    private static string FindPtxDirOrSkip()
    {
        foreach (var dir in new[] { Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx") })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        Skip.If(true, "PTX files not found");
        return "";
    }

    private static T WithWatchdog<T>(string what, Func<T> body)
    {
        var task = Task.Factory.StartNew(body, TaskCreationOptions.LongRunning);
        if (!task.Wait(Watchdog))
            Assert.Fail($"{what} did not finish within {Watchdog.TotalSeconds:F0} s (a graph launched around an attached modifier or tap can wedge the stream).");
        return task.GetAwaiter().GetResult();
    }

    /// <summary>Prefill, then <see cref="Steps"/> teacher-forced decode steps with the requested attachment installed after prefill.</summary>
    private static Run Decode(string ptx, bool stepGraph, Attach attach, int[]? fed)
    {
        string path = Environment.GetEnvironmentVariable("DOTLLM_SMOLLM_Q8_GGUF")
            ?? TestFixtureResolver.ResolveFile([], "QuantFactory", "SmolLM-135M-GGUF", ["SmolLM-135M.Q8_0.gguf"], null).Path ?? "";
        Skip.If(!File.Exists(path), $"SmolLM-135M Q8_0 GGUF not found at {path}");
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] prompt = GgufBpeTokenizerFactory.Load(gguf.Metadata).Encode(
            "The quick brown fox jumps over the lazy dog. The capital of France is Paris, the capital of Germany is Berlin, and the capital of Italy is");

        // Same configuration the E2 step-graph tests use: tq4 in-place KV is admitted by StepGraphEligible; the shallow graph's depth cap
        // (1) is below the prompt so the step graph, not the device-scalar graph, takes the decode step.
        using var v2 = StaticToggle.DenseV2(true, 1);
        using var cap = StaticToggle.GraphMaxDepth(1);
        using var sg = StaticToggle.StepGraph(stepGraph);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptx);
        model.UseGraphCapture = true;
        using var cache = model.CreateKvCache(prompt.Length + Steps + 4,
            new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, 4));
        int vocab = config.VocabSize;
        float[] last;
        using (var l = model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), 0, cache)) last = Row(l, vocab);

        var mod = new CountingModifier();
        int samples = 0;
        IDisposable? tap = null;
        if (attach == Attach.Modifier) model.ResidualModifier = mod;
        if (attach == Attach.Tap)
            tap = model.RegisterTap(new HiddenStateTapSubscription(null, [HookPoint.PostLayer]), _ => samples++);

        var rows = new List<float[]>();
        var tokens = new int[Steps];
        int sg0 = model.StepGraphLaunches, pos = prompt.Length;
        try
        {
            for (int s = 0; s < Steps; s++)
            {
                tokens[s] = fed?[s] ?? Argmax(last);
                using var l = model.Forward([tokens[s]], [pos++], 0, cache);
                last = Row(l, vocab);
                rows.Add(last);
            }
        }
        finally
        {
            tap?.Dispose();
            model.ResidualModifier = null;
        }
        return new Run(rows, tokens, model.StepGraphLaunches - sg0, attach == Attach.Modifier ? mod.Calls : samples, config.NumLayers);
    }

    /// <summary>Precondition shared by both tests: with nothing attached the step graph takes every decode step. Fails loudly, never skips.</summary>
    private Run Baseline(string ptx)
    {
        var run = Decode(ptx, stepGraph: true, Attach.None, null);
        _out.WriteLine($"baseline (nothing attached): stepGraphLaunches={run.StepGraphLaunches}");
        Assert.True(run.StepGraphLaunches == Steps,
            $"PRECONDITION: the step graph did not engage in the guard-test configuration ({run.StepGraphLaunches} launches over {Steps} steps); " +
            "the guard tests cannot discriminate, so this is a failure, not a skip.");
        return run;
    }

    /// <summary>A residual modifier keeps decode eager: it is called on every layer of every step and the step graph never replays.</summary>
    [SkippableFact]
    public void StepGraph_IsNotTaken_WhenAResidualModifierIsAttached()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = FindPtxDirOrSkip();
        WithWatchdog("modifier guard test", () =>
        {
            var baseline = Baseline(ptx);
            var guarded = Decode(ptx, stepGraph: true, Attach.Modifier, baseline.Tokens);
            _out.WriteLine($"modifier attached: stepGraphLaunches={guarded.StepGraphLaunches} modifierCalls={guarded.Calls}");
            Assert.Equal(0, guarded.StepGraphLaunches);
            Assert.Equal(Steps * guarded.NumLayers, guarded.Calls);   // AppliesTo is true everywhere, Apply is a no-op
            Assert.Equal(0, BitDiffs(baseline.Rows, guarded.Rows));    // a counting no-op modifier changes nothing
            return 0;
        });
    }

    /// <summary>A PostLayer tap keeps decode eager: the callback fires on every layer of every step, the step graph never replays, logits are unchanged.</summary>
    [SkippableFact]
    public void StepGraph_IsNotTaken_WhenAHiddenStateTapIsAttached()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = FindPtxDirOrSkip();
        WithWatchdog("tap guard test", () =>
        {
            var baseline = Baseline(ptx);
            var eager = Decode(ptx, stepGraph: false, Attach.None, baseline.Tokens);   // untapped eager reference
            var guarded = Decode(ptx, stepGraph: true, Attach.Tap, baseline.Tokens);
            _out.WriteLine($"tap attached: stepGraphLaunches={guarded.StepGraphLaunches} samples={guarded.Calls}");
            Assert.Equal(0, guarded.StepGraphLaunches);
            Assert.Equal(Steps * guarded.NumLayers, guarded.Calls);   // default rows = last token; one sample per layer per decode step
            Assert.Equal(0, BitDiffs(eager.Rows, guarded.Rows));      // a read-only tap on the eager path leaves logits bit-identical
            return 0;
        });
    }

    private static float[] Row(DotLLM.Core.Tensors.ITensor t, int vocab)
    {
        var r = new float[vocab];
        new ReadOnlySpan<float>((void*)t.DataPointer, vocab).CopyTo(r);
        return r;
    }

    private static int Argmax(float[] a)
    {
        int b = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[b]) b = i;
        return b;
    }

    private static int BitDiffs(List<float[]> a, List<float[]> b)
    {
        int n = 0;
        for (int s = 0; s < a.Count; s++)
            for (int i = 0; i < a[s].Length; i++)
                if (BitConverter.SingleToInt32Bits(a[s][i]) != BitConverter.SingleToInt32Bits(b[s][i])) n++;
        return n;
    }
}
