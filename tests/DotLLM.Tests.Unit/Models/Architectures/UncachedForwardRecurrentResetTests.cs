using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tests.Unit.Models.Lfm2Moe;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// #1786: an uncached <c>Forward</c> (no KV cache, no caller-threaded state) that starts at position 0 is a new sequence, so a
/// CPU recurrent host must not let the previous prompt's model-owned recurrent state (GDN/SSM matrix AND short-conv history)
/// leak into it. Each host is checked three ways, each killing a different wrong implementation:
/// <list type="number">
///   <item><description><b>Reset</b>: prompt A then prompt B through ONE model gives every row of B bit-identical to a fresh
///   model's B. Fails when the reset is missing (the pre-#1786 behaviour).</description></item>
///   <item><description><b>Continuation</b>: A at 0, then an uncached call at <c>positions[0] &gt; 0</c>, differs from the same
///   call on a fresh model. Fails when the reset fires on every uncached call instead of only at position 0.</description></item>
///   <item><description><b>Threaded state</b>: A then B through one caller-owned state container differs from fresh B. Fails when
///   the reset ignores the "no state passed" condition and zeroes a caller's container.</description></item>
/// </list>
/// Nemotron-H's tests live in <see cref="NemotronHTransformerModelRecurrentBatchTests"/> (its synthetic fixture is private there).
/// </summary>
public sealed class UncachedForwardRecurrentResetTests : IDisposable
{
    private static readonly int[] PromptA = [3, 1, 4, 1, 5, 2];
    private static readonly int[] PromptB = [2, 7, 1, 6, 3];
    private static readonly int[] Continuation = [5, 4, 6];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-1786-{Guid.NewGuid():N}");
    private readonly List<IDisposable> _owned = [];

    public UncachedForwardRecurrentResetTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        for (int i = _owned.Count - 1; i >= 0; i--) _owned[i].Dispose();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ── Qwen3HybridDense (qwen35 dense: GDN + conv1d + full attention) ──

    private Func<Qwen3HybridDenseTransformerModel> DenseFactory()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "dense.gguf"), withMtp: false);
        var gguf = GgufFile.Open(path);
        _owned.Add(gguf);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        return () => Own(Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));
    }

    [Fact]
    public void Qwen3HybridDense_UncachedPositionZero_ResetsDefaultState()
    {
        var make = DenseFactory();
        var used = make();
        AssertResets(used, make(), (m, ids, pos) => m.Forward(ids, pos, -1));
    }

    [Fact]
    public void Qwen3HybridDense_UncachedContinuation_KeepsDefaultState()
    {
        var make = DenseFactory();
        AssertContinues(make(), make(), (m, ids, pos) => m.Forward(ids, pos, -1));
    }

    [Fact]
    public void Qwen3HybridDense_ThreadedState_IsNotReset()
    {
        var make = DenseFactory();
        var model = make();
        var state = (IGdnState)Own(model.CreateSequenceState()!);
        AssertThreadedStateKept(model, make(),
            (m, ids, pos) => m.Forward(ids, pos, -1, kvCache: null, gdnState: state));
    }

    // ── Qwen3MoeHybrid (qwen35moe: GDN + conv1d + MoE) ──

    private Func<Qwen3MoeHybridTransformerModel> MoeFactory()
    {
        string path = SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "moe.gguf"), contextLength: 32);
        var gguf = GgufFile.Open(path);
        _owned.Add(gguf);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        return () => Own((Qwen3MoeHybridTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config));
    }

    [Fact]
    public void Qwen3MoeHybrid_UncachedPositionZero_ResetsDefaultState()
    {
        var make = MoeFactory();
        AssertResets(make(), make(), (m, ids, pos) => m.Forward(ids, pos, -1));
    }

    [Fact]
    public void Qwen3MoeHybrid_UncachedContinuation_KeepsDefaultState()
    {
        var make = MoeFactory();
        AssertContinues(make(), make(), (m, ids, pos) => m.Forward(ids, pos, -1));
    }

    [Fact]
    public void Qwen3MoeHybrid_ThreadedState_IsNotReset()
    {
        var make = MoeFactory();
        var model = make();
        var state = (IGdnState)Own(model.CreateSequenceState()!);
        AssertThreadedStateKept(model, make(),
            (m, ids, pos) => m.Forward(ids, pos, -1, kvCache: null, gdnState: state));
    }

    // ── LFM2 (short-conv + attention; the state is the conv history only) ──

    private Lfm2MoeTransformerModel Lfm2() { var f = new Lfm2TestFixture(); _owned.Add(f); return f.Model; }

    [Fact]
    public void Lfm2_UncachedPositionZero_ResetsDefaultState()
        => AssertResets(Lfm2(), Lfm2(), (m, ids, pos) => m.Forward(ids, pos, -1));

    [Fact]
    public void Lfm2_UncachedContinuation_KeepsDefaultState()
        => AssertContinues(Lfm2(), Lfm2(), (m, ids, pos) => m.Forward(ids, pos, -1));

    [Fact]
    public void Lfm2_ThreadedState_IsNotReset()
    {
        var model = Lfm2();
        var state = (ISsmState)Own(model.CreateSequenceState()!);
        AssertThreadedStateKept(model, Lfm2(), (m, ids, pos) => m.Forward(ids, pos, -1, kvCache: null, ssmState: state));
    }

    // ── LFM3 (no synthetic fixture exists: the smallest real GGUF, skipped when absent) ──

    [SkippableFact]
    [Trait("Category", "RealModel")]
    public void Lfm3_UncachedPositionZero_ResetsDefaultState_AndKeepsContinuation()
    {
        string? path = TestFixtureResolver.ResolveFile("DOTLLM_LFM3_GGUF", "LiquidAI", "LFM3-1.7B-GGUF", "LFM3-1.7B-Q8_0.gguf").Path;
        Skip.If(path is null, "LFM3 GGUF not found: set DOTLLM_LFM3_GGUF or fetch LiquidAI/LFM3-1.7B-GGUF.");
        var gguf = GgufFile.Open(path!);
        _owned.Add(gguf);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Lfm3TransformerModel Make() => Own(Lfm3TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));

        int[] a = [597, 5205, 302, 3980, 355, 4741];   // "The capital of France is Paris"
        int[] b = [5205, 302, 5881, 355];                // "capital of Australia is"
        AssertResets(Make(), Make(), (m, ids, pos) => m.Forward(ids, pos, -1), a, b);
        AssertContinues(Make(), Make(), (m, ids, pos) => m.Forward(ids, pos, -1), a, b);
    }

    // ── Shared checks ──

    private static void AssertResets<TModel>(TModel used, TModel fresh, Func<TModel, int[], int[], ITensor> forward,
                                             int[]? a = null, int[]? b = null)
    {
        a ??= PromptA; b ??= PromptB;
        Run(used, forward, a, From(0, a.Length));
        float[] reused = Run(used, forward, b, From(0, b.Length));
        float[] expected = Run(fresh, forward, b, From(0, b.Length));
        Assert.True(expected.AsSpan().SequenceEqual(reused),
            $"prompt B after prompt A through one model differs from a fresh model's B (max |diff| {MaxAbsDiff(expected, reused):G4}): " +
            "the uncached position-0 forward kept the previous prompt's recurrent state (#1786).");
    }

    private static void AssertContinues<TModel>(TModel used, TModel fresh, Func<TModel, int[], int[], ITensor> forward,
                                                int[]? a = null, int[]? c = null)
    {
        a ??= PromptA; c ??= Continuation;
        Run(used, forward, a, From(0, a.Length));
        float[] continued = Run(used, forward, c, From(a.Length, c.Length));
        float[] coldStart = Run(fresh, forward, c, From(a.Length, c.Length));
        Assert.True(MaxAbsDiff(continued, coldStart) > 1e-4f,
            "an uncached continuation (positions[0] > 0) matched a cold model: the recurrent state was reset on a call that " +
            "must continue it.");
    }

    private static void AssertThreadedStateKept<TModel>(TModel model, TModel fresh, Func<TModel, int[], int[], ITensor> threaded)
        where TModel : IModel
    {
        Run(model, threaded, PromptA, From(0, PromptA.Length));
        float[] carried = Run(model, threaded, PromptB, From(0, PromptB.Length));
        float[] freshB = Run(fresh, (m, ids, pos) => m.Forward(ids, pos, -1), PromptB, From(0, PromptB.Length));
        Assert.True(MaxAbsDiff(carried, freshB) > 1e-4f,
            "a caller-threaded state container was zeroed by a position-0 forward: the auto-reset must only touch the " +
            "model-owned default state.");
    }

    private static unsafe float[] Run<TModel>(TModel model, Func<TModel, int[], int[], ITensor> forward, int[] ids, int[] pos)
    {
        using ITensor logits = forward(model, ids, pos);
        Assert.Equal(ids.Length, logits.Shape[0]);   // every row: the conv window is short, so the last row alone is weak
        return new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount).ToArray();
    }

    private static int[] From(int start, int count) => Enumerable.Range(start, count).ToArray();

    private static float MaxAbsDiff(float[] x, float[] y)
    {
        Assert.Equal(x.Length, y.Length);
        float max = 0;
        for (int i = 0; i < x.Length; i++) max = MathF.Max(max, MathF.Abs(x[i] - y[i]));
        return max;
    }

    private T Own<T>(T item) where T : IDisposable { _owned.Add(item); return item; }
}
