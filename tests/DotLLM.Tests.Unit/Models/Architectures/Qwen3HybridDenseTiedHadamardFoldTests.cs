using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1276: <c>prism.hadamard</c> version 2 (<c>tied_output</c>) on the CPU forward pass.
/// </summary>
/// <remarks>
/// <para>
/// Unlike <see cref="Qwen3HybridDenseSyntheticHadamardFoldTests"/>, the fixture here is <b>really
/// folded</b>: every declared weight's rows (and the tied <c>token_embd</c> table's rows) are
/// forward-transformed offline, exactly as PrismML's converter does. A correct runtime then computes
/// the <i>same function</i> as the unfolded model, which makes this an exact oracle rather than a
/// "the transform moves the output" smoke check.
/// </para>
/// <para>
/// In the tied layout the one table is used both ways: rows fetched by lookup must be un-rotated
/// (inverse), and the final hidden state must be forward-rotated before the tied lm_head. Dropping or
/// inverting either site breaks the equivalence — <see cref="TiedFold_WithoutHeadRotation_IsWrong"/>
/// pins that the head rotation is not a no-op on this fixture.
/// </para>
/// </remarks>
public sealed class Qwen3HybridDenseTiedHadamardFoldTests : IDisposable
{
    private static readonly int[] Prompt = [1, 3, 5, 7, 9];

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public Qwen3HybridDenseTiedHadamardFoldTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35-tied-fold-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void TiedFold_ReproducesTheUnfoldedModel()
    {
        var (plainPath, foldedPath, fold) = WriteFolded();

        float[] reference = Forward(plainPath, fold: null);
        float[] folded = Forward(foldedPath, fold);

        float maxDiff = MaxAbsDiff(reference, folded);
        _out.WriteLine($"tied fold vs unfolded: max |diff| = {maxDiff:E3}");
        Assert.True(maxDiff < 1e-4f, $"tied fold diverges from the unfolded model: max |diff| {maxDiff:E3}");
    }

    /// <summary>
    /// The discriminating mirror. Rotating only the tied table (no fold declared) must move the
    /// logits well past the equivalence tolerance — so a runtime that skipped the head rotation, or
    /// un-rotated the lookup only, could not pass <see cref="TiedFold_ReproducesTheUnfoldedModel"/>.
    /// And reading a v2 declaration as v1 (no head site) must be refused at load.
    /// </summary>
    [Fact]
    public void TiedFold_WithoutHeadRotation_IsWrong()
    {
        var (plainPath, foldedPath, fold) = WriteFolded();

        string tableOnlyPath = Path.Combine(_scratch, "table-only-rotated.gguf");
        RewriteRows(plainPath, tableOnlyPath, fold, ["token_embd.weight"]);

        float[] reference = Forward(plainPath, fold: null);
        float[] wrong = Forward(tableOnlyPath, fold: null);

        float maxDiff = MaxAbsDiff(reference, wrong);
        _out.WriteLine($"unrotated tied table: max |diff| = {maxDiff:E3}");
        Assert.True(maxDiff > 2e-2f, $"the tied-table rotation is degenerate on this fixture: max |diff| {maxDiff:E3}");

        // The full folded file loads only with the head site registered.
        using var gguf = GgufFile.Open(foldedPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var v1Style = fold with
        {
            FoldedWeights = fold.FoldedWeights.Where(n => n != "output.weight").ToFrozenSetOrdinal(),
            TiedOutput = false,
        };
        Assert.Throws<NotSupportedException>(() =>
            Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config with { HadamardFold = v1Style },
                ThreadingConfig.SingleThreaded));
    }

    [Fact]
    public void TiedOutputDeclaration_WithSeparateOutputWeight_IsRefused()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "untied.gguf"), withMtp: false);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var fold = SyntheticHadamardFold.For(config, gdnVGrouped: false, tiedOutput: true);

        var ex = Assert.Throws<NotSupportedException>(() =>
            Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config with { HadamardFold = fold },
                ThreadingConfig.SingleThreaded));
        Assert.Contains("tied_output", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void V1Declaration_WithTiedRotatedEmbedding_IsRefused()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "tied.gguf"), withMtp: false,
            tiedOutput: true);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var fold = SyntheticHadamardFold.For(config, gdnVGrouped: false, tiedOutput: false);

        Assert.Throws<NotSupportedException>(() =>
            Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config with { HadamardFold = fold },
                ThreadingConfig.SingleThreaded));
    }

    /// <summary>Writes the tied fixture and a really-folded copy of it.</summary>
    private (string Plain, string Folded, HadamardFoldConfig Fold) WriteFolded()
    {
        string plainPath = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "tied.gguf"), withMtp: false,
            tiedOutput: true);

        HadamardFoldConfig fold;
        using (var gguf = GgufFile.Open(plainPath))
        {
            // gdn_v_grouped off: this fixture folds ssm_out in tiled order, so no permute is needed.
            fold = SyntheticHadamardFold.For(GgufModelConfigExtractor.Extract(gguf.Metadata),
                gdnVGrouped: false, tiedOutput: true);
        }

        // output.weight is the LM-head SITE; the tensor it names is token_embd (folded below).
        var tables = fold.FoldedWeights.Where(n => n != "output.weight").Append("token_embd.weight").ToArray();
        string foldedPath = Path.Combine(_scratch, "tied-folded.gguf");
        RewriteRows(plainPath, foldedPath, fold, tables);
        return (plainPath, foldedPath, fold);
    }

    /// <summary>
    /// Copies <paramref name="src"/> to <paramref name="dst"/>, forward-transforming every row of the
    /// named F32 tensors (rows run along <c>ne0</c>, the input dimension). Folding a weight is that
    /// row transform: <c>(H·S·w)·(H·S·x) = w·x</c> because <c>H·S</c> is orthogonal.
    /// </summary>
    private static unsafe void RewriteRows(string src, string dst, HadamardFoldConfig fold,
        IReadOnlyList<string> names)
    {
        byte[] bytes = File.ReadAllBytes(src);
        using (var gguf = GgufFile.Open(src))
        {
            foreach (string name in names)
            {
                var desc = gguf.TensorsByName[name];
                Assert.Equal(QuantizationType.F32, desc.QuantizationType);
                int width = desc.Shape[0];
                long rows = desc.Shape.ElementCount / width;
                var signs = fold.SignsFor(width) ?? [];
                var tmp = new float[width];

                fixed (byte* basePtr = bytes)
                {
                    float* data = (float*)(basePtr + gguf.DataSectionOffset + (long)desc.DataOffset);
                    for (long r = 0; r < rows; r++)
                    {
                        var row = new Span<float>(data + r * width, width);
                        Hadamard.ForwardRow(row, signs, tmp, fold.BlockSize);
                        tmp.CopyTo(row);
                    }
                }
            }
        }

        File.WriteAllBytes(dst, bytes);
    }

    private static unsafe float[] Forward(string path, HadamardFoldConfig? fold)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata) with { HadamardFold = fold };

        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        int[] positions = Enumerable.Range(0, Prompt.Length).ToArray();
        using ITensor logits = model.Forward(Prompt, positions, deviceId: -1);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, Prompt.Length * config.VocabSize).ToArray();
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(float.IsFinite(b[i]));
            max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        }
        return max;
    }
}
