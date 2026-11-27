using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// The F32 <c>patch_embd</c> / <c>position_embd</c> tensors of the mmproj are not an automatic exception to the no-FP32 rule (CLAUDE.md "Precision and
/// Expansion Rules"; user ruling 2026-11-27, #2035): the encoder converts them once at load to F16 or BF16 (<see cref="VisionEmbedPrecision"/>), F32 stays
/// as the A/B arm and the default until the user rules. This class pins the conversion and measures what each arm costs.
/// The sharp measurement is the F32-weights control (the Q8_0 matrices dequantised to an F32 GGUF, which removes the reference's own Q8_0 activation
/// quantisation noise: ours vs llama.cpp relRMS 2.1e-4), so a precision change shows up un-masked. Control files: <c>vision-f32-control/</c> next to the oracles.
/// </summary>
public sealed class Inkling2VisionEmbedPrecisionTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2VisionEmbedPrecisionTests(ITestOutputHelper output) => _output = output;

    // ───────────── the conversions (no model needed) ─────────────

    /// <summary>Round to nearest even, like hardware and like llama.cpp's <c>ggml_fp32_to_bf16</c>. Kills: truncation (round toward zero) and round-half-up.</summary>
    [Theory]
    [InlineData(1.0f, 0x3F80)]
    [InlineData(-2.5f, 0xC020)]
    [InlineData(0.0f, 0x0000)]
    [InlineData(1.00390625f, 0x3F80)]       // 0x3F808000: a tie, the even neighbour 0x3F80 wins (round-half-up would give 0x3F81)
    [InlineData(1.01171875f, 0x3F82)]       // 0x3F818000: a tie, the even neighbour 0x3F82 wins (truncation would give 0x3F81)
    [InlineData(1.0039063f + 1e-7f, 0x3F81)]    // just above the tie: rounds up
    [InlineData(3.4028235e38f, 0x7F80)]     // float.MaxValue rounds up to infinity, as IEEE round-to-nearest does
    public void FloatToBf16_RoundsToNearestEven(float value, int bits)
        => Assert.Equal((ushort)bits, Inkling2VisionEncoder.FloatToBf16(value));

    [Fact]
    public void FloatToBf16_KeepsNanAndInfinity()
    {
        Assert.Equal(0x7F80, Inkling2VisionEncoder.FloatToBf16(float.PositiveInfinity));
        Assert.Equal(0xFF80, Inkling2VisionEncoder.FloatToBf16(float.NegativeInfinity));
        ushort nan = Inkling2VisionEncoder.FloatToBf16(float.NaN);
        Assert.True((nan & 0x7F80) == 0x7F80 && (nan & 0x007F) != 0, "NaN stays a NaN (not rounded into an infinity)");
    }

    [Fact]
    public void EmbedPrecisionEnvironment_ParsesTheThreeArms_AndRejectsTypos()
    {
        Assert.Equal(VisionEmbedPrecision.F32, Inkling2VisionEncoder.EmbedPrecisionFromEnvironment(_ => null));
        Assert.Equal(VisionEmbedPrecision.F16, Inkling2VisionEncoder.EmbedPrecisionFromEnvironment(_ => "F16"));
        Assert.Equal(VisionEmbedPrecision.BF16, Inkling2VisionEncoder.EmbedPrecisionFromEnvironment(_ => " bf16 "));
        Assert.Throws<ArgumentException>(() => Inkling2VisionEncoder.EmbedPrecisionFromEnvironment(_ => "fp8"));
    }

    // ───────────── the arms on the real mmproj ─────────────

    private static string ControlDir => Path.Combine(Inkling2VisionFixtures.OracleDir, "..", "vision-f32-control");

    private static string? ControlMmproj()
    {
        string p = Path.GetFullPath(Path.Combine(ControlDir, "mmproj-f32.gguf"));
        return File.Exists(p) ? p : null;
    }

    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return f;
    }

    private static readonly VisionEmbedPrecision[] Arms = [VisionEmbedPrecision.F32, VisionEmbedPrecision.F16, VisionEmbedPrecision.BF16];

    private static float[] Embeddings(ImageEmbeddings e, int dim)
    {
        var all = new float[e.Tokens * dim];
        var row = new float[dim];
        for (int r = 0; r < e.Tokens; r++) { e.ReadRow(r, row); row.CopyTo(all, r * dim); }
        return all;
    }

    private static double RelRms(float[] got, float[] want)
    {
        double num = 0, den = 0;
        for (int i = 0; i < want.Length; i++) { double d = got[i] - want[i]; num += d * d; den += (double)want[i] * want[i]; }
        return Math.Sqrt(num / den);
    }

    /// <summary>
    /// The measurement for the user's decision (docs/INKLING2.md "Embedding precision"): embedding relRMS of each arm against the F32 llama.cpp oracle, on the
    /// F32-weights control (sharp) and on the shipping Q8_0 mmproj (the 2% noise floor of the reference's activation quantisation), plus the arm against our own F32 arm.
    /// Asserts only what must hold for the numbers to mean anything: the arms really differ (a no-op conversion would give identical output), F32 on the control
    /// is at the 2.1e-4 floor, and F16 does not cost more than 10x the floor (a gross-error bound, not the ruling).
    /// </summary>
    [SkippableFact]
    public void Report_EmbedPrecisionArms_Embeddings()
    {
        string? control = ControlMmproj(), shipped = Inkling2VisionFixtures.FindMmproj();
        Skip.If(control is null || shipped is null, "F32 control mmproj (vision-f32-control/) or the shipped mmproj not available");
        const int dim = 2304;
        var lines = new List<string> { "arm | image | control relRMS vs llama.cpp F32 | control vs our F32 arm | shipped Q8_0 relRMS vs llama.cpp | shipped vs our F32 arm" };
        var images = new[] { "rainbow80", "rainbow190x110", "rainbow640" };
        var controlResults = new Dictionary<(VisionEmbedPrecision, string), float[]>();
        var shippedResults = new Dictionary<(VisionEmbedPrecision, string), float[]>();
        foreach (var arm in Arms)
        {
            foreach (var (path, store) in new[] { (control!, controlResults), (shipped!, shippedResults) })
            {
                using var enc = Inkling2VisionEncoder.Load(path, new ThreadingConfig(8), embedPrecision: arm);
                Assert.Equal(arm, enc.EmbedPrecision);
                foreach (string name in images)
                {
                    if (path == control && name == "rainbow640") continue;   // the control has no 640 oracle (the encode is ~25 s); the shipped mmproj covers it
                    var image = ImageDecoder.Load(Path.Combine(Inkling2VisionFixtures.OracleDir, name, "image.ppm"));
                    using var e = enc.Encode(image);
                    store[(arm, name)] = Embeddings(e, dim);
                }
            }
        }
        foreach (var arm in Arms)
            foreach (string name in images)
            {
                string c = "n/a", cSelf = "n/a";
                if (controlResults.TryGetValue((arm, name), out var cm))
                {
                    string f = Path.Combine(ControlDir, $"llamacpp_f32_{name}.f32");
                    if (File.Exists(f)) c = RelRms(cm, ReadF32(f)).ToString("E2");
                    cSelf = RelRms(cm, controlResults[(VisionEmbedPrecision.F32, name)]).ToString("E2");
                }
                var sm = shippedResults[(arm, name)];
                string s = RelRms(sm, ReadF32(Path.Combine(Inkling2VisionFixtures.OracleDir, name, "embd.f32"))).ToString("E2");
                string sSelf = RelRms(sm, shippedResults[(VisionEmbedPrecision.F32, name)]).ToString("E2");
                lines.Add($"{arm,-4} | {name,-14} | {c} | {cSelf} | {s} | {sSelf}");
            }
        foreach (string l in lines) _output.WriteLine(l);

        // The arms are real: a conversion that did nothing would make F16 and BF16 identical to F32.
        foreach (string name in new[] { "rainbow80", "rainbow190x110" })
        {
            Assert.NotEqual(controlResults[(VisionEmbedPrecision.F32, name)], controlResults[(VisionEmbedPrecision.F16, name)]);
            Assert.NotEqual(controlResults[(VisionEmbedPrecision.F16, name)], controlResults[(VisionEmbedPrecision.BF16, name)]);
            double floor = RelRms(controlResults[(VisionEmbedPrecision.F32, name)], ReadF32(Path.Combine(ControlDir, $"llamacpp_f32_{name}.f32")));
            Assert.True(floor < 5e-4, $"{name}: F32 arm on the F32-weights control is {floor:E2} from llama.cpp (the reviewer measured 2.1e-4)");
            double f16 = RelRms(controlResults[(VisionEmbedPrecision.F16, name)], ReadF32(Path.Combine(ControlDir, $"llamacpp_f32_{name}.f32")));
            Assert.True(f16 < 10 * 2.1e-4 + 1e-3, $"{name}: the F16 arm moved the embeddings by {f16:E2}");
        }
    }
}
