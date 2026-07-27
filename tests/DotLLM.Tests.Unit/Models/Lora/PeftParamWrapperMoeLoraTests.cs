using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Architectures;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lora;

/// <summary>
/// Tests for the peft "ParamWrapper" fused-MoE LoRA serialization format
/// (produced by <c>LoraConfig(target_parameters=[...])</c>): LoRA over raw
/// fused expert <c>nn.Parameter</c> tensors is serialized as STACKED factor
/// pairs with the parameter name ABSENT from the key —
/// <c>...layers.{i}.experts(.base_layer)?.lora_{A,B}.weight</c> — where
/// A is <c>[r*E, in]</c> (expert-MAJOR) and B is <c>[out, r*E]</c>
/// (expert-MINOR: column c → rank c/E, expert c%E). Which nesting level is
/// which parameter is inferred from shapes (in &gt; out ⇒ gate_up_proj,
/// in &lt; out ⇒ down_proj), cross-checked against adapter_config.json's
/// <c>target_parameters</c>.
/// </summary>
/// <remarks>
/// Fixture dims are chosen tiny but DISCRIMINATING: with E=2, r=2 the
/// expert-minor B de-interleave (expert 0 takes stacked columns {0, 2})
/// differs from a naive expert-major slicing (columns {0, 1}), so a wrong
/// implementation produces different stored factors and the exact-value
/// asserts below fail.
/// </remarks>
public sealed unsafe class PeftParamWrapperMoeLoraTests : IDisposable
{
    private const int Rank = 2;
    private const int Experts = 2;          // E
    private const int Hidden = 8;           // in_features of gate_up, out_features of down
    private const int Inter = 2;            // per-expert MoE intermediate size
    private const int GateUpOut = 2 * Inter; // fused gate‖up output dim (4) — < Hidden (8)
    private const int RTimesE = Rank * Experts;

    // Deterministic, all-distinct ramp bases so every stacked element is unique.
    private const float AGuBase = 100f;  // stacked gate_up A  [r*E, Hidden]
    private const float BGuBase = 200f;  // stacked gate_up B  [GateUpOut, r*E]
    private const float ADnBase = 300f;  // stacked down A     [r*E, Inter]
    private const float BDnBase = 400f;  // stacked down B     [Hidden, r*E]

    private readonly string _scratch;

    public PeftParamWrapperMoeLoraTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-peft-pw-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private static ModelConfig BuildMoeBaseConfig() => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 32,
        HiddenSize = Hidden,
        IntermediateSize = 16,
        NumLayers = 1,
        NumAttentionHeads = 2,
        NumKvHeads = 2,
        HeadDim = 4,
        MaxSequenceLength = 32,
        RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: 4, Type: RoPEType.Norm),
        Moe = new MoeConfig { NumExperts = Experts, NumExpertsPerTok = 1, MoeIntermediateSize = Inter },
    };

    private static float[] Ramp(float start, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = start + i;
        return v;
    }

    /// <summary>
    /// Writes a ParamWrapper-format PEFT adapter directory for decoder layer 0.
    /// The two nesting levels carry the two stacked pairs; which parameter sits
    /// at which level is controlled by <paramref name="gateUpInner"/> so tests
    /// can prove the loader disambiguates by SHAPE, not by nesting position.
    /// </summary>
    private string BuildParamWrapperFixture(bool gateUpInner, bool includeTargetParameters,
                                            bool includeLinearQProj = false)
    {
        string dir = Path.Combine(_scratch, $"pw-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        WriteConfig(dir, includeTargetParameters
            ? new[] { "decoder.layers.0.experts.gate_up_proj", "decoder.layers.0.experts.down_proj" }
            : null);

        float[] aGu = Ramp(AGuBase, RTimesE * Hidden);
        float[] bGu = Ramp(BGuBase, GateUpOut * RTimesE);
        float[] aDn = Ramp(ADnBase, RTimesE * Inter);
        float[] bDn = Ramp(BDnBase, Hidden * RTimesE);

        const string prefix = "base_model.model.model.decoder.layers.0.experts";
        string gateUpKey = gateUpInner ? $"{prefix}.base_layer" : prefix;
        string downKey = gateUpInner ? prefix : $"{prefix}.base_layer";

        var b = new SafetensorsFixtureBuilder()
            .AddFloat32($"{gateUpKey}.lora_A.weight", [RTimesE, Hidden], aGu)
            .AddFloat32($"{gateUpKey}.lora_B.weight", [GateUpOut, RTimesE], bGu)
            .AddFloat32($"{downKey}.lora_A.weight", [RTimesE, Inter], aDn)
            .AddFloat32($"{downKey}.lora_B.weight", [Hidden, RTimesE], bDn);

        if (includeLinearQProj)
        {
            // Ordinary nn.Linear LoRA in the SAME file — must keep flowing
            // through the untouched ProjectionPathRegex path.
            b.AddFloat32("base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_A.weight",
                [Rank, Hidden], Ramp(500f, Rank * Hidden));
            b.AddFloat32("base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_B.weight",
                [Hidden, Rank], Ramp(600f, Hidden * Rank));
        }

        b.WriteTo(Path.Combine(dir, "adapter_model.safetensors"));
        return dir;
    }

    private static void WriteConfig(string dir, string[]? targetParameters)
    {
        var cfgObj = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["r"] = Rank,
            ["lora_alpha"] = 4.0,
            // Real ParamWrapper adapters carry a regex STRING here.
            ["target_modules"] = @".*(decoder|language_model)\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
            ["task_type"] = null,
            ["use_rslora"] = false,
            ["use_dora"] = false,
        };
        if (targetParameters is not null)
            cfgObj["target_parameters"] = targetParameters;
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), JsonSerializer.Serialize(cfgObj));
    }

    private static ReadOnlySpan<float> AsSpan(nint handle, int elements)
        => new((void*)handle, elements);

    // ── Expected per-expert factor values (derive from the ramp bases) ──
    //
    // Stacked A [r*E, in] is expert-MAJOR: A_e = rows [e*r, (e+1)*r).
    //   dotLLM B factor element (k, i)   = base + (e*Rank + k)*in + i
    // Stacked B [out, r*E] is expert-MINOR: B_e[:, k] = B[:, k*E + e].
    //   dotLLM A factor element (o, k)   = base + o*(r*E) + k*Experts + e

    private static float ExpectedBFactor(float stackedABase, int inDim, int e, int k, int i)
        => stackedABase + (e * Rank + k) * inDim + i;

    private static float ExpectedAFactor(float stackedBBase, int stackedRow, int e, int k)
        => stackedBBase + stackedRow * RTimesE + k * Experts + e;

    [Fact]
    public void LoadFromDirectory_ParamWrapper_FactorizesPerExpert_ExpertMinorB()
    {
        var cfg = BuildMoeBaseConfig();
        string dir = BuildParamWrapperFixture(gateUpInner: true, includeTargetParameters: true,
            includeLinearQProj: true);

        using var adapter = PeftAdapterLoader.LoadFromDirectory("pw", dir, cfg);

        for (int e = 0; e < Experts; e++)
        {
            var gate = adapter.GetLayerWeights(0, $"mlp.experts.{e}.gate_proj", LoraRegion.Decoder);
            var up = adapter.GetLayerWeights(0, $"mlp.experts.{e}.up_proj", LoraRegion.Decoder);
            var down = adapter.GetLayerWeights(0, $"mlp.experts.{e}.down_proj", LoraRegion.Decoder);
            Assert.NotNull(gate);
            Assert.NotNull(up);
            Assert.NotNull(down);

            Assert.Equal(Hidden, gate!.Value.InputDim);
            Assert.Equal(Inter, gate.Value.OutputDim);
            Assert.Equal(Hidden, up!.Value.InputDim);
            Assert.Equal(Inter, up.Value.OutputDim);
            Assert.Equal(Inter, down!.Value.InputDim);
            Assert.Equal(Hidden, down.Value.OutputDim);
            Assert.Equal(LoraWeightDType.F32, gate.Value.WeightDType);

            // B factor ([Rank, in]) = expert-major slice of stacked A; shared by gate & up.
            var gateB = AsSpan(gate.Value.BHandle, Rank * Hidden);
            var upB = AsSpan(up.Value.BHandle, Rank * Hidden);
            for (int k = 0; k < Rank; k++)
            {
                for (int i = 0; i < Hidden; i++)
                {
                    float want = ExpectedBFactor(AGuBase, Hidden, e, k, i);
                    Assert.Equal(want, gateB[k * Hidden + i]);
                    Assert.Equal(want, upB[k * Hidden + i]);
                }
            }

            // A factor ([out, Rank]) = expert-MINOR de-interleave of stacked B.
            // gate takes fused rows [0, Inter); up takes rows [Inter, 2*Inter).
            var gateA = AsSpan(gate.Value.AHandle, Inter * Rank);
            var upA = AsSpan(up.Value.AHandle, Inter * Rank);
            for (int o = 0; o < Inter; o++)
            {
                for (int k = 0; k < Rank; k++)
                {
                    Assert.Equal(ExpectedAFactor(BGuBase, o, e, k), gateA[o * Rank + k]);
                    Assert.Equal(ExpectedAFactor(BGuBase, Inter + o, e, k), upA[o * Rank + k]);
                }
            }

            // down_proj: unfused — one entry per expert.
            var downB = AsSpan(down.Value.BHandle, Rank * Inter);
            var downA = AsSpan(down.Value.AHandle, Hidden * Rank);
            for (int k = 0; k < Rank; k++)
                for (int i = 0; i < Inter; i++)
                    Assert.Equal(ExpectedBFactor(ADnBase, Inter, e, k, i), downB[k * Inter + i]);
            for (int o = 0; o < Hidden; o++)
                for (int k = 0; k < Rank; k++)
                    Assert.Equal(ExpectedAFactor(BDnBase, o, e, k), downA[o * Rank + k]);
        }

        // The ordinary nn.Linear q_proj entry in the same file still loads
        // through the existing projection path, region-tagged as before.
        Assert.NotNull(adapter.GetLayerWeights(0, "q_proj", LoraRegion.Decoder));

        Assert.True(adapter.IsCompatible(cfg));
    }

    [Fact]
    public void LoadFromDirectory_ParamWrapper_DisambiguatesByShape_NotNestingLevel()
    {
        // Swap the nesting: down_proj at the INNER (.base_layer) level and
        // gate_up at the OUTER level. Shape inference (in>out ⇒ gate_up,
        // in<out ⇒ down) must still map each stacked pair to the right
        // parameter. No target_parameters in the config — shapes alone decide.
        var cfg = BuildMoeBaseConfig();
        string dir = BuildParamWrapperFixture(gateUpInner: false, includeTargetParameters: false);

        using var adapter = PeftAdapterLoader.LoadFromDirectory("pw-swap", dir, cfg);

        var gate = adapter.GetLayerWeights(0, "mlp.experts.0.gate_proj", LoraRegion.Decoder);
        var down = adapter.GetLayerWeights(0, "mlp.experts.1.down_proj", LoraRegion.Decoder);
        Assert.NotNull(gate);
        Assert.NotNull(down);
        Assert.Equal(Hidden, gate!.Value.InputDim);
        Assert.Equal(Inter, gate.Value.OutputDim);
        Assert.Equal(Inter, down!.Value.InputDim);
        Assert.Equal(Hidden, down.Value.OutputDim);

        // Values must come from the gate_up ramp (AGuBase) regardless of the
        // nesting level the pair was serialized at.
        var gateB = AsSpan(gate.Value.BHandle, Rank * Hidden);
        Assert.Equal(ExpectedBFactor(AGuBase, Hidden, e: 0, k: 0, i: 0), gateB[0]);
        var downA = AsSpan(down.Value.AHandle, Hidden * Rank);
        Assert.Equal(ExpectedAFactor(BDnBase, stackedRow: 0, e: 1, k: 0), downA[0]);

        Assert.True(adapter.IsCompatible(cfg));
    }

    [Fact]
    public void LoadFromDirectory_ParamWrapper_TargetParametersMismatchThrows()
    {
        // A single gate_up-shaped stacked pair (in=Hidden > out=GateUpOut), but
        // adapter_config.json's target_parameters only declares down_proj —
        // the shape-inferred name must be cross-checked and rejected.
        string dir = Path.Combine(_scratch, $"pw-mismatch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        WriteConfig(dir, new[] { "decoder.layers.0.experts.down_proj" });

        new SafetensorsFixtureBuilder()
            .AddFloat32("base_model.model.model.decoder.layers.0.experts.lora_A.weight",
                [RTimesE, Hidden], Ramp(AGuBase, RTimesE * Hidden))
            .AddFloat32("base_model.model.model.decoder.layers.0.experts.lora_B.weight",
                [GateUpOut, RTimesE], Ramp(BGuBase, GateUpOut * RTimesE))
            .WriteTo(Path.Combine(dir, "adapter_model.safetensors"));

        // IDISP005 false positive: LoadFromDirectory already returns the disposable
        // LoraAdapter type; this call is expected to throw before any adapter
        // instance is constructed, so there is nothing to dispose here.
        Assert.Throws<InvalidDataException>(() =>
            PeftAdapterLoader.LoadFromDirectory("pw-mismatch", dir, BuildMoeBaseConfig()));
    }

    [Fact]
    public void LoadFromDirectory_ParamWrapper_F16SourceUpcastsToF32()
    {
        // ParamWrapper factorization requires an element-wise de-interleave, so
        // F16/BF16 sources are decoded to F32 regardless of preserveSourceDType.
        string dir = Path.Combine(_scratch, $"pw-f16-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        WriteConfig(dir, new[] { "decoder.layers.0.experts.gate_up_proj" });

        float[] aGu = Ramp(AGuBase, RTimesE * Hidden);
        float[] bGu = Ramp(BGuBase, GateUpOut * RTimesE);
        new SafetensorsFixtureBuilder()
            .AddRaw("base_model.model.model.decoder.layers.0.experts.lora_A.weight",
                "F16", [RTimesE, Hidden], ToF16Bytes(aGu))
            .AddRaw("base_model.model.model.decoder.layers.0.experts.lora_B.weight",
                "F16", [GateUpOut, RTimesE], ToF16Bytes(bGu))
            .WriteTo(Path.Combine(dir, "adapter_model.safetensors"));

        using var adapter = PeftAdapterLoader.LoadFromDirectory("pw-f16", dir,
            baseConfig: BuildMoeBaseConfig(), preserveSourceDType: true);

        var gate = adapter.GetLayerWeights(0, "mlp.experts.1.gate_proj", LoraRegion.Decoder);
        Assert.NotNull(gate);
        Assert.Equal(LoraWeightDType.F32, gate!.Value.WeightDType);
        var gateB = AsSpan(gate.Value.BHandle, Rank * Hidden);
        // Ramp values here are exactly representable in half precision.
        Assert.Equal(ExpectedBFactor(AGuBase, Hidden, e: 1, k: 0, i: 3), gateB[3]);
    }

    private static byte[] ToF16Bytes(ReadOnlySpan<float> values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)values[i]);
            bytes[i * 2] = (byte)bits;
            bytes[i * 2 + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }

    /// <summary>
    /// Env-gated end-to-end validation against the REAL DiffusionGemma fused
    /// adapter: loads the raw ParamWrapper directory and the known-good
    /// converted per-expert directory and asserts every per-expert factor
    /// buffer is numerically identical (both are pure re-slicings of the same
    /// trained F32 tensors, so equality is exact). Needs no GGUF/model.
    /// Set <c>DOTLLM_DG_MOE_LORA_ROOT</c> to the directory containing both
    /// <c>diffusiongemma_csharp_moe_lora</c> (raw) and
    /// <c>diffusiongemma_csharp_moe_lora_dotllm</c> (converted).
    /// </summary>
    [SkippableFact]
    public void LoadFromDirectory_RealFusedAdapter_MatchesConvertedReference()
    {
        string? root = Environment.GetEnvironmentVariable("DOTLLM_DG_MOE_LORA_ROOT");
        Skip.If(string.IsNullOrEmpty(root), "DOTLLM_DG_MOE_LORA_ROOT not set.");
        string rawDir = Path.Combine(root!, "diffusiongemma_csharp_moe_lora");
        string convDir = Path.Combine(root!, "diffusiongemma_csharp_moe_lora_dotllm");
        Skip.IfNot(Directory.Exists(rawDir) && Directory.Exists(convDir),
            "Raw/converted adapter directories not found under DOTLLM_DG_MOE_LORA_ROOT.");

        using var raw = PeftAdapterLoader.LoadFromDirectory("dg-raw", rawDir);
        using var conv = PeftAdapterLoader.LoadFromDirectory("dg-conv", convDir);

        // Real adapter geometry: E=128 experts, hidden=2816, moe inter=704,
        // r=8, targeted decoder layers {0,4,...,28}.
        int[] layers = [0, 4, 8, 12, 16, 20, 24, 28];
        const int E = 128, RealHidden = 2816, RealInter = 704, R = 8;
        Assert.Equal(R, raw.Rank);

        (string Proj, int In, int Out)[] projs =
        [
            ("gate_proj", RealHidden, RealInter),
            ("up_proj", RealHidden, RealInter),
            ("down_proj", RealInter, RealHidden),
        ];

        foreach (int layer in layers)
        {
            for (int e = 0; e < E; e++)
            {
                foreach (var (proj, inDim, outDim) in projs)
                {
                    string key = $"mlp.experts.{e}.{proj}";
                    var w1 = raw.GetLayerWeights(layer, key, LoraRegion.Decoder);
                    var w2 = conv.GetLayerWeights(layer, key, LoraRegion.Decoder);
                    Assert.True(w1.HasValue, $"raw missing layer {layer} {key}");
                    Assert.True(w2.HasValue, $"converted missing layer {layer} {key}");
                    Assert.Equal(inDim, w1!.Value.InputDim);
                    Assert.Equal(w2!.Value.InputDim, w1.Value.InputDim);
                    Assert.Equal(w2.Value.OutputDim, w1.Value.OutputDim);

                    var b1 = AsSpan(w1.Value.BHandle, R * inDim);
                    var b2 = AsSpan(w2.Value.BHandle, R * inDim);
                    Assert.True(b1.SequenceEqual(b2), $"B factor mismatch at layer {layer} {key}");

                    var a1 = AsSpan(w1.Value.AHandle, outDim * R);
                    var a2 = AsSpan(w2.Value.AHandle, outDim * R);
                    Assert.True(a1.SequenceEqual(a2), $"A factor mismatch at layer {layer} {key}");
                }
            }
        }

        // The plain nn.Linear entries (q/v_proj) coexist in both files and must
        // still load through the untouched projection path.
        Assert.NotNull(raw.GetLayerWeights(0, "q_proj", LoraRegion.Decoder));
        Assert.NotNull(conv.GetLayerWeights(0, "q_proj", LoraRegion.Decoder));
    }
}
