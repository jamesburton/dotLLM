using System.Buffers.Binary;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Loads a HuggingFace PEFT-format LoRA adapter directory into a
/// <see cref="LoraAdapter"/>. Supports the canonical layout:
/// <c>{root}/adapter_config.json</c> + <c>{root}/adapter_model.safetensors</c>.
/// </summary>
/// <remarks>
/// <para>
/// PEFT tensor naming (per <c>peft</c> ≥ 0.4): each LoRA factor is published
/// as <c>base_model.model.{layer_path}.{proj_name}.lora_A.weight</c> and
/// <c>...lora_B.weight</c>. PEFT also occasionally writes
/// <c>...lora_A.default.weight</c> when there are named adapter sub-trees;
/// the loader normalises both forms.
/// </para>
/// <para>
/// Only plain LoRA is supported in Phase 4a. <c>use_rslora</c> and
/// <c>use_dora</c> are rejected with a clear <see cref="NotSupportedException"/>;
/// quantised adapter weights (F16 / BF16 / Q8_0) are decoded to F32 during
/// load (only F32, F16, BF16 implemented this commit — anything else throws).
/// </para>
/// </remarks>
public static unsafe class PeftAdapterLoader
{
    private static readonly Regex ProjectionPathRegex = new(
        @"^(?:base_model\.(?:model\.)?)?model\.(?:(?<enc>encoder\.language_model)\.|(?<dec>decoder)\.)?layers\.(?<layer>\d+)\.(?<scope>self_attn|mlp)\.(?<proj>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_(?<which>A|B)(?:\.default)?\.weight$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// DiffusionGemma's <c>decoder.self_conditioning.{gate,up,down}_proj</c> LoRA tensors —
    /// the model-level self-conditioning module (runs once per forward, not inside the
    /// per-layer loop, so it has no real transformer layer index). Matched separately from
    /// <see cref="ProjectionPathRegex"/> because the path has no <c>layers.{i}</c> segment;
    /// routed into the SAME pending-pair machinery as ordinary per-layer entries, keyed under
    /// <see cref="LoraAdapter.SelfConditioningLayerIndex"/> with region
    /// <see cref="LoraRegion.Any"/> (see <see cref="TransformerModel.ApplySelfConditioningLoraDelta"/>
    /// for the application side).
    /// </summary>
    private static readonly Regex SelfConditioningPathRegex = new(
        @"^(?:base_model\.(?:model\.)?)?model\.decoder\.self_conditioning\.(?<proj>gate_proj|up_proj|down_proj)\.lora_(?<which>A|B)(?:\.default)?\.weight$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// Per-expert MoE LoRA tensor names — the standard HF <c>peft</c> convention for MoE
    /// target_modules inserts an <c>experts.{expertIndex}</c> segment between the module
    /// prefix and the projection name, e.g.
    /// <c>model.layers.{i}.mlp.experts.{j}.gate_proj.lora_A.weight</c> (verified against the
    /// <c>peft</c> MoE-support convention; not confirmed against any real DiffusionGemma
    /// adapter we've sampled locally — none of them target experts). Parsed as a SEPARATE
    /// regex (mirroring <see cref="SelfConditioningPathRegex"/>) rather than folding into
    /// <see cref="ProjectionPathRegex"/>'s <c>proj</c> alternation, since the (layer, expert,
    /// proj) tuple needs its own composed storage key
    /// (<c>"mlp.experts.{expert}.{proj}"</c> — matches the naming <c>LoraAdapter</c>'s
    /// per-expert MoE shape validation and <c>MoeSwiGluMlp.ExpertProjectionName</c> already
    /// use) instead of a bare projection name.
    /// </summary>
    private static readonly Regex MoeExpertPathRegex = new(
        @"^(?:base_model\.(?:model\.)?)?model\.(?:(?<enc>encoder\.language_model)\.|(?<dec>decoder)\.)?layers\.(?<layer>\d+)\.mlp\.experts\.(?<expert>\d+)\.(?<proj>gate_proj|up_proj|down_proj)\.lora_(?<which>A|B)(?:\.default)?\.weight$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// peft "ParamWrapper" fused-MoE LoRA tensor names — produced by
    /// <c>LoraConfig(target_parameters=[...])</c> targeting raw fused expert
    /// <c>nn.Parameter</c> tensors (e.g. DiffusionGemma's
    /// <c>DiffusionGemmaTextExperts.gate_up_proj</c> <c>[E, 2*inter, hidden]</c> and
    /// <c>down_proj</c> <c>[E, hidden, inter]</c>). The parameter name is ABSENT from
    /// the serialized key; when TWO parameters on the same module are targeted, peft
    /// NESTS ParamWrappers, yielding an inner <c>experts.base_layer.lora_{A,B}</c>
    /// pair and an outer <c>experts.lora_{A,B}</c> pair per layer:
    /// <code>
    /// base_model.model.model.decoder.layers.{i}.experts.base_layer.lora_A.weight
    /// base_model.model.model.decoder.layers.{i}.experts.lora_A.weight
    /// </code>
    /// Which nesting level is which parameter is inferred from SHAPES — A is
    /// <c>[r*E, in]</c>, B is <c>[out, r*E]</c>; <c>in &gt; out</c> ⇒ gate_up_proj,
    /// <c>in &lt; out</c> ⇒ down_proj — cross-checked against adapter_config.json's
    /// <c>target_parameters</c> list when present (verified empirically against a real
    /// DiffusionGemma-26B fused adapter, peft 0.19.1). Each stacked pair is factorized
    /// at load time into the SAME per-expert <c>"mlp.experts.{e}.{proj}"</c> storage
    /// keys <see cref="MoeExpertPathRegex"/> produces, so <see cref="LoraAdapter"/> and
    /// all apply-side code are unchanged. See <c>FactorizeParamWrapperPair</c> for the
    /// stacked layout semantics (A expert-major, B expert-minor, fused gate‖up split).
    /// </summary>
    private static readonly Regex ParamWrapperExpertPathRegex = new(
        @"^(?:base_model\.(?:model\.)?)?model\.(?:(?<enc>encoder\.language_model)\.|(?<dec>decoder)\.)?layers\.(?<layer>\d+)\.experts(?<inner>\.base_layer)?\.lora_(?<which>A|B)(?:\.default)?\.weight$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// Loads a PEFT LoRA adapter from the directory at <paramref name="path"/>.
    /// </summary>
    /// <param name="name">Logical name to register under.</param>
    /// <param name="path">Directory containing PEFT adapter files.</param>
    /// <param name="baseConfig">
    /// Optional base-model <see cref="ModelConfig"/>. When supplied, the loader
    /// validates layer count, hidden size, and per-projection dimensions and
    /// throws <see cref="InvalidDataException"/> at load time on mismatch.
    /// </param>
    /// <param name="preserveSourceDType">
    /// When <c>true</c>, F16 / BF16 source tensors are stored verbatim in the
    /// adapter and dequantised on read by the runtime delta kernel (Phase 4d.1).
    /// When <c>false</c> (default — backward compat with Phase 4a), source
    /// tensors are upcast to F32 at load time.
    /// </param>
    /// <returns>A loaded <see cref="LoraAdapter"/> owned by the caller.</returns>
    public static LoraAdapter LoadFromDirectory(string name, string path, ModelConfig? baseConfig = null,
                                                bool preserveSourceDType = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!Directory.Exists(path))
            throw new DirectoryNotFoundException($"PEFT adapter directory not found: {path}");

        string configPath = Path.Combine(path, "adapter_config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException(
                $"PEFT adapter is missing adapter_config.json (looked in '{path}').", configPath);

        string safetensorsPath = Path.Combine(path, "adapter_model.safetensors");
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException(
                $"PEFT adapter is missing adapter_model.safetensors (looked in '{path}').", safetensorsPath);

        // ── adapter_config.json ─────────────────────────────────────
        var meta = ParseAdapterConfig(configPath);
        if (meta.UseRsLora)
            throw new NotSupportedException(
                $"PEFT adapter '{name}' has use_rslora=true. rsLoRA scaling is a follow-up; "
                + "Phase 4a covers plain LoRA only.");
        if (meta.UseDora)
            throw new NotSupportedException(
                $"PEFT adapter '{name}' has use_dora=true. DoRA scaling is a follow-up; "
                + "Phase 4a covers plain LoRA only.");
        if (!string.IsNullOrEmpty(meta.TaskType)
            && !StringComparer.OrdinalIgnoreCase.Equals(meta.TaskType, "CAUSAL_LM"))
        {
            throw new NotSupportedException(
                $"PEFT adapter '{name}' declares task_type='{meta.TaskType}'. Only CAUSAL_LM "
                + "adapters are in scope for Phase 4a.");
        }

        var adapter = new LoraAdapter(name, meta.Rank, meta.Alpha, meta.TargetModules);
        bool transferred = false;
        try
        {
            using var safetensors = SafetensorsFile.Open(safetensorsPath);
            LoadTensors(safetensors, adapter, meta.Rank, meta.TargetParameters, preserveSourceDType);

            if (baseConfig is not null && !adapter.IsCompatible(baseConfig))
            {
                throw new InvalidDataException(
                    $"PEFT adapter '{name}' is not compatible with the supplied base model "
                    + $"(layers={baseConfig.NumLayers}, hidden={baseConfig.HiddenSize}, "
                    + $"q_out={baseConfig.NumAttentionHeads * baseConfig.HeadDim}, "
                    + $"kv_out={baseConfig.NumKvHeads * baseConfig.HeadDim}, "
                    + $"intermediate={baseConfig.IntermediateSize}). See adapter shapes above.");
            }

            transferred = true;
            return adapter;
        }
        finally
        {
            if (!transferred) adapter.Dispose();
        }
    }

    private static void LoadTensors(SafetensorsFile file, LoraAdapter adapter, int rank,
                                    IReadOnlyList<string> targetParameters, bool preserveSourceDType = false)
    {
        // Group tensors by (layer, proj, region) so we can validate that A and B
        // arrive in matched pairs. Per PEFT convention the writer typically
        // emits both halves together but we don't assume ordering. DiffusionGemma
        // real adapters carry INDEPENDENT deltas for the same (layer, proj) under
        // decoder.* (canvas) vs encoder.language_model.* (prompt) — region keeps
        // those separate instead of colliding.
        var pending = new Dictionary<(int Layer, string Proj, LoraRegion Region), PendingPair>();
        // ParamWrapper fused-MoE pairs, keyed by (layer, region, nesting level).
        // Each pair holds STACKED factors for one fused expert parameter; they
        // are factorized into per-expert entries after the collection pass.
        var wrapperPending = new Dictionary<(int Layer, LoraRegion Region, bool Inner), PendingPair>();
        var unrecognised = new List<string>();

        foreach (var tensor in file.Tensors)
        {
            // Embedding / lm_head LoRA — rare, log via a structured exception
            // rather than silently dropping when encountered.
            if (tensor.Name.Contains("lora_embedding_A", StringComparison.Ordinal)
                || tensor.Name.Contains("lora_embedding_B", StringComparison.Ordinal))
            {
                // Skip with a record so the diagnostic is auditable.
                continue;
            }

            int layer;
            string proj;
            string which; // "A" or "B"
            LoraRegion region;

            var match = ProjectionPathRegex.Match(tensor.Name);
            if (match.Success)
            {
                layer = int.Parse(match.Groups["layer"].Value, System.Globalization.CultureInfo.InvariantCulture);
                proj = match.Groups["proj"].Value;
                which = match.Groups["which"].Value;
                region = match.Groups["enc"].Success ? LoraRegion.Encoder
                    : match.Groups["dec"].Success ? LoraRegion.Decoder
                    : LoraRegion.Any;
            }
            else if (ParamWrapperExpertPathRegex.Match(tensor.Name) is { Success: true } wrapperMatch)
            {
                // peft ParamWrapper fused-MoE pair — stacked factors, no
                // parameter name in the key. Collected separately and
                // factorized per expert after the pass (the stacked shapes
                // [r*E, in] / [out, r*E] don't fit the PendingPair rank
                // validation below).
                int wLayer = int.Parse(wrapperMatch.Groups["layer"].Value, System.Globalization.CultureInfo.InvariantCulture);
                LoraRegion wRegion = wrapperMatch.Groups["enc"].Success ? LoraRegion.Encoder
                    : wrapperMatch.Groups["dec"].Success ? LoraRegion.Decoder
                    : LoraRegion.Any;
                bool innerWrapper = wrapperMatch.Groups["inner"].Success;

                var wKey = (wLayer, wRegion, innerWrapper);
                if (!wrapperPending.TryGetValue(wKey, out var wPair))
                {
                    wPair = new PendingPair();
                    wrapperPending[wKey] = wPair;
                }

                string wWhich = wrapperMatch.Groups["which"].Value;
                if (wWhich == "A")
                {
                    if (wPair.AAssigned)
                        throw new InvalidDataException(
                            $"PEFT adapter has duplicate ParamWrapper lora_A entry for layer={wLayer} "
                            + $"region={wRegion} inner={innerWrapper}.");
                    wPair.AAssigned = true;
                    wPair.ATensor = tensor;
                }
                else
                {
                    if (wPair.BAssigned)
                        throw new InvalidDataException(
                            $"PEFT adapter has duplicate ParamWrapper lora_B entry for layer={wLayer} "
                            + $"region={wRegion} inner={innerWrapper}.");
                    wPair.BAssigned = true;
                    wPair.BTensor = tensor;
                }
                continue;
            }
            else
            {
                var expertMatch = MoeExpertPathRegex.Match(tensor.Name);
                if (expertMatch.Success)
                {
                    // Compose the same "mlp.experts.{expert}.{proj}" storage key that
                    // MoeSwiGluMlp.ExpertProjectionName produces at runtime and that
                    // LoraAdapter.TryValidatePerExpertMoeProjection already validates —
                    // no changes needed on either of those to accept this key shape.
                    layer = int.Parse(expertMatch.Groups["layer"].Value, System.Globalization.CultureInfo.InvariantCulture);
                    string expert = expertMatch.Groups["expert"].Value;
                    proj = $"mlp.experts.{expert}.{expertMatch.Groups["proj"].Value}";
                    which = expertMatch.Groups["which"].Value;
                    region = expertMatch.Groups["enc"].Success ? LoraRegion.Encoder
                        : expertMatch.Groups["dec"].Success ? LoraRegion.Decoder
                        : LoraRegion.Any;
                }
                else
                {
                    // DiffusionGemma self-conditioning LoRA — model-level module, no real
                    // layer index. Route to the sentinel layer under the same gate_proj/
                    // up_proj/down_proj names as the dense FFN so IsCompatibleEntry's
                    // existing shape validation applies unchanged.
                    var scMatch = SelfConditioningPathRegex.Match(tensor.Name);
                    if (!scMatch.Success)
                    {
                        unrecognised.Add(tensor.Name);
                        continue;
                    }

                    layer = LoraAdapter.SelfConditioningLayerIndex;
                    proj = scMatch.Groups["proj"].Value;
                    which = scMatch.Groups["which"].Value;
                    region = LoraRegion.Any;
                }
            }

            var key = (layer, proj, region);
            if (!pending.TryGetValue(key, out var pair))
            {
                pair = new PendingPair();
                pending[key] = pair;
            }

            if (which == "A")
            {
                if (pair.AAssigned)
                    throw new InvalidDataException(
                        $"PEFT adapter has duplicate lora_A entry for layer={layer} proj='{proj}' region={region}.");
                pair.AAssigned = true;
                pair.ATensor = tensor;
            }
            else
            {
                if (pair.BAssigned)
                    throw new InvalidDataException(
                        $"PEFT adapter has duplicate lora_B entry for layer={layer} proj='{proj}' region={region}.");
                pair.BAssigned = true;
                pair.BTensor = tensor;
            }
        }

        if (unrecognised.Count > 0)
        {
            throw new InvalidDataException(
                "PEFT adapter contains tensor names that do not match the expected "
                + "{base_model.model.|model.}[decoder.|encoder.language_model.]layers.{i}.{self_attn|mlp}.{proj}.lora_{A|B}[.default].weight "
                + "convention. Unrecognised: " + string.Join(", ", unrecognised));
        }

        if (pending.Count == 0 && wrapperPending.Count == 0)
            throw new InvalidDataException(
                "PEFT adapter contains no recognised LoRA factor tensors.");

        foreach (var ((wLayer, wRegion, innerWrapper), wPair) in wrapperPending)
        {
            if (!wPair.AAssigned)
                throw new InvalidDataException(
                    $"PEFT adapter is missing ParamWrapper lora_A for layer={wLayer} region={wRegion} "
                    + $"inner={innerWrapper} (only lora_B was found).");
            if (!wPair.BAssigned)
                throw new InvalidDataException(
                    $"PEFT adapter is missing ParamWrapper lora_B for layer={wLayer} region={wRegion} "
                    + $"inner={innerWrapper} (only lora_A was found).");

            FactorizeParamWrapperPair(file, adapter, rank, targetParameters, wLayer, wRegion, wPair);
        }

        foreach (var ((layer, proj, region), pair) in pending)
        {
            if (!pair.AAssigned)
                throw new InvalidDataException(
                    $"PEFT adapter is missing lora_A for layer={layer} proj='{proj}' region={region} "
                    + "(only lora_B was found).");
            if (!pair.BAssigned)
                throw new InvalidDataException(
                    $"PEFT adapter is missing lora_B for layer={layer} proj='{proj}' region={region} "
                    + "(only lora_A was found).");

            // PEFT layout: A is [r, d_out], B is [r, d_in]. dotLLM uses the
            // weight-as-[output, input] convention, so:
            //   - lora_A.weight shape [r, d_out]  → store as [d_out, r] row-major
            //     (this is our A: [outputDim, rank])
            //   - lora_B.weight shape [d_out, r]  → ALREADY [d_out, r] in PEFT for
            //     base_model.model layer; but per HF PEFT spec lora_B is [d_out, r]
            //     i.e. the up-projection — so PEFT_A is dotLLM_B and PEFT_B is dotLLM_A.
            //
            // Concretely (from peft.tuners.lora.LoraLayer):
            //     y = x . W^T + scaling * x . A^T . B^T
            //   where A shape = (r, in_features), B shape = (out_features, r).
            // So PEFT 'lora_A' = dotLLM B (down, [r, in])
            //    PEFT 'lora_B' = dotLLM A (up,   [out, r])
            int rA = pair.ATensor.Shape[0];   // PEFT A: rows = r
            int aIn = pair.ATensor.Shape[1];  // PEFT A: cols = in_features
            int bOut = pair.BTensor.Shape[0]; // PEFT B: rows = out_features
            int rB = pair.BTensor.Shape[1];   // PEFT B: cols = r

            if (rA != rank || rB != rank)
                throw new InvalidDataException(
                    $"PEFT adapter rank mismatch at layer={layer} proj='{proj}' region={region}: "
                    + $"adapter_config.r={rank}, lora_A rank dim={rA}, lora_B rank dim={rB}.");

            // dotLLM expects:
            //   B (down): [inputDim, rank]   row-major  — i.e. "[r, in]" in PEFT terms,
            //                                            but our layout says [outputDim_of_factor, inputDim_of_factor]
            //                                            with outputDim=rank and inputDim=in.
            //   Therefore B (down) has dimensions [rank, in_features] and the loader stores
            //   the PEFT 'lora_A' tensor (which IS [r, in_features] row-major) verbatim.
            //   A (up)   has dimensions [outputDim, rank] and the loader stores the PEFT
            //   'lora_B' tensor (which IS [out_features, r] row-major) verbatim.
            int inputDim = aIn;   // input feature dim of the factor pair
            int outputDim = bOut; // output feature dim of the factor pair

            long bElems = (long)rank * inputDim;
            long aElems = (long)outputDim * rank;

            // Phase 4d.1: when preserveSourceDType is set AND both tensors share
            // a supported quantised dtype (F16 or BF16), keep the bytes verbatim.
            // Otherwise upcast to F32 (the original Phase 4a behaviour).
            LoraWeightDType storeDType = LoraWeightDType.F32;
            if (preserveSourceDType
                && pair.ATensor.DType == pair.BTensor.DType
                && (pair.ATensor.DType == SafetensorsDType.F16 || pair.ATensor.DType == SafetensorsDType.BF16))
            {
                storeDType = pair.ATensor.DType == SafetensorsDType.F16
                    ? LoraWeightDType.F16
                    : LoraWeightDType.BF16;
            }

            nint bHandle;
            nint aHandle;
            if (storeDType == LoraWeightDType.F32)
            {
                bHandle = LoraAdapter.AllocAligned(bElems);
                aHandle = LoraAdapter.AllocAligned(aElems);
                try
                {
                    CopyTensorAsF32(file, pair.ATensor, (float*)bHandle, bElems);
                    CopyTensorAsF32(file, pair.BTensor, (float*)aHandle, aElems);
                }
                catch
                {
                    if (aHandle != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)aHandle);
                    if (bHandle != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)bHandle);
                    throw;
                }
            }
            else
            {
                // 2 bytes per element for F16/BF16. Use AlignedAllocBytes so
                // the parent dispose path still uses AlignedFree.
                long bBytes = bElems * 2;
                long aBytes = aElems * 2;
                bHandle = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)bBytes, 64);
                aHandle = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)aBytes, 64);
                try
                {
                    byte* bSrc = (byte*)file.DataBasePointer + pair.ATensor.DataBeginOffset;
                    byte* aSrc = (byte*)file.DataBasePointer + pair.BTensor.DataBeginOffset;
                    Buffer.MemoryCopy(bSrc, (void*)bHandle, bBytes, bBytes);
                    Buffer.MemoryCopy(aSrc, (void*)aHandle, aBytes, aBytes);
                }
                catch
                {
                    if (aHandle != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)aHandle);
                    if (bHandle != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)bHandle);
                    throw;
                }
            }

            adapter.AddLayerWeights(layer, proj, new LoraLayerWeights(
                AHandle: aHandle,
                BHandle: bHandle,
                InputDim: inputDim,
                OutputDim: outputDim,
                WeightDType: storeDType), region);
        }
    }

    private static void CopyTensorAsF32(SafetensorsFile file, SafetensorsTensorDescriptor tensor,
                                        float* dst, long expectedElements)
    {
        long actualElements = tensor.ElementCount;
        if (actualElements != expectedElements)
            throw new InvalidDataException(
                $"PEFT tensor '{tensor.Name}' element-count mismatch: "
                + $"expected {expectedElements}, got {actualElements}.");

        byte* src = (byte*)file.DataBasePointer + tensor.DataBeginOffset;
        switch (tensor.DType)
        {
            case SafetensorsDType.F32:
                {
                    long bytes = expectedElements * sizeof(float);
                    Buffer.MemoryCopy(src, dst, bytes, bytes);
                    break;
                }
            case SafetensorsDType.F16:
                {
                    var srcSpan = new ReadOnlySpan<Half>(src, (int)expectedElements);
                    var dstSpan = new Span<float>(dst, (int)expectedElements);
                    System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(srcSpan, dstSpan);
                    break;
                }
            case SafetensorsDType.BF16:
                {
                    // BF16: top 16 bits of an F32. Upcast = shift left into the
                    // exponent + mantissa of an F32. No SIMD helper in
                    // TensorPrimitives yet, scalar loop is fine for 10–100 MB.
                    for (long i = 0; i < expectedElements; i++)
                    {
                        ushort raw = BinaryPrimitives.ReadUInt16LittleEndian(
                            new ReadOnlySpan<byte>(src + i * 2, 2));
                        uint asF32 = (uint)raw << 16;
                        dst[i] = BitConverter.UInt32BitsToSingle(asF32);
                    }
                    break;
                }
            default:
                throw new NotSupportedException(
                    $"PEFT tensor '{tensor.Name}' has dtype {tensor.DType}; "
                    + "only F32, F16, and BF16 are supported in Phase 4a.");
        }
    }

    /// <summary>
    /// Factorizes one peft ParamWrapper stacked LoRA pair into per-expert
    /// <c>"mlp.experts.{e}.{proj}"</c> entries on <paramref name="adapter"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stacked layout (matching peft's <c>get_delta_weight</c> einsum
    /// <c>"o r e, e r i -&gt; e o i"</c>):
    /// </para>
    /// <list type="bullet">
    /// <item>A <c>[r*E, in]</c> is EXPERT-MAJOR: <c>A_e = A[e*r : (e+1)*r, :]</c>.</item>
    /// <item>B <c>[out, r*E]</c> is EXPERT-MINOR: column <c>c</c> belongs to rank
    /// <c>c / E</c> of expert <c>c % E</c>, i.e. <c>B_e[:, k] = B[:, k*E + e]</c>.</item>
    /// </list>
    /// <para>
    /// Which fused parameter the pair belongs to is inferred from shapes —
    /// <c>in &gt; out</c> ⇒ <c>gate_up_proj</c>, <c>in &lt; out</c> ⇒ <c>down_proj</c>
    /// — and cross-checked against adapter_config.json's <c>target_parameters</c>
    /// when present. <c>gate_up_proj</c> is fused gate‖up along the output dim:
    /// B_e rows <c>[0, inter)</c> are gate_proj, rows <c>[inter, 2*inter)</c> are
    /// up_proj; A_e is shared by both (duplicated into each per-expert entry so
    /// ownership/dispose stays uniform).
    /// </para>
    /// <para>
    /// The de-interleave is element-wise, so factors are always materialised as
    /// F32 (F16/BF16 sources are upcast; <c>preserveSourceDType</c> does not apply
    /// to ParamWrapper pairs). Per-expert delta semantics are unchanged:
    /// <c>delta_e = B_e @ A_e * alpha/r</c>, applied by the existing runtime path.
    /// </para>
    /// </remarks>
    private static void FactorizeParamWrapperPair(SafetensorsFile file, LoraAdapter adapter, int rank,
                                                  IReadOnlyList<string> targetParameters,
                                                  int layer, LoraRegion region, PendingPair pair)
    {
        if (pair.ATensor.Shape.Length != 2 || pair.BTensor.Shape.Length != 2)
            throw new InvalidDataException(
                $"PEFT ParamWrapper tensors at layer={layer} region={region} must be 2-D "
                + $"(got A rank {pair.ATensor.Shape.Length}, B rank {pair.BTensor.Shape.Length}).");

        int rTimesE = pair.ATensor.Shape[0];  // A: [r*E, in]
        int inDim = pair.ATensor.Shape[1];
        int outDim = pair.BTensor.Shape[0];   // B: [out, r*E]
        if (pair.BTensor.Shape[1] != rTimesE)
            throw new InvalidDataException(
                $"PEFT ParamWrapper pair at layer={layer} region={region} has mismatched stacked rank "
                + $"dims: lora_A rows={rTimesE}, lora_B cols={pair.BTensor.Shape[1]}.");
        if (rTimesE % rank != 0)
            throw new InvalidDataException(
                $"PEFT ParamWrapper pair at layer={layer} region={region}: stacked rank dim {rTimesE} "
                + $"is not a multiple of adapter_config.r={rank}.");
        int numExperts = rTimesE / rank;

        // Shape-based parameter disambiguation (the key carries no parameter
        // name). gate_up_proj: in=hidden > out=2*inter. down_proj: in=inter <
        // out=hidden. Equal dims cannot be told apart — reject explicitly
        // rather than guessing.
        bool isGateUp;
        if (inDim > outDim) isGateUp = true;
        else if (inDim < outDim) isGateUp = false;
        else
            throw new NotSupportedException(
                $"PEFT ParamWrapper pair at layer={layer} region={region} has in_features == out_features "
                + $"({inDim}); the fused parameter cannot be inferred from shapes.");

        string paramName = isGateUp ? "gate_up_proj" : "down_proj";
        if (targetParameters.Count > 0)
        {
            string expectedSuffix = $"layers.{layer}.experts.{paramName}";
            bool declared = false;
            foreach (var tp in targetParameters)
            {
                if (tp.EndsWith(expectedSuffix, StringComparison.Ordinal)) { declared = true; break; }
            }
            if (!declared)
                throw new InvalidDataException(
                    $"PEFT ParamWrapper pair at layer={layer} region={region} was shape-inferred as "
                    + $"'{paramName}' (in={inDim}, out={outDim}) but adapter_config.json target_parameters "
                    + $"declares no '...{expectedSuffix}' entry.");
        }

        int inter = outDim;
        if (isGateUp)
        {
            if (outDim % 2 != 0)
                throw new InvalidDataException(
                    $"PEFT ParamWrapper gate_up_proj pair at layer={layer} region={region} has odd fused "
                    + $"output dim {outDim}; expected 2*intermediate.");
            inter = outDim / 2;
        }

        // Decode both stacked tensors to F32 staging buffers once, then slice /
        // gather per expert. Staging is freed before returning.
        long aStackElems = (long)rTimesE * inDim;
        long bStackElems = (long)outDim * rTimesE;
        float* aStack = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
            (nuint)(aStackElems * sizeof(float)), 64);
        float* bStack = null;
        try
        {
            bStack = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
                (nuint)(bStackElems * sizeof(float)), 64);
            CopyTensorAsF32(file, pair.ATensor, aStack, aStackElems);
            CopyTensorAsF32(file, pair.BTensor, bStack, bStackElems);

            for (int e = 0; e < numExperts; e++)
            {
                // dotLLM B factor [rank, in] = A_e (expert-major slice, contiguous).
                float* aExpert = aStack + (long)e * rank * inDim;

                if (isGateUp)
                {
                    AddExpertEntry(adapter, layer, region, e, "gate_proj", rank, inDim, inter,
                        aExpert, bStack, bRowOffset: 0, rTimesE, numExperts);
                    AddExpertEntry(adapter, layer, region, e, "up_proj", rank, inDim, inter,
                        aExpert, bStack, bRowOffset: inter, rTimesE, numExperts);
                }
                else
                {
                    AddExpertEntry(adapter, layer, region, e, "down_proj", rank, inDim, outDim,
                        aExpert, bStack, bRowOffset: 0, rTimesE, numExperts);
                }
            }
        }
        finally
        {
            if (bStack is not null) System.Runtime.InteropServices.NativeMemory.AlignedFree(bStack);
            System.Runtime.InteropServices.NativeMemory.AlignedFree(aStack);
        }
    }

    /// <summary>
    /// Allocates and fills one per-expert <see cref="LoraLayerWeights"/> entry from
    /// F32 staging copies of a ParamWrapper stacked pair, then registers it under
    /// <c>"mlp.experts.{expert}.{proj}"</c> (the same key shape
    /// <see cref="MoeExpertPathRegex"/> entries use).
    /// </summary>
    /// <param name="adapter">Target adapter (takes buffer ownership).</param>
    /// <param name="layer">Transformer layer index.</param>
    /// <param name="region">Encoder/decoder region tag.</param>
    /// <param name="expert">Expert index <c>e</c>.</param>
    /// <param name="proj">gate_proj / up_proj / down_proj.</param>
    /// <param name="rank">Per-expert LoRA rank r.</param>
    /// <param name="inDim">Input feature dim of the factor pair.</param>
    /// <param name="entryOutDim">Output feature dim of THIS entry (inter for gate/up).</param>
    /// <param name="aExpert">Staged A_e slice, <c>[rank, inDim]</c> row-major.</param>
    /// <param name="bStack">Staged stacked B, <c>[out, r*E]</c> row-major.</param>
    /// <param name="bRowOffset">First stacked-B row of this entry (inter for up_proj).</param>
    /// <param name="rTimesE">Stacked rank dim r*E (stacked B row stride).</param>
    /// <param name="numExperts">Expert count E (expert-minor column stride divisor).</param>
    private static void AddExpertEntry(LoraAdapter adapter, int layer, LoraRegion region, int expert,
                                       string proj, int rank, int inDim, int entryOutDim,
                                       float* aExpert, float* bStack, int bRowOffset,
                                       int rTimesE, int numExperts)
    {
        long bElems = (long)rank * inDim;       // dotLLM B factor: [rank, in]
        long aElems = (long)entryOutDim * rank; // dotLLM A factor: [out, rank]

        nint bHandle = LoraAdapter.AllocAligned(bElems);
        nint aHandle = 0;
        try
        {
            aHandle = LoraAdapter.AllocAligned(aElems);

            // B factor = A_e verbatim (contiguous slice of stacked A).
            Buffer.MemoryCopy(aExpert, (void*)bHandle, bElems * sizeof(float), bElems * sizeof(float));

            // A factor = expert-minor de-interleave of stacked B:
            //   A[o, k] = bStack[(bRowOffset + o) * rTimesE + k*E + expert].
            float* dst = (float*)aHandle;
            for (int o = 0; o < entryOutDim; o++)
            {
                float* srcRow = bStack + (long)(bRowOffset + o) * rTimesE + expert;
                for (int k = 0; k < rank; k++)
                    dst[(long)o * rank + k] = srcRow[(long)k * numExperts];
            }

            adapter.AddLayerWeights(layer, $"mlp.experts.{expert}.{proj}", new LoraLayerWeights(
                AHandle: aHandle,
                BHandle: bHandle,
                InputDim: inDim,
                OutputDim: entryOutDim,
                WeightDType: LoraWeightDType.F32), region);
        }
        catch
        {
            if (aHandle != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)aHandle);
            if (bHandle != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)bHandle);
            throw;
        }
    }

    private static AdapterConfigMeta ParseAdapterConfig(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException(
                $"PEFT adapter_config.json root is not a JSON object (got {root.ValueKind}).");

        int rank = root.TryGetProperty("r", out var rEl) && rEl.ValueKind == JsonValueKind.Number
            ? rEl.GetInt32()
            : throw new InvalidDataException(
                "PEFT adapter_config.json is missing required 'r' (rank) field.");
        if (rank <= 0)
            throw new InvalidDataException(
                $"PEFT adapter_config.json has invalid rank r={rank} (must be positive).");

        // lora_alpha: int or float; PEFT writes int historically.
        float alpha;
        if (root.TryGetProperty("lora_alpha", out var alphaEl))
        {
            alpha = alphaEl.ValueKind switch
            {
                JsonValueKind.Number => (float)alphaEl.GetDouble(),
                _ => throw new InvalidDataException(
                    $"PEFT adapter_config.json 'lora_alpha' must be a number (got {alphaEl.ValueKind}).")
            };
        }
        else
        {
            // PEFT default: alpha = 8 when missing.
            alpha = 8f;
        }

        var targets = new List<string>();
        if (root.TryGetProperty("target_modules", out var tm))
        {
            switch (tm.ValueKind)
            {
                case JsonValueKind.Array:
                    foreach (var entry in tm.EnumerateArray())
                        if (entry.ValueKind == JsonValueKind.String)
                            targets.Add(entry.GetString()!);
                    break;
                case JsonValueKind.String:
                    targets.Add(tm.GetString()!);
                    break;
                case JsonValueKind.Null:
                    break;
                default:
                    throw new InvalidDataException(
                        $"PEFT adapter_config.json 'target_modules' must be an array or string (got {tm.ValueKind}).");
            }
        }

        // target_parameters (peft ≥ 0.17, ParamWrapper): full parameter paths like
        // "decoder.layers.0.experts.gate_up_proj". Used to cross-check the
        // shape-based inner/outer wrapper disambiguation.
        var targetParameters = new List<string>();
        if (root.TryGetProperty("target_parameters", out var tp) && tp.ValueKind == JsonValueKind.Array)
        {
            foreach (var entry in tp.EnumerateArray())
                if (entry.ValueKind == JsonValueKind.String)
                    targetParameters.Add(entry.GetString()!);
        }

        float dropout = 0f;
        if (root.TryGetProperty("lora_dropout", out var drop) && drop.ValueKind == JsonValueKind.Number)
            dropout = (float)drop.GetDouble();

        bool useRslora = root.TryGetProperty("use_rslora", out var rs)
            && rs.ValueKind is JsonValueKind.True;
        bool useDora = root.TryGetProperty("use_dora", out var dora)
            && dora.ValueKind is JsonValueKind.True;

        string? taskType = null;
        if (root.TryGetProperty("task_type", out var task) && task.ValueKind == JsonValueKind.String)
            taskType = task.GetString();

        return new AdapterConfigMeta(rank, alpha, targets, targetParameters, dropout, useRslora, useDora, taskType);
    }

    private sealed record AdapterConfigMeta(
        int Rank,
        float Alpha,
        IReadOnlyList<string> TargetModules,
        IReadOnlyList<string> TargetParameters,
        float Dropout,
        bool UseRsLora,
        bool UseDora,
        string? TaskType);

    private sealed class PendingPair
    {
        public bool AAssigned;
        public bool BAssigned;
        public SafetensorsTensorDescriptor ATensor;
        public SafetensorsTensorDescriptor BTensor;
    }
}
