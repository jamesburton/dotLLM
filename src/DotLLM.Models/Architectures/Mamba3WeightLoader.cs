using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Resolves an opened <see cref="SafetensorsFile"/> against
/// <see cref="Mamba3TensorMapping"/> and returns a fully-populated
/// <see cref="Mamba3Weights"/> — or a structured report describing which
/// tensors are missing / shape-mismatched / dtype-unsupported.
/// </summary>
/// <remarks>
/// <para>
/// Stage D2 only materialises an F32 ingest path. Every tensor in
/// <c>ib-ssm/mamba3-370M-10BT</c> is F32, so the loader rehydrates the
/// mmap view directly as <see cref="Mamba3TensorHandle"/>s with
/// <c>OwnsMemory = false</c>. When the checkpoint's alignment happens to
/// be sharper than 64 bytes the loader still accepts it — the aligned
/// allocation path exists for dtype fallbacks (bf16 → f32) in Stage D3+.
/// </para>
/// <para>
/// <b>Structural missing-tensor handling.</b> The loader does NOT throw
/// on a missing tensor. It emits a
/// <see cref="Mamba3TensorIssueKind.Missing"/> diagnostic and leaves the
/// corresponding <see cref="Mamba3TensorHandle"/> as
/// <see cref="Mamba3TensorHandle.Empty"/>.
/// </para>
/// <para>
/// <b>A_log is not loaded.</b> Canonical Mamba-3
/// (<c>state-spaces/mamba</c>) does not store <c>A_log</c>: the decay
/// <c>A</c> is per-token per-head, derived at forward time from
/// <c>dd_A</c> (a slice of <c>in_proj(u)</c>) as
/// <c>-softplus(dd_A)</c> clamped to <c>&lt;= -A_floor</c>. The
/// minimal-reference name <c>A_log</c> used during Stage A-C has no
/// canonical analogue, so the loader no longer probes for it.
/// </para>
/// </remarks>
public static class Mamba3WeightLoader
{
    /// <summary>
    /// Loads every tensor named by <see cref="Mamba3TensorMapping"/> from
    /// <paramref name="file"/> against the shapes implied by
    /// <paramref name="config"/>. Never throws for missing or
    /// shape-mismatched tensors — all findings are attached to the
    /// returned <see cref="Mamba3Weights.Report"/>.
    /// </summary>
    /// <param name="config">A Mamba-3 model config with <see cref="Mamba3Config"/> populated.</param>
    /// <param name="file">An opened safetensors file. Must outlive the returned weights.</param>
    /// <returns>A populated <see cref="Mamba3Weights"/> with structured diagnostics.</returns>
    /// <exception cref="ArgumentException"><paramref name="config"/> is not a Mamba-3 config.</exception>
    public static Mamba3Weights Load(ModelConfig config, ISafetensorsTensorSource file)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(file);

        if (config.Mamba3Config is null)
            throw new ArgumentException(
                "ModelConfig.Mamba3Config must be populated to load Mamba-3 weights.",
                nameof(config));

        var m3 = config.Mamba3Config;
        int numLayers = config.NumLayers;
        int hidden = config.HiddenSize;
        int vocab = config.VocabSize;
        int numHeads = m3.NumHeads;
        int dState = m3.StateSize;
        int dInner = m3.DInner;
        int headDim = m3.HeadDim;
        int dInProj = m3.InputProjectionDim;
        bool isMimo = m3.IsMimo;
        int mimoRank = m3.MimoRank;
        // B_bias / C_bias shape depends on MIMO flag:
        //   SISO: [num_heads, 1, state_size]        (HF 370M convention; middle axis is a
        //                                            singleton broadcast slot).
        //   MIMO: [num_heads, mimo_rank, state_size] (canonical per-rank bias; matches
        //                                            `mamba3_mimo_fwd.py` and capture script
        //                                            `tests/.../Fixtures/Mamba3/capture_fixtures_canonical.py`).
        int bcBiasMiddle = isMimo ? mimoRank : 1;
        int perLayer = isMimo
            ? Mamba3TensorMapping.PerLayerMimoTensorCount
            : Mamba3TensorMapping.PerLayerTensorCount;

        var entries = new List<Mamba3TensorDiagnostic>(3 + numLayers * perLayer);

        // --- globals ---
        var emb = ResolveRequired(
            file, Mamba3TensorMapping.TokenEmbedding, [vocab, hidden], entries);

        var finalNorm = ResolveRequired(
            file, Mamba3TensorMapping.FinalNorm, [hidden], entries);

        Mamba3TensorHandle lmHead;
        if (config.TiedEmbeddings)
        {
            // Per HF convention, if embeddings are tied the safetensors file
            // may legitimately omit lm_head.weight. Alias to the embedding
            // handle so Stage D3 does not have to special-case this.
            if (!file.TensorsByName.ContainsKey(Mamba3TensorMapping.LmHead))
            {
                lmHead = emb with { OwnsMemory = false };
                entries.Add(new Mamba3TensorDiagnostic(
                    Mamba3TensorMapping.LmHead,
                    Mamba3TensorIssueKind.Ok,
                    "tied to token embedding (tie_word_embeddings=true)",
                    IsRequired: false));
            }
            else
            {
                lmHead = ResolveRequired(
                    file, Mamba3TensorMapping.LmHead, [vocab, hidden], entries);
            }
        }
        else
        {
            lmHead = ResolveRequired(
                file, Mamba3TensorMapping.LmHead, [vocab, hidden], entries);
        }

        // --- per-layer ---
        var layers = new Mamba3LayerWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            var norm = ResolveRequired(
                file, Mamba3TensorMapping.LayerNorm(i), [hidden], entries);
            var inProj = ResolveRequired(
                file, Mamba3TensorMapping.InProj(i), [dInProj, hidden], entries);
            var outProj = ResolveRequired(
                file, Mamba3TensorMapping.OutProj(i), [hidden, dInner], entries);
            var bNorm = ResolveRequired(
                file, Mamba3TensorMapping.BNorm(i), [dState], entries);
            var cNorm = ResolveRequired(
                file, Mamba3TensorMapping.CNorm(i), [dState], entries);
            // B_bias / C_bias shape depends on MIMO: SISO uses a [H, 1, N]
            // HF-convention layout with a singleton middle axis (block consumer
            // squeezes it); MIMO uses the canonical rank-expanded [H, R, N].
            var bBias = ResolveRequired(
                file, Mamba3TensorMapping.BBias(i), [numHeads, bcBiasMiddle, dState], entries);
            var cBias = ResolveRequired(
                file, Mamba3TensorMapping.CBias(i), [numHeads, bcBiasMiddle, dState], entries);
            var d = ResolveRequired(
                file, Mamba3TensorMapping.D(i), [numHeads], entries);
            var dtBias = ResolveRequired(
                file, Mamba3TensorMapping.DtBias(i), [numHeads], entries);

            Mamba3TensorHandle mimoX = Mamba3TensorHandle.Empty;
            Mamba3TensorHandle mimoZ = Mamba3TensorHandle.Empty;
            Mamba3TensorHandle mimoO = Mamba3TensorHandle.Empty;
            if (isMimo)
            {
                // Canonical MIMO per-rank weights — shape [H, R, P]. All three
                // are required when is_mimo=true; missing tensors surface via
                // the diagnostics report like any other required tensor.
                mimoX = ResolveRequired(
                    file, Mamba3TensorMapping.MimoX(i), [numHeads, mimoRank, headDim], entries);
                mimoZ = ResolveRequired(
                    file, Mamba3TensorMapping.MimoZ(i), [numHeads, mimoRank, headDim], entries);
                mimoO = ResolveRequired(
                    file, Mamba3TensorMapping.MimoO(i), [numHeads, mimoRank, headDim], entries);
            }

            layers[i] = new Mamba3LayerWeights(
                Norm: norm,
                InProj: inProj,
                OutProj: outProj,
                BNorm: bNorm,
                CNorm: cNorm,
                BBias: bBias,
                CBias: cBias,
                D: d,
                DtBias: dtBias,
                MimoX: mimoX,
                MimoZ: mimoZ,
                MimoO: mimoO);
        }

        // Canonical Mamba-3 has no A_log parameter — A is data-derived from
        // dd_A at forward time (see Mamba3Block.Forward). Stage P2b retired
        // the pre-canonical A_log probe that used to live here.

        int loaded = 0;
        int missingRequired = 0;
        foreach (var e in entries)
        {
            if (e.Kind == Mamba3TensorIssueKind.Ok) loaded++;
            else if (e.IsRequired) missingRequired++;
        }

        return new Mamba3Weights
        {
            TokenEmbedding = emb,
            FinalNorm = finalNorm,
            LmHead = lmHead,
            Layers = layers,
            Report = new Mamba3WeightLoadReport
            {
                Entries = entries,
                LoadedCount = loaded,
                MissingRequiredCount = missingRequired,
            },
        };
    }

    /// <summary>
    /// Resolves a required tensor by name, validating its shape, and
    /// wrapping it as a <see cref="Mamba3TensorHandle"/>. Records a
    /// diagnostic (OK or Missing or ShapeMismatch) into
    /// <paramref name="diagnostics"/> regardless of outcome.
    /// </summary>
    [SkipLocalsInit]
    private static unsafe Mamba3TensorHandle ResolveRequired(
        ISafetensorsTensorSource file,
        string name,
        ReadOnlySpan<int> expectedShape,
        List<Mamba3TensorDiagnostic> diagnostics)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
        {
            diagnostics.Add(new Mamba3TensorDiagnostic(
                name, Mamba3TensorIssueKind.Missing,
                "tensor not found in safetensors header",
                IsRequired: true));
            return Mamba3TensorHandle.Empty;
        }

        if (desc.DType != SafetensorsDType.F32)
        {
            diagnostics.Add(new Mamba3TensorDiagnostic(
                name, Mamba3TensorIssueKind.UnsupportedDType,
                $"dtype {desc.DType} not yet supported by the Mamba-3 F32 ingest path",
                IsRequired: true));
            return Mamba3TensorHandle.Empty;
        }

        if (!ShapesEqual(desc.Shape, expectedShape))
        {
            diagnostics.Add(new Mamba3TensorDiagnostic(
                name, Mamba3TensorIssueKind.ShapeMismatch,
                $"expected shape [{string.Join(',', expectedShape.ToArray())}] but header reports [{string.Join(',', desc.Shape)}]",
                IsRequired: true));
            return Mamba3TensorHandle.Empty;
        }

        nint ptr = file.GetTensorPointer(name);

        // Mmap-backed data inherits whatever alignment the OS grants (page
        // aligned — so 4 KiB / 16 KiB, strictly >= 64). If that ever stops
        // being true we copy into a 64-byte-aligned scratch buffer; the
        // Stage D2 path never exercises this branch but the plumbing is
        // wired for future dtypes.
        if ((((ulong)ptr) & 63UL) != 0)
        {
            long byteCount = desc.ByteCount;
            nint aligned = (nint)NativeMemory.AlignedAlloc((nuint)byteCount, 64);
            Buffer.MemoryCopy((void*)ptr, (void*)aligned, byteCount, byteCount);
            diagnostics.Add(new Mamba3TensorDiagnostic(
                name, Mamba3TensorIssueKind.Ok,
                "copied into 64-byte-aligned scratch (mmap pointer was under-aligned)",
                IsRequired: true));
            return new Mamba3TensorHandle(aligned, desc.Shape, desc.DType, OwnsMemory: true);
        }

        diagnostics.Add(new Mamba3TensorDiagnostic(
            name, Mamba3TensorIssueKind.Ok, "loaded", IsRequired: true));
        return new Mamba3TensorHandle(ptr, desc.Shape, desc.DType, OwnsMemory: false);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool ShapesEqual(int[] actual, ReadOnlySpan<int> expected)
    {
        if (actual.Length != expected.Length) return false;
        for (int i = 0; i < actual.Length; i++)
            if (actual[i] != expected[i]) return false;
        return true;
    }
}
