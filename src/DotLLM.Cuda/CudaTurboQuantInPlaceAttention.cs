namespace DotLLM.Cuda;

/// <summary>
/// Attention for one layer over a <see cref="CudaTurboQuantInPlaceKvCache"/>, read in place (#1892 step 2, Task 6). This is the one
/// sequence the dense <see cref="CudaTransformerModel"/> forward runs for that cache, kept here so kernel-level tests exercise the
/// exact same code:
/// <list type="number">
/// <item><b>Encode</b> the fresh K / V rows into the cache's codes and norms (<see cref="CudaTurboQuantInPlaceKvCache.UpdateDevice"/>).</item>
/// <item><b>Rotate Q</b> into K code space in place with the <b>K</b> signs: <c>q~ = H(S_k . q)/sqrt(d)</c>.</item>
/// <item><b>Attend</b> over rows <c>[0, CurrentLength)</c> of the cache only, current chunk included, in code space: the tensor-core
/// prefill kernel when <see cref="CudaTransformerModel.UseKvqFlashPrefill"/> allows (head dim 64 / 128, at least 16 rows, the newest
/// rows, no sliding window or sinks, group &lt;= 8), else the scalar <c>attention_kvtq</c> (decode, short prefill, sliding window,
/// sinks, head dim 256 or any other power of two, larger groups).</item>
/// <item><b>Rotate O</b> back out of V code space in place with the <b>V</b> signs: <c>o = S_v . H(a~)/sqrt(d)</c> (F16).</item>
/// </list>
/// The fresh K / V buffers are read only by the encode: attention never sees them. Nothing is allocated, and there is no fallback
/// to a decoded history: a missing kernel throws.
/// </summary>
internal static class CudaTurboQuantInPlaceAttention
{
    private static int s_scalarPrefillLogged;

    /// <summary>Throws unless every kernel the in-place TurboQuant forward needs is loaded (no silent dequant fallback exists).</summary>
    internal static void EnsureKernels(CudaKernels kernels)
    {
        if (!kernels.TurboQuantF16Available)
            throw new NotSupportedException("In-place TurboQuant KV needs turboquant.ptx (encode); it is missing or stale (#1892).");
        if (!kernels.TqRotateAvailable || !kernels.TqRotateOF16Available)
            throw new NotSupportedException("In-place TurboQuant KV needs turboquant_rotate.ptx (Q/O rotations); it is missing or stale (#1892).");
        if (!kernels.HasAttentionKvTq)
            throw new NotSupportedException("In-place TurboQuant KV needs attention_kvtq.ptx; it is missing or stale (#1892).");
    }

    /// <summary>
    /// Runs encode, Q rotation, attention and O rotation for one layer (see the class remarks). <paramref name="q"/> is F16
    /// <c>[seqLen, numHeads*headDim]</c> and is <b>overwritten</b> with the rotated query; <paramref name="output"/> receives the F16
    /// attention output <c>[seqLen, numHeads*headDim]</c> in the original space. Returns whether the tensor-core arm ran.
    /// </summary>
    internal static bool Run(CudaKernels kernels, CudaTurboQuantInPlaceKvCache cache, int layer,
        nint q, nint kFresh, nint vFresh, ReadOnlySpan<int> positions, int seqLen,
        int numHeads, int numKvHeads, int headDim, int slidingWindow, nint sinks, nint output, nint stream,
        bool flashEnabled)
    {
        EnsureKernels(kernels);
        if (headDim != cache.Layout.HeadDim || numKvHeads != cache.Layout.NumKvHeads)
            throw new ArgumentException($"Attention geometry {numKvHeads}x{headDim} does not match the cache's " +
                                        $"{cache.Layout.NumKvHeads}x{cache.Layout.HeadDim}.");

        // 1. Encode the current rows. Attention below reads them back from the codes, like every other history row.
        cache.UpdateDevice(kFresh, vFresh, positions, seqLen, layer, stream, kernels);
        int seqKv = cache.CurrentLength;
        int positionOffset = positions[0];

        // 2. Query into K code space (K signs).
        kernels.LaunchTqRotateQ(q, seqLen, numHeads, headDim, cache.SignsKPtr, stream);

        // 3. Attention over the cache, in code space.
        int bits = cache.Layout.MseBits;
        int cu = cache.CodeUintsPerVec;
        bool flash = CudaTransformerModel.UseKvqFlashPrefill(flashEnabled, kernels.HasAttentionFlashPrefillKvTq(headDim),
            seqLen, seqKv, positionOffset, slidingWindow, sinks != 0, window: 0, numHeads, numKvHeads, headDim);
        if (flash)
        {
            kernels.LaunchAttentionFlashPrefillKvTq(q, cache.GetKeyCodesPtr(layer), cache.GetValueCodesPtr(layer),
                cache.GetKeyNormsPtr(layer), cache.GetValueNormsPtr(layer), cache.CentroidsPtr, output,
                seqLen, seqKv, numHeads, numKvHeads, headDim, positionOffset, bits, cu, cache.NormFormat, stream);
        }
        else
        {
            if (seqLen >= CudaTransformerModel.KvqFlashPrefillMinRows && Interlocked.Exchange(ref s_scalarPrefillLogged, 1) == 0)
                Console.Error.WriteLine(
                    $"[dotLLM.Cuda] TurboQuant tq{bits} KV prefill runs the scalar in-place attention_kvtq kernel (head_dim {headDim}, " +
                    $"{numHeads}/{numKvHeads} heads, sliding {slidingWindow}, sinks {sinks != 0}, tensor-core kernel loaded " +
                    $"{kernels.HasAttentionFlashPrefillKvTq(headDim)}): correct and allocation-free, but slower than the tensor-core prefill.");
            kernels.LaunchAttentionKvTq(q, cache.GetKeyCodesPtr(layer), cache.GetValueCodesPtr(layer),
                cache.GetKeyNormsPtr(layer), cache.GetValueNormsPtr(layer), cache.CentroidsPtr, output,
                seqLen, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow, stream, sinks,
                bits, cu, cache.NormFormat);
        }

        // 4. Output back out of V code space (V signs).
        kernels.LaunchTqRotateOF16(output, seqLen, numHeads, headDim, cache.SignsVPtr, stream);
        return flash;
    }
}
