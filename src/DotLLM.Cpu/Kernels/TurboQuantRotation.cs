namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Q/O rotations that let attention run in TurboQuant code space (#1892). A TurboQuant side stores
/// <c>x ~ n · R⁻¹ c</c> with <c>R = H·D/√d</c> (<c>D</c> the ±1 sign diagonal, <c>H</c> the unnormalised Walsh-Hadamard
/// transform) and <c>c</c> the centroid vector, so with <c>R</c> orthogonal:
/// <list type="bullet">
/// <item><description>score: <c>q · x = n · (R q) · c</c>, so the query is rotated once (<c>q̃ = H(S_k ⊙ q)/√d</c>) and then
/// dotted against centroids with no per-key inverse rotation;</description></item>
/// <item><description>output: <c>Σ p_j x_j = R⁻¹ (Σ p_j n_j c_j)</c>, so the accumulated code-space vector <c>ã</c> is rotated back once:
/// <c>o = S_v ⊙ H(ã)/√d</c>.</description></item>
/// </list>
/// Both wrap <see cref="Hadamard"/> with <c>blockSize = headDim</c> (one block per head vector). The K side uses the codec seed
/// and the V side <c>seed ^ 0xD1B54A32D192ED03</c> (see <c>TurboQuantKvCache</c>): pass the matching signs.
/// </summary>
public static unsafe class TurboQuantRotation
{
    /// <summary>
    /// Converts a codec's ±1 float sign vector to the <see cref="sbyte"/> form <see cref="Hadamard"/> takes. Do this once when
    /// the cache is built, not per call.
    /// </summary>
    /// <param name="signs">The codec's <c>RotationSigns</c> (±1.0f).</param>
    /// <returns>The same signs as ±1 bytes.</returns>
    public static sbyte[] SignsToSbyte(ReadOnlySpan<float> signs)
    {
        var r = new sbyte[signs.Length];
        for (int i = 0; i < r.Length; i++) r[i] = signs[i] < 0f ? (sbyte)-1 : (sbyte)1;
        return r;
    }

    /// <summary>Rotates the query into K code space in place: <c>q̃ = H(S_k ⊙ q)/√d</c> per (row, head).</summary>
    /// <param name="q">Queries, <c>[rows, numHeads, headDim]</c> contiguous F32.</param>
    /// <param name="rows">Number of query rows.</param>
    /// <param name="numHeads">Query heads per row.</param>
    /// <param name="headDim">Head dimension (a power of two).</param>
    /// <param name="signsK">The K-side ±1 signs, length <paramref name="headDim"/>.</param>
    public static void RotateQueryInPlace(float* q, int rows, int numHeads, int headDim, ReadOnlySpan<sbyte> signsK)
    {
        Validate(rows, numHeads, headDim, signsK);
        for (int v = 0; v < rows * numHeads; v++)
        {
            var row = new Span<float>(q + (long)v * headDim, headDim);
            Hadamard.ForwardRow(row, signsK, row, headDim);
        }
    }

    /// <summary>Rotates the code-space attention output back in place: <c>o = S_v ⊙ H(ã)/√d</c> per (row, head).</summary>
    /// <param name="o">Attention output, <c>[rows, numHeads, headDim]</c> contiguous F32.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="numHeads">Heads per row.</param>
    /// <param name="headDim">Head dimension (a power of two).</param>
    /// <param name="signsV">The V-side ±1 signs, length <paramref name="headDim"/>.</param>
    public static void RotateOutputInPlace(float* o, int rows, int numHeads, int headDim, ReadOnlySpan<sbyte> signsV)
    {
        Validate(rows, numHeads, headDim, signsV);
        for (int v = 0; v < rows * numHeads; v++)
        {
            var row = new Span<float>(o + (long)v * headDim, headDim);
            Hadamard.InverseRow(row, signsV, row, headDim);
        }
    }

    private static void Validate(int rows, int numHeads, int headDim, ReadOnlySpan<sbyte> signs)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0)
            throw new ArgumentException($"headDim must be a power of two; got {headDim}.", nameof(headDim));
        if (signs.Length != headDim)
            throw new ArgumentException($"signs length {signs.Length} != headDim {headDim}.", nameof(signs));
    }
}
