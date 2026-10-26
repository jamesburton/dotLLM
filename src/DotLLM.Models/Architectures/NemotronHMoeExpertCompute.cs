using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;

namespace DotLLM.Models.Architectures;

/// <summary>
/// One routed <c>nemotron_h_moe</c> expert on the CPU (issue #1071): <c>y[n, expertDim] = down(relu²(up(x[n, expertDim])))</c>
/// from the expert's slice of the memory-mapped banks (<c>bank + e · perExpertBytes</c>), for every bank type the CPU model reads
/// (Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS, IQ4_XS, IQ4_NL, NVFP4, ...). It runs <see cref="NemotronHTransformerModel.GemmDispatch"/>, the
/// model's own GEMM dispatcher, so the CPU model's per-expert path and the CUDA offload's CPU cold leg share one code path.
/// </summary>
/// <remarks>
/// Thread-pool free when <c>pool</c> is null: the CUDA cold leg computes several experts in parallel, one per worker, each
/// single-threaded (one expert is too small to split across threads, #1063 section 6). The kernels' scratch comes from
/// <see cref="System.Buffers.ArrayPool{T}"/>, so concurrent calls are safe. Expert banks carry no <c>X.scale</c> tensor (the
/// per-tensor scale of NVFP4 SSM projections), so no scale is applied.
/// </remarks>
internal static unsafe class NemotronHMoeExpertCompute
{
    /// <summary>Width the routed experts of <paramref name="moe"/> read and write: the latent, or the hidden size when there is none.</summary>
    /// <param name="moe">Layer weights.</param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <returns>The expert width.</returns>
    public static int ExpertDim(NemotronHMoeWeights moe, int hiddenSize) => moe.HasLatent ? moe.LatentSize : hiddenSize;

    /// <summary>Host pointer of expert <paramref name="expert"/>'s up matrix.</summary>
    /// <param name="moe">Layer weights.</param>
    /// <param name="expert">Expert index.</param>
    /// <returns>The pointer.</returns>
    public static nint UpPointer(NemotronHMoeWeights moe, int expert) => moe.UpExpsWeight + (nint)(expert * moe.UpPerExpertBytes);

    /// <summary>Host pointer of expert <paramref name="expert"/>'s down matrix.</summary>
    /// <param name="moe">Layer weights.</param>
    /// <param name="expert">Expert index.</param>
    /// <returns>The pointer.</returns>
    public static nint DownPointer(NemotronHMoeWeights moe, int expert) => moe.DownExpsWeight + (nint)(expert * moe.DownPerExpertBytes);

    /// <summary>
    /// Computes expert <paramref name="expert"/> of <paramref name="moe"/> for <paramref name="n"/> rows: up, relu² in place, down.
    /// </summary>
    /// <param name="moe">Layer weights (the banks are read in place).</param>
    /// <param name="expert">Expert index.</param>
    /// <param name="expertDim">Expert width (<see cref="ExpertDim"/>).</param>
    /// <param name="x">Input rows <c>[n, expertDim]</c>.</param>
    /// <param name="n">Rows.</param>
    /// <param name="mid">Scratch <c>[n, MoeIntermediateSize]</c> (holds relu²(up(x)) afterwards).</param>
    /// <param name="y">Output rows <c>[n, expertDim]</c>; must not alias <paramref name="x"/> or <paramref name="mid"/>.</param>
    /// <param name="pool">Compute pool, or null for single-threaded (the result is bit-identical either way).</param>
    /// <exception cref="ArgumentOutOfRangeException">The expert index is out of range or <paramref name="n"/> is not positive.</exception>
    public static void Compute(NemotronHMoeWeights moe, int expert, int expertDim, float* x, int n, float* mid, float* y,
        ComputeThreadPool? pool = null)
    {
        if ((uint)expert >= (uint)moe.NumExperts) throw new ArgumentOutOfRangeException(nameof(expert));
        ArgumentOutOfRangeException.ThrowIfLessThan(n, 1);
        int inter = moe.MoeIntermediateSize;
        NemotronHTransformerModel.GemmDispatch(UpPointer(moe, expert), moe.UpExpsQuantType, x, mid, inter, expertDim, n, pool, null);
        var m = new Span<float>(mid, n * inter);
        ReluSquared.Execute(m, m);
        NemotronHTransformerModel.GemmDispatch(DownPointer(moe, expert), moe.DownExpsQuantType, mid, y, expertDim, inter, n, pool, null);
    }

    /// <summary>
    /// Touches every page of expert <paramref name="expert"/>'s up and down matrices, so a timed run afterwards does not pay the
    /// first-touch page faults of the memory-mapped file (calibration, #1071).
    /// </summary>
    /// <param name="moe">Layer weights.</param>
    /// <param name="expert">Expert index.</param>
    /// <returns>A checksum (keeps the reads from being optimised away).</returns>
    public static long Touch(NemotronHMoeWeights moe, int expert)
    {
        long sum = 0;
        sum += TouchRange((byte*)UpPointer(moe, expert), moe.UpPerExpertBytes);
        sum += TouchRange((byte*)DownPointer(moe, expert), moe.DownPerExpertBytes);
        return sum;

        static long TouchRange(byte* p, long bytes)
        {
            long s = 0;
            for (long i = 0; i < bytes; i += 4096) s += p[i];
            if (bytes > 0) s += p[bytes - 1];
            return s;
        }
    }
}
