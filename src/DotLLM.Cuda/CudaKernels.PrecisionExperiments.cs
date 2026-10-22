using DotLLM.Core.Configuration;

namespace DotLLM.Cuda;

/// <summary>
/// Issue #1109 precision-study kernels: the F16-accumulating flash prefill variants of
/// <c>attention_flash_mma_prefill_hd256_xacc.ptx</c>. Loaded lazily on first use, so a process that never
/// selects an experiment never loads the module.
/// </summary>
public sealed unsafe partial class CudaKernels
{
    private const string FlashXaccPtx = "attention_flash_mma_prefill_hd256_xacc.ptx";

    private CudaModule? _flashXaccModule;
    private bool _flashXaccLoadAttempted;
    private readonly Dictionary<string, nint> _flashXaccFuncs = new(StringComparer.Ordinal);

    /// <summary>
    /// Launches the F16-accumulating flash prefill variant <paramref name="mode"/> (issue #1109). Same arguments,
    /// grid, block and shared memory as <see cref="LaunchAttentionFlashPrefillHd256"/>; with
    /// <paramref name="kvq"/> the K/V pointers are raw Q4_0 blocks as for <see cref="LaunchAttentionFlashPrefillHd256Kvq"/>.
    /// </summary>
    /// <param name="mode">Accumulator variant; not <see cref="FlashAccumulatorMode.F32"/>.</param>
    /// <param name="kvq">True for a Q4_0 K + Q4_0 V cache, false for the F16 cache.</param>
    /// <param name="q">F16 queries <c>[seqQ, numHeads*256]</c>.</param>
    /// <param name="k">K history (F16, or raw Q4_0 blocks).</param>
    /// <param name="v">V history (F16, or raw Q4_0 blocks).</param>
    /// <param name="output">F32 output <c>[seqQ, numHeads*256]</c>.</param>
    /// <param name="seqQ">Query rows.</param>
    /// <param name="seqKv">KV rows (history including the query rows).</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="positionOffset">Absolute position of query row 0.</param>
    /// <param name="stream">CUDA stream.</param>
    /// <exception cref="InvalidOperationException">The experiment PTX or the requested entry is missing.</exception>
    internal void LaunchAttentionFlashPrefillHd256Xacc(FlashAccumulatorMode mode, bool kvq, nint q, nint k, nint v,
        nint output, int seqQ, int seqKv, int numHeads, int numKvHeads, int positionOffset, nint stream)
    {
        string entry = (kvq ? "attention_flash_mma_prefill_hd256_kvq_q4q4_" : "attention_flash_mma_prefill_hd256_f16_")
                       + CudaFlashAccumulatorExperiment.Suffix(mode);
        nint func = ResolveFlashXacc(entry);
        LaunchAttentionFlashPrefillHd256Core(func, q, k, v, output, seqQ, seqKv, numHeads, numKvHeads,
            positionOffset, stream, scale: 0f);
    }

    private nint ResolveFlashXacc(string entry)
    {
        if (_flashXaccFuncs.TryGetValue(entry, out nint f)) return f;
        if (!_flashXaccLoadAttempted)
        {
            _flashXaccLoadAttempted = true;
            string path = Path.Combine(PtxDirectory, FlashXaccPtx);
            if (File.Exists(path)) _flashXaccModule = CudaModule.LoadFromFile(path);
        }
        if (_flashXaccModule is null)
            throw new InvalidOperationException(
                $"{CudaFlashAccumulatorExperiment.EnvVar} selects an F16-accumulate flash variant but {FlashXaccPtx} is not in {PtxDirectory}.");
        f = _flashXaccModule.TryGetFunction(entry);
        if (f == 0) throw new InvalidOperationException($"{FlashXaccPtx} has no entry {entry}.");
        SetMaxDynamicSharedBytes(f, AttentionFlashPrefillHd256MaxSmemBytes);
        _flashXaccFuncs[entry] = f;
        return f;
    }

    /// <summary>Whether a quantized-KV pair has an F16-accumulate experiment entry (Q4_0 K + Q4_0 V only).</summary>
    /// <param name="k">K cache format.</param>
    /// <param name="v">V cache format.</param>
    /// <returns><see langword="true"/> for Q4_0/Q4_0.</returns>
    internal static bool FlashXaccSupportsKvq(KvCacheDType k, KvCacheDType v)
        => k == KvCacheDType.Q4_0 && v == KvCacheDType.Q4_0;

    private void DisposePrecisionExperimentModules()
    {
        _flashXaccModule?.Dispose();
        _flashXaccModule = null;
        _flashXaccFuncs.Clear();
    }
}
