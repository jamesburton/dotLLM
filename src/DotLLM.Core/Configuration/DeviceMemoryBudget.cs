namespace DotLLM.Core.Configuration;

/// <summary>
/// Admission arithmetic for device buffers that grow with the context (issue #1068). On Windows the
/// WDDM driver does not fail an allocation that exceeds dedicated VRAM: it backs it with shared system
/// memory, and every kernel that touches it then reads over PCIe (~3 GB/s on a Gen3 x4 eGPU link, against
/// ~360 GB/s from VRAM). A request that should take minutes then takes hours. So a growth is admitted only
/// when it fits in free VRAM with <see cref="ResolveReserveBytes"/> left over; otherwise the caller refuses.
/// Pure and GPU-free, so it is unit-testable.
/// </summary>
public static class DeviceMemoryBudget
{
    /// <summary>Default free VRAM a growth must leave untouched: cuBLAS workspace, the display, other processes.</summary>
    public const long DefaultReserveBytes = 512L << 20;

    /// <summary>Environment variable overriding <see cref="DefaultReserveBytes"/>, in MiB.</summary>
    public const string ReserveEnvVar = "DOTLLM_CUDA_VRAM_RESERVE_MB";

    /// <summary>Granularity the fitted context is rounded down to.</summary>
    public const int ContextGranularity = 1024;

    /// <summary>
    /// Parses <see cref="ReserveEnvVar"/>'s value: a non-negative MiB count, else <see cref="DefaultReserveBytes"/>.
    /// </summary>
    /// <param name="envValue">The raw environment value, or null.</param>
    /// <returns>The reserve in bytes.</returns>
    public static long ResolveReserveBytes(string? envValue)
        => long.TryParse(envValue, out long mib) && mib >= 0 ? mib << 20 : DefaultReserveBytes;

    /// <summary>
    /// True when <paramref name="requiredBytes"/> more device memory can be allocated while leaving
    /// <paramref name="reserveBytes"/> of <paramref name="freeBytes"/> free.
    /// </summary>
    /// <param name="freeBytes">Free device memory now (after any free-first release).</param>
    /// <param name="requiredBytes">Bytes the growth allocates.</param>
    /// <param name="reserveBytes">Bytes that must stay free.</param>
    /// <returns><see langword="true"/> when the growth is admitted.</returns>
    public static bool Fits(long freeBytes, long requiredBytes, long reserveBytes)
        => requiredBytes <= 0 || freeBytes - reserveBytes >= requiredBytes;

    /// <summary>
    /// The user-facing refusal for a growth that does not fit. Names what grew, the numbers, and the remedies.
    /// </summary>
    /// <param name="what">What the allocation is for, e.g. "KV cache (q8_0) for 98304 tokens".</param>
    /// <param name="requiredBytes">Bytes the growth needs.</param>
    /// <param name="freeBytes">Free device memory.</param>
    /// <param name="reserveBytes">Bytes kept free.</param>
    /// <returns>The message.</returns>
    public static string DescribeRefusal(string what, long requiredBytes, long freeBytes, long reserveBytes)
        => $"Not enough free GPU memory for the {what}: it needs {Mib(requiredBytes)} MiB, {Mib(freeBytes)} MiB is free " +
           $"and {Mib(reserveBytes)} MiB is kept in reserve. Refusing rather than letting the driver spill into shared " +
           "system memory, which makes generation orders of magnitude slower. Shorten the conversation, lower the " +
           $"context with 'dotllm serve --ctx-size', use a smaller KV type (--kv-quant-k/-v q8_0 or q4_0), or free GPU memory " +
           $"({ReserveEnvVar} sets the reserve).";

    /// <summary>
    /// The longest context whose per-token device memory fits: <c>(free − reserve − fixed) / perToken</c>, rounded down to
    /// <see cref="ContextGranularity"/>, capped at <paramref name="modelMaxContext"/>. Returns 0 when not even one
    /// granule fits.
    /// </summary>
    /// <param name="freeBytes">Free device memory with the model loaded.</param>
    /// <param name="reserveBytes">Bytes kept free.</param>
    /// <param name="fixedBytes">Context-independent buffers not yet allocated (snapshots, checkpoints, forward scratch).</param>
    /// <param name="bytesPerToken">Device bytes each context token costs at its peak (KV plus context-sized scratch).</param>
    /// <param name="modelMaxContext">The model's own context window.</param>
    /// <returns>The fitted context, in tokens.</returns>
    public static int MaxFittingContext(long freeBytes, long reserveBytes, long fixedBytes, long bytesPerToken, int modelMaxContext)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bytesPerToken);
        long budget = freeBytes - reserveBytes - fixedBytes;
        if (budget <= 0) return 0;
        long tokens = budget / bytesPerToken;
        tokens -= tokens % ContextGranularity;
        return (int)Math.Min(tokens, modelMaxContext);
    }

    private static long Mib(long bytes) => (bytes + (1L << 20) - 1) >> 20;
}
