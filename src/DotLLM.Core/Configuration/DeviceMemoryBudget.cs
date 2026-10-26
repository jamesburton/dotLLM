using System.Collections.Concurrent;

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
    /// <summary>
    /// Default free VRAM a growth must leave untouched: cuBLAS workspace, transient allocations, another process's growth.
    /// Measured on an RTX 3060 12 GB under WDDM (issue #1068): CUDA's free figure equals the DXGI per-process budget minus
    /// usage, and shared-memory spill begins ~280 MiB past that budget, so this reserve sits on top of that slack.
    /// </summary>
    public const long DefaultReserveBytes = 256L << 20;

    /// <summary>Environment variable overriding <see cref="DefaultReserveBytes"/>, in MiB.</summary>
    public const string ReserveEnvVar = "DOTLLM_CUDA_VRAM_RESERVE_MB";

    /// <summary>Granularity the fitted context is rounded down to.</summary>
    public const int ContextGranularity = 1024;

    /// <summary>
    /// Environment variable (issue #1295) that simulates a smaller GPU: every planner that reads free or total device memory
    /// (CUDA <c>cuMemGetInfo</c> / NVML, Vulkan heap capacity and the device-local allocation limit) sees at most this many MiB,
    /// so a large card exercises the small-card offload paths. Test knob; unset or non-positive disables it.
    /// </summary>
    public const string SimulatedBudgetEnvVar = "DOTLLM_VRAM_BUDGET_MB";

    private static readonly ConcurrentDictionary<int, long> s_simulatedBaselineFree = new();
    private static long s_loggedSimulatedBudget = -1;

    /// <summary>
    /// The simulated device memory budget from <see cref="SimulatedBudgetEnvVar"/>, in bytes, or <see langword="null"/> when
    /// unset. Read on every call (planning time only, never the hot path) so a test can toggle it; logged once per value.
    /// </summary>
    /// <returns>The budget in bytes, or <see langword="null"/>.</returns>
    public static long? SimulatedBudgetBytes()
    {
        if (!long.TryParse(Environment.GetEnvironmentVariable(SimulatedBudgetEnvVar), out long mib) || mib <= 0) return null;
        long bytes = mib << 20;
        if (Interlocked.Exchange(ref s_loggedSimulatedBudget, bytes) != bytes)
            Console.Error.WriteLine($"[dotLLM] {SimulatedBudgetEnvVar}={mib}: device memory planners see a simulated {mib} MiB GPU.");
        return bytes;
    }

    /// <summary>
    /// Records <paramref name="freeBytes"/> as the free memory of <paramref name="device"/> before this process allocates on it
    /// (call right after context creation). Later <see cref="ClampSimulatedFree"/> calls charge everything allocated since
    /// against the simulated budget. No-op while the budget is off.
    /// </summary>
    /// <param name="device">Device ordinal.</param>
    /// <param name="freeBytes">Measured free bytes now.</param>
    public static void ResetSimulatedBaseline(int device, long freeBytes)
    {
        // shortcut: one baseline per device, re-taken at each context creation; two models loaded side by side on one device
        // share it only from the second load on. Fine for a test knob; track per-context usage if it ever matters.
        if (SimulatedBudgetBytes() is not null) s_simulatedBaselineFree[device] = freeBytes;
    }

    /// <summary>
    /// Measured free device memory as a planner should see it under <see cref="SimulatedBudgetEnvVar"/>: the budget minus what
    /// this process has allocated since the baseline, never more than really is free. Returns <paramref name="freeBytes"/>
    /// unchanged when the budget is off.
    /// </summary>
    /// <param name="device">Device ordinal.</param>
    /// <param name="freeBytes">Measured free bytes.</param>
    /// <returns>The free bytes to plan with.</returns>
    public static long ClampSimulatedFree(int device, long freeBytes)
        => SimulatedBudgetBytes() is { } budget
            ? SimulatedFree(budget, s_simulatedBaselineFree.GetOrAdd(device, freeBytes), freeBytes)
            : freeBytes;

    /// <summary>
    /// Device capacity (heap or total size) as a planner should see it: at most the <see cref="SimulatedBudgetEnvVar"/> budget.
    /// </summary>
    /// <param name="capacityBytes">Measured capacity.</param>
    /// <returns>The capacity to plan with.</returns>
    public static long ClampSimulatedCapacity(long capacityBytes)
        => SimulatedBudgetBytes() is { } budget ? Math.Min(capacityBytes, budget) : capacityBytes;

    /// <summary>
    /// Pure arithmetic of <see cref="ClampSimulatedFree"/>: <c>clamp(budget − (baseline − free), 0, free)</c>.
    /// </summary>
    /// <param name="budgetBytes">Simulated budget.</param>
    /// <param name="baselineFreeBytes">Free bytes before this process allocated.</param>
    /// <param name="freeBytes">Free bytes now.</param>
    /// <returns>The simulated free bytes.</returns>
    public static long SimulatedFree(long budgetBytes, long baselineFreeBytes, long freeBytes)
        => Math.Clamp(budgetBytes - Math.Max(0, baselineFreeBytes - freeBytes), 0, Math.Max(0, freeBytes));

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
