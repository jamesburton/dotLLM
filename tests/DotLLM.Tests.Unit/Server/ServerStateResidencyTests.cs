using DotLLM.Engine;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// End-to-end (real-weight, CPU-only) tests for the model-residency lifecycle added by #369:
/// multi-model residency, idle-timeout auto-unload with lazy reload, budget-driven LRU eviction,
/// and a regression check that the original single-model hot-swap path is unchanged by default.
/// Uses the smallest cached GGUF quants available (SmolLM-135M-Instruct, two different quants
/// standing in for two distinct models — their <see cref="ServerOptions.ModelId"/> differs by
/// filename, which is all the residency machinery keys on). Warm-up and paged KV-cache are
/// disabled to keep these fast and dependency-free; skipped automatically when the fixture GGUFs
/// aren't present in the local Hugging Face hub cache (e.g. a fresh clone/CI without the model
/// downloaded), matching the existing skip pattern used by other real-weight tests in this repo.
/// </summary>
public sealed class ServerStateResidencyTests : IDisposable
{
    private static readonly string? PathA = FindCachedGguf("Q5_K_M");
    private static readonly string? PathB = FindCachedGguf("Q3_K_M");

    private readonly List<ServerState> _states = new();

    public void Dispose()
    {
        foreach (var s in _states)
        {
            try { s.Dispose(); } catch { /* best-effort */ }
        }
    }

    private ServerState NewBareState(int maxResident = 1, long memoryBudgetBytes = 0, double defaultKeepAliveSeconds = 300)
    {
        var state = new ServerState
        {
            Options = new ServerOptions
            {
                Model = "",
                ModelId = "none",
                Device = "cpu",
                UsePaged = false,
                PromptCacheEnabled = false,
                Warmup = WarmupOptions.Disabled,
            },
            Residency = new ModelResidencyManager
            {
                MaxResidentModels = maxResident,
                MemoryBudgetBytes = memoryBudgetBytes,
                DefaultKeepAliveSeconds = defaultKeepAliveSeconds,
            },
        };
        _states.Add(state);
        return state;
    }

    private static string? FindCachedGguf(string quantMarker)
    {
        string hub = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub");
        if (!Directory.Exists(hub)) return null;

        return Directory.EnumerateFiles(hub, "*.gguf", SearchOption.AllDirectories)
            .FirstOrDefault(f =>
                f.Contains("SmolLM-135M", StringComparison.OrdinalIgnoreCase) &&
                f.Contains(quantMarker, StringComparison.OrdinalIgnoreCase) &&
                File.Exists(f));
    }

    private static bool FixturesAvailable => PathA is not null && PathB is not null;

    [SkippableFact]
    public async Task MultiModelResidency_LoadingSecondModel_KeepsFirstResidentAndReactivatableWithoutReload()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");

        var state = NewBareState(maxResident: 2);
        var ct = CancellationToken.None;

        var errA = await state.EnsureActiveAsync(PathA, keepAliveOverride: null, ct);
        Assert.Null(errA);
        string idA = state.Options.ModelId;
        Assert.True(state.IsReady);
        Assert.NotNull(state.Model);

        var errB = await state.EnsureActiveAsync(PathB, keepAliveOverride: null, ct);
        Assert.Null(errB);
        string idB = state.Options.ModelId;
        Assert.NotEqual(idA, idB);
        Assert.True(state.IsReady);

        // Model A must still be resident (stashed), not disposed - acceptance criterion #1.
        Assert.Equal(1, state.Residency.StashedCount);
        Assert.True(state.Residency.Contains(idA));

        // Reactivating A should be a cheap field-swap (no ServerStartup.LoadModel call) - verified
        // indirectly by success + correct bookkeeping; B becomes the stashed one.
        var errReactivate = await state.EnsureActiveAsync(idA, keepAliveOverride: null, ct);
        Assert.Null(errReactivate);
        Assert.Equal(idA, state.Options.ModelId);
        Assert.True(state.IsReady);
        Assert.NotNull(state.Model);
        Assert.True(state.Residency.Contains(idB));
        Assert.False(state.Residency.Contains(idA));
    }

    [SkippableFact]
    public async Task IdleUnload_PastKeepAlive_UnloadsActiveModel_NextRequestLazilyReloads()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");

        var state = NewBareState(maxResident: 1, defaultKeepAliveSeconds: 0.05); // 50ms
        var ct = CancellationToken.None;

        var err = await state.EnsureActiveAsync(PathA, keepAliveOverride: null, ct);
        Assert.Null(err);
        string modelId = state.Options.ModelId;
        string loadedPath = state.LoadedModelPath;
        Assert.True(state.IsReady);

        await Task.Delay(200); // past the 50ms keep-alive

        await state.SweepIdleAsync();

        Assert.False(state.IsReady);
        Assert.Null(state.Model);
        // Options/LoadedModelPath survive the idle-unload so a later request can lazily reload.
        Assert.Equal(modelId, state.Options.ModelId);
        Assert.Equal(loadedPath, state.LoadedModelPath);

        // "a request against it triggers reload as expected" (acceptance criterion #2).
        var reloadErr = await state.EnsureActiveAsync(requestedModel: null, keepAliveOverride: null, ct);
        Assert.Null(reloadErr);
        Assert.True(state.IsReady);
        Assert.NotNull(state.Model);
        Assert.Equal(modelId, state.Options.ModelId);
    }

    [SkippableFact]
    public async Task IdleUnload_NeverExpiresWhenKeepAliveNegative()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");

        var state = NewBareState(maxResident: 1, defaultKeepAliveSeconds: 0.05);
        var ct = CancellationToken.None;

        var err = await state.EnsureActiveAsync(PathA, keepAliveOverride: -1, ct); // pin: never unload
        Assert.Null(err);

        await Task.Delay(200);
        await state.SweepIdleAsync();

        Assert.True(state.IsReady);
        Assert.NotNull(state.Model);
    }

    [SkippableFact]
    public async Task EvictionUnderBudgetPressure_EvictsLruStashedModelWhenBudgetExceeded()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");

        // ServerStartup.SafeFileLength (not raw FileInfo.Length) - Hugging Face hub cache entries
        // are NTFS symlinks on Windows, and FileInfo.Length reports 0 for the reparse point itself
        // rather than the target's size (see docs/perf/ISSUE_369_MODEL_KEEPALIVE.md).
        long sizeA = ServerStartup.SafeFileLength(PathA!);
        Assert.True(sizeA > 0, "sanity: resolved GGUF size must be nonzero");

        // Budget only fits one model at a time, even though MaxResidentModels allows more.
        var state = NewBareState(maxResident: 3, memoryBudgetBytes: sizeA + sizeA / 10);
        var ct = CancellationToken.None;

        var errA = await state.EnsureActiveAsync(PathA, keepAliveOverride: null, ct);
        Assert.Null(errA);
        string idA = state.Options.ModelId;

        var errB = await state.EnsureActiveAsync(PathB, keepAliveOverride: null, ct);
        Assert.Null(errB);
        string idB = state.Options.ModelId;

        // A was stashed then immediately evicted by the byte budget - count stays bounded even
        // though MaxResidentModels(3) would otherwise have allowed keeping it.
        Assert.Equal(0, state.Residency.StashedCount);
        Assert.False(state.Residency.Contains(idA));
        Assert.Equal(idB, state.Options.ModelId);
    }

    [SkippableFact]
    public async Task SingleModelHotSwap_DefaultConfiguration_RegressionUnchanged()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");

        // Default MaxResidentModels = 1 - the exact pre-#369 configuration. Exercises the same
        // SwapModelAsync path ModelManagementEndpoint's POST /v1/models/load uses.
        var state = NewBareState(); // maxResident defaults to 1
        var ct = CancellationToken.None;

        await state.SwapModelAsync(async () =>
        {
            var loaded = await Task.Run(() => ServerStartup.LoadModel(PathA!, state.Options with
            {
                Model = PathA!,
                ModelId = Path.GetFileNameWithoutExtension(PathA!),
            }));
            state.Options = loaded.Options;
            state.Model = loaded.Model;
            state.Tokenizer = loaded.Tokenizer;
            state.ChatTemplate = loaded.ChatTemplate;
            state.Generator = loaded.Generator;
            state.LoadedModelPath = loaded.LoadedModelPath;
            state.CurrentGguf = loaded.CurrentGguf;
            state.KvCacheConfig = loaded.KvCacheConfig;
            state.KvCacheFactory = loaded.KvCacheFactory;
            state.EstimatedBytes = loaded.EstimatedBytes;
        }, ct);
        string idA = state.Options.ModelId;
        Assert.True(state.IsReady);

        await state.SwapModelAsync(async () =>
        {
            var loaded = await Task.Run(() => ServerStartup.LoadModel(PathB!, state.Options with
            {
                Model = PathB!,
                ModelId = Path.GetFileNameWithoutExtension(PathB!),
            }));
            state.Options = loaded.Options;
            state.Model = loaded.Model;
            state.Tokenizer = loaded.Tokenizer;
            state.ChatTemplate = loaded.ChatTemplate;
            state.Generator = loaded.Generator;
            state.LoadedModelPath = loaded.LoadedModelPath;
            state.CurrentGguf = loaded.CurrentGguf;
            state.KvCacheConfig = loaded.KvCacheConfig;
            state.KvCacheFactory = loaded.KvCacheFactory;
            state.EstimatedBytes = loaded.EstimatedBytes;
        }, ct);
        string idB = state.Options.ModelId;

        Assert.NotEqual(idA, idB);
        Assert.True(state.IsReady);
        // Model A was evicted (disposed) immediately, not kept resident - matches pre-#369 behavior.
        Assert.Equal(0, state.Residency.StashedCount);
        Assert.False(state.Residency.Contains(idA));
    }
}
