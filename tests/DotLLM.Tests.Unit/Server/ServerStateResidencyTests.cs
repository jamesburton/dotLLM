using DotLLM.Server.Models;
using DotLLM.Engine;
using DotLLM.Server;
using Xunit;
using DotLLM.Tests.Integration.Fixtures;

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
[Collection("SequentialFileIO")]   // reads the process-global model-store env vars that SequentialFileIO tests mutate (#1395)
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

    private static string? FindCachedGguf(string quantMarker) =>
        TestFixtureResolver.ResolveFile([], "QuantFactory", "SmolLM-135M-GGUF", [$"SmolLM-135M*{quantMarker}*.gguf"], null).Path;

    private static bool FixturesAvailable => PathA is not null && PathB is not null;

    [SkippableFact]
    public async Task MultiModelResidency_LoadingSecondModel_KeepsFirstResidentAndReactivatableWithoutReload()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");

        // Not locally `using`: NewBareState() registers state in _states, disposed by this class's Dispose().
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

        // Not locally `using`: NewBareState() registers state in _states, disposed by this class's Dispose().
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

        // Not locally `using`: NewBareState() registers state in _states, disposed by this class's Dispose().
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
        // Not locally `using`: NewBareState() registers state in _states, disposed by this class's Dispose().
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
        // Not locally `using`: NewBareState() registers state in _states, disposed by this class's Dispose().
        var state = NewBareState(); // maxResident defaults to 1
        var ct = CancellationToken.None;

        await state.SwapModelAsync(async () =>
        {
            // Not locally `using`: loaded's live fields (Model, CurrentGguf, ...) are copied onto
            // `state` below, which owns them from here on; disposing `loaded` itself would dispose
            // the same underlying objects out from under `state`.
            var loaded = await Task.Run(() => ServerStartup.LoadModel(PathA!, state.Options with
            {
                Model = PathA!,
                ModelId = Path.GetFileNameWithoutExtension(PathA!),
            })).ConfigureAwait(false);
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
            // Not locally `using`: loaded's live fields (Model, CurrentGguf, ...) are copied onto
            // `state` below, which owns them from here on; disposing `loaded` itself would dispose
            // the same underlying objects out from under `state`.
            var loaded = await Task.Run(() => ServerStartup.LoadModel(PathB!, state.Options with
            {
                Model = PathB!,
                ModelId = Path.GetFileNameWithoutExtension(PathB!),
            })).ConfigureAwait(false);
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

    /// <summary>
    /// #1023: a swap must free the old model BEFORE loading the new one. The loader is observed at the moment it runs: the old
    /// model is already detached and (default budget) evicted, nothing is ready, and the status reports the load. The old order
    /// loaded B while A was still installed - on a GPU, both at once.
    /// </summary>
    [SkippableFact]
    public async Task Swap_FreesTheActiveModelBeforeLoading_AndReportsLoading()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");
        var state = NewBareState(); // default MaxResidentModels = 1
        var ct = CancellationToken.None;
        Assert.Null(await state.EnsureActiveAsync(PathA, keepAliveOverride: null, ct));

        bool? modelDetachedAtLoad = null, readyAtLoad = null;
        int stashedAtLoad = -1;
        ServerStatusResponse? statusAtLoad = null;
        state.ModelLoader = (path, options) =>
        {
            modelDetachedAtLoad = state.Model is null;
            readyAtLoad = state.IsReady;
            stashedAtLoad = state.Residency.StashedCount;
            statusAtLoad = state.GetStatus();
            return ServerStartup.LoadModel(path, options);
        };

        Assert.Null(await state.EnsureActiveAsync(PathB, keepAliveOverride: null, ct));

        Assert.True(modelDetachedAtLoad, "the old model must be detached before the new one loads");
        Assert.Equal(0, stashedAtLoad);      // ...and evicted (disposed), not just stashed, under the default budget
        Assert.False(readyAtLoad);
        Assert.Equal("loading", statusAtLoad!.State);
        Assert.Equal(ServerStartup.ModelIdFor(PathB!, PathB!), statusAtLoad.LoadingModelId);
        Assert.NotNull(statusAtLoad.LoadingSince);
        Assert.Equal("idle", state.GetStatus().State);
        Assert.True(state.IsReady);
    }

    /// <summary>
    /// #1416: a chat-triggered load (EnsureActiveAsync) clones the current Options, which carry the PREVIOUS model's per-load KV choice.
    /// A Q8 chosen for model A must not become model B's default; the operator's startup value is the fallback. Needs no model fixture:
    /// the loader seam records the options it is handed and fails.
    /// </summary>
    [Fact]
    public async Task ChatTriggeredSwap_DoesNotInheritThePreviousModelsPerLoadKvChoice()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-1416-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string fake = Path.Combine(dir, "b.gguf");
            File.WriteAllBytes(fake, "GGUF"u8.ToArray());
            var state = NewBareState();
            state.Options = state.Options with
            {
                StartupCacheTypeK = null, StartupCacheTypeV = "f16",
                CacheTypeK = "q8_0", CacheTypeV = "q8_0",   // what a successful explicit load of the previous model leaves behind
            };

            ServerOptions? seen = null;
            state.ModelLoader = (path, options) => { seen = options; throw new InvalidOperationException("stop after capturing the options"); };
            await state.EnsureActiveAsync(fake, keepAliveOverride: null, CancellationToken.None);

            Assert.NotNull(seen);
            Assert.Null(seen!.CacheTypeK);
            Assert.Equal("f16", seen.CacheTypeV);
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    /// <summary>#1023: a load whose requester has already gone must not free the active model, let alone load.</summary>
    [SkippableFact]
    public async Task CancelledRequest_DoesNotFreeTheActiveModel_OrStartALoad()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");
        var state = NewBareState();
        Assert.Null(await state.EnsureActiveAsync(PathA, keepAliveOverride: null, CancellationToken.None));
        var modelA = state.Model;
        bool loaderCalled = false;
        state.ModelLoader = (p, o) => { loaderCalled = true; return ServerStartup.LoadModel(p, o); };

        using var gone = new CancellationTokenSource();
        gone.Cancel();
        string? error = await state.EnsureActiveAsync(PathB, keepAliveOverride: null, gone.Token);

        Assert.NotNull(error);
        Assert.False(loaderCalled);
        Assert.Same(modelA, state.Model);
        Assert.True(state.IsReady);
    }

    /// <summary>
    /// #1023: a request admitted for A that runs after a swap to B must run on A (re-activated under the gate), handed the CURRENT
    /// generator - never B's, and never A's disposed one.
    /// </summary>
    [SkippableFact]
    public async Task QueuedRequest_AfterASwap_RunsOnItsOwnModel_WithTheCurrentGenerator()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");
        var state = NewBareState();
        var ct = CancellationToken.None;
        Assert.Null(await state.EnsureActiveAsync(PathA, keepAliveOverride: null, ct));
        string idA = state.Options.ModelId;
        var admittedGenerator = state.Generator;

        Assert.Null(await state.EnsureActiveAsync(PathB, keepAliveOverride: null, ct));   // the swap the request queued behind
        Assert.NotEqual(idA, state.Options.ModelId);

        string? ranOn = null;
        TextGenerator? handed = null;
        await state.ExecuteAsync(idA, g => { ranOn = Path.GetFileName(state.LoadedModelPath); handed = g; return Task.CompletedTask; }, ct);

        Assert.Equal(Path.GetFileName(PathA), ranOn);   // A's file (its id string may be re-qualified on reload)
        Assert.Same(state.Generator, handed);
        Assert.NotSame(admittedGenerator, handed);   // A was evicted and reloaded: the captured instance is dead
    }

    /// <summary>
    /// #1023 (seen live): req#2 loaded B, released the gate to render its prompt, and an older waiter needing A took the gate first,
    /// swapped back to A, and req#2 then had to reload B - two extra loads. The activating request now keeps first claim: a waiter
    /// that would swap the model away yields until it has generated.
    /// </summary>
    [SkippableFact]
    public async Task ActivatingRequest_KeepsFirstClaim_AWaiterForAnotherModelYieldsInsteadOfSwappingItAway()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");
        var state = NewBareState();
        var ct = CancellationToken.None;
        Assert.Null(await state.EnsureActiveAsync(PathA, keepAliveOverride: null, ct));
        string idA = state.Options.ModelId;

        int loads = 0;
        state.ModelLoader = (p, o) => { Interlocked.Increment(ref loads); return ServerStartup.LoadModel(p, o); };
        var order = new System.Collections.Concurrent.ConcurrentQueue<string>();
        var activated = new TaskCompletionSource<string>();
        var generate = new TaskCompletionSource();

        var r2 = Task.Run(async () =>
        {
            using var scope = ServerLog.BeginRequest("req#2");
            Assert.Null(await state.EnsureActiveAsync(PathB, keepAliveOverride: null, ct));
            activated.SetResult(state.Options.ModelId);
            await generate.Task;   // rendering its prompt...
            await state.ExecuteAsync(state.Options.ModelId, _ => { order.Enqueue("r2:" + Path.GetFileName(state.LoadedModelPath)); return Task.CompletedTask; }, ct);
        });
        await activated.Task;

        var r3 = Task.Run(async () =>
        {
            using var scope = ServerLog.BeginRequest("req#3");
            await state.ExecuteAsync(idA, _ => { order.Enqueue("r3:" + Path.GetFileName(state.LoadedModelPath)); return Task.CompletedTask; }, ct);
        });

        await WaitUntilAsync(() => state.GetStatus().WaitingRequests == 1);
        Assert.Equal(1, loads);          // B only: req#3 is yielding, not reloading A
        Assert.Empty(order);
        // Review of #1023: the gate is free during the hand-off, but the server is not idle and req#3 is queued.
        var during = state.GetStatus();
        Assert.Equal("busy", during.State);
        Assert.Contains("req#2", during.BusyWith);
        Assert.Equal(1, during.WaitingRequests);

        generate.SetResult();
        await Task.WhenAll(r2, r3);

        Assert.Equal(["r2:" + Path.GetFileName(PathB), "r3:" + Path.GetFileName(PathA)], order.ToArray());
        Assert.Equal(2, loads);          // B, then A once - not B, A, B
    }

    private static async Task WaitUntilAsync(Func<bool> condition)
    {
        for (int i = 0; i < 3000 && !condition(); i++) await Task.Delay(10);
        Assert.True(condition());
    }

    /// <summary>Review of #1023: a load that can never fit the target GPU is refused BEFORE the resident model is freed.</summary>
    [SkippableTheory]
    [InlineData(1L, 1024L, "cannot load there")]                       // weights > the device's total memory
    [InlineData(1L, 1L << 40, "nothing was unloaded")]                 // weights > free + what this server's models there would release
    public async Task LoadThatCanNeverFit_IsRefusedBeforeFreeingTheResidentModel(long free, long total, string expected)
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");
        var state = NewBareState();
        Assert.Null(await state.EnsureActiveAsync(PathA, keepAliveOverride: null, CancellationToken.None));
        var modelA = state.Model;
        state.Options = state.Options with { Device = "gpu" };   // the next load targets an explicit, full-offload GPU
        state.CudaMemoryProbe = _ => (free, total);
        bool loaderCalled = false;
        state.ModelLoader = (p, o) => { loaderCalled = true; return ServerStartup.LoadModel(p, o); };

        string? error = await state.EnsureActiveAsync(PathB, keepAliveOverride: null, CancellationToken.None);

        Assert.Contains(expected, error);
        Assert.False(loaderCalled);
        Assert.Same(modelA, state.Model);   // still resident and serving
        Assert.True(state.IsReady);
    }

    /// <summary>
    /// Review of #1023: once freeing has begun, a client disconnect must not abandon the load - that left the server with no model.
    /// The token is cancelled from inside the free step (the device probe it consults).
    /// </summary>
    [SkippableFact]
    public async Task DisconnectAfterFreeingBegan_StillCompletesTheLoad()
    {
        Skip.IfNot(FixturesAvailable, "SmolLM-135M cached GGUF quants not found locally.");
        var state = NewBareState(maxResident: 2);   // A is stashed (not evicted), so the device-room step runs and probes
        Assert.Null(await state.EnsureActiveAsync(PathA, keepAliveOverride: null, CancellationToken.None));
        state.Options = state.Options with { Device = "auto" };
        using var cts = new CancellationTokenSource();
        state.CudaMemoryProbe = _ => { cts.Cancel(); return (1L << 40, 1L << 40); };   // the client goes away mid-free
        state.ModelLoader = (p, o) => ServerStartup.LoadModel(p, o with { Device = "cpu" });

        string? error = await state.EnsureActiveAsync(PathB, keepAliveOverride: null, cts.Token);

        Assert.True(cts.IsCancellationRequested);
        Assert.Null(error);
        Assert.True(state.IsReady);
        Assert.Equal(Path.GetFileName(PathB), Path.GetFileName(state.LoadedModelPath));
    }
}
