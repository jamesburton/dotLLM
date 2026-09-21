using System.Text;
using System.Text.Json;
using DotLLM.HuggingFace;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// Model lifecycle endpoints.
/// <list type="bullet">
///   <item><c>GET  /v1/models/available</c> — list locally downloaded models (ungated).</item>
///   <item><c>POST /v1/models/load</c> — hot-swap the loaded model.</item>
///   <item><c>POST /v1/models/unload</c> — explicitly unload one or all resident models (#454, gated).</item>
///   <item><c>POST /v1/models/enable</c> / <c>/v1/models/disable</c> — curate what is loadable (#454, gated).</item>
///   <item><c>POST /v1/models/pull</c> — download into the HF hub cache, streaming progress (#454, gated).</item>
///   <item><c>GET  /v1/models/pull</c>, <c>GET /v1/models/pull/{id}</c> — poll jobs (ungated).</item>
///   <item><c>DELETE /v1/models/pull/{id}</c> — cancel a job (#454, gated).</item>
/// </list>
/// </summary>
public static class ModelManagementEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/v1/models/available", (ServerState state) =>
        {
            var models = HuggingFaceDownloader.ListLocalModels();
            return new AvailableModelsResponse
            {
                Models = models.Select(m =>
                {
                    var modelId = Path.GetFileNameWithoutExtension(m.Filename);
                    return new AvailableModelDto
                    {
                        RepoId = m.RepoId,
                        Filename = m.Filename,
                        FullPath = m.FullPath,
                        SizeBytes = m.SizeBytes,
                        // (#454) The key a tray correlates against GET /v1/models and passes to
                        // the enable/disable routes — the same id a load of this file produces.
                        ModelId = modelId,
                        Enabled = state.Catalog.IsEnabled(modelId),
                    };
                }).ToArray(),
            };
        });

        app.MapPost("/v1/models/load", async (ModelLoadRequest request, ServerState state, CancellationToken ct) =>
        {
            var resolvedPath = ServerStartup.ResolveModelPath(request.Model, request.Quant);
            if (resolvedPath is null)
                return Results.BadRequest(ErrorResponse.InvalidRequest($"Model not found: {request.Model}", param: "model", code: "model_not_found"));

            // (#454) Honour the operator's enable/disable curation. Checked against both the
            // requested argument and the model id this load would produce, since a client may
            // legitimately name either.
            var loadKey = Path.GetFileNameWithoutExtension(resolvedPath);
            if (!state.Catalog.IsEnabled(request.Model) || !state.Catalog.IsEnabled(loadKey))
                return Results.BadRequest(ErrorResponse.InvalidRequest($"Model is disabled: {request.Model}", param: "model", code: "model_disabled"));

            try
            {
                await state.SwapModelAsync(async () =>
                {
                    var newOptions = state.Options with
                    {
                        Model = request.Model,
                        Quant = request.Quant,
                        Device = request.Device ?? state.Options.Device,
                        GpuLayers = request.GpuLayers ?? state.Options.GpuLayers,
                        CacheTypeK = request.CacheTypeK ?? state.Options.CacheTypeK,
                        CacheTypeV = request.CacheTypeV ?? state.Options.CacheTypeV,
                        Threads = request.Threads ?? state.Options.Threads,
                        DecodeThreads = request.DecodeThreads ?? state.Options.DecodeThreads,
                        SpeculativeModel = request.SpeculativeModel,
                        SpeculativeCandidates = request.SpeculativeK ?? state.Options.SpeculativeCandidates,
                        ModelId = Path.GetFileNameWithoutExtension(resolvedPath),
                        RopeOverride = ServerOptions.BuildRopeOverride(
                            request.RopeScaling, request.RopeFreqBase, request.RopeScale,
                            request.YarnOrigCtx, request.YarnAttnFactor,
                            request.YarnBetaFast, request.YarnBetaSlow)
                            ?? state.Options.RopeOverride,
                    };
                    var newState = await Task.Run(() => ServerStartup.LoadModel(resolvedPath, newOptions), ct);

                    // Transfer new state fields into the existing ServerState
                    state.Options = newOptions;
                    state.Config = newState.Config;
                    state.Model = newState.Model;
                    state.Tokenizer = newState.Tokenizer;
                    state.ChatTemplate = newState.ChatTemplate;
                    state.Generator = newState.Generator;
                    state.ToolCallParser = newState.ToolCallParser;
                    state.KvCacheConfig = newState.KvCacheConfig;
                    state.KvCacheFactory = newState.KvCacheFactory;
                    state.PrefixCache = newState.PrefixCache;
                    state.PrefixTrieManager = newState.PrefixTrieManager;
                    state.PagedFactory = newState.PagedFactory;
                    state.LoadedModelPath = resolvedPath;
                    state.CurrentGguf = newState.CurrentGguf;
                    state.DraftModel = newState.DraftModel;
                    state.DraftModelPath = newState.DraftModelPath;
                    state.DraftGguf = newState.DraftGguf;
                    // (#369) Multi-model residency bookkeeping and the continuous-batch scheduler
                    // (previously dropped on every explicit /v1/models/load swap — the server
                    // silently fell back to the single-request gate after the first swap).
                    state.EstimatedBytes = SafeFileLength(resolvedPath);
                    state.KeepAliveSecondsOverride = request.KeepAlive;
                    state.Scheduler = newState.Scheduler;
                    state.StartSchedulerLoop();
                    // Preserve the existing LoRA registry across model swap so loaded
                    // adapters survive (LoadModel mints a fresh registry for fresh starts).
                    if (newState.LoraRegistry is not null && !ReferenceEquals(newState.LoraRegistry, state.LoraRegistry))
                        newState.LoraRegistry.Dispose();

                    await Task.CompletedTask;
                }, ct);

                return Results.Ok(new ModelLoadResponse
                {
                    Status = "loaded",
                    Model = request.Model,
                });
            }
            catch (Exception ex)
            {
                return Results.BadRequest(ErrorResponse.InvalidRequest(ex.Message));
            }
        });

        MapUnload(app);
        MapEnableDisable(app);
        MapPull(app);
    }

    // ───────────────────────────── unload (#454) ─────────────────────────────

    private static void MapUnload(WebApplication app) =>
        app.MapPost("/v1/models/unload", async (ModelUnloadRequest request, ServerState state, CancellationToken ct) =>
        {
            if (!state.Options.AllowModelAdminApi)
                return AdminGate.Forbidden("POST /v1/models/unload");

            var unloaded = await state.UnloadAsync(request.Model, request.All, ct);
            return Results.Ok(new ModelUnloadResponse
            {
                Status = unloaded.Count > 0 ? "unloaded" : "not_resident",
                Unloaded = unloaded.ToArray(),
            });
        });

    // ────────────────────────── enable / disable (#454) ──────────────────────

    private static void MapEnableDisable(WebApplication app)
    {
        app.MapPost("/v1/models/enable", (ModelEnableRequest request, ServerState state) =>
        {
            if (!state.Options.AllowModelAdminApi)
                return AdminGate.Forbidden("POST /v1/models/enable");
            if (string.IsNullOrWhiteSpace(request.Model))
                return Results.BadRequest(ErrorResponse.InvalidRequest("model is required", param: "model"));

            state.Catalog.Enable(request.Model);
            return Results.Ok(new ModelEnableResponse
            {
                Model = request.Model,
                Enabled = true,
                StillLoaded = IsActive(state, request.Model),
            });
        });

        app.MapPost("/v1/models/disable", (ModelEnableRequest request, ServerState state) =>
        {
            if (!state.Options.AllowModelAdminApi)
                return AdminGate.Forbidden("POST /v1/models/disable");
            if (string.IsNullOrWhiteSpace(request.Model))
                return Results.BadRequest(ErrorResponse.InvalidRequest("model is required", param: "model"));

            state.Catalog.Disable(request.Model);

            // Disabling blocks the next activation; it deliberately does not interrupt a model
            // that is loaded and serving right now. `still_loaded` says so, and
            // POST /v1/models/unload is the verb that actually frees it.
            return Results.Ok(new ModelEnableResponse
            {
                Model = request.Model,
                Enabled = false,
                StillLoaded = IsActive(state, request.Model),
            });
        });
    }

    private static bool IsActive(ServerState state, string key) =>
        state.IsReady && state.Model is not null
        && string.Equals(state.Options.ModelId, key, StringComparison.OrdinalIgnoreCase);

    // ────────────────────────────── pull (#454) ──────────────────────────────

    private static void MapPull(WebApplication app)
    {
        app.MapGet("/v1/models/pull", (ServerState state) =>
            Results.Ok(new ModelPullJobListResponse
            {
                Jobs = state.PullManager.List().Select(j => j.ToDto()).ToArray(),
            }));

        app.MapGet("/v1/models/pull/{id}", (string id, ServerState state) =>
        {
            var job = state.PullManager.Get(id);
            return job is null
                ? Results.NotFound(ErrorResponse.NotFound($"No such pull job: {id}", param: "id", code: "pull_job_not_found"))
                : Results.Ok(job.ToDto());
        });

        app.MapDelete("/v1/models/pull/{id}", (string id, ServerState state) =>
        {
            if (!state.Options.AllowModelAdminApi)
                return AdminGate.Forbidden("DELETE /v1/models/pull/{id}");

            return state.PullManager.Cancel(id)
                ? Results.Ok(new StatusResponse { Status = "cancelling" })
                : Results.NotFound(ErrorResponse.NotFound($"No such pull job: {id}", param: "id", code: "pull_job_not_found"));
        });

        app.MapPost("/v1/models/pull", async (ModelPullRequest request, HttpContext http, ServerState state) =>
        {
            if (!state.Options.AllowModelAdminApi)
                return AdminGate.Forbidden("POST /v1/models/pull");
            if (string.IsNullOrWhiteSpace(request.RepoId))
                return Results.BadRequest(ErrorResponse.InvalidRequest("repo_id is required", param: "repo_id"));
            if (string.IsNullOrWhiteSpace(request.Filename))
                return Results.BadRequest(ErrorResponse.InvalidRequest("filename is required", param: "filename"));

            var job = state.PullManager.Start(request.RepoId, request.Filename, request.Revision);

            // Non-streaming: hand back the job and let the caller poll. The download keeps
            // running regardless.
            if (request.Stream is false)
                return Results.Json(job.ToDto(), statusCode: StatusCodes.Status202Accepted);

            await StreamPullProgressAsync(http, job);
            return Results.Empty;
        });
    }

    /// <summary>
    /// Streams a pull job's progress as SSE until the job reaches a terminal state.
    /// </summary>
    /// <remarks>
    /// The client disconnecting (<see cref="HttpContext.RequestAborted"/>) ends the <i>stream</i>
    /// only — it never cancels the job. A tray app closing its window must not abort a
    /// multi-gigabyte download; <c>DELETE /v1/models/pull/{id}</c> is the only thing that cancels.
    /// </remarks>
    private static async Task StreamPullProgressAsync(HttpContext http, PullJob job)
    {
        SseResponse.ApplyHeaders(http);

        // A bounded DropOldest channel keeps the writer (the download loop, which raises Progress
        // synchronously) from ever blocking on a slow SSE consumer. Only intermediate progress
        // ticks are dropped; the terminal state is written last, so it always lands.
        var channel = System.Threading.Channels.Channel.CreateBounded<ModelPullJobDto>(
            new System.Threading.Channels.BoundedChannelOptions(4)
            {
                FullMode = System.Threading.Channels.BoundedChannelFullMode.DropOldest,
                SingleReader = true,
            });

        void OnProgress(PullJob j) => channel.Writer.TryWrite(j.ToDto());
        job.Progress += OnProgress;
        _ = job.WaitAsync().ContinueWith(_ =>
        {
            channel.Writer.TryWrite(job.ToDto());
            channel.Writer.TryComplete();
        }, TaskScheduler.Default);

        try
        {
            channel.Writer.TryWrite(job.ToDto());
            await foreach (var dto in channel.Reader.ReadAllAsync(http.RequestAborted))
            {
                var json = JsonSerializer.Serialize(dto, ServerJsonContext.Default.ModelPullJobDto);
                await http.Response.WriteAsync($"data: {json}\n\n", Encoding.UTF8, http.RequestAborted);
                await http.Response.Body.FlushAsync(http.RequestAborted);
            }
            await http.Response.WriteAsync("data: [DONE]\n\n", Encoding.UTF8, http.RequestAborted);
            await http.Response.Body.FlushAsync(http.RequestAborted);
        }
        catch (OperationCanceledException)
        {
            // Client went away. The job keeps running - see the remarks above.
        }
        finally
        {
            job.Progress -= OnProgress;
        }
    }

    private static long SafeFileLength(string path) => ServerStartup.SafeFileLength(path);
}
