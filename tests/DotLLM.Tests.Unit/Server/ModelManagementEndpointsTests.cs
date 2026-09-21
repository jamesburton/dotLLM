using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Contract tests for the model-lifecycle / settings surface added in #454.
/// </summary>
/// <remarks>
/// <para>
/// <b>Route presence</b> is asserted against the real <see cref="EndpointDataSource"/> produced by
/// <see cref="EndpointExtensions.MapDotLLMEndpoints"/>, not a mock. Before the routes existed
/// every one of these assertions failed with "route not mapped" — the in-process equivalent of the
/// 404 a client would get — which is what makes them discriminating rather than tautological
/// (#417).
/// </para>
/// <para>
/// <b>Serialization</b> is asserted through <see cref="ServerJsonContext"/> rather than a plain
/// <see cref="JsonSerializer"/> call, because that context is source-generated: a DTO missing its
/// <c>[JsonSerializable]</c> registration compiles fine and only fails at runtime, when a client
/// is already waiting on the response.
/// </para>
/// </remarks>
public sealed class ModelManagementEndpointsTests
{
    private static ServerState NewState(bool allowAdmin = false, string modelId = "test-model") =>
        new()
        {
            Options = new ServerOptions { Model = "test", ModelId = modelId, AllowModelAdminApi = allowAdmin },
        };

    /// <summary>All routes mapped by the production endpoint wiring, as "METHOD /path".</summary>
    private static HashSet<string> MappedRoutes()
    {
        var builder = WebApplication.CreateSlimBuilder();
        // Minimal-API parameter binding is resolved at Map time, so the handlers' DI dependency
        // has to be registered even though no request is ever issued.
        builder.Services.AddSingleton(NewState());
        var app = builder.Build();
        app.MapDotLLMEndpoints();

        var routes = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var source in ((IEndpointRouteBuilder)app).DataSources)
        {
            foreach (var endpoint in source.Endpoints)
            {
                if (endpoint is not RouteEndpoint re) continue;
                var methods = re.Metadata.GetMetadata<Microsoft.AspNetCore.Routing.IHttpMethodMetadata>()?.HttpMethods
                              ?? ["*"];
                foreach (var m in methods)
                    routes.Add($"{m} /{re.RoutePattern.RawText?.TrimStart('/')}");
            }
        }
        return routes;
    }

    // ───────────────────────────── route presence ─────────────────────────────

    [Theory]
    [InlineData("POST /v1/models/unload")]
    [InlineData("POST /v1/models/enable")]
    [InlineData("POST /v1/models/disable")]
    [InlineData("POST /v1/models/pull")]
    [InlineData("GET /v1/models/pull")]
    [InlineData("GET /v1/models/pull/{id}")]
    [InlineData("DELETE /v1/models/pull/{id}")]
    [InlineData("GET /v1/settings")]
    [InlineData("PUT /v1/settings")]
    [InlineData("GET /v1/devices")]
    public void Route_IsMapped(string route)
    {
        Assert.Contains(route, MappedRoutes());
    }

    [Fact]
    public void PreexistingRoutes_StillMapped()
    {
        var routes = MappedRoutes();
        Assert.Contains("GET /v1/models", routes);
        Assert.Contains("POST /v1/models/load", routes);
        Assert.Contains("GET /v1/models/available", routes);
        Assert.Contains("GET /v1/config", routes);
    }

    // RETIRED: `MessagesRoute_NotIntroducedHere` asserted that #454 did not introduce
    // POST /v1/messages, which was true and worth pinning while #454 and #448 were developed
    // in parallel worktrees. #448 has since landed and legitimately owns that route, so the
    // assertion is now false BY DESIGN rather than by regression. It is removed instead of
    // inverted: "some other issue registers this route" is not a property this file should own.

    // ───────────────────────── enable / disable semantics ─────────────────────

    [Fact]
    public void Catalog_UnknownKey_IsEnabled()
    {
        Assert.True(new ModelCatalog().IsEnabled("anything"));
    }

    [Fact]
    public void Catalog_Disable_ThenEnable_RoundTrips()
    {
        var catalog = new ModelCatalog();
        Assert.True(catalog.Disable("m"));
        Assert.False(catalog.Disable("m")); // idempotent, reports no change
        Assert.False(catalog.IsEnabled("m"));
        Assert.Equal(["m"], catalog.DisabledKeys());

        Assert.True(catalog.Enable("m"));
        Assert.True(catalog.IsEnabled("m"));
        Assert.Empty(catalog.DisabledKeys());
    }

    [Fact]
    public void Catalog_KeyMatchingIsCaseInsensitive()
    {
        var catalog = new ModelCatalog();
        catalog.Disable("Qwen3-4B-Q8_0");
        Assert.False(catalog.IsEnabled("qwen3-4b-q8_0"));
    }

    /// <summary>
    /// Disabling must block the implicit activation a chat request's <c>model</c> field triggers,
    /// not just the explicit load route — otherwise a "curated list" is cosmetic.
    /// </summary>
    [Fact]
    public async Task EnsureActive_DisabledModel_IsRefused()
    {
        var state = NewState();
        state.Catalog.Disable("other-model");

        var error = await state.EnsureActiveAsync("other-model", keepAliveOverride: null, CancellationToken.None);

        Assert.NotNull(error);
        Assert.Contains("disabled", error!, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// A request may name a file path or repo id rather than the model key, so checking only the
    /// raw request string leaves the curation bypassable: <c>model: "C:/…/foo.gguf"</c> resolves
    /// to a file whose key ("foo") is disabled, and would load anyway.
    /// </summary>
    /// <remarks>
    /// Discriminating without a real model: the fixture is an <b>empty</b> .gguf, so if the
    /// catalog check did not fire before <c>LoadModel</c> the error would be a GGUF parse failure,
    /// not "disabled". Asserting the message therefore also pins the check's position.
    /// </remarks>
    [Fact]
    public async Task EnsureActive_DisabledKey_IsRefusedEvenWhenRequestedByPath()
    {
        var dir = Path.Combine(Path.GetTempPath(), "dotllm-454-key-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "curated-off.gguf");
        await File.WriteAllBytesAsync(path, []);
        try
        {
            var state = NewState();
            state.Catalog.Disable("curated-off"); // the derived key, not the path

            var error = await state.EnsureActiveAsync(path, keepAliveOverride: null, CancellationToken.None);

            Assert.NotNull(error);
            Assert.Contains("disabled", error!, StringComparison.OrdinalIgnoreCase);
        }
        finally { try { Directory.Delete(dir, recursive: true); } catch { } }
    }

    [Fact]
    public async Task EnsureActive_EnabledButMissingModel_ReportsNotFound_NotDisabled()
    {
        var state = NewState();
        var error = await state.EnsureActiveAsync("definitely-not-a-real-model-454", null, CancellationToken.None);

        Assert.NotNull(error);
        Assert.DoesNotContain("disabled", error!, StringComparison.OrdinalIgnoreCase);
    }

    // ──────────────────────────────── unload ──────────────────────────────────

    [Fact]
    public async Task Unload_NothingResident_ReportsNotResident()
    {
        var state = NewState();
        var unloaded = await state.UnloadAsync(key: null, all: false, CancellationToken.None);
        Assert.Empty(unloaded);
    }

    [Fact]
    public async Task Unload_ByKey_DisposesStashedSnapshot()
    {
        var state = NewState();
        var snapshot = new ResidentModelSnapshot
        {
            Key = "stashed-model",
            Options = state.Options with { ModelId = "stashed-model" },
            LastUsedUtc = DateTimeOffset.UtcNow,
            EstimatedBytes = 1234,
        };
        state.Residency.Stash(snapshot);
        Assert.True(state.Residency.Contains("stashed-model"));

        var unloaded = await state.UnloadAsync("stashed-model", all: false, CancellationToken.None);

        Assert.Equal(["stashed-model"], unloaded);
        Assert.False(state.Residency.Contains("stashed-model"));
    }

    [Fact]
    public async Task Unload_All_DropsEveryStashedModel()
    {
        var state = NewState();
        foreach (var key in new[] { "a", "b", "c" })
        {
            state.Residency.Stash(new ResidentModelSnapshot
            {
                Key = key,
                Options = state.Options with { ModelId = key },
                LastUsedUtc = DateTimeOffset.UtcNow,
            });
        }

        var unloaded = await state.UnloadAsync(key: null, all: true, CancellationToken.None);

        Assert.Equal(3, unloaded.Count);
        Assert.Equal(0, state.Residency.StashedCount);
    }

    [Fact]
    public async Task Unload_UnknownKey_LeavesOtherModelsAlone()
    {
        var state = NewState();
        state.Residency.Stash(new ResidentModelSnapshot
        {
            Key = "keep-me",
            Options = state.Options with { ModelId = "keep-me" },
            LastUsedUtc = DateTimeOffset.UtcNow,
        });

        var unloaded = await state.UnloadAsync("not-resident", all: false, CancellationToken.None);

        Assert.Empty(unloaded);
        Assert.True(state.Residency.Contains("keep-me"));
    }

    // ──────────────────────────────── settings ────────────────────────────────

    [Fact]
    public void Settings_Read_ReflectsResidencyManager_NotOptions()
    {
        var state = NewState();
        state.Residency.DefaultKeepAliveSeconds = 77;
        state.Residency.MaxResidentModels = 3;
        state.Residency.MemoryBudgetBytes = 999;
        state.IdleSweepIntervalSeconds = 2.5;

        var dto = SettingsEndpoint.Read(state);

        Assert.Equal(77, dto.KeepAliveSeconds);
        Assert.Equal(3, dto.MaxResidentModels);
        Assert.Equal(999, dto.ResidentMemoryBudgetBytes);
        Assert.Equal(2.5, dto.IdleSweepIntervalSeconds);
    }

    [Fact]
    public void Settings_Apply_PartialUpdate_LeavesUnsetFieldsAlone()
    {
        var state = NewState();
        state.Residency.DefaultKeepAliveSeconds = 300;
        state.Residency.MaxResidentModels = 2;

        var result = SettingsEndpoint.Apply(state, new SettingsUpdateRequest { KeepAliveSeconds = 60 });

        Assert.Equal(60, state.Residency.DefaultKeepAliveSeconds);
        Assert.Equal(2, state.Residency.MaxResidentModels); // untouched
        Assert.Equal(["keep_alive_seconds"], result.Applied);
        Assert.Empty(result.RestartRequired);
    }

    /// <summary>
    /// The whole point of the endpoint: a settings change must be live. Storing these on
    /// <see cref="ServerOptions"/> would look correct here but revert on the next model swap,
    /// which replaces the whole record — hence the assertion that the residency manager (the
    /// object the sweep actually reads) changed.
    /// </summary>
    [Fact]
    public void Settings_Apply_WritesThroughToTheObjectTheSweepReads()
    {
        var state = NewState();
        SettingsEndpoint.Apply(state, new SettingsUpdateRequest
        {
            KeepAliveSeconds = -1,
            MaxResidentModels = 4,
            ResidentMemoryBudgetBytes = 8_000_000_000,
            IdleSweepIntervalSeconds = 1,
        });

        Assert.Equal(-1, state.Residency.DefaultKeepAliveSeconds);
        Assert.Equal(4, state.Residency.MaxResidentModels);
        Assert.Equal(8_000_000_000, state.Residency.MemoryBudgetBytes);
        Assert.Equal(1, state.IdleSweepIntervalSeconds);
    }

    /// <summary>
    /// Lowering the residency budget must evict <i>now</i>, not merely at the next load —
    /// "settings changes take effect without a restart" is an acceptance criterion of #454.
    /// </summary>
    [Fact]
    public void Settings_Apply_TighteningResidency_EvictsImmediately()
    {
        var state = NewState();
        state.Residency.MaxResidentModels = 5;
        foreach (var key in new[] { "a", "b", "c" })
        {
            state.Residency.Stash(new ResidentModelSnapshot
            {
                Key = key,
                Options = state.Options with { ModelId = key },
                LastUsedUtc = DateTimeOffset.UtcNow.AddMinutes(-Array.IndexOf(new[] { "a", "b", "c" }, key)),
            });
        }
        Assert.Equal(3, state.Residency.StashedCount);

        var result = SettingsEndpoint.Apply(state, new SettingsUpdateRequest { MaxResidentModels = 2 });

        // Budget of 2 counts the active slot, so exactly one stashed model may remain.
        Assert.Equal(1, state.Residency.StashedCount);
        Assert.Equal(2, result.Evicted.Length);
    }

    [Fact]
    public void Settings_Apply_KeepAliveZeroIsHonoured_NotTreatedAsUnset()
    {
        var state = NewState();
        state.Residency.DefaultKeepAliveSeconds = 300;

        SettingsEndpoint.Apply(state, new SettingsUpdateRequest { KeepAliveSeconds = 0 });

        Assert.Equal(0, state.Residency.DefaultKeepAliveSeconds);
    }

    // ───────────────────────────────── devices ────────────────────────────────

    [Fact]
    public void Devices_AlwaysReportsCpuAsServable()
    {
        var response = DeviceEndpoint.Describe();
        var cpu = Assert.Single(response.Backends, b => b.Name == "cpu");
        Assert.True(cpu.Available);
        Assert.True(cpu.Servable);
        Assert.Equal("cpu", Assert.Single(cpu.Devices).DeviceString);
    }

    /// <summary>
    /// The server's load path dispatches to the CPU or CUDA loader only, so a tray must not be
    /// told it can place a model on Vulkan even where Vulkan devices exist.
    /// </summary>
    [Fact]
    public void Devices_VulkanIsNeverReportedServable()
    {
        var vulkan = Assert.Single(DeviceEndpoint.Describe().Backends, b => b.Name == "vulkan");
        Assert.False(vulkan.Servable);
        Assert.NotNull(vulkan.Note);
        Assert.All(vulkan.Devices, d => Assert.Null(d.DeviceString));
    }

    [Fact]
    public void Devices_CudaDeviceStringsMatchWhatLoadAccepts()
    {
        var cuda = Assert.Single(DeviceEndpoint.Describe().Backends, b => b.Name == "cuda");
        foreach (var d in cuda.Devices)
            Assert.Equal($"gpu:{d.Index}", d.DeviceString);
    }

    [Fact]
    public void Devices_ProbeIsCached()
    {
        Assert.Same(DeviceEndpoint.Describe(), DeviceEndpoint.Describe());
    }

    // ──────────────────────── serialization (source-gen) ──────────────────────

    [Fact]
    public void SettingsDto_SerializesThroughSourceGenContext()
    {
        var json = JsonSerializer.Serialize(
            SettingsEndpoint.Read(NewState(allowAdmin: true)), ServerJsonContext.Default.SettingsDto);

        Assert.Contains("\"keep_alive_seconds\"", json);
        Assert.Contains("\"model_admin_api_enabled\":true", json);
        Assert.Contains("\"disabled_models\"", json);
    }

    [Fact]
    public void SettingsUpdateRequest_PartialBody_DeserializesMissingFieldsAsNull()
    {
        var req = JsonSerializer.Deserialize(
            """{"keep_alive_seconds":0}""", ServerJsonContext.Default.SettingsUpdateRequest);

        Assert.NotNull(req);
        Assert.Equal(0, req!.KeepAliveSeconds);
        Assert.Null(req.MaxResidentModels);
        Assert.Null(req.IdleSweepIntervalSeconds);
    }

    [Fact]
    public void UnloadRequest_EmptyBody_DefaultsToActiveModelOnly()
    {
        var req = JsonSerializer.Deserialize("{}", ServerJsonContext.Default.ModelUnloadRequest);
        Assert.NotNull(req);
        Assert.Null(req!.Model);
        Assert.False(req.All);
    }

    /// <summary>An omitted <c>stream</c> must mean "stream", the documented default.</summary>
    /// <remarks>
    /// This case caught a real trap rather than restating the DTO: written as
    /// <c>public bool Stream { get; init; } = true;</c> the property initializer holds for
    /// <c>new ModelPullRequest{…}</c> but is <b>skipped</b> by the source-generated deserializer
    /// because the record carries <c>required</c> members — an omitted field arrived as
    /// <c>false</c>, silently turning every default pull non-streaming. Hence <c>bool?</c>.
    /// </remarks>
    [Fact]
    public void PullRequest_OmittedStream_IsTreatedAsStreaming()
    {
        var req = JsonSerializer.Deserialize(
            """{"repo_id":"o/r","filename":"f.gguf"}""", ServerJsonContext.Default.ModelPullRequest);

        Assert.NotNull(req);
        Assert.Null(req!.Stream);           // absent, so the endpoint's "not false" default applies
        Assert.False(req.Stream is false);  // the condition the endpoint actually branches on
        Assert.Null(req.Revision);
    }

    [Fact]
    public void PullRequest_ExplicitStreamFalse_IsHonoured()
    {
        var req = JsonSerializer.Deserialize(
            """{"repo_id":"o/r","filename":"f.gguf","stream":false}""",
            ServerJsonContext.Default.ModelPullRequest);

        Assert.True(req!.Stream is false);
    }

    [Fact]
    public void DeviceListResponse_SerializesThroughSourceGenContext()
    {
        var json = JsonSerializer.Serialize(DeviceEndpoint.Describe(), ServerJsonContext.Default.DeviceListResponse);
        Assert.Contains("\"backends\"", json);
        Assert.Contains("\"servable\"", json);
    }

    [Fact]
    public void PullJobDto_OmitsNullPathsWhileRunning()
    {
        var dto = new ModelPullJobDto
        {
            Id = "x", RepoId = "o/r", Filename = "f.gguf", Revision = "main",
            Status = "running", BytesDownloaded = 10,
        };
        var json = JsonSerializer.Serialize(dto, ServerJsonContext.Default.ModelPullJobDto);

        Assert.DoesNotContain("model_path", json);
        Assert.DoesNotContain("error", json);
        Assert.Contains("\"status\":\"running\"", json);
    }

    // ───────────────────────────── CLI / options gate ─────────────────────────

    [Fact]
    public void ServerOptions_AdminGate_IsOffByDefault()
    {
        Assert.False(new ServerOptions { Model = "m" }.AllowModelAdminApi);
        Assert.False(ServerOptions.Parse(["--model", "m"]).AllowModelAdminApi);
    }

    [Fact]
    public void ServerOptions_Parse_AllowModelAdminFlag()
    {
        Assert.True(ServerOptions.Parse(["--model", "m", "--allow-model-admin"]).AllowModelAdminApi);
        Assert.True(ServerOptions.Parse(["--model", "m", "--allow-lora-admin"]).AllowLoraAdminApi);
    }

    /// <summary>
    /// The gate must survive a model swap: <c>POST /v1/models/load</c> rebuilds
    /// <see cref="ServerOptions"/> with a <c>with</c> expression, so a flag that did not carry
    /// forward would silently re-open (or close) the admin API on the first swap.
    /// </summary>
    [Fact]
    public void AdminGate_SurvivesOptionsWithExpression()
    {
        var options = new ServerOptions { Model = "m", AllowModelAdminApi = true };
        var swapped = options with { Model = "other", ModelId = "other" };
        Assert.True(swapped.AllowModelAdminApi);
    }
}
