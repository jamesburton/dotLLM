using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.Models;
using DotLLM.Tray.Api;
using Xunit;

namespace DotLLM.Tray.Tests;

/// <summary>
/// Guards the tray's mirrored DTOs against drift from the real #454 server contract.
/// </summary>
/// <remarks>
/// <para>
/// The tray links no server code on purpose, so its DTOs are copies and copies rot. Each test here
/// serializes a <i>server</i> DTO through <see cref="ServerJsonContext"/> — the exact
/// source-generated path the running server uses — and deserializes it through the tray's
/// <see cref="TrayJsonContext"/>, asserting field for field. A renamed or dropped
/// <c>[JsonPropertyName]</c> on either side fails immediately instead of silently producing a
/// default-valued UI.
/// </para>
/// <para>
/// <b>Discrimination (#417).</b> Demonstrated by mutation: changing the tray's
/// <c>[JsonPropertyName("expires_in_seconds")]</c> to <c>"expires_in"</c> and rebuilding turned
/// <see cref="ResidentModel_RoundTripsThroughTheTrayMirror"/> red on the <c>ExpiresInSeconds</c>
/// assertion (Assert.Equal() Failure: Values differ, Expected: 42, Actual: null) while every other
/// test stayed green; reverting restored it. The same mutation applied to
/// <c>device_string</c> and to <c>model_id</c> failed their respective tests. The assertions are
/// therefore sensitive to exactly the defect class they exist for.
/// </para>
/// </remarks>
public sealed class TrayContractTests
{
    private static T RoundTrip<T>(
        object serverDto,
        System.Text.Json.Serialization.Metadata.JsonTypeInfo serverInfo,
        System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> trayInfo)
    {
        var json = JsonSerializer.Serialize(serverDto, serverInfo);
        var value = JsonSerializer.Deserialize(json, trayInfo);
        Assert.NotNull(value);
        return value!;
    }

    [Fact]
    public void ResidentModel_RoundTripsThroughTheTrayMirror()
    {
        var server = new ModelListResponse
        {
            Data =
            [
                new ModelInfoDto
                {
                    Id = "qwen2.5-3b-instruct-q4_k_m",
                    Created = 1_700_000_000,
                    IsActive = true,
                    IdleSeconds = 12.5,
                    KeepAliveSeconds = 300,
                    ExpiresInSeconds = 42,
                    SizeBytes = 2_147_483_648,
                },
            ],
        };

        var tray = RoundTrip(server, ServerJsonContext.Default.ModelListResponse,
            TrayJsonContext.Default.TrayModelList);

        var model = Assert.Single(tray.Data);
        Assert.Equal("qwen2.5-3b-instruct-q4_k_m", model.Id);
        Assert.True(model.IsActive);
        Assert.Equal(12.5, model.IdleSeconds);
        Assert.Equal(300, model.KeepAliveSeconds);
        // The time-to-auto-unload the tray menu shows. #455 names this field explicitly.
        Assert.Equal(42, model.ExpiresInSeconds);
        Assert.Equal(2_147_483_648, model.SizeBytes);
    }

    [Fact]
    public void ResidentModel_NeverExpiring_ArrivesAsNullNotZero()
    {
        // A negative keep-alive means "never auto-unload" and the server omits expires_in_seconds
        // entirely. Zero would render as "unloading now" in the tray — the opposite of the truth.
        var server = new ModelListResponse
        {
            Data = [new ModelInfoDto { Id = "pinned", KeepAliveSeconds = -1, ExpiresInSeconds = null }],
        };

        var tray = RoundTrip(server, ServerJsonContext.Default.ModelListResponse,
            TrayJsonContext.Default.TrayModelList);

        Assert.Null(Assert.Single(tray.Data).ExpiresInSeconds);
    }

    [Fact]
    public void AvailableModel_CarriesTheCorrelationKeyAndEnabledFlag()
    {
        var server = new AvailableModelsResponse
        {
            Models =
            [
                new AvailableModelDto
                {
                    RepoId = "bartowski/Qwen2.5-3B-Instruct-GGUF",
                    Filename = "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                    FullPath = @"C:\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                    SizeBytes = 123,
                    ModelId = "Qwen2.5-3B-Instruct-Q4_K_M",
                    Enabled = false,
                },
            ],
        };

        var tray = RoundTrip(server, ServerJsonContext.Default.AvailableModelsResponse,
            TrayJsonContext.Default.TrayAvailableModelList);

        var model = Assert.Single(tray.Models);
        // model_id is what the tray passes to enable/disable and correlates with GET /v1/models.
        Assert.Equal("Qwen2.5-3B-Instruct-Q4_K_M", model.ModelId);
        Assert.False(model.Enabled);
    }

    [Fact]
    public void Settings_RoundTripEveryField()
    {
        var server = new SettingsDto
        {
            KeepAliveSeconds = 120,
            MaxResidentModels = 3,
            ResidentMemoryBudgetBytes = 8L * 1024 * 1024 * 1024,
            IdleSweepIntervalSeconds = 2.5,
            ModelAdminApiEnabled = true,
            LoraAdminApiEnabled = false,
            DisabledModels = ["old-model"],
        };

        var tray = RoundTrip(server, ServerJsonContext.Default.SettingsDto,
            TrayJsonContext.Default.TraySettingsDto);

        Assert.Equal(120, tray.KeepAliveSeconds);
        Assert.Equal(3, tray.MaxResidentModels);
        Assert.Equal(8L * 1024 * 1024 * 1024, tray.ResidentMemoryBudgetBytes);
        Assert.Equal(2.5, tray.IdleSweepIntervalSeconds);
        Assert.True(tray.ModelAdminApiEnabled);
        Assert.False(tray.LoraAdminApiEnabled);
        Assert.Equal(["old-model"], tray.DisabledModels);
    }

    [Fact]
    public void SettingsUpdate_OmitsNullFieldsSoAPartialUpdateStaysPartial()
    {
        // The server applies exactly the fields present in the JSON. If the tray serialized nulls
        // (or defaults) for the fields the user did not touch, "set keep-alive" would also reset
        // max_resident_models and the budget.
        var update = new TraySettingsUpdate { KeepAliveSeconds = 60 };
        var json = JsonSerializer.Serialize(update, TrayJsonContext.Default.TraySettingsUpdate);

        Assert.Contains("keep_alive_seconds", json, StringComparison.Ordinal);
        Assert.DoesNotContain("max_resident_models", json, StringComparison.Ordinal);
        Assert.DoesNotContain("resident_memory_budget_bytes", json, StringComparison.Ordinal);
        Assert.DoesNotContain("idle_sweep_interval_seconds", json, StringComparison.Ordinal);

        // And the server must be able to read what the tray wrote.
        var server = JsonSerializer.Deserialize(json, ServerJsonContext.Default.SettingsUpdateRequest);
        Assert.NotNull(server);
        Assert.Equal(60, server!.KeepAliveSeconds);
        Assert.Null(server.MaxResidentModels);
        Assert.Null(server.ResidentMemoryBudgetBytes);
        Assert.Null(server.IdleSweepIntervalSeconds);
    }

    [Fact]
    public void SettingsUpdateResponse_RoundTrips()
    {
        var server = new SettingsUpdateResponse
        {
            Settings = new SettingsDto { KeepAliveSeconds = 60, MaxResidentModels = 1 },
            Applied = ["keep_alive_seconds"],
            RestartRequired = [],
            Evicted = ["evicted-model"],
        };

        var tray = RoundTrip(server, ServerJsonContext.Default.SettingsUpdateResponse,
            TrayJsonContext.Default.TraySettingsUpdateResult);

        Assert.Equal(60, tray.Settings.KeepAliveSeconds);
        Assert.Equal(["keep_alive_seconds"], tray.Applied);
        Assert.Empty(tray.RestartRequired);
        Assert.Equal(["evicted-model"], tray.Evicted);
    }

    [Fact]
    public void UnloadRequest_IsReadableByTheServer()
    {
        var json = JsonSerializer.Serialize(
            new TrayUnloadRequest { Model = "foo", All = false }, TrayJsonContext.Default.TrayUnloadRequest);
        var server = JsonSerializer.Deserialize(json, ServerJsonContext.Default.ModelUnloadRequest);

        Assert.NotNull(server);
        Assert.Equal("foo", server!.Model);
        Assert.False(server.All);
    }

    [Fact]
    public void UnloadRequest_UnloadAll_OmitsModel()
    {
        var json = JsonSerializer.Serialize(
            new TrayUnloadRequest { All = true }, TrayJsonContext.Default.TrayUnloadRequest);
        Assert.DoesNotContain("\"model\"", json, StringComparison.Ordinal);

        var server = JsonSerializer.Deserialize(json, ServerJsonContext.Default.ModelUnloadRequest);
        Assert.True(server!.All);
    }

    [Fact]
    public void UnloadResponse_RoundTrips()
    {
        var server = new ModelUnloadResponse { Status = "unloaded", Unloaded = ["a", "b"] };
        var tray = RoundTrip(server, ServerJsonContext.Default.ModelUnloadResponse,
            TrayJsonContext.Default.TrayUnloadResult);

        Assert.Equal("unloaded", tray.Status);
        Assert.Equal(["a", "b"], tray.Unloaded);
    }

    [Fact]
    public void EnableResponse_CarriesStillLoaded()
    {
        // Disabling does not unload. The tray's UI depends on still_loaded to say so rather than
        // implying the model is gone.
        var server = new ModelEnableResponse { Model = "m", Enabled = false, StillLoaded = true };
        var tray = RoundTrip(server, ServerJsonContext.Default.ModelEnableResponse,
            TrayJsonContext.Default.TrayEnableResult);

        Assert.Equal("m", tray.Model);
        Assert.False(tray.Enabled);
        Assert.True(tray.StillLoaded);
    }

    [Fact]
    public void EnableRequest_IsReadableByTheServer()
    {
        var json = JsonSerializer.Serialize(
            new TrayEnableRequest { Model = "m" }, TrayJsonContext.Default.TrayEnableRequest);
        var server = JsonSerializer.Deserialize(json, ServerJsonContext.Default.ModelEnableRequest);
        Assert.Equal("m", server!.Model);
    }

    [Fact]
    public void LoadRequest_IsReadableByTheServer_IncludingKeepAliveOverride()
    {
        var json = JsonSerializer.Serialize(
            new TrayLoadRequest
            {
                Model = "m",
                Device = "gpu:0",
                GpuLayers = 32,
                CacheTypeK = "q8_0",
                CacheTypeV = "q8_0",
                KeepAlive = -1,
            },
            TrayJsonContext.Default.TrayLoadRequest);

        var server = JsonSerializer.Deserialize(json, ServerJsonContext.Default.ModelLoadRequest);

        Assert.NotNull(server);
        Assert.Equal("m", server!.Model);
        Assert.Equal("gpu:0", server.Device);
        Assert.Equal(32, server.GpuLayers);
        Assert.Equal("q8_0", server.CacheTypeK);
        Assert.Equal("q8_0", server.CacheTypeV);
        Assert.Equal(-1, server.KeepAlive);
    }

    [Fact]
    public void Devices_RoundTrip_IncludingTheVulkanNotServableShape()
    {
        var server = new DeviceListResponse
        {
            Backends =
            [
                new BackendInfoDto
                {
                    Name = "cpu",
                    Available = true,
                    DeviceCount = 1,
                    Servable = true,
                    Devices = [new DeviceInfoDto { Index = 0, Name = "CPU", DeviceString = "cpu" }],
                },
                new BackendInfoDto
                {
                    Name = "vulkan",
                    Available = true,
                    DeviceCount = 1,
                    Servable = false,
                    Note = "The server's load path dispatches to CPU or CUDA only.",
                    Devices = [new DeviceInfoDto { Index = 0, Name = "Radeon 8060S", DeviceString = null }],
                },
            ],
        };

        var tray = RoundTrip(server, ServerJsonContext.Default.DeviceListResponse,
            TrayJsonContext.Default.TrayDeviceList);

        var vulkan = Assert.Single(tray.Backends, b => b.Name == "vulkan");
        // available && !servable is the exact shape a device picker must not offer.
        Assert.True(vulkan.Available);
        Assert.False(vulkan.Servable);
        Assert.NotNull(vulkan.Note);
        Assert.Null(Assert.Single(vulkan.Devices).DeviceString);

        var cpu = Assert.Single(tray.Backends, b => b.Name == "cpu");
        Assert.Equal("cpu", Assert.Single(cpu.Devices).DeviceString);
    }

    [Fact]
    public void PullJob_RoundTripsEveryProgressField()
    {
        var server = new ModelPullJobDto
        {
            Id = "job-1",
            RepoId = "org/repo",
            Filename = "model.gguf",
            Revision = "main",
            Status = "running",
            BytesDownloaded = 500,
            TotalBytes = 1000,
            Percent = 50,
            StartedAt = 1_700_000_000,
        };

        var tray = RoundTrip(server, ServerJsonContext.Default.ModelPullJobDto,
            TrayJsonContext.Default.TrayPullJob);

        Assert.Equal("job-1", tray.Id);
        Assert.Equal("org/repo", tray.RepoId);
        Assert.Equal("model.gguf", tray.Filename);
        Assert.Equal("main", tray.Revision);
        Assert.Equal("running", tray.Status);
        Assert.Equal(500, tray.BytesDownloaded);
        Assert.Equal(1000, tray.TotalBytes);
        Assert.Equal(50, tray.Percent);
        Assert.False(tray.IsTerminal);
    }

    [Theory]
    [InlineData("running", false)]
    [InlineData("completed", true)]
    [InlineData("failed", true)]
    [InlineData("cancelled", true)]
    public void PullJob_TerminalStatesMatchTheServersVocabulary(string status, bool expected)
    {
        // These four strings are the server's, verbatim (ModelPullManager). If the tray's set
        // drifts, a progress UI either spins forever or stops early.
        Assert.Equal(expected, new TrayPullJob { Status = status }.IsTerminal);
    }

    [Fact]
    public void PullRequest_AlwaysSendsStreamExplicitly()
    {
        // The server's source-generated deserializer skips property initializers on a record with
        // `required` members, so an omitted `stream` arrives as false. #454 made the field bool?
        // with null meaning "stream". The tray must never rely on that subtlety: it sends the
        // value it means.
        var streaming = JsonSerializer.Serialize(
            new TrayPullRequest { RepoId = "o/r", Filename = "f.gguf", Stream = true },
            TrayJsonContext.Default.TrayPullRequest);
        var polling = JsonSerializer.Serialize(
            new TrayPullRequest { RepoId = "o/r", Filename = "f.gguf", Stream = false },
            TrayJsonContext.Default.TrayPullRequest);

        Assert.Contains("\"stream\":true", streaming, StringComparison.Ordinal);
        Assert.Contains("\"stream\":false", polling, StringComparison.Ordinal);

        var server = JsonSerializer.Deserialize(polling, ServerJsonContext.Default.ModelPullRequest);
        Assert.NotNull(server);
        Assert.Equal("o/r", server!.RepoId);
        Assert.Equal("f.gguf", server.Filename);
        Assert.False(server.Stream);
    }

    [Fact]
    public void ErrorResponse_RoundTripsSoTheGateMessageSurvives()
    {
        var server = new ErrorResponse
        {
            Error = "POST /v1/models/unload is disabled. Start the server with --allow-model-admin "
                  + "(ServerOptions.AllowModelAdminApi) to enable the model-administration API.",
        };

        var tray = RoundTrip(server, ServerJsonContext.Default.ErrorResponse,
            TrayJsonContext.Default.TrayErrorResponse);

        Assert.Contains("--allow-model-admin", tray.Error, StringComparison.Ordinal);
    }
}
