using System;
using System.Linq;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #450: only the collection route <c>GET /v1/models</c> existed, so the OpenAI SDK's
/// <c>client.models.retrieve()</c> 404'd.
/// </summary>
public sealed class ModelRetrieveEndpointTests
{
    private static ServerState CreateState(string modelId) =>
        new() { Options = new ServerOptions { Model = "test", ModelId = modelId } };

    /// <summary>
    /// A bare server that has never loaded a model still reports its configured id from the
    /// collection route, so retrieve has to agree — a 404 there would contradict the list.
    /// </summary>
    [Fact]
    public void Retrieve_FindsTheConfiguredModel_OnABareServer()
    {
        using var state = CreateState("Qwen/Qwen3-4B");

        var model = ModelEndpoint.Retrieve(state, "Qwen/Qwen3-4B");

        Assert.NotNull(model);
        Assert.Equal("Qwen/Qwen3-4B", model!.Id);
        Assert.Equal("model", model.Object);
    }

    [Fact]
    public void Retrieve_AgreesWithTheCollectionRoute()
    {
        using var state = CreateState("some-model");

        foreach (var listed in ModelEndpoint.BuildList(state).Data)
            Assert.NotNull(ModelEndpoint.Retrieve(state, listed.Id));
    }

    [Fact]
    public void Retrieve_ReturnsNullForAnUnknownId()
    {
        using var state = CreateState("some-model");

        Assert.Null(ModelEndpoint.Retrieve(state, "not-a-model"));
        Assert.Null(ModelEndpoint.Retrieve(state, ""));
    }

    /// <summary>
    /// Ids here are HuggingFace repo ids, which contain a slash. A single-segment
    /// <c>{id}</c> route would not match them at all, so the route has to be a catch-all.
    /// </summary>
    [Fact]
    public void Retrieve_HandlesSlashBearingRepoIds()
    {
        using var state = CreateState("unsloth/Qwen3-4B-GGUF");

        Assert.NotNull(ModelEndpoint.Retrieve(state, "unsloth/Qwen3-4B-GGUF"));
    }

    /// <summary>
    /// The retrieve route is a catch-all under <c>/v1/models</c>. Asserting the literal routes
    /// are still registered guards the ordering assumption: ASP.NET prefers a literal segment
    /// over a catch-all, so <c>/v1/models/available</c> must not be swallowed.
    /// </summary>
    [Fact]
    public void RouteTable_HasBothTheCatchAllAndTheLiteralModelRoutes()
    {
        var builder = WebApplication.CreateSlimBuilder();
        using var state = CreateState("m");
        builder.Services.AddSingleton(state);
        var app = builder.Build();

        ModelEndpoint.Map(app);
        ModelManagementEndpoint.Map(app);
        ModelInspectEndpoint.Map(app);

        var patterns = ((IEndpointRouteBuilder)app).DataSources
            .SelectMany(d => d.Endpoints)
            .OfType<RouteEndpoint>()
            .Select(e => e.RoutePattern.RawText)
            .ToArray();

        Assert.Contains("/v1/models", patterns);
        Assert.Contains("/v1/models/{**id}", patterns);
        Assert.Contains("/v1/models/available", patterns);
        Assert.Contains("/v1/models/inspect", patterns);
    }
}
