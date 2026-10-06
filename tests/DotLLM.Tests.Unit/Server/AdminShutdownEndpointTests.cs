using System.Net;
using System.Text;
using System.Text.Json;
using DotLLM.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>Issue #722: <c>POST /v1/admin/shutdown</c> - gated, answers 202, then the host really stops.</summary>
public sealed class AdminShutdownEndpointTests
{
    private static async Task<(WebApplication App, ServerState State, HttpClient Http)> StartAsync(bool admin)
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        var state = ServerStartup.CreateBareState(new ServerOptions { Model = "none", AllowModelAdminApi = admin });
        builder.Services.AddSingleton(state);
        var app = builder.Build();
        app.MapDotLLMEndpoints(serveUi: false);
        await app.StartAsync();
        return (app, state, new HttpClient { BaseAddress = new Uri(app.Urls.First()) });
    }

    [Fact]
    public async Task WithoutAdminFlag_ItIs403_AndTheServerKeepsRunning()
    {
        var (app, state, http) = await StartAsync(admin: false);
        try
        {
            var resp = await http.PostAsync("/v1/admin/shutdown", new StringContent("", Encoding.UTF8, "application/json"));
            Assert.Equal(HttpStatusCode.Forbidden, resp.StatusCode);
            Assert.Contains("--allow-model-admin", await resp.Content.ReadAsStringAsync());

            await Task.Delay(AdminEndpointDelay() + TimeSpan.FromMilliseconds(500));
            Assert.False(app.Lifetime.ApplicationStopping.IsCancellationRequested);
            Assert.Equal(HttpStatusCode.OK, (await http.GetAsync("/health")).StatusCode);
        }
        finally { http.Dispose(); await app.DisposeAsync(); state.Dispose(); }
    }

    [Fact]
    public async Task WithAdminFlag_AnswersAcceptedThenTheHostStops()
    {
        var (app, state, http) = await StartAsync(admin: true);
        try
        {
            var stopped = new TaskCompletionSource();
            app.Lifetime.ApplicationStopping.Register(() => stopped.TrySetResult());

            var resp = await http.PostAsync("/v1/admin/shutdown", new StringContent("", Encoding.UTF8, "application/json"));

            Assert.Equal(HttpStatusCode.Accepted, resp.StatusCode);   // the answer is delivered before the stop
            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
            Assert.Equal("shutting_down", doc.RootElement.GetProperty("status").GetString());
            Assert.True(await Task.WhenAny(stopped.Task, Task.Delay(TimeSpan.FromSeconds(10))) == stopped.Task, "the host never began stopping");
        }
        finally { http.Dispose(); await app.DisposeAsync(); state.Dispose(); }
    }

    private static TimeSpan AdminEndpointDelay() => TimeSpan.FromMilliseconds(250);
}
