using DotLLM.Server.Endpoints;

namespace DotLLM.Server;

/// <summary>
/// Extension methods for registering all dotLLM API endpoints.
/// </summary>
public static class EndpointExtensions
{
    /// <summary>
    /// Maps all dotLLM API endpoints: the OpenAI-compatible surface plus the
    /// Anthropic-compatible <c>POST /v1/messages</c> surface (#448).
    /// </summary>
    /// <param name="app">The web application.</param>
    /// <param name="serveUi">When true, also serves the embedded web chat UI at <c>GET /</c>.</param>
    public static WebApplication MapDotLLMEndpoints(this WebApplication app, bool serveUi = false)
    {
        ChatCompletionEndpoint.Map(app);
        MessagesEndpoint.Map(app);
        CompletionEndpoint.Map(app);
        ModelEndpoint.Map(app);
        TokenizeEndpoint.Map(app);
        HealthEndpoint.Map(app);
        PropsEndpoint.Map(app);
        ConfigEndpoint.Map(app);
        PromptCacheEndpoint.Map(app);
        ModelManagementEndpoint.Map(app);
        ModelInspectEndpoint.Map(app);
        EmbeddingsEndpoint.Map(app);
        LoraEndpoints.Map(app);
        SettingsEndpoint.Map(app);
        DeviceEndpoint.Map(app);

        if (serveUi)
            WebUIEndpoint.Map(app);

        return app;
    }
}
