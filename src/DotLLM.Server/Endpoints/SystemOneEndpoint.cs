using System.Text.Json;
using DotLLM.Engine;
using DotLLM.Engine.Decisions;
using DotLLM.Engine.Scheduler;
using DotLLM.Server.Models;
using DotLLM.Server.RateLimiting;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// POST /v1/systemone — Jev-compatible decision endpoint (issue #708).
/// </summary>
/// <remarks>
/// <para>
/// Same wire shape as TypeSafe's System One API, so Jev SDKs, the OpenJev servers and the Microsoft Agent Framework
/// <c>Microsoft.Agents.AI.TypeSafe</c> provider can point at a dotLLM server (set the provider's endpoint to
/// <c>http://host:port/v1/systemone</c>). Each question is answered with ONE prefill and no decode loop: the option letters' logits at the
/// first generated position are renormalised into per-option probabilities (see <see cref="DecisionEvaluator"/>). Works with any loaded
/// instruction-tuned model on any backend; Tev1 is the model its prompt contract was trained for.
/// </para>
/// <para>
/// Questions run sequentially in request order. The state comes first in each prompt, so the prefix caches reuse it across questions.
/// Probabilities are the model's own, restricted to the options; they are not calibrated like Jev's.
/// </para>
/// </remarks>
public static class SystemOneEndpoint
{
    /// <summary>Most score levels accepted (Jev documents 10; the label alphabet allows 52).</summary>
    public const int MaxScoreLevels = 10;

    public static void Map(WebApplication app) =>
        app.MapPost("/v1/systemone", HandleAsync);

    /// <summary>A request failure the endpoint reports as an error envelope.</summary>
    internal sealed class SystemOneException(int status, string message, string? param = null) : Exception(message)
    {
        public int Status { get; } = status;
        public string? Param { get; } = param;
    }

    private static async Task HandleAsync(SystemOneRequest request, ServerState state, HttpContext httpContext)
    {
        var ct = httpContext.RequestAborted;

        var activationError = await state.EnsureActiveAsync(requestedModel: null, keepAliveOverride: null, ct);
        if (activationError is not null)
        {
            await WriteErrorAsync(httpContext, 400, activationError);
            return;
        }

        if (!state.IsReady || state.Generator is null || state.Tokenizer is null || state.ChatTemplate is null || state.Config is null)
        {
            await WriteErrorAsync(httpContext, 503, "No model loaded");
            return;
        }

        DecisionEvaluator evaluator;
        try
        {
            evaluator = new DecisionEvaluator(state.Tokenizer, state.ChatTemplate, (prompt, options, token) => RunAsync(state, httpContext, prompt, options, token));
        }
        catch (NotSupportedException ex)
        {
            await WriteErrorAsync(httpContext, 501, ex.Message);
            return;
        }

        try
        {
            var response = await AnswerAsync(request, evaluator, state.Options.ModelId, state.Config.MaxSequenceLength, ct);
            RateLimitMiddleware.GetLease(httpContext)?.ReportActualTokens((int)response.Usage!.InputTokens);
            await httpContext.Response.WriteAsJsonAsync(response, ServerJsonContext.Default.SystemOneResponse, contentType: null, ct);
        }
        catch (SystemOneException ex)
        {
            await WriteErrorAsync(httpContext, ex.Status, ex.Message, ex.Param);
        }
    }

    /// <summary>Validates the request and answers every question. Throws <see cref="SystemOneException"/> on a bad request.</summary>
    internal static async Task<SystemOneResponse> AnswerAsync(
        SystemOneRequest request, DecisionEvaluator evaluator, string modelId, int maxSequenceLength, CancellationToken ct)
    {
        if (request.Questions is not { Count: > 0 })
            throw new SystemOneException(422, "'questions' must contain at least one question.", "questions");

        // Validate every question up front so a bad one fails the request before any forward pass runs.
        var plans = new List<(string Id, string Type, string Instructions, DecisionOption[] Options, string[] Keys)>();
        foreach (var (id, q) in request.Questions)
            plans.Add(Plan(id, q));

        var answers = new Dictionary<string, SystemOneAnswer>(plans.Count, StringComparer.Ordinal);
        long inputTokens = 0;
        JsonElement? state = request.State.ValueKind == JsonValueKind.Undefined ? null : request.State;

        foreach (var (id, type, instructions, options, keys) in plans)
        {
            var dist = await evaluator.ScoreAsync(state, instructions, options, ct).ConfigureAwait(false);
            if (dist.PromptTokens > maxSequenceLength)
                throw new SystemOneException(422, $"The prompt for question '{id}' is {dist.PromptTokens} tokens, over the model's context of {maxSequenceLength}.", "state");
            inputTokens += dist.PromptTokens;
            double[] p = dist.Probabilities;

            answers[id] = type switch
            {
                "noul" => new SystemOneAnswer { Type = "noul", Noul = p[0] },
                "choice" => new SystemOneAnswer
                {
                    Type = "choice",
                    Choice = keys[DecisionMath.ArgMax(p)],
                    Probabilities = ToMap(keys, p),
                    Confidence = DecisionMath.Confidence(p),
                },
                _ => new SystemOneAnswer
                {
                    Type = "score",
                    Score = DecisionMath.ExpectedIndex(p),
                    Probabilities = ToMap(keys, p),
                    Confidence = DecisionMath.Confidence(p),
                    Legend = LegendOf(keys, options),
                },
            };
        }

        return new SystemOneResponse
        {
            Model = modelId,
            Answers = answers,
            Usage = new SystemOneUsage { InputTokens = inputTokens, OutputTokens = 0 },
        };
    }

    private static Dictionary<string, double> ToMap(string[] keys, double[] p)
    {
        var map = new Dictionary<string, double>(keys.Length, StringComparer.Ordinal);
        for (int i = 0; i < keys.Length; i++) map[keys[i]] = p[i];
        return map;
    }

    private static Dictionary<string, string> LegendOf(string[] keys, DecisionOption[] options)
    {
        var map = new Dictionary<string, string>(keys.Length, StringComparer.Ordinal);
        for (int i = 0; i < keys.Length; i++) map[keys[i]] = options[i].Description;
        return map;
    }

    private static (string Id, string Type, string Instructions, DecisionOption[] Options, string[] Keys) Plan(string id, SystemOneQuestion q)
    {
        string param = $"questions.{id}";
        if (string.IsNullOrWhiteSpace(q.Instructions))
            throw new SystemOneException(422, $"Question '{id}' has no instructions.", param);

        switch (q.Type)
        {
            case "noul":
            {
                string trueText = "Yes, this is true.", falseText = "No, this is not true.";
                if (q.Criteria.ValueKind == JsonValueKind.Object)
                {
                    if (q.Criteria.TryGetProperty("true", out var t) && t.ValueKind == JsonValueKind.String) trueText = t.GetString()!;
                    if (q.Criteria.TryGetProperty("false", out var f) && f.ValueKind == JsonValueKind.String) falseText = f.GetString()!;
                }
                else if (q.Criteria.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null))
                    throw new SystemOneException(422, $"Question '{id}': noul criteria must be an object with optional 'true' and 'false'.", param);

                string[] keys = ["true", "false"];
                return (id, "noul", q.Instructions!, DecisionEvaluator.BuildOptions([("true", trueText), ("false", falseText)]), keys);
            }

            case "choice":
            {
                if (q.Criteria.ValueKind != JsonValueKind.Object)
                    throw new SystemOneException(422, $"Question '{id}': choice criteria must be an object of name -> description.", param);
                var items = new List<(string, string)>();
                foreach (var prop in q.Criteria.EnumerateObject())
                    items.Add((prop.Name, prop.Value.ValueKind == JsonValueKind.String ? prop.Value.GetString()! : prop.Name));
                if (items.Count < 2)
                    throw new SystemOneException(422, $"Question '{id}': a choice question needs at least two choices.", param);
                if (items.Count > DecisionEvaluator.MaxOptions)
                    throw new SystemOneException(422, $"Question '{id}': at most {DecisionEvaluator.MaxOptions} choices are supported.", param);
                return (id, "choice", q.Instructions!, DecisionEvaluator.BuildOptions(items), items.Select(i => i.Item1).ToArray());
            }

            case "score":
            {
                if (q.Criteria.ValueKind != JsonValueKind.Array)
                    throw new SystemOneException(422, $"Question '{id}': score criteria must be an array of level descriptions.", param);
                var items = new List<(string, string)>();
                int i = 0;
                foreach (var level in q.Criteria.EnumerateArray())
                {
                    if (level.ValueKind != JsonValueKind.String)
                        throw new SystemOneException(422, $"Question '{id}': score levels must be strings.", param);
                    items.Add((i.ToString(System.Globalization.CultureInfo.InvariantCulture), level.GetString()!));
                    i++;
                }
                if (items.Count < 2 || items.Count > MaxScoreLevels)
                    throw new SystemOneException(422, $"Question '{id}': a score question needs 2..{MaxScoreLevels} levels, got {items.Count}.", param);
                return (id, "score", q.Instructions!, DecisionEvaluator.BuildOptions(items), items.Select(x => x.Item1).ToArray());
            }

            default:
                throw new SystemOneException(422, $"Question '{id}': unknown type '{q.Type}' (expected noul, choice or score).", param);
        }
    }

    /// <summary>Runs one prompt through the continuous-batch scheduler when active, otherwise the direct generator.</summary>
    private static async Task<InferenceResponse> RunAsync(
        ServerState state, HttpContext httpContext, string prompt, DotLLM.Core.Configuration.InferenceOptions options, CancellationToken ct)
    {
        if (state.Scheduler is { } scheduler)
        {
            var inferenceRequest = new InferenceRequest
            {
                TokenIds = state.Tokenizer!.Encode(prompt),
                Options = options,
                ApiKey = httpContext.Items.TryGetValue(RateLimitMiddleware.ApiKeyItemKey, out var k) ? k as string : null,
            };
            return await scheduler.EnqueueAsync(inferenceRequest, ct);
        }

        InferenceResponse? result = null;
        await state.ExecuteAsync(() =>
        {
            result = state.Generator!.Generate(prompt, options);
            return Task.CompletedTask;
        }, ct);
        return result!;
    }

    private static Task WriteErrorAsync(HttpContext httpContext, int statusCode, string message, string? param = null)
    {
        httpContext.Response.StatusCode = statusCode;
        return httpContext.Response.WriteAsJsonAsync(
            statusCode >= 500 ? ErrorResponse.Internal(message) : ErrorResponse.InvalidRequest(message, param: param),
            ServerJsonContext.Default.ErrorResponse,
            contentType: null,
            httpContext.RequestAborted);
    }
}
