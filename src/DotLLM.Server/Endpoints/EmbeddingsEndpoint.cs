using System.Buffers.Binary;
using System.Text.Json;
using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// POST /v1/embeddings — OpenAI-compatible embeddings endpoint (issue #451).
/// </summary>
/// <remarks>
/// <para><b>Backend coverage: CPU only.</b> The pooled hidden state comes from
/// <see cref="IEmbeddingModel.ForwardHidden"/>, which only the CPU <c>TransformerModel</c>
/// implements. When a Vulkan or CUDA model is loaded this endpoint returns <c>501 Not
/// Implemented</c> rather than silently producing an unvalidated vector. A GPU path is a
/// follow-on.</para>
/// <para>Each input item is its own forward pass with positions <c>0..n-1</c>; sequences are not
/// packed into one batch, because the CPU forward has no per-sequence attention mask.</para>
/// <para>The model's scratch buffers (and its compute thread pool) are shared mutable state, so
/// the request holds <b>both</b> locks that guard the model: <see cref="ServerState.ExecuteAsync"/>
/// against the direct-generator path, and — when a continuous-batch scheduler is active —
/// <c>ContinuousBatchSchedulerService.AcquireModelAsync</c> against its run loop, which drives
/// forward passes outside the request gate by design.</para>
/// </remarks>
public static class EmbeddingsEndpoint
{
    public static void Map(WebApplication app) =>
        app.MapPost("/v1/embeddings", HandleAsync);

    private static async Task HandleAsync(
        EmbeddingRequest request,
        ServerState state,
        HttpContext httpContext)
    {
        var ct = httpContext.RequestAborted;

        var activationError = await state.EnsureActiveAsync(request.Model, keepAliveOverride: null, ct);
        if (activationError is not null)
        {
            await WriteErrorAsync(httpContext, 400, activationError);
            return;
        }

        if (!state.IsReady || state.Model is null || state.Tokenizer is null || state.Config is null)
        {
            await WriteErrorAsync(httpContext, 503, "No model loaded");
            return;
        }

        if (state.Model is not IEmbeddingModel embeddingModel)
        {
            await WriteErrorAsync(httpContext, 501,
                $"The loaded model ({state.Model.GetType().Name}) does not support embedding extraction. "
                + "POST /v1/embeddings is implemented for the CPU backend only; run the server without "
                + "a GPU backend to use it.");
            return;
        }

        if (request.Dimensions is not null)
        {
            await WriteErrorAsync(httpContext, 400,
                "'dimensions' is not supported: dotLLM returns the model's full hidden size and does not "
                + "perform Matryoshka truncation.");
            return;
        }

        bool base64;
        switch (request.EncodingFormat)
        {
            case null or "" or "float":
                base64 = false;
                break;
            case "base64":
                base64 = true;
                break;
            default:
                await WriteErrorAsync(httpContext, 400,
                    $"'encoding_format' must be 'float' or 'base64', got '{request.EncodingFormat}'.");
                return;
        }

        PoolingType? requestedPooling = null;
        if (!string.IsNullOrEmpty(request.Pooling))
        {
            requestedPooling = request.Pooling.ToLowerInvariant() switch
            {
                "last" => PoolingType.Last,
                "mean" => PoolingType.Mean,
                "cls" => PoolingType.Cls,
                _ => null,
            };
            if (requestedPooling is null)
            {
                await WriteErrorAsync(httpContext, 400,
                    $"'pooling' must be 'last', 'mean' or 'cls', got '{request.Pooling}'.");
                return;
            }
        }

        var tokenizer = state.Tokenizer;
        var parsed = EmbeddingInputParser.Parse(request.Input, tokenizer.Encode, state.Config.VocabSize);
        if (!parsed.Ok)
        {
            await WriteErrorAsync(httpContext, 400, parsed.Error!);
            return;
        }

        var sequences = parsed.Sequences!;
        int maxSeq = state.Config.MaxSequenceLength;
        for (int i = 0; i < sequences.Count; i++)
        {
            if (sequences[i].Length > maxSeq)
            {
                await WriteErrorAsync(httpContext, 400,
                    $"'input[{i}]' is {sequences[i].Length} tokens, which exceeds the model's context length of {maxSeq}.");
                return;
            }
        }

        var pooling = EmbeddingPooler.Resolve(requestedPooling, embeddingModel.DeclaredPoolingType);
        if (pooling is PoolingType.None or PoolingType.Rank)
        {
            await WriteErrorAsync(httpContext, 400,
                $"The loaded checkpoint declares pooling type '{pooling}', which this endpoint cannot represent. "
                + "Pass \"pooling\": \"last\" (or \"mean\"/\"cls\") explicitly.");
            return;
        }

        bool normalize = request.Normalize ?? true;
        int hiddenSize = state.Config.HiddenSize;

        var data = new EmbeddingData[sequences.Count];
        int promptTokens = 0;

        try
        {
            // Two locks, both needed. ExecuteAsync serialises against the direct-generator path;
            // the scheduler lease serialises against the continuous-batch run loop, which drives
            // forward passes on this same model OUTSIDE the request gate. Either alone leaves the
            // model's shared scratch buffers exposed to a concurrent forward - and that is not a
            // subtle numerical wobble: with the lease removed, a generation running alongside an
            // embedding tears down the shared ComputeThreadPool and the process dies. The lease is
            // taken inside the gate so the two are always acquired in the same order.
            await state.ExecuteAsync(async () =>
            {
                using var lease = state.Scheduler is { } scheduler
                    ? await scheduler.AcquireModelAsync(ct)
                    : null;

                for (int i = 0; i < sequences.Count; i++)
                {
                    ct.ThrowIfCancellationRequested();

                    int[] tokens = sequences[i];
                    var positions = new int[tokens.Length];
                    for (int p = 0; p < positions.Length; p++)
                        positions[p] = p;

                    var vector = new float[hiddenSize];
                    using (var hidden = embeddingModel.ForwardHidden(tokens, positions, deviceId: 0))
                    {
                        var hiddenSpan = AsSpan(hidden, tokens.Length * hiddenSize);
                        EmbeddingPooler.Pool(hiddenSpan, tokens.Length, hiddenSize, pooling, vector);
                    }

                    if (normalize)
                        EmbeddingPooler.L2Normalize(vector);

                    data[i] = new EmbeddingData
                    {
                        Index = i,
                        Embedding = base64 ? EncodeBase64(vector) : EncodeFloats(vector),
                    };
                    promptTokens += tokens.Length;
                }
            }, ct);
        }
        catch (NotSupportedException ex)
        {
            await WriteErrorAsync(httpContext, 400, ex.Message);
            return;
        }

        var response = new EmbeddingResponse
        {
            Data = data,
            Model = state.Options.ModelId,
            Usage = new EmbeddingUsage { PromptTokens = promptTokens, TotalTokens = promptTokens },
        };

        await httpContext.Response.WriteAsJsonAsync(
            response, ServerJsonContext.Default.EmbeddingResponse, contentType: null, ct);
    }

    /// <summary>Views an f32 CPU tensor's payload as a span. The tensor stays alive for the call.</summary>
    private static unsafe ReadOnlySpan<float> AsSpan(Core.Tensors.ITensor tensor, int length)
        => new((void*)tensor.DataPointer, length);

    private static JsonElement EncodeFloats(float[] vector)
        => JsonSerializer.SerializeToElement(vector, ServerJsonContext.Default.SingleArray);

    /// <summary>
    /// OpenAI's <c>base64</c> encoding is the raw little-endian IEEE-754 float32 payload,
    /// base64-encoded — what <c>numpy.frombuffer(base64.b64decode(s), dtype="float32")</c> reads.
    /// </summary>
    private static JsonElement EncodeBase64(float[] vector)
    {
        var bytes = new byte[vector.Length * sizeof(float)];
        for (int i = 0; i < vector.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), vector[i]);

        return JsonSerializer.SerializeToElement(Convert.ToBase64String(bytes), ServerJsonContext.Default.String);
    }

    /// <summary>
    /// Maps an HTTP status onto the matching <see cref="ErrorResponse"/> factory (#452), so the
    /// envelope's <c>error.type</c> agrees with the status code rather than defaulting to one
    /// class for every failure.
    /// </summary>
    private static ErrorResponse ErrorForStatus(int statusCode, string message) => statusCode switch
    {
        StatusCodes.Status404NotFound => ErrorResponse.NotFound(message),
        StatusCodes.Status429TooManyRequests => ErrorResponse.RateLimit(message),
        >= 500 => ErrorResponse.Internal(message),
        _ => ErrorResponse.InvalidRequest(message),
    };

    private static Task WriteErrorAsync(HttpContext httpContext, int statusCode, string message)
    {
        httpContext.Response.StatusCode = statusCode;
        return httpContext.Response.WriteAsJsonAsync(
            ErrorForStatus(statusCode, message),
            ServerJsonContext.Default.ErrorResponse,
            contentType: null,
            httpContext.RequestAborted);
    }
}
