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
        var result = await ComputeAsync(state, request, ct);
        if (result.Error is not null)
        {
            await WriteErrorAsync(httpContext, result.Status, result.Error);
            return;
        }

        bool base64 = result.Base64;
        var data = new EmbeddingData[result.Vectors!.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = new EmbeddingData
            {
                Index = i,
                Embedding = base64 ? EncodeBase64(result.Vectors[i]) : EncodeFloats(result.Vectors[i]),
            };
        int promptTokens = result.PromptTokens;
        var response = new EmbeddingResponse
        {
            Data = data,
            Model = state.Options.ModelId,
            Usage = new EmbeddingUsage { PromptTokens = promptTokens, TotalTokens = promptTokens },
        };

        await httpContext.Response.WriteAsJsonAsync(
            response, ServerJsonContext.Default.EmbeddingResponse, contentType: null, ct);
    }


    /// <summary>Outcome of <see cref="ComputeAsync"/>: either an HTTP error or the embedding vectors.</summary>
    internal sealed record EmbedResult(int Status, string? Error, float[][]? Vectors, int PromptTokens, bool Base64)
    {
        public static EmbedResult Fail(int status, string message) => new(status, message, null, 0, false);
    }

    /// <summary>
    /// Shared embedding core for <c>/v1/embeddings</c> and ollama's <c>/api/embed</c>: model activation,
    /// validation, pooling, truncation and normalisation. Returns errors rather than writing them so
    /// each surface can use its own error envelope.
    /// </summary>
    internal static async Task<EmbedResult> ComputeAsync(ServerState state, EmbeddingRequest request, CancellationToken ct)
    {
        var activationError = await state.EnsureActiveAsync(request.Model, keepAliveOverride: null, ct);
        if (activationError is not null)
        {
            return EmbedResult.Fail(400, activationError);
        }

        if (!state.IsReady || state.Model is null || state.Tokenizer is null || state.Config is null)
        {
            return EmbedResult.Fail(503, "No model loaded");
        }

        if (state.Model is not IEmbeddingModel embeddingModel)
        {
            return EmbedResult.Fail(501, $"The loaded model ({state.Model.GetType().Name}) does not support embedding extraction. "
                + "POST /v1/embeddings is implemented for the CPU backend only; run the server without "
                + "a GPU backend to use it.");
        }

        int outDims = state.Config.HiddenSize;
        if (request.Dimensions is { } dims)
        {
            if (dims < 1 || dims > state.Config.HiddenSize)
            {
                return EmbedResult.Fail(400, $"'dimensions' must be between 1 and the model's hidden size ({state.Config.HiddenSize}), got {dims}.");
            }
            outDims = dims;
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
                return EmbedResult.Fail(400, $"'encoding_format' must be 'float' or 'base64', got '{request.EncodingFormat}'.");
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
                return EmbedResult.Fail(400, $"'pooling' must be 'last', 'mean' or 'cls', got '{request.Pooling}'.");
            }
        }

        var tokenizer = state.Tokenizer;
        var parsed = EmbeddingInputParser.Parse(request.Input, tokenizer.Encode, state.Config.VocabSize);
        if (!parsed.Ok)
        {
            return EmbedResult.Fail(400, parsed.Error!);
        }

        var sequences = parsed.Sequences!;
        int maxSeq = state.Config.MaxSequenceLength;
        for (int i = 0; i < sequences.Count; i++)
        {
            if (sequences[i].Length > maxSeq)
            {
                return EmbedResult.Fail(400, $"'input[{i}]' is {sequences[i].Length} tokens, which exceeds the model's context length of {maxSeq}.");
            }
        }

        var pooling = EmbeddingPooler.Resolve(requestedPooling, embeddingModel.DeclaredPoolingType);
        if (pooling is PoolingType.None or PoolingType.Rank)
        {
            return EmbedResult.Fail(400, $"The loaded checkpoint declares pooling type '{pooling}', which this endpoint cannot represent. "
                + "Pass \"pooling\": \"last\" (or \"mean\"/\"cls\") explicitly.");
        }

        bool normalize = request.Normalize ?? true;
        int hiddenSize = state.Config.HiddenSize;

        var vectors = new float[sequences.Count][];
        int promptTokens = 0;

        try
        {
            // Both locks are still needed and both are still taken -- ExecuteAsync now acquires
            // the scheduler lease itself (#461 follow-up), so this endpoint no longer takes it
            // separately. Re-adding an inner AcquireModelAsync here would self-deadlock: the
            // semaphore is not reentrant.
            await state.ExecuteAsync(async () =>
            {
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

                    // Matryoshka-style truncation: keep the leading outDims components, then renormalise.
                    if (outDims < hiddenSize)
                        vector = vector.AsSpan(0, outDims).ToArray();

                    if (normalize)
                        EmbeddingPooler.L2Normalize(vector);

                    vectors[i] = vector;
                    promptTokens += tokens.Length;
                }
            }, ct);
        }
        catch (NotSupportedException ex)
        {
            return EmbedResult.Fail(400, ex.Message);
        }

        return new EmbedResult(200, null, vectors, promptTokens, base64);
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
