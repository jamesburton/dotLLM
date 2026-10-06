using DotLLM.HuggingFace;
using DotLLM.Models.Gguf;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /v1/models/inspect?path=... — read GGUF metadata without loading the model.
/// Returns layer count, architecture, and file size for UI configuration.
/// Path is restricted to the configured model directory to prevent path traversal.
/// </summary>
public static class ModelInspectEndpoint
{
    public static void Map(WebApplication app) =>
        app.MapGet("/v1/models/inspect", (string path, ServerState state) =>
        {
            if (string.IsNullOrEmpty(path))
                return Results.BadRequest(ErrorResponse.InvalidRequest("Path is required", param: "path"));

            var fullPath = Path.GetFullPath(path);

            if (!IsAllowedModelPath(fullPath, state))
                return Results.Json(
                    ErrorResponse.InvalidRequest("Path is outside allowed model directories", param: "path"),
                    ServerJsonContext.Default.ErrorResponse,
                    statusCode: 403);

            // Ollama blobs are listed as extension-less sha256-* files; accept those only when the model list offers them.
            if (!fullPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && !OllamaStore.IsBlobPath(fullPath))
                return Results.BadRequest(ErrorResponse.InvalidRequest("Only .gguf files are supported", param: "path"));

            if (!File.Exists(fullPath))
                return Results.BadRequest(ErrorResponse.InvalidRequest("File not found", param: "path"));

            try
            {
                using var gguf = GgufFile.Open(fullPath);
                var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                var fileSize = new FileInfo(fullPath).Length;

                return Results.Ok(new ModelInspectResponse
                {
                    Architecture = config.Architecture.ToString(),
                    NumLayers = config.NumLayers,
                    HiddenSize = config.HiddenSize,
                    NumKvHeads = config.NumKvHeads,
                    HeadDim = config.HeadDim,
                    VocabSize = config.VocabSize,
                    MaxSequenceLength = config.MaxSequenceLength,
                    FileSizeBytes = fileSize,
                });
            }
            catch
            {
                return Results.BadRequest(ErrorResponse.InvalidRequest("Failed to read GGUF metadata", param: "path"));
            }
        });

    /// <summary>
    /// Checks whether the given normalized path is within an allowed model directory.
    /// Allowed directories: the default HuggingFace model cache and the directory of the currently loaded model.
    /// </summary>
    /// <summary>True when <paramref name="fullPath"/> is one of the listed local models.</summary>
    internal static bool IsListedModelPath(string fullPath, IEnumerable<LocalModel> listed) =>
        listed.Any(m => string.Equals(Path.GetFullPath(m.FullPath), fullPath, StringComparison.OrdinalIgnoreCase));

    internal static bool IsAllowedModelPath(string fullPath, ServerState state)
    {
        var modelsDir = Path.GetFullPath(HuggingFaceDownloader.DefaultModelsDirectory);
        if (!modelsDir.EndsWith(Path.DirectorySeparatorChar))
            modelsDir += Path.DirectorySeparatorChar;

        if (fullPath.StartsWith(modelsDir, StringComparison.OrdinalIgnoreCase))
            return true;

        if (!string.IsNullOrEmpty(state.LoadedModelPath))
        {
            var loadedDir = Path.GetFullPath(Path.GetDirectoryName(state.LoadedModelPath)!);
            if (!loadedDir.EndsWith(Path.DirectorySeparatorChar))
                loadedDir += Path.DirectorySeparatorChar;

            if (fullPath.StartsWith(loadedDir, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        // Anything the model list itself offers (HF hub cache, ollama store, ...) must be inspectable, otherwise the UI's layer slider
        // silently keeps its default when inspect is refused.
        try
        {
            if (IsListedModelPath(fullPath, ModelResolver.EnumerateLocal(includeOllama: true)))
                return true;
        }
        catch { /* unreadable store: not allowed */ }

        return false;
    }
}
