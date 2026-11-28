using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures.DSpark;

/// <summary>
/// Finds and attaches the Inkling 2 DSpark drafter (issue #2039). The drafter ships as a separate GGUF (<c>dspark-Inkling-2-Mini-Q8_0.gguf</c>,
/// <c>unsloth/Inkling-2-Mini-GGUF</c>); the usual case is an auto-detected <c>dspark-*.gguf</c> beside the trunk. An explicit path (<c>--dspark</c>)
/// always wins and a failure to attach it is an error; an auto-detected file that cannot be attached is the caller's to log and ignore.
/// </summary>
public static class Inkling2DSparkResolver
{
    /// <summary>Env var that turns the sibling auto-attach off (<c>DOTLLM_DSPARK=0</c>); an explicit <c>--dspark</c> still attaches.</summary>
    public const string EnvVar = "DOTLLM_DSPARK";

    /// <summary>False when <c>DOTLLM_DSPARK=0</c>.</summary>
    public static bool AutoAttachEnabled => Environment.GetEnvironmentVariable(EnvVar) != "0";

    /// <summary>The <c>dspark-*.gguf</c> next to <paramref name="modelPath"/> (its directory, then the one above); the largest wins. Null when none.</summary>
    /// <param name="modelPath">Path of the trunk GGUF or its directory.</param>
    public static string? Find(string modelPath)
    {
        if (string.IsNullOrEmpty(modelPath)) return null;
        string? dir = Directory.Exists(modelPath) ? modelPath : Path.GetDirectoryName(Path.GetFullPath(modelPath));
        for (int up = 0; up < 2 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            if (!Directory.Exists(dir)) continue;
            string? best = null;
            long bestSize = -1;
            foreach (string f in Directory.EnumerateFiles(dir, "dspark-*.gguf"))
            {
                long size = new FileInfo(f).Length;
                if (size > bestSize) { best = f; bestSize = size; }
            }
            if (best is not null) return best;
        }
        return null;
    }

    /// <summary>
    /// Attaches the drafter to <paramref name="model"/> when it is an Inkling 2 target without one.
    /// </summary>
    /// <param name="model">The loaded model; other types are left alone (an explicit path then throws).</param>
    /// <param name="modelPath">Trunk path for auto-detection.</param>
    /// <param name="explicitPath">An explicit drafter path; null = auto-detect (and honour <see cref="AutoAttachEnabled"/>).</param>
    /// <returns>The attached path, or null when nothing was attached.</returns>
    /// <exception cref="FileNotFoundException">An explicit path does not exist.</exception>
    /// <exception cref="NotSupportedException">An explicit path was given for a model that cannot take it, or the file is not an Inkling 2 DSpark drafter.</exception>
    public static string? TryAttach(IModel model, string modelPath, string? explicitPath = null)
    {
        if (model is not IDFlashDraftAttachable target || target.HasDFlashDrafter || model is not Inkling2TransformerModel)
        {
            if (explicitPath is not null) throw new NotSupportedException($"--dspark: {model.GetType().Name} cannot take an Inkling 2 DSpark drafter.");
            return null;
        }
        string? path = explicitPath;
        if (path is null)
        {
            if (!AutoAttachEnabled) return null;
            path = Find(modelPath);
            if (path is null) return null;
        }
        else if (!File.Exists(path)) throw new FileNotFoundException("DSpark drafter GGUF not found.", path);
        target.AttachDFlashDrafter(path);
        return path;
    }
}
