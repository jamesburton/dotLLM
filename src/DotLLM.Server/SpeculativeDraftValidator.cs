using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models;

namespace DotLLM.Server;

/// <summary>
/// A server option that cannot be honoured because of what the user asked for (not because of the device or the machine).
/// Carries a complete, actionable message; <c>serve</c> prints it as-is and never re-labels it as a device failure.
/// </summary>
public sealed class InvalidServerOptionException : InvalidOperationException
{
    /// <inheritdoc cref="InvalidServerOptionException"/>
    public InvalidServerOptionException(string message) : base(message) { }
}

/// <summary>
/// Validates a <c>--speculative-model</c> draft before any weights are loaded (#1319). The draft is loaded through the dense
/// Llama-style <c>TransformerModel</c>; a hybrid / recurrent architecture (qwen4exp, Qwen3.5/3.6 hybrids, Nemotron-H, LFM2-MoE)
/// would only fail there, after the (possibly 87 GiB) target is already resident, with a message that blames the target and the device.
/// </summary>
public static class SpeculativeDraftValidator
{
    /// <summary>
    /// Throws <see cref="InvalidServerOptionException"/> when <paramref name="draft"/> cannot serve as a speculative draft model.
    /// </summary>
    /// <param name="draft">The draft model's configuration.</param>
    /// <param name="draftPath">The draft file (used in the message and to recognise a standalone MTP head).</param>
    /// <param name="target">The target model's configuration (used to point at the right alternative).</param>
    public static void Validate(ModelConfig draft, string draftPath, ModelConfig target)
    {
        ArgumentNullException.ThrowIfNull(draft);
        ArgumentNullException.ThrowIfNull(target);

        string? why = ModelLoader.DenseLoaderUnsupportedReason(draft);
        if (why is null)
            return;

        string file = Path.GetFileName(draftPath);
        string advice = draft.Architecture == Architecture.Qwen4Exp
            ? "qwen4exp has no separate draft model: its speculation head ships as a standalone mtp-*.gguf beside the quantisation folders, "
              + "and serve attaches it automatically (or pass it with --mtp-head <path>; --no-mtp disables it). "
              + "Drop --speculative-model."
            : "Pick a dense (Llama/Qwen/Mistral-style) draft model with the same tokenizer, or drop --speculative-model.";

        throw new InvalidServerOptionException(
            $"--speculative-model '{file}' cannot be used as a draft model: its architecture {draft.Architecture} does not use the dense "
            + $"Llama-style layout the draft loader requires, because {why}. {advice} "
            + $"(Checked before loading the target {target.Architecture} model; nothing was loaded.)");
    }
}
