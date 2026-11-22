namespace DotLLM.Engine;

/// <summary>
/// Reason why token generation stopped.
/// </summary>
public enum FinishReason
{
    /// <summary>A stop condition was met (EOS token, stop sequence).</summary>
    Stop,

    /// <summary>Maximum token limit was reached.</summary>
    Length,

    /// <summary>Generation stopped to execute tool calls.</summary>
    ToolCalls,

    /// <summary>A <see cref="DotLLM.Core.Constraints.IDecodingConstraint"/> is satisfied and permits nothing but end-of-sequence, so a <see cref="GenerationSession"/> segment ended without sampling it (#1921).</summary>
    ConstraintComplete
}
