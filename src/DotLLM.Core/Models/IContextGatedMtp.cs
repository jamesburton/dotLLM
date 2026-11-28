namespace DotLLM.Core.Models;

/// <summary>
/// A model whose speculative drafter is only worth its memory and its calibration cost above some context length (the Vulkan Inkling 2 DSpark drafter, issue #2046:
/// it never won below several thousand tokens). The engine asks per request, with the request's context (prompt tokens + tokens to generate), whether MTP may run;
/// the model attaches the drafter on demand and answers false below its threshold, in which case nothing is allocated and the request decodes plainly.
/// </summary>
public interface IContextGatedMtp
{
    /// <summary>
    /// Called once per request before the MTP decision. Attaches the drafter when <paramref name="contextTokens"/> reaches the model's threshold (or when it was attached explicitly,
    /// which forces it on at every length) and returns whether this request may speculate.
    /// </summary>
    /// <param name="contextTokens">Prompt tokens plus the generation budget.</param>
    bool PrepareMtpFor(int contextTokens);
}
