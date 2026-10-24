namespace DotLLM.Core.Models;

/// <summary>
/// A target model that can drive a DFlash / DFlash2 block-diffusion drafter (issue #1175): a separate <c>dflash</c> GGUF that
/// conditions on the target's intermediate layer inputs and drafts a whole block in one pass. Once attached the drafter takes
/// the model's MTP slot (<see cref="IModel.SupportsMtp"/> is true and <see cref="IModel.CreateMtpState()"/> returns its state),
/// so the MTP speculative decoder verifies its drafts.
/// </summary>
public interface IDFlashDraftAttachable
{
    /// <summary>True once a drafter is attached.</summary>
    bool HasDFlashDrafter { get; }

    /// <summary>Opens <paramref name="path"/> as a <c>dflash</c> drafter GGUF and attaches it; the model owns the file.</summary>
    /// <param name="path">Drafter GGUF path.</param>
    void AttachDFlashDrafter(string path);
}
