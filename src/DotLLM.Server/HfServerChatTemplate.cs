using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

namespace DotLLM.Server;

/// <summary>Chooses the checkpoint's precompiled default or tool template for each request.</summary>
internal sealed class HfServerChatTemplate(IChatTemplate defaultTemplate, JinjaChatTemplate toolTemplate) : IChatTemplate
{
    internal IChatTemplate Default => defaultTemplate;
    internal JinjaChatTemplate Tool => toolTemplate;

    public string Apply(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options)
        => Select(this, options.Tools is { Length: > 0 })!.Apply(messages, options);

    internal static IChatTemplate? Select(IChatTemplate? template, bool useTools)
        => template is HfServerChatTemplate hf ? useTools ? hf.Tool : hf.Default : template;
}
