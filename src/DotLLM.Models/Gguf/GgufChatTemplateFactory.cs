using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Bridge between GGUF metadata and the Jinja2 chat template engine.
/// Creates a <see cref="JinjaChatTemplate"/> from model metadata and tokenizer info.
/// </summary>
public static class GgufChatTemplateFactory
{
    /// <summary>
    /// Conservative fallback for base models that do not declare a GGUF chat template.
    /// It avoids model-specific control tokens such as ChatML's &lt;|im_start|&gt;.
    /// </summary>
    public const string PlainCompletionTemplateText =
        "{% for message in messages %}" +
        "{% if message['role'] == 'system' %}{{ message['content'] + '\\n\\n' }}" +
        "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n' }}" +
        "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\\n' }}" +
        "{% elif message['role'] == 'tool' %}{{ 'Tool: ' + message['content'] + '\\n' }}" +
        "{% endif %}" +
        "{% endfor %}" +
        "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}";

    /// <summary>
    /// Tool-aware Jinja2 chat template for BitNet b1.58.
    /// Preserves BitNet's <c>User:/Assistant:/&lt;|eot_id|&gt;</c> turn format and injects
    /// a Hermes-style <c>&lt;tools&gt;/&lt;tool_call&gt;</c> preamble when tools are present.
    /// Source of truth: kept byte-identical to <c>scripts/lora/templates/bitnet_tooluse.jinja</c>
    /// and guarded by <c>BitNetToolFormatVerificationTests</c>.
    /// </summary>
    public const string BitNetToolAwareTemplateText =
        "{%- if tools %}\n" +
        "{%- if messages | length > 0 and messages[0].role == 'system' %}\n" +
        "{{- 'System: ' + messages[0].content | trim + '\\n\\n' }}\n" +
        "{%- endif %}\n" +
        "{{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n" +
        "{%- for tool in tools %}\n" +
        "{{- \"\\n\" }}\n" +
        "{{- tool | tojson }}\n" +
        "{%- endfor %}\n" +
        "{{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call>\" }}\n" +
        "{{- '<|eot_id|>' }}\n" +
        "{%- else %}\n" +
        "{%- if messages | length > 0 and messages[0].role == 'system' %}\n" +
        "{{- 'System: ' + messages[0].content | trim + '<|eot_id|>' }}\n" +
        "{%- endif %}\n" +
        "{%- endif %}\n" +
        "{%- for message in messages %}\n" +
        "{%- if message.role == 'user' %}\n" +
        "{{- 'User: ' + message.content | trim + '<|eot_id|>' }}\n" +
        "{%- elif message.role == 'assistant' %}\n" +
        "{{- 'Assistant: ' + message.content | trim + '<|eot_id|>' }}\n" +
        "{%- elif message.role == 'tool' %}\n" +
        "{{- 'Tool: ' + message.content | trim + '<|eot_id|>' }}\n" +
        "{%- endif %}\n" +
        "{%- endfor %}\n" +
        "{%- if add_generation_prompt %}\n" +
        "{{- 'Assistant: ' }}\n" +
        "{%- endif %}";

    /// <summary>
    /// Tries to create a <see cref="JinjaChatTemplate"/> from GGUF metadata.
    /// Returns null if no chat template is present in the metadata.
    /// </summary>
    /// <param name="metadata">GGUF metadata containing the template string and token info.</param>
    /// <param name="tokenizer">Tokenizer for resolving BOS/EOS token strings.</param>
    public static JinjaChatTemplate? TryCreate(GgufMetadata metadata, ITokenizer tokenizer)
    {
        string? template = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(template))
            return null;

        string bosToken = tokenizer.DecodeToken(tokenizer.BosTokenId);
        string eosToken = tokenizer.DecodeToken(tokenizer.EosTokenId);
        return new JinjaChatTemplate(template, bosToken, eosToken);
    }

    /// <summary>
    /// Tries to create a <see cref="JinjaChatTemplate"/> from GGUF metadata.
    /// Returns null if no chat template is present in the metadata.
    /// For <see cref="Architecture.BitNet"/>, substitutes the built-in tool-aware template
    /// so that <c>--tools</c> injects tool definitions correctly.
    /// </summary>
    /// <param name="metadata">GGUF metadata containing the template string and token info.</param>
    /// <param name="tokenizer">Tokenizer for resolving BOS/EOS token strings.</param>
    /// <param name="architecture">Model architecture (used for per-arch template substitution).</param>
    public static JinjaChatTemplate? TryCreate(GgufMetadata metadata, ITokenizer tokenizer,
        Architecture architecture)
    {
        string bosToken = tokenizer.DecodeToken(tokenizer.BosTokenId);
        string eosToken = tokenizer.DecodeToken(tokenizer.EosTokenId);

        // Approach A: BitNet's built-in GGUF template has no tool support.
        // Substitute the tool-aware template unconditionally for BitNet.
        if (architecture == Architecture.BitNet)
            return new JinjaChatTemplate(BitNetToolAwareTemplateText, bosToken, eosToken);

        string? template = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(template))
            return null;

        return new JinjaChatTemplate(template, bosToken, eosToken);
    }

    /// <summary>
    /// Creates a plain text completion-style fallback template for models without chat metadata.
    /// </summary>
    /// <param name="tokenizer">Tokenizer for resolving BOS/EOS token strings.</param>
    public static JinjaChatTemplate CreatePlainFallback(ITokenizer tokenizer)
        => new(PlainCompletionTemplateText,
            tokenizer.DecodeToken(tokenizer.BosTokenId),
            tokenizer.DecodeToken(tokenizer.EosTokenId));

    /// <summary>
    /// Tries to create a <see cref="JinjaChatTemplate"/> from a ModelConfig.
    /// Returns null if no chat template is present in the config.
    /// </summary>
    /// <param name="config">Model configuration with chat template string.</param>
    /// <param name="bosToken">BOS token string.</param>
    /// <param name="eosToken">EOS token string.</param>
    public static JinjaChatTemplate? TryCreate(ModelConfig config, string bosToken, string eosToken)
    {
        if (string.IsNullOrEmpty(config.ChatTemplate))
            return null;

        return new JinjaChatTemplate(config.ChatTemplate, bosToken, eosToken);
    }

    /// <summary>
    /// Creates a tool call parser appropriate for the model based on architecture
    /// and chat template content.
    /// </summary>
    /// <param name="metadata">GGUF metadata for template heuristics.</param>
    /// <param name="architecture">Model architecture.</param>
    /// <returns>A tool call parser for this model.</returns>
    public static IToolCallParser CreateToolCallParser(GgufMetadata metadata, Architecture architecture)
    {
        string? template = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        return ToolCallParserFactory.Create(architecture, template);
    }

    /// <summary>
    /// Creates a tool call parser appropriate for the model based on architecture
    /// and chat template content.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <returns>A tool call parser for this model.</returns>
    public static IToolCallParser CreateToolCallParser(ModelConfig config)
        => ToolCallParserFactory.Create(config.Architecture, config.ChatTemplate);
}
