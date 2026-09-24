using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Integration.Tokenizers.ChatTemplates;

/// <summary>
/// Verifies the real <c>Qwen/Qwen3.8-27B</c> <c>chat_template.jinja</c> tool-response branch
/// renders correctly through the Jinja2-subset interpreter (issue #399). That branch is the
/// motivating case for <c>loop.previtem</c> / <c>loop.nextitem</c> support: it merges
/// consecutive <c>role == "tool"</c> messages into a single <c>&lt;|im_start|&gt;user ... &lt;|im_end|&gt;</c>
/// turn by checking the neighbouring message's role —
///
/// <code>
/// {%- elif message.role == "tool" %}
///     {%- if loop.previtem and loop.previtem.role != "tool" %}
///         {{- '&lt;|im_start|&gt;user' }}
///     {%- endif %}
///     {{- '\n&lt;tool_response&gt;\n' }}
///     {{- content }}
///     {{- '\n&lt;/tool_response&gt;' }}
///     {%- if not loop.last and loop.nextitem.role != "tool" %}
///         {{- '&lt;|im_end|&gt;\n' }}
///     {%- elif loop.last %}
///         {{- '&lt;|im_end|&gt;\n' }}
///     {%- endif %}
/// {%- endif %}
/// </code>
///
/// Without <c>loop.previtem</c>/<c>loop.nextitem</c> this branch cannot render at all, so tool
/// calling was unavailable for the model even though plain text generation worked.
/// </summary>
/// <remarks>
/// <see cref="TemplateExcerpt"/> is the <b>verbatim</b> text of
/// https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja (fetched 2026-08-14),
/// lines 1-41 (the <c>render_content</c> macro) and lines 88-170 (the multi-step-tool detection
/// loop, the per-message render loop containing the branch above, and the generation-prompt
/// tail) concatenated with no other edits. Lines 42-87 — the reasoning-effort / <c>tools</c>
/// system-prompt prelude, unconditional but unrelated to the tool-response branch — are excised
/// because they contain <c>not in ('xhigh', 'medium', 'low')</c>, a parenthesized tuple literal
/// the evaluator's parser does not yet support (tracked separately as issue #409; the full,
/// unmodified template still fails to parse until that lands). The excised region is not
/// exercised by any assertion below.
/// </remarks>
public sealed class Qwen38ChatTemplateTests
{
    // ── Verbatim excerpt of Qwen/Qwen3.8-27B's chat_template.jinja — see <remarks> above ──
    private const string TemplateExcerpt =
        """
        {%- set image_count = namespace(value=0) %}
        {%- set video_count = namespace(value=0) %}
        {%- macro render_content(content, do_vision_count, is_system_content=false) %}
            {%- if content is string %}
                {{- content }}
            {%- elif content is iterable and content is not mapping %}
                {%- for item in content %}
                    {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                        {%- if is_system_content %}
                            {{- raise_exception('System message cannot contain images.') }}
                        {%- endif %}
                        {%- if do_vision_count %}
                            {%- set image_count.value = image_count.value + 1 %}
                        {%- endif %}
                        {%- if add_vision_id %}
                            {{- 'Picture ' ~ image_count.value ~ ': ' }}
                        {%- endif %}
                        {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
                    {%- elif 'video' in item or item.type == 'video' %}
                        {%- if is_system_content %}
                            {{- raise_exception('System message cannot contain videos.') }}
                        {%- endif %}
                        {%- if do_vision_count %}
                            {%- set video_count.value = video_count.value + 1 %}
                        {%- endif %}
                        {%- if add_vision_id %}
                            {{- 'Video ' ~ video_count.value ~ ': ' }}
                        {%- endif %}
                        {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
                    {%- elif 'text' in item %}
                        {{- item.text }}
                    {%- else %}
                        {{- raise_exception('Unexpected item type in content.') }}
                    {%- endif %}
                {%- endfor %}
            {%- elif content is none or content is undefined %}
                {{- '' }}
            {%- else %}
                {{- raise_exception('Unexpected content type.') }}
            {%- endif %}
        {%- endmacro %}
        {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
        {%- for message in messages[::-1] %}
            {%- set index = (messages|length - 1) - loop.index0 %}
            {%- if ns.multi_step_tool and message.role == "user" %}
                {%- set content = render_content(message.content, false)|trim %}
                {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
                    {%- set ns.multi_step_tool = false %}
                    {%- set ns.last_query_index = index %}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
        {%- if ns.multi_step_tool %}
            {{- raise_exception('No user query found in messages.') }}
        {%- endif %}
        {%- for message in messages %}
            {%- set content = render_content(message.content, true)|trim %}
            {%- if message.role == "system" %}
                {%- if not loop.first %}
                    {{- raise_exception('System message must be at the beginning.') }}
                {%- endif %}
            {%- elif message.role == "user" %}
                {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
            {%- elif message.role == "assistant" %}
                {%- set reasoning_content = '' %}
                {%- if message.reasoning_content is string %}
                    {%- set reasoning_content = message.reasoning_content %}
                {%- endif %}
                {%- set reasoning_content = reasoning_content|trim %}
                {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
                    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
                {%- else %}
                    {{- '<|im_start|>' + message.role + '\n' + content }}
                {%- endif %}
                {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
                    {%- for tool_call in message.tool_calls %}
                        {%- if tool_call.function is defined %}
                            {%- set tool_call = tool_call.function %}
                        {%- endif %}
                        {%- if loop.first %}
                            {%- if content|trim %}
                                {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                            {%- else %}
                                {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                            {%- endif %}
                        {%- else %}
                            {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                        {%- endif %}
                        {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                            {%- for args_name, args_value in tool_call.arguments|items %}
                                {{- '<parameter=' + args_name + '>\n' }}
                                {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                                {{- args_value }}
                                {{- '\n</parameter>\n' }}
                            {%- endfor %}
                        {%- endif %}
                        {{- '</function>\n</tool_call>' }}
                    {%- endfor %}
                {%- endif %}
                {{- '<|im_end|>\n' }}
            {%- elif message.role == "tool" %}
                {%- if loop.previtem and loop.previtem.role != "tool" %}
                    {{- '<|im_start|>user' }}
                {%- endif %}
                {{- '\n<tool_response>\n' }}
                {{- content }}
                {{- '\n</tool_response>' }}
                {%- if not loop.last and loop.nextitem.role != "tool" %}
                    {{- '<|im_end|>\n' }}
                {%- elif loop.last %}
                    {{- '<|im_end|>\n' }}
                {%- endif %}
            {%- else %}
                {{- raise_exception('Unexpected message role.') }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
            {%- else %}
                {{- '<think>\n' }}
            {%- endif %}
        {%- endif %}
        """;

    private static JinjaChatTemplate CreateTemplate() =>
        new(TemplateExcerpt, bosToken: "", eosToken: "<|im_end|>");

    [Fact]
    public void RealChatTemplate_ConsecutiveToolResponses_MergeIntoSingleUserTurn()
    {
        var template = CreateTemplate();

        // Two tool calls in one assistant turn, answered by two consecutive tool messages.
        // loop.previtem on the SECOND tool message is the FIRST tool message (role == "tool"),
        // so the branch must NOT re-open "<|im_start|>user" — the two <tool_response> blocks
        // merge into a single user turn, closed only once loop.last (or loop.nextitem.role !=
        // "tool") is reached. A broken implementation (previtem always undefined, or previtem
        // always equal to the current item) would either re-open per message or never open at
        // all — both produce a different tag sequence than asserted below.
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What's the weather in Paris and Tokyo?" },
            new()
            {
                Role = "assistant",
                Content = "",
                ToolCalls =
                [
                    new ToolCall(Id: "call_1", FunctionName: "get_weather", Arguments: """{"location":"Paris"}"""),
                    new ToolCall(Id: "call_2", FunctionName: "get_weather", Arguments: """{"location":"Tokyo"}"""),
                ],
            },
            new() { Role = "tool", Content = "{\"temp_c\": 18}", ToolCallId = "call_1" },
            new() { Role = "tool", Content = "{\"temp_c\": 25}", ToolCallId = "call_2" },
        };

        var tools = new[]
        {
            new ToolDefinition(
                Name: "get_weather",
                Description: "Get current weather for a location.",
                ParametersSchema: """{"type":"object","properties":{"location":{"type":"string"}}}"""),
        };

        string rendered = template.Apply(messages,
            new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools });

        // Two "<|im_start|>user" openers total: one for the real initial user turn, and exactly
        // one (not two) covering both tool_response blocks — the second tool message's
        // loop.previtem.role == "tool" suppresses what would otherwise be a second opener per
        // tool message (which would make this 3, not 2).
        Assert.Equal(2, CountOccurrences(rendered, "<|im_start|>user"));
        Assert.Contains(
            "<|im_start|>user\n<tool_response>\n{\"temp_c\": 18}\n</tool_response>\n<tool_response>\n{\"temp_c\": 25}\n</tool_response><|im_end|>\n",
            rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void RealChatTemplate_SingleToolResponse_OpensAndClosesUserTurnInSameIteration()
    {
        var template = CreateTemplate();

        // A lone tool message: loop.previtem (the assistant message) has role != "tool", so the
        // opener renders; loop.last is true, so the closer renders in the same iteration. This
        // exercises loop.previtem being DEFINED (not the first-iteration-undefined case) and
        // loop.last / loop.nextitem's "undefined on the last iteration" boundary together —
        // reading loop.nextitem here must not throw.
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What's the weather in Paris?" },
            new()
            {
                Role = "assistant",
                Content = "",
                ToolCalls = [new ToolCall(Id: "call_1", FunctionName: "get_weather", Arguments: """{"location":"Paris"}""")],
            },
            new() { Role = "tool", Content = "{\"temp_c\": 18}", ToolCallId = "call_1" },
        };

        var tools = new[]
        {
            new ToolDefinition(
                Name: "get_weather",
                Description: "Get current weather for a location.",
                ParametersSchema: """{"type":"object","properties":{"location":{"type":"string"}}}"""),
        };

        string rendered = template.Apply(messages,
            new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools });

        Assert.Contains(
            "<|im_start|>user\n<tool_response>\n{\"temp_c\": 18}\n</tool_response><|im_end|>\n",
            rendered, StringComparison.Ordinal);
    }

    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0;
        int index = 0;
        while ((index = haystack.IndexOf(needle, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += needle.Length;
        }
        return count;
    }
}
