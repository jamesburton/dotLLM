using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

public class JinjaChatTemplateTests
{
    // Normalize line endings: raw string literals on Windows contain \r\n,
    // but real GGUF templates always use \n.
    private static string Normalize(string s) => s.Replace("\r\n", "\n");

    // ── Real template strings from popular models ──

    // ChatML format used by Qwen2, many models
    private const string ChatMlTemplate =
        """
        {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
        """;

    // Llama 3.1 Instruct template (simplified but representative)
    private const string Llama3Template =
        """
        {{- bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>

        {{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>

        {{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>

        {{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

        {% endif %}
        """;

    // Mistral Instruct template
    private const string MistralTemplate =
        """
        {{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %} {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}
        """;

    // SmolLM-style template (ChatML variant)
    private const string SmolLmTemplate =
        """
        {% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
        You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
        {% endif %}<|im_start|>{{ message['role'] }}
        {{ message['content'] }}<|im_end|>
        {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        {% endif %}
        """;

    // Template with namespace() for loop state (Llama 3.1 pattern)
    private const string NamespaceTemplate =
        """
        {%- set ns = namespace(has_system=false) -%}
        {%- for message in messages -%}
        {%- if message['role'] == 'system' -%}
        {%- set ns.has_system = true -%}
        {%- endif -%}
        {%- endfor -%}
        {%- if not ns.has_system -%}
        <|system|>You are a helpful assistant.<|end|>
        {% endif -%}
        {%- for message in messages -%}
        <|{{ message['role'] }}|>{{ message['content'] }}<|end|>
        {% endfor -%}
        {%- if add_generation_prompt -%}
        <|assistant|>
        {%- endif -%}
        """;

    // Real chat_template.jinja from SyzygyResearch/Mach-1-Additive-35B (Qwen3.5-MoE family),
    // fetched verbatim from the HF repo 2026-08-07 for issue #273. Defines and calls a
    // {% macro render_content(content, do_vision_count, is_system_content=false) %} — the exact
    // macro-with-defaults, closure-over-namespace, multi-call-site shape the issue reports
    // JinjaParser previously crashed on (issue #266 quotes this same "render_content" macro as
    // the real-world trigger). This is the actual declared GGUF chat_template a `dotllm serve`
    // process would hit for this model family, not a synthetic stand-in.
    private const string Mach1TemplateText =
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
        {%- if not messages %}
            {{- raise_exception('No messages provided.') }}
        {%- endif %}
        {%- if tools and tools is iterable and tools is not mapping %}
            {{- '<|im_start|>system\n' }}
            {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
            {%- for tool in tools %}
                {{- "\n" }}
                {{- tool | tojson }}
            {%- endfor %}
            {{- "\n</tools>" }}
            {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
            {%- if messages[0].role == 'system' %}
                {%- set content = render_content(messages[0].content, false, true)|trim %}
                {%- if content %}
                    {{- '\n\n' + content }}
                {%- endif %}
            {%- endif %}
            {{- '<|im_end|>\n' }}
        {%- else %}
            {%- if messages[0].role == 'system' %}
                {%- set content = render_content(messages[0].content, false, true)|trim %}
                {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
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
                {%- else %}
                    {%- if '</think>' in content %}
                        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                    {%- endif %}
                {%- endif %}
                {%- set reasoning_content = reasoning_content|trim %}
                {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
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
                        {%- if tool_call.arguments is defined %}
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

    // ── ChatML tests ──

    [Fact]
    public void ChatML_UserAssistant_SimpleConversation()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello!" },
            new() { Role = "assistant", Content = "Hi there!" },
            new() { Role = "user", Content = "How are you?" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Contains("<|im_start|>user\nHello!<|im_end|>", result, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>assistant\nHi there!<|im_end|>", result, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>user\nHow are you?<|im_end|>", result, StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>assistant\n", result, StringComparison.Ordinal);
    }

    [Fact]
    public void ChatML_NoGenerationPrompt()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });

        Assert.Contains("<|im_start|>user\nHello!<|im_end|>", result, StringComparison.Ordinal);
        Assert.DoesNotContain("<|im_start|>assistant", result, StringComparison.Ordinal);
    }

    [Fact]
    public void ChatML_WithSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are helpful." },
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Contains("<|im_start|>system\nYou are helpful.<|im_end|>", result, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>user\nHello!<|im_end|>", result, StringComparison.Ordinal);
    }

    // ── Llama 3.1 tests ──

    [Fact]
    public void Llama3_BasicConversation()
    {
        var template = new JinjaChatTemplate(Normalize(Llama3Template), "<|begin_of_text|>", "<|eot_id|>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are helpful." },
            new() { Role = "user", Content = "What is 2+2?" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.StartsWith("<|begin_of_text|>", result, StringComparison.Ordinal);
        Assert.Contains("<|start_header_id|>system<|end_header_id|>", result, StringComparison.Ordinal);
        Assert.Contains("You are helpful.", result, StringComparison.Ordinal);
        Assert.Contains("<|start_header_id|>user<|end_header_id|>", result, StringComparison.Ordinal);
        Assert.Contains("What is 2+2?", result, StringComparison.Ordinal);
        Assert.Contains("<|start_header_id|>assistant<|end_header_id|>", result, StringComparison.Ordinal);
    }

    [Fact]
    public void Llama3_MultiTurn()
    {
        var template = new JinjaChatTemplate(Normalize(Llama3Template), "<|begin_of_text|>", "<|eot_id|>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi" },
            new() { Role = "assistant", Content = "Hello!" },
            new() { Role = "user", Content = "Bye" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.StartsWith("<|begin_of_text|>", result, StringComparison.Ordinal);
        // All messages should appear in order
        int userHi = result.IndexOf("Hi", StringComparison.Ordinal);
        int assistantHello = result.IndexOf("Hello!", StringComparison.Ordinal);
        int userBye = result.IndexOf("Bye", StringComparison.Ordinal);
        Assert.True(userHi < assistantHello);
        Assert.True(assistantHello < userBye);
    }

    // ── Mistral tests ──

    [Fact]
    public void Mistral_UserAssistant()
    {
        var template = new JinjaChatTemplate(Normalize(MistralTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is AI?" },
            new() { Role = "assistant", Content = "AI is artificial intelligence." },
            new() { Role = "user", Content = "Tell me more." },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });

        Assert.StartsWith("<s>", result, StringComparison.Ordinal);
        Assert.Contains("[INST] What is AI? [/INST]", result, StringComparison.Ordinal);
        Assert.Contains("AI is artificial intelligence.</s>", result, StringComparison.Ordinal);
        Assert.Contains("[INST] Tell me more. [/INST]", result, StringComparison.Ordinal);
    }

    // ── SmolLM tests ──

    [Fact]
    public void SmolLM_DefaultSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(SmolLmTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        // When no system message, SmolLM inserts a default system message
        Assert.Contains("<|im_start|>system\nYou are a helpful AI assistant named SmolLM", result, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>user\nHello!", result, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>assistant\n", result, StringComparison.Ordinal);
    }

    [Fact]
    public void SmolLM_WithExplicitSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(SmolLmTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are a poet." },
            new() { Role = "user", Content = "Write a haiku." },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        // Should NOT have the default system message
        Assert.DoesNotContain("SmolLM", result, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>system\nYou are a poet.", result, StringComparison.Ordinal);
    }

    // ── Namespace pattern tests ──

    [Fact]
    public void Namespace_DetectsSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(NamespaceTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "Custom system." },
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        // Should NOT have default system because we provided one
        Assert.DoesNotContain("You are a helpful assistant.", result, StringComparison.Ordinal);
        Assert.Contains("<|system|>Custom system.<|end|>", result, StringComparison.Ordinal);
        Assert.Contains("<|user|>Hello!<|end|>", result, StringComparison.Ordinal);
        Assert.Contains("<|assistant|>", result, StringComparison.Ordinal);
    }

    [Fact]
    public void Namespace_InjectsDefaultSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(NamespaceTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Contains("<|system|>You are a helpful assistant.<|end|>", result, StringComparison.Ordinal);
        Assert.Contains("<|user|>Hello!<|end|>", result, StringComparison.Ordinal);
    }

    // ── Edge cases ──

    [Fact]
    public void EmptyMessages()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>();

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Equal("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void SingleUserMessage()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", result);
    }

    [Fact]
    public void BosAndEosTokens_Accessible()
    {
        // Template that uses bos_token and eos_token
        const string template = "{{ bos_token }}hello{{ eos_token }}";
        var tmpl = new JinjaChatTemplate(template, "<BOS>", "<EOS>");
        var result = tmpl.Apply(new List<ChatMessage>(), new ChatTemplateOptions());

        Assert.Equal("<BOS>hello<EOS>", result);
    }

    // ── Tool definitions ──

    [Fact]
    public void Tools_AvailableInContext()
    {
        const string template = "{% if tools %}Tools: {{ tools | length }}{% else %}No tools{% endif %}";
        var tmpl = new JinjaChatTemplate(template, "<s>", "</s>");

        var withTools = tmpl.Apply(
            new List<ChatMessage>(),
            new ChatTemplateOptions
            {
                Tools = [new ToolDefinition("get_weather", "Get weather info", """{"type":"object"}""")]
            });
        Assert.Equal("Tools: 1", withTools);

        var withoutTools = tmpl.Apply(
            new List<ChatMessage>(),
            new ChatTemplateOptions());
        Assert.Equal("No tools", withoutTools);
    }

    // ── Whitespace control in real templates ──

    [Fact]
    public void WhitespaceControl_InForLoop()
    {
        const string template = "{%- for msg in messages -%}[{{ msg.role }}]{{ msg.content }}{%- endfor -%}";
        var tmpl = new JinjaChatTemplate(template, "", "");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "a" },
            new() { Role = "assistant", Content = "b" },
        };

        var result = tmpl.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });
        Assert.Equal("[user]a[assistant]b", result);
    }

    // ── Filter chains in templates ──

    [Fact]
    public void FilterChain_TrimThenTojson()
    {
        const string template = "{{ '  hello  ' | trim | tojson }}";
        var tmpl = new JinjaChatTemplate(template, "", "");
        var result = tmpl.Apply(new List<ChatMessage>(), new ChatTemplateOptions());
        Assert.Equal("\"hello\"", result);
    }

    // ── Not in operator ──

    [Fact]
    public void NotInOperator()
    {
        const string template = "{% if 'x' not in items %}missing{% else %}found{% endif %}";
        var tmpl = new JinjaChatTemplate(template, "", "");

        // Can't easily set custom vars through Apply, so test through evaluator directly
        var lexer = new JinjaLexer(template);
        var tokens = lexer.Tokenize();
        var parser = new JinjaParser(tokens);
        var ast = parser.Parse();

        var eval = new JinjaEvaluator(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["items"] = new List<object?> { "a", "b" },
            ["messages"] = new List<object?>(),
            ["add_generation_prompt"] = false,
        });
        var result = eval.Evaluate(ast);
        Assert.Equal("missing", result);
    }

    // ── Tool call round-trip ──

    [Fact]
    public void ToolCalls_InAssistantMessage_RenderedInContext()
    {
        // Template that renders tool_calls from assistant messages
        const string template =
            "{%- for msg in messages -%}" +
            "[{{ msg.role }}]" +
            "{% if msg.tool_calls %}CALLS:{% for tc in msg.tool_calls %}{{ tc.function.name }}{% endfor %}" +
            "{% else %}{{ msg.content }}" +
            "{% endif %}" +
            "{%- endfor -%}";

        var tmpl = new JinjaChatTemplate(template, "", "");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What's the weather?" },
            new()
            {
                Role = "assistant", Content = "",
                ToolCalls = [new ToolCall("call_0", "get_weather", """{"location":"Paris"}""")]
            },
        };

        var result = tmpl.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });

        Assert.Contains("[assistant]CALLS:get_weather", result, StringComparison.Ordinal);
    }

    [Fact]
    public void ToolResult_Message_RenderedInContext()
    {
        // Template that renders tool result messages
        const string template =
            "{%- for msg in messages -%}" +
            "[{{ msg.role }}{% if msg.tool_call_id %}:{{ msg.tool_call_id }}{% endif %}]{{ msg.content }}" +
            "{%- endfor -%}";

        var tmpl = new JinjaChatTemplate(template, "", "");
        var messages = new List<ChatMessage>
        {
            new() { Role = "tool", Content = """{"temp":22}""", ToolCallId = "call_0" },
        };

        var result = tmpl.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });

        Assert.Contains("[tool:call_0]", result, StringComparison.Ordinal);
        Assert.Contains("""{"temp":22}""", result, StringComparison.Ordinal);
    }

    [Fact]
    public void Tools_Tojson_RendersToolDefinitions()
    {
        const string template = "{{ tools | tojson }}";
        var tmpl = new JinjaChatTemplate(template, "", "");

        var result = tmpl.Apply(
            new List<ChatMessage>(),
            new ChatTemplateOptions
            {
                Tools = [new ToolDefinition("search", "Search the web", """{"type":"object","properties":{"q":{"type":"string"}}}""")]
            });

        // Should contain the tool definition as JSON
        Assert.Contains("\"search\"", result, StringComparison.Ordinal);
        Assert.Contains("\"function\"", result, StringComparison.Ordinal);
    }

    [Fact]
    public void MultiTurn_ToolCallConversation()
    {
        // Simplified ChatML-like template
        const string template =
            "{%- for msg in messages -%}" +
            "<|{{ msg.role }}|>{{ msg.content }}" +
            "{% if msg.tool_calls %}{% for tc in msg.tool_calls %}[TC:{{ tc.function.name }}]{% endfor %}{% endif %}" +
            "<|end|>" +
            "{%- endfor -%}" +
            "{% if add_generation_prompt %}<|assistant|>{% endif %}";

        var tmpl = new JinjaChatTemplate(template, "", "");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Weather in Paris?" },
            new()
            {
                Role = "assistant", Content = "",
                ToolCalls = [new ToolCall("c0", "get_weather", """{"location":"Paris"}""")]
            },
            new() { Role = "tool", Content = """{"temp":22}""", ToolCallId = "c0" },
            new() { Role = "assistant", Content = "It's 22 degrees in Paris." },
        };

        var result = tmpl.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });

        Assert.Contains("<|user|>Weather in Paris?", result, StringComparison.Ordinal);
        Assert.Contains("[TC:get_weather]", result, StringComparison.Ordinal);
        Assert.Contains("<|tool|>", result, StringComparison.Ordinal);
        Assert.Contains("It's 22 degrees", result, StringComparison.Ordinal);
    }

    // ── Real-world macro template (#273): SyzygyResearch/Mach-1-Additive-35B chat_template.jinja ──
    // Before the fix, constructing JinjaChatTemplate with this template threw JinjaException
    // ("Unexpected statement keyword: Macro") from the JinjaParser constructor call — which is
    // exactly the exception ServerStartup.LoadModel previously let crash the whole server process.
    // These exercise the macro actually being called (not just parsed): render_content is invoked
    // 2-3x per conversation with different argument counts (2-arg call relies on the
    // is_system_content=false default; the 3-arg call overrides it).

    [Fact]
    public void Mach1Macro_ConstructsWithoutThrowing()
    {
        // The regression this issue is about: this used to throw during the constructor
        // (lexing/parsing happens eagerly, not lazily on first Apply()).
        var exception = Record.Exception(() => new JinjaChatTemplate(Normalize(Mach1TemplateText), "<s>", "</s>"));
        Assert.Null(exception);
    }

    [Fact]
    public void Mach1Macro_SystemAndUserMessage_RendersContentThroughMacro()
    {
        var template = new JinjaChatTemplate(Normalize(Mach1TemplateText), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are a helpful assistant." },
            new() { Role = "user", Content = "What is 2+2?" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        // System content goes through render_content(messages[0].content, false, true) in the
        // preamble (3-arg call, overriding the is_system_content default) — the macro's
        // `content is string` branch must have executed and echoed it back out.
        Assert.Contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", result, StringComparison.Ordinal);

        // User content goes through render_content(message.content, true) in the main loop
        // (2-arg call — relies on the is_system_content=false default parameter).
        Assert.Contains("<|im_start|>user\nWhat is 2+2?<|im_end|>\n", result, StringComparison.Ordinal);

        // add_generation_prompt path; enable_thinking is never set by BuildContext, so
        // `enable_thinking is defined` is false and the else branch ('<think>\n') applies.
        Assert.EndsWith("<|im_start|>assistant\n<think>\n", result, StringComparison.Ordinal);

        // The system message must not also produce a duplicate <|im_start|>system block from the
        // main for-loop pass (the template's own `if not loop.first: raise_exception(...)` guard,
        // and the fact the system branch of the main loop emits nothing for loop.first).
        Assert.Equal(1, CountOccurrences(result, "<|im_start|>system"));
    }

    [Fact]
    public void Mach1Macro_NoSystemMessage_PreambleSkipsMacroCallCleanly()
    {
        // messages[0].role != 'system' -> the preamble's inner `if messages[0].role == 'system'`
        // is false, so render_content is never called from the preamble at all for this case.
        var template = new JinjaChatTemplate(Normalize(Mach1TemplateText), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello there" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.DoesNotContain("<|im_start|>system", result, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>user\nHello there<|im_end|>\n", result, StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n", result, StringComparison.Ordinal);
    }

    [Fact]
    public void Mach1Macro_MultiTurn_AssistantReplyRendersViaMacro()
    {
        // Adds an assistant turn so render_content is also exercised for message.role == "assistant"
        // (loop.index0 <= ns.last_query_index branch — reasoning_content handling, no <think> block
        // since there's no '</think>' in the content and reasoning_content is undefined).
        var template = new JinjaChatTemplate(Normalize(Mach1TemplateText), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "Be concise." },
            new() { Role = "user", Content = "Hi" },
            new() { Role = "assistant", Content = "Hello! How can I help?" },
            new() { Role = "user", Content = "What's 2+2?" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });

        Assert.Contains("<|im_start|>system\nBe concise.<|im_end|>\n", result, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>user\nHi<|im_end|>\n", result, StringComparison.Ordinal);
        Assert.Contains("Hello! How can I help?", result, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>user\nWhat's 2+2?<|im_end|>\n", result, StringComparison.Ordinal);

        // Ordering: messages must render in conversation order.
        int sysIdx = result.IndexOf("Be concise.", StringComparison.Ordinal);
        int userHiIdx = result.IndexOf("Hi<|im_end|>", StringComparison.Ordinal);
        int assistantIdx = result.IndexOf("Hello! How can I help?", StringComparison.Ordinal);
        int userLastIdx = result.IndexOf("What's 2+2?", StringComparison.Ordinal);
        Assert.True(sysIdx < userHiIdx);
        Assert.True(userHiIdx < assistantIdx);
        Assert.True(assistantIdx < userLastIdx);
    }

    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0, index = 0;
        while ((index = haystack.IndexOf(needle, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += needle.Length;
        }
        return count;
    }
}
