using System.Runtime.CompilerServices;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// End-to-end acceptance test for issue #409: the real, unmodified
/// <c>Qwen/Qwen3.8-27B</c> <c>chat_template.jinja</c> must at least PARSE successfully once the
/// tuple-literal grouping bug is fixed. The fixture is byte-for-byte what HuggingFace serves at
/// https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja (fetched 2026-08-14).
/// </summary>
public class JinjaQwen3_8_27BAcceptanceTests
{
    private static string LoadFixture([CallerFilePath] string callerFilePath = "")
    {
        var dir = Path.GetDirectoryName(callerFilePath)!;
        var path = Path.Combine(dir, "Fixtures", "qwen3.8-27b-chat-template.jinja");
        return File.ReadAllText(path);
    }

    [Fact]
    public void FullTemplate_Parses_WithoutThrowing()
    {
        // This is the exact regression from #409: before the tuple-literal fix, this throws
        // "Line 48, Col 53: Expected RightParen, got Comma" on
        // `resolved_reasoning_effort not in ('xhigh', 'medium', 'low')`, which sits in the
        // reasoning-instructions prelude that executes unconditionally for every render.
        var source = LoadFixture();
        var tokens = new JinjaLexer(source).Tokenize();
        var parser = new JinjaParser(tokens);
        var ast = parser.Parse();
        Assert.NotEmpty(ast.Nodes);
    }

    [Fact]
    public void FullTemplate_ConstructsAsJinjaChatTemplate_WithoutThrowing()
    {
        // JinjaChatTemplate's constructor lexes+parses eagerly; this is the same assertion as
        // above but through the public API callers actually use.
        var source = LoadFixture();
        _ = new JinjaChatTemplate(source, bosToken: "<|endoftext|>", eosToken: "<|im_end|>");
    }

    [Fact]
    public void FullTemplate_Render_KnownRemainingGapIsTracked()
    {
        // Rendering (as opposed to parsing) additionally requires #399's loop.previtem /
        // loop.nextitem support and `is undefined` handling — both used unconditionally in this
        // template (the reasoning-effort prelude at line 46: `enable_thinking is undefined or ...`,
        // and the tool-response run detection at lines 148/154: `loop.previtem` / `loop.nextitem`).
        // #399 is tracked separately by PR #411, which is NOT merged as of this branch.
        //
        // This test documents the current end-to-end state rather than silently skipping:
        //   - If #411 has NOT landed: rendering must fail, and it must fail for the KNOWN #399
        //     reason (an unsupported `is undefined`/`is defined`-style test name), not for a
        //     tuple/comma parsing reason — proving #409's fix is not masking a different bug.
        //   - If #411 HAS landed (merge this branch onto a newer `dev` and re-run): rendering
        //     should succeed outright; this test's `catch` branch will no longer be reached and
        //     the success path below is asserted instead.
        var source = LoadFixture();
        var template = new JinjaChatTemplate(source, bosToken: "<|endoftext|>", eosToken: "<|im_end|>");

        var messages = new[]
        {
            new ChatMessage { Role = "user", Content = "What is 2 + 2?" },
        };
        var options = new ChatTemplateOptions { AddGenerationPrompt = true };

        try
        {
            var result = template.Apply(messages, options);

            // #411 has landed (or the gap has otherwise closed) — full end-to-end render works.
            Assert.Contains("What is 2 + 2?", result);
        }
        catch (JinjaException ex)
        {
            // Must NOT be a tuple/grouping parse failure (that would mean #409 regressed).
            Assert.DoesNotContain("RightParen", ex.Message);
            Assert.DoesNotContain("got Comma", ex.Message);

            // Must be the known, tracked #399 gap: an "is undefined"-style test name the
            // evaluator doesn't recognize yet (see JinjaEvaluator.EvalIsTest).
            Assert.Contains("Unknown test", ex.Message);
        }
    }
}
