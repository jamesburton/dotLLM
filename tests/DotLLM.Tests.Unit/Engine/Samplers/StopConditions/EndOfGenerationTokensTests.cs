using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers.StopConditions;

/// <summary>
/// Issue #776: Gemma-4's GGUF declares eos_token_id=106 (&lt;turn|&gt;) but ends a tool-call turn with
/// &lt;eos&gt; (id 1), so generation ran to max_tokens and leaked "&lt;eos&gt;&lt;eos&gt;...". llama.cpp stops on every EOG token.
/// </summary>
public class EndOfGenerationTokensTests
{
    private sealed class FakeTokenizer(int eos, params (string Text, int Id)[] specials) : ITokenizer
    {
        public int[] Encode(string text)
        {
            foreach (var (t, id) in specials)
                if (t == text) return [id];
            return [.. text.Select(c => 1000 + c)]; // ordinary text: one id per char
        }
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Concat(tokenIds.ToArray().Select(DecodeToken));
        public string DecodeToken(int id) => specials.FirstOrDefault(s => s.Id == id).Text ?? ((char)(id - 1000)).ToString();
        public int VocabSize => 2000;
        public int BosTokenId => 2;
        public int EosTokenId => eos;
        public int CountTokens(string text) => Encode(text).Length;
    }

    [Fact]
    public void Gemma4_Resolves_DeclaredEosAndEosToken()
    {
        var tok = new FakeTokenizer(106, ("<turn|>", 106), ("<eos>", 1));

        int[] ids = EndOfGenerationTokens.Resolve(tok);

        Assert.Equal(106, ids[0]);              // declared EOS stays first
        Assert.Contains(1, ids);                // <eos>
        Assert.Equal(2, ids.Length);            // <turn|> is the declared one: not duplicated
    }

    [Fact]
    public void StopCondition_FiresOnAnyEogId_NotOnOrdinaryTokens()
    {
        var cond = EndOfGenerationTokens.CreateStopCondition(new FakeTokenizer(106, ("<turn|>", 106), ("<eos>", 1)));

        Assert.Equal(StopResult.Stop, cond.ShouldStop(106, [], ""));
        Assert.Equal(StopResult.Stop, cond.ShouldStop(1, [], ""));
        Assert.Equal(StopResult.Continue, cond.ShouldStop(1000 + 'a', [], "a"));
    }

    [Fact]
    public void CandidateThatIsOrdinaryText_IsNotAdded()
    {
        // A vocabulary where "<eos>" is NOT a single special token must not gain a stop id.
        int[] ids = EndOfGenerationTokens.Resolve(new FakeTokenizer(7));
        Assert.Equal([7], ids);
    }

    [Fact]
    public void ChatMlAndLlamaTurnEnders_AreRecognised()
    {
        var tok = new FakeTokenizer(128009, ("<|eot_id|>", 128009), ("<|eom_id|>", 128008), ("<|im_end|>", 5));
        int[] ids = EndOfGenerationTokens.Resolve(tok);
        Assert.Equal([128009, 128008, 5], ids.OrderByDescending(x => x == 128009).ThenByDescending(x => x).ToArray());
    }

    [Fact]
    public void SingleId_ConditionBehavesLikeTheOriginalEosCondition()
    {
        var cond = new EosStopCondition(9);
        Assert.Equal(StopResult.Stop, cond.ShouldStop(9, [], ""));
        Assert.Equal(StopResult.Continue, cond.ShouldStop(8, [], ""));
    }
}
