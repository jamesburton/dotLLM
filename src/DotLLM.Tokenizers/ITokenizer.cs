namespace DotLLM.Tokenizers;

/// <summary>
/// Tokenizer that encodes text to token IDs and decodes token IDs back to text.
/// </summary>
public interface ITokenizer
{
    /// <summary>Encodes text into token IDs.</summary>
    /// <param name="text">Input text to tokenize.</param>
    /// <returns>Array of token IDs.</returns>
    int[] Encode(string text);

    /// <summary>
    /// Encodes text WITHOUT any automatic BOS prepending. Identical to <see cref="Encode"/> for
    /// tokenizers that never add BOS on their own (the default). Used where the caller owns
    /// stream framing — corpus streaming for perplexity, which prepends BOS once itself, and
    /// sampler token-set construction.
    /// </summary>
    int[] EncodeRaw(string text) => Encode(text);

    /// <summary>
    /// Marks every special-token spelling in <paramref name="text"/> (untrusted, caller-supplied chat content)
    /// so a later <see cref="Encode"/> / <see cref="EncodeRaw"/> tokenizes it as plain text instead of as a control
    /// token (#1339; see <see cref="ControlTokenGuard"/>). The default returns the text unchanged, for tokenizers
    /// that do not split out special-token literals.
    /// </summary>
    /// <param name="text">Untrusted text.</param>
    /// <param name="exemptPrefixes">
    /// Spellings that stay recognised when the text at a match position starts with one of them (e.g. the reasoning
    /// tags an assistant turn legitimately carries). Null/empty = none.
    /// </param>
    /// <returns>The text with markers inserted; equal to <paramref name="text"/> when nothing matched.</returns>
    string MarkUntrustedSpecials(string text, IReadOnlyList<string>? exemptPrefixes = null) => text;

    /// <summary>Decodes a sequence of token IDs back to text.</summary>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <returns>Decoded text.</returns>
    string Decode(ReadOnlySpan<int> tokenIds);

    /// <summary>
    /// Decodes a sequence of token IDs back to text, optionally preserving the leading space
    /// that SentencePiece tokenizers normally strip (the inverse of BOS ▁ prepending).
    /// Use <paramref name="stripBosSpace"/> = <c>false</c> when decoding generated continuation
    /// tokens that were NOT encoded with BOS space prepending.
    /// </summary>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <param name="stripBosSpace">When true (default), strips the leading space introduced by BOS ▁ prepending.</param>
    /// <returns>Decoded text.</returns>
    string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);

    /// <summary>Decodes a single token ID to its string representation.</summary>
    /// <param name="tokenId">Token ID to decode.</param>
    /// <returns>String representation of the token.</returns>
    string DecodeToken(int tokenId);

    /// <summary>Total vocabulary size.</summary>
    int VocabSize { get; }

    /// <summary>Beginning-of-sequence token ID.</summary>
    int BosTokenId { get; }

    /// <summary>End-of-sequence token ID.</summary>
    int EosTokenId { get; }

    /// <summary>
    /// Additional end-of-generation token ids declared by the model file beyond <see cref="EosTokenId"/>
    /// (GGUF <c>tokenizer.ggml.eot_token_id</c> / <c>eom_token_id</c>: GLM-4.x declares <c>&lt;|endoftext|&gt;</c> as
    /// EOS but ends its turns with <c>&lt;|user|&gt;</c> (eot) or <c>&lt;|observation|&gt;</c> after a tool call (eom), #797).
    /// Empty when the model declares none.
    /// </summary>
    IReadOnlyList<int> ExtraEndOfGenerationTokenIds => [];

    /// <summary>
    /// Counts the number of tokens without performing a full encode.
    /// May be approximate for some tokenizer implementations.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <returns>Token count.</returns>
    int CountTokens(string text);
}
