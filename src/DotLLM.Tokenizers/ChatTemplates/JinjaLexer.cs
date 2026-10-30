namespace DotLLM.Tokenizers.ChatTemplates;

/// <summary>
/// Tokenizes a Jinja2 template string into a flat token list.
/// Two-mode lexer: text mode (outside tags) and tag mode (inside {{ }}, {% %}).
/// Handles whitespace control markers ({%- -%} {{- -}}) and, like HF transformers and llama.cpp's minja
/// (both build their environment with <c>trim_blocks</c> + <c>lstrip_blocks</c>), removes the newline after a block
/// tag and the indentation before one. Templates are authored against that behaviour (#928).
/// </summary>
internal sealed class JinjaLexer
{
    private readonly string _source;
    private int _pos;
    private int _line = 1;
    private int _col = 1;
    private int _braceDepth;

    private readonly bool _trimBlocks;
    private readonly bool _lstripBlocks;

    /// <param name="source">Template source.</param>
    /// <param name="trimBlocks">Drop the first newline after a block tag (<c>%}</c>) or comment (<c>#}</c>) - Jinja2's <c>trim_blocks</c>.</param>
    /// <param name="lstripBlocks">Drop spaces/tabs between the start of a line and a block tag or comment - Jinja2's <c>lstrip_blocks</c>.</param>
    public JinjaLexer(string source, bool trimBlocks = true, bool lstripBlocks = true)
    {
        _source = source;
        _trimBlocks = trimBlocks;
        _lstripBlocks = lstripBlocks;
    }

    public List<JinjaToken> Tokenize()
    {
        var tokens = new List<JinjaToken>();
        _pos = 0;
        _line = 1;
        _col = 1;
        _braceDepth = 0;

        while (_pos < _source.Length)
        {
            // Check for comment {# ... #}
            if (Match("{#"))
            {
                int commentStart = _pos - 2;
                // Whitespace control on comments: {#- trims the preceding text, -#} the following.
                if (_pos < _source.Length && _source[_pos] == '-')
                    StripTrailingWhitespace(tokens);
                else if (_lstripBlocks)
                    StripLineIndentation(tokens, commentStart);
                bool stripRight = SkipComment();
                if (stripRight)
                    StripLeadingWhitespace(tokens);
                else if (_trimBlocks)
                    SkipOneNewline();
                continue;
            }

            // Check for expression start {{ or {{-
            if (Match("{{"))
            {
                bool stripLeft = false;
                if (_pos < _source.Length && _source[_pos] == '-')
                {
                    stripLeft = true;
                    Advance();
                }

                if (stripLeft)
                    StripTrailingWhitespace(tokens);

                tokens.Add(new JinjaToken(JinjaTokenType.ExprStart, null, _line, _col));
                TokenizeTag(tokens, "}}");
                continue;
            }

            // Check for statement start {% or {%-
            if (Match("{%"))
            {
                int tagStart = _pos - 2;
                bool stripLeft = false;
                bool keepIndent = false;
                if (_pos < _source.Length && _source[_pos] == '-')
                {
                    stripLeft = true;
                    Advance();
                }
                else if (_pos < _source.Length && _source[_pos] == '+')
                {
                    keepIndent = true;   // "{%+" disables lstrip_blocks for this tag
                    Advance();
                }

                if (stripLeft)
                    StripTrailingWhitespace(tokens);
                else if (_lstripBlocks && !keepIndent)
                    StripLineIndentation(tokens, tagStart);

                tokens.Add(new JinjaToken(JinjaTokenType.StmtStart, null, _line, _col));
                TokenizeTag(tokens, "%}");
                continue;
            }

            // Text mode: scan until next tag
            int textStart = _pos;
            int textLine = _line;
            int textCol = _col;
            while (_pos < _source.Length && !LookAhead("{{") && !LookAhead("{%") && !LookAhead("{#"))
            {
                Advance();
            }

            if (_pos > textStart)
            {
                string text = _source[textStart.._pos];
                tokens.Add(new JinjaToken(JinjaTokenType.Text, text, textLine, textCol));
            }
        }

        tokens.Add(new JinjaToken(JinjaTokenType.Eof, null, _line, _col));
        return tokens;
    }

    private void TokenizeTag(List<JinjaToken> tokens, string endMarker)
    {
        SkipWhitespaceInTag();

        while (_pos < _source.Length)
        {
            SkipWhitespaceInTag();

            if (_pos >= _source.Length)
                throw new JinjaException("Unexpected end of template inside tag", _line, _col);

            // Only check for end markers when not inside a dict literal
            if (_braceDepth == 0)
            {
                // Check for strip-right + end marker: -}} or -%}
                if (_source[_pos] == '-' && _pos + endMarker.Length < _source.Length &&
                    _source.AsSpan(_pos + 1, endMarker.Length).SequenceEqual(endMarker))
                {
                    _pos += 1 + endMarker.Length;
                    _col += 1 + endMarker.Length;
                    var endType = endMarker == "}}" ? JinjaTokenType.ExprEnd : JinjaTokenType.StmtEnd;
                    tokens.Add(new JinjaToken(endType, null, _line, _col));
                    StripLeadingWhitespace(tokens);
                    return;
                }

                // Check for end marker without strip
                if (LookAhead(endMarker))
                {
                    _pos += endMarker.Length;
                    _col += endMarker.Length;
                    var endType = endMarker == "}}" ? JinjaTokenType.ExprEnd : JinjaTokenType.StmtEnd;
                    tokens.Add(new JinjaToken(endType, null, _line, _col));
                    if (_trimBlocks && endType == JinjaTokenType.StmtEnd)
                        SkipOneNewline();
                    return;
                }
            }

            char c = _source[_pos];

            // String literal
            if (c is '"' or '\'')
            {
                tokens.Add(ReadString());
                continue;
            }

            // Number literal
            if (char.IsDigit(c))
            {
                tokens.Add(ReadNumber());
                continue;
            }

            // Identifier or keyword
            if (char.IsLetter(c) || c == '_')
            {
                tokens.Add(ReadIdentifierOrKeyword());
                continue;
            }

            // Two-character operators
            if (_pos + 1 < _source.Length)
            {
                string twoChar = _source.Substring(_pos, 2);
                var twoCharType = twoChar switch
                {
                    "==" => JinjaTokenType.Eq,
                    "!=" => JinjaTokenType.Ne,
                    "<=" => JinjaTokenType.Le,
                    ">=" => JinjaTokenType.Ge,
                    _ => (JinjaTokenType?)null
                };

                if (twoCharType.HasValue)
                {
                    tokens.Add(new JinjaToken(twoCharType.Value, null, _line, _col));
                    Advance();
                    Advance();
                    continue;
                }
            }

            // Single-character operators/punctuation
            // Handle braces with depth tracking for dict literals
            if (c == '{')
            {
                _braceDepth++;
                tokens.Add(new JinjaToken(JinjaTokenType.LeftBrace, null, _line, _col));
                Advance();
                continue;
            }

            if (c == '}' && _braceDepth > 0)
            {
                _braceDepth--;
                tokens.Add(new JinjaToken(JinjaTokenType.RightBrace, null, _line, _col));
                Advance();
                continue;
            }

            var singleType = c switch
            {
                '.' => JinjaTokenType.Dot,
                '[' => JinjaTokenType.LeftBracket,
                ']' => JinjaTokenType.RightBracket,
                '(' => JinjaTokenType.LeftParen,
                ')' => JinjaTokenType.RightParen,
                '|' => JinjaTokenType.Pipe,
                ',' => JinjaTokenType.Comma,
                ':' => JinjaTokenType.Colon,
                '=' => JinjaTokenType.Assign,
                '<' => JinjaTokenType.Lt,
                '>' => JinjaTokenType.Gt,
                '+' => JinjaTokenType.Plus,
                '-' => JinjaTokenType.Minus,
                '*' => JinjaTokenType.Multiply,
                '/' => JinjaTokenType.Divide,
                '%' => JinjaTokenType.Modulo,
                '~' => JinjaTokenType.Tilde,
                _ => (JinjaTokenType?)null
            };

            if (singleType.HasValue)
            {
                tokens.Add(new JinjaToken(singleType.Value, null, _line, _col));
                Advance();
                continue;
            }

            throw new JinjaException($"Unexpected character '{c}' in tag", _line, _col);
        }

        throw new JinjaException("Unclosed tag", _line, _col);
    }

    private JinjaToken ReadString()
    {
        char quote = _source[_pos];
        int startLine = _line;
        int startCol = _col;
        Advance(); // skip opening quote

        var sb = new System.Text.StringBuilder();
        while (_pos < _source.Length && _source[_pos] != quote)
        {
            if (_source[_pos] == '\\' && _pos + 1 < _source.Length)
            {
                Advance(); // skip backslash
                char escaped = _source[_pos];

                // Python string-literal hex escapes: \xNN, \uNNNN, \UNNNNNNNN (Flint's template strips "\x00").
                int hexDigits = escaped switch { 'x' => 2, 'u' => 4, 'U' => 8, _ => 0 };
                if (hexDigits > 0 && TryReadHex(_pos + 1, hexDigits, out int codePoint))
                {
                    sb.Append(char.ConvertFromUtf32(codePoint));
                    for (int i = 0; i <= hexDigits; i++)
                        Advance();
                    continue;
                }

                sb.Append(escaped switch
                {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    _ => escaped
                });
                Advance();
            }
            else
            {
                sb.Append(_source[_pos]);
                Advance();
            }
        }

        if (_pos >= _source.Length)
            throw new JinjaException("Unterminated string literal", startLine, startCol);

        Advance(); // skip closing quote
        return new JinjaToken(JinjaTokenType.StringLiteral, sb.ToString(), startLine, startCol);
    }

    /// <summary>Parses exactly <paramref name="count"/> hex digits starting at <paramref name="start"/> as a Unicode scalar value.</summary>
    /// <param name="start">Index of the first hex digit.</param>
    /// <param name="count">Number of hex digits to read.</param>
    /// <param name="codePoint">The parsed code point when successful.</param>
    /// <returns><see langword="true"/> when all digits are hex and the value is a valid scalar (not a surrogate).</returns>
    private bool TryReadHex(int start, int count, out int codePoint)
    {
        codePoint = 0;
        if (start + count > _source.Length)
            return false;
        for (int i = 0; i < count; i++)
        {
            int d = HexValue(_source[start + i]);
            if (d < 0)
                return false;
            codePoint = (codePoint << 4) | d;
        }
        return codePoint is >= 0 and <= 0x10FFFF and not (>= 0xD800 and <= 0xDFFF);
    }

    private static int HexValue(char c) => c switch
    {
        >= '0' and <= '9' => c - '0',
        >= 'a' and <= 'f' => c - 'a' + 10,
        >= 'A' and <= 'F' => c - 'A' + 10,
        _ => -1,
    };

    private JinjaToken ReadNumber()
    {
        int startLine = _line;
        int startCol = _col;
        int start = _pos;

        while (_pos < _source.Length && char.IsDigit(_source[_pos]))
            Advance();

        string numStr = _source[start.._pos];
        return new JinjaToken(JinjaTokenType.IntLiteral, int.Parse(numStr), startLine, startCol);
    }

    private JinjaToken ReadIdentifierOrKeyword()
    {
        int startLine = _line;
        int startCol = _col;
        int start = _pos;

        while (_pos < _source.Length && (char.IsLetterOrDigit(_source[_pos]) || _source[_pos] == '_'))
            Advance();

        string word = _source[start.._pos];

        var type = word switch
        {
            "if" => JinjaTokenType.If,
            "elif" => JinjaTokenType.Elif,
            "else" => JinjaTokenType.Else,
            "endif" => JinjaTokenType.Endif,
            "for" => JinjaTokenType.For,
            "in" => JinjaTokenType.In,
            "endfor" => JinjaTokenType.Endfor,
            "set" => JinjaTokenType.Set,
            "macro" => JinjaTokenType.Macro,
            "endmacro" => JinjaTokenType.EndMacro,
            "endset" => JinjaTokenType.EndSet,
            "not" => JinjaTokenType.Not,
            "and" => JinjaTokenType.And,
            "or" => JinjaTokenType.Or,
            "is" => JinjaTokenType.Is,
            "defined" => JinjaTokenType.Defined,
            "namespace" => JinjaTokenType.Namespace,
            "true" or "True" => JinjaTokenType.BoolLiteral,
            "false" or "False" => JinjaTokenType.BoolLiteral,
            "none" or "None" => JinjaTokenType.NoneLiteral,
            _ => JinjaTokenType.Identifier
        };

        object? value = type switch
        {
            JinjaTokenType.BoolLiteral => word is "true" or "True",
            JinjaTokenType.Identifier => word,
            _ => null
        };

        return new JinjaToken(type, value, startLine, startCol);
    }

    /// <returns>True when the comment closed with <c>-#}</c> (strip following whitespace).</returns>
    private bool SkipComment()
    {
        // Already consumed {#, now scan for #}
        while (_pos < _source.Length)
        {
            if (_source[_pos] == '#' && _pos + 1 < _source.Length && _source[_pos + 1] == '}')
            {
                bool strip = _pos > 0 && _source[_pos - 1] == '-';
                Advance();
                Advance();
                return strip;
            }
            Advance();
        }
        throw new JinjaException("Unclosed comment", _line, _col);
    }

    private void SkipWhitespaceInTag()
    {
        while (_pos < _source.Length && _source[_pos] is ' ' or '\t' or '\r' or '\n')
            Advance();
    }

    private bool Match(string s)
    {
        if (_pos + s.Length > _source.Length)
            return false;
        if (!_source.AsSpan(_pos, s.Length).SequenceEqual(s))
            return false;
        _pos += s.Length;
        _col += s.Length;
        return true;
    }

    private bool LookAhead(string s)
    {
        if (_pos + s.Length > _source.Length)
            return false;
        return _source.AsSpan(_pos, s.Length).SequenceEqual(s);
    }

    private void Advance()
    {
        if (_pos < _source.Length)
        {
            if (_source[_pos] == '\n')
            {
                _line++;
                _col = 1;
            }
            else
            {
                _col++;
            }
            _pos++;
        }
    }

    /// <summary>Skips a single LF or CRLF at the current position (trim_blocks).</summary>
    private void SkipOneNewline()
    {
        if (_pos + 1 < _source.Length && _source[_pos] == '\r' && _source[_pos + 1] == '\n')
            Advance();
        if (_pos < _source.Length && _source[_pos] == '\n')
            Advance();
    }

    /// <summary>
    /// lstrip_blocks: removes the spaces/tabs between the start of the SOURCE line and a block tag, but only when nothing
    /// else precedes the tag on that line. Looks at the source rather than the token (trim_blocks may already have eaten the
    /// newline that ended the previous line, leaving the indentation as the whole preceding Text token).
    /// </summary>
    private void StripLineIndentation(List<JinjaToken> tokens, int tagStart)
    {
        int i = tagStart - 1;
        while (i >= 0 && _source[i] is ' ' or '\t')
            i--;
        int indent = tagStart - 1 - i;
        if (indent == 0 || (i >= 0 && _source[i] != '\n'))
            return;
        if (tokens.Count == 0)
            return;
        int last = tokens.Count - 1;
        if (tokens[last].Type != JinjaTokenType.Text)
            return;
        string text = (string)tokens[last].Value!;
        if (text.Length < indent)
            return;
        string head = text[..^indent];
        if (head.Length == 0)
            tokens.RemoveAt(last);
        else
            tokens[last] = tokens[last] with { Value = head };
    }

    /// <summary>
    /// Strips trailing whitespace (including newlines) from the last Text token.
    /// Used for {{- and {%- whitespace control.
    /// </summary>
    private static void StripTrailingWhitespace(List<JinjaToken> tokens)
    {
        if (tokens.Count == 0)
            return;

        int last = tokens.Count - 1;
        if (tokens[last].Type != JinjaTokenType.Text)
            return;

        string text = (string)tokens[last].Value!;
        string trimmed = text.TrimEnd(' ', '\t', '\r', '\n');
        if (trimmed.Length == 0)
            tokens.RemoveAt(last);
        else
            tokens[last] = tokens[last] with { Value = trimmed };
    }

    /// <summary>
    /// Sets a flag so the next Text token emitted has its leading whitespace stripped.
    /// Used for -}} and -%} whitespace control.
    /// Actually modifies the token list inline — the flag is implemented by modifying the next text token.
    /// We achieve this by adding a sentinel that the tokenizer main loop handles.
    /// Instead, we just mark position and strip when the next text is emitted.
    /// </summary>
    private void StripLeadingWhitespace(List<JinjaToken> tokens)
    {
        // Strip leading whitespace from the source at the current position
        // (the text that would become the next Text token)
        while (_pos < _source.Length && _source[_pos] is ' ' or '\t' or '\r' or '\n')
            Advance();
    }
}
