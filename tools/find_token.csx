// Find token id for a string.
#r "C:/Development/dotLLM/src/DotLLM.Models/bin/Release/net10.0/DotLLM.Models.dll"
#r "C:/Development/dotLLM/src/DotLLM.Core/bin/Release/net10.0/DotLLM.Core.dll"
#r "C:/Development/dotLLM/src/DotLLM.Tokenizers/bin/Release/net10.0/DotLLM.Tokenizers.dll"

using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

string path = Args[0];
string target = Args[1];
var gguf = GgufFile.Open(path);
try
{
    var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
    var ids = tok.Encode(target);
    Console.WriteLine($"Encode('{target}') = [{string.Join(",", ids.ToArray())}]");
    foreach (int id in ids)
    {
        string text = tok.DecodeToken(id);
        Console.WriteLine($"  {id} -> {System.Text.Json.JsonSerializer.Serialize(text)}");
    }
}
finally
{
    gguf.Dispose();
}
