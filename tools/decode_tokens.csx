// Quick token-id decode helper.
#r "C:/Development/dotLLM/src/DotLLM.Models/bin/Release/net10.0/DotLLM.Models.dll"
#r "C:/Development/dotLLM/src/DotLLM.Core/bin/Release/net10.0/DotLLM.Core.dll"
#r "C:/Development/dotLLM/src/DotLLM.Tokenizers/bin/Release/net10.0/DotLLM.Tokenizers.dll"

using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

string path = Args[0];
var gguf = GgufFile.Open(path);
try
{
    var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
    for (int i = 1; i < Args.Count; i++)
    {
        int id = int.Parse(Args[i]);
        string text = tok.DecodeToken(id);
        Console.WriteLine($"token {id} -> {System.Text.Json.JsonSerializer.Serialize(text)}");
    }

    string prompt = "The capital of France is";
    var ids = tok.Encode(prompt);
    Console.WriteLine($"prompt '{prompt}' -> [{string.Join(",", ids.ToArray())}]");
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
