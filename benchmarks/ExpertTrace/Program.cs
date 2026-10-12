// ExpertTrace (#938, parent #825): records per-layer routed-expert selections of the qwen4exp CPU oracle for one prompt (greedy decode).
// Diagnostics tool, not shipped.
//   ExpertTrace <gguf> <promptFile|-> <maxNew> <out.bin> [chunk=256] [needleTokens=0]
//     promptFile : UTF-8 text (raw, includes any chat markers). needleTokens > 0 instead builds a long-context needle prompt of about that many tokens.
// Output (little endian): "RTR1", int32 layers, int32 topK, int32 nTokens, int32 nPrefill, then per token layers*topK uint16 (token-major),
// tokens [0,nPrefill) are prompt rows, the rest greedy-decoded rows (the row that was FED to the model at that step).
using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;

string gguf = args[0], promptPath = args[1], outPath = args[3];
int maxNew = int.Parse(args[2]);
int chunk = args.Length > 4 ? int.Parse(args[4]) : 256;
int needleTokens = args.Length > 5 ? int.Parse(args[5]) : 0;

var sw = Stopwatch.StartNew();
var (model, file, cfg) = ModelLoader.LoadFromGguf(gguf, new DotLLM.Core.Configuration.ThreadingConfig(0));
var q4 = (Qwen4ExpTransformerModel)model;
var tok = GgufTokenizerFactory.Load(file.Metadata);
Console.WriteLine($"[trace] loaded in {sw.Elapsed.TotalSeconds:F0} s; vocab {cfg.VocabSize}");

int[] prompt;
if (needleTokens > 0)
{
    var filler = string.Join("\n\n", Directory.GetFiles(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "docs"), "*.md")
        .OrderBy(x => x).Select(File.ReadAllText));
    var fids = tok.Encode(filler);
    int n = Math.Min(needleTokens, fids.Length);
    var head = tok.Encode("<|im_start|>user\nBelow is a long document. Read it, then answer the question at the end.\n\n");
    var needle = tok.Encode("\n\nNote: the secret passphrase for the vault is amber-falcon-7731.\n\n");
    var tail = tok.Encode("\n\nQuestion: what is the secret passphrase for the vault?<|im_end|>\n<|im_start|>assistant\n");
    var list = new List<int>(head); list.AddRange(fids.Take(n * 2 / 5)); list.AddRange(needle); list.AddRange(fids.Skip(n * 2 / 5).Take(n - n * 2 / 5)); list.AddRange(tail);
    prompt = list.ToArray();
}
else prompt = tok.Encode(File.ReadAllText(promptPath));
Console.WriteLine($"[trace] prompt {prompt.Length} tokens, maxNew {maxNew}");

int L = cfg.NumLayers + 1; // generous; unused layers are written as 0xFFFF
int K = 0;
var routes = new List<ushort[]>();
var cur = new List<ushort[]>();
int baseTok = 0;
ushort[]? rowBuf = null;
Qwen4ExpTransformerModel.RoutingTrace = (layer, rows, k, ex) =>
{
    if (K == 0) K = k;
    for (int t = 0; t < rows; t++)
    {
        int idx = baseTok + t;
        while (routes.Count <= idx) { var a = new ushort[L * 16]; Array.Fill(a, (ushort)0xFFFF); routes.Add(a); }
        var r = routes[idx];
        for (int j = 0; j < k; j++) r[layer * 16 + j] = (ushort)ex[t * k + j];
    }
};

unsafe int ArgMaxLast(DotLLM.Core.Tensors.ITensor lg, int rows)
{
    var p = (float*)(lg.DataPointer + (nint)((long)(rows - 1) * cfg.VocabSize * 4));
    int best = 0; float bv = p[0];
    for (int i = 1; i < cfg.VocabSize; i++) if (p[i] > bv) { bv = p[i]; best = i; }
    return best;
}

int pos = 0; int next = -1;
for (int s = 0; s < prompt.Length; s += chunk)
{
    var ids = prompt.AsSpan(s, Math.Min(chunk, prompt.Length - s));
    var ps = Enumerable.Range(pos, ids.Length).ToArray();
    baseTok = pos;
    var t0 = Stopwatch.StartNew();
    using var lg = q4.Forward(ids, ps, -1, (DotLLM.Core.Attention.IKvCache?)null, true);
    pos += ids.Length;
    next = ArgMaxLast(lg, 1);
    Console.WriteLine($"[trace] prefill {pos}/{prompt.Length} ({ids.Length / t0.Elapsed.TotalSeconds:F1} tok/s)");
}
int nPrefill = prompt.Length;
var eos = new HashSet<int>(tok.ExtraEndOfGenerationTokenIds) { tok.EosTokenId };
var text = new System.Text.StringBuilder();
int generated = 0;
for (; generated < maxNew; generated++)
{
    if (eos.Contains(next)) break;
    text.Append(tok.DecodeToken(next));
    baseTok = pos;
    using var lg = q4.Forward(new[] { next }, new[] { pos }, -1, (DotLLM.Core.Attention.IKvCache?)null, true);
    pos++;
    next = ArgMaxLast(lg, 1);
    if (generated % 20 == 0) Console.WriteLine($"[trace] decode {generated}/{maxNew}");
}
Console.WriteLine($"[trace] generated {generated} tokens: {text.ToString().Replace("\n", "\n")[..Math.Min(text.Length, 300)]}");

int usedLayers = 0;
foreach (var r in routes) for (int l = 0; l < L; l++) if (r[l * 16] != 0xFFFF) usedLayers = Math.Max(usedLayers, l + 1);
using var fs = File.Create(outPath);
using var bw = new BinaryWriter(fs);
bw.Write(new[] { (byte)'R', (byte)'T', (byte)'R', (byte)'1' });
bw.Write(usedLayers); bw.Write(K); bw.Write(routes.Count); bw.Write(nPrefill);
foreach (var r in routes)
    for (int l = 0; l < usedLayers; l++)
        for (int j = 0; j < K; j++) bw.Write(r[l * 16 + j]);
Console.WriteLine($"[trace] wrote {outPath}: {routes.Count} tokens x {usedLayers} layers x top-{K}");
