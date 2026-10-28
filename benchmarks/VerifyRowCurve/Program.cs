// Diagnostics tool (not shipped) for issue #1359 stage 1: multi-row verify wall time vs rows R.
//   VerifyRowCurve <gguf> <vulkan|cpu> [reps=3] [inner=5] [ctx=64] [rows=1,2,4,8,16,32] [ab]
// With `ab` (Vulkan dense qwen35 #1365, MoE host #1374; models without recurrent row snapshots time plain Forward instead) every timing is taken twice in the same process: "old" = the multi-column GEMVs capped at 8 rows
// (the pre-#1365 dispatch: DOTLLM_VK_SMALLROW_WIDE=0 + DOTLLM_VK_PQ2_0_MULTICOL_MAX_N=8) and "new" = up to 16; the order alternates with the rep.
// Loads ONCE, prefills ctx tokens, then times a forward over R rows at positions ctx..ctx+R-1 (token content is irrelevant to cost),
// rolls the KV and the recurrent state back, and repeats. Two series per R: "fwd" (plain Forward) and "verify" (ForwardWithRecurrentSnapshots,
// the call MtpSpeculativeDecoder makes, only where the model supports it and R fits its snapshot cap). Reps are interleaved: every rep visits
// every R (ascending on even reps, descending on odd) so a clock-ramp or UMA-contention drift hits all R alike. Warm-up passes are discarded.
using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

string path = args[0];
string dev = args[1];
int reps = args.Length > 2 ? int.Parse(args[2]) : 3;
int inner = args.Length > 3 ? int.Parse(args[3]) : 5;
int ctx = args.Length > 4 ? int.Parse(args[4]) : 64;
int[] rowsList = (args.Length > 5 ? args[5] : "1,2,4,8,16,32").Split(',').Select(int.Parse).ToArray();
int maxRows = rowsList.Max();
bool ab = args.Length > 6 && args[6] == "ab";
void SetWide(bool wide)
{
    DotLLM.Vulkan.VulkanQwen3HybridDenseTransformerModel.SmallRowWideEnabled = wide;
    DotLLM.Vulkan.VulkanQwen3MoeHybridTransformerModel.SmallRowWideEnabled = wide;   // #1374
    DotLLM.Vulkan.Kernels.PQ2_0SmallNDispatch.MultiColumnMaxNOverride = wide ? 16 : 8;
}
if (ab) SetWide(true);

var sw0 = Stopwatch.StartNew();
IModel model; Func<int, IKvCache> kvf; GgufFile gguf; ModelConfig cfg; VulkanDevice? vdev = null;
if (dev == "vulkan")
{
    gguf = GgufFile.Open(path);
    cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
    vdev = VulkanDevice.Create();
    (model, kvf) = VulkanModelLoader.CreateFromGguf(vdev, gguf, cfg, VulkanModelLoader.ResolveSpvDir());
}
else
{
    (model, gguf, cfg) = ModelLoader.LoadFromGguf(path, new ThreadingConfig(0));
    var geom = KvGeometry.FromConfig(cfg, model.SupportsCompactSlidingKv);
    kvf = n => new SimpleKvCache(geom, n);
}
Console.WriteLine($"[vrc] {Path.GetFileName(path)} {dev} model={model.GetType().Name} loaded in {sw0.Elapsed.TotalSeconds:F0} s ctx={ctx} reps={reps} inner={inner}");
bool canSnap = model.SupportsRecurrentRowSnapshots;
bool canCkpt = model.SupportsRecurrentStateCheckpoint;
Console.WriteLine($"[vrc] SupportsRecurrentRowSnapshots={canSnap} SupportsRecurrentStateCheckpoint={canCkpt}");

var rng = new Random(1234);
int[] Ids(int n) { var a = new int[n]; for (int i = 0; i < n; i++) a[i] = rng.Next(1000, 20000); return a; }

var kv = kvf(ctx + maxRows + 8);
model.ResetSequenceState();
if (ctx > 0)
{
    var ctxIds = Ids(ctx);
    for (int p = 0; p < ctx; p += 32)
    {
        int n = Math.Min(32, ctx - p);
        using var w = model.Forward(ctxIds.AsSpan(p, n), Enumerable.Range(p, n).ToArray(), -1, kv);
    }
}
object? ckpt = canCkpt ? model.CheckpointRecurrentState() : null;
Console.WriteLine($"[vrc] checkpoint taken: {ckpt is not null}");

double Time(int rows, bool verify)
{
    var ids = Ids(rows);
    var pos = Enumerable.Range(ctx, rows).ToArray();
    long t0 = Stopwatch.GetTimestamp();
    using var lg = verify
        ? model.ForwardWithRecurrentSnapshots(ids, pos, -1, kv, null)
        : model.Forward(ids, pos, -1, kv);
    double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
    kv.Rollback(ctx);
    if (ckpt is not null) model.RestoreRecurrentState(ckpt);
    return ms;
}

bool VerifyOk(int r) => canSnap && r >= 2 && r <= 16;
bool AbOk(int r) => r >= 2 && r <= 16;   // ab arm: verify series when the model snapshots, plain Forward otherwise
if (ab)
{
    // Old/new interleaved per timing, alternating which goes first; verify series only (ForwardWithRecurrentSnapshots, <= 16 rows).
    foreach (int r in rowsList.Where(AbOk)) { SetWide(false); Time(r, canSnap); SetWide(true); Time(r, canSnap); }
    var oldL = rowsList.ToDictionary(r => r, _ => new List<double>());
    var newL = rowsList.ToDictionary(r => r, _ => new List<double>());
    for (int rep = 0; rep < reps; rep++)
    {
        foreach (int r in (rep % 2 == 0 ? rowsList : rowsList.Reverse().ToArray()).Where(AbOk))
            for (int i = 0; i < inner; i++)
            {
                bool newFirst = ((rep + i) & 1) == 1;
                for (int pass = 0; pass < 2; pass++)
                {
                    bool wide = (pass == 0) == newFirst;
                    SetWide(wide);
                    (wide ? newL : oldL)[r].Add(Time(r, canSnap));
                }
            }
        Console.WriteLine($"[vrc] ab rep {rep + 1}/{reps} done {sw0.Elapsed.TotalSeconds:F0} s");
    }
    Console.WriteLine("ab	rows	arm	min	med	max");
    foreach (int r in rowsList.Where(AbOk))
    {
        Console.WriteLine($"ab	{r}	old	{oldL[r].Min():F1}	{Med(oldL[r]):F1}	{oldL[r].Max():F1}");
        Console.WriteLine($"ab	{r}	new	{newL[r].Min():F1}	{Med(newL[r]):F1}	{newL[r].Max():F1}");
    }
    (kv as IDisposable)?.Dispose(); (model as IDisposable)?.Dispose(); vdev?.Dispose();
    return;
}
for (int w = 0; w < 2; w++)
    foreach (int r in rowsList) { Time(r, false); if (VerifyOk(r)) Time(r, true); }

var fwd = rowsList.ToDictionary(r => r, _ => new List<double>());
var ver = rowsList.ToDictionary(r => r, _ => new List<double>());
var fwdRepMed = rowsList.ToDictionary(r => r, _ => new List<double>());
var verRepMed = rowsList.ToDictionary(r => r, _ => new List<double>());
static double Med(List<double> l) { var s = l.OrderBy(x => x).ToList(); return s[s.Count / 2]; }
for (int rep = 0; rep < reps; rep++)
{
    var order = rep % 2 == 0 ? rowsList : rowsList.Reverse().ToArray();
    foreach (int r in order)
    {
        var a = new List<double>(); var b = new List<double>();
        for (int i = 0; i < inner; i++)
        {
            a.Add(Time(r, false));
            if (VerifyOk(r)) b.Add(Time(r, true));
        }
        fwd[r].AddRange(a); fwdRepMed[r].Add(Med(a));
        if (b.Count > 0) { ver[r].AddRange(b); verRepMed[r].Add(Med(b)); }
    }
    Console.WriteLine($"[vrc] rep {rep + 1}/{reps} done {sw0.Elapsed.TotalSeconds:F0} s");
}
Console.WriteLine("series\trows\tmin\tmed\tmax\trepMedians");
foreach (int r in rowsList)
{
    Console.WriteLine($"fwd\t{r}\t{fwd[r].Min():F1}\t{Med(fwd[r]):F1}\t{fwd[r].Max():F1}\t{string.Join(" ", fwdRepMed[r].Select(x => x.ToString("F1")))}");
    if (ver[r].Count > 0)
        Console.WriteLine($"verify\t{r}\t{ver[r].Min():F1}\t{Med(ver[r]):F1}\t{ver[r].Max():F1}\t{string.Join(" ", verRepMed[r].Select(x => x.ToString("F1")))}");
}
(kv as IDisposable)?.Dispose();
(model as IDisposable)?.Dispose();
vdev?.Dispose();
