// Reads a PerfView .etl.zip (kernel CPU sample trace) and emits an aggregated
// self-time + inclusive-time breakdown per method, restricted to a target
// process name (default: dotnet).
//
// Usage:
//   EtlStackDump <etl.zip path> [process-name] [top-N]

using Microsoft.Diagnostics.Tracing.Etlx;
using Microsoft.Diagnostics.Tracing.Parsers.Kernel;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: EtlStackDump <etl.zip> [process-name] [top-N]");
    return 1;
}

string etlPath = args[0];
string procName = args.Length > 1 ? args[1] : "dotnet";
int topN = args.Length > 2 ? int.Parse(args[2]) : 40;

Console.Error.WriteLine($"Opening {etlPath}...");

using var traceLog = TraceLog.OpenOrConvert(etlPath);
Console.Error.WriteLine($"  Session: {traceLog.SessionStartTime:HH:mm:ss.fff} → {traceLog.SessionEndTime:HH:mm:ss.fff}");
Console.Error.WriteLine($"  Processes: {traceLog.Processes.Count}, Events: {traceLog.EventCount}");

var selfTime = new Dictionary<string, long>(StringComparer.Ordinal);
var inclusiveTime = new Dictionary<string, long>(StringComparer.Ordinal);
long totalSamples = 0;
int missingStacks = 0;

foreach (var ev in traceLog.Events)
{
    if (ev is not SampledProfileTraceData sample) continue;
    var proc = sample.Process();
    if (proc == null || !string.Equals(proc.Name, procName, StringComparison.OrdinalIgnoreCase))
        continue;

    totalSamples++;

    var stack = sample.CallStack();
    if (stack == null)
    {
        missingStacks++;
        continue;
    }

    string? leaf = null;
    var seen = new HashSet<string>(StringComparer.Ordinal);
    var current = stack;
    while (current != null)
    {
        string name = FrameName(current);
        if (leaf == null) leaf = name;
        if (seen.Add(name))
            inclusiveTime[name] = inclusiveTime.GetValueOrDefault(name) + 1;
        current = current.Caller;
    }
    if (leaf != null)
        selfTime[leaf] = selfTime.GetValueOrDefault(leaf) + 1;
}

Console.Error.WriteLine($"  CPU samples for {procName}: {totalSamples} (no-stack: {missingStacks})");

PrintTop("self", selfTime, topN);
PrintTop("self (DotLLM)", selfTime, topN, "DotLLM");
PrintTop("inclusive (DotLLM)", inclusiveTime, topN, "DotLLM");
PrintTop("self (VecDot/Quantize/Attention/Matmul)", selfTime, topN,
    "VecDot", "Quantize", "Attention", "MatMul");

// Second pass: for each unresolved coreclr! leaf with > 100 samples,
// aggregate its nearest managed caller so we can tell what it is
// (spin wait, JIT, wake, etc.).
Console.WriteLine();
Console.WriteLine("=== nearest managed caller of unresolved coreclr leaves (> 100 self samples) ===");
var hotUnresolved = selfTime
    .Where(kv => kv.Key.StartsWith("coreclr!0x", StringComparison.OrdinalIgnoreCase) && kv.Value > 100)
    .Select(kv => kv.Key)
    .ToHashSet();

var callerAgg = new Dictionary<string, Dictionary<string, long>>();
foreach (var ev in traceLog.Events)
{
    if (ev is not SampledProfileTraceData sample) continue;
    var proc = sample.Process();
    if (proc == null || !string.Equals(proc.Name, procName, StringComparison.OrdinalIgnoreCase))
        continue;
    var s = sample.CallStack();
    if (s == null) continue;
    string leaf = FrameName(s);
    if (!hotUnresolved.Contains(leaf)) continue;
    // Walk up until a managed frame
    var cur = s.Caller;
    while (cur != null)
    {
        string n = FrameName(cur);
        if (!n.StartsWith("coreclr!", StringComparison.OrdinalIgnoreCase)
            && !n.StartsWith("ntoskrnl!", StringComparison.OrdinalIgnoreCase)
            && !n.StartsWith("ntdll!", StringComparison.OrdinalIgnoreCase)
            && !n.StartsWith("kernel", StringComparison.OrdinalIgnoreCase)
            && !n.StartsWith("?!", StringComparison.Ordinal))
        {
            var d = callerAgg.GetValueOrDefault(leaf) ?? new Dictionary<string, long>();
            d[n] = d.GetValueOrDefault(n) + 1;
            callerAgg[leaf] = d;
            break;
        }
        cur = cur.Caller;
    }
}
foreach (var (addr, d) in callerAgg.OrderByDescending(kv => kv.Value.Values.Sum()))
{
    long total = d.Values.Sum();
    Console.WriteLine($"\n  {addr}  (attributed: {total})");
    foreach (var (name, w) in d.OrderByDescending(kv => kv.Value).Take(6))
        Console.WriteLine($"    {100.0 * w / total,6:F1}%  {w,8}  {name}");
}

return 0;

static string FrameName(TraceCallStack stack)
{
    var ca = stack.CodeAddress;
    var method = ca.Method;
    if (method != null)
    {
        var module = method.MethodModuleFile?.Name ?? "";
        return $"{module}!{method.FullMethodName}";
    }
    var mod = ca.ModuleFile?.Name ?? "?";
    return $"{mod}!0x{ca.Address:X}";
}

static void PrintTop(string label, Dictionary<string, long> d, int n, params string[] filters)
{
    Console.WriteLine();
    string filterLabel = filters.Length > 0 ? $" (any of: {string.Join(", ", filters)})" : "";
    Console.WriteLine($"=== top {n} by {label}{filterLabel} ===");
    long total = d.Values.Sum();
    var items = d
        .Where(kv => filters.Length == 0 ||
                     filters.Any(f => kv.Key.Contains(f, StringComparison.OrdinalIgnoreCase)))
        .OrderByDescending(kv => kv.Value)
        .Take(n);
    foreach (var (name, w) in items)
    {
        double pct = 100.0 * w / Math.Max(total, 1);
        Console.WriteLine($"  {pct,6:F2}%  {w,10}  {name}");
    }
}
