// Routing trace for MoE offload studies (#1063): DOTLLM_MOE_ROUTE_TRACE=<file> appends one record per routed token:
// int16 layerSeq, int16 nExpert, int16 topK, then topK int16 expert ids. layerSeq numbers MoE layers by first appearance,
// which is layer order because layers run in sequence. tools/moe_cache_sim.py replays a trace through cache policies.
// Off (one static null check) unless the variable is set.
using System.Runtime.CompilerServices;

namespace DotLLM.Models.Architectures;

/// <summary>Optional per-token routing trace of the CPU Nemotron-H MoE router (see the file header).</summary>
internal static class RouteTrace
{
    private static readonly string? Path = Environment.GetEnvironmentVariable("DOTLLM_MOE_ROUTE_TRACE");
    private static readonly object Gate = new();
    private static readonly ConditionalWeakTable<object, StrongBox<int>> Ids = new();
    private static BinaryWriter? _w;
    private static int _next;

    /// <summary>Appends one token's selected expert ids when tracing is enabled.</summary>
    /// <param name="moe">The layer's MoE weights (identifies the layer).</param>
    /// <param name="ids">The selected expert ids, in selection order.</param>
    internal static void Record(NemotronHMoeWeights moe, ReadOnlySpan<int> ids)
    {
        if (Path is null) return;
        lock (Gate)
        {
            _w ??= new BinaryWriter(File.Open(Path, FileMode.Append, FileAccess.Write, FileShare.Read));
            if (!Ids.TryGetValue(moe, out var box)) { box = new StrongBox<int>(_next++); Ids.Add(moe, box); }
            _w.Write((short)box.Value); _w.Write((short)moe.NumExperts); _w.Write((short)ids.Length);
            foreach (int e in ids) _w.Write((short)e);
            _w.Flush();
        }
    }
}
