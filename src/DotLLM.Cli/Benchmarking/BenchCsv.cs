using System.Globalization;

namespace DotLLM.Cli.Benchmarking;

/// <summary>
/// Formats a ready-to-paste row for <c>benchmarks/perf-matrix/results.csv</c>.
/// Column order MUST match that file's header:
/// <c>date,host,device,backend,runtime,runtime_version,model,quant,pp_tok_s,tg_tok_s,tg_ctx_depth,settings,notes</c>.
/// <c>settings</c> and <c>notes</c> are always double-quoted (matching the
/// hand-entered convention in the file); embedded quotes are doubled per RFC 4180.
/// </summary>
public static class BenchCsv
{
    /// <summary>The results.csv header this formatter targets.</summary>
    public const string Header =
        "date,host,device,backend,runtime,runtime_version,model,quant,pp_tok_s,tg_tok_s,tg_ctx_depth,settings,notes";

    /// <summary>
    /// Builds one results.csv row.
    /// </summary>
    /// <param name="date">Measurement date (formatted yyyy-MM-dd).</param>
    /// <param name="host">Host label (e.g. <c>strix-halo</c>).</param>
    /// <param name="device">Device label (e.g. <c>Radeon-8060S-gfx1151</c>, <c>Zen5-16c</c>).</param>
    /// <param name="backend">Backend: <c>cpu</c>, <c>vulkan</c>, or <c>cuda</c>.</param>
    /// <param name="runtimeVersion">Point-in-time version (e.g. <c>dev-96a892bd</c>).</param>
    /// <param name="model">Model name (file name without quant suffix / extension).</param>
    /// <param name="quant">Quantization label (e.g. <c>Q4_K_M</c>).</param>
    /// <param name="ppTokS">Median prefill tokens/second.</param>
    /// <param name="tgTokS">Median decode tokens/second.</param>
    /// <param name="tgCtxDepth">Context depth the decode ran at.</param>
    /// <param name="settings">Free-form settings summary (quoted).</param>
    /// <param name="notes">Free-form notes (quoted).</param>
    /// <param name="runtime">Runtime name; defaults to <c>dotLLM</c>.</param>
    public static string FormatRow(
        DateOnly date, string host, string device, string backend, string runtimeVersion,
        string model, string quant, double ppTokS, double tgTokS, int tgCtxDepth,
        string settings, string notes, string runtime = "dotLLM")
    {
        return string.Join(",",
            date.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            Bare(host), Bare(device), Bare(backend), Bare(runtime), Bare(runtimeVersion),
            Bare(model), Bare(quant),
            BenchStats.FormatTokS(ppTokS), BenchStats.FormatTokS(tgTokS),
            tgCtxDepth.ToString(CultureInfo.InvariantCulture),
            Quote(settings), Quote(notes));
    }

    /// <summary>
    /// Sanitizes an unquoted field: commas / quotes / newlines would corrupt the row,
    /// so they are replaced with <c>;</c> / <c>'</c> / space respectively.
    /// </summary>
    private static string Bare(string value) =>
        value.Replace(',', ';').Replace('"', '\'').Replace('\r', ' ').Replace('\n', ' ').Trim();

    /// <summary>Quotes a free-form field, doubling embedded quotes (RFC 4180).</summary>
    private static string Quote(string value) =>
        "\"" + value.Replace("\"", "\"\"").Replace('\r', ' ').Replace('\n', ' ') + "\"";
}
