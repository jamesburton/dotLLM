namespace DotLLM.Server;

/// <summary>
/// Tracks which model keys an operator has <b>disabled</b> (#454): visible in listings but not
/// loadable, so a client (e.g. the tray app) can present a curated subset of what is on disk
/// without deleting anything.
/// </summary>
/// <remarks>
/// <para>
/// The key is the same string chat/completion requests match against and
/// <c>GET /v1/models</c> reports — <see cref="ServerOptions.ModelId"/>, which for a file-resolved
/// model is <c>Path.GetFileNameWithoutExtension(path)</c>. Comparison is ordinal-ignore-case, the
/// same as <see cref="ServerState.EnsureActiveAsync"/>'s own key matching.
/// </para>
/// <para>
/// <b>Disabling does not unload.</b> An already-active model stays active and keeps serving until
/// it is unloaded explicitly (<c>POST /v1/models/unload</c>) or idles out; disabling only blocks
/// the next <i>activation</i>. Unload and disable are separate verbs on purpose so a tray can
/// "hide this from the list" without interrupting a running generation.
/// </para>
/// <para>
/// <b>State is in-memory only</b> and is lost on restart. Persisting it would mean introducing a
/// server-owned settings file, which is out of scope for this issue.
/// </para>
/// </remarks>
public sealed class ModelCatalog
{
    private readonly object _lock = new();
    private readonly HashSet<string> _disabled = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Whether the given model key may be activated. Unknown keys are enabled.</summary>
    public bool IsEnabled(string? key)
    {
        if (string.IsNullOrWhiteSpace(key)) return true;
        lock (_lock) { return !_disabled.Contains(key); }
    }

    /// <summary>Marks a key disabled. Returns true when this call changed the state.</summary>
    public bool Disable(string key)
    {
        lock (_lock) { return _disabled.Add(key); }
    }

    /// <summary>Marks a key enabled. Returns true when this call changed the state.</summary>
    public bool Enable(string key)
    {
        lock (_lock) { return _disabled.Remove(key); }
    }

    /// <summary>Snapshot of the currently-disabled keys, sorted for stable output.</summary>
    public IReadOnlyList<string> DisabledKeys()
    {
        lock (_lock) { return _disabled.OrderBy(k => k, StringComparer.OrdinalIgnoreCase).ToArray(); }
    }
}
