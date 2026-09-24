using DotLLM.HuggingFace;
using DotLLM.Server.Models;

namespace DotLLM.Server;

/// <summary>
/// Owns in-flight <c>POST /v1/models/pull</c> downloads (#454).
/// </summary>
/// <remarks>
/// <para>
/// A job's lifetime is deliberately <b>decoupled from the HTTP request that started it</b>: a tray
/// app closing its SSE stream (or being killed) must not abort a multi-gigabyte download. Only
/// <c>DELETE /v1/models/pull/{id}</c> cancels, and cancelling leaves the hub cache's
/// <c>.incomplete</c> file in place — which is exactly what makes a later re-POST resume rather
/// than restart.
/// </para>
/// <para>
/// Jobs are keyed by <c>repo/revision/filename</c> for dedupe: POSTing the same file twice while
/// the first is running returns the running job instead of starting a competing writer against the
/// same <c>.incomplete</c> file.
/// </para>
/// </remarks>
public sealed class ModelPullManager : IDisposable
{
    private readonly object _lock = new();
    private readonly Dictionary<string, PullJob> _byId = new(StringComparer.Ordinal);
    private readonly Dictionary<string, PullJob> _byTarget = new(StringComparer.OrdinalIgnoreCase);
    private readonly Func<HuggingFaceDownloader> _downloaderFactory;
    private readonly string? _cacheRoot;
    private readonly string? _modelsDir;

    /// <summary>
    /// Creates a manager.
    /// </summary>
    /// <param name="downloaderFactory">
    /// Produces a downloader per job. Tests inject one pointed at a local stub server so no test
    /// ever reaches the real Hub.
    /// </param>
    /// <param name="cacheRoot">Hub cache root override (tests). Null = <see cref="HubCache.CacheRoot"/>.</param>
    /// <param name="modelsDir">Models mirror root override (tests).</param>
    public ModelPullManager(
        Func<HuggingFaceDownloader>? downloaderFactory = null,
        string? cacheRoot = null,
        string? modelsDir = null)
    {
        _downloaderFactory = downloaderFactory ?? (() => new HuggingFaceDownloader());
        _cacheRoot = cacheRoot;
        _modelsDir = modelsDir;
    }

    /// <summary>
    /// Starts a download, or returns the already-running job for the same target.
    /// The returned job is running in the background; await <see cref="PullJob.WaitAsync"/> or
    /// subscribe via <see cref="PullJob.Progress"/> to follow it.
    /// </summary>
    public PullJob Start(string repoId, string filename, string? revision)
    {
        revision ??= "main";
        string target = $"{repoId}@{revision}/{filename}";

        lock (_lock)
        {
            if (_byTarget.TryGetValue(target, out var existing) && existing.Status == "running")
                return existing;

            var job = new PullJob(Guid.NewGuid().ToString("N"), repoId, filename, revision);
            _byId[job.Id] = job;
            _byTarget[target] = job;
            job.Run(_downloaderFactory(), _cacheRoot, _modelsDir);
            return job;
        }
    }

    /// <summary>Looks up a job by id, or null.</summary>
    public PullJob? Get(string id)
    {
        lock (_lock) { return _byId.GetValueOrDefault(id); }
    }

    /// <summary>All known jobs, newest first.</summary>
    public IReadOnlyList<PullJob> List()
    {
        lock (_lock) { return _byId.Values.OrderByDescending(j => j.StartedAt).ToArray(); }
    }

    /// <summary>Cancels a job. Returns false when the id is unknown.</summary>
    public bool Cancel(string id)
    {
        var job = Get(id);
        if (job is null) return false;
        job.Cancel();
        return true;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            foreach (var job in _byId.Values) job.Cancel();
            _byId.Clear();
            _byTarget.Clear();
        }
    }
}

/// <summary>One background model download (#454).</summary>
public sealed class PullJob
{
    private readonly CancellationTokenSource _cts = new();
    private readonly TaskCompletionSource _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly object _stateLock = new();

    internal PullJob(string id, string repoId, string filename, string revision)
    {
        Id = id;
        RepoId = repoId;
        Filename = filename;
        Revision = revision;
        StartedAt = DateTimeOffset.UtcNow;
    }

    /// <summary>Opaque job id used by the <c>/v1/models/pull/{id}</c> routes.</summary>
    public string Id { get; }

    /// <summary>Source repo.</summary>
    public string RepoId { get; }

    /// <summary>Source filename within the repo.</summary>
    public string Filename { get; }

    /// <summary>Resolved git revision.</summary>
    public string Revision { get; }

    /// <summary>When the job started.</summary>
    public DateTimeOffset StartedAt { get; }

    /// <summary><c>running</c> | <c>completed</c> | <c>failed</c> | <c>cancelled</c>.</summary>
    public string Status { get; private set; } = "running";

    /// <summary>Bytes transferred so far, including bytes resumed from a previous attempt.</summary>
    public long BytesDownloaded { get; private set; }

    /// <summary>Total bytes once the server reports a length.</summary>
    public long? TotalBytes { get; private set; }

    private string? _error;
    private string? _blobPath;
    private string? _snapshotPath;
    private string? _modelPath;
    private DateTimeOffset? _completedAt;

    /// <summary>
    /// Raised on every progress report, and once more when the job reaches a terminal state.
    /// Handlers must not throw and must not block.
    /// </summary>
    /// <remarks>
    /// Delivery is on a thread pool thread, not the download loop: <see cref="Progress{T}"/> posts
    /// its callbacks. Ticks can therefore arrive out of order with respect to the download, and a
    /// tick dispatched after completion is dropped rather than applied (#521) — so read state from
    /// <see cref="ToDto"/> rather than assuming the last raise carries the last value.
    /// </remarks>
    public event Action<PullJob>? Progress;

    /// <summary>Requests cancellation. Leaves the <c>.incomplete</c> file so a re-POST resumes.</summary>
    public void Cancel()
    {
        try { _cts.Cancel(); } catch (ObjectDisposedException) { /* already finished */ }
    }

    /// <summary>Completes when the job reaches a terminal state. Never faults.</summary>
    public Task WaitAsync() => _completion.Task;

    /// <summary>
    /// Applies one progress report and raises <see cref="Progress"/>, unless the job has already
    /// reached a terminal state (#521).
    /// </summary>
    /// <remarks>
    /// <see cref="Progress{T}"/> has no synchronization context here, so it posts every callback to
    /// the thread pool: a tick reported in the last moments of a download can be dispatched
    /// <b>after</b> <see cref="Run"/>'s completion block has stamped the exact size. Without this
    /// guard that tick rolls <see cref="BytesDownloaded"/> backwards and <c>/v1/models/pull/{id}</c>
    /// serves a <c>completed</c> job at under 100 percent.
    /// </remarks>
    internal void ApplyProgress(long bytesDownloaded, long? totalBytes)
    {
        lock (_stateLock)
        {
            if (Status != "running") return;
            BytesDownloaded = bytesDownloaded;
            TotalBytes = totalBytes;
        }
        Progress?.Invoke(this);
    }

    internal void Run(HuggingFaceDownloader downloader, string? cacheRoot, string? modelsDir)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                var progress = new Progress<(long bytesDownloaded, long? totalBytes)>(
                    p => ApplyProgress(p.bytesDownloaded, p.totalBytes));

                var result = await downloader.DownloadToHubCacheAsync(
                    RepoId, Filename, Revision, cacheRoot, modelsDir, progress, _cts.Token)
                    .ConfigureAwait(false);

                lock (_stateLock)
                {
                    _blobPath = result.BlobPath;
                    _snapshotPath = result.SnapshotPath;
                    _modelPath = result.ModelPath;
                    BytesDownloaded = result.SizeBytes;
                    TotalBytes = result.SizeBytes;
                    Status = "completed";
                    _completedAt = DateTimeOffset.UtcNow;
                }
            }
            catch (OperationCanceledException)
            {
                lock (_stateLock) { Status = "cancelled"; _completedAt = DateTimeOffset.UtcNow; }
            }
            catch (Exception ex)
            {
                lock (_stateLock) { Status = "failed"; _error = ex.Message; _completedAt = DateTimeOffset.UtcNow; }
            }
            finally
            {
                downloader.Dispose();
                Progress?.Invoke(this);
                _completion.TrySetResult();
            }
        });
    }

    /// <summary>Point-in-time DTO snapshot of this job.</summary>
    public ModelPullJobDto ToDto()
    {
        lock (_stateLock)
        {
            return new ModelPullJobDto
            {
                Id = Id,
                RepoId = RepoId,
                Filename = Filename,
                Revision = Revision,
                Status = Status,
                BytesDownloaded = BytesDownloaded,
                TotalBytes = TotalBytes,
                Percent = TotalBytes is > 0 ? Math.Round(100.0 * BytesDownloaded / TotalBytes.Value, 2) : null,
                Error = _error,
                BlobPath = _blobPath,
                SnapshotPath = _snapshotPath,
                ModelPath = _modelPath,
                StartedAt = StartedAt.ToUnixTimeSeconds(),
                CompletedAt = _completedAt?.ToUnixTimeSeconds(),
            };
        }
    }
}
