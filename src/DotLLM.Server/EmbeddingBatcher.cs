using System.Collections.Concurrent;
using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;

namespace DotLLM.Server;

/// <summary>
/// Server-side dynamic batching for <c>/v1/embeddings</c> and <c>/api/embed</c> (#1405).
/// </summary>
/// <remarks>
/// <para>Concurrent requests used to queue on the request gate and run one sequence at a time. Here each request
/// enqueues a <see cref="Job"/>; one drain loop collects every queued job (after an optional short wait window when the
/// batch is small), takes the gate <b>once</b>, and hands all sequences to
/// <see cref="IEmbeddingModel.ForwardHiddenBatch"/> so a backend with inter-sequence parallelism fills every core.
/// With nothing queued behind it a lone request runs after at most the window.</para>
/// <para>Results land in per-job arrays by sequence index, so output order is the request's input order regardless of
/// execution order. Pooling / normalisation / truncation are per job.</para>
/// </remarks>
internal sealed class EmbeddingBatcher(ServerState state)
{
    /// <summary>One request's worth of sequences plus how to post-process them.</summary>
    internal sealed class Job(string modelId, IReadOnlyList<int[]> sequences, PoolingType pooling, int hiddenSize, int outDims, bool normalize, CancellationToken ct)
    {
        public readonly string ModelId = modelId;
        public readonly IReadOnlyList<int[]> Sequences = sequences;
        public readonly PoolingType Pooling = pooling;
        public readonly int HiddenSize = hiddenSize;
        public readonly int OutDims = outDims;
        public readonly bool Normalize = normalize;
        public readonly CancellationToken Ct = ct;
        public readonly float[][] Vectors = new float[sequences.Count][];
        public readonly TaskCompletionSource Done = new(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    private readonly ConcurrentQueue<Job> _queue = new();
    private int _draining;

    /// <summary>Wait window used to let concurrent requests coalesce when the pending batch is small. 0 disables (natural batching only).</summary>
    internal TimeSpan Window { get; set; } = TimeSpan.FromMilliseconds(
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_EMBED_BATCH_WINDOW_MS"), out int ms) ? ms : 2);

    /// <summary>Stop collecting jobs once this many sequences are pending (a job is never split).</summary>
    internal int MaxSequences { get; set; } = 256;

    /// <summary>Largest batch executed so far (diagnostics / tests).</summary>
    internal int LargestBatchSequences => Volatile.Read(ref _largest);
    private int _largest;

    /// <summary>Number of batches executed so far (diagnostics / tests).</summary>
    internal int BatchesRun => Volatile.Read(ref _batches);
    private int _batches;

    /// <summary>Queues <paramref name="job"/> and completes when its <see cref="Job.Vectors"/> are filled or it failed.</summary>
    internal async Task RunAsync(Job job)
    {
        _queue.Enqueue(job);
        if (Interlocked.CompareExchange(ref _draining, 1, 0) == 0)
            _ = Task.Run(DrainAsync);
        await job.Done.Task.WaitAsync(job.Ct).ConfigureAwait(false);
    }

    private async Task DrainAsync()
    {
        try
        {
            while (true)
            {
                if (_queue.IsEmpty)
                {
                    Volatile.Write(ref _draining, 0);
                    // A job may have been enqueued between the emptiness check and the flag release.
                    if (_queue.IsEmpty || Interlocked.CompareExchange(ref _draining, 1, 0) != 0) return;
                    continue;
                }

                if (Window > TimeSpan.Zero && PendingSequences() < Environment.ProcessorCount)
                    await Task.Delay(Window).ConfigureAwait(false);

                var jobs = TakeBatch();
                if (jobs.Count > 0)
                    await ExecuteAsync(jobs).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            // Never leave a drain loop dead with queued jobs: fail them and let the next request restart the loop.
            while (_queue.TryDequeue(out var j)) j.Done.TrySetException(ex);
            Volatile.Write(ref _draining, 0);
        }
    }

    private int PendingSequences()
    {
        int n = 0;
        foreach (var j in _queue) n += j.Sequences.Count;
        return n;
    }

    /// <summary>Dequeues jobs for the head job's model up to <see cref="MaxSequences"/>; other models' jobs wait for the next round.</summary>
    private List<Job> TakeBatch()
    {
        var taken = new List<Job>();
        var deferred = new List<Job>();
        string? model = null;
        int seqs = 0;
        while (_queue.TryDequeue(out var j))
        {
            if (j.Ct.IsCancellationRequested) { j.Done.TrySetCanceled(j.Ct); continue; }
            model ??= j.ModelId;
            if (!string.Equals(model, j.ModelId, StringComparison.OrdinalIgnoreCase) || (taken.Count > 0 && seqs + j.Sequences.Count > MaxSequences))
            {
                deferred.Add(j);
                continue;
            }
            taken.Add(j);
            seqs += j.Sequences.Count;
        }
        foreach (var j in deferred) _queue.Enqueue(j);   // order among deferred jobs is preserved (single drain loop)
        return taken;
    }

    private async Task ExecuteAsync(List<Job> jobs)
    {
        try
        {
            await state.ExecuteOnModelAsync(jobs[0].ModelId, () =>
            {
                var model = state.Model as IEmbeddingModel
                    ?? throw new ModelUnavailableException(jobs[0].ModelId, "the active model does not support embedding extraction");
                var live = jobs.Where(j => !j.Ct.IsCancellationRequested).ToList();
                foreach (var j in jobs) if (j.Ct.IsCancellationRequested) j.Done.TrySetCanceled(j.Ct);
                if (live.Count == 0) return Task.CompletedTask;
                RunBatch(model, live);
                return Task.CompletedTask;
            }, CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            foreach (var j in jobs) j.Done.TrySetException(ex);
        }
    }

    private void RunBatch(IEmbeddingModel model, List<Job> live)
    {
        var all = new List<int[]>();
        var owner = new List<(Job Job, int Index)>();
        foreach (var j in live)
            for (int i = 0; i < j.Sequences.Count; i++) { all.Add(j.Sequences[i]); owner.Add((j, i)); }

        Interlocked.Increment(ref _batches);
        int prev;
        while ((prev = Volatile.Read(ref _largest)) < all.Count && Interlocked.CompareExchange(ref _largest, all.Count, prev) != prev) { }

        try
        {
            model.ForwardHiddenBatch(all, (idx, hidden) =>
            {
                var (job, local) = owner[idx];
                job.Vectors[local] = Finish(job, hidden, all[idx].Length);
            }, deviceId: 0);
            foreach (var j in live) j.Done.TrySetResult();
        }
        catch (Exception) when (live.Count > 1)
        {
            // One bad request must not fail its neighbours: redo each job alone and report its own outcome.
            foreach (var j in live)
            {
                try
                {
                    var seqs = j.Sequences;
                    model.ForwardHiddenBatch(seqs, (idx, hidden) => j.Vectors[idx] = Finish(j, hidden, seqs[idx].Length), deviceId: 0);
                    j.Done.TrySetResult();
                }
                catch (Exception ex) { j.Done.TrySetException(ex); }
            }
        }
        catch (Exception ex)
        {
            foreach (var j in live) j.Done.TrySetException(ex);
        }
    }

    private static float[] Finish(Job job, ReadOnlySpan<float> hidden, int tokenCount)
    {
        var vector = new float[job.HiddenSize];
        EmbeddingPooler.Pool(hidden, tokenCount, job.HiddenSize, job.Pooling, vector);
        if (job.OutDims < job.HiddenSize)
            vector = vector.AsSpan(0, job.OutDims).ToArray();
        if (job.Normalize)
            EmbeddingPooler.L2Normalize(vector);
        return vector;
    }
}
