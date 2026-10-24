using DotLLM.Core.Attention;

namespace DotLLM.Core.Models;

/// <summary>Storage owned by an independent recurrent prefix snapshot.</summary>
public enum RecurrentPrefixSnapshotStorage
{
    /// <summary>Backend device memory.</summary>
    Device,
    /// <summary>Page-locked host memory for GPU transport, or native host memory on CPU.</summary>
    Host,
}

/// <summary>Owned, immutable KV and recurrent state at one exact token boundary.</summary>
public interface IRecurrentPrefixSnapshot : IDisposable
{
    /// <summary>Raw state bytes owned by the snapshot.</summary>
    long ByteCount { get; }
    /// <summary>Compatibility identity, including weights, runtime geometry and KV formats.</summary>
    string Identity { get; }
    /// <summary>Number of consumed tokens.</summary>
    int PrefixLength { get; }
    /// <summary>Storage tier containing the independent bytes.</summary>
    RecurrentPrefixSnapshotStorage Storage { get; }
}

/// <summary>Captures every model-owned recurrent input alongside attention KV, independently of the live cache.</summary>
public interface IRecurrentPrefixSnapshotProvider
{
    /// <summary>Changes whenever the model or forward-affecting configuration changes.</summary>
    string PrefixSnapshotIdentity { get; }
    /// <summary>Reports available storage transports.</summary>
    bool SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage storage);
    /// <summary>Whether the active request can allocate its live KV capacity while leaving backend scratch headroom. Cold device snapshots may be reclaimed until this succeeds.</summary>
    bool HasPrefixSnapshotRequestHeadroom(IKvCache kvCache) => true;
    /// <summary>Estimates exact owned state bytes before allocating.</summary>
    long EstimatePrefixSnapshotBytes(IKvCache kvCache, int prefixLength);
    /// <summary>Captures the current model-owned state. The caller guarantees that it has consumed exactly prefixLength tokens.</summary>
    IRecurrentPrefixSnapshot CapturePrefixSnapshot(IKvCache kvCache, int prefixLength, RecurrentPrefixSnapshotStorage storage);
    /// <summary>Copies an independent snapshot to another tier without touching live state.</summary>
    IRecurrentPrefixSnapshot TransferPrefixSnapshot(IRecurrentPrefixSnapshot source, RecurrentPrefixSnapshotStorage storage);
    /// <summary>Restores exact state into the live model and cache. Callers must serialize this against forward passes.</summary>
    void RestorePrefixSnapshot(IRecurrentPrefixSnapshot snapshot, IKvCache kvCache);
}

/// <summary>Optional provider-owned, versioned persistence format.</summary>
public interface IPersistableRecurrentPrefixSnapshot : IRecurrentPrefixSnapshot
{
    /// <summary>Writes a complete bounded snapshot payload to a stream.</summary>
    void WriteTo(Stream destination);
}

/// <summary>Optional import capability for a provider's persisted snapshots.</summary>
public interface IPersistableRecurrentPrefixSnapshotProvider : IRecurrentPrefixSnapshotProvider
{
    /// <summary>Validates and loads a snapshot, rejecting a payload larger than maxBytes before allocating.</summary>
    IRecurrentPrefixSnapshot ReadPrefixSnapshot(Stream source, RecurrentPrefixSnapshotStorage storage, long maxBytes);
}

