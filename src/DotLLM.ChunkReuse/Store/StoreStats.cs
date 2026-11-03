namespace DotLLM.ChunkReuse.Store;

/// <summary>Size of a <see cref="FileChunkStore"/>: entries and leftover temp files counted separately.</summary>
/// <param name="Entries">Entry files (<c>.ckv</c> with a well-formed key name; content not validated).</param>
/// <param name="EntryBytes">Bytes of the entry files (<see cref="IChunkStore.TotalBytes"/>).</param>
/// <param name="TempFiles">Store temp files (<c>&lt;key&gt;.&lt;guid&gt;.tmp</c>): writes in flight or left by a crashed writer.</param>
/// <param name="TempBytes">Bytes of the temp files.</param>
public readonly record struct StoreStats(int Entries, long EntryBytes, int TempFiles, long TempBytes);
