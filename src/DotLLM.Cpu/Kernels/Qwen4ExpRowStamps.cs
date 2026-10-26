using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Content-identity stamps for append-only row stores (the Qwen4-Exp pooled indexer keys and the state's own K/V rows), used to make
/// a checkpoint / restore copy only the rows that actually differ (#840).
/// </summary>
/// <remarks>
/// <para>Rows are grouped (<see cref="Group"/> rows per stamp). Every write to a group stamps it with a fresh process-unique value
/// and a copy between two stores carries the stamp across, so <b>equal stamps mean byte-identical group content</b> regardless of
/// which store or lineage the groups came from: a checkpoint shell that last synced from this store, or a store restored from a
/// different history, are both handled by the same comparison, with no assumption that only appends happened in between.</para>
/// <para>The stamp table is a managed <c>long[]</c> grown geometrically (amortised; steady state allocates nothing).</para>
/// </remarks>
internal sealed unsafe class Qwen4ExpRowStamps
{
    /// <summary>Rows per stamp.</summary>
    public const int Group = 4;

    private static long s_next;
    private long[] _stamps = new long[64];

    /// <summary>Rows physically copied by the last <see cref="DeltaCopy"/> (test / accounting hook).</summary>
    public long LastCopiedRows { get; private set; }

    /// <summary>Records that the last sync copied nothing.</summary>
    public void ResetCopied() => LastCopiedRows = 0;

    /// <summary>Stamps the groups covering rows <c>[firstRow, firstRow + count)</c> with a fresh value (call after writing them).</summary>
    public void Touch(int firstRow, int count)
    {
        if (count <= 0) return;
        int g0 = firstRow / Group, g1 = (firstRow + count - 1) / Group;
        Ensure(g1 + 1);
        long st = Interlocked.Increment(ref s_next);
        for (int g = g0; g <= g1; g++) _stamps[g] = st;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Ensure(int groups)
    {
        if (_stamps.Length >= groups) return;
        Array.Resize(ref _stamps, Math.Max(groups, _stamps.Length * 2));
    }

    /// <summary>
    /// Makes <paramref name="dst"/>'s first <paramref name="rows"/> rows equal <paramref name="src"/>'s, copying only the groups whose
    /// stamps differ (adjacent differing groups are coalesced into one copy) and adopting the source stamps.
    /// </summary>
    /// <returns>Number of rows copied.</returns>
    public static long DeltaCopy(Qwen4ExpRowStamps src, Qwen4ExpRowStamps dst, int rows, float* srcData, float* dstData, int rowFloats)
    {
        long copied = 0;
        if (rows > 0)
        {
            int groups = (rows + Group - 1) / Group;
            src.Ensure(groups); dst.Ensure(groups);
            long[] s = src._stamps, d = dst._stamps;
            int g = 0;
            while (g < groups)
            {
                if (s[g] == d[g]) { g++; continue; }
                int run = g;
                while (run < groups && s[run] != d[run]) { d[run] = s[run]; run++; }
                int r0 = g * Group, r1 = Math.Min(run * Group, rows);
                long floats = (long)(r1 - r0) * rowFloats;
                Buffer.MemoryCopy(srcData + (long)r0 * rowFloats, dstData + (long)r0 * rowFloats, floats * sizeof(float), floats * sizeof(float));
                copied += r1 - r0;
                g = run;
            }
        }
        dst.LastCopiedRows = copied;
        return copied;
    }
}
