using System.Runtime.InteropServices;
using DotLLM.Core.Attention;

namespace DotLLM.Engine.KvCache;

/// <summary>Host-memory moves for <see cref="SlidingWindowRows"/> (CPU caches, issue #1169).</summary>
internal static unsafe class SlidingWindowRowsHost
{
    /// <summary>
    /// Makes positions <c>[minPos, maxPos]</c> of a windowed slot addressable, compacting or resizing its K and V buffers
    /// when they would overflow. No-op for a full slot.
    /// </summary>
    /// <param name="rows">The cache's window bookkeeping.</param>
    /// <param name="layer">Slot index.</param>
    /// <param name="minPos">Lowest position about to be written.</param>
    /// <param name="maxPos">Highest position about to be written.</param>
    /// <param name="currentLength">Committed length before the write.</param>
    /// <param name="keys">The slot's key buffer; replaced when resized.</param>
    /// <param name="values">The slot's value buffer; replaced when resized.</param>
    /// <param name="keyRowBytes">Bytes per key row.</param>
    /// <param name="valueRowBytes">Bytes per value row.</param>
    internal static void Prepare(this SlidingWindowRows rows, int layer, int minPos, int maxPos, int currentLength,
        ref nint keys, ref nint values, int keyRowBytes, int valueRowBytes)
    {
        if (!rows.TryPlan(layer, minPos, maxPos, currentLength, out var m)) return;
        if (m.Resize)
        {
            keys = Resize(keys, m.Shift, m.Keep, keyRowBytes, m.Capacity);
            values = Resize(values, m.Shift, m.Keep, valueRowBytes, m.Capacity);
        }
        else
        {
            // Overlapping move to the front; Buffer.MemoryCopy is overlap-safe.
            Move(keys, m.Shift, m.Keep, keyRowBytes);
            Move(values, m.Shift, m.Keep, valueRowBytes);
        }

        rows.Commit(layer, m);

        static void Move(nint buffer, int shift, int keep, int rowBytes)
        {
            if (shift == 0 || keep == 0) return;
            long bytes = (long)keep * rowBytes;
            Buffer.MemoryCopy((byte*)buffer + (long)shift * rowBytes, (void*)buffer, bytes, bytes);
        }

        static nint Resize(nint old, int shift, int keep, int rowBytes, int capacity)
        {
            nuint newBytes = (nuint)((long)capacity * rowBytes);
            void* p = NativeMemory.AlignedAlloc(newBytes, 64);
            long bytes = (long)keep * rowBytes;
            Buffer.MemoryCopy((byte*)old + (long)shift * rowBytes, p, (long)newBytes, bytes);
            NativeMemory.AlignedFree((void*)old);
            return (nint)p;
        }
    }
}
