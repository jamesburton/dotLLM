using System.Runtime.InteropServices;

namespace DotLLM.Vulkan.Interop;

/// <summary>
/// Minimal SPIR-V reflection: extracts, for each storage-buffer binding of a
/// compute shader, whether the shader can WRITE that binding. Drives the
/// hazard-scoped barrier tracker (<see cref="VulkanHazardTracker"/>, issue #144):
/// a dispatch's read set / write set is derived from the shader's own
/// <c>readonly</c> qualifiers (SPIR-V <c>NonWritable</c> decorations), so the
/// per-kernel read/write declaration can never drift from the shader source.
/// </summary>
/// <remarks>
/// <para>
/// Conservative by construction: a binding is treated as writable unless the
/// shader explicitly marks it non-writable (GLSL <c>readonly buffer</c> →
/// <c>OpDecorate/OpMemberDecorate NonWritable</c>). Any parse anomaly yields
/// the all-writable mask, which degenerates to blanket barriers — always safe,
/// never silently under-synchronized.
/// </para>
/// <para>
/// Handles both storage-class encodings glslang emits: <c>StorageBuffer</c>
/// (SPIR-V ≥ 1.3 / vulkan1.1+ targets) and <c>Uniform</c> + <c>BufferBlock</c>
/// (vulkan1.0 targets).
/// </para>
/// </remarks>
internal static class SpirvReflection
{
    private const uint SpirvMagic = 0x07230203;

    // Opcodes.
    private const int OpTypeStruct = 30;
    private const int OpTypePointer = 32;
    private const int OpVariable = 59;
    private const int OpDecorate = 71;
    private const int OpMemberDecorate = 72;

    // Decorations.
    private const uint DecorationBufferBlock = 3;
    private const uint DecorationNonWritable = 24;
    private const uint DecorationBinding = 33;

    // Storage classes.
    private const uint StorageClassUniform = 2;
    private const uint StorageClassStorageBuffer = 12;

    /// <summary>
    /// Computes the storage-buffer writes mask for a SPIR-V compute shader:
    /// bit <c>b</c> is set when the shader may write descriptor binding
    /// <c>b</c> (set 0). Bindings the shader marks fully <c>NonWritable</c>
    /// (GLSL <c>readonly</c>) have their bit clear. Returns <c>~0u</c>
    /// (all-writable, fully conservative) when the blob cannot be parsed.
    /// </summary>
    internal static uint ComputeStorageWritesMask(ReadOnlySpan<byte> spv)
    {
        try
        {
            return ComputeCore(spv);
        }
        catch
        {
            return ~0u; // Conservative: treat every binding as written.
        }
    }

    private static uint ComputeCore(ReadOnlySpan<byte> spv)
    {
        if (spv.Length < 20 || (spv.Length & 3) != 0)
            return ~0u;
        ReadOnlySpan<uint> words = MemoryMarshal.Cast<byte, uint>(spv);
        if (words[0] != SpirvMagic)
            return ~0u;

        uint bound = words[3];
        if (bound == 0 || bound > 4 * 1024 * 1024)
            return ~0u;

        // Per-id facts gathered in one pass. Ids are dense (< bound).
        var binding = new int[bound];           // id -> binding + 1 (0 = none)
        var varNonWritable = new bool[bound];   // OpDecorate NonWritable on the variable
        var structMemberCount = new int[bound]; // OpTypeStruct member count
        var structNwMembers = new int[bound];   // count of NonWritable member decorations
        var ptrPointee = new uint[bound];       // OpTypePointer -> pointee type id
        var blockIsBufferBlock = new bool[bound];

        // (varId, storageClass, pointerTypeId) triples.
        Span<int> varIds = stackalloc int[64];
        Span<uint> varStorage = stackalloc uint[64];
        Span<uint> varPtrType = stackalloc uint[64];
        int varCount = 0;

        int i = 5;
        while (i < words.Length)
        {
            uint instr = words[i];
            int wordCount = (int)(instr >> 16);
            int opcode = (int)(instr & 0xFFFF);
            if (wordCount <= 0 || i + wordCount > words.Length)
                return ~0u;

            switch (opcode)
            {
                case OpDecorate when wordCount >= 3:
                {
                    uint target = words[i + 1];
                    uint decoration = words[i + 2];
                    if (target < bound)
                    {
                        if (decoration == DecorationNonWritable)
                            varNonWritable[target] = true;
                        else if (decoration == DecorationBinding && wordCount >= 4)
                            binding[target] = (int)words[i + 3] + 1;
                        else if (decoration == DecorationBufferBlock)
                            blockIsBufferBlock[target] = true;
                    }
                    break;
                }
                case OpMemberDecorate when wordCount >= 4:
                {
                    uint structId = words[i + 1];
                    uint decoration = words[i + 3];
                    if (structId < bound && decoration == DecorationNonWritable)
                        structNwMembers[structId]++;
                    break;
                }
                case OpTypeStruct when wordCount >= 2:
                {
                    uint id = words[i + 1];
                    if (id < bound)
                        structMemberCount[id] = wordCount - 2;
                    break;
                }
                case OpTypePointer when wordCount >= 4:
                {
                    uint id = words[i + 1];
                    if (id < bound)
                        ptrPointee[id] = words[i + 3];
                    break;
                }
                case OpVariable when wordCount >= 4:
                {
                    if (varCount < varIds.Length)
                    {
                        varIds[varCount] = (int)words[i + 2];
                        varStorage[varCount] = words[i + 3];
                        varPtrType[varCount] = words[i + 1];
                        varCount++;
                    }
                    break;
                }
            }
            i += wordCount;
        }

        uint mask = 0;
        for (int v = 0; v < varCount; v++)
        {
            uint id = (uint)varIds[v];
            if (id >= bound || binding[id] == 0)
                continue;

            uint pointee = varPtrType[v] < bound ? ptrPointee[varPtrType[v]] : 0;
            bool isStorage = varStorage[v] == StorageClassStorageBuffer
                || (varStorage[v] == StorageClassUniform
                    && pointee < bound && blockIsBufferBlock[pointee]);
            if (!isStorage)
                continue;

            int b = binding[id] - 1;
            if (b is < 0 or > 31)
                return ~0u; // Out-of-range binding — bail conservative.

            // Non-writable when either the variable itself is decorated, or
            // every member of the backing block struct is decorated.
            bool nonWritable = varNonWritable[id];
            if (!nonWritable && pointee < bound)
            {
                int members = structMemberCount[pointee];
                nonWritable = members > 0 && structNwMembers[pointee] >= members;
            }
            if (!nonWritable)
                mask |= 1u << b;
        }
        return mask;
    }
}
