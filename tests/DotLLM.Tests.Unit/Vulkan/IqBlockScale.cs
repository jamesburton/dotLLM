namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Forces the fp16 super-scale of a random IQ-family block to a sane value (random bytes would produce inf / NaN halves). For every format
/// but IQ1_M the scale is the block's first two bytes; IQ1_M (56-byte blocks) stores it in the top nibbles of its four 16-bit scale words
/// (bytes 48..55), keeping the low 12 bits of each word random.
/// </summary>
internal static class IqBlockScale
{
    public static void Set(byte[] bytes, long blockOffset, int blockBytes, Half d)
    {
        ushort bits = BitConverter.HalfToUInt16Bits(d);
        if (blockBytes == 56)
        {
            for (int w = 0; w < 4; w++)
            {
                long o = blockOffset + 48 + 2 * w;
                ushort word = (ushort)(bytes[o] | (bytes[o + 1] << 8));
                word = (ushort)((word & 0x0FFF) | (((bits >> (4 * w)) & 0xF) << 12));
                bytes[o] = (byte)word;
                bytes[o + 1] = (byte)(word >> 8);
            }
        }
        else
        {
            bytes[blockOffset] = (byte)bits;
            bytes[blockOffset + 1] = (byte)(bits >> 8);
        }
    }
}
