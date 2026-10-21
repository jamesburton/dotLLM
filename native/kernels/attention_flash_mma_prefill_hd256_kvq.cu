// attention_flash_mma_prefill_hd256_kvq.cu -- issue #1094.
//
// Quantized-KV variants of attention_flash_mma_prefill_hd256_f16: the K/V history is read straight
// from KV-format Q8_0 (half d + 32 int8, 34 B / 32 elems) or Q4_0 (half d + 16 bytes of
// ADJACENT-PAIR nibbles -- the KV convention, not the GGUF weight layout; 18 B / 32 elems) blocks,
// instead of first dequantizing the whole history [0, seq_kv) into an F16 scratch per attention
// layer per prefill tile.
//
// ─── How: raw cp.async + in-place decode ────────────────────────────────────────────────
// The body is attention_flash_mma_prefill_hd256.cuh, shared verbatim with the F16 entry; only the
// tile-loader policy differs.
//   * A KV head-row (one key, one KV head: HEAD_DIM/32 = 8 whole blocks) is 272 B (Q8_0) or
//     144 B (Q4_0) of raw bytes, starting at (key*num_kv_heads + kv_head) * row_bytes. Both are
//     multiples of 16, so with a 16-byte-aligned base every raw row is 16-byte aligned and is
//     copied with 17 / 9 cp.async.cg of 16 B -- issued exactly where the F16 kernel issues its
//     copies of tile t+1, so the copy overlaps tile t's MMAs, bypassing registers.
//   * The raw row lands in the TAIL of that row's own padded F16 slot in the tile: bytes
//     [528 - row_bytes, 528) of the 528-byte slot (SMEM_STRIDE 264 halves) -- [256, 528) for Q8_0,
//     [384, 528) for Q4_0. No extra shared memory: same layout, size and 99 KB cap as the F16
//     kernel (a separate staging buffer would not fit: group 8 fills the cap exactly).
//   * The raw tile is decoded in place, row by row. Row r of a tile is owned by ONE warp for both
//     the copy and the decode (thread i -> row i>>5, the same mapping as load_tile), so only
//     warp-level ordering is needed (stage_decode_row / stage_decode_tile in the .cuh):
//       1. each lane cp_async_wait's its own copies, then __syncwarp(): the warp's raw rows are
//          complete and visible to all its lanes;
//       2. each lane decodes its 8-element chunk c of K row r and V row r from the staged raw
//          bytes INTO REGISTERS;
//       3. __syncwarp(): every lane of the warp has finished reading both rows' raw bytes;
//       4. each lane writes its 16 F16 bytes to [16c, 16c + 16) of each row slot.
//     The in-place overlap (F16 chunks c >= 16 for Q8_0 / c >= 24 for Q4_0 overwrite the raw tail)
//     is therefore safe: no raw byte of a row is overwritten before step 3, i.e. before every read
//     of it; other rows are disjoint. The pad bytes [512, 528) hold raw data but are never read by
//     ldmatrix (it reads halves 0..255 of a row). CudaAttentionFlashPrefillHd256KvqTests checks
//     the result bit for bit against the scratch path.
//   * When: tile 0 is decoded before the first barrier. Tile t+1 (copies issued at the top of
//     iteration t, landed during tile t's QK) is decoded INSIDE tile t's PV loop, one K+V row pair
//     after PV steps 3, 7, 11, so its LDS / ALU / STS issue while the tensor pipe executes the
//     preceding HMMAs; buffer (t+1)&1 is not read in iteration t and the closing barrier publishes it.
//   * Decoding uses the KVQ decode kernel's block decoders (V2LoadQ8 / V2LoadQ4::decode8 in
//     attention_mma_decode_v2.cuh -- the same code path as its global load8, only the element
//     loads come from shared memory): each element is __float2half(d * (float)q), the eager
//     dequant kernels' expression, so the tile holds exactly the halves the F16 kernel reads from a
//     dequantized scratch and the output is bit-identical to dequant-into-F16 + the F16 kernel.
//
// ─── Measured (RTX 3060, seqQ 512, 24/4/256; vs dequant-into-scratch + the F16 kernel) ─────
//   per-thread __ldg of 2-byte-aligned block bytes -> decode -> st.shared .......... 0.72-0.75x
//   raw cp.async + in-place decode before the barrier ............................. 0.83-0.84x
//   raw cp.async with the decode skipped (wrong output; bounds the copy) ........... 1.05x
//   + decode of tile t+1 as one block between QK and PV ........................... 0.86-0.88x
//   + decode interleaved with PV, one row per 2 steps .............................. 0.92-0.96x
//   + K and V rows paired per unit, tile-0 decode by the same per-row helper (this version,
//     206 registers) ................................. 0.97-1.01x @ seqKv 64k, 0.98-1.09x @ 4k
// So the copy is free once it is raw cp.async; what costs is the decode work (every q-tile block
// re-decodes the history), and it is only cheap when it overlaps the MMAs. Register prefetch,
// packed-half2 decode, hoisted addressing and wider __ldg in the first scheme changed nothing:
// there the per-thread synchronous load stage itself was the cost.
//
// Requirement: k and v must be 16-byte aligned (cp.async); the C# launcher checks it.
//
// Entries (same launch geometry / arguments as the F16 kernel; k and v are raw block bytes):
//   attention_flash_mma_prefill_hd256_kvq_q8q8 / _q4q4 / _q8q4 (K Q8_0, V Q4_0) / _q4q8

#include <stdint.h>
#include "attention_flash_mma_prefill_hd256.cuh"
#include "attention_mma_decode_v2.cuh"   // V2LoadQ8 / V2LoadQ4 block decoders (issue #988)

template <class L>
struct PrefillTileLoadStaged
{
    static constexpr int kRowBytes = HEAD_DIM / 32 * L::kBlockBytes;   // 272 (Q8_0) / 144 (Q4_0)
    static constexpr int kSlotBytes = SMEM_STRIDE * 2;                   // 528
    static constexpr int kRawOff = kSlotBytes - kRowBytes;               // 256 / 384
    static constexpr int kRawChunks = kRowBytes / 16;                    // 17 / 9
    static_assert(kRowBytes % 16 == 0 && kRawOff % 16 == 0, "raw row must be 16-byte copyable");
    static_assert(kRawChunks <= 32, "one warp copies one raw row");
    static constexpr bool kStaged = true;

    // Same interface as PrefillTileLoadF16::chunk: dst = row slot + c*8 halves, elem = the F16
    // element index (row * HEAD_DIM + c*8). Lane c < kRawChunks copies raw 16-byte chunk c.
    __device__ __forceinline__ static void chunk(half* dst, const void* base, size_t elem)
    {
        const int c = (int)((elem >> 3) & 31);
        if (c < kRawChunks)
        {
            unsigned char* slot = reinterpret_cast<unsigned char*>(dst - c * 8);
            const unsigned char* src = reinterpret_cast<const unsigned char*>(base) + (elem >> 8) * kRowBytes;
            cp_async16(slot + kRawOff + 16 * c, src + 16 * c);
        }
    }

    static constexpr int kSlots = 3;   // rows per warp interleaved with PV: ceil(KV_TILE / 6)

    __device__ __forceinline__ static uint4 decode_chunk(const half* row, int c)
    {
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(row) + kRawOff;
        return L::template decode8<V2PlainLd>(raw + (c >> 2) * L::kBlockBytes, (c & 3) * 8);
    }

    // Row r of the tile, this lane's chunk: read the staged raw bytes and decode into registers /
    // write the decoded F16 chunk. The caller orders all reads of a row before any write of it.
    __device__ __forceinline__ static uint4 read_row(const half* tile, int r)
    {
        return decode_chunk(tile + (size_t)r * SMEM_STRIDE, threadIdx.x & 31);
    }

    __device__ __forceinline__ static void write_row(half* tile, int r, uint4 h)
    {
        *reinterpret_cast<uint4*>(tile + (size_t)r * SMEM_STRIDE + (threadIdx.x & 31) * 8) = h;
    }
};

#define KVQ_PREFILL_ENTRY(NAME, KLOAD, VLOAD)                                                     \
extern "C" __global__ void __launch_bounds__(MAX_GROUP * 32, 1) NAME(                             \
    const half* __restrict__ q, const uint8_t* __restrict__ k, const uint8_t* __restrict__ v,     \
    float* __restrict__ output, const int seq_q, const int seq_kv, const int num_heads,           \
    const int num_kv_heads, const int position_offset, const float scale)                         \
{                                                                                                 \
    attn_flash_prefill_hd256_body<PrefillTileLoadStaged<KLOAD>, PrefillTileLoadStaged<VLOAD>>(    \
        q, k, v, output, seq_q, seq_kv, num_heads, num_kv_heads, position_offset, scale);         \
}

KVQ_PREFILL_ENTRY(attention_flash_mma_prefill_hd256_kvq_q8q8, V2LoadQ8, V2LoadQ8)
KVQ_PREFILL_ENTRY(attention_flash_mma_prefill_hd256_kvq_q4q4, V2LoadQ4, V2LoadQ4)
KVQ_PREFILL_ENTRY(attention_flash_mma_prefill_hd256_kvq_q8q4, V2LoadQ8, V2LoadQ4)
KVQ_PREFILL_ENTRY(attention_flash_mma_prefill_hd256_kvq_q4q8, V2LoadQ4, V2LoadQ8)
