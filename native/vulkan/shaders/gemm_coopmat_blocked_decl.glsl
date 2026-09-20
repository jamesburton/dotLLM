// SHARED BLOCKED COOPMAT GEMM TEMPLATE — declarations half. Issue #443.
//
// Extracted verbatim (geometry, LDS layout, fragment order, store paths) from the PROVEN
// matmul_pq2_0_f32_gemm_ladder.glsl, which shipped as PQ2_0GemmVariant.Ladder128x128x4:
// 12.2x on the kernel, 5.47x end to end on the real model, all 11 correctness gates.
// PQ2_0's own shaders are deliberately NOT retargeted onto this template (issue #443
// constraint 8: the PQ2_0 path is shipped and validated; its .spv stay byte-identical).
//
// ----------------------------------------------------------------------------------
// THE MECHANISM, so it is not re-derived per quant.
//
// Total weight stage/unpack work = M*K*N / BN. Each weight element is re-staged once per
// N-tile — independent of BM, inversely proportional to BN. At BN=16 with N=512 that is 32x;
// at BN=128, 4x. RGP measured 8.47x fewer VALU ops per output element against 8.0x predicted
// (#440 addendum). This is WORK DELETION, not an occupancy or arithmetic-intensity subtlety:
// (c) wins at the SAME 4/16 occupancy as the 16x16 kernel.
// ----------------------------------------------------------------------------------
//
// GLSL has no function pointers, so the per-quant dequant is a hook FUNCTION the wrapper
// defines between this file and gemm_coopmat_blocked_main.glsl. Two includes, not one,
// because the hook must see `sharedA` / `STRIDE`, which are declared here.
//
// CONTRACT — the wrapper must, BEFORE including this file:
//   #define BM / BN      output tile (weight rows x token rows) per workgroup
//   #define WM / WN      per-subgroup warp tile; (BM/WM) * (BN/WN) must equal NSG
//   #define NSG          subgroups per workgroup
//   #define WAVE         subgroup width the workgroup is sized in (64 = driver native here)
//   declare its own bindings and push constants (they differ per quant: MoE has four
//   bindings and a six-field push block).
//
// ... and BEFORE including gemm_coopmat_blocked_main.glsl:
//   #define GEMM_M           uint expr: weight rows of the whole problem
//   #define GEMM_K           uint expr: contraction length
//   #define GEMM_CHUNKS      uint expr: number of BK-sized K chunks = GEMM_K / BK
//   #define GEMM_ROW_BASE    uint expr: global token row of local token row 0
//   #define GEMM_ROW_LIMIT   uint expr: number of valid LOCAL token rows
//   #define GEMM_B_BUF       name of the activation array (row-major [rows, GEMM_K])
//   #define GEMM_C_BUF       name of the output array (row-major [rows, GEMM_M])
//   void gemmStageA(uint ch);   writes sharedA[BM * STRIDE] for K chunk `ch`
//   optionally #define GEMM_EPILOGUE_SCALE <float expr>  (a per-tensor tail scale)
//   optionally #define GEMM_HAS_PROLOGUE and `bool gemmPrologue();` returning false to bail
//
// The hook may read g_tid / g_mBase / g_tBase, set up by main() before it is called.

#extension GL_KHR_cooperative_matrix      : require
#extension GL_KHR_memory_scope_semantics  : require
#extension GL_KHR_shader_subgroup_basic   : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage    : require
// [[unroll]] on the fragment loops. Required: the coopmat accumulator array is indexed by the
// loop variables and must be register-resident, exactly as llama.cpp's mul_mm.comp relies on.
#extension GL_EXT_control_flow_attributes : require

#define BLOCK_SIZE (NSG * WAVE)

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

const uint TM = 16u;    // coopmat M per fragment
const uint TN = 16u;    // coopmat N per fragment
const uint TK = 16u;    // coopmat K per coopMatMulAdd

// BK=32 is not a tuning choice, it is what makes the big tile fit: 128x128 at BK=128 needs
// 64 KB of LDS against this device's 32 KB limit and the pipeline would not create at all.
const uint BK = 32u;

// LDS row padding, as llama.cpp's SHMEM_STRIDE_PAD: an unpadded 32-element f16 row is 64 B,
// so rows 0 and 2 would land on the same LDS banks.
const uint PAD    = 8u;
const uint STRIDE = BK + PAD;   // 40 f16 = 80 B per staged row

const uint SG_ROWS     = BM / WM;   // subgroup grid: SG_ROWS x (NSG/SG_ROWS) == NSG
const uint CMS_PER_ROW = WM / TM;   // 16x16 fragments down one subgroup's tile
const uint CMS_PER_COL = WN / TN;   // 16x16 fragments across it

const uint B_ELEMS   = BN * BK;                // activations staged per K chunk
const uint B_PER_THR = B_ELEMS / BLOCK_SIZE;

// Weight tile [BM rows][STRIDE] f16, written by the wrapper's gemmStageA.
shared float16_t sharedA[BM * STRIDE];
// Activation tile [BN rows][STRIDE] f16.
shared float16_t sharedB[BN * STRIDE];
// Per-subgroup 16x16 staging for the boundary store path.
shared float storeStage[NSG * TM * TN];

uint g_tid;     // gl_LocalInvocationID.x
uint g_mBase;   // first weight row of this workgroup's tile
uint g_tBase;   // first LOCAL token row of this workgroup's tile
