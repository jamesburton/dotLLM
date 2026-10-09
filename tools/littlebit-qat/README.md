# littlebit-qat

Clean-room LittleBit-style sub-1-bit QAT trainer (issue #860). Written from a functional spec and the dotLLM spike
(`src/DotLLM.Cpu/Kernels/Experimental/LittleBit.cs`, #832); no third-party LittleBit code. Not wired into CI except the
packing fixture test (`tests/DotLLM.Tests.Unit/Cpu/Kernels/LittleBitPackedFormatTests.cs`).

Files: `littlebit.py` (rank formula, SmoothSign, int32 packing, Dual-SVID+ITQ init, `LittleBitLinear`, export/load),
`train.py` (frozen-teacher KL + 10 x hidden-state MSE, AdamW cosine), `data.py` (WikiText-2 train + C4 shard slice -> uint32 blocks),
`eval_ppl.py` (WikiText-2 test PPL), `smoke_test.py` (CPU; `--write-fixture PATH` regenerates the C# fixture).

Run (T5500, Python 3.11 with torch cu126): `python data.py --out train.bin`, then
`python train.py --data train.bin --out run --eff-bit 0.55 --steps N --accum 4`. Take `scripts/gpu-lock.sh` first.

## Export format (checkpoint dir)
`model.safetensors` + `config.json` + tokenizer + `littlebit_config.json`. Per quantised linear `<prefix>.`:
`U_packed`/`V_packed` int32 `(out, ceil(r/32))` / `(r, ceil(in/32))`, `*_shape` int64, `u1 (1,out) u2 (1,r) v1 (1,r) v2 (1,in)` bf16,
`_R` suffix for the residual path, `bias`, scalar buffers `_eff_bit_target`, `_split_dim_final`, `_eff_bit_actual`.
Bit 1 = -1, LSB-first, padded with +1. Everything except `lm_head*` linears is quantised; embeddings/norms stay bf16.

## Choices that deviate from / extend the spec
- fp32 master weights (latents, scales; fp32 copies for bf16 norms/biases) with bf16 compute: bf16 params cannot absorb lr 4e-5 updates.
- KL + lm_head computed in 256-token chunks under checkpointing (vocab 152k x seq 2048 fp32 logits are 1.2 GB each).
- KD reduction defaults to the spec's `batchmean` on 3-D logits (sum over positions); `--kd-reduction token` gives the per-token mean.
- Grad-norm clip 1.0 (HF Trainer default; the spec is silent). Rank-1 scale SVD uses 4 power iterations (spec silent). Joint-ITQ on by default.
- Token budget is sized to the 3060 (see issue), not the 5-epoch C4 shard recipe. No resume logic yet (train_state.pt is written).

## Mismatches with the dotLLM spike
1. Packing is byte-identical on little-endian (int32 word `c/32` bit `c%32` == byte `c/8` bit `c%8`; U row = `RPad/8` bytes, V row-per-latent = `DInPad/8` bytes; both pad with 0). Proven by the C# test.
2. Zero handling: spec `sign(0)=+1`; spike `FromSigns` maps `<=0` to -1. Moot for already-packed input.
3. Scales: checkpoint bf16 `u1,u2,v1,v2` vs spike fp16 `h,g,l` with `l = bf16(v1*u2)`. bf16->fp16 is exact only for |x| in [6.1e-5, 65504]; export reports the count outside.
4. Spike `LittleBitPath.PaperBits` = `2r(a+b+1)+16(a+b+r)` per path double-counts sign bits (spec/authors: `r(a+b)+16(a+b)+16r`). 1024x3072, r=192, 2 paths: spec 1,710,080 bits (0.544 bpw), spike 3,283,712. Not changed here.
5. The checkpoint stores v1 and u2 separately (+16r bits/path over the authors' accounting); trainer reports both `bits_spec` and `bits_stored`.

## Resumable run on the T5500
`t5500/launch.ps1 -Out C:/littlebit/out -Log C:/littlebit/main.log -ArgFile args.txt` starts `run_train.sh` (takes the gpu-lock, runs `train.py --resume`) and
`watcher.sh` (refreshes the lock every 4 min while the trainer's Windows pid lives, releases it when it exits) as detached processes.
`train.py` checkpoints every 100 steps (keeps 2), evals WikiText-2 PPL (first 40 windows) every 100, exports every 500 and at the end.
First 0.55 bpw run (1500 steps x 8192 tokens = 12.3M tokens, ~4.7 h, ~720 tok/s): 40-window PPL 3496 (step 100) -> 401 (step 1500); teacher 19.6.
