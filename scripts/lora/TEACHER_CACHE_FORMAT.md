# Teacher Cache Format — byte-for-byte contract

This document specifies the exact on-disk layout consumed by `mote_train.py --teacher-cache`.
Both the producer (`materialize_kd_corpus.py` / `strix_teacher_dump.py`) and the consumer
(`mote_train.py`) must follow this contract precisely.

---

## Directory layout

```
<kd-corpus>/            # passed to --kd-corpus
    ids.npy             # [N_SEQS, SEQ_LEN]  int32  tokenized corpus (right-padded)
    meta.json           # corpus metadata (see below)

<teacher-cache>/        # passed to --teacher-cache
    topk_idx.npy        # [N_SEQS, SEQ_LEN, K]  int32   top-K vocab indices per position
    topk_logit.npy      # [N_SEQS, SEQ_LEN, K]  float16 teacher RAW logits (NOT softmaxed)
    meta.json           # cache metadata (see below)
```

---

## `<kd-corpus>/ids.npy`

- **dtype**: `int32`
- **shape**: `[N_SEQS, SEQ_LEN]`
- **encoding**: rows are right-padded with `pad_id` to a uniform `SEQ_LEN`.
- **byte order**: native (little-endian on x86).
- **load**: `np.load("ids.npy", mmap_mode='r')` — never read fully into RAM.

## `<kd-corpus>/meta.json`

```json
{
  "tokenizer": "<hf-model-id-or-local-path>",
  "seq_len":   <int>,
  "n_seqs":    <int>,
  "pad_id":    <int>,
  "mix":       "<corpus-description-string>"
}
```

---

## `<teacher-cache>/topk_idx.npy`

- **dtype**: `int32`
- **shape**: `[N_SEQS, SEQ_LEN, K]`
- **semantics**: `topk_idx[i, t, :]` contains the K vocabulary indices for which
  the teacher assigns the highest raw logits when predicting token `ids[i, t+1]`
  (standard teacher-forcing / next-token alignment).
  - Position `t = SEQ_LEN - 1` has no valid next-token target; its slice should be
    treated as padding and is excluded from the KD loss by the `[:-1]` slice in the
    training loop.
- **byte order**: native (little-endian on x86).
- **load**: `np.load("topk_idx.npy", mmap_mode='r')` — file can be many GB; rows
  are gathered on demand inside the training loop.

## `<teacher-cache>/topk_logit.npy`

- **dtype**: `float16`
- **shape**: `[N_SEQS, SEQ_LEN, K]`
- **semantics**: `topk_logit[i, t, k]` is the teacher's **raw (un-softmaxed) logit**
  for vocabulary token `topk_idx[i, t, k]`.  The consumer applies temperature scaling
  and softmax; do NOT pre-softmax on the producer side.
- **byte order**: native (little-endian on x86).
- **load**: `np.load("topk_logit.npy", mmap_mode='r')` — same mmap guidance as above.

## `<teacher-cache>/meta.json`

```json
{
  "model_id":   "<hf-model-id-or-local-path>",
  "K":          <int>,
  "seq_len":    <int>,
  "n_seqs":     <int>,
  "vocab_size": <int>
}
```

---

## Alignment invariant

The trainer validates:

```
cache.n_seqs  == corpus.n_seqs
cache.seq_len == corpus.seq_len
```

If either check fails, training is aborted with an alignment error.  Regenerate
the cache against the exact same corpus file.

---

## KD loss formula (consumer-side)

Given a batch row index `i` and temperature `T` (default 2.0):

```python
s_logits  = student(ids[i])[0, :-1, :]        # [L-1, V]  shifted student logits
tc_idx    = topk_idx[i, :-1, :].astype(int64) # [L-1, K]
tc_logit  = topk_logit[i, :-1, :].astype(f32) # [L-1, K]

s_topk     = s_logits.gather(-1, tc_idx)       # [L-1, K]
teacher_p  = softmax(tc_logit / T, dim=-1)     # [L-1, K]
student_lp = log_softmax(s_topk / T, dim=-1)   # [L-1, K]
kd         = sum(teacher_p * (log(teacher_p + eps) - student_lp), -1).mean() * T^2
```

`total_loss = lm_loss + kd_weight * kd`

---

## Notes

- If the cache was produced by the **same base model** being trained (not a stronger
  external teacher), the teacher is similarly weak at Japanese.  In that case, enable
  EN-gating (wrap the KD branch with `if is_en:`) or lower `--kd-weight` for JA heal runs.
- The cache arrays can easily exceed tens of GB.  Always load with `mmap_mode='r'` and
  gather individual rows inside the training loop; never call `np.array(topk_idx)` to
  materialise the whole file.
- Producer tools: `scripts/lora/materialize_kd_corpus.py` (corpus) and
  `scripts/lora/strix_teacher_dump.py` (teacher logits).
