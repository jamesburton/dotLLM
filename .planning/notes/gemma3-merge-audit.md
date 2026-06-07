---
topic: Gemma 3 merge audit + plan
status: audit_complete_implementation_deferred
date: 2026-05-18
sequence_item: 6
referenced_commits:
  - 9bd0967  # Gemma3: Architecture enum + HF config (step 57)
  - 71ca9c3  # Gemma3: GeGLU activation kernel (step 57)
---

# Gemma 3 merge — audit + plan

## Status

Two Gemma 3 commits exist in `feature/qwen3.6` reflog but not on any
branch (`worktree-agent-a56ac89c4decfb398` removed). Audit conducted
2026-05-18 on Strix Halo dev host.

| Commit | Files | Lines | Cherry-pickable? |
|---|---|---:|---|
| `9bd0967` Architecture enum + HF config | 4 (Architecture.cs, ModelConfig.cs, HfConfigExtractor.cs, HfConfigExtractorTests.cs) | +332 | **NO — text merge corrupts the test file** |
| `71ca9c3` GeGLU activation kernel | 2 (FusedOps.cs, FusedOpsTests.cs) | +224 | Probably yes (not tried) |

## Cherry-pick failure mode (verified)

Tried `git cherry-pick --no-commit 9bd0967`. Git's text-merge reported
**no conflict** — but the resulting `HfConfigExtractorTests.cs` was
syntactically broken.

Root cause: the original commit was authored on a tree where the SmolLM3
NoPE / YaRN tests did not yet exist. The patch added the three Gemma3
tests immediately after the DeepSeekV3 test (line 503) and before the
class's closing `}`. On main, the SmolLM3 tests were inserted between
the DeepSeekV3 test and the class close in the meantime — so the
patch context `} (close DeepSeekV3 test) → } (close class)` no longer
matched, and git applied the diff at the *first* matching `}\n}` it
found, eating the opening of the SmolLM3 test in the process.

The resulting test file has:
- DeepSeekV3 test closes at line 503 with `}`.
- Lines 504+ start with the orphan body of the SmolLM3 test
  (`"rope_theta": 5000000.0,` etc.) with no `[Fact]` / `public void` /
  `{` declaration above.
- Three Gemma3 tests appended at the very end OUTSIDE the class
  (the class brace was inadvertently opened up).

This is a real "looks-clean-but-isnt" 3-way-merge failure — exactly the
class of issue that motivated the user's earlier "audit commits first,
then plan the merge" decision.

## Recommended merge plan (when ready)

**Manual file-by-file integration**, NOT cherry-pick:

### Step 1 — `src/DotLLM.Core/Configuration/Architecture.cs`

Cherry-pickable in isolation (additive enum + dispatch arm + activation
mapping). Apply the diff from `git show 9bd0967 -- src/DotLLM.Core/Configuration/Architecture.cs`
verbatim. Verify the new `Gemma3` enum variant doesn't clash with any
later additions:

```
git show 9bd0967:src/DotLLM.Core/Configuration/Architecture.cs > /tmp/gemma3-arch.cs
diff src/DotLLM.Core/Configuration/Architecture.cs /tmp/gemma3-arch.cs
```

Splice the Gemma3-specific blocks manually.

### Step 2 — `src/DotLLM.Core/Models/ModelConfig.cs`

Four new optional fields:

- `PerLayerSlidingWindow` — `IReadOnlyList<int?>?` for the
  `sliding_window_pattern` formula
- `AttnLogitSoftcap` — `float?`
- `FinalLogitSoftcap` — `float?`
- `QueryPreAttnScalar` — `float?`

All nullable / optional, so additive. Splice manually; verify no
existing fields clash with the names.

### Step 3 — `src/DotLLM.Models/SafeTensors/HfConfigExtractor.cs`

100 lines of new dispatch logic for Gemma 3 text-only and multimodal
configs, plus the `sliding_window_pattern` formula. Apply manually;
verify the `ResolveArchitecture` switch is structurally compatible
with the current state.

### Step 4 — `tests/DotLLM.Tests.Unit/Models/SafeTensors/HfConfigExtractorTests.cs`

**Manually copy** the three Gemma3 tests from `git show 71ca9c3`
(or rather, from 9bd0967 — the tests are in the same commit as the
extractor changes). Append them INSIDE the existing class (before the
final `}`), NOT cherry-pick.

### Step 5 — `src/DotLLM.Cpu/Kernels/FusedOps.cs` + tests

Probably clean cherry-pick of `71ca9c3` since `FusedOps.cs` may not
have evolved as much. Try `git cherry-pick --no-commit 71ca9c3` first;
if the test file errors, fall back to manual splice for both files.

### Step 6 — verify

- Build: `dotnet build src/DotLLM.Core/ src/DotLLM.Models/ src/DotLLM.Cpu/`
- Tests: `dotnet test tests/DotLLM.Tests.Unit/ --filter "FullyQualifiedName~HfConfigExtractor|FullyQualifiedName~FusedOps"`

### Step 7 — what's still NOT done by these two commits

The user's note in `.continue-here.md` referenced "ALiBi + softcap
parameter overloads need careful surgical merge". Those refer to FUTURE
Gemma 3 work that builds on these two commits — specifically:

- Attention.Execute soft-cap parameter integration (Gemma 2/3 cap raw
  scores via `softCap * tanh(s/softCap)`; the FA shader already supports
  this — see `attention_flash_f32.comp` line 32-33 — but the CPU
  Attention.Execute likely doesn't yet).
- Per-layer attention type dispatch (sliding vs full per
  `PerLayerSlidingWindow[layer]`).
- Final-logit soft-cap at the LM head (`FinalLogitSoftcap`).
- End-to-end Gemma 3 forward pass with all of the above.

That's the substantive Gemma 3 work. The two reflog commits are
prerequisites; they wire the config plumbing only. **End-to-end Gemma 3
inference is still 1-2 sessions of focused work even after these are
merged.**

## Recommendation

**Defer.** Gemma 3 isn't a current-model priority (no real-world dotLLM
deployment is blocked on Gemma 3 inference). The two prerequisite
commits can be revived from reflog any time. The substantive Gemma 3
work (soft-cap in Attention.Execute + per-layer attention dispatch +
final-logit cap + end-to-end test) is the real cost; the prerequisite
revival is sub-1-hour mechanical splicing on top.

If a Gemma 3 use case arrives, the merge plan above is ready to
execute. Until then, the reflog SHAs `9bd0967` and `71ca9c3` keep
the work recoverable — they won't be GC'd as long as someone runs
`git reflog` periodically. To pin them harder, create local tags:

```
git tag gemma3-step57-config-pending 9bd0967
git tag gemma3-step57-geglu-pending 71ca9c3
```

(Tags are optional; reflog persistence is fine for the next 90 days
under git's default settings.)
