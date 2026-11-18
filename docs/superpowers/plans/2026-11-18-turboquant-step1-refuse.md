# TurboQuant step 1: refuse loudly — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A TurboQuant KV request either runs TurboQuant for real or fails at startup with an actionable message. It is never
silently served by another cache, on any backend, command or setting.

**Architecture:**
- A pure resolver, `TurboQuantKvSupport.RefusalReason`, in `DotLLM.Server`, decides whether TurboQuant is real for a given
  context.
- `ServerStartup`, `RunCommand` and `ChatCommand` call it before loading (cheap rules) and again after loading (host rules).
- In every KV branch chain, the TurboQuant branch comes first, so no later branch can catch a tq config.
- After step 1, the only real path is CPU `serve --no-paged` on the plain `TransformerModel`, through the existing
  `TurboQuantKvCache`. That path was already there before this step.

**Tech Stack:** .NET 10, xUnit, Spectre.Console CLI.

**Spec:** `docs/superpowers/specs/2026-11-18-turboquant-in-place-design.md` (Goal 1, step 1).

## Global Constraints

- One issue, one branch, one PR, targeting `dev`:
  - Branch `issue/1888-turboquant-refuse`.
  - Commits carry `(#1888)`.
  - The PR body says `Closes #1888`. Close the issue by hand after the merge, because `Closes` doesn't auto-close on `dev`.
- **Every refusal message:**
  - names the unsupported part;
  - suggests `q8_0`/`q4_0`;
  - cites the epic issue number.
- K and V are judged independently. A config is TurboQuant when **either** side is `KvCacheDType.TurboQuant`.
- **Mixed or V-only tq:** refused, with "mixed K/V pairs with TurboQuant arrive in step 6".
- **CPU `tq*q` (QJL):** allowed on the real path. It is pre-existing and not removed (spec call).
- **run/chat:** refuse every tq form in step 1. Step 2 wires all three commands at once, and no new consumer of the scratch
  path is added now. **(plan call)**
- **Prove each guard test can fail:** list the mutants applied and the ones that died in the PR body (CLAUDE.md).
- CPU-only test filter: `FullyQualifiedName!~DotLLM.Tests.Unit.Vulkan&Category!=GPU`.
- `docs/CLI.md` tables are generated. Update the help constants, regenerate, and keep `CliHelpTests` green.

## Review Focus

1. **`--device auto`**, which resolves to a GPU only after loading. The pre-load check must not wave it through; the
   post-load check must refuse once the model turns out to be CUDA, Vulkan or the hybrid split. Test in Task 3.
2. **`serve` with the paged default and tq4.** This is the most common accidental input. It must refuse with a hint to pass
   `--no-paged`. Test in Task 3.
3. **A model with an embedded MTP head and `MtpEnabled` left at its default `true`.** Refuse with a hint to pass `--no-mtp`.
   Never drop MTP silently. Test in Task 2.
4. **`--cache-type-v tq4` with K left unset.** Today `IsTurboQuant` reads only K, so this slips through. It must refuse as
   V-only. Test in Task 1 and Task 2.
5. **Bare `turboquant` / `tq` (default bits).** It must go through the same gate as `tq4`. Test in Task 2.

---

### Task 0: Issues and branch

- [ ] **Step 1:** File the epic "TurboQuant KV in place on every backend". The body links the spec, lists steps 1-6, and
  states the merge gates.
- [ ] **Step 2:** File the step-1 issue ("refuse loudly"). Its acceptance criteria are this plan's Goal, Review Focus and
  Global Constraints. Link it from the epic.
- [ ] **Step 3:** Comment on #1881:
  - The premise doesn't hold: nothing in production builds the Vulkan TurboQuant cache, and tq is silently dropped on every
    backend.
  - James redirected the work to the epic.
  - #1881's ask becomes step 4.
  
  Leave #1881 open. Step 4 closes it.
- [ ] **Step 4:** Create a worktree on `issue/1888-turboquant-refuse` from `dev`. Commit the spec and this plan as the first
  commit.

### Task 1: Config predicates and the G3 fix

**Files:**
- Modify: `src/DotLLM.Core/Configuration/KvCacheConfig.cs` (beside `IsTurboQuant`, line 38)
- Modify: `src/DotLLM.Cuda/CudaTransformerModel.cs` (`G3ServesKvCache`, about line 4966)
- Test: `tests/DotLLM.Tests.Unit/Core/KvCacheConfigTurboQuantTests.cs` (create)

**Interfaces:**
- Produces:
  - `bool KvCacheConfig.IsTurboQuantAny` (K or V is TurboQuant);
  - `bool KvCacheConfig.IsTurboQuantMixed` (exactly one side is TurboQuant).
  - `IsTurboQuant` keeps its meaning (K is TurboQuant). Update its XML doc to point at `IsTurboQuantAny`.
- `CudaTransformerModel.G3ServesKvCache(kv)` returns `!kv.IsQuantized && !kv.IsTurboQuantAny`.

- [ ] **Step 1: Write the failing tests:**
  - `VOnlyTurboQuant_IsAny_AndMixed`: K=F32, V=TurboQuant → `IsTurboQuantAny` true, `IsTurboQuantMixed` true,
    `IsTurboQuant` false.
  - `BothTurboQuant_IsAny_NotMixed`.
  - `NeitherTurboQuant_IsNotAny`: Q8_0/Q4_0 pairs.
  - `G3_DoesNotServe_TurboQuant`: `G3ServesKvCache(new(TurboQuant, TurboQuant))` is false. A V-only TurboQuant config is
    also false. `G3ServesKvCache(KvCacheConfig.Default)` stays true.
- [ ] **Step 2:** Run with `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~KvCacheConfigTurboQuant"`.
  Expected: FAIL, because the members don't exist yet.
- [ ] **Step 3:** Implement both predicates and the G3 change. If `G3ServesKvCache` is `internal`, make it visible to the
  test assembly. Check `InternalsVisibleTo` first, and match how existing tests call it.
- [ ] **Step 4:** Run again. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(kv): IsTurboQuantAny/Mixed; G3 never serves TurboQuant KV (#1888)`.

### Task 2: The resolver

**Files:**
- Create: `src/DotLLM.Server/TurboQuantKvSupport.cs`
- Test: `tests/DotLLM.Tests.Unit/Server/TurboQuantKvSupportTests.cs`

**Interfaces:**
- Produces:
  ```csharp
  public enum KvBackendKind { Unknown, Cpu, Cuda, Vulkan }           // Unknown = --device auto before load
  public enum KvCommand { Serve, Run, Chat }
  public sealed record TurboQuantContext(
      KvCommand Command, KvBackendKind Backend, bool Paged,
      ModelConfig Config, bool? HostIsPlainCpuTransformer,          // null = not loaded yet
      bool HasDraftModel, bool MtpWillAttach);
  public static class TurboQuantKvSupport
  {
      public const int MaxHeadDim = 256;
      public static int EpicIssue => /* epic number from Task 0 */;
      public static string? RefusalReason(KvCacheConfig kv, TurboQuantContext ctx);   // null = TurboQuant is real here
      public static KvBackendKind BackendOf(IModel model);           // Vulkan*/Cuda* (incl. HybridTransformerModel) by type
  }
  ```
- **Rule order** (the first match wins; every message ends with the epic reference):
  1. `!kv.IsTurboQuantAny` → null.
  2. `kv.IsTurboQuantMixed` → mixed/V-only refusal (step 6).
  3. `Backend is Cuda or Vulkan` → "TurboQuant KV does not run on {backend} yet (step 2)".
  4. `Command is Run or Chat` → "use `dotllm serve --device cpu --no-paged` until step 2".
  5. `Paged` → "the paged cache has no TurboQuant blocks yet (step 3); pass `--no-paged`".
  6. Head dim not a power of two, or above `MaxHeadDim`, or `!KvGeometry.FromConfig(Config).IsUniform` → refuse, giving the
     head dim.
  7. `HasDraftModel` → refuse (the draft's KV would not be TurboQuant).
  8. `MtpWillAttach` → refuse, hint `--no-mtp`.
  9. `HostIsPlainCpuTransformer == false` → refuse, naming the architecture.
  10. Otherwise → null.
  
  `Backend == Unknown` and `HostIsPlainCpuTransformer == null` skip rules 3 and 9. That is the pre-load pass.

- [ ] **Step 1: Write the failing tests.** Build configs like `PagedKvBlockTypeDefaultTests.Dense` with
  `HeadDim = 64, NumKvHeads = 4`.
  - `[Theory]` `Refuses(case)`, one row per rule 2-9, each asserting:
    - non-null;
    - the expected distinguishing substring (`"step 6"`, `"vulkan"`, `"cuda"`, `"--no-paged"`, `"head dim 80"`, `"draft"`,
      `"--no-mtp"`, `"Qwen3Hybrid"`);
    - `$"#{TurboQuantKvSupport.EpicIssue}"`.
  - `AllowsRealCpuPath`: `(Serve, Cpu, Paged false, HostIsPlainCpuTransformer true)` with tq4, tq4q and bare `turboquant`
    (via `KvCacheConfig.ParseDType`) → null.
  - `NotTurboQuant_IsNeverRefused`: Default, Q8_0 and Q4_0 configs on every backend → null.
  - `PreLoadPass_SkipsHostAndBackendRules`: `Backend Unknown`, `Host null` → null. `Paged true` in that same context still
    refuses.
  - `VOnly_IsRefused`: K=F32, V=TurboQuant on CPU, serve, no-paged → contains `"step 6"`.
- [ ] **Step 2:** Run with `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~TurboQuantKvSupport"`.
  Expected: FAIL (no type).
- [ ] **Step 3:** Implement `TurboQuantKvSupport`. `BackendOf` matches on the concrete type's namespace (`DotLLM.Vulkan*`
  → Vulkan, `DotLLM.Cuda*` → Cuda, else Cpu). Add a test for a CPU `TransformerModel` (TinyQwen) → Cpu.
- [ ] **Step 4:** Run again. Expected: PASS.
- [ ] **Step 5: Mutation.** Apply each mutant and confirm a test dies; record them for the PR:
  - swap rules 2 and 3;
  - delete rule 5;
  - make `IsTurboQuantAny` read K only;
  - drop the epic suffix.
  
  Restore. Commit before mutating, because `git checkout` can't restore untracked files.
- [ ] **Step 6:** Commit: `feat(server): TurboQuantKvSupport resolver (#1888)`.

### Task 3: Wire serve

**Files:**
- Modify: `src/DotLLM.Server/ServerStartup.cs`:
  - pre-load pass after `ValidateSpeculativeDraftEarly` (about line 654);
  - post-load pass and the tq-first branch where `kvConfig` is built (about lines 710-910);
  - CUDA `CreateKvCache` callers.
- Test: `tests/DotLLM.Tests.Unit/Server/ServerTurboQuantWiringTests.cs` (uses `PrefixCacheFailedRequestTests.TinyQwen`)

**Interfaces:**
- Consumes: `TurboQuantKvSupport`, `TurboQuantContext`, `KvCacheConfig.IsTurboQuantAny` (Tasks 1-2).
- Refusals throw `InvalidServerOptionException(reason)`.
- **Pre-load:**
  - `Backend` comes from `DeviceSpec.Parse(options.Device).Kind`. `Auto` maps to `Unknown`.
  - `MtpWillAttach` is false in this pass; it is checked after loading.
- **Post-load:**
  - `Backend = BackendOf(model)`;
  - `HostIsPlainCpuTransformer = model.GetType() == typeof(TransformerModel)`;
  - `MtpWillAttach = options.MtpEnabled && model.SupportsMtp`;
  - `HasDraftModel = !string.IsNullOrEmpty(options.SpeculativeModel)`.
- Parse V with bits as well. `KvCacheConfig.TurboQuantBits` comes from whichever side is tq. Both sides already agree, since
  mixed is refused.
- **Branch order:** after the post-load pass returns null, the CPU branch chain starts with `if (kvConfig.IsTurboQuantAny)`,
  which builds the existing `TurboQuantKvCache` factory. Move the current `else if (kvConfig.IsTurboQuant)` branch there.
  No other branch may receive a tq config.

- [ ] **Step 1: Write the failing tests.** `LoadModel(_tiny.GgufPath, options)` with `Warmup = Disabled`, `Device = "cpu"`:
  - `Tq4_NoPaged_BuildsTurboQuantKvCache`:
    - `UsePaged = false`, `CacheTypeK = "tq4"`, `CacheTypeV = "tq4"`.
    - `state.KvCacheFactory!(config, 32)` is a `TurboQuantKvCache` with `Bits == 4`.
  - `Tq4_Paged_Refuses`: `UsePaged = true` → `InvalidServerOptionException` whose message contains `"--no-paged"`.
  - `Tq4_VOnly_Refuses`: `CacheTypeV = "tq4"` only → the message contains `"step 6"`.
  - `Tq4q_NoPaged_BuildsQjlCache`: `UseQjl == true`.
  - `Tq4_DeviceVulkan_RefusesBeforeLoad`: `[SkippableFact]` needing a Vulkan device, plus a CPU-only variant that asserts the
    pre-load pass with `Device = "vulkan"` refuses. The resolver is pure, so call the startup's pre-load helper directly.
    Expose it as `internal static string? PreLoadTurboQuantRefusal(ServerOptions, ModelConfig)`.
- [ ] **Step 2:** Run with `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~ServerTurboQuantWiring"`.
  Expected: FAIL. `Tq4_Paged_Refuses` gets a `PagedKvCache`; `Tq4_VOnly` gets the default cache.
- [ ] **Step 3:** Implement the two passes and the tq-first branch.
- [ ] **Step 4:** Run again. Expected: PASS. Also run the CPU-only filter on `FullyQualifiedName~Server` and confirm no
  regressions.
- [ ] **Step 5: Mutation.** Each mutant must kill a test; record them:
  - remove the tq-first branch, leaving only the old one after the paged branch;
  - skip the post-load pass;
  - pass `Paged: false` always.
- [ ] **Step 6:** Commit: `feat(server): TurboQuant KV refuses unless it really runs (#1888)`.

### Task 4: Wire run and chat, and update the help text

**Files:**
- Modify:
  - `src/DotLLM.Cli/Commands/RunCommand.cs` (about line 771, where `kvConfig` is built);
  - `src/DotLLM.Cli/Commands/ChatCommand.cs` (about line 632);
  - `src/DotLLM.Cli/Helpers/KvStorageCli.cs` (`CacheTypeKHelp`/`CacheTypeVHelp` list `tq[1..8][q]` and say where it
    runs);
  - `docs/CLI.md` (regenerated).
- Test: `tests/DotLLM.Tests.Unit/Cli/CliTurboQuantRefusalTests.cs`

**Interfaces:**
- Consumes `TurboQuantKvSupport.RefusalReason` with `KvCommand.Run` / `KvCommand.Chat`.
- On refusal: write the reason to stderr (`AnsiConsole.MarkupLine("[red]…[/]")` with the reason escaped) and return exit
  code 1 **before the model loads**. Rule 4 doesn't depend on the host, so the pre-load pass refuses every tq form.
- Extract the check into `internal static string? KvStorageCli.TurboQuantRefusal(KvCommand, KvStorageRequest, ModelConfig?)`,
  so both commands share it and it can be tested without a model.

- [ ] **Step 1: Write the failing tests:**
  - `Run_Tq4_IsRefused_BeforeLoad` and `Chat_Tq4_IsRefused_BeforeLoad`: call `KvStorageCli.TurboQuantRefusal` → contains
    `"dotllm serve --device cpu --no-paged"`.
  - `Run_Q8_IsNotRefused` → null.
- [ ] **Step 2:** Run with `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CliTurboQuantRefusal"`.
  Expected: FAIL.
- [ ] **Step 3:** Implement. Call the check in both commands right after the KV flags are parsed. Update the help constants
  and regenerate `docs/CLI.md` with the generator `CliHelpTests` names.
- [ ] **Step 4:** Run `CliTurboQuantRefusal`, then `CliHelpTests`. Expected: PASS.
- [ ] **Step 5: Mutation.** The mutant removes the call in `RunCommand`. This is a call-site mutant, so the helper test still
  passes. To kill it, add `Run_Tq4_ExitsNonZero`, which invokes the command's settings path. If the command can't be invoked
  without a model, record that the mutant survived and why, and keep the helper tests.
- [ ] **Step 6:** Commit: `feat(cli): run/chat refuse TurboQuant KV until step 2 (#1888)`.

### Task 5: Docs, PR, merge

- [ ] **Step 1:** In `docs/CLI.md` (KV sizing section) and in `docs/KV_CACHE.md`, add a short "TurboQuant availability"
  table with one row per backend and setting. The only ✅ is CPU serve `--no-paged`, plain transformer. Link the epic.
- [ ] **Step 2:** Run the whole CPU-only unit filter:
  `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName!~DotLLM.Tests.Unit.Vulkan&Category!=GPU"`.
  Expected: no new failures against `dev`. Compare by failing test name, not by count, so Theory cases are included.
- [ ] **Step 3:** Open the PR to `dev`. The body includes:
  - `Closes #1888`;
  - the availability table;
  - the mutants per task (applied and died);
  - "no GPU run needed: every GPU path refuses before load".
- [ ] **Step 4:** Once CI is green and the run's head equals the PR head, merge with `gh pr merge PR --merge --admin`.
  Gate each later step with `&&` on `state == MERGED`.
- [ ] **Step 5:** Close the issue by hand. Fast-forward the primary `dev`. Run `git branch -d` and remove the worktree.
- [ ] **Step 6:** Update `.docs/HANDOFF.md`: step 1 is merged, and step 2's plan is next.
