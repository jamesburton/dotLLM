# BenchmarkDotNet silent class-discovery skip — `sealed` is the culprit

**Date**: 2026-05-18
**Branch**: feature/qwen3.6
**Investigator**: dotnet-perf-expert agent (Claude Opus 4.7 1M-ctx)
**File touched**: `benchmarks/DotLLM.Benchmarks/HybridPrefillDecodeBenchmarks.cs`

## Symptom

`HybridPrefillDecodeBenchmarks` compiled cleanly into `DotLLM.Benchmarks.dll`
(verified via `Assembly.LoadFrom` + `GetTypes()` reflection — the class is
present, public, has a parameterless ctor, and one `[Benchmark]`-attributed
method). Despite that, `BenchmarkSwitcher.FromAssembly(...).Run(args, ...)`
with `--list flat` did NOT include it, and `--filter '*HybridPrefillDecode*'`
returned **`0 benchmarks`**. No warning, no error, no diagnostic.

Prior debugging notes in `docs/PERFORMANCE.md` §6.5 hypothesised a
nested-enum `[Params(BenchmarkMode.PureVulkan, BenchmarkMode.Hybrid)]`
conflict. That hypothesis was wrong.

## Root cause

**The benchmark class was declared `public sealed class`.**

BenchmarkDotNet's `BenchmarkConverter.TypeToBenchmarks(Type)` filters out
sealed classes during enumeration without raising any diagnostic. BDN's
runtime toolchains — including the `InProcessEmitToolchain` we activate via
the `inProcessMutator` job in `Program.cs` — emit a runtime-generated
subclass of each benchmark class (one per `[Params]` × `[Arguments]`
combination) via `Reflection.Emit`. A `sealed` type cannot be a base class,
so BDN's discovery pipeline silently drops it. The class is excluded from
both `--list` output and `--filter` resolution as if it never existed.

This is consistent with the prior session's observation that
"`[Params(BenchmarkMode...)]` was nested" — the nested enum was a red
herring; the only `sealed`-decorated benchmark class in the project
happened to also be the only one with a nested enum.

## Minimal fix applied

Single-character delete: removed the `sealed` keyword from
`benchmarks/DotLLM.Benchmarks/HybridPrefillDecodeBenchmarks.cs:51`.
A NOTE comment was added immediately above the class declaration explaining
why future authors should not re-add `sealed`. No other code change.

```diff
- public sealed class HybridPrefillDecodeBenchmarks
+ // NOTE: Cannot be `sealed` — BenchmarkDotNet's BenchmarkConverter silently
+ // filters out sealed classes during discovery because its runtime
+ // toolchains (in-process Emit included) generate a subclass per benchmark
+ // case via Reflection.Emit. With `sealed` the type is unsubclassable and
+ // BDN drops it without diagnostic.
+ public class HybridPrefillDecodeBenchmarks
```

## Verification

After the fix:
- `--list flat | grep Hybrid` →
  `DotLLM.Benchmarks.HybridPrefillDecodeBenchmarks.RunGeneration` (was empty).
- `--filter '*HybridPrefillDecode*' --list flat` → 1 line (was 0).
- `--filter '*HybridPrefillDecode*'` → BDN actually enters the run path:
  enumerates 8 cases (`{16,64,256,1024}` PromptTokens × `{PureVulkan,Hybrid}` Mode),
  builds them under `InProcessEmitToolchain`, executes `GlobalSetup`,
  downloads TinyLlama Q8_0 if absent, runs the workload. The first case
  (PromptTokens=16, Mode=PureVulkan) ran in ~743 ms ± 14 ms over 3 iters.
  (Full run aborted by `timeout 120` after Mode=Hybrid iter 1 — the
  benchmark works; it's just slow per `[SimpleJob(warmupCount: 1, iterationCount: 3)]`
  × 8 cases × ~750ms.)
- Regression check: `--filter '*VulkanFlashAttention*' --list flat` still
  returns its 2 entries.
- Total `--list flat` line count: 64 → 65 (one new entry for the recovered class).

## Were any other classes silently filtered?

No. The task brief mentioned three suspected silently-skipped classes
(`HybridPrefillDecodeBenchmarks`, `TelemetryOverheadBenchmark` /
`LoraDeltaOverheadBenchmark`, `VulkanLoraDeltaDispatchBenchmark`) inferred
from gaps in the interactive listing's benchmark-ID sequence (#15, #19, #21).
Direct inspection of `--list flat` before the fix showed all of
`TelemetryOverheadBenchmark`, `LoraDeltaOverheadBenchmark`,
`LoraMacroBenchmarks`, and `VulkanLoraDeltaDispatchBenchmark` discovered
correctly. The "gaps" in the interactive UI are almost certainly the
interactive listing's own filtering/grouping behaviour rather than
discovery failures — only `HybridPrefillDecodeBenchmarks` was actually
dropped. Grepping the project confirms it was the only `sealed` benchmark
class.

## Is this worth an upstream BDN issue?

**Probably yes** — BDN should at least emit a warning when it finds a
class with `[Benchmark]`-attributed methods that it then refuses to run
because the class is sealed. Silent drops cost two sessions of debugging
on this project alone, and the failure mode is the same one would hit
trying to add `sealed` to a benchmark class for performance / inlining
reasons (a not-unreasonable instinct in `Performance Improvements in .NET`
style code).

### Draft upstream issue

> **Title**: Benchmark classes marked `sealed` are silently dropped from discovery
>
> **Repo**: dotnet/BenchmarkDotNet
>
> **Body**:
>
> BDN's `BenchmarkConverter.TypeToBenchmarks(Type)` drops sealed classes
> during enumeration without any diagnostic. Reproduction (verified against
> BDN 0.14.0, .NET 10.0.103):
>
> ```csharp
> [SimpleJob]
> public sealed class MyBench   // <-- sealed
> {
>     [Benchmark] public int Run() => 42;
> }
> ```
>
> `BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args)` with
> `--list flat` does not list `MyBench`. `--filter '*MyBench*'` returns
> `0 benchmarks`. Removing `sealed` makes the class discoverable.
>
> Root cause is the runtime-generated subclass that BDN emits per benchmark
> case (via Reflection.Emit / InProcessEmitToolchain) requires a
> non-sealed base. Suggested fix: emit a warning at discovery time —
> `"Type 'MyBench' has [Benchmark] methods but is sealed; BDN must be able
>  to subclass benchmark types. Make it non-sealed or virtual."`
>
> Affected versions: 0.14.0 (and likely earlier — the silent filter
> predates the InProcessEmit refactor).

## Anti-future-regression

The new inline NOTE comment plus this `.planning/notes/` entry should
prevent the next author from adding `sealed` for performance "tidying."
The wider project lint could be tightened — a Roslyn analyser that flags
`[Benchmark]`-bearing types with `sealed` would be a one-rule analyser
worth ~20 lines of code — but that's out of scope for this fix.
