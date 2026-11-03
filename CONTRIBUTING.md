# Contributing to dotLLM

Thanks for your interest in contributing! dotLLM is a ground-up LLM inference engine for .NET and we welcome issues, discussions, and pull requests.

## Branches in this fork (`jamesburton/dotLLM`)

This repository is a fork of [`kkokosa/dotLLM`](https://github.com/kkokosa/dotLLM). Its branches have fixed roles:

| Branch | Role | What you do with it |
|--------|------|---------------------|
| `main` | A mirror of upstream `kkokosa/dotLLM` `main`, kept at exactly 0 commits ahead and 0 behind. | **Never open a PR against it and never push to it.** A ruleset blocks everything except the owner's upstream sync. |
| `dev` | The integration branch. Every push to `dev` publishes a **prerelease** (`0.3.0-dev.N` GitHub release assets and the `dotllm` NuGet prerelease). | **Branch from `dev` and open your PR against `dev`.** |

- Only the repository owner pushes to or merges into `dev`. Collaborators open a PR to `dev`; it is reviewed and merged by the owner. This matters because a merge to `dev` publishes immediately.
- Branch names stay `issue/{number}-{short-kebab-description}`, but they start from `dev`.
- `Closes #N` in a PR description does **not** auto-close the issue when the PR merges into `dev` (GitHub only does that for the default branch). The owner closes the issue by hand after merging, so please still reference it.
- Work meant for upstream is extracted as a separate small PR against `kkokosa/dotLLM` `main` by the owner; do not do that from this fork.

Where the workflow below says `main`, read `dev` when contributing to this fork.

## Before you start

- Read [CLAUDE.md](CLAUDE.md) for the project identity, design philosophy, and code-style rules.
- Browse [docs/](docs/) — each module has a dedicated design doc. Read the doc relevant to the area you plan to touch before writing code.
- Check [docs/ROADMAP.md](docs/ROADMAP.md) to see what phase is in flight and what's planned next.

## Development workflow

We follow a strict issue-driven workflow. **Every change starts with a GitHub issue.**

1. **Open an issue** describing the bug, feature, or improvement. Include acceptance criteria and link to the relevant roadmap step or design doc if applicable.
2. **Branch from `dev`** (`main` upstream; see [Branches in this fork](#branches-in-this-fork-jamesburtondotllm)) using the pattern `issue/{number}-{short-kebab-description}` (e.g., `issue/42-gguf-q6_k-support`).
3. **Keep the scope tight** — one issue, one branch, one PR. If scope grows, split into a new issue.
4. **Reference the issue in commit messages** with `(#42)` at the end of the subject line.
5. **Open a PR against `dev`** (against `main` only in upstream `kkokosa/dotLLM`) with `Closes #42` in the description. Auto-generated release notes pull from PR titles, so write a descriptive title.
6. **Update docs as you go.** When a PR completes a roadmap step, update `docs/ROADMAP.md` (add a checkmark) and the Roadmap table in `README.md`.

## Building and testing

```bash
# Build the whole solution
dotnet build

# Run unit tests (GPU tests excluded by default)
dotnet test tests/DotLLM.Tests.Unit/ -c Release --filter "Category!=GPU"

# Run integration tests (downloads small test models on first run)
dotnet test tests/DotLLM.Tests.Integration/ -c Release

# Run the CLI without installing
dotnet run --project src/DotLLM.Cli -c Release -- run QuantFactory/SmolLM-135M-GGUF -p "hello" -n 32
```

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for performance work.

## Code style

- File-scoped namespaces, nullable enabled, latest C# language version.
- `readonly record struct` for small value types.
- `Span<T>` / `ReadOnlySpan<T>` in method signatures.
- Zero managed allocations on the inference hot path — use `NativeMemory.AlignedAlloc` (64-byte aligned for AVX-512, 32-byte for AVX2) and `ArrayPool<T>.Shared` for scratch buffers.
- SIMD: prefer `System.Numerics.Tensors.TensorPrimitives` for standard ops and `System.Runtime.Intrinsics` (`Vector128<T>`/`Vector256<T>`) for hand-tuned hot loops. Always provide a scalar fallback.
- XML doc comments on all public APIs in `src/`.
- `TreatWarningsAsErrors=true` — fix warnings, don't suppress them unless justified.

## Performance changes

- Benchmark before and after with [BenchmarkDotNet](benchmarks/DotLLM.Benchmarks/). Include numbers in the PR description.
- For SIMD kernels: verify numerical correctness against a scalar reference first, then optimize.

## Reporting bugs

Open an issue with:

- dotLLM version (or commit SHA)
- OS, CPU architecture, and .NET runtime version
- GPU model and CUDA version (if applicable)
- Model name / GGUF file being used
- A minimal repro (prompt + command line is usually enough)

## Security issues

Please **do not** open public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).

## Licensing

dotLLM is licensed under [GPL-3.0-only](LICENSE). By contributing, you agree that your contributions will be licensed under the same terms.
