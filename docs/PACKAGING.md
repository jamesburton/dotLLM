# Releases, the dev channel, and NuGet / `dnx`

This fork publishes its own builds. Upstream (`kkokosa/dotLLM`) has been dark since 2026-07-30 and carries none of the fork's
features, so nothing here reads or writes upstream.

## Channels

| Channel | Trigger | Version | Tag | Contents |
|---|---|---|---|---|
| **dev** | every push to `dev` that passes CI (`ci.yml`: `dev-version` -> `dev-release`) | `0.3.0-dev.<commit count>` | `v0.3.0-dev.<n>` | win-x64 / linux-x64 / osx-arm64 archives (the Windows one includes `dotllm-tray.exe`), `SHA256SUMS`, nupkgs |
| **tagged** | a pushed tag `v*` (not `v*-dev.*`) | the tag | the tag | the same, plus experimental Native AOT archives |

Both go through one pipeline, `release.yml` (it is also a reusable workflow). Dev releases are always GitHub *prereleases*, so
they are never "latest", and only the newest 15 are kept (`dev-prune` deletes older releases and their tags).

**Version ordering.** `<commit count>` only grows on `dev`, numeric prerelease identifiers compare numerically
(`dev.9999 < dev.12000`), and `0.3.0-dev.N` outranks upstream's `0.2.0-alpha.*` tags and precedes the stable `0.3.0`. When a stable
fork release is cut (`git tag v0.3.0`, push), bump `DEV_BASE_VERSION` in `ci.yml` to the next version so dev builds keep sorting
after it. The version is pinned into every artifact with `-p:MinVerVersionOverride`, so the tray's "current version", the nupkg and the
tag always agree.

The tray reads `https://api.github.com/repos/jamesburton/dotLLM/releases` by default ([TRAY.md](TRAY.md)).

## Verifying a download

These builds are **not Authenticode-signed**, so Windows SmartScreen will show "unrecognized app". What is published instead:

- `SHA256SUMS` on every release.
- A **build-provenance attestation** (GitHub Artifact Attestations - free for public repositories) tying each asset to this repository,
  the workflow, and the commit:

  ```
  gh attestation verify dotllm-0.3.0-dev.2250-win-x64.zip --repo jamesburton/dotLLM
  ```

This proves *where a file was built*, not *who vouches for it*; it adds no signing identity and keeps no secret in CI. Authenticode
signing is a separate, later decision (see below).

## NuGet and `dnx` - decision deferred

`dnx <package>` (.NET 10) runs a dotnet-tool package without installing it, so the CLI is already shaped for it: `DotLLM.Cli` is a
`PackAsTool` project with command `dotllm`, and its package carries the Vulkan `spv/` shaders (verified: `dnx` ran a locally packed
build). **The blocker is naming**: upstream owns `DotLLM.Cli`, `DotLLM.Engine`, ... on nuget.org (versions `0.1.0-preview.1-3`), so this
fork cannot publish under those ids.

Until a name is chosen, nothing is pushed. The package id is one repository variable, so choosing is a settings change, not a code change:

- repo variable `DOTLLM_TOOL_PACKAGE_ID` = the id to publish the tool under (unset = skip the push),
- repo secret `NUGET_API_KEY` = a key scoped to that id.

Only the tool package is pushed; the `DotLLM.*` libraries are still packed (so a build proves they pack) but not published.

Availability checked on nuget.org on 2026-10-05:

| Option | `dnx` invocation | Notes |
|---|---|---|
| **`dotllm`** | `dnx dotllm` | Free. Matches the command name; the shortest. Claims the bare product name, which a later upstream revival would also want. |
| `DotLLM.Tool` | `dnx DotLLM.Tool` | Free. Stays in the `DotLLM.` family but reads as a different package from upstream's `DotLLM.Cli`. |
| `dotllm-cli` | `dnx dotllm-cli` | Free. Unambiguous, a little redundant. |
| `<Owner>.DotLLM` (e.g. `JamesBurton.DotLLM`) | `dnx JamesBurton.DotLLM` | Free. Clearly a fork; unlikely ever to collide. Owner-prefix reservation is also available on nuget.org. |
| keep `DotLLM.Cli` | - | Not possible without upstream transferring ownership. |

Whatever the id, the installed command stays `dotllm`. Two things are not yet automated and would need the account: reserving a
package-id prefix, and switching from a long-lived `NUGET_API_KEY` to nuget.org *Trusted Publishing* (OIDC, no stored secret) - the
latter is the better end state once the id exists.

## What is deliberately not here

- **Authenticode signing.** A certificate (or a signing service) ties an identity to a person and its key would sit in the Actions
  secrets of a single-maintainer fork. Hash + attestation is the honest level until that is decided. The tray therefore still checks and
  links, and does not self-apply ([TRAY.md](TRAY.md) section 3).
- **scoop / winget manifests.** Straightforward on top of the checksums above (a manifest pins the hash); not yet added.
