# Windows System Tray — dotLLM

The tray (`src/DotLLM.Tray`) makes dotLLM a background service you forget about rather than a
terminal you keep open. It is **a client**: every action is an HTTP call to the management API
from [#454](https://github.com/jamesburton/dotLLM/issues/454). It references neither
`DotLLM.Engine` nor `DotLLM.Server`, so an in-process privileged path is not merely discouraged —
it does not compile.

Windows only for v1. macOS/Linux trays, a chat UI in the tray (the Web UI already exists) and
multi-server management are deliberately out of scope.

## Layout

| Project | What it is |
|---|---|
| `src/DotLLM.Tray.Core` | `net10.0-windows` class library. Every decision the tray makes: the API client, the lifecycle state machine, the process runner, the job object, autostart, the update checker, the settings store. No UI. |
| `src/DotLLM.Tray` | `net10.0-windows` WinForms `WinExe`. Tray icon, menu, three dialogs. Thin. |
| `tests/DotLLM.Tray.Tests` | `net10.0-windows` xUnit. 105 tests over `Tray.Core`. |

The split exists so the behaviour is testable at all — see [Testing](#testing) for exactly which
parts are covered and which are manual-only.

All three set `EnableWindowsTargeting=true`, because `ci.yml` builds the whole solution on
`ubuntu-latest`. The tray suite is *built* there and *run* on Windows; `ci.yml` executes only
`tests/DotLLM.Tests.Unit`.

## Settled decisions

### 1. Packaging: self-contained, single-file, **not** trimmed, **not** AOT

`dotnet publish src/DotLLM.Tray -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true`

Why:

- **Self-contained, not framework-dependent.** The tray's entire premise is "install it and forget
  it". Framework-dependent means a user whose .NET 10 runtime is absent, or is removed by a later
  Windows servicing update, gets a tray that silently stops appearing at login — the failure mode
  that is hardest to diagnose precisely because the app is invisible by design.
- **Single file, because it makes the update story one file.** "Download the new archive, replace
  the exe" is something a user can do and undo. A loose-file layout makes a partial replacement
  possible, and a partially-updated app is worse than an out-of-date one.
- **Not Native AOT.** WinForms does not support AOT: designer serialization and control activation
  are reflection-driven, and the trim analyzer produces findings that cannot be fixed. This project
  therefore disables the trim analyzer it inherits from `src/Directory.Build.props`.
  `docs/AOT.md`'s argument against AOT — 10–40% slower CPU inference from losing Dynamic PGO —
  does not apply here (the tray does no inference), but the WinForms constraint decides it anyway.
- **It matches what the repo already ships.** `.github/workflows/release.yml` already publishes
  `DotLLM.Cli` as a self-contained single file per RID. The tray follows the same shape, so a
  release archive contains `dotllm.exe` and `dotllm-tray.exe` side by side — which is exactly the
  layout `DotLlmExecutableLocator`'s sibling probe expects.

The tray does **not** bundle the engine. It locates `dotllm.exe` as: a configured path → a sibling
of `Environment.ProcessPath` → `PATH`. (`Environment.ProcessPath`, not `Assembly.Location`, which
is empty in a single-file publish.)

### 2. Hosting: attach if one is running, otherwise spawn as a child — ownership is explicit

`ServerSupervisor.StartAsync` probes `GET /health` **before** launching anything.

- Something answers → **attach** (`RunningAttached`). No second process is spawned; spawning anyway
  would produce a process that fails to bind the port and dies, leaving a confusing corpse and a
  tray that believes it owns a server it does not. This is the same "refuse to start over a live
  server" check #454's own harness uses.
- Nothing answers → **spawn** (`RunningOwned`), always with `--allow-model-admin` and
  `--no-browser`.

Ownership is one-way and not negotiable: **the tray stops only what it started.** Stop and Restart
are greyed out for an attached server, with a tooltip saying why. The tray does not look up the
listening PID and kill it — that process may be a developer's `dotnet run`, and the tray has no
mandate over it.

**Orphan prevention is a kernel invariant, not a code path.** At startup the tray puts *itself*
into a job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Children started with `Process` inherit
job membership from `CreateProcess`, so one assignment covers every server the tray ever launches
and leaves no window in which a child exists outside the job — closing that window per-child would
need `CREATE_SUSPENDED`, which `ProcessStartInfo` cannot express. When the tray goes away by any
means, including `taskkill /F`, the kernel closes the last handle and reaps the job. Verified: see
[Testing](#testing).

If the OS refuses to create the job, the tray still works, still reaps its child on every ordinary
exit path, and **says so in the menu** ("⚠ Orphan protection unavailable on this machine") rather
than quietly losing the guarantee.

The converse — server dies, tray notices — is `IServerProcessHandle.Exited`, which moves the tray to
`Failed` instead of leaving a green icon over a dead backend.

### 3. Signing / SmartScreen: **unresolved — so the tray does not self-update**

There is no code-signing certificate in this project. `.github/workflows/release.yml` has no signing
step and no secret for one; every published archive is unsigned.

The consequence is concrete, not theoretical. An unsigned executable downloaded from the internet
carries the Mark of the Web, so SmartScreen shows "Windows protected your PC — unrecognized app"
and Defender may quarantine it. A tray that downloaded and swapped its own executable would
therefore (a) hand the user a binary Windows tells them not to run, and (b) risk a quarantine
*partway through the swap*, leaving a broken install with no tray left to repair it.

So v1 **checks, shows the changelog, and opens the release page**. It never downloads and never
applies. This is stated plainly rather than shipped as a staged-apply path that would be blocked in
practice.

What would change it, in order:

1. An OV or EV Authenticode certificate plus a signing step in `release.yml` (EV, or an OV cert with
   accumulated SmartScreen reputation, is what actually clears the prompt).
2. Publishing SHA-256 digests with the release assets, and verifying them before any swap.
3. Only then an updater that stages beside the running exe and renames on next start.

The update check is also **opt-in** (`check_for_updates`, default `false`): the tray makes no
request to GitHub until the user turns it on.

## The three tiers of setting

Conflating these would be the most misleading thing the UI could do, so the Settings dialog
captions each group with when it applies.

| Tier | Where it lives | When it applies |
|---|---|---|
| **Live** — keep-alive default, max resident models, residency byte budget, idle sweep interval | server, via `PUT /v1/settings` | immediately; the response reports `applied`, `restart_required` and anything `evicted` |
| **Launch** — host, port, `dotllm.exe` path, startup model, initial device, GPU layers, KV cache types | `%APPDATA%\dotLLM\tray.json`, rendered into the child's command line | next time the tray starts a server |
| **Per load** — device, GPU layers, KV cache types, keep-alive override | `POST /v1/models/load` | that load |

`--allow-model-admin` is always emitted for a tray-owned server and is not configurable: without it
every write route answers 403 and the tray's own controls are dead buttons against a server it
started itself. An *attached* server's flags are not the tray's to choose, so when one was started
without the flag the live settings group goes read-only and shows the server's own explanation.

The device picker is populated from `GET /v1/devices` gated on **`servable`**, not `available`.
Vulkan reports available-but-not-servable (the server's load path dispatches to CPU or CUDA only),
and offering it would produce a load that silently lands on the CPU.

## Autostart

Per-user `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`, never `HKLM` — an all-users entry
needs elevation, would start the tray for accounts that never asked, and could not be removed by the
user who enabled it. The `Run` key is also what Task Manager's Startup tab lists and can disable, so
the user has a way out that does not involve the tray.

- **Off until explicitly enabled.** A fresh install has *no entry*, not a disabled one.
- **The registry is the source of truth.** `IsEnabled()` reads the key on every menu open, never a
  cached flag — a removal from Task Manager must not make the menu lie.
- **The path is quoted unconditionally.** The default install location contains a space, and an
  unquoted path is both a startup failure and the classic unquoted-path hijack.
- **Disable deletes; it does not blank.** An empty `Run` value still lists in the Startup tab, so
  blanking would fail "cleanly removable".
- The manifest is `asInvoker`. The `Run` key does not launch elevated entries, and the tray needs
  nothing beyond `HKCU` and `%APPDATA%`.

## Known gap — needs a change in #454

**There is no shutdown endpoint.** Two consequences:

1. **An attached server cannot be stopped or restarted from the tray at all.** Those commands are
   greyed out with the reason.
2. **Stopping an owned child is a hard `Process.Kill`.** In-flight generations are cut. In-flight
   *downloads* are not lost — they resume from their `.incomplete` file.

The fix belongs in #454, not here: a gated `POST /v1/admin/shutdown` that drains behind the request
gate using the same stop-scheduler-then-dispose sequence `ServerState.UnloadAsync` already has.
The tray would call it, wait for `/health` to stop answering, and only then fall back to a kill.
**The tray deliberately does not work around this** by finding the listening PID — that would be
exactly the reach-around its client-only contract forbids.

## Testing

### Automated (105 tests, `dotnet test tests/DotLLM.Tray.Tests`)

| Suite | Covers |
|---|---|
| `TrayContractTests` | every mirrored DTO round-tripped through the **real** source-generated `ServerJsonContext` (hence one test-only `InternalsVisibleTo` on `DotLLM.Server`) |
| `DotLlmApiClientTests` | routes, verbs, bodies, SSE frame parsing, the admin gate's 403 becoming an actionable message |
| `ServerSupervisorTests` | attach-vs-spawn, ownership, start timeout, child death, restart, event coalescing |
| `AutostartManagerTests` | the rules above, plus one real-registry round-trip under a throwaway value name |
| `ServerLaunchTests` | the exact `dotllm serve` command line, invariant number formatting, executable discovery |
| `UpdateCheckerTests` | SemVer/MinVer ordering, with the naive comparators as an explicit negative control |
| `TraySettingsStoreTests` | opt-in defaults, corrupt and partial files |

Per [#417](https://github.com/jamesburton/dotLLM/issues/417), discrimination is demonstrated. Each
suite's remarks record which mutants were applied and which died — **including one that survived**
(deleting the explicit `handle.Kill()` on the start-timeout path is redundant with the dispose that
follows; removing both reap paths is what turns the test red).

### Verified against a real server

A throwaway harness drove the real `DotLlmApiClient`, `ProcessServerProcessRunner` and
`ServerSupervisor` against a real `dotllm serve` (no model, so no GPU) on ports 18097–18099:

- **24/24** — spawn, attach-instead-of-spawn, refuse-to-stop-an-attached-server, every ungated GET,
  live `PUT /v1/settings` (including that a partial update leaves other fields alone),
  enable/disable round-trip visible in `disabled_models`, `not_resident` unload, 404 on an unknown
  pull job, stop, and the PID actually gone.
- **8/8** — a server started *without* `--allow-model-admin`: `GET /v1/settings` ungated and
  reporting `model_admin_api_enabled: false`, and all five write routes refused with the flag named
  in the message.
- **Orphan guarantee** — parent process holding the job object spawned a server (PID 34904), was
  hard-killed with `taskkill /F`, and the server was gone within 3 s with the port unreachable.
- Confirmed on this machine: `cpu(available, servable)`, `cuda(unavailable)`,
  `vulkan(available, NOT servable)` — the exact shape the device picker must handle.

### Manual only — the UI

Nothing below is automated. WinForms message pumps, shell notification icons and real user gestures
are not reachable from a unit test.

1. **First run.** Launch `dotllm-tray.exe` with no `tray.json` present. Grey icon, tooltip
   "Stopped". Right-click: Start enabled; Stop, Restart and Models disabled. Confirm
   `HKCU\...\Run` has **no** dotLLM value and no update check has occurred.
2. **Start.** Click Start. Icon goes amber then green; tooltip reads "Running (started by the
   tray) — no model loaded". Confirm exactly one new `dotllm` process, and a log file under
   `%LOCALAPPDATA%\dotLLM\tray\`.
3. **Attach.** Exit the tray (leaving the server running is not possible — see 8 — so instead start
   a server manually with `dotllm serve --allow-model-admin --no-browser`, then launch the tray).
   Icon should be **blue**, tooltip "Running (started elsewhere)", Stop and Restart **greyed** with
   the explanatory tooltip. Confirm no second `dotllm` process appeared.
4. **Web UI / copy URL.** Double-click the icon opens the browser at the base URL. "Copy base URL"
   puts it on the clipboard.
5. **Models.** Open Models. Resident list shows the loaded model with size and a counting-down
   "Auto-unload in"; a model with keep-alive −1 shows "never", not "0:00". Load an available model,
   watch it appear in the resident list. Disable it and confirm the status line says it is still
   loaded and that the setting resets on restart. Unload it. Unload All.
6. **Pull.** Enter a small repo/file and Start download. Progress advances. **Close the Models
   window mid-download, reopen it, and confirm the job is still listed and still progressing** —
   disconnecting must not cancel. Then Cancel and confirm the `.incomplete` file survives.
7. **Settings.** Change keep-alive and press Apply; the status line names the applied field. Change
   the port, press OK, and confirm the tray re-points and stops the old owned child. Against a
   server started without `--allow-model-admin`, confirm the live group is read-only with the
   server's message. Confirm Vulkan does not appear in the device list on a machine that has it.
8. **Exit.** Exit from the menu. Icon disappears; the owned `dotllm` process is gone.
9. **Orphan, by hand.** Start a server from the tray, note its PID, then `taskkill /F /IM
   dotllm-tray.exe`. The server PID must be gone within a few seconds. (Automated equivalent
   verified above, but repeat it on any machine where the menu shows the orphan-protection warning.)
10. **Server dies underneath.** Start from the tray, then `taskkill /F` the `dotllm` process. The
    icon must go red and the menu must say the server exited unexpectedly.
11. **Autostart.** Tick "Start dotLLM at login". Confirm the `Run` value appears, quoted. Sign out
    and back in; the tray reappears. Untick it; confirm the value is **gone**, not empty, and that
    Task Manager's Startup tab no longer lists it.
12. **Update check.** Enable it in Settings, then "Check for updates". With a newer release
    published you get the changelog dialog with **no Install button** and "Open release page";
    otherwise a plain "up to date". Confirm that with the option off, no request to
    `api.github.com` is made.
13. **Second instance.** Launch the tray twice. The second exits silently; one icon only.
14. **DPI.** Move the tray between a 100% and a 150% display and confirm the icon stays crisp.
