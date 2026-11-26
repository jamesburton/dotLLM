# Agent Instructions

Codex and other repository agents should read `CLAUDE.md` first for the project architecture, coding standards, and workflow rules.

If `.docs/HANDOFF.md` exists, read it at session start before planning or editing. `.docs/` is a private, git-ignored local handoff folder for concise progress and next tasks. Maintain `.docs/HANDOFF.md` when durable progress, benchmark results, or task context changes. Use sibling files such as `.docs/FUTURE_TASKS.md` for longer load-on-demand notes and link them from the handoff.

## Keep `docs/SUPPORTED_MODELS.md` current

When a change adds a model architecture, or adds, optimises, removes or breaks support for one on any device, update the [Device support](docs/SUPPORTED_MODELS.md#device-support) table in the same PR:

- Every `Architecture` enum variant has a row. A new variant needs a new row.
- Re-rate the affected CPU, CUDA and Vulkan cells using the tier rubric above the table. Highly-Optimised requires recorded performance numbers; record them in the docs in the same PR.
- Update the Notes column when a feature lands or is found missing (MTP, DFlash, speculative decoding, attention variant, KV-cache modes, vision, per-device limits).
- Write `?` for anything you have not verified. Remove a suspected-issue footnote once it is fixed or disproved.
- Correct the technical matrix and per-architecture notes if the change makes them wrong, and update the "Last reviewed" line when you re-check the whole table.
