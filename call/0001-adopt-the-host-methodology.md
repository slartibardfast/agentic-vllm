# Adopt the host methodology

- Status: accepted
- Scope: project
- Date: 2026-08-27

## Context and Problem Statement

This repository began as an empty folder named `agentic-vllm`, intended to hold
the thought work for vLLM: plans, decisions, personas, and verification. Without
a governing manual and mechanical checks, an agent working here repeats known
mistakes: ordinal file names, slopped prose, unrecorded decisions, and
methodology re-litigated per session.

## Decision

Adopt the agentic-host methodology from https://github.com/connollydavid/host.
The template's techniques are copied in at revision `44b4c84` (recorded in full
in the `.host` stamp), the revision the host install manifest publishes as the
verified set. The steps follow the host README procedure:

- Migration case a (no prior CLAUDE.md): the spine (`CLAUDE.md`,
  `STRUCTURE.md`) is copied unchanged at the chosen revision; this project's
  own conventions are recorded under the project-specifics heading in
  `CLAUDE.md`.
- Mode Shallow: live files only, history untouched. The repository had no
  history and no files, so the rename map is empty and no `.host-remap`
  dictionary applies.
- The software under governance, vLLM, is embedded as the Where room from the
  `.host-software` recipe; its build waiver is recorded separately
  (call/0002).
- The verification tools are wired by role: all four as referenced submodules
  under `tools/` (reference, never vendor), and the two the host itself
  executes, host-lint and host-lifecycle, additionally as Where-room
  components, so the commit gate runs from a binary whose source pin and
  artifact hash are recorded in the recipe and audited by `software --check`.

## Consequences

- Good: commits are gated by the host-lint hooks, authored docs are held to
  zero slop by the prose lane, lifecycle phases carry tool-written receipts,
  and upgrades proceed from the template's ledger at the recorded baseline.
- Acknowledged: `host-lint --all` warns on the lem section's dotted sub-headings
  inside `CLAUDE.md`, and its prose lane warns on the dashes and arrows in that
  section's tables. The spine is inherited verbatim by copy-at-version and
  re-copied at each upgrade, so those warnings are triaged and acknowledged
  rather than reworded. The gates themselves (`host-lifecycle prose .` and the
  commit hooks) are clean on this copy.
