# Waive vLLM build reproducibility pending a recorded recipe

- Status: accepted
- Scope: software
- Date: 2026-08-27

## Context and Problem Statement

The `.host-software` recipe records vLLM as the Where-room component. The
methodology requires software initiated under it to carry a reproducible-build
recipe: a pinned toolchain, a build command, and a deployed artifact hash that a
clean rebuild reproduces byte for byte. vLLM is migrated software: it predates
this host, it is developed upstream, and its release artifacts (wheels and
container images) are produced by upstream CI inside heavy GPU build images that
this host does not operate. No recipe recorded here can rebuild those artifacts
byte for byte today.

## Decision

Carry `repro-waiver = call/0002` on the `vllm` stanza. The interim provenance
for the component is the recorded source pin plus upstream's published release
artifacts: what this host audits is the exact source revision, and anything it
deploys would be a build published by upstream for that revision. The stanza
records no deploy line and no artifact hash.

## Consequences

- `host-lifecycle software --verify-build` warns and skips the rebuild
  comparison for this component; `host-lifecycle software --check` still
  requires this citation to resolve.
- The waiver is meant to retire. When a deterministic recipe for a vLLM
  artifact is recorded (toolchain, command, hash) and proven to reproduce, the
  stanza gains its build lines and this decision is superseded the MADR way.
