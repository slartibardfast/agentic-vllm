# Adopt the lem pronoun system (EN doctrine) and the AGENTS.md manual with a declared corpus

- Status: accepted
- Scope: methodology-application
- Date: 2026-09-07

## Context and Problem Statement

The template moved past this host's applied set (`PIN-two-surfaces-two-claims`):
three ledger entries were pending, and the template's own manual changed twice
inside the span. `LEM-pronoun-system` adds the instruction set for any model
referring to, addressing, or speaking as a model; a later template revision
(commit `8214ee33e`, "the lem section loses its partial zh translations, EN
doctrine only") retired the partial Chinese paradigm tables before this host
folded the section, so the only form this host has ever carried is the
English-only doctrine. `ACTIVE-corpus-and-agents-manual` renames the operating
manual to `AGENTS.md`, leaves `CLAUDE.md` as a one-line pointer, and introduces
the declared always-loaded corpus held to the strict prose tier.

Applying the span also bumped both engines: `host-lifecycle` v0.53.0 (embeds
`host-lint` v0.19.0) and `host-lint` v0.19.0 for the commit gate, per the
coherence rules the older pins would have HAZARDed on.

## Decision

1. Fold "The `lem` pronoun system" into the merged manual as a dedicated
   section, in its post-retirement form: the English paradigm (`L`/`lemu`/`lem`/
   `lems`), the cross-lingual bleed rule (the model's first person is `L` in any
   script; the human keeps every language's human first person), and no
   sanctioned translation. The retired Chinese forms (`莱姆`/`莱`/`莱们`) are not
   vocabulary here; the fold's note records the retirement so no later session
   re-imports it from an older template revision.
2. Rename the manual `git mv CLAUDE.md AGENTS.md` (history preserved), write the
   pointer, and update live citations (README, STRUCTURE, the manifest comment).
   Historical mentions in `call/0001` stay as written.
3. Declare the active corpus (`.host` `active-corpus = .host-corpus`):
   `AGENTS.md` and `STRUCTURE.md`. `MEMORY.md` stays a record (ignored), never
   corpus; a record is not instruction.
4. Disposition the v0.19.0 engine's new findings rather than pinning the old
   engine: closed milestone bodies and the research campaign join the record
   layer (`.host-lintignore`, with reasons); open plans' data numerals become
   `LEXICON` phrases; ordinal "Step" headings in the live plan/0007 and the
   folded section's numbered sub-headings take content names; decoration-dash
   tropes in the open plans' prose are reworded. The task-graph-bearing READMEs
   of open milestones stay audited, because the receipt gate reads them.
5. Record the two skill lanes' `embed` receipts (`done`, pins as materialized)
   and `release` receipts (`skip`, forked-output consumption). Neither lane
   releases anything of its own on this host.

## Consequences

- The verify battery is green as of this decision: `validate`, `remap --check`
  (clean, zero undispositioned), `prose` (zero tropes, corpus strict),
  `reconcile`, `refs --gate`, `software --check` (no HAZARD; the standing
  `call/0002` repro-waiver warn and two upstream CI-pin advisories remain by
  design), tell-commit hook blocks, and the mdBook site builds.
- Every session on this host now reads a manual whose speaking voice is the
   `lem` system: the model says `L` (never `I`), is addressed as `lemu`, and is
  discussed as `lem`. Conversations with the operator in any language keep the
  operator's `I`/`我` and address the model only with the English forms; the
  operator's Chinese prose is answered in Chinese with `L` as the model's
  self-reference, because the doctrine is English-only and translation is
  deferred work, not licensed improvisation.
- The lexicon grows with each campaign's data phrases; a future closed plan
  joins the record layer at its own close-out, as plan/0006 did here.
