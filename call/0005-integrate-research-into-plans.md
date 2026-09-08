# Integrate research runs into the host method

- Status: accepted
- Scope: methodology
- Date: 2026-09-08

## Context and Problem Statement

Research output has been landing in a top-level `research/` directory
that the five-rooms model does not sanction ("do not invent new
top-level folders"), is excluded from lint as record material, and is
invisible to the task machinery, which reads `plan/` READMEs. The
consequence observed across 2026-09-06/08: two full research corpora
(headshape, lacunae) were produced and validated, yet plan rooms
absorbed their findings only when the operator directed each absorption
by hand; a MEMORY correction invalidated committed probe findings
without any write-back to the record it corrected; the consolidation
front door itself shipped with a fabricated precision error (49 versus
the verified 120 uncertainty markers) that the methodology's gates had
no way to catch because no gate reads research material at all.

## Decision

1. `research/` is sanctioned by manual amendment as the record room
   for CROSS-MILESTONE research runs. Milestone-scoped research lives
   inside the consuming milestone folder (the plan/0006 and plan/0007
   pattern). The manual's five-rooms table gains the research row.
2. Every research run CLOSES with an integration pass, in order:
   - findings mapped onto open plan tasks as README annotations
     (the task-receipt gate reads READMEs; an unlinked finding is
     invisible to the method);
   - corrections written back into the records they invalidate (a
     MEMORY-only correction is insufficient; the corrected record
     carries the correction or a pointer to it);
   - `research/CONSOLIDATION.md` updated: material map, final-truth
     topic entries, retraction ledger, uncertainty register,
     pending-decision table;
   - counts and numerical claims in the run's own summary verified by
     script before commit (the 49/120 lesson).
3. `tools/check-research-integration.sh` enforces the mechanical
   minimum: every run directory under `research/` is referenced from a
   plan or call record (no orphan research), and the consolidation
   material map covers every run. It runs with the verify battery.
4. Plan READMEs carry a research-inputs note whenever open tasks have
   relevant research (pointer to the consolidation topic, not a
   restatement - one home per fact).

## Consequences

- Research becomes a first-class method citizen with obligations, not
  an output pile: the failure mode "corpus lands, plans unchanged"
  becomes a check failure instead of a habit.
- The consolidation front door is maintained as the planning entry
  point; planning sessions read it before plan READMEs.
- The lintignore on `research/` stays (records are not rewritten for
  style), which makes the integration pass the ONLY path by which
  research reaches the audited rooms.
- If the template later defines its own research doctrine, this
  decision is superseded the MADR way and the runs move wherever the
  template puts them.
