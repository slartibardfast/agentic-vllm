# DOCTRINE.md — Long-range agentic campaigns on weco local (2026-09-11)

The rules for running multi-day agentic optimization campaigns on this
host's weco local fork, synthesized from the validated corpus
(results/*.json, 10/10 items) and this host's own campaign records.
Every rule cites its evidence. The runner (tools/campaign-runner)
implements the mechanical subset; this document is the doctrine the
runner enforces and the operator ratifies.

## D1. The exit contract: queue-drain, never phase-list

A window exits on QUEUE-DRAIN (the named work queue is empty) or
GOAL-RESOLUTION, never "all plan phases delivered." When the queue
near's empty with time remaining, the run ESCALATES — asks for more
work — rather than winding down. The runbook's named queue was always
the finish line; the 72h window that closed at 11h failed this rule
(MEMORY 2026-09-11; cadence item: Claude Code /goal + Stop-hook
pattern, 0din.ai completion-promise).

## D2. Failure routing: gate / supervision / human — asked in order

For any failure class, ask in strict order (taxonomy item):
1. GATE — a deterministic, cheap, independence-checkable check whose
   failure means the work does not count. Gate classes: authoring
   bugs, unverified numeric self-reports (script-verify every count —
   the 49-vs-120 class), silent skips, cross-arm contamination.
2. SUPERVISION — failure observable only from outside the loop;
   answered by restart/re-anchor, NEVER by verdict. Classes: context
   rot, goal drift, deadlock/stall, resource collision, budget
   runaway.
3. HUMAN — judgment under novelty, or the checker itself is the
   adversary (monitor-sabotage class). Hard-always-ask set: external
   auth flows, destructive/irreversible ops, genuine scope change.
Alarm discipline: a gate firing non-actionably more than one run in
ten gets redesigned (EEMUA 191; desensitization is measured).

## D3. The ask budget: rarely, batched, never blocking

Everything reversible proceeds-and-records. Asks batch into a
morning-review document; a blocking ask parks one lane, never the
window; silence never grants authority. Users rubber-stamp 93-97
percent of prompts and scrutiny falls with habituation — asking more
is not safety (cadence item). Defaults taken are logged, not silent.

## D4. Eval boundary: three-valued outcomes, circuit breaker, canary

No mature framework silently continues on null (eval item; Optuna
aborts by default, SMAC3 imputes with a finite sentinel). The
contract: VALUE / CANDIDATE_CRASH (finite floor sentinel, continue) /
INFRA_CRASH (retry once, then no value). Breaker: 3 consecutive
infra-crashes or 30 percent over a 10-step window stops the run with
a distinct exit code. Frozen-baseline canary every N steps (our
preflight's known-good-control-first, elevated to a loop invariant).
Poisoned-metric screens: physical bounds, 5x-band jump requires
oracle re-run, eval script outside the mutation surface.

## D5. Supervision: heartbeat below timeout, progress not liveness

HeartbeatSeconds < TimeoutSeconds, the two causes distinct (worker
death versus task-too-slow; supervision item). The stall signal is
PROGRESS (steps.jsonl appends, file mtimes), never GPU/CPU liveness
(busy GPUs hide stalls — Coreweave). Watchdog: stale-mtime on the
step log, TERM->grace->KILL on the process GROUP (top-level-only
signals orphan GPU children), py-spy/faulthandler dump captured
BEFORE the kill. Per-step budget adapts from the rolling P95.

## D6. Resume: commit-before-append, the ref is the incumbent

Per-step git commit in the lane; the sha written into the step
record; a weco/best ref moved on improved_best. Resume: skip the
torn tail, reset --hard to the ref, continue at N+1
(checkpoint item; the phantom-step window is the WAL inversion —
the log line lands only after the snapshot exists, so a crash wastes
a step rather than fabricating one). NEVER reuse logged metric
values on resume — re-measure (our median-of-N doctrine; MR the
cross-restart bands). Resume preserves A,B,A,B interleave parity
(AutoML item — drift neutralization survives interruption).

## D7. The ledger: lifecycle states, intent fields, replayability

Trial states (RUNNING/COMPLETE/FAIL/SKIPPED — AutoML item) on every
step; the four-valued verdict on every comparison: WIN / LOSS /
UNKNOWN-collect-more (Pinpoint's statistical skip) / ABORT — an
abort is never classified as data (bisection item). Intent fields on
every verdict: question, mechanism, disposition{use,next,deciding
row}, provenance{sha, dirty, diff} (provenance item — state is
captured mechanically, intent must not be the thinnest slot). The
ledger is append-only and replayable (the bisect-log property).
Incumbent calibration tripwire: a known-good control that must pass
or the run aborts as miscalibrated (git's verify_good analog — the
rule that would have caught the founding variance question on day
one).

## D8. Lanes: one consumer per worktree, promote at sync points

Wire the fork's existing consumer_lock into the local loop (lanes
item — it exists, wired only cloud-side). GPU fencing stays
environmental (CUDA_VISIBLE_DEVICES; the gpu-lease script's fd
leases). Champion promotion between lanes happens ONLY at ledger
synchronization points (island-model: frequent migration collapses
diversity); each lane commits its champion, the loser re-bases.

## D9. The harness contract (opencode specifics)

stdin=DEVNULL (the non-TTY stdin hang is reproduced on 1.18.25);
PATH includes ~/.opencode/bin and the runner preflights which +
auth before step one; success requires exit 0 AND an assistant
text event (the exit-0-lying class); OPENCODE_DISABLE_AUTOUPDATE=1;
per-step timeout is the only defense against session-idle hangs
(#17516 is our exact pattern, closed not-planned). For long
campaigns: opencode serve + --attach amortizes the ~7s per-step
boot. Drift defenses: the window is a bounded-task queue (METR's
horizon data); gates live in an agent-unreadable lane (reward
hacking 43x when the agent sees the scorer); supervision config
outside the agent-writable tree.

## D10. The daily record: intent survives the night

Per-day window record (provenance item schema): intent, plan-as-
opened, verdict pointers, deltas, corrections-with-write-back,
script-verified counts, open questions, next_queue with gates,
provenance, and the fixed cold-start resume read-order. The test
stays AGENTS.md's: a new session reads the records and continues
without repeating a mistake. The morning review reads THIS document
plus the ledger; nothing else is required context.
