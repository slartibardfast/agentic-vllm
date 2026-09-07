
## Decode bisect: interleaved paired A/B (2026-09-07, four rounds)

Same-session paired comparison, fresh engine per probe, ctx512_decode
and ctx2048_decode tok/s (medians of 5 in-process reps), stale kernel
(2f1ab8b) vs tuned kernel (HEAD):

| round | stale 512 | tuned 512 | stale 2048 | tuned 2048 |
|---|---|---|---|---|
| r1 | 15.4 | 12.2 | 15.4 | 15.8 |
| r2 | 15.7 | 12.0 | 15.1 | 15.8 |
| r3 | 12.2 | 17.5 | 14.4 | 16.0 |
| r4 | 13.5 | 17.3 | 14.7 | 14.7 |

Verdict: NO kernel-attributable decode regression. ctx2048: the tuned
kernel is faster in all four paired rounds (+0.4 to +1.6 tok/s). ctx512:
both kernels swing 12.0-17.5 across restarts - engine-restart variance,
kernel-version-independent (the stale kernel shows the same swings).
The earlier 0.79-0.90 "deficits" against the seeded single-restart
numbers are within this variance band. Consequently the tp2 decode/mixed
rows are judged VARIANCE-BOUND by the gate until cross-restart sampling
history accumulates; they are excluded from the failure count but
recorded, and the next campaign should add restarts (target: medians
within a 5 percent band) before sub-10 percent deltas are interpreted.
