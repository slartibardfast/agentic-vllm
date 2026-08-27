# MEMORY.md

## Session Log

- Project purpose: governance, plans, decisions, and verification for the vLLM inference engine, hosted as the Where room

- 2026-08-27: Adopted the host methodology at template revision 44b4c84 (case a, mode Shallow; no prior files, so the rename map is empty). The spine, rooms, recipe, tools, gate, receipts, and the upgrade baseline are in place; the reasoning is in call/0001.
- 2026-08-27: The adopted revision is the revision the host install manifest publishes as the verified set. Template HEAD is one commit later and adds the lem pronoun section, which fails the prose gate on the template's own text (dashes and arrows in its tables); until upstream lands it green, keep the adopted pin and let the upgrade ledger bring the section in.
- 2026-08-27: The gate binaries are the published release binaries placed at the recorded artifact paths: host-lint 0.18.1 and host-lifecycle 0.50.0, linux-amd64. Their sha256 match the artifact hashes in .host-software, so the installed hooks run verified against the canonical hash.
- 2026-08-27: host-prove v0.3.0 (the musl release binary) is installed at ~/.local/bin/host-prove, with ~/.local/bin added to PATH in ~/.bashrc. Reason: the gate components' obligations declare kani rungs, and software --check hazards while the re-deriver does not run.
- 2026-08-27: software --check flagged eight scripts at the vllm pin that are invoked as executables but recorded 100644 (an upstream defect at that revision). The printed remedy was applied index-only in software/vllm/main (update-index --chmod=+x), so that worktree is deliberately dirty against its pin. Never commit that change in the worktree: the pin is the audit anchor, the fix belongs upstream.
- 2026-08-27: Upgrade ledger: the stamp was migrated to baseline e4ed590 and all six pending entries were applied and recorded. Two were no-ops (the recipe already used the current keys); the reference discipline and its verify clause were already in the copied spine and the v0.50.0 binary.
