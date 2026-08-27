# agentic-vllm

An agentic project hosting [vLLM](https://github.com/vllm-project/vllm): the
thought about the engine (plans, decisions, personas, specifications) lives in
this repository, versioned and audited, and the engine itself lives beneath it
as the Where room, materialized locally from the recipe in `.host-software`.

Read [CLAUDE.md](CLAUDE.md) first. It is the operating manual, and it tells any
agent exactly how to work here. [STRUCTURE.md](STRUCTURE.md) is the one-page
map of the rooms:

- `cast/` holds the personas: the people the software serves
- `plan/` holds the milestones, one folder per milestone
- `call/` holds the decisions, in MADR format
- `software/` holds the hosted software (gitignored, materialized locally)
- `tools/` holds the verification tools as pinned submodules

Setup on a fresh machine:

```sh
git submodule update --init
./link-skills.sh
host-lifecycle software --materialize .
host-lifecycle software --install-hooks .
host-lifecycle software --verify-setup .
```

`host-lifecycle` and `host-lint` come from their GitHub releases or a
`cargo build` at their recorded pins; `.host-software` records which. The
methodology is described at https://github.com/connollydavid/host.
