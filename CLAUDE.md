# CLAUDE.md

Rust reimplementation of AutoEQ's parametric EQ optimizer. See `.claude/MAP.md` for project structure.

**Key commands:** `cargo test`, `cargo bench`, `cargo build`

**Golden test matrix:** 90 combinations (5 IEMs × 6 targets × 3 constraint sets). RMSE ≤ 0.5 dB vs. AutoEQ.

**Start here:** `docs/ARCHITECTURE.md` (algorithm spec) and `docs/PLAN.md` (9-phase roadmap).


## BEHAVIOR:

```
claude tokens are limited.  our activity shall be optimized to limit token use.  

Is this in a skill or memory?   → Trust it. Skip the file read.
Is this speculative?            → Kill the tool call.
Can calls run in parallel?      → Parallelize them.
Output > 20 lines you won't use → Route to subagent.
About to restate what user said → Delete it.

Grep before Read. Never read a whole file to find one thing.
Do not re-read files already in context this session.
```

