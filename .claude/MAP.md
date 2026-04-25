# PROJECT MAP

## [LEGEND]

### Authority (primary structure)
- **GOSPEL**: Authoritative reference — stable, tested, patterns to follow.
- **VERIFIED**: Working and tested; not the canonical reference.
- **WIP**: Active development — may be incomplete or untested.
- **DEPRECATED**: Last resort. DO NOT CONSULT without asking User first.
- **TEMP**: Ephemeral/Disposable. Plans and inventories; delete when done.
- **SECRET**: Sensitive credentials or secrets. NEVER read into context. Must always be gitignored.

### Audience (inline tags, applied within any tier)
- `[dev-eyes]` — Internal use; not for end users.
- `[user-eyes]` — User-facing; no internal dev-speak or TODOs.

### Signal Score (inline tag, applied within any tier except SECRET)
- `[s:N]` — Relevance score. Seeded at init from tier baseline; updated by refresh based on session signals.
- Tier baselines: GOSPEL=10, VERIFIED=5, WIP=0, TEMP=-5, DEPRECATED=-10.

---

## [GOSPEL]

- ~/claude/reference-implementations/autoEQ/autoeq/  # the reference implementation - gold-standard
- ~/claude/biquad-fit/src/                # **Prior port.** Faithful pipeline, L-BFGS optimizer (diverges from AutoEQ's SLSQP).

---

## [VERIFIED]

- CLAUDE.md                               # Project instructions and orientation
- docs/ARCHITECTURE.md [dev-eyes] [s:10] # Algorithm specification, core reference
- docs/PLAN.md [dev-eyes] [s:10]         # 9-phase implementation roadmap
- Cargo.toml                              # Package manifest
- Cargo.lock                              # Dependency lock file
- docs/setup.txt [dev-eyes]              # Setup and development guide
- benches/optimize_benchmark.rs          # Benchmark suite
- tests/fixtures/fr/*.json               # IEM frequency response measurements (5 IEMs)
- tests/fixtures/targets/*.json          # Target EQ curves (6 targets)
- tests/fixtures/golden/*.json           # Golden reference outputs vs AutoEQ (90 combinations)

---

## [WIP]

- src/lib.rs [dev-eyes]                  # Main library implementation (active development)

---

## [TEMP]

- CLAUDE.md- [dev-eyes]                  # Backup of CLAUDE.md

---

## [SECRET]

- .gitignore                              # (verified gitignored: /target, Cargo.lock in repo by design)
