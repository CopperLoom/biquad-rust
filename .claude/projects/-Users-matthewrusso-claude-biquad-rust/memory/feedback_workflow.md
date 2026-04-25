---
name: Git workflow — branch first
description: Always cut a feature branch before starting any implementation work
type: feedback
---

Always `git checkout -b <branch-name>` **before writing any code**. Never work directly on main.

**Why:** User called this out explicitly as basics. Previous phases all used feature branches (phase-1-types, phase-3-smooth-compensate). Creating the branch retroactively after the work is done defeats the purpose.

**How to apply:** First action when starting any phase or feature is to cut the branch. No exceptions.
