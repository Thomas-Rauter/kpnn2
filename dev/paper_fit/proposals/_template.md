# Proposal: <title>

<!-- Written by dev/paper_fit/propose_prompt.md. Delete this
comment and every placeholder line in italics. The maintainer
sets Status and Decision; the session leaves them as given. -->

| Field | Value |
|-------|-------|
| Status | `proposed` |
| Decision | *empty until the maintainer decides: date and one line* |
| Date | YYYY-MM-DD |
| Based on | *`runs/<dir>/report.md`* |
| kpnn2 commit | *short SHA; add `dirty` if the tree was not clean* |
| Kind | *add, change, or remove* |
| Locks touched | *none, or the `CONTEXT.md` section* |
| Agent | *agent and model* |

## Problem

*One sentence that names the needs. Then at most one paragraph.*

## Evidence

| Need | Paper | Run mapping | Glue today | Card |
|------|-------|-------------|------------|------|
| *N18* | *`<id>`* | *glue or missing* | *what the sketch does, lines* | *section* |

*One line on papers counted once because they share code, if
any.*

## Proposal

- Public surface: *signatures, defaults, and the first line of
  each docstring.*
- Behavior: *rules, edge cases, returned objects with dims and
  names, and what it does not do.*
- Errors: *each caller mistake and its `Kpnn2Error` message.*
- Where it lives: *`src/` modules, new or existing, and the
  internal code it reuses.*

## Why in kpnn2

*The responsibility in **Package philosophy** this extends, and
why user code is not enough.*

## Before and after

### *`<id>`*

```python
# before
...  # GLUE
# after
...  # PROPOSED
```

## Predicted effect on the next run

| Paper | Need | Now | After | `# GLUE` lines |
|-------|------|-----|-------|----------------|

*Verdicts that would move, and orphans that would gain users.*

## Cost and risks

*Public names, keyword arguments, and methods added or removed;
internal code; migration for users; numbers that would change.*

## Slices

1. **Title.** *What changes; files; tests; lockstep yes or no;
   changelog yes or no; numbers changed yes or no.*

## Open questions

*Choices the maintainer must settle before slice 1 is planned,
each with a recommendation. "None" is fine.*

## Considered and not chosen

| Candidate | Supporting papers | Why not now |
|-----------|-------------------|-------------|

## Earlier proposals

*For each `done` proposal: whether its predicted effect held in
this run. "None" if there are none.*

## Caveats

*Freshness-gate mismatches, cards that look wrong, and anything
else that limits this proposal.*
