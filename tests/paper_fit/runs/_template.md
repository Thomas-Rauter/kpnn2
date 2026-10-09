# Paper-fit run YYYY-MM-DD

<!-- Written by tests/paper_fit/assess_prompt.md. Delete this
comment and every placeholder line in italics. -->

| Field | Value |
|-------|-------|
| kpnn2 version | |
| kpnn2 commit | *short SHA; add `dirty` if the tree was not clean* |
| Scope | *all included papers, or the ids assessed* |
| Cards | *count, then ids* |
| Previous run | *`runs/<dir>/report.md`, or "none"* |
| Agent | *agent and model* |

## Summary

*At most five sentences: verdict counts, the strongest gaps, the
orphans, and the biggest change since the previous run.*

## Verdicts

| Paper | Verdict | Sketch | kpnn2 lines | Glue lines | Needs left to glue or missing |
|-------|---------|--------|-------------|------------|-------------------------------|
| *`<id>`* | *R, P, or N* | *pass or fail* | | | |

## Per paper

### *`<id>`*

- Sketch: *`sketches/<id>.py`, pass or fail.*
- Needs: *one entry per need: ID and `covered` (kpnn2 names),
  `glue` (what remains), `missing`, or `out of scope`
  (`CONTEXT.md` section).*
- Author glue: *for each row of the card's Glue table: removed,
  shrunk, or left.*
- Notes: *anything the verdict does not show.*

## Gaps

Needs that are `missing` or `glue` in two or more papers.

| Need | Papers | What an abstraction would do | Lock touched |
|------|--------|------------------------------|--------------|

Watch list, one paper only:

| Need | Paper | Note |
|------|-------|------|

## Orphans

Every name in `kpnn2.__all__`, and each documented keyword
argument and public method of those names.

| kpnn2 name or option | Used by | Need served, or "no application in this sample" |
|----------------------|---------|-------------------------------------------------|

## Friction

*Glue that repeats in three or more sketches, and kpnn2 calls the
sketches had to work around.*

## Changes since the previous run

*Verdicts that moved, gaps that closed or opened, orphans that
gained or lost users. "First run" if there is none.*

## Caveats

*Paper code opened and why, cards that look wrong, sketches that
did not run, and anything else that limits this run.*
