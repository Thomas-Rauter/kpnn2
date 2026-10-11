# Paper-fit proposal prompt

You are writing one **proposal** for the paper-fit check in
`tests/paper_fit/`. An assessment run lists where kpnn2 falls
short of published KPNN code: gaps, orphans, and friction. A
proposal turns that list into one change to kpnn2 that the
maintainer can accept or reject. You recommend. The maintainer
decides. You do not implement.

A proposal may be large and take a dozen commits. It must still
be **one thing**: one problem, solved by parts that belong
together.

## Scope

Work from the newest directory under `tests/paper_fit/runs/`,
unless the user names a run. If the user names a gap, need, or
kpnn2 name, propose for that one, or explain why it fails the
rules below.

## Read, in this order

1. `tests/paper_fit/README.md`.
2. The run's `report.md` and its `sketches/`. Skim the
   `report.md` of older runs for trends only.
3. Every file in `tests/paper_fit/proposals/` except
   `_template.md`.
4. `AGENTS.md`, `CONTEXT.md` in full, and
   `dev/src_guidelines.txt`.
5. The state of the tree. `<run-sha>` is the short SHA in the run
   directory's name.

   ```bash
   git rev-parse --short HEAD
   git status --short
   git diff --stat <run-sha> HEAD -- src/
   .venv/bin/python -c "import kpnn2; print(kpnn2.__version__)"
   ```

6. After the shortlist (step 1 below): the `needs.md` rows it
   cites, and the Needs, Glue, and Fragile spots sections of the
   cards of the papers it cites.
7. After choosing (step 3 below): the modules in `src/` the change
   would touch, and the tests that pin them.

Do not open paper code in `.paper-code/`. If a card is unclear,
say so under **Caveats**.

## Freshness gate

Check this before step 1. Stop and ask the user to run an
assessment first if:

- no run exists;
- the run's Scope misses a row of `papers.csv` whose `status` is
  `included`; or
- `git diff <run-sha> HEAD -- src/` is not empty.

Go on only if the user tells you to, and list the mismatch under
**Caveats**. If the run directory ends in `-dirty`, note that too.

## Rules

- **One thing.** State the problem in one sentence that names the
  needs it serves. The parts of the proposal belong together
  when:
  - every part serves that sentence;
  - the parts share one concept, name, or data structure; and
  - dropping any part leaves the problem partly unsolved or makes
    another part pointless.

  Two fixes that only share a module are two proposals. Pick
  one.
- **Evidence comes from the run.** A paper supports a need when
  the run maps that need to `glue` or `missing` for that paper.
  `covered` and `out of scope` do not count. The run's
  `out of scope` mappings are settled, including pre-edgelist
  ones. Papers whose cards say one builds on the other's code
  count once. A candidate needs at least two supporting papers.
  A watch-list need joins only a pattern that already has two.
- **One pattern, not two needs.** Treat two needs as one pattern
  only when `needs.md` merges them, the run says they are one, or
  you show that one call would serve both.
- **Removals count.** An orphan can be the candidate: remove,
  narrow, or merge a public name or option. "No application in
  this sample" is weak alone. Say what keeping it costs, and which
  use case in `CONTEXT.md` it serves, if any. Give the deprecation
  path.
- **Why in kpnn2.** `CONTEXT.md` **Package philosophy** splits
  the work between kpnn2 and the user. Name the responsibility of
  kpnn2 the change extends. Glue a user writes in a few lines on
  public names is not reason enough, unless that glue is a
  fragile spot: easy to get silently wrong, or dependent on
  kpnn2's internal layout.
- **Locks.** A candidate that breaks a locked decision in
  `CONTEXT.md` qualifies only with three or more supporting papers
  and no workaround inside the lock. Its title then starts with
  `Lock change:`. Name the section, what the lock protects, and
  what lifting it costs. Otherwise, list it under **Considered
  and not chosen** with the section.
- **Concrete.** Give signatures, defaults, errors, returned
  objects with their dims and names, and where the code lives.
  Leave nothing to the implementer except what you list under
  **Open questions**.
- **Smallest public surface.** Count every public name, keyword
  argument, and method the change adds or removes.
- **Earlier proposals.** Do not repeat a `rejected` or `deferred`
  proposal unless the run brings evidence it lacked; say what is
  new. Do not overlap an `accepted` one.
- **Write only** `tests/paper_fit/proposals/<YYYY-MM-DD>_<slug>.md`.
  Do not edit `src/`, `tests/` outside that file, `docs/`,
  `CONTEXT.md`, `AGENTS.md`, `CHANGELOG.md`, runs, cards,
  `needs.md`, `papers.csv`, or earlier proposals. Do not write
  planner task files. Do not stage or commit.
- **Budget.** Aim for well under 150k tokens. Read `src/` only
  where the chosen change lands.

## Procedure

### 1. Shortlist

Collect candidates from the run: each row of **Gaps**, each
**Friction** item, each orphan without users, and watch-list
needs that join a gap. Group needs that one mechanism would
serve. Drop candidates that fail the evidence rule. Keep at most
five.

### 2. Choose

First drop candidates that fail **Why in kpnn2** or **Locks**.
Rank the rest by these, in order. A later one only breaks ties.

1. **Breadth:** supporting papers, counted as above.
2. **Effect:** needs that would become `covered`, verdicts that
   would move, `# GLUE` lines removed, and fragile spots closed.
3. **Cost:** public surface added (removed surface counts in
   favor), internal code, migration for users, and numbers that
   would change.

If no candidate is left, write no file. Reply with the shortlist
and the rule each candidate failed.

### 3. Design

1. Read the `src/` modules the change touches and the tests that
   pin them. Check whether internal code already does part of it.
2. Specify the public surface, the behavior including edge cases,
   the errors (`CONTEXT.md` **Errors**), and the files. Follow any
   `AGENTS.md` procedure that applies, for example **How to add an
   aggregation method**.
3. Rewrite the `# GLUE` lines of two to four affected sketches
   with the proposed API. Mark each new call `# PROPOSED`. These
   excerpts do not run.
4. Predict the next run: for each supporting paper and need, the
   mapping now and after. Add the verdicts that would move and the
   orphans that would gain users.
5. Split the work into slices, one commit each, in order:
   - Each slice leaves pytest, mypy, numpydoc, and Ruff green.
   - Each slice is one concern that `dev/llm_dev/planner.txt` can
     take as its TASK.
   - The first slice is the smallest one useful on its own.
   - Per slice, say whether **Public API lockstep** applies,
     whether it earns a `CHANGELOG.md` line, and whether it
     changes a formula, a numerical default, a reduction, or a
     tie-break (**Scientific acceptance** in `AGENTS.md`).
6. List the open questions: design choices the maintainer must
   settle before slice 1 can be planned. Give your recommendation
   for each.

### 4. Write

Copy `tests/paper_fit/proposals/_template.md` to
`tests/paper_fit/proposals/<YYYY-MM-DD>_<slug>.md`. The slug is
two to four lowercase words joined by hyphens. Set **Status** to
`proposed` and fill every section. Keep it under about 300 lines.

For each earlier proposal whose status is `done`, check its
predicted effect against the run. Record whether it held under
**Earlier proposals**.

Reply with the file path, the title, the problem sentence, the
supporting papers, the number of slices, and the open questions.
End with the commit message block that `AGENTS.md` asks for.
