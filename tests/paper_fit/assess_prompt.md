# Paper-fit assessment prompt

You are running one assessment of the paper-fit check in
`tests/paper_fit/`. It tests the rule "derive abstractions from
applications, not applications from abstractions". Published
KPNN code is the application. kpnn2 is the abstraction. Answer
three questions from the cards:

1. For each paper, which part of its connectivity code kpnn2
   would have replaced, and what glue would remain.
2. Which needs shared across papers kpnn2 does not cover. These
   are candidate abstractions.
3. Which kpnn2 features no paper needs. These are candidate
   orphans.

## Scope

Assess every row of `tests/paper_fit/papers.csv` whose `status`
is `included`. If the user names ids, assess only those, mark the
cross-paper sections as not updated, and say so at the top of the
report.

## Read, in this order

1. `tests/paper_fit/README.md`, `papers.csv`, and `needs.md`.
2. Every `tests/paper_fit/cards/<id>.md` in scope.
3. `report.md` in the newest directory under
   `tests/paper_fit/runs/`, if there is one.
4. `AGENTS.md` and `CONTEXT.md` in full. `CONTEXT.md` is the
   kpnn2 contract: public API, locked decisions, and what the
   package is not.
5. The version and the state of the tree:

   ```bash
   .venv/bin/python -c "import kpnn2; print(kpnn2.__version__)"
   .venv/bin/python -c "import kpnn2; print(kpnn2.__all__)"
   git rev-parse --short HEAD
   git status --short
   ```

   Use the Python that has this checkout of kpnn2 installed.
   `.venv/bin/python` is the usual one.

Open `src/` only to check a public docstring that `CONTEXT.md`
does not settle. Open paper code in `.paper-code/<id>/` only at a
path a card cites, and only when the card is unclear. List each
such lookup under **Caveats**. If `.paper-code/<id>/` is missing,
the card is the only source.

## Rules

- **Score against the paper's needs, not against kpnn2's
  features.** Start from each card's Needs and Glue tables.
- **Public API only.** Sketches use `import kpnn2` and names in
  `kpnn2.__all__`, as `AGENTS.md` **Code style** says. Private
  modules do not count.
- **Every sketch runs.** Plain PyTorch the user would still
  write ends with a `# GLUE` comment. A need covered only by
  glue is not covered.
- **Locks are context, not gaps to close.** If closing a gap
  would break a locked decision in `CONTEXT.md`, name the section
  and stop there.
- **Recommend, do not decide.** Gaps and orphans are input for
  the maintainer. Do not edit `src/`, `docs/`, `CONTEXT.md`,
  `AGENTS.md`, cards, `needs.md`, or `papers.csv`. If a card
  looks wrong, report it under **Caveats**.
- **Write only** the run directory
  `tests/paper_fit/runs/<YYYY-MM-DD>_<short-sha>/`. Add
  `-dirty` to the name if `git status --short` was not empty. Do
  not stage or commit.

## Procedure

### 1. Per paper

For each card in scope:

1. Map each need in the card's Needs table to one of:
   - `covered`: name the kpnn2 public names that cover it.
   - `glue`: kpnn2 helps, but plain PyTorch remains; name it.
   - `missing`: kpnn2 does not help.
   - `out of scope`: the contract leaves it to the user, for
     example the training loop. Cite the `CONTEXT.md` section.
2. Write `sketches/<id>.py`: the paper's connectivity core with
   kpnn2. Build the graph as a `source`/`target` DataFrame, parse
   it, build the constrained layers and `forward`, and map
   attributions if the card has an Interpretation section. Use a
   toy graph of at most 30 nodes with the paper's shape: the same
   depth pattern, units per node, skip connections, and cycles.
   Keep it under about 60 lines. Seed with
   `torch.manual_seed(42)`. For attributions, use Captum if it
   imports, otherwise input times gradient.
   If the previous run has a sketch for this paper, start from
   it. Check its verdict against the card again; do not carry it
   over.
3. Run the sketch on one random batch. Fix kpnn2 misuse until it
   exits cleanly. Record pass or fail.
4. Count the lines that call kpnn2 and the `# GLUE` lines.
5. For each row of the card's Glue table, say whether kpnn2
   removes it, shrinks it, or leaves it.
6. Give a verdict:
   - `R`, replaces: the connectivity core uses kpnn2, and what
     remains is ordinary model code.
   - `P`, partial: kpnn2 covers part of it. Name the needs left
     to glue.
   - `N`, no fit: a core need blocks kpnn2 or makes it pointless.
     Name that need.

### 2. Across papers

1. **Gaps.** Needs that are `missing` or `glue` in two or more
   papers. For each, list the papers, what an abstraction would
   have to do, and any `CONTEXT.md` lock it touches. Needs from a
   single paper go on a separate watch list.
2. **Orphans.** Every name in `kpnn2.__all__`, plus each
   documented keyword argument and public method of those names
   (for example `widths=`, `ranks=`, `transpose()`). For each,
   list the papers whose sketch uses it. For an unused one, name
   a need in `needs.md` it serves, or write "no application in
   this sample".
3. **Friction.** `# GLUE` patterns that repeat in three or more
   sketches, and kpnn2 calls that sketches had to work around.
4. **Changes.** Against the previous run: verdicts that moved,
   gaps that closed or opened, orphans that gained or lost users.

### 3. Report

Copy `tests/paper_fit/runs/_template.md` to `report.md` in the run
directory and fill it. Reply with the run directory, the verdict
counts, the main gaps, and the orphans. End with the commit
message block that `AGENTS.md` asks for.
