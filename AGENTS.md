# kpnn2 — agent instructions

This file is **how to work** in this repository: commands,
conventions, and process. **Read it first.**

`CONTEXT.md` is **what you are working on**: purpose,
architecture, locked product decisions, public API, and
primitive specs. It is often the next file to open. It is
not required for every task.

Do not paste this file, `CONTEXT.md`, or
`dev/src_guidelines.txt` into replies or into other
rules.

## When to read CONTEXT.md

Read `CONTEXT.md` before changing:

- `src/`
- tests that pin behavior of public names or internals
- public docs that describe behavior (including
  `docs/reference/`)
- `pyproject.toml` (dependencies, build, or tool config
  that is part of the contract)

Skip it for process-only work, git or commit-message nits,
tests-only wording, or docs that already match the
contract.

If a later prompt disagrees with this file on process, this
file wins. If it disagrees with `CONTEXT.md` on product or
architecture, `CONTEXT.md` wins. The user can override
either in that prompt.

Product locks (no compiler, no `torch.sparse`, two parsers,
frozen topology, and the rest) live in `CONTEXT.md`. Point
at named sections there. Do not copy those locks into this
file.

## When to read `dev/src_guidelines.txt`

Read `dev/src_guidelines.txt` before changing `src/`.
It is library-craft for package code, not the product
contract. Skip it otherwise (tests-only, docs, process,
git).

If it conflicts with `CONTEXT.md` on product or
architecture, `CONTEXT.md` wins. If it conflicts with
this file on process (pytest, seeds in tests, ruff,
lockstep, git), this file wins.

## Do not

- Do not rename the distribution, import, or `src/kpnn2/`
  directory.
- Do not grow or shrink the public API unless the user
  asked. Names are exactly `kpnn2.__all__`.
- Do not stage, commit, or push. Do not amend or skip
  hooks. The user stages and commits.
- Do not pin `kpnn2.__version__` in `CONTEXT.md`; it lives
  in `pyproject.toml` and `src/kpnn2/__init__.py`.

## Public API lockstep

If `__all__`, a public signature, or a public docstring
changes, update **all** of these in the same change:

- `src/kpnn2/__init__.py` (`__all__` and imports)
- `CONTEXT.md` (Public API table and related contract text)
- `docs/reference/` (one mkdocstrings page per public name;
  `index.md` is the grouped listing; docstrings stay the
  source of truth)
- `tests/api/test_public_api.py` (`_PUBLIC_NAMES` and leftover
  guards)

Do not export compiler leftovers (`compile_graph`,
`customize_model`, and the rest listed in that test).

## How to add an aggregation method

`aggregate_node_attributions` is a dispatcher. Methods live
under `src/kpnn2/_aggregation/_methods/`. They share one
internal signature
`(attributions, labels, **kwargs) -> xarray.Dataset`.
Do not change the dispatcher to add a method.

1. Add a module, for example
   `_aggregation/_methods/_someone_2027.py`.
2. Decorate the function with
   `@register_aggregation_method(name, status, description,
   references, added_in, ...)`. Statuses are `recommended`,
   `supported`, `experimental`, `deprecated`, `removed`.
   Deprecated entries need `deprecated_in` and
   `replacement`; removed entries need `removed_in` and
   `replacement`. Use `FutureWarning` for deprecated
   (not `DeprecationWarning`).
3. Import the module in
   `_aggregation/_methods/__init__.py` so the decorator
   runs.
4. The method checks its own dims and whether `labels` is
   required. Optional `seed` is this method's concern, not
   the dispatcher's. Reject unexpected dims.
5. Put the full numpy contract on that function's
   docstring (formulas, dims, kwargs, return variables).
   Do not add them to the dispatcher docstring.
6. Add `docs/reference/aggregation/{name}.md` that
   mkdocstrings-renders the private function. Note that
   the function is not imported from `kpnn2`; users pass
   `method="{name}"`.
7. Add a row to the Methods table on
   `docs/reference/aggregate_node_attributions.md`.

Do not export the method function or the decorator.
Do not put method pages in `docs/reference/*.md` or in
the Reference index Callables table. Do not change the
dispatcher function or its docstring to add a method.
`list_aggregation_methods()` and the dispatcher pick the
new name up with no other code changes. Tests may import
`register_aggregation_method` from
`kpnn2._aggregation._registry` and must unregister dummy
entries.

## Changelog

`CHANGELOG.md` is for important API and core changes only,
concise. See `CONTEXT.md` **Changelog**. Do not log docs
pages, notebooks, nav, wording, tests, or other small
edits. Leave the file alone when nothing notable shipped.
Lockstep above does **not** by itself justify a changelog
line.

## Commands

After changing anything under `src/`, run the **full**
pytest suite, including slow tests, before finishing:

```
pytest
```

Do not pass `-m "not slow"`. Fix failures from that run
before declaring the work done. Docs-only, rules-only, or
tests-only edits do not require this full run unless `src/`
also changed.

After Python edits, format with the `dev` extra pin:

```
python -m ruff format .
```

Do not use a global `ruff` on `PATH`; it can disagree with
CI. Exact pin: `pyproject.toml`.

Execute tutorial notebooks with:

```
python dev/docs_notebooks.py
```

That clears cell `execution_count` and Jupyter
`metadata.execution` afterward (outputs and cell ids
stay). Use `--fix-only` to clear without re-running.
Use the venv kernel, not
`ipykernel install --user --name python3`. `mkdocs serve`
repairs missing stream `name` fields on pre-build.
Literature notebooks under `docs/literature/` are frozen:
repair them, do not execute them in CI. Re-run with
`python dev/docs_notebooks.py --literature` after
downloading files into gitignored `.literature-data/`
(see `docs/literature/README.md`). Do not add bulk omics
matrices to git.

`tests/manual/` is Colab GPU/TPU smoke, not pytest and not
docs. Do not execute it in CI or with
`dev/docs_notebooks.py`. Open from GitHub via
`dev/colab.txt`. Install from TestPyPI with `--no-deps`.

MkDocs output, coverage, and wheels go under gitignored
`out/` (`out/site`, `out/htmlcov`, `out/coverage.xml`,
`out/dist`). Do not write `site/`, `htmlcov/`, or `dist/`
at the repo root. `python -m build` needs `-o out/dist`.

## Code style

- Lines `<= 80` characters.
- Function definitions and calls with 2+ arguments: each
  argument on its own line; closing `)` on its own line.
- Docs, README, and doctests use `import kpnn2` and
  `kpnn2.parse_layered(...)` (same for the other public
  names). Do not introduce `import kpnn2 as k2`. Users
  never import private modules (`kpnn2._masked_linear`).
- Public failures: `Kpnn2Error` only.
- Randomness in tests: `random.seed(42)`,
  `numpy.random.seed(42)`, `torch.manual_seed(42)`.

## Docs tutorials

Feedforward-example, skip-edges, and layered vs adjacency
use `PackedLinear` on hops. `MaskedLinear(hop.to_mask())`
is the dense hatch. Cyclic graph and time-series stay on
`AdjacencySpec`. `docs/packed_linear.md` is the PackedLinear
page (including the reparse hatch). The transformer example
is the `PackedMultiheadAttention` walkthrough. Do not
sprinkle that class through feedforward-example.

Notebooks must be valid nbformat v4. Stream outputs need
`name` (`stdout` / `stderr`).

Terminology (concepts vs abbreviations vs docstrings): see
`CONTEXT.md` **Docs terminology**. `docs/concepts.md` owns
concepts, `docs/includes/abbreviations.md` owns expansions,
docstrings own contracts.

## Git

Never run `git add`, `git commit`, `git push`, amend, or
skip hooks. Do not stage files.

After any change that would be a commit, end the reply
with **one** copyable fenced code block: imperative
subject (≈50 chars, no trailing period), blank line, then
1–3 short sentences on why / what matters, wrapped near
72 characters. No commentary inside the fence. A brief
note outside is fine (for example files to exclude).

## When to update this file vs CONTEXT.md

Update **this file** when process, commands, or conventions
move.

Update **`CONTEXT.md`** in the same change only if the
product contract moved: public API, package philosophy,
repository layout, or core dependencies.

Do not edit `CONTEXT.md` for typos in other files,
tests-only work, or docs that already match the contract.
Do not copy product locks from `CONTEXT.md` into this file.

Library-craft rules live in `dev/src_guidelines.txt`.
Update that file when those rules move. Do not copy
them here. Update this file only if the pointer or
when-to-read condition changes.
