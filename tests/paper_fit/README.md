# Paper fit

A manual check, run with an AI agent, of one rule: derive
abstractions from applications, not applications from
abstractions. Published KPNN code is the application. kpnn2 is
the abstraction. The check asks three things:

1. Which part of each paper's code kpnn2 would have replaced.
2. Which needs, shared across papers, kpnn2 misses.
3. Which kpnn2 features no paper needs.

This directory is **not** pytest and **not** CI.
`pyproject.toml` tells pytest and Ruff to skip it. Nothing here
runs unless you start a session for it.

## Two kinds of session

Start each one in a fresh agent session. That keeps the token
budget per session small.

| Session | Send | Writes |
|---------|------|--------|
| Extract | `Follow tests/paper_fit/extract_prompt.md for <id>.` | `cards/<id>.md`, its row in `papers.csv`, `needs.md` |
| Assess | `Follow tests/paper_fit/assess_prompt.md.` | `runs/<date>_<commit>/` |

`Follow tests/paper_fit/extract_prompt.md for the next pending
paper.` takes the first `pending` row in `papers.csv`.

Extraction reads one paper and its code. It is the expensive
step and runs once per paper. Assessment reads only cards, the
needs catalog, and the kpnn2 contract. It is cheap enough to
repeat for each release.

Run extractions one at a time. Two parallel sessions can give
two different needs the same new ID in `needs.md`.

## Pilot

Run these five first, then adjust `cards/_template.md` and
`needs.md` before the rest. They differ in framework and in how
the graph enters the model.

| id | Why |
|----|-----|
| `kpnn-2020` | TensorFlow 1; a network of named biological nodes |
| `pnet-2021` | Keras; ontology levels, outputs at inner layers |
| `drugcell-2020` | PyTorch; several units per ontology term |
| `lembas-2022` | PyTorch; recurrent signaling network with cycles |
| `sctransformer-2026` | Attention restricted by a regulatory prior |

## Files

| Path | Role |
|------|------|
| `papers.csv` | Target list; one row per paper, never deleted |
| `extract_prompt.md` | Prompt for one extraction session |
| `assess_prompt.md` | Prompt for one assessment session |
| `needs.md` | Paper-neutral catalog of needs |
| `cards/_template.md` | Card layout |
| `cards/<id>.md` | One card per included paper |
| `runs/_template.md` | Report layout |
| `runs/<date>_<commit>/report.md` | One report per assessment |
| `runs/<date>_<commit>/sketches/<id>.py` | kpnn2 sketch per paper |

Paper code is cloned into `.paper-code/<id>/` at the repo root.
That directory is gitignored. Delete it whenever you like. The
pinned `code_ref` in `papers.csv` lets a session fetch the same
code again.

## `papers.csv`

| Column | Content |
|--------|---------|
| `id` | Stable handle; also the card file name |
| `year`, `title`, `venue`, `doi` | Citation |
| `code_url` | Repository or archive the authors link |
| `code_ref` | Commit SHA, or `sha256:<hash>` for an archive |
| `framework` | For example `PyTorch`, `TensorFlow 1`, `Keras` |
| `status` | `pending`, `included`, or `excluded` |
| `note` | Free text; excluded rows start with a reason tag |
| `source` | `csbg/kpnn-papers` or `manual` |

Reason tags for `excluded` rows are `review:`, `gnn:`,
`no-code:`, and `out-of-scope:`. Extraction sets them. The rules
are under **Triage** in `extract_prompt.md`. Six rows were
excluded from their titles alone when the list was set up; the
note says so.

An `id` is the lowercase model name plus the year
(`pnet-2021`). If the paper names no model, use two or three
distinctive title words (`self-pruning-2025`). Do not rename an
`id` once a card or a run uses it.

## Adding papers

1. Add a row with `status` `pending`, the citation columns, and
   `source`. Leave `code_url`, `code_ref`, and `framework` empty
   unless you know them.
2. Run an extraction session for it.
3. Run an assessment when you want the cross-paper picture to
   include it.

To sync with <https://github.com/csbg/kpnn-papers>, add rows for
entries whose DOI is not here yet. Do not delete rows. Excluded
rows stay so nobody triages them twice. Set an excluded row back
to `pending` when its situation changes, for example when code is
released.

## Re-extracting

Ask for it explicitly:

```text
Follow tests/paper_fit/extract_prompt.md and re-extract <id>.
```

The session fetches the pinned `code_ref`, unless you also ask it
to move to the latest commit. It rewrites the card and keeps
existing need IDs. Re-extract when the card template gains a
field the assessment needs, or when a run reports a card as
wrong.

## Why extraction does not see kpnn2

The extraction prompt forbids reading kpnn2 sources and docs. An
agent that knows kpnn2 tends to describe a paper in kpnn2's
terms, and the assessment then confirms itself. Needs are written
in plain deep-learning terms for the same reason.

## Excerpts

Cards quote short code excerpts with path, commit, and the
repository's license. Keep them short. The card points to the
code instead of copying it.
