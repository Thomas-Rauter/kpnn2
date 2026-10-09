# Paper-fit extraction prompt

You are writing one **paper card** for the paper-fit check in
`tests/paper_fit/`. A card is a compact, faithful record of how
one published model puts prior knowledge into a neural network. A
later session works from cards alone, so the card must be precise
enough that nobody has to open the code again.

## Target

The user names a paper `id` from `tests/paper_fit/papers.csv`. If
they ask for the next pending paper, take the first row whose
`status` is `pending`. Work on that one paper only.

- `pending`: run the whole procedure.
- `included` or `excluded`: stop and tell the user, unless they
  asked to re-extract. A re-extraction fetches the row's
  `code_ref` (or the latest commit, if the user asked for that),
  rewrites the card, and keeps existing need IDs.

## Rules

1. **Do not look at the library in this repository.** Do not open
   `src/`, `CONTEXT.md`, `README.md`, `docs/`, `dev/`, `local/`,
   or any path under `tests/` outside `tests/paper_fit/`. Inside
   `tests/paper_fit/`, do not open `assess_prompt.md`, `runs/`,
   or other papers' cards. If your harness loaded `AGENTS.md`,
   ignore what it says about the package. Its git rules still
   apply.
2. **Plain terms.** Describe the paper in its own words or in
   plain deep-learning terms: node, edge, layer, unit, mask,
   weight. Do not use names from this repository's package.
3. **Evidence.** Every claim about code cites `path:line` or
   `path:start-end`, relative to the root of the paper's
   repository at `code_ref`. Write "not found" or "unclear"
   instead of guessing.
4. **Describe, do not judge.** No verdicts on library fit, on
   code quality, or on what some library could do for the paper.
5. **The paper's code is read-only.** Do not install its
   dependencies, run it, or download its data.
6. **Write only** `tests/paper_fit/cards/<id>.md`, the target's
   row in `tests/paper_fit/papers.csv`, and
   `tests/paper_fit/needs.md`. Fetch code only into
   `.paper-code/<id>/` at the repo root. Do not stage or commit.
7. **Budget.** Aim for well under 100k tokens. Search before you
   read, and read files in line ranges. Do not read data files,
   logs, or notebook outputs.

## Procedure

### 1. Paper

Find an open full text: the DOI landing page, PubMed Central,
arXiv, or bioRxiv. Read the abstract, the model or architecture
section, the code availability statement, and the caption of the
architecture figure. Skip results and discussion unless the
architecture is only described there. If no full text is open,
use the abstract and the code README, and say so in the card.

### 2. Triage

Pick one status. Stop early on an exclusion.

| Status | When | `note` starts with |
|--------|------|--------------------|
| `excluded` | Review, commentary, or perspective without a model of its own | `review:` |
| `excluded` | Message-passing graph neural network (see below) | `gnn:` |
| `excluded` | No public code in the paper, its supplement, or a search for the model name or title on GitHub and Zenodo | `no-code:` |
| `excluded` | The prior-knowledge graph does not constrain the network | `out-of-scope:` |
| `included` | Everything else | |

Decide `gnn:` by mechanism, not by imports. A message-passing
network updates each node's state by aggregating its neighbors
with a function whose weights are shared across edges, whether it
uses PyG, DGL, or hand-written code. A model in which each
prior-knowledge edge has its own weight or mask entry is
`included`, even if it uses a graph library for indexing. A model
with both parts is `included`; the card covers the constrained
part and names the other one.

For an exclusion, update the row: `status`, a one-sentence `note`
with the reason and what you searched, and `code_url` if you
found one. Write no card, fetch no code, and go to step 7.

### 3. Code

Use the code the paper links to. If there are several
repositories, use the authors' official one and name the others
in the card.

For a git repository, run this from the repo root. On a
re-extraction, set `REF` to the row's `code_ref`.

```bash
ID=<id>; URL=<code_url>; REF=HEAD
D=.paper-code/$ID
git init -q "$D"
git -C "$D" remote add origin "$URL"
git -C "$D" fetch -q --depth 1 --filter=blob:none origin "$REF"
git -C "$D" sparse-checkout set --no-cone \
  '*.py' '*.ipynb' '*.R' '*.r' '*.lua' '*.m' '*.jl' '*.sh' \
  '*.md' '*.rst' '*.cfg' '*.toml' '*.yml' '*.yaml' \
  'requirements*.txt' 'LICENSE*' 'COPYING*'
git -C "$D" checkout -q FETCH_HEAD
git -C "$D" rev-parse HEAD
git -C "$D" ls-files
```

This downloads source files only. `ls-files` still lists the
whole tree, data included. If you need one small file that is not
source, for example the ontology file to see its format, add it
with `git -C "$D" sparse-checkout add <path>` and read only its
first lines.

For an archive (Zenodo, journal supplement), download it into
`.paper-code/<id>/`, unpack it there, and record
`sha256:<hash of the archive>` as `code_ref`.

### 4. Find the connectivity code

1. Read the repository README and the `ls-files` tree. Skip
   vendored libraries, plotting, and data preparation that does
   not touch the graph.
2. Search before you read. Add `-l` first if the output is long.

   ```bash
   X=--exclude-dir=.git
   grep -rnIiE $X 'mask|sparse|adjacen|connect' "$D"
   grep -rnIiE $X 'ontolog|pathway|reactome|kegg|hierarch' "$D"
   grep -rnIE $X 'nn\.Module|keras\.layers|def (forward|call)\(' "$D"
   grep -rnIiE $X 'deeplift|shap|captum|integrated|attribut' "$D"
   ```

3. Read the model definition in full. Of the rest, read only the
   code that builds the graph, builds masks or index arrays,
   aligns data features to nodes, applies the connectivity, and
   computes importance scores.
4. For notebooks, print the code cells only:

   ```bash
   python3 - path/to/notebook.ipynb <<'EOF'
   import json, sys
   for cell in json.load(open(sys.argv[1]))["cells"]:
       if cell["cell_type"] == "code":
           print("".join(cell["source"]), "\n# ---")
   EOF
   ```

5. Stop exploring once every card section is filled or marked
   "not found".

Where the code and the paper disagree, the card describes the
code. Record the difference under **Paper vs code**.

### 5. Card

Copy `tests/paper_fit/cards/_template.md` to
`tests/paper_fit/cards/<id>.md` and fill every section. Keep it
under about 250 lines. Excerpts are verbatim, at most 60 lines in
total, each headed by its `path:start-end`.

### 6. Needs

Read `tests/paper_fit/needs.md` and follow its rules. List what
the model has to do to get prior knowledge into, through, or out
of the network. For each item:

- If an existing need means the same thing, add `<id>` to its
  Papers column.
- Otherwise add a row with the next free ID.

Every need you touch appears in the card's **Needs** table with
code evidence.

### 7. Row and reply

Update the target's row in `papers.csv`: `status`, `note`,
`code_url`, `code_ref`, and `framework`. Correct `title`,
`venue`, `year`, or `doi` if the row is wrong or empty. Keep the
CSV valid: quote any field that contains a comma or a quote
character, and do not reorder rows or columns.

Reply with the id, the status, the card path, the needs you
reused, the needs you added, and what you could not determine.
End with the commit message block that `AGENTS.md` asks for.
