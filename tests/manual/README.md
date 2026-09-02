# Manual Colab accelerator smoke

This directory is **not** pytest and **not** CI. Pytest does not
collect it. Do not execute these notebooks with
`python scripts/docs_notebooks.py`.

## When to run

Before a release, after the candidate is on **TestPyPI**:

1. Open the notebook from GitHub (link in `dev/colab.txt`).
   Do not upload a local copy.
2. **Runtime → Change runtime type → T4 GPU**, run all cells.
3. **Runtime → Change runtime type → TPU v2**, run all cells
   again (new session).

The first code cell installs `kpnn2` from TestPyPI with
`--no-deps` (and `--pre`, so release candidates are visible)
so Colab's CUDA or XLA PyTorch is not replaced.

The one unavoidable click is the runtime accelerator. The
notebook then picks CUDA or XLA from what Colab actually
provides.
