# Installation

## Install from PyPI

Install the core `kpnn2` package from PyPI with:

```bash
pip install kpnn2
```

`kpnn2` requires Python 3.10 or later. See the tested Python
versions in the badge on the [**Home**](index.md) page.

Core dependencies are `torch`, `pandas`, `numpy`, and `xarray`.

## Attribution tools

`kpnn2` does not run attribution for you. Captum is not a package
dependency.

```bash
pip install captum
```

Any other attribution method that yields a tensor of named-layer
nodes also works.

## Development installation

To work on the package locally, clone the repository and install it
in editable mode from the project root:

```bash
git clone git@github.com:Thomas-Rauter/kpnn2.git
cd kpnn2
pip install -e .
```

## Optional dependency groups

Install development dependencies with:

```bash
pip install -e ".[dev]"
```

That extra pins the Ruff version CI uses. Run Ruff as
`python -m ruff`, so a different `ruff` on `PATH` cannot stand in
for the pinned one. The full list of local checks is
[`dev/check_code_locally.txt`](https://github.com/Thomas-Rauter/kpnn2/blob/main/dev/check_code_locally.txt).
To upgrade Ruff, bump the pin in `pyproject.toml`, run
`python -m ruff format .`, and commit the pin and any rewrites
together.

Install documentation dependencies with:

```bash
pip install -e ".[docs]"
```

The `docs` extra includes the tools used by the documentation
notebooks, such as Captum, Graphviz (Python package), seaborn, and
Jupyter.

## Notebooks and documentation

Some documentation notebooks draw graphs with Graphviz. If a
notebook requires Graphviz rendering, you also need the system-level
Graphviz installation, not only the Python package.

For example, on Ubuntu or Debian:

```bash
sudo apt install graphviz
```

## Verify the installation

A minimal core-installation smoke test is:

```bash
python -c "import kpnn2; print('kpnn2 imported successfully')"
```
