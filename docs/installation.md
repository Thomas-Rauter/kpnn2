# Installation

## Install from PyPI

Install the core `kpnn2` package from PyPI with:

```bash
pip install kpnn2
```

The core installation supports parsing an edgelist into a
`LayeredSpec`, building `MaskedLinear` layers, aligning named inputs,
and mapping a layer tensor back to named nodes. Training stays
ordinary PyTorch.

`kpnn2` requires Python 3.10 or later. See the tested Python
versions in the badge on the [**Home**](index.md) page.

Core dependencies are `torch`, `pandas`, `numpy`, and `xarray`.

## Attribution tools

`kpnn2` does not run attribution for you. Captum is not a package
dependency.

If you want Captum (as in
[**Getting started**](getting-started.ipynb)), install it yourself:

```bash
pip install captum
```

Then pass the resulting tensor to
[`map_node_attributions`][kpnn2.map_node_attributions].
Any other attribution method that yields a tensor of named-layer
units works the same way. Extra axes (class, step, …) stay as extra
xarray dimensions.

## AnnData

AnnData input is not supported in v1. `align_inputs()` accepts a
pandas DataFrame. Tensors are not accepted; pass a pre-ordered
tensor straight to the model.

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

That extra pins an exact Ruff version. CI runs the same pin.
A global `ruff` on `PATH` (for example `~/.local/bin/ruff`) can
be a different version and will accept code that CI then rejects.

After installing the `dev` extra, lint and format with the
environment's interpreter so `PATH` cannot shadow Ruff:

```bash
python -m ruff --version
python -m ruff check .
python -m ruff format --check .
```

To apply formatting:

```bash
python -m ruff format .
```

When upgrading Ruff, bump the pin in `pyproject.toml`, run
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
