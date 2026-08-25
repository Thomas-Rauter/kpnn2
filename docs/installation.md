# Installation

## Install from PyPI

Install the core `kpnn2` package from PyPI with:

```bash
pip install kpnn2
```

The core installation supports parsing an edgelist into a
`GraphSpec`, building `MaskedLinear` layers, aligning named inputs,
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
[`map_node_attributions`](api.md#kpnn2.map_node_attributions.map_node_attributions).
Any other attribution method that yields a tensor of named-layer
units works the same way. Extra axes (class, step, …) stay as extra
xarray dimensions.

## AnnData

AnnData input is not supported in v1. `align_inputs()` accepts a
pandas DataFrame or a pre-ordered tensor.

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
