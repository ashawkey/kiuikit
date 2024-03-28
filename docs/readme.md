# Docs for kiuikit

We use [sphinx](https://www.sphinx-doc.org/en/master/) and [m2r2](https://github.com/CrossNox/m2r2) to build the documentation in markdown language.

### Usage

Since sphinx needs to import the module for auto-doc, we need to install full dependency first:
```bash
pip install -e ".[full]"
pip install -r docs/requirements.txt
```

Build the docs by:
```bash
cd docs
make html
```

View the local [html](docs/build/html/index.html) by VSCode `Live Server` extension.