# GEMINI.md: AI-Assisted Development Guide for `dftio`

This document provides context and instructions for an AI assistant to effectively contribute to the `dftio` project.

## Project Overview

`dftio` is a Python library designed to parse and process output files from various Density Functional Theory (DFT) software packages. Its primary goal is to convert complex DFT outputs into standardized, machine-learning-ready formats. The library supports packages such as ABACUS, VASP, SIESTA, Gaussian, RESCU, and PYATB.

The project provides a command-line interface (CLI) for parsing operations and for plotting derived data like electronic band structures.

**Key Technologies:**
- **Language:** Python 3.9+
- **Core Libraries:** NumPy, SciPy, PyTorch, ASE (Atomic Simulation Environment), sisl
- **Package Management:** `uv`
- **Testing:** `pytest`

## Building and Running

### Installation

The project uses `uv` for dependency management. To set up a development environment, including testing dependencies, run:

```bash
# Install all dependencies, including development tools
uv sync --group dev
```
This command installs packages defined in `pyproject.toml`.

### Running the CLI

The main entry point is the `dftio` command. It has several subcommands, with `parse` being the most central one.

**Example for parsing ABACUS output:**
```bash
dftio parse --mode abacus --root /path/to/abacus/output --hamiltonian --overlap -o /path/to/save
```

For a full list of commands and options, use the help flag:
```bash
dftio --help
dftio parse --help
```

### Running Tests

The project uses `pytest` for testing. The standard test suite can be run with the following command, which excludes slower "integration" tests:

```bash
uv run pytest -v -m "not integration"
```

To run the full suite including code coverage analysis (as done in CI):
```bash
uv run pytest -v -m "not integration" --cov=dftio
```

## Development Conventions

### Project Structure

- **`dftio/`**: Main source code for the library.
    - **`io/`**: Contains the parsing logic for different DFT packages. Each package (e.g., `abacus`, `siesta`) has its own submodule.
    - **`data/`**: Data structures for handling atomic configurations and computational results.
    - **`__main__.py`**: Defines the CLI entry point and its arguments.
- **`test/`**: Contains all `pytest` tests. The structure mirrors the main `dftio/` directory.
- **`docs/`**: Project documentation, built with Jupyter Book.
- **`pyproject.toml`**: Defines project metadata, dependencies, and tool configurations (including `pytest`).

### Coding Style

Follow the existing coding style in the file you are editing. While no specific linter is enforced in the project configuration, adhere to standard PEP 8 conventions.

### Adding a New Parser

To add support for a new DFT package, you would typically:
1. Create a new module under `dftio/io/`, e.g., `dftio/io/new_package/`.
2. Implement a parser class within that module.
3. Register the new parser in `dftio/io/parse.py`'s `ParserRegister` to make it available via the CLI.
4. Add corresponding tests in the `test/` directory to validate the parser's correctness.
