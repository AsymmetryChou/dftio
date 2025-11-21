# Installation Guide

## Prerequisites

- Python 3.9 - 3.12
- pip (Python package installer)

## Recommended: Using the Install Script

The easiest way to install dftio is using the provided installation script, which automatically handles the `torch-scatter` dependency:

### CPU Version (Default)

```bash
git clone https://github.com/deepmodeling/dftio.git
cd dftio
./install.sh
```

### GPU Version

For CUDA-enabled GPUs, specify your CUDA version:

```bash
# CUDA 11.8
./install.sh cu118

# CUDA 12.1
./install.sh cu121

# CUDA 12.4
./install.sh cu124
```

## Manual Installation with UV

If you prefer manual control:

### 1. Install UV

```bash
pip install uv
```

### 2. Clone the Repository

```bash
git clone https://github.com/deepmodeling/dftio.git
cd dftio
```

### 3. Install Dependencies

**CPU version:**
```bash
uv sync
```

**GPU version (CUDA 12.1 example):**
```bash
uv sync --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

## Development Installation

For development, install with dev dependencies:

```bash
uv sync --group dev
```

This includes additional tools for:
- Testing (pytest, pytest-cov)
- Documentation (jupyter-book, sphinx-autodoc-typehints)

## Verify Installation

After installation, verify that dftio is working:

```bash
uv run dftio --help
```

You should see the dftio command-line interface help message.

## Troubleshooting

### torch-scatter Installation Issues

The most common installation issue involves `torch-scatter`. This package requires special handling because it needs to match your PyTorch and CUDA versions.

**Solution**: Use the install script or the `--find-links` option as shown above.

### Python Version Issues

dftio requires Python 3.9-3.12. Check your Python version:

```bash
python --version
```

If you have multiple Python versions, specify the correct one with UV:

```bash
uv sync --python python3.11
```

### Network Issues

If you experience network timeouts, try:

```bash
uv sync --no-cache
```

## Next Steps

- [Quick Start Guide](user-guide/quickstart.md)
- [CLI Reference](user-guide/cli-reference.md)
