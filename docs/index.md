# dftio Documentation

Welcome to the **dftio** documentation! dftio is a powerful tool to assist machine learning communities in transcribing and manipulating DFT (Density Functional Theory) output into formats that are easy to read and use with machine learning models.

## Key Features

- 🚀 **Multi-DFT Support**: Parse outputs from ABACUS, RESCU, SIESTA, Gaussian, VASP, and PYATB
- ⚡ **Parallel Processing**: Utilize multiprocessing for fast data processing
- 📊 **Multiple Output Formats**: Export to DAT, ASE, or LMDB formats
- 🔧 **Flexible Data Extraction**: Extract structures, eigenvalues, Hamiltonians, density matrices, and overlap matrices
- 🎯 **ML-Ready**: Standard dataset classes compatible with PyTorch and PyTorch Geometric

## Quick Links

```{tableofcontents}
```

## Supported DFT Packages

| Package  | Structure | Eigenvalues | Hamiltonian | Density Matrix | Overlap Matrix |
|:--------:|:---------:|:-----------:|:-----------:|:--------------:|:--------------:|
|  ABACUS  |     ✓     |      ✓      |      ✓      |       ✓        |       ✓        |
|  RESCU   |     ✓     |             |      ✓      |                |       ✓        |
|  SIESTA  |     ✓     |             |      ✓      |       ✓        |       ✓        |
| Gaussian |     ✓     |             |      ✓      |       ✓        |       ✓        |
|   VASP   |     ✓     |      ✓      |             |                |                |
|  PYATB   |     ✓     |      ✓      |             |                |                |

## Getting Started

New to dftio? Check out the [Installation Guide](installation.md) and [Quick Start Tutorial](user-guide/quickstart.md).

## Contributing

dftio is an open-source project that welcomes contributions! See our [Contributing Guide](contributing.md) for details.

### Current Contributors

Qiangqiang Gu, Jijie Zou, Mingkang Liu, Zixi Gan, Zhanghao Zhouyin
