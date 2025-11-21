# Quick Start

This guide will help you get started with dftio in just a few minutes.

## Basic Usage

The main command for dftio is `dftio parse`, which parses DFT output files and converts them to your desired format.

### Example 1: Parse ABACUS Output

```bash
uv run dftio parse \
    --mode abacus \
    --root /path/to/abacus/outputs \
    --prefix "frame*" \
    --outroot ./parsed_data \
    --format lmdb \
    --hamiltonian \
    --overlap
```

This command:
- Parses ABACUS output files
- Looks for directories matching `frame*` pattern
- Extracts Hamiltonian and overlap matrices
- Saves to LMDB format in `./parsed_data`

### Example 2: Parse SIESTA Output with Eigenvalues

```bash
uv run dftio parse \
    --mode siesta \
    --root /path/to/siesta/outputs \
    --outroot ./siesta_parsed \
    --format dat \
    --eigenvalue \
    --hamiltonian \
    --overlap
```

### Example 3: Parallel Processing

For large datasets, use multiple workers:

```bash
uv run dftio parse \
    --mode abacus \
    --root /path/to/large/dataset \
    --num_workers 8 \
    --format lmdb \
    --hamiltonian
```

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--mode` | DFT software name | `abacus`, `siesta`, `gaussian`, `vasp`, `pyatb` |
| `--root` | Root directory of DFT files | `/path/to/data` |
| `--prefix` | Pattern to match subdirectories | `frame*`, `calc_*` |
| `--outroot` | Output directory | `./parsed_data` |
| `--format` | Output format | `dat`, `ase`, `lmdb` |
| `--num_workers` | Number of parallel workers | `1` (default), `4`, `8` |

## What to Extract

### Structure Information

Always extracted by default:
- Atomic numbers
- Positions
- Cell parameters
- Periodic boundary conditions

### Optional Quantities

Add flags to extract additional data:

- `--hamiltonian` - Hamiltonian matrices
- `--overlap` - Overlap matrices
- `--density_matrix` - Density matrices
- `--eigenvalue` - K-points and eigenvalues

### Band Index Selection

For eigenvalues, you can skip low-energy bands:

```bash
uv run dftio parse \
    --mode abacus \
    --eigenvalue \
    --band_index_min 10  # Skip first 10 bands
```

## Output Formats

### DAT Format

Simple text-based format, easy to inspect:

```bash
--format dat
```

### ASE Format

Compatible with Atomic Simulation Environment:

```bash
--format ase
```

### LMDB Format (Recommended for ML)

Lightning Memory-Mapped Database, efficient for large datasets:

```bash
--format lmdb
```

## Next Steps

- [Detailed Parsing Guide](parsing.md) - Learn about parsing each DFT software
- [Data Formats](data-formats.md) - Understand output data structures
- [CLI Reference](cli-reference.md) - Complete command-line reference
