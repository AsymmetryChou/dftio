# Contributing to dftio

Thank you for your interest in contributing to dftio!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/deepmodeling/dftio.git
cd dftio
```

2. Install with development dependencies:
```bash
./install.sh  # or specify GPU version
uv sync --group dev
```

3. Run tests:
```bash
uv run pytest
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Use pytest markers for integration tests

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Adding Support for New DFT Software

See the [Developer Guide](developer-guide.md) for details on implementing parsers for new DFT packages.
