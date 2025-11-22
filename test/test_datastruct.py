import pytest
import torch
from dftio.datastruct.atomicbasis import AtomicBasis

def test_atomic_basis_init():
    """Test AtomicBasis initialization."""
    basis = AtomicBasis(
        element="Si",
        basis="2s2p",
        rcut=5.0,
        radial_type="spline",
        dtype=torch.float64
    )
    assert isinstance(basis, AtomicBasis)
    assert basis.element == "Si"
    assert basis.basis == "2s2p"
    assert basis.rcut == 5.0
    assert basis.radial_type == "spline"
    assert basis.dtype == torch.float64

def test_atomic_basis_str():
    """Test the string representation of AtomicBasis."""
    basis = AtomicBasis(
        element="Si",
        basis="2s2p",
        rcut=5.0,
        radial_type="spline",
        dtype=torch.float64
    )
    assert str(basis) == "Si 2s2p 5.0 spline"
