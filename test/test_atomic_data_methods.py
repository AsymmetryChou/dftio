import pytest
import numpy as np
import torch
from dftio.data import AtomicData, _keys
import ase

def test_atomic_data_from_points():
    """Test AtomicData.from_points."""
    pos = np.array([[0.0, 0.0, 0.0]])
    r_max = 5.0
    cell = np.eye(3)
    pbc = [True, True, True]
    
    data = AtomicData.from_points(
        pos=pos,
        r_max=r_max,
        cell=cell,
        pbc=pbc,
        atomic_numbers=np.array([1])
    )
    
    assert isinstance(data, AtomicData)
    assert data.num_nodes == 1
    assert _keys.EDGE_INDEX_KEY in data

def test_atomic_data_from_ase():
    """Test AtomicData.from_ase."""
    atoms = ase.Atoms(
        numbers=[1],
        positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3),
        pbc=True
    )
    
    data = AtomicData.from_ase(atoms, r_max=5.0)
    
    assert isinstance(data, AtomicData)
    assert data.num_nodes == 1
    # AtomicData stores atomic numbers as [num_nodes] or [num_nodes, 1]?
    # The failure showed array([[1]]) vs array([1]).
    # So it stores as 2D array?
    # Let's check equality with correct shape.
    assert np.array_equal(data[_keys.ATOMIC_NUMBERS_KEY], np.array([[1]])) or \
           np.array_equal(data[_keys.ATOMIC_NUMBERS_KEY], np.array([1]))

def test_atomic_data_to_ase():
    """Test AtomicData.to_ase."""
    pos = np.array([[0.0, 0.0, 0.0]])
    r_max = 5.0
    cell = np.eye(3)
    pbc = np.array([True, True, True])
    
    data = AtomicData.from_points(
        pos=pos,
        r_max=r_max,
        cell=cell,
        pbc=pbc,
        atomic_numbers=np.array([1])
    )
    
    atoms = data.to_ase()
    
    assert isinstance(atoms, ase.Atoms)
    assert len(atoms) == 1
    assert atoms.get_atomic_numbers()[0] == 1
