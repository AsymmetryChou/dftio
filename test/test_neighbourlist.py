import pytest
import torch
import numpy as np
from dftio.datastruct.neighbourlist import PrimitiveFieldsNeighborList

@pytest.fixture
def neighbourlist_data():
    """Fixture for PrimitiveFieldsNeighborList class."""
    cutoffs = np.array([2.0, 2.0])
    pbc = np.array([True, True, True])
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    grids = np.array([[0.5, 0.5, 0.5]])
    return cutoffs, pbc, cell, coordinates, grids

def test_neighbourlist_init(neighbourlist_data):
    """Test PrimitiveFieldsNeighborList initialization."""
    cutoffs, _, _, _, _ = neighbourlist_data
    nl = PrimitiveFieldsNeighborList(cutoffs=cutoffs)
    assert isinstance(nl, PrimitiveFieldsNeighborList)
    assert np.array_equal(nl.cutoffs, cutoffs)

def test_neighbourlist_build(neighbourlist_data):
    """Test building the neighbor list."""
    cutoffs, pbc, cell, coordinates, grids = neighbourlist_data
    nl = PrimitiveFieldsNeighborList(cutoffs=cutoffs)
    nl.build(pbc, cell, coordinates, grids)
    assert nl.nupdates == 1
    assert nl.nneighbors > 0
    neighbors, displacements = nl.get_neighbors(0)
    assert len(neighbors) > 0
    assert len(displacements) > 0

def test_neighbourlist_update(neighbourlist_data):
    """Test updating the neighbor list."""
    cutoffs, pbc, cell, coordinates, grids = neighbourlist_data
    nl = PrimitiveFieldsNeighborList(cutoffs=cutoffs, skin=0.1)
    updated = nl.update(pbc, cell, coordinates, grids)
    assert updated is True
    assert nl.nupdates == 1
    updated = nl.update(pbc, cell, coordinates, grids)
    assert updated is False
    assert nl.nupdates == 1
    new_coordinates = coordinates + 0.2
    updated = nl.update(pbc, cell, new_coordinates, grids)
    assert updated is True
    assert nl.nupdates == 2
