import pytest
import numpy as np
from dftio.data import AtomicData
from dftio.data import _keys

def test_atomic_data_init():
    """Test AtomicData initialization."""
    data = AtomicData(
        pos=np.array([[0.0, 0.0, 0.0]]),
        edge_index=np.array([[0], [0]], dtype=np.int64),
        atomic_numbers=np.array([1], dtype=np.int64),
        cell=np.eye(3),
        pbc=np.array([True, True, True])
    )
    assert np.array_equal(data[_keys.POSITIONS_KEY], np.array([[0.0, 0.0, 0.0]]))
    assert np.array_equal(data[_keys.ATOMIC_NUMBERS_KEY], np.array([[1]], dtype=np.int64))

def test_atomic_data_is_dict():
    """Test that AtomicData behaves like a dictionary."""
    data = AtomicData(
        pos=np.array([[0.0, 0.0, 0.0]]),
        edge_index=np.array([[0], [0]], dtype=np.int64),
        atomic_numbers=np.array([1], dtype=np.int64)
    )
    data["custom_key"] = "custom_value"
    assert "custom_key" in data
    assert data["custom_key"] == "custom_value"

def test_atomic_data_from_dict():
    """Test creating AtomicData from a dictionary."""
    d = {
        _keys.POSITIONS_KEY: np.array([[0.0, 0.0, 0.0]]),
        _keys.EDGE_INDEX_KEY: np.array([[0], [0]], dtype=np.int64),
        _keys.ATOMIC_NUMBERS_KEY: np.array([1], dtype=np.int64),
        _keys.CELL_KEY: np.eye(3),
        _keys.PBC_KEY: np.array([True, True, True])
    }
    data = AtomicData.from_dict(d)
    assert isinstance(data, AtomicData)
    assert np.array_equal(data[_keys.POSITIONS_KEY], np.array([[0.0, 0.0, 0.0]]))
    assert np.array_equal(data[_keys.ATOMIC_NUMBERS_KEY], np.array([1], dtype=np.int64))
