import pytest
import numpy as np
import torch
from dftio.data.data_np import Data

def test_data_init():
    """Test Data initialization."""
    x = np.array([[1.0]])
    data = Data(x=x)
    assert np.array_equal(data.x, x)

def test_data_from_dict():
    """Test Data.from_dict."""
    d = {"x": np.array([[1.0]])}
    data = Data.from_dict(d)
    assert np.array_equal(data.x, np.array([[1.0]]))

def test_data_properties():
    """Test Data properties."""
    x = np.array([[1.0]])
    data = Data(x=x)
    
    assert data.num_nodes == 1
    assert data.num_features == 1
    assert data.keys == ["x"]
