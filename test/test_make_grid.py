import pytest
import torch
from dftio.op.make_grid import make_simple_grid, make_simple_grid2, make_uniform_grid

def test_make_simple_grid():
    """Test make_simple_grid."""
    cell = torch.eye(3)
    nx, ny, nz = 2, 2, 2
    shape, grid = make_simple_grid(cell, nx, ny, nz)
    
    assert shape == (nx, ny, nz)
    assert grid.shape == (nx * ny * nz, 3)

def test_make_simple_grid2():
    """Test make_simple_grid2."""
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1, y1, z1 = 1.0, 1.0, 1.0
    nx, ny, nz = 2, 2, 2
    shape, grid = make_simple_grid2(x0, y0, z0, x1, y1, z1, nx, ny, nz)
    
    assert shape == (nx, ny, nz)
    assert grid.shape == (nx * ny * nz, 3)

def test_make_uniform_grid():
    """Test make_uniform_grid."""
    cell = torch.eye(3)
    dr = 0.5
    shape, grid = make_uniform_grid(cell, dr)
    
    assert len(shape) == 3
    assert grid.shape[1] == 3
