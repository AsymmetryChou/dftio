import pytest
import torch
import numpy as np
from dftio.datastruct.field import Field

@pytest.fixture
def field_data():
    """Fixture for Field class."""
    data = np.random.rand(10, 10, 10)
    cell = torch.eye(3) * 10
    pos = torch.rand(5, 3) * 10
    atomic_numbers = torch.randint(1, 10, (5,))
    return data, cell, 10, 10, 10, pos, atomic_numbers, (0, 0, 0)

def test_field_init(field_data):
    """Test Field initialization."""
    data, cell, na, nb, nc, pos, atomic_numbers, origin = field_data
    field = Field(
        data=data,
        cell=cell,
        na=na,
        nb=nb,
        nc=nc,
        pos=pos,
        atomic_numbers=atomic_numbers,
        origin=origin
    )
    assert isinstance(field, Field)
    assert field.na == 10
    assert field.nb == 10
    assert field.nc == 10

def test_field_call(field_data):
    """Test calling the Field object."""
    data, cell, na, nb, nc, pos, atomic_numbers, origin = field_data
    field = Field(
        data=data,
        cell=cell,
        na=na,
        nb=nb,
        nc=nc,
        pos=pos,
        atomic_numbers=atomic_numbers,
        origin=origin
    )
    coords = torch.rand(10, 3) * 10
    values = field(coords)
    assert values.shape == (10,)

def test_field_rotate(field_data):
    """Test rotating the field."""
    data, cell, na, nb, nc, pos, atomic_numbers, origin = field_data
    field = Field(
        data=data,
        cell=cell,
        na=na,
        nb=nb,
        nc=nc,
        pos=pos,
        atomic_numbers=atomic_numbers,
        origin=origin
    )
    field.rotate('x', np.pi / 2)
    assert len(field._rot_mat) == 1
    field.reset_rotations()
    assert len(field._rot_mat) == 0

def test_field_set_origin(field_data):
    """Test setting the origin of the field."""
    data, cell, na, nb, nc, pos, atomic_numbers, origin = field_data
    field = Field(
        data=data,
        cell=cell,
        na=na,
        nb=nb,
        nc=nc,
        pos=pos,
        atomic_numbers=atomic_numbers,
        origin=origin
    )
    new_origin = [1, 1, 1]
    field.set_origin(new_origin)
    assert torch.allclose(field._origin_shift, torch.tensor([1.0, 1.0, 1.0]))

# TODO: Add tests for from_cube method once sample .cube files are available.
# It is important to test this method as it is a crucial part of the Field class.
