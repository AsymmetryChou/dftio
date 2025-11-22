import os
import pytest
import numpy as np
from dftio.io.pyatb.pyatb_parser import PyatbParser
from dftio.data import _keys

@pytest.fixture
def pyatb_parser(tmp_path):
    """Fixture for PyatbParser with mock data."""
    # Create a temporary directory structure
    root = tmp_path / "pyatb_calc"
    root.mkdir()
    
    # Create pyatb directory structure
    pyatb_dir = root / "pyatb"
    pyatb_dir.mkdir()
    
    # Create mock STRU file (Abacus format)
    # The error "ValueError: invalid literal for int() with base ..." suggests that something expected to be an int is not.
    # In pyatb_parser.py:
    # structure = {
    #     _keys.ATOMIC_NUMBERS_KEY: np.array([ase.atom.atomic_numbers[i] for i in sys.data["atom_names"]], dtype=np.int32)[sys.data["atom_types"]],
    # }
    # It uses dpdata to read the STRU file.
    # dpdata might be failing to parse my mock STRU if it's not perfect.
    # Or maybe the atom types are not integers?
    
    # Let's look at the mock STRU again.
    # ATOMIC_SPECIES
    # Si 28.0855 Si.upf
    # ...
    # ATOMIC_POSITIONS
    # Direct
    # Si
    # 0.0 0.0 0.0 1 1 1
    
    # The error might be coming from dpdata parsing.
    # "ValueError: invalid literal for int() with base 10: 'Si'"?
    # Or maybe "1.0" where int is expected?
    
    # Let's try to make the mock STRU more standard.
    # Maybe "1 1 1" (move flags) should be integers? They are.
    
    # The error trace would be helpful but I only have the summary.
    # ERROR test/test_pyatb_parser.py::test_pyatb_parser_init - ValueError: invalid literal for int() with base ...
    # This happens in init?
    # self.raw_sys = [dpdata.System(..., fmt="abacus/stru") ...]
    # So yes, dpdata parsing fails.
    
    # Let's try to use a simpler STRU format or ensure it matches exactly what dpdata expects.
    # Maybe remove the move flags if they are optional or ensure they are parsed correctly.
    # Or maybe the LATTICE_CONSTANT should be just a number.
    
    with open(pyatb_dir / "STRU", "w") as f:
        f.write("ATOMIC_SPECIES\n")
        f.write("Si 28.0855 Si.upf\n")
        f.write("LATTICE_CONSTANT\n")
        f.write("10.2\n") 
        f.write("LATTICE_VECTORS\n")
        f.write("0.5 0.5 0.0\n")
        f.write("0.5 0.0 0.5\n")
        f.write("0.0 0.5 0.5\n")
        f.write("ATOMIC_POSITIONS\n")
        f.write("Direct\n")
        f.write("Si\n")
        f.write("0.0\n") # Magnetism? No.
        f.write("2\n") # Number of atoms
        f.write("0.0 0.0 0.0 1 1 1\n")
        f.write("0.25 0.25 0.25 1 1 1\n")

    # Create Out/Band_Structure directory
    out_dir = pyatb_dir / "Out"
    out_dir.mkdir()
    band_dir = out_dir / "Band_Structure"
    band_dir.mkdir()
    
    # Create mock band.dat
    # Shape: [nk, nbands]
    # Let's say 2 kpoints, 4 bands
    band_data = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [0.1, 1.1, 2.1, 3.1]
    ])
    np.savetxt(band_dir / "band.dat", band_data)
    
    # Create mock kpt.dat
    # Shape: [nk, 3]
    kpt_data = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0]
    ])
    np.savetxt(band_dir / "kpt.dat", kpt_data)
    
    return PyatbParser(root=str(tmp_path), prefix="pyatb_calc")

def test_pyatb_parser_init(pyatb_parser, tmp_path):
    """Test PyatbParser initialization."""
    assert pyatb_parser.root == str(tmp_path)
    assert pyatb_parser.prefix == "pyatb_calc"
    assert len(pyatb_parser.raw_datas) == 1

def test_get_structure(pyatb_parser):
    """Test parsing of structure."""
    structure = pyatb_parser.get_structure(0)
    assert structure is not None
    assert _keys.ATOMIC_NUMBERS_KEY in structure
    assert _keys.POSITIONS_KEY in structure
    assert _keys.CELL_KEY in structure
    assert _keys.PBC_KEY in structure
    assert len(structure[_keys.ATOMIC_NUMBERS_KEY]) == 2
    assert structure[_keys.ATOMIC_NUMBERS_KEY][0] == 14 # Si

def test_get_eigenvalue(pyatb_parser):
    """Test parsing of eigenvalues."""
    eigenvalues = pyatb_parser.get_eigenvalue(0)
    assert eigenvalues is not None
    assert _keys.ENERGY_EIGENVALUE_KEY in eigenvalues
    assert _keys.KPOINT_KEY in eigenvalues
    
    # Check shapes
    # eigs: [1, nk, nbands] -> [1, 2, 4]
    assert eigenvalues[_keys.ENERGY_EIGENVALUE_KEY].shape == (1, 2, 4)
    assert eigenvalues[_keys.KPOINT_KEY].shape == (2, 3)
