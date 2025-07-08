import pytest
import numpy as np
import os
from dftio.io.siesta.siesta_parser import SiestaParser


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)

def test_convert_kpoints_bohr_inv_to_twopi_over_a_scalar():
    # Test with scalar input
    kpoint_bohr_inv = 1.0
    lattice_const_a = 5.0  # Angstrom
    a0 = 0.529177
    expected = kpoint_bohr_inv * lattice_const_a / (a0 * 2 * np.pi)
    result = SiestaParser.convert_kpoints_bohr_inv_to_twopi_over_a(kpoint_bohr_inv, lattice_const_a)
    assert np.isclose(result, expected)

    # Test with 1D array input
    kpoints_bohr_inv = np.array([0.0, 1.0, 2.0])
    lattice_const_a = 3.5
    a0 = 0.529177
    scale = lattice_const_a / (a0 * 2 * np.pi)
    expected = kpoints_bohr_inv * scale
    result = SiestaParser.convert_kpoints_bohr_inv_to_twopi_over_a(kpoints_bohr_inv, lattice_const_a)
    assert np.allclose(result, expected)

    # Test with 2D array input
    kpoints_bohr_inv = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lattice_const_a = 4.2
    a0 = 0.529177
    scale = lattice_const_a / (a0 * 2 * np.pi)
    expected = kpoints_bohr_inv * scale
    result = SiestaParser.convert_kpoints_bohr_inv_to_twopi_over_a(kpoints_bohr_inv, lattice_const_a)
    assert np.allclose(result, expected)

    # Test with 3D array input
    kpoints_bohr_inv = np.array([
        [0.000000, 0.000000, 0.000000],
        [0.407635, 0.000000, 0.000000],
    ])

    lattice_const_a_angstrom = np.array([  [4.0782999992,0.0000000000,0.0000000000],
                                        [0.0000000000,4.0782999992,0.0000000000],
                                        [0.0000000000,0.0000000000,4.0782999992]])

    result = SiestaParser.convert_kpoints_bohr_inv_to_twopi_over_a(kpoints_bohr_inv, lattice_const_a_angstrom)
    diff = np.abs(result - np.array([[0. ,        0. ,        0.    ],
                                     [0.49999977 ,0.    ,     0.    ]]))
    assert np.all(diff < 1e-5), "The conversion did not yield the expected result."


def test_find_content(root_directory):

    path = "test/data/siesta/siesta_out_withband"
    path = os.path.join(root_directory, path)
    assert os.path.exists(path), f"Path does not exist: {path}"

    lattice_ves_path,_ = SiestaParser.find_content(path=path,str_to_find='LatticeVectors')
    assert lattice_ves_path == os.path.join(path, 'STRUCT.fdf')

    struct,_ = SiestaParser.find_content(path= path,str_to_find='AtomicCoordinatesAndAtomicSpecies')
    assert struct == os.path.join(path, 'STRUCT.fdf')

    chemspecis,_ = SiestaParser.find_content(path= path,str_to_find='ChemicalSpeciesLabel')
    assert chemspecis == os.path.join(path, 'STRUCT.fdf')

    log_file,_ = SiestaParser.find_content(path=path,str_to_find='WELCOME',for_Kpt_bands=True)
    assert log_file == os.path.join(path, 'out.log')

    _,system_label = SiestaParser.find_content(path=path, str_to_find='SystemLabel', for_system_label=True)
    assert system_label == "Au_cell"


