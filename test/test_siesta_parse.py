import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import pytest
import numpy as np
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

def test_parse_siesta_output(root_directory):

    prefix = "siesta_out_withband"
    root_directory = os.path.join(root_directory, "test", "data", "siesta")
    parser = SiestaParser(root=root_directory, prefix=prefix)
    assert len(parser.raw_datas) == 1, "Parser should have one raw data entry."

    struct_info = parser.get_structure(0)
    assert np.allclose(struct_info['atomic_numbers'], np.array([79, 79, 79, 79], dtype=np.int32))
    assert np.array_equal(struct_info['pbc'], np.array([True, True, True]))
    assert np.allclose(struct_info['pos'], np.array([[[0.     , 0.     , 0.     ],
                                                      [0.     , 2.03915, 2.03915],
                                                      [2.03915, 0.     , 2.03915],
                                                      [2.03915, 2.03915, 0.     ]]], dtype=np.float32))
    assert np.allclose(struct_info['cell'], np.array([[[4.0783, 0.    , 0.    ],
                                                       [0.    , 4.0783, 0.    ],
                                                       [0.    , 0.    , 4.0783]]], dtype=np.float32))

    eigs_info = parser.get_eigenvalue(0)
    assert np.allclose(eigs_info['kpoint'][0], np.array([0.0, 0.0, 0.0]))
    assert np.allclose(eigs_info['kpoint'][2], np.array([0.05000059, 0.        , 0.05000059]))
    assert np.allclose(eigs_info['kpoint'][11], parser.get_eigenvalue(0)['kpoint'][11])
    assert eigs_info['eigenvalue'].shape == (1, 81, 60)
    assert eigs_info['eigenvalue'][0, 0, 0] == pytest.approx(-11.9863)
    assert eigs_info['eigenvalue'][0, 2, 3] == pytest.approx(-9.1764)
    assert eigs_info['eigenvalue'][0, 10, 20] == pytest.approx(-4.189)

    assert parser.get_basis(0) == {'Au': '2s1p2d'}

    # store H,S,D results in a tuple: ([H1, H2, ...], [S1, S2, ...], [D1, D2, ...])
    assert type(parser.get_blocks(idx=0,hamiltonian=True,overlap=False,density_matrix=False)) == tuple  

    assert type(parser.get_blocks(idx=0,hamiltonian=True,overlap=False,density_matrix=False)[0][0]) == dict
    assert parser.get_blocks(idx=0, hamiltonian=True, overlap=False, density_matrix=False)[0][0]['0_0_0_0_0'].shape == (15, 15)
    assert parser.get_blocks(idx=0, hamiltonian=True, overlap=False, density_matrix=False)[0][0]['2_3_0_-1_0'].shape == (15, 15)
    assert parser.get_blocks(idx=0, hamiltonian=True, overlap=False, density_matrix=False)[0][0]['0_0_0_0_0'][0,1] == pytest.approx(-2.3722997)
    assert parser.get_blocks(idx=0, hamiltonian=True, overlap=False, density_matrix=False)[0][0]['1_2_0_-1_0'][0,1] == pytest.approx(-4.234535e-07)

    assert type(parser.get_blocks(idx=0, hamiltonian=False, overlap=True, density_matrix=False)[1][0]) == dict
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=True, density_matrix=False)[1][0]['0_0_0_0_0'].shape == (15, 15)
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=True, density_matrix=False)[1][0]['2_3_0_-1_0'].shape == (15, 15)
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=True, density_matrix=False)[1][0]['0_0_0_0_0'][0,0] == pytest.approx(1.00000000)
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=True, density_matrix=False)[1][0]['0_0_0_0_0'][0,1] == pytest.approx(0.9065072)
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=True, density_matrix=False)[1][0]['0_0_0_0_0'][0,2] == pytest.approx(0.0)
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=True, density_matrix=False)[1][0]['1_2_0_1_0'][1,0] == pytest.approx(0.07965754)

    assert type(parser.get_blocks(idx=0, hamiltonian=False, overlap=False, density_matrix=True)[2][0]) == dict
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=False, density_matrix=True)[2][0]['0_0_0_0_0'].shape == (15, 15)
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=False, density_matrix=True)[2][0]['2_3_0_-1_0'].shape == (15, 15)
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=False, density_matrix=True)[2][0]['0_0_0_0_0'][0,0] == pytest.approx(0.007212014)
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=False, density_matrix=True)[2][0]['0_0_0_0_0'][0,1] == pytest.approx(0.00013299382)
    assert parser.get_blocks(idx=0, hamiltonian=False, overlap=False, density_matrix=True)[2][0]['1_2_0_1_0'][1,0] == pytest.approx(0.000278148)
