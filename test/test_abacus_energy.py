import os
import pytest
import numpy as np
from dftio.io.abacus.abacus_parser import AbacusParser
from dftio.data import _keys


def test_abacus_scf_energy():
    """Test energy extraction for SCF calculation."""
    # Use the existing test data
    test_data_dir = os.path.join(os.path.dirname(__file__), "data", "abacus_scf")

    parser = AbacusParser(
        root=[test_data_dir],
        prefix=""
    )

    # Test get_etot method
    energy_data = parser.get_etot(idx=0)

    assert energy_data is not None, "Energy data should not be None"
    assert _keys.TOTAL_ENERGY_KEY in energy_data, f"Energy data should contain {_keys.TOTAL_ENERGY_KEY}"

    energy = energy_data[_keys.TOTAL_ENERGY_KEY]

    # Check shape - should be [1,] for SCF
    assert energy.shape == (1,), f"Expected shape (1,), got {energy.shape}"

    # Check dtype
    assert energy.dtype == np.float64, f"Expected dtype float64, got {energy.dtype}"

    # Check value (from the log file: -1879.7169812 eV)
    expected_energy = -1879.7169812
    assert np.isclose(energy[0], expected_energy, atol=1e-5), \
        f"Expected energy ~{expected_energy}, got {energy[0]}"

    print(f"SCF energy extraction test passed: {energy[0]} eV")


def test_abacus_energy_write_dat():
    """Test energy writing to dat format."""
    import tempfile
    import shutil

    test_data_dir = os.path.join(os.path.dirname(__file__), "data", "abacus_scf")

    parser = AbacusParser(
        root=[test_data_dir],
        prefix=""
    )

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        parser.write(
            idx=0,
            outroot=tmpdir,
            format="dat",
            eigenvalue=False,
            hamiltonian=False,
            overlap=False,
            density_matrix=False,
            band_index_min=0,
            energy=True
        )

        # Check if energy file was created
        output_dir = os.path.join(tmpdir, parser.formula(idx=0) + ".0")
        energy_file = os.path.join(output_dir, "total_energy.npy")

        assert os.path.exists(energy_file), f"Energy file should exist at {energy_file}"

        # Load and verify energy
        loaded_energy = np.load(energy_file)
        assert loaded_energy.shape == (1,), f"Expected shape (1,), got {loaded_energy.shape}"
        assert np.isclose(loaded_energy[0], -1879.7169812, atol=1e-5), \
            f"Loaded energy doesn't match expected value"

        print(f"DAT format energy write test passed")


def test_abacus_energy_write_lmdb():
    """Test energy writing to LMDB format."""
    import tempfile
    import lmdb
    import pickle

    test_data_dir = os.path.join(os.path.dirname(__file__), "data", "abacus_scf")

    parser = AbacusParser(
        root=[test_data_dir],
        prefix=""
    )

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        parser.write(
            idx=0,
            outroot=tmpdir,
            format="lmdb",
            eigenvalue=False,
            hamiltonian=False,
            overlap=False,
            density_matrix=False,
            band_index_min=0,
            energy=True
        )

        # Check if LMDB was created
        lmdb_path = os.path.join(tmpdir, f"data.{os.getpid()}.lmdb")
        assert os.path.exists(lmdb_path), f"LMDB should exist at {lmdb_path}"

        # Open and verify energy in LMDB
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            data = txn.get((0).to_bytes(length=4, byteorder='big'))
            assert data is not None, "LMDB should contain data"

            data_dict = pickle.loads(data)
            assert _keys.TOTAL_ENERGY_KEY in data_dict, \
                f"LMDB data should contain {_keys.TOTAL_ENERGY_KEY}"

            energy = data_dict[_keys.TOTAL_ENERGY_KEY]
            assert np.isclose(energy, -1879.7169812, atol=1e-5), \
                f"LMDB energy doesn't match expected value"

        env.close()
        print(f"LMDB format energy write test passed")


def test_abacus_md_energy():
    """Test energy extraction for MD calculation."""
    test_data_dir = os.path.join(os.path.dirname(__file__), "data", "abacus_md")

    parser = AbacusParser(
        root=[test_data_dir],
        prefix=""
    )

    # Test get_etot method
    energy_data = parser.get_etot(idx=0)

    assert energy_data is not None, "Energy data should not be None"
    assert _keys.TOTAL_ENERGY_KEY in energy_data, f"Energy data should contain {_keys.TOTAL_ENERGY_KEY}"

    energy = energy_data[_keys.TOTAL_ENERGY_KEY]

    # Check shape - should be [nframes,] for MD (11 frames in test data)
    expected_nframes = 11
    assert energy.shape == (expected_nframes,), f"Expected shape ({expected_nframes},), got {energy.shape}"

    # Check dtype
    assert energy.dtype == np.float64, f"Expected dtype float64, got {energy.dtype}"

    # Check first few energy values from the log file
    expected_energies = np.array([
        -845.88136814,
        -845.88023386,
        -845.87688911,
        -845.87146567,
        -845.86416254,
        -845.85525948,
        -845.84511407,
        -845.83414351,
        -845.82280254,
        -845.81156302,
        -845.80089500
    ], dtype=np.float64)

    assert np.allclose(energy, expected_energies, atol=1e-5), \
        f"MD energies don't match expected values.\nExpected:\n{expected_energies}\nGot:\n{energy}"

    print(f"MD energy extraction test passed: {energy.shape[0]} frames extracted")


def test_abacus_relax_energy():
    """Test energy extraction for RELAX calculation."""
    test_data_dir = os.path.join(os.path.dirname(__file__), "data", "abacus_relax")

    parser = AbacusParser(
        root=[test_data_dir],
        prefix=""
    )

    # Test get_etot method
    energy_data = parser.get_etot(idx=0)

    assert energy_data is not None, "Energy data should not be None"
    assert _keys.TOTAL_ENERGY_KEY in energy_data, f"Energy data should contain {_keys.TOTAL_ENERGY_KEY}"

    energy = energy_data[_keys.TOTAL_ENERGY_KEY]

    # Check shape - should be [nframes,] for RELAX (4 frames in test data)
    expected_nframes = 1
    assert energy.shape == (expected_nframes,), f"Expected shape ({expected_nframes},), got {energy.shape}"

    # Check dtype
    assert energy.dtype == np.float64, f"Expected dtype float64, got {energy.dtype}"

    # Check energy values from the log file
    expected_energies = np.array([
        -208.06746292
    ], dtype=np.float64)

    assert np.allclose(energy, expected_energies, atol=1e-5), \
        f"RELAX energies don't match expected values.\nExpected:\n{expected_energies}\nGot:\n{energy}"

    print(f"RELAX energy extraction test passed: {energy.shape[0]} frames extracted")


if __name__ == "__main__":
    test_abacus_scf_energy()
    test_abacus_energy_write_dat()
    test_abacus_energy_write_lmdb()
    test_abacus_md_energy()
    test_abacus_relax_energy()
    print("\nAll energy extraction tests passed!")
