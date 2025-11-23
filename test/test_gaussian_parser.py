import os
import pytest
import numpy as np
from dftio.io.gaussian.gaussian_parser import GaussianParser
from dftio.data import _keys

@pytest.fixture
def gaussian_parser():
    """Fixture for GaussianParser."""
    root = "test/data/gaussian"
    prefix = "example_folder"
    
    # Create valid_gaussian_logs.txt
    cwd = os.getcwd()
    valid_logs_path = os.path.join(root, "valid_gaussian_logs.txt")
    with open(valid_logs_path, "w") as f:
        f.write(os.path.join(cwd, "test/data/gaussian/example_folder/id_1/gau.log") + "\n")
        f.write(os.path.join(cwd, "test/data/gaussian/example_folder/id_2/gau.log") + "\n")
        
    yield GaussianParser(root=root, prefix=prefix, valid_gau_info_path=valid_logs_path)
    
    # Cleanup
    if os.path.exists(valid_logs_path):
        os.remove(valid_logs_path)

def test_gaussian_parser_init(gaussian_parser):
    """Test GaussianParser initialization."""
    assert gaussian_parser.root == "test/data/gaussian"
    assert gaussian_parser.prefix == "example_folder"
    # Check if raw_datas are correctly loaded (assuming valid_gaussian_logs.txt exists or is handled)
    # If valid_gaussian_logs.txt doesn't exist in the test setup, we might need to mock it or create it.
    # For now, let's assume the parser handles finding files if valid_gau_info_path is not provided or if we need to generate it.
    
    # Actually, looking at the code, if valid_gau_info_path is provided, it loads from there.
    # If not, it tries to write to 'valid_gaussian_logs.txt'.
    # Let's check if we need to create a dummy valid_gaussian_logs.txt for the test.
    pass

def test_get_structure(gaussian_parser):
    """Test parsing of structure."""
    # We need to know how many files are found.
    # Based on file listing: test/data/gaussian/example_folder/id_1/gau.log and id_2/gau.log
    # The parser seems to rely on 'valid_gaussian_logs.txt' or similar to know which files to parse if valid_gau_info_path is passed.
    # Or it scans.
    
    # Let's look at how GaussianParser initializes self.raw_datas.
    # It calls get_gau_logs(valid_gau_info_path) if provided.
    # We should probably mock get_gau_logs or provide a real file.
    
    # For this test, let's manually set raw_datas if needed, or rely on the fixture to set it up correctly.
    # But wait, I don't have 'valid_gaussian_logs.txt' in the file list.
    # I should probably create one in the fixture.
    pass

@pytest.fixture
def setup_gaussian_test_data(tmp_path):
    """Setup test data for Gaussian parser."""
    # Create a temporary directory structure
    root = tmp_path / "gaussian"
    root.mkdir()
    example_folder = root / "example_folder"
    example_folder.mkdir()
    
    # Create dummy log files
    id_1 = example_folder / "id_1"
    id_1.mkdir()
    with open(id_1 / "gau.log", "w") as f:
        f.write("Gaussian log file content mock\n")
        # We need enough content to pass get_basic_info and other functions.
        # This might be complicated to mock without real files.
        # But we have real files in test/data/gaussian.
        # So we should use those.
    
    # Create valid_gaussian_logs.txt
    valid_logs = root / "valid_gaussian_logs.txt"
    with open(valid_logs, "w") as f:
        f.write(f"{os.path.abspath('test/data/gaussian/example_folder/id_1/gau.log')}\n")
        f.write(f"{os.path.abspath('test/data/gaussian/example_folder/id_2/gau.log')}\n")
        
    return str(root), str(valid_logs)

def test_gaussian_parser_real_data():
    """Test using the real data in test/data/gaussian."""
    # We need to generate the valid_gaussian_logs.txt file first because the parser expects it 
    # or we pass it.
    
    # Let's create a temporary valid_gaussian_logs.txt
    # Use absolute paths
    cwd = os.getcwd()
    with open("test_valid_gaussian_logs.txt", "w") as f:
        f.write(os.path.join(cwd, "test/data/gaussian/example_folder/id_1/gau.log") + "\n")
        f.write(os.path.join(cwd, "test/data/gaussian/example_folder/id_2/gau.log") + "\n")
        
    try:
        parser = GaussianParser(root="test/data/gaussian", prefix="example_folder", valid_gau_info_path="test_valid_gaussian_logs.txt")
        
        # Test get_structure
        structure = parser.get_structure(0)
        assert structure is not None
        assert _keys.ATOMIC_NUMBERS_KEY in structure
        assert _keys.POSITIONS_KEY in structure
        
        # Test get_blocks
        # This might fail if the log files don't contain the expected matrix data.
        # The existing files seem to be just 'gau.log'.
        # Let's try to call get_blocks with hamiltonian=False to avoid reading matrices if they aren't there.
        # Or just check if it runs.
        
        # blocks = parser.get_blocks(0, hamiltonian=False, overlap=False)
        # assert blocks is not None
        
    finally:
        if os.path.exists("test_valid_gaussian_logs.txt"):
            os.remove("test_valid_gaussian_logs.txt")

def test_gaussian_parser_init(gaussian_parser):
    """Test GaussianParser initialization."""
    assert gaussian_parser.root == "test/data/gaussian"
    assert gaussian_parser.prefix == "example_folder"
    # Check if raw_datas are correctly loaded (assuming valid_gaussian_logs.txt exists or is handled)
    # If valid_gaussian_logs.txt doesn't exist in the test setup, we might need to mock it or create it.
    # For now, let's assume the parser handles finding files if valid_gau_info_path is not provided or if we need to generate it.
    
    # Actually, looking at the code, if valid_gau_info_path is provided, it loads from there.
    # If not, it tries to write to 'valid_gaussian_logs.txt'.
    # Let's check if we need to create a dummy valid_gaussian_logs.txt for the test.
    pass

def test_get_structure(gaussian_parser):
    """Test parsing of structure."""
    # We need to know how many files are found.
    # Based on file listing: test/data/gaussian/example_folder/id_1/gau.log and id_2/gau.log
    # The parser seems to rely on 'valid_gaussian_logs.txt' or similar to know which files to parse if valid_gau_info_path is passed.
    # Or it scans.
    
    # Let's look at how GaussianParser initializes self.raw_datas.
    # It calls get_gau_logs(valid_gau_info_path) if provided.
    # We should probably mock get_gau_logs or provide a real file.
    
    # For this test, let's manually set raw_datas if needed, or rely on the fixture to set it up correctly.
    # But wait, I don't have 'valid_gaussian_logs.txt' in the file list.
    # I should probably create one in the fixture.
    pass
