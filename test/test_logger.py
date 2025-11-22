import pytest
import logging
import os
from dftio.logger import set_log_handles

def test_set_log_handles(tmp_path):
    """Test set_log_handles function."""
    log_file = tmp_path / "test.log"
    
    # Test setting log handles
    set_log_handles(level=logging.DEBUG, log_path=log_file)
    
    # Check if log file is created (it might be created only when logging happens or immediately)
    # The implementation creates the directory immediately.
    # FileHandler opens the file.
    
    logger = logging.getLogger()
    logger.debug("Test message")
    
    assert log_file.exists()
    with open(log_file, "r") as f:
        content = f.read()
        assert "Test message" in content
