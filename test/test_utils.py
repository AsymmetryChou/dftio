import pytest
from dftio.utils import j_must_have

def test_j_must_have():
    """Test j_must_have function."""
    data = {"key1": "value1", "key2": "value2"}
    
    # Test existing key
    assert j_must_have(data, "key1") == "value1"
    
    # Test deprecated key
    assert j_must_have(data, "new_key", deprecated_key=["key2"]) == "value2"
    
    # Test missing key
    with pytest.raises(RuntimeError):
        j_must_have(data, "missing_key")
