"""
Simple installation verification test

This test can be run immediately after installation to verify
that lirpapy is working correctly.
"""

import pytest


def test_import_lirpapy():
    """Test that lirpapy can be imported"""
    try:
        import lirpapy
        assert lirpapy is not None
    except ImportError as e:
        pytest.fail(f"Failed to import lirpapy: {e}")


def test_import_core_components():
    """Test importing core components"""
    from lirpapy import TorchModel, BoundedTensor, LirpaConfiguration
    
    assert TorchModel is not None
    assert BoundedTensor is not None
    assert LirpaConfiguration is not None


def test_create_bounded_tensor():
    """Test creating a simple BoundedTensor"""
    import numpy as np
    from lirpapy import BoundedTensor
    
    lower = np.array([0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0], dtype=np.float32)
    
    bt = BoundedTensor(lower, upper)
    
    assert bt is not None
    assert bt.lower() is not None
    assert bt.upper() is not None


def test_version():
    """Test that version is accessible"""
    import lirpapy
    
    assert hasattr(lirpapy, '__version__')
    assert isinstance(lirpapy.__version__, str)


def test_configuration_accessible():
    """Test that LirpaConfiguration is accessible"""
    from lirpapy import LirpaConfiguration
    
    # Should be able to read configuration values
    assert hasattr(LirpaConfiguration, 'VERBOSE')
    assert hasattr(LirpaConfiguration, 'ALPHA_ITERATIONS')
    assert hasattr(LirpaConfiguration, 'ALPHA_LR')


if __name__ == '__main__':
    """Run installation tests"""
    print("Running lirpapy installation tests...")
    
    try:
        test_import_lirpapy()
        print("✓ Import test passed")
        
        test_import_core_components()
        print("✓ Core components import test passed")
        
        test_create_bounded_tensor()
        print("✓ BoundedTensor creation test passed")
        
        test_version()
        print("✓ Version test passed")
        
        test_configuration_accessible()
        print("✓ Configuration test passed")
        
        print("\n" + "="*50)
        print("All installation tests passed!")
        print("lirpapy is installed correctly.")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Installation test failed: {e}")
        print("\nPlease ensure lirpapy is built correctly:")
        print("  pip install -e .")
        print("or")
        print("  python setup.py build_ext --inplace")
        raise
