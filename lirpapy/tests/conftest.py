"""
Pytest configuration and fixtures for lirpapy tests
"""

import pytest
import numpy as np
from pathlib import Path

# Get paths to test resources
REPO_ROOT = Path(__file__).parent.parent.parent
RESOURCES_DIR = REPO_ROOT / "resources"
ONNX_DIR = RESOURCES_DIR / "onnx"
PROPERTIES_DIR = RESOURCES_DIR / "properties"


@pytest.fixture
def spec_test_onnx():
    """Path to spec_test.onnx"""
    return str(ONNX_DIR / "spec_test.onnx")


@pytest.fixture
def spec_test_vnnlib():
    """Path to spec_test.vnnlib"""
    return str(PROPERTIES_DIR / "spec_test.vnnlib")


@pytest.fixture
def simple_input_bounds():
    """Simple input bounds for testing"""
    return {
        'lower': np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        'upper': np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }


@pytest.fixture
def check_resources():
    """Check if test resources are available"""
    if not ONNX_DIR.exists():
        pytest.skip(f"ONNX directory not found: {ONNX_DIR}")
    if not PROPERTIES_DIR.exists():
        pytest.skip(f"Properties directory not found: {PROPERTIES_DIR}")
