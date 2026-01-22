# Import core C++ bindings
try:
    from .LirpaPyCore import LirpaConfiguration
except ImportError as e:
    raise ImportError(
        "Failed to import LirpaPyCore. Make sure the C++ extension is built. "
        "Run 'python setup.py build_ext --inplace' or 'pip install -e .' "
        f"Error: {e}"
    )

# Import high-level Python wrappers
from .lirpapy import TorchModel, BoundedTensor

# Public API exports
__all__ = ['TorchModel', 'BoundedTensor', 'LirpaConfiguration']

__version__ = '1.0.0'
