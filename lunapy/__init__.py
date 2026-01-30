# Import core C++ bindings
try:
    from .LunaPyCore import LunaConfiguration
except ImportError as e:
    raise ImportError(
        "Failed to import LunaPyCore. Make sure the C++ extension is built. "
        "Run 'python setup.py build_ext --inplace' or 'pip install -e .' "
        f"Error: {e}"
    )

# Import high-level Python wrappers
from .lunapy import TorchModel, BoundedTensor

# Public API exports
__all__ = ['TorchModel', 'BoundedTensor', 'LunaConfiguration']

__version__ = '1.0.0'
