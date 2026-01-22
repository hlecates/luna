from .LirpaPyCore import TorchModel as _CoreTorchModel
from .LirpaPyCore import BoundedTensor as _CoreBoundedTensor
from .LirpaPyCore import LirpaConfiguration
import numpy as np

__all__ = ['TorchModel', 'BoundedTensor', 'LirpaConfiguration']


class BoundedTensor:
    """
    Represents lower and upper bounds for a tensor.
    
    Attributes:
        lower: Lower bounds as NumPy array
        upper: Upper bounds as NumPy array
    """
    
    def __init__(self, lower=None, upper=None, _core_obj=None):
        """
        Create a BoundedTensor.
        
        Args:
            lower: Lower bounds (NumPy array)
            upper: Upper bounds (NumPy array)
            _core_obj: Internal C++ BoundedTensor object (for internal use)
        """
        if _core_obj is not None:
            self._core = _core_obj
        elif lower is not None and upper is not None:
            lower = np.asarray(lower, dtype=np.float32)
            upper = np.asarray(upper, dtype=np.float32)
            self._core = _CoreBoundedTensor(lower, upper)
        else:
            self._core = _CoreBoundedTensor()
    
    def lower(self):
        """Get lower bounds as NumPy array."""
        return self._core.lower()
    
    def upper(self):
        """Get upper bounds as NumPy array."""
        return self._core.upper()
    
    def __repr__(self):
        return f"<BoundedTensor shape={self.lower().shape}>"


class TorchModel:
    """
    Neural network model with bound propagation support.
    
    This class wraps the C++ TorchModel and provides methods for:
    - Loading ONNX models
    - Setting input bounds
    - Computing certified output bounds using CROWN and Alpha-CROWN
    """
    
    def __init__(self, onnx_path, vnnlib_path=None):
        """
        Create a TorchModel from an ONNX file.
        
        Args:
            onnx_path: Path to ONNX model file
            vnnlib_path: Optional path to VNN-LIB file containing input bounds
                        and output specifications
        
        Example:
            >>> model = TorchModel("model.onnx", "spec.vnnlib")
            >>> bounds = model.compute_bounds(method='CROWN')
            >>> print(bounds.lower(), bounds.upper())
        """
        if vnnlib_path is not None:
            self._core = _CoreTorchModel(onnx_path, vnnlib_path)
        else:
            self._core = _CoreTorchModel(onnx_path)
    
    def getInputSize(self):
        """Get the total number of input elements."""
        return self._core.getInputSize()
    
    def getOutputSize(self):
        """Get the total number of output elements."""
        return self._core.getOutputSize()
    
    def getNumNodes(self):
        """Get the number of nodes in the computational graph."""
        return self._core.getNumNodes()
    
    def setInputBounds(self, lower, upper):
        """
        Set input bounds for the model.
        
        Args:
            lower: Lower bounds (NumPy array or list)
            upper: Upper bounds (NumPy array or list)
        
        Example:
            >>> model = TorchModel("model.onnx")
            >>> model.setInputBounds(
            ...     lower=np.zeros(784),
            ...     upper=np.ones(784)
            ... )
        """
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)
        self._core.setInputBounds(lower, upper)
    
    def getInputBounds(self):
        """
        Get the current input bounds.
        
        Returns:
            BoundedTensor with lower and upper bounds
        """
        core_bt = self._core.getInputBounds()
        return BoundedTensor(_core_obj=core_bt)
    
    def hasInputBounds(self):
        """Check if input bounds have been set."""
        return self._core.hasInputBounds()
    
    def forward(self, input_data):
        """
        Perform forward pass through the model.
        
        Args:
            input_data: Input tensor (NumPy array)
        
        Returns:
            Output tensor (NumPy array)
        """
        input_data = np.asarray(input_data, dtype=np.float32)
        return self._core.forward(input_data)
    
    def setSpecificationMatrix(self, C):
        """
        Set output specification matrix.
        
        The specification matrix C transforms the output: C @ output
        This is useful for verifying properties about specific output combinations.
        
        Args:
            C: Specification matrix (NumPy array)
        """
        C = np.asarray(C, dtype=np.float32)
        self._core.setSpecificationMatrix(C)
    
    def hasSpecificationMatrix(self):
        """Check if a specification matrix has been set."""
        return self._core.hasSpecificationMatrix()
    
    def compute_bounds(self, method='CROWN', bound_lower=True, 
                      bound_upper=True, C=None):
        """
        Compute certified output bounds using the specified method.
        
        Args:
            method: Analysis method - 'CROWN' or 'alpha-CROWN' (default: 'CROWN')
            bound_lower: Compute lower bounds (default: True)
            bound_upper: Compute upper bounds (default: True)
            C: Optional specification matrix (NumPy array)
        
        Returns:
            BoundedTensor containing lower and upper bounds
        
        Example:
            >>> model = TorchModel("model.onnx", "spec.vnnlib")
            >>> bounds = model.compute_bounds(method='CROWN')
            >>> print("Lower bounds:", bounds.lower())
            >>> print("Upper bounds:", bounds.upper())
            
            >>> # With custom specification matrix
            >>> C = np.eye(10)
            >>> bounds = model.compute_bounds(method='alpha-CROWN', C=C)
        """
        if C is not None:
            C = np.asarray(C, dtype=np.float32)
        
        core_bt = self._core.compute_bounds(
            method=method,
            bound_lower=bound_lower,
            bound_upper=bound_upper,
            C=C
        )
        return BoundedTensor(_core_obj=core_bt)
    
    def runCROWN(self):
        """
        Run CROWN analysis directly.
        
        Returns:
            BoundedTensor with computed bounds
        """
        core_bt = self._core.runCROWN()
        return BoundedTensor(_core_obj=core_bt)
    
    def runAlphaCROWN(self, optimizeLower=True, optimizeUpper=False):
        """
        Run Alpha-CROWN analysis directly.
        
        Args:
            optimizeLower: Optimize lower bounds (default: True)
            optimizeUpper: Optimize upper bounds (default: False)
        
        Returns:
            BoundedTensor with computed bounds
        """
        core_bt = self._core.runAlphaCROWN(optimizeLower, optimizeUpper)
        return BoundedTensor(_core_obj=core_bt)
    
    def getFinalAnalysisBounds(self):
        """
        Get bounds from the most recent analysis.
        
        Returns:
            BoundedTensor with final bounds
        """
        core_bt = self._core.getFinalAnalysisBounds()
        return BoundedTensor(_core_obj=core_bt)
    
    def hasFinalAnalysisBounds(self):
        """Check if final analysis bounds are available."""
        return self._core.hasFinalAnalysisBounds()
    
    def __repr__(self):
        return (f"<TorchModel nodes={self.getNumNodes()}, "
                f"input_size={self.getInputSize()}, "
                f"output_size={self.getOutputSize()}>")
