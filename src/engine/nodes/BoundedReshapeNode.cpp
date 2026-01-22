#include "BoundedReshapeNode.h"

namespace NLR {

BoundedReshapeNode::BoundedReshapeNode(const Operations::ReshapeWrapper& reshape_module) 
    : _reshape_module(reshape_module) {
    _nodeName = "reshape";  // Set default name
    _nodeIndex = 0;
    _input_size = 0;  // Will be set dynamically
    _output_size = 0; // Will be set dynamically
}

// Standard PyTorch forward pass
torch::Tensor BoundedReshapeNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel(); // Reshape preserves total number of elements
    }
    
    // Capture input shape for backward propagation
    _input_shape.clear();
    for (int i = 0; i < input.dim(); ++i) {
        _input_shape.push_back(input.size(i));
    }
    
    // Use the reshape module's forward method
    torch::Tensor output = _reshape_module.forward(input);
    
    return output;
}

// Auto-LiRPA style boundBackward method (NEW)
// Reshape operations don't change the linear relationships, just pass through A matrices
void BoundedReshapeNode::boundBackward(
    const BoundA& last_lA, 
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedReshapeNode expects at least one input");
    }
    
    if (last_lA.isPatches() || last_uA.isPatches()) {
         throw std::runtime_error("BoundedReshapeNode: Patches propagation not implemented (convert to matrix)");
    }
    
    torch::Tensor lA = last_lA.asTensor();
    torch::Tensor uA = last_uA.asTensor();
    
    // Reshape operations don't change the linear relationships
    // We need to reshape the A matrices to match the input shape
    auto _bound_oneside = [&](const torch::Tensor& A) -> torch::Tensor {
        if (!A.defined()) {
            return torch::Tensor();
        }

        // Use the stored input shape
        if (_input_shape.empty()) {
            return A;
        }

        // Build new shape: [batch_spec, output_spec, *input_shape[1:]]
        // A has shape [batch_spec, output_spec, reshaped_dim...]
        // We need to reshape the dimensions after output_spec to match the original input shape
        
        // Assuming A is [batch, spec, ...]
        if (A.dim() < 2) return A;
        
        std::vector<int64_t> new_shape;
        new_shape.push_back(A.size(0));  // batch_spec dimension
        new_shape.push_back(A.size(1));  // output_spec dimension

        // Add input shape dimensions (skip batch dimension at index 0)
        // We assume _input_shape[0] is batch size, which is handled implicitly/separately or matches A.size(0)?
        // Actually A.size(0) is usually batch size.
        // The linear maps from input (excluding batch) to output.
        // A maps output_spec to input.
        
        // If _input_shape includes batch at index 0, we skip it.
        for (size_t i = 1; i < _input_shape.size(); ++i) {
            new_shape.push_back(_input_shape[i]);
        }

        return A.reshape(new_shape);
    };

    // Reshape both A matrices to match input shape
    torch::Tensor reshaped_lA = _bound_oneside(lA);
    torch::Tensor reshaped_uA = _bound_oneside(uA);
    
    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(reshaped_lA), BoundA(reshaped_uA)));
    
    // Reshape operations don't add bias - initialize to zeros with correct size
    if (lA.defined()) {
        // Get the output size from the A matrix
        int output_size = lA.size(1); // Second dimension is output size
        
        if (!lbias.defined()) {
            lbias = torch::zeros({output_size}, lA.options());
        }
    } else {
        if (!lbias.defined()) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
            lbias = torch::zeros({1}, options);
        }
    }
    
    if (uA.defined()) {
        // Get the output size from the A matrix
        int output_size = uA.size(1); // Second dimension is output size
        
        if (!ubias.defined()) {
            ubias = torch::zeros({output_size}, uA.options());
        }
    } else {
        if (!ubias.defined()) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
            ubias = torch::zeros({1}, options);
        }
    }
}



// IBP (Interval Bound Propagation): Fast interval-based bound computation for Reshape
BoundedTensor<torch::Tensor> BoundedReshapeNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("Reshape module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();
    
    // Apply reshape to both lower and upper bounds
    torch::Tensor reshapedLower = _reshape_module.forward(inputLowerBound);
    torch::Tensor reshapedUpper = _reshape_module.forward(inputUpperBound);
    
    return BoundedTensor<torch::Tensor>(reshapedLower, reshapedUpper);
}

void NLR::BoundedReshapeNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedReshapeNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
