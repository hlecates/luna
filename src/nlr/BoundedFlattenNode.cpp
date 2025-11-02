#include "BoundedFlattenNode.h"

namespace NLR {

BoundedFlattenNode::BoundedFlattenNode(const Operations::FlattenWrapper& flatten_module)
    : _flatten_module(flatten_module) {
    _nodeName = "flatten";  // Set default name
    _nodeIndex = 0;
    _input_size = 0;  // Will be set dynamically
    _output_size = 0; // Will be set dynamically
    std::cout << "[BoundedFlattenNode] Constructor called with name: " << _nodeName << std::endl;
}

// Standard PyTorch forward pass
torch::Tensor BoundedFlattenNode::forward(const torch::Tensor& input) {
    std::cout << "[BoundedFlattenNode] Forward called with input shape: " << input.sizes() << std::endl;

    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel(); // Flatten preserves total number of elements
    }

    // Use the flatten module's forward method
    torch::Tensor output = _flatten_module.forward(input);

    std::cout << "[BoundedFlattenNode] Forward output shape: " << output.sizes() << std::endl;
    return output;
}

// Auto-LiRPA style boundBackward method
// Flatten operations are identical to Reshape - they don't change the linear relationships, just pass through A matrices
void BoundedFlattenNode::boundBackward(
    const torch::Tensor& last_lA,
    const torch::Tensor& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<torch::Tensor, torch::Tensor>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    /*
    std::cout << "[BoundedFlattenNode::boundBackward] Starting boundBackward" << std::endl;
    std::cout << "[BoundedFlattenNode::boundBackward] last_lA defined: " << last_lA.defined() << std::endl;
    std::cout << "[BoundedFlattenNode::boundBackward] last_uA defined: " << last_uA.defined() << std::endl;
    std::cout << "[BoundedFlattenNode::boundBackward] lbias defined: " << lbias.defined() << std::endl;
    std::cout << "[BoundedFlattenNode::boundBackward] ubias defined: " << ubias.defined() << std::endl;
    std::cout << "[BoundedFlattenNode::boundBackward] inputBounds size: " << inputBounds.size() << std::endl;
    */
    if (last_lA.defined()) {
        //std::cout << "[BoundedFlattenNode::boundBackward] last_lA shape: " << last_lA.sizes() << std::endl;
    }
    if (last_uA.defined()) {
        //std::cout << "[BoundedFlattenNode::boundBackward] last_uA shape: " << last_uA.sizes() << std::endl;
    }

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedFlattenNode expects at least one input");
    }

    // Flatten operations don't change the linear relationships
    // We need to reshape the A matrices to match the input shape
    // This is exactly like BoundReshape in the Python code: A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])
    auto _bound_oneside = [&](const torch::Tensor& A) -> torch::Tensor {
        if (!A.defined()) {
            return torch::Tensor();
        }

        // Use the stored input shape (excluding batch dimension)
        // If input shape is not set, just pass through A unchanged
        if (_input_shape.empty()) {
            //std::cout << "[BoundedFlattenNode::boundBackward] Warning: input shape not set, passing A through unchanged" << std::endl;
            return A;
        }

        // Build new shape: [batch_spec, output_spec, *input_shape[1:]]
        // A has shape [batch_spec, output_spec, flattened_input_dim]
        // We need to reshape the last dimension to match the original input shape
        std::vector<int64_t> new_shape;
        new_shape.push_back(A.size(0));  // batch_spec dimension
        new_shape.push_back(A.size(1));  // output_spec dimension

        // Add input shape dimensions (skip batch dimension at index 0)
        for (size_t i = 1; i < _input_shape.size(); ++i) {
            new_shape.push_back(_input_shape[i]);
        }

        //std::cout << "[BoundedFlattenNode::boundBackward] Reshaping A from " << A.sizes()<< " to shape: [";
        for (size_t i = 0; i < new_shape.size(); ++i) {
            //std::cout << new_shape[i];
            if (i < new_shape.size() - 1) std::cout << ", ";
        }
        //std::cout << "]" << std::endl;

        return A.reshape(new_shape);
    };

    // Reshape both A matrices to match input shape
    torch::Tensor reshaped_lA = _bound_oneside(last_lA);
    torch::Tensor reshaped_uA = _bound_oneside(last_uA);

    // Pass through the reshaped A matrices
    outputA_matrices.clear();
    outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(reshaped_lA, reshaped_uA));
    //std::cout << "[BoundedFlattenNode::boundBackward] Added reshaped A matrices to output" << std::endl;

    // Flatten operations don't add bias - initialize to zeros with correct size
    if (last_lA.defined()) {
        // Get the output size from the A matrix
        int output_size = last_lA.size(1); // Second dimension is output size
        //std::cout << "[BoundedFlattenNode::boundBackward] Output size from A matrix: " << output_size << std::endl;

        if (!lbias.defined()) {
            lbias = torch::zeros({output_size});
            //std::cout << "[BoundedFlattenNode::boundBackward] Initialized lbias to zeros with size: " << output_size << std::endl;
        }
    } else {
        if (!lbias.defined()) {
            lbias = torch::zeros({1});
            //std::cout << "[BoundedFlattenNode::boundBackward] Initialized lbias to zeros (fallback)" << std::endl;
        }
    }

    if (last_uA.defined()) {
        // Get the output size from the A matrix
        int output_size = last_uA.size(1); // Second dimension is output size
        //std::cout << "[BoundedFlattenNode::boundBackward] Output size from A matrix: " << output_size << std::endl;

        if (!ubias.defined()) {
            ubias = torch::zeros({output_size});
            //std::cout << "[BoundedFlattenNode::boundBackward] Initialized ubias to zeros with size: " << output_size << std::endl;
        }
    } else {
        if (!ubias.defined()) {
            ubias = torch::zeros({1});
            //std::cout << "[BoundedFlattenNode::boundBackward] Initialized ubias to zeros (fallback)" << std::endl;
        }
    }

    //std::cout << "[BoundedFlattenNode::boundBackward] Final lbias shape: " << lbias.sizes() << std::endl;
    //std::cout << "[BoundedFlattenNode::boundBackward] Final ubias shape: " << ubias.sizes() << std::endl;
}



// IBP (Interval Bound Propagation): Fast interval-based bound computation for Flatten
BoundedTensor<torch::Tensor> BoundedFlattenNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    std::cout << "[BoundedFlattenNode::computeIntervalBoundPropagation] Starting IBP computation" << std::endl;

    if (inputBounds.size() < 1) {
        throw std::runtime_error("Flatten module requires at least one input");
    }

    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();

    std::cout << "[BoundedFlattenNode::computeIntervalBoundPropagation] Input lower shape: " << inputLowerBound.sizes() << std::endl;
    std::cout << "[BoundedFlattenNode::computeIntervalBoundPropagation] Input upper shape: " << inputUpperBound.sizes() << std::endl;

    // Apply flatten to both lower and upper bounds
    torch::Tensor flattenedLower = _flatten_module.forward(inputLowerBound);
    torch::Tensor flattenedUpper = _flatten_module.forward(inputUpperBound);

    std::cout << "[BoundedFlattenNode::computeIntervalBoundPropagation] Flattened lower shape: " << flattenedLower.sizes() << std::endl;
    std::cout << "[BoundedFlattenNode::computeIntervalBoundPropagation] Flattened upper shape: " << flattenedUpper.sizes() << std::endl;

    return BoundedTensor<torch::Tensor>(flattenedLower, flattenedUpper);
}

void NLR::BoundedFlattenNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedFlattenNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
