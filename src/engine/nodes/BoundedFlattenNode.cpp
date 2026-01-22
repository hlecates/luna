#include "BoundedFlattenNode.h"

namespace NLR {

BoundedFlattenNode::BoundedFlattenNode(const Operations::FlattenWrapper& flatten_module)
    : _flatten_module(flatten_module) {
    _nodeName = "flatten";  // Set default name
    _nodeIndex = 0;
    _input_size = 0;  // Will be set dynamically
    _output_size = 0; // Will be set dynamically
}

// Standard PyTorch forward pass
torch::Tensor BoundedFlattenNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel(); // Flatten preserves total number of elements
    }

    // Use the flatten module's forward method
    torch::Tensor output = _flatten_module.forward(input);

    // Capture input shape for backward propagation (excluding batch dimension if needed, but here we store full shape)
    // Note: The boundBackward logic assumes _input_shape includes batch dimension at index 0?
    // Let's check boundBackward:
    // new_shape.push_back(A.size(0)); // batch
    // new_shape.push_back(A.size(1)); // spec
    // for (size_t i = 1; i < _input_shape.size(); ++i) ...
    // So it assumes _input_shape[0] is batch.

    _input_shape.clear();
    for (int i = 0; i < input.dim(); ++i) {
        _input_shape.push_back(input.size(i));
    }
    
    return output;
}

// Auto-LiRPA style boundBackward method
// Flatten operations are identical to Reshape - they don't change the linear relationships, just pass through A matrices
void BoundedFlattenNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedFlattenNode expects at least one input");
    }

    if (last_lA.isPatches() || last_uA.isPatches()) {
         throw std::runtime_error("BoundedFlattenNode: Patches propagation not implemented (convert to matrix)");
    }
    
    torch::Tensor lA = last_lA.asTensor();
    torch::Tensor uA = last_uA.asTensor();

    // IMPORTANT:
    // In this C++ pipeline, most nodes (Conv/BN/Add/...) operate on *flattened* bounds
    // (e.g. [C*H*W]) and treat spatial structure via local reshape heuristics when needed.
    //
    // Flatten is therefore a no-op for bound propagation: it must NOT reshape A into a higher-rank
    // tensor (e.g. [B,S,C,H,W]) because downstream nodes (especially Add and the CROWN concretizer)
    // expect the last dimension to remain a single "features" axis.
    //
    // Auto_LiRPA's internal (patches) representation is different; here we keep A flattened.
    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(lA), BoundA(uA)));

    torch::Device device = _device;
    auto zero_bias_like_A = [device](const torch::Tensor& A) -> torch::Tensor {
        if (!A.defined()) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
            return torch::zeros({1, 1}, options);
        }
        if (A.dim() == 3) {
            // A format: [spec, batch, features] -> bias [spec, batch]
            // A.size(0) = spec, A.size(1) = batch
            return torch::zeros({A.size(0), A.size(1)}, A.options());
        }
        if (A.dim() == 2) {
            // A format: [spec, features] -> bias [spec, 1] (batch=1)
            return torch::zeros({A.size(0), 1}, A.options());
        }
        // Fallback: [1, 1] to maintain [spec, batch] format
        return torch::zeros({1, 1}, A.options());
    };

    lbias = zero_bias_like_A(lA);
    ubias = zero_bias_like_A(uA);
}



// IBP (Interval Bound Propagation): Fast interval-based bound computation for Flatten
BoundedTensor<torch::Tensor> BoundedFlattenNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("Flatten module requires at least one input");
    }

    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();

    // Apply flatten to both lower and upper bounds
    torch::Tensor flattenedLower = _flatten_module.forward(inputLowerBound);
    torch::Tensor flattenedUpper = _flatten_module.forward(inputUpperBound);

    return BoundedTensor<torch::Tensor>(flattenedLower, flattenedUpper);
}

void NLR::BoundedFlattenNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedFlattenNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
