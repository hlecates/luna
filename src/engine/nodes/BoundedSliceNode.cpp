// BoundedSliceNode.cpp - Slice operation with bound propagation
#include "BoundedSliceNode.h"
#include <torch/torch.h>

namespace NLR {

BoundedSliceNode::BoundedSliceNode(int start, int end, int axis, int step, const String& name)
    : _start(start), _end(end), _axis(axis), _step(step), 
      _nodeName(name), _nodeIndex(0), _input_size(0), _output_size(0) {
}

std::pair<int64_t, int64_t> BoundedSliceNode::fixupParams(
    const std::vector<int64_t>& shape, 
    int64_t start, int64_t end, 
    int64_t axis, int64_t step) const {
    
    // Handle negative indices
    if (start < 0) {
        start += shape[axis];
    }
    if (end < 0) {
        // Special case: -9223372036854775807 is -inf in ONNX
        if (end == -9223372036854775807) {
            end = 0;  // only possible when step == -1
        } else {
            end += shape[axis];
        }
    }
    
    // Handle negative step
    if (step == -1) {
        // Swap start and end for negative step
        std::swap(start, end);
        start = start + 1;
    }
    
    // Clamp end to valid range
    end = std::min(end, shape[axis]);
    
    return {start, end};
}

torch::Tensor BoundedSliceNode::forward(const torch::Tensor& input) {
    // Get input shape
    std::vector<int64_t> shape;
    for (int i = 0; i < input.dim(); ++i) {
        shape.push_back(input.size(i));
    }
    
    // Fix up parameters
    auto [start, end] = fixupParams(shape, _start, _end, _axis, _step);
    
    // Perform slice using torch::narrow
    int64_t length = end - start;
    torch::Tensor result = torch::narrow(input, _axis, start, length);
    
    // Handle negative step (flip)
    if (_step == -1) {
        result = torch::flip(result, {_axis});
    }
    
    return result;
}

void BoundedSliceNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {
    
    // Slice backward: pad the A matrix with zeros to restore original input shape
    // Following auto_LiRPA logic: create zero tensor with input shape, then index_copy
    
    auto boundOneside = [&](const BoundA& A) -> BoundA {
        if (!A.defined() || !A.isTensor()) {
            return BoundA();
        }
        
        torch::Tensor A_tensor = A.asTensor();
        
        // Get fixed up parameters
        auto [start, end] = fixupParams(_input_shape, _start, _end, _axis, _step);
        
        // In auto_LiRPA: A has shape [spec, batch, ...] 
        // We need to pad along dimension (axis + 1) to account for spec dimension
        int64_t dim = _axis + 1;
        
        // Create new A with input shape
        // A.shape[:2] + input_shape[1:]
        std::vector<int64_t> new_shape;
        new_shape.push_back(A_tensor.size(0));  // spec dimension
        if (A_tensor.dim() > 1) {
            new_shape.push_back(A_tensor.size(1));  // batch dimension
        }
        
        // Add remaining dimensions from input shape (skip batch dimension)
        for (size_t i = 1; i < _input_shape.size(); ++i) {
            new_shape.push_back(_input_shape[i]);
        }
        
        // Create zero tensor
        torch::Tensor new_A = torch::zeros(
            new_shape, 
            torch::TensorOptions()
                .device(A_tensor.device())
                .dtype(A_tensor.dtype())
                .requires_grad(A_tensor.requires_grad())
        );
        
        // Create indices for the sliced region
        torch::Tensor indices = torch::arange(start, end, torch::TensorOptions().device(A_tensor.device()));
        
        // Use index_copy to place A into the appropriate slice of new_A
        new_A = torch::index_copy(new_A, dim, indices, A_tensor);
        
        return BoundA(new_A);
    };
    
    BoundA lA = boundOneside(last_lA);
    BoundA uA = boundOneside(last_uA);
    
    // Return A matrices for the single input
    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(lA, uA));
    
    // Bias terms pass through unchanged
    auto options = last_lA.defined() && last_lA.isTensor()
        ? last_lA.asTensor().options()
        : (last_uA.defined() && last_uA.isTensor() ? last_uA.asTensor().options()
                                                   : torch::TensorOptions().dtype(torch::kFloat32).device(_device));
    lbias = torch::zeros({1}, options);
    ubias = torch::zeros({1}, options);
}

BoundedTensor<torch::Tensor> BoundedSliceNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.empty()) {
        throw std::runtime_error("BoundedSliceNode: no inputs for IBP");
    }
    
    // Get input bounds
    torch::Tensor lower = inputBounds[0].lower();
    torch::Tensor upper = inputBounds[0].upper();
    
    // Get input shape
    std::vector<int64_t> shape;
    for (int i = 0; i < lower.dim(); ++i) {
        shape.push_back(lower.size(i));
    }
    
    // Adjust axis if tensor dimensions don't match the stored input_shape
    // This handles the case where batch dimension is removed (e.g., [1, 8] → [8])
    int64_t adjusted_axis = _axis;
    if (!_input_shape.empty() && lower.dim() < (int64_t)_input_shape.size()) {
        // Tensor has fewer dimensions than stored shape (batch dim removed)
        int64_t dim_diff = _input_shape.size() - lower.dim();
        adjusted_axis = std::max(0LL, _axis - dim_diff);
    }
    
    // Fix up parameters with adjusted axis
    auto [start, end] = fixupParams(shape, _start, _end, adjusted_axis, _step);
    
    // Apply narrow to both bounds
    int64_t length = end - start;
    torch::Tensor sliced_lower = torch::narrow(lower, adjusted_axis, start, length);
    torch::Tensor sliced_upper = torch::narrow(upper, adjusted_axis, start, length);
    
    // Handle negative step (flip)
    if (_step == -1) {
        sliced_lower = torch::flip(sliced_lower, {_axis});
        sliced_upper = torch::flip(sliced_upper, {_axis});
    }
    
    return BoundedTensor<torch::Tensor>(sliced_lower, sliced_upper);
}

} // namespace NLR
