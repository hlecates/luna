// BoundedConcatNode.cpp - Concatenation with bound propagation
#include "BoundedConcatNode.h"
#include <torch/torch.h>
#include <cmath>
#include <fstream>
#include <chrono>

namespace NLR {

BoundedConcatNode::BoundedConcatNode(int axis, unsigned numInputs, const String& name)
    : _axis(axis), _numInputs(numInputs), _nodeName(name), _nodeIndex(0), 
      _input_size(0), _output_size(0) {
}

torch::Tensor BoundedConcatNode::forward(const torch::Tensor& input) {
    // Single input case - just return it
    return input;
}

torch::Tensor BoundedConcatNode::forward(const std::vector<torch::Tensor>& inputs) {
    // #region agent log
    { std::ofstream log("/Users/hlecates/Desktop/autolirpa/.cursor/debug.log", std::ios::app); log << "{\"location\":\"BoundedConcatNode.cpp:18\",\"message\":\"forward entry\",\"data\":{\"num_inputs\":" << inputs.size() << ",\"axis\":" << _axis << ",\"node_name\":\"" << _nodeName.ascii() << "\"},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\"sessionId\":\"debug-session\",\"hypothesisId\":\"FORWARD\"}\n"; }
    // #endregion
    
    std::cerr << "[BoundedConcatNode::forward] Called with " << inputs.size() << " inputs" << std::endl;
    
    if (inputs.empty()) {
        throw std::runtime_error("BoundedConcatNode::forward - no inputs provided");
    }
    
    if (inputs.size() == 1) {
        std::cerr << "[BoundedConcatNode::forward] Single input, returning as-is" << std::endl;
        return inputs[0];
    }
    
    // Print input shapes
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::cerr << "[BoundedConcatNode::forward] Input " << i << " shape: [";
        std::string shape_str = "[";
        for (int j = 0; j < inputs[i].dim(); ++j) {
            std::cerr << inputs[i].size(j);
            shape_str += std::to_string(inputs[i].size(j));
            if (j < inputs[i].dim() - 1) {
                std::cerr << ", ";
                shape_str += ",";
            }
        }
        std::cerr << "]" << std::endl;
        shape_str += "]";
        // #region agent log
        { std::ofstream log("/Users/hlecates/Desktop/autolirpa/.cursor/debug.log", std::ios::app); log << "{\"location\":\"BoundedConcatNode.cpp:34\",\"message\":\"input tensor shape\",\"data\":{\"input_idx\":" << i << ",\"shape\":\"" << shape_str << "\",\"ndim\":" << inputs[i].dim() << "},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\"sessionId\":\"debug-session\",\"hypothesisId\":\"FORWARD\"}\n"; }
        // #endregion
    }
    
    // Concatenate all inputs along the specified axis
    torch::Tensor result = torch::cat(inputs, _axis);
    std::cerr << "[BoundedConcatNode::forward] Output shape: [";
    std::string output_shape_str = "[";
    for (int j = 0; j < result.dim(); ++j) {
        std::cerr << result.size(j);
        output_shape_str += std::to_string(result.size(j));
        if (j < result.dim() - 1) {
            std::cerr << ", ";
            output_shape_str += ",";
        }
    }
    std::cerr << "]" << std::endl;
    output_shape_str += "]";
    // #region agent log
    { std::ofstream log("/Users/hlecates/Desktop/autolirpa/.cursor/debug.log", std::ios::app); log << "{\"location\":\"BoundedConcatNode.cpp:50\",\"message\":\"forward output\",\"data\":{\"output_shape\":\"" << output_shape_str << "\",\"output_ndim\":" << result.dim() << ",\"_input_size\":" << _input_size << ",\"_output_size\":" << _output_size << "},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\"sessionId\":\"debug-session\",\"hypothesisId\":\"FORWARD\"}\n"; }
    // #endregion
    
    return result;
}

void BoundedConcatNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {
    
    // Concatenation backward: split the A matrix along concat axis
    // and distribute chunks to each input
    
    if (inputBounds.size() != _numInputs) {
        throw std::runtime_error("BoundedConcatNode: input count mismatch");
    }
    
    // Initialize output A matrices for each input
    outputA_matrices.clear();
    for (unsigned i = 0; i < _numInputs; ++i) {
        outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(), BoundA()));
    }
    
    // Split A matrices along the concatenation axis
    // Following auto_LiRPA logic: split(A, input_size, dim=axis+1)
    auto splitA = [&](const BoundA& A) -> std::vector<BoundA> {
        std::vector<BoundA> result;
        
        if (!A.defined() || !A.isTensor()) {
            // Return empty BoundA for each input
            for (unsigned i = 0; i < _numInputs; ++i) {
                result.push_back(BoundA());
            }
            return result;
        }
        
        torch::Tensor A_tensor = A.asTensor();
        
        // In auto_LiRPA, A matrix is split along dimension (axis + 1)
        // But we need to adjust for the actual dimensions of A_tensor
        // A_tensor might be [spec, ...] without explicit batch dimension
        int split_dim = _axis + 1;
        
        // Adjust split_dim if it's out of range (batch dim might be missing)
        if (split_dim >= A_tensor.dim() && split_dim > 0) {
            split_dim = _axis;  // Try without the +1 offset
        }
        
        // Get sizes along split dimension for each input
        std::vector<int64_t> split_sizes;
        int64_t total_size = A_tensor.size(split_dim);
        
        if (_input_sizes.empty()) {
            // Equal split if sizes not provided
            int64_t chunk_size = total_size / _numInputs;
            for (unsigned i = 0; i < _numInputs; ++i) {
                split_sizes.push_back(chunk_size);
            }
        } else {
            // Use provided input sizes (sizes along concat axis)
            int64_t sum_sizes = 0;
            for (unsigned size : _input_sizes) {
                sum_sizes += size;
            }
            
            // Check if input_sizes match the actual A tensor dimension
            if (sum_sizes == total_size) {
                // Perfect match - use input_sizes directly
                for (unsigned size : _input_sizes) {
                    split_sizes.push_back(static_cast<int64_t>(size));
                }
            } else {
                // Mismatch - split proportionally based on input_sizes ratios
                // This happens when the A matrix represents a subset/projection of the output
                double scale = static_cast<double>(total_size) / sum_sizes;
                int64_t accumulated = 0;
                for (unsigned i = 0; i < _input_sizes.size(); ++i) {
                    if (i == _input_sizes.size() - 1) {
                        // Last chunk gets remainder to ensure exact sum
                        split_sizes.push_back(total_size - accumulated);
                    } else {
                        int64_t chunk = static_cast<int64_t>(std::round(_input_sizes[i] * scale));
                        split_sizes.push_back(chunk);
                        accumulated += chunk;
                    }
                }
            }
        }
        
        // Split the tensor
        auto chunks = torch::split(A_tensor, split_sizes, split_dim);
        
        for (const auto& chunk : chunks) {
            result.push_back(BoundA(chunk));
        }
        
        return result;
    };
    
    std::vector<BoundA> lA_chunks = splitA(last_lA);
    std::vector<BoundA> uA_chunks = splitA(last_uA);
    
    // Assign chunks to output matrices
    for (unsigned i = 0; i < _numInputs && i < lA_chunks.size(); ++i) {
        outputA_matrices[i] = Pair<BoundA, BoundA>(lA_chunks[i], uA_chunks[i]);
    }
    
    // Bias terms pass through unchanged
    auto options = last_lA.defined() && last_lA.isTensor()
        ? last_lA.asTensor().options()
        : (last_uA.defined() && last_uA.isTensor() ? last_uA.asTensor().options()
                                                   : torch::TensorOptions().dtype(torch::kFloat32).device(_device));
    lbias = torch::zeros({1}, options);
    ubias = torch::zeros({1}, options);
}

BoundedTensor<torch::Tensor> BoundedConcatNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    // #region agent log
    { std::ofstream log("/Users/hlecates/Desktop/autolirpa/.cursor/debug.log", std::ios::app); log << "{\"location\":\"BoundedConcatNode.cpp:163\",\"message\":\"IBP entry\",\"data\":{\"numInputBounds\":" << inputBounds.size() << ",\"expected_numInputs\":" << _numInputs << ",\"axis\":" << _axis << "},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\"sessionId\":\"debug-session\",\"hypothesisId\":\"IBP_ENTRY\"}\n"; }
    // #endregion
    
    if (inputBounds.size() != _numInputs) {
        throw std::runtime_error("BoundedConcatNode: IBP input count mismatch");
    }
    
    if (inputBounds.empty()) {
        throw std::runtime_error("BoundedConcatNode: no inputs for IBP");
    }
    
    // Single input case
    if (inputBounds.size() == 1) {
        return inputBounds[0];
    }
    
    // Collect lower and upper bounds from all inputs
    std::vector<torch::Tensor> lower_bounds;
    std::vector<torch::Tensor> upper_bounds;
    
    for (unsigned i = 0; i < inputBounds.size(); ++i) {
        lower_bounds.push_back(inputBounds[i].lower());
        upper_bounds.push_back(inputBounds[i].upper());
    }
    
    // Make axis non-negative and adjust for actual tensor dimensions
    // In auto_LiRPA, axis is made non-negative in forward() before concatenation
    // The axis refers to the dimension in the actual tensor (which may not have batch dim)
    int64_t axis = _axis;
    if (!lower_bounds.empty()) {
        int64_t ndim = lower_bounds[0].dim();
        
        // #region agent log
        { std::ofstream log("/Users/hlecates/Desktop/autolirpa/.cursor/debug.log", std::ios::app); log << "{\"location\":\"BoundedConcatNode.cpp:194\",\"message\":\"before axis adjustment\",\"data\":{\"original_axis\":" << _axis << ",\"ndim\":" << ndim << ",\"axis_negative\":" << (axis < 0 ? "true" : "false") << "},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\"sessionId\":\"debug-session\",\"hypothesisId\":\"A\"}\n"; }
        // #endregion
        
        // Make axis non-negative
        if (axis < 0) {
            axis += ndim;
        }
        
        // If axis is still out of range, it might be because the batch dimension
        // was removed. Try adjusting by -1 if axis > 0
        if (axis >= ndim && axis > 0) {
            axis = axis - 1;
        }
        
        // #region agent log
        { std::ofstream log("/Users/hlecates/Desktop/autolirpa/.cursor/debug.log", std::ios::app); log << "{\"location\":\"BoundedConcatNode.cpp:207\",\"message\":\"after axis adjustment\",\"data\":{\"adjusted_axis\":" << axis << ",\"ndim\":" << ndim << ",\"axis_in_bounds\":" << (axis >= 0 && axis < ndim ? "true" : "false") << "},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\"sessionId\":\"debug-session\",\"hypothesisId\":\"A\"}\n"; }
        // #endregion
        
        // Ensure axis is valid after adjustment
        if (axis < 0 || axis >= ndim) {
            std::cerr << "[BoundedConcatNode] Error: _axis=" << _axis << " (adjusted=" << axis 
                      << "), ndim=" << ndim << ", tensor shape: [";
            for (int i = 0; i < lower_bounds[0].dim(); ++i) {
                std::cerr << lower_bounds[0].size(i);
                if (i < lower_bounds[0].dim() - 1) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;
            // #region agent log
            { std::ofstream log("/Users/hlecates/Desktop/autolirpa/.cursor/debug.log", std::ios::app); log << "{\"location\":\"BoundedConcatNode.cpp:219\",\"message\":\"ERROR: axis out of range\",\"data\":{\"original_axis\":" << _axis << ",\"adjusted_axis\":" << axis << ",\"ndim\":" << ndim << "},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\"sessionId\":\"debug-session\",\"hypothesisId\":\"A\"}\n"; }
            // #endregion
            throw std::runtime_error("BoundedConcatNode: axis out of range");
        }
    }
    
    // Concatenate bounds along the specified axis
    torch::Tensor concat_lower = torch::cat(lower_bounds, axis);
    torch::Tensor concat_upper = torch::cat(upper_bounds, axis);
    
    // #region agent log
    { std::ofstream log("/Users/hlecates/Desktop/autolirpa/.cursor/debug.log", std::ios::app); log << "{\"location\":\"BoundedConcatNode.cpp:231\",\"message\":\"IBP exit success\",\"data\":{\"concat_axis\":" << axis << "},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\"sessionId\":\"debug-session\",\"hypothesisId\":\"IBP_EXIT\"}\n"; }
    // #endregion
    
    return BoundedTensor<torch::Tensor>(concat_lower, concat_upper);
}

} // namespace NLR
