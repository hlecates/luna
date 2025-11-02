#include "BoundedSubNode.h"

namespace NLR {

BoundedSubNode::BoundedSubNode() {
    _nodeName = "sub";  // Set default name
    _nodeIndex = 0;
    _input_size = 0;  // Will be set dynamically
    _output_size = 0; // Will be set dynamically
    _constantIsSecond = true; // Default: x - constant
//     std::cout << "[BoundedSubNode] Constructor called with name: " << _nodeName << std::endl;
}

// Standard PyTorch forward pass (single input - with constant if available)
torch::Tensor BoundedSubNode::forward(const torch::Tensor& input) {
    if (_constantValue.defined()) {
        if (_constantIsSecond) {
            // x - constant
            return input - _constantValue;
        } else {
            // constant - x
            return _constantValue - input;
        }
    }
    std::cerr << "[BoundedSubNode] Warning: forward() called with single input and no constant. Returning input unchanged." << std::endl;
    return input;
}

// Multi-input forward pass for Sub: x - y
torch::Tensor BoundedSubNode::forward(const std::vector<torch::Tensor>& inputs) {
//     std::cout << "[BoundedSubNode] Forward called with " << inputs.size() << " inputs" << std::endl;

    if (inputs.size() != 2) {
        throw std::runtime_error("BoundedSubNode::forward() expects exactly 2 inputs for subtraction (x - y)");
    }

    const torch::Tensor& x = inputs[0];
    const torch::Tensor& y = inputs[1];

//     std::cout << "[BoundedSubNode] Input 0 (x) shape: " << x.sizes() << std::endl;
//     std::cout << "[BoundedSubNode] Input 1 (y) shape: " << y.sizes() << std::endl;

    // Perform element-wise subtraction (PyTorch handles broadcasting)
    torch::Tensor result = x - y;

    // Update sizes dynamically
    if (x.dim() > 0) {
        _input_size = x.numel();
        _output_size = result.numel();
    }

//     std::cout << "[BoundedSubNode] Forward output shape: " << result.sizes() << std::endl;
    return result;
}

// CROWN Backward Mode bound propagation
// For Sub: output = x - y
// Lower bound: lA @ lower(x) - uA @ upper(y)
// Upper bound: uA @ upper(x) - lA @ lower(y)
void BoundedSubNode::boundBackward(
    const torch::Tensor& last_lA,
    const torch::Tensor& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<torch::Tensor, torch::Tensor>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

//     std::cout << "[BoundedSubNode::boundBackward] Starting boundBackward" << std::endl;
//     std::cout << "[BoundedSubNode::boundBackward] last_lA defined: " << last_lA.defined() << std::endl;
//     std::cout << "[BoundedSubNode::boundBackward] last_uA defined: " << last_uA.defined() << std::endl;
//     std::cout << "[BoundedSubNode::boundBackward] inputBounds size: " << inputBounds.size() << std::endl;

    if (last_lA.defined()) {
//         std::cout << "[BoundedSubNode::boundBackward] last_lA shape: " << last_lA.sizes() << std::endl;
    }
    if (last_uA.defined()) {
//         std::cout << "[BoundedSubNode::boundBackward] last_uA shape: " << last_uA.sizes() << std::endl;
    }

    // Clear and prepare output A matrices
    outputA_matrices.clear();

    if (inputBounds.size() == 1) {
        // Single input case with constant
        if (_constantValue.defined()) {
            if (_constantIsSecond) {
                // x - constant: gradient w.r.t. x is 1
                outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(last_lA, last_uA));

                // Contribution to bias from constant: compute A @ (-constant)
                // For y = x - c, backward propagation gives: bound = A @ (x - c) = A @ x - A @ c
                // The bias term is -A @ c
                if (last_lA.defined()) {
                    torch::Tensor constant = _constantValue.flatten();

                    // Compute -A @ constant to get the bias contribution
                    torch::Tensor constant_contrib;
                    if (last_lA.dim() == 3) {
                        // Shape: (batch, spec, features) @ (features,) -> (batch, spec)
                        constant_contrib = torch::matmul(last_lA, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) {
                            constant_contrib = constant_contrib.squeeze(0);
                        }
                    } else if (last_lA.dim() == 2) {
                        // Shape: (spec, features) @ (features,) -> (spec,)
                        constant_contrib = torch::matmul(last_lA, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_lA dimensions");
                    }

                    // Negate because we're subtracting the constant
                    lbias = -constant_contrib;
                }

                if (last_uA.defined()) {
                    torch::Tensor constant = _constantValue.flatten();

                    // Compute -A @ constant for upper bound
                    torch::Tensor constant_contrib;
                    if (last_uA.dim() == 3) {
                        // Shape: (batch, spec, features) @ (features,) -> (batch, spec)
                        constant_contrib = torch::matmul(last_uA, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) {
                            constant_contrib = constant_contrib.squeeze(0);
                        }
                    } else if (last_uA.dim() == 2) {
                        // Shape: (spec, features) @ (features,) -> (spec,)
                        constant_contrib = torch::matmul(last_uA, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_uA dimensions");
                    }

                    // Negate because we're subtracting the constant
                    ubias = -constant_contrib;
                }
            } else {
                // constant - x: gradient w.r.t. x is -1, so negate the A matrices
                torch::Tensor neg_lA = last_lA.defined() ? -last_lA : torch::Tensor();
                torch::Tensor neg_uA = last_uA.defined() ? -last_uA : torch::Tensor();
                outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(neg_lA, neg_uA));

                // Contribution to bias from constant: compute A @ constant
                // For y = c - x, backward propagation gives: bound = A @ (c - x) = A @ c - A @ x
                // The bias term is +A @ c
                if (last_lA.defined()) {
                    torch::Tensor constant = _constantValue.flatten();

                    // Compute A @ constant to get the bias contribution
                    torch::Tensor constant_contrib;
                    if (last_lA.dim() == 3) {
                        // Shape: (batch, spec, features) @ (features,) -> (batch, spec)
                        constant_contrib = torch::matmul(last_lA, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) {
                            constant_contrib = constant_contrib.squeeze(0);
                        }
                    } else if (last_lA.dim() == 2) {
                        // Shape: (spec, features) @ (features,) -> (spec,)
                        constant_contrib = torch::matmul(last_lA, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_lA dimensions");
                    }

                    // Positive because constant comes first
                    lbias = constant_contrib;
                }

                if (last_uA.defined()) {
                    torch::Tensor constant = _constantValue.flatten();

                    // Compute A @ constant for upper bound
                    torch::Tensor constant_contrib;
                    if (last_uA.dim() == 3) {
                        // Shape: (batch, spec, features) @ (features,) -> (batch, spec)
                        constant_contrib = torch::matmul(last_uA, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) {
                            constant_contrib = constant_contrib.squeeze(0);
                        }
                    } else if (last_uA.dim() == 2) {
                        // Shape: (spec, features) @ (features,) -> (spec,)
                        constant_contrib = torch::matmul(last_uA, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_uA dimensions");
                    }

                    // Positive because constant comes first
                    ubias = constant_contrib;
                }
            }
        } else {
            throw std::runtime_error("BoundedSubNode::boundBackward with 1 input requires a constant value");
        }
    } else if (inputBounds.size() >= 2) {
        // Two input case
        // First input (x): positive sign, normal A matrices
        // For lower bound: lA @ lower(x)
        // For upper bound: uA @ upper(x)
        outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(last_lA, last_uA));
//         std::cout << "[BoundedSubNode::boundBackward] Added A matrices for first input (x): (last_lA, last_uA)" << std::endl;

        // Second input (y): negative sign, SWAPPED A matrices!
        // For lower bound of (x - y): lA @ lower(x) - uA @ upper(y) → use -uA for y
        // For upper bound of (x - y): uA @ upper(x) - lA @ lower(y) → use -lA for y
        torch::Tensor neg_lA = last_lA.defined() ? -last_lA : torch::Tensor();
        torch::Tensor neg_uA = last_uA.defined() ? -last_uA : torch::Tensor();

        // CRITICAL: Swap neg_uA and neg_lA when appending!
        // The pair is (lA_for_y, uA_for_y) = (-uA, -lA)
        outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(neg_uA, neg_lA));
//         std::cout << "[BoundedSubNode::boundBackward] Added A matrices for second input (y): (-last_uA, -last_lA) [SWAPPED]" << std::endl;
    } else {
        throw std::runtime_error("BoundedSubNode::boundBackward expects at least 1 input bound");
    }

    // Initialize bias if needed - but DON'T override if already set by constant handling above
    // The bias dimension should match dimension 1 of the A matrix (number of output neurons)
    if (last_lA.defined()) {
        // For A matrix shape [batch, output_neurons, input_neurons]
        // The bias should have shape [output_neurons]
        int output_size = last_lA.size(1); // Second dimension is number of output neurons
//         std::cout << "[BoundedSubNode::boundBackward] Output size from A matrix: " << output_size << std::endl;

        if (!lbias.defined()) {
            // Only initialize if not already set by constant handling
            lbias = torch::zeros({output_size});
//             std::cout << "[BoundedSubNode::boundBackward] Initialized lbias to zeros with size: " << output_size << std::endl;
        } else if (lbias.numel() != output_size) {
            // If bias size doesn't match, we need to handle it properly
//             std::cout << "[BoundedSubNode::boundBackward] Warning: lbias size " << lbias.numel()
//                       << " doesn't match expected output size " << output_size << std::endl;

            // If bias came from a constant contribution and doesn't match, we need to expand/broadcast it
            if (lbias.numel() == 1) {
                // Scalar bias - expand to output size
                lbias = lbias.expand({output_size});
//                 std::cout << "[BoundedSubNode::boundBackward] Expanded scalar lbias to size " << output_size << std::endl;
            } else if (_constantValue.defined() && lbias.numel() == _constantValue.numel()) {
                // Bias matches constant size, not output size - this is the problematic case
                // We need to handle broadcasting properly based on the operation semantics
//                 std::cout << "[BoundedSubNode::boundBackward] Bias from constant has incompatible size, keeping as-is" << std::endl;
                // Don't modify - let the error happen so we can debug further
            }
        }
    } else {
        if (!lbias.defined()) {
            lbias = torch::zeros({1});
//             std::cout << "[BoundedSubNode::boundBackward] Initialized lbias to zeros (fallback)" << std::endl;
        }
    }

    if (last_uA.defined()) {
        int output_size = last_uA.size(1); // Second dimension is number of output neurons
//         std::cout << "[BoundedSubNode::boundBackward] Output size from A matrix: " << output_size << std::endl;

        if (!ubias.defined()) {
            ubias = torch::zeros({output_size});
//             std::cout << "[BoundedSubNode::boundBackward] Initialized ubias to zeros with size: " << output_size << std::endl;
        } else if (ubias.numel() != output_size) {
            // If bias size doesn't match, we need to handle it properly
//             std::cout << "[BoundedSubNode::boundBackward] Warning: ubias size " << ubias.numel()
//                       << " doesn't match expected output size " << output_size << std::endl;

            // If bias came from a constant contribution and doesn't match, we need to expand/broadcast it
            if (ubias.numel() == 1) {
                // Scalar bias - expand to output size
                ubias = ubias.expand({output_size});
//                 std::cout << "[BoundedSubNode::boundBackward] Expanded scalar ubias to size " << output_size << std::endl;
            } else if (_constantValue.defined() && ubias.numel() == _constantValue.numel()) {
                // Bias matches constant size, not output size - this is the problematic case
                // We need to handle broadcasting properly based on the operation semantics
//                 std::cout << "[BoundedSubNode::boundBackward] Bias from constant has incompatible size, keeping as-is" << std::endl;
                // Don't modify - let the error happen so we can debug further
            }
        }
    } else {
        if (!ubias.defined()) {
            ubias = torch::zeros({1});
//             std::cout << "[BoundedSubNode::boundBackward] Initialized ubias to zeros (fallback)" << std::endl;
        }
    }

//     std::cout << "[BoundedSubNode::boundBackward] Final lbias shape: " << lbias.sizes() << std::endl;
//     std::cout << "[BoundedSubNode::boundBackward] Final ubias shape: " << ubias.sizes() << std::endl;
}

// IBP (Interval Bound Propagation): Fast interval-based bound computation for Sub
// For Sub: output = x - y
// Lower bound: lower(x) - upper(y)
// Upper bound: upper(x) - lower(y)
BoundedTensor<torch::Tensor> BoundedSubNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

//     std::cout << "[BoundedSubNode::computeIntervalBoundPropagation] Starting IBP computation" << std::endl;

    if (inputBounds.size() == 1) {
        // Single input case with constant
        const auto& xBounds = inputBounds[0];
        if (_constantValue.defined()) {
            if (_constantIsSecond) {
                // x - constant: [lower(x) - constant, upper(x) - constant]
                torch::Tensor resultLower = xBounds.lower() - _constantValue;
                torch::Tensor resultUpper = xBounds.upper() - _constantValue;
                return BoundedTensor<torch::Tensor>(resultLower, resultUpper);
            } else {
                // constant - x: [constant - upper(x), constant - lower(x)]
                torch::Tensor resultLower = _constantValue - xBounds.upper();
                torch::Tensor resultUpper = _constantValue - xBounds.lower();
                return BoundedTensor<torch::Tensor>(resultLower, resultUpper);
            }
        } else {
            throw std::runtime_error("BoundedSubNode::computeIntervalBoundPropagation with 1 input requires a constant value");
        }
    } else if (inputBounds.size() >= 2) {
        const auto& xBounds = inputBounds[0];
        const auto& yBounds = inputBounds[1];

        torch::Tensor xLower = xBounds.lower();
        torch::Tensor xUpper = xBounds.upper();
        torch::Tensor yLower = yBounds.lower();
        torch::Tensor yUpper = yBounds.upper();

//         std::cout << "[BoundedSubNode::computeIntervalBoundPropagation] x lower shape: " << xLower.sizes() << std::endl;
//         std::cout << "[BoundedSubNode::computeIntervalBoundPropagation] x upper shape: " << xUpper.sizes() << std::endl;
//         std::cout << "[BoundedSubNode::computeIntervalBoundPropagation] y lower shape: " << yLower.sizes() << std::endl;
//         std::cout << "[BoundedSubNode::computeIntervalBoundPropagation] y upper shape: " << yUpper.sizes() << std::endl;

        // For subtraction: [lower(x) - upper(y), upper(x) - lower(y)]
        torch::Tensor resultLower = xLower - yUpper;
        torch::Tensor resultUpper = xUpper - yLower;

//         std::cout << "[BoundedSubNode::computeIntervalBoundPropagation] Result lower shape: " << resultLower.sizes() << std::endl;
//         std::cout << "[BoundedSubNode::computeIntervalBoundPropagation] Result upper shape: " << resultUpper.sizes() << std::endl;

        return BoundedTensor<torch::Tensor>(resultLower, resultUpper);
    } else {
        throw std::runtime_error("BoundedSubNode::computeIntervalBoundPropagation requires at least 1 input");
    }
}

void BoundedSubNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedSubNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
