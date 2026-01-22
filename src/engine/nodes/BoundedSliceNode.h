// BoundedSliceNode.h - Slice operation with bound propagation
#ifndef __BOUNDED_SLICE_NODE_H__
#define __BOUNDED_SLICE_NODE_H__

#include "BoundedTorchNode.h"
#include <vector>

namespace NLR {

class BoundedSliceNode : public BoundedTorchNode {
public:
    BoundedSliceNode(int start, int end, int axis, int step = 1, const String& name = "");

    // Node identification
    NodeType getNodeType() const override { return NodeType::IDENTITY; } // Reuse IDENTITY for now
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }

    // Forward pass - slice input tensor
    torch::Tensor forward(const torch::Tensor& input) override;

    // Backward bound propagation - pad A matrix with zeros
    void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;

    // IBP computation - narrow bounds
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) override;

    // Node information
    unsigned getInputSize() const override { return _input_size; }
    unsigned getOutputSize() const override { return _output_size; }
    bool isPerturbed() const override { return true; }

    // Size setters
    void setInputSize(unsigned size) override { _input_size = size; }
    void setOutputSize(unsigned size) override { _output_size = size; }

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }

    // Slice-specific
    int getStart() const { return _start; }
    int getEnd() const { return _end; }
    int getAxis() const { return _axis; }
    int getStep() const { return _step; }
    
    void setInputShape(const std::vector<int64_t>& shape) { _input_shape = shape; }
    const std::vector<int64_t>& getInputShape() const { return _input_shape; }

private:
    // Helper to fix up negative indices and bounds
    std::pair<int64_t, int64_t> fixupParams(const std::vector<int64_t>& shape, 
                                             int64_t start, int64_t end, 
                                             int64_t axis, int64_t step) const;
    
    int _start;             // Start index for slice
    int _end;               // End index for slice
    int _axis;              // Axis along which to slice
    int _step;              // Step size (1 or -1)
    
    std::vector<int64_t> _input_shape;  // Shape of input tensor
    
    String _nodeName;
    unsigned _nodeIndex;
    unsigned _input_size;
    unsigned _output_size;
};

} // namespace NLR

#endif // __BOUNDED_SLICE_NODE_H__
