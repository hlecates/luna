// BoundedConcatNode.h - Concatenation with bound propagation
#ifndef __BOUNDED_CONCAT_NODE_H__
#define __BOUNDED_CONCAT_NODE_H__

#include "BoundedTorchNode.h"
#include <vector>

namespace NLR {

class BoundedConcatNode : public BoundedTorchNode {
public:
    BoundedConcatNode(int axis, unsigned numInputs, const String& name = "");

    // Node identification
    NodeType getNodeType() const override { return NodeType::CONCAT; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }

    // Forward pass - concatenate inputs
    torch::Tensor forward(const torch::Tensor& input) override;
    torch::Tensor forward(const std::vector<torch::Tensor>& inputs) override;

    // Backward bound propagation - split A matrix
    void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;

    // IBP computation - concatenate bounds
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

    // Concat-specific
    int getAxis() const { return _axis; }
    unsigned getNumInputs() const { return _numInputs; }
    void setInputSizes(const std::vector<unsigned>& sizes) { _input_sizes = sizes; }

private:
    int _axis;              // Concatenation axis
    unsigned _numInputs;    // Number of inputs to concatenate
    std::vector<unsigned> _input_sizes;  // Size of each input along concat axis
    
    String _nodeName;
    unsigned _nodeIndex;
    unsigned _input_size;
    unsigned _output_size;
};

} // namespace NLR

#endif // __BOUNDED_CONCAT_NODE_H__
