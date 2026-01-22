#include "BoundedAlphaOptimizedNode.h"
#include "AlphaCROWNAnalysis.h"

namespace NLR {

torch::Tensor BoundedAlphaOptimizeNode::getAlphaForBound(bool isLowerBound, unsigned specIndex) const {
    if (!_alphaCrownAnalysis) {
        // Return default alpha value if no AlphaCROWN analysis is available
        return torch::full({getOutputSize()}, 0.5f, torch::dtype(torch::kFloat32));
    }

    // Get alpha parameters from AlphaCROWNAnalysis
    return _alphaCrownAnalysis->getAlphaForNode(getNodeIndex(), isLowerBound, specIndex);
}

// Optimization side queries (delegated to AlphaCROWNAnalysis)
bool BoundedAlphaOptimizeNode::isOptimizingLower() const {
    return _alphaCrownAnalysis ? _alphaCrownAnalysis->isOptimizingLower() : true;  // Default to lower
}

bool BoundedAlphaOptimizeNode::isOptimizingUpper() const {
    return _alphaCrownAnalysis ? _alphaCrownAnalysis->isOptimizingUpper() : false; // Default to not upper
}

bool BoundedAlphaOptimizeNode::isOptimizingBoth() const {
    return false;  // Always false - removed "both" mode in favor of binary selection
}

} // namespace NLR