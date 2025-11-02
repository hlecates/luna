// LIRPA TorchModel - Standalone neural network model for CROWN/Alpha-CROWN analysis
// This model is decoupled from the Marabou NLR engine

#ifndef __TorchModel_h__
#define __TorchModel_h__

#include "Map.h"
#include "Set.h"
#include "MString.h"
#include "Vector.h"
#include "BoundedTorchNode.h"
#include "BoundedInputNode.h"
#include "BoundedLinearNode.h"
#include "BoundedReLUNode.h"
#include "BoundedIdentityNode.h"
#include "BoundedConstantNode.h"
#include "BoundedReshapeNode.h"

// Forward declarations to avoid circular dependency
class CROWNAnalysis;
class AlphaCROWNAnalysis;

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

namespace NLR {

// Configuration structure for analysis (mirrors auto_LiRPA bound_opts)
struct AnalysisConfig {
    // Analysis type selection
    enum class Method { CROWN, AlphaCROWN };
    Method method = Method::CROWN;

    // Bound computation flags
    bool compute_lower = true;
    bool compute_upper = true;

    // AlphaCROWN specific options (mirroring auto_LiRPA)
    unsigned alpha_iterations = 20;
    float alpha_lr = 0.5;
    bool optimize_lower = true;
    bool optimize_upper = false;

    // Other options
    bool verbose = true;

    AnalysisConfig() = default;
};

class TorchModel {
public:
    // Traditional constructor with pre-built nodes
    TorchModel(const Vector<std::shared_ptr<BoundedTorchNode>>& nodes,
               const Vector<unsigned>& inputIndices,
               unsigned outputIndex,
               const Map<unsigned, Vector<unsigned>>& dependencies);

    // Constructor that loads from ONNX file (mirrors auto_LiRPA BoundedModule)
    TorchModel(const String& onnxPath);

    // Constructor that loads from ONNX file and VNN-LIB file for input bounds
    TorchModel(const String& onnxPath,
               const String& vnnlibPath);

    // Forward pass through the entire model
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor forward(unsigned nodeIndex, Map<unsigned, torch::Tensor>& activations, 
                         const Map<unsigned, torch::Tensor>& inputs);

    
    
    // forward pass that returns activations for all nodes
    Map<unsigned, torch::Tensor> forwardAndStoreActivations(const torch::Tensor& input);
    Map<unsigned, torch::Tensor> forwardAndStoreActivations(const Map<unsigned, torch::Tensor>& inputs);            

    // Get model information
    unsigned getInputSize() const { return _input_size; }
    unsigned getOutputSize() const { return _output_size; }
    unsigned getNumNodes() const { return _nodes.size(); }
    
    // Access to nodes
    const Vector<std::shared_ptr<BoundedTorchNode>>& getNodes() const { return _nodes; }
    std::shared_ptr<BoundedTorchNode> getNode(unsigned index) const;
    Vector<unsigned> getAllNodeIndices() const;
    Vector<unsigned> getNodesByType(NodeType type) const;
    const Vector<unsigned>& getInputIndices() const { return _inputIndices; }
    unsigned getOutputIndex() const { return _outputIndex; }
    
    // PRIMARY BOUND MANAGEMENT INTERFACE
    void setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds);
    
    // CONCRETE BOUND STORAGE (for CROWN analysis to call)
    void setConcreteBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& concreteBounds);
    void clearConcreteBounds();  // Clear all cached concrete bounds
    // FOR TESTING AND OUTPUTTING
    BoundedTensor<torch::Tensor> getConcreteBounds(unsigned nodeIndex) const;
    bool hasConcreteBounds(unsigned nodeIndex) const;
    
    // Input bound access
    BoundedTensor<torch::Tensor> getInputBounds() const;
    bool hasInputBounds() const;
    torch::Tensor getInputLowerBounds() const;
    torch::Tensor getInputUpperBounds() const;

    // SPECIFICATION MATRIX MANAGEMENT
    void setSpecificationMatrix(const torch::Tensor& specMatrix);
    torch::Tensor getSpecificationMatrix() const;
    bool hasSpecificationMatrix() const;

    // ANALYSIS CONFIGURATION MANAGEMENT (mirrors auto_LiRPA interface)
    void setAnalysisConfig(const AnalysisConfig& config);
    AnalysisConfig getAnalysisConfig() const;
    void setAnalysisMethod(AnalysisConfig::Method method);
    void setVerbose(bool verbose);

    // UNIFIED ANALYSIS ENTRY METHOD (mirrors auto_LiRPA's compute_bounds)
    BoundedTensor<torch::Tensor> compute_bounds(
        const BoundedTensor<torch::Tensor>& input_bounds,
        const torch::Tensor* specification_matrix = nullptr,
        AnalysisConfig::Method method = AnalysisConfig::Method::CROWN,
        bool bound_lower = true,
        bool bound_upper = true
    );

    // ANALYSIS ENTRY METHODS
    BoundedTensor<torch::Tensor> runCROWN();
    BoundedTensor<torch::Tensor> runCROWN(const BoundedTensor<torch::Tensor>& inputBounds);

    // NEW ALPHA-CROWN ENTRY METHODS (REFACTORED)
    // TorchModel controls optimization flow and calls AlphaCROWN for bound computation
    BoundedTensor<torch::Tensor> runAlphaCROWN(bool optimizeLower = true, bool optimizeUpper = false);
    BoundedTensor<torch::Tensor> runAlphaCROWN(const BoundedTensor<torch::Tensor>& inputBounds,
                                                bool optimizeLower = true, bool optimizeUpper = false);

    /* DEPRECATED - OLD ENTRY METHOD
    BoundedTensor<torch::Tensor> runAlphaCROWN();
    BoundedTensor<torch::Tensor> runAlphaCROWN(const BoundedTensor<torch::Tensor>& inputBounds);
    */

    // ANALYSIS BOUNDS STORAGE
    void setCROWNBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& bounds);
    void setAlphaCROWNBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& bounds);
    BoundedTensor<torch::Tensor> getCROWNBounds(unsigned nodeIndex) const;
    BoundedTensor<torch::Tensor> getAlphaCROWNBounds(unsigned nodeIndex) const;
    bool hasCROWNBounds(unsigned nodeIndex) const;
    bool hasAlphaCROWNBounds(unsigned nodeIndex) const;

    // FINAL ANALYSIS BOUNDS (output bounds from most recent analysis)
    void setFinalAnalysisBounds(const BoundedTensor<torch::Tensor>& bounds);
    BoundedTensor<torch::Tensor> getFinalAnalysisBounds() const;
    bool hasFinalAnalysisBounds() const;
    
    // Dependency graph access
    const Map<unsigned, Vector<unsigned>>& getDependenciesMap() const { return _dependencies; }

    // Full graph 
    void buildDependencyGraph();
    void buildDependents();
    void computeDegrees();

    // Traversal 
    Vector<unsigned> topologicalSort() const;
    Vector<unsigned> getRoots() const;
    Vector<unsigned> getLeaves() const;
    Vector<unsigned> getDependents(unsigned nodeIndex) const;
    Vector<unsigned> getDependencies(unsigned nodeIndex) const;

    // Degree and Processing states
    unsigned getDegreeOut(unsigned nodeIndex) const;
    unsigned getDegreeIn(unsigned nodeIndex) const;
    void resetProcessingState();
    bool isProcessed(unsigned nodeIndex) const;
    void markProcessed(unsigned nodeIndex);

    // Logging
    void log(const String& message) const;

private:
    Vector<std::shared_ptr<BoundedTorchNode>> _nodes;
    Vector<unsigned> _inputIndices;
    unsigned _outputIndex;
    Map<unsigned, Vector<unsigned>> _dependencies;

    // Graph traversal state
    Map<unsigned, Vector<unsigned>> _dependents;
    Map<unsigned, unsigned> _degreeOut;
    Map<unsigned, unsigned> _degreeIn;
    Map<unsigned, bool> _processed;

    // Model dimensions
    unsigned _input_size;
    unsigned _output_size;

    BoundedTensor<torch::Tensor> _inputBounds;      // Input bounds for the model
    Map<unsigned, BoundedTensor<torch::Tensor>> _concreteBounds;  // CROWN concrete bounds

    // SPECIFICATION AND ANALYSIS STORAGE
    torch::Tensor _specificationMatrix;            // Specification matrix for analysis
    bool _hasSpecificationMatrix;                  // Whether specification matrix is set
    Map<unsigned, BoundedTensor<torch::Tensor>> _crownBounds;      // CROWN analysis bounds
    Map<unsigned, BoundedTensor<torch::Tensor>> _alphaCrownBounds; // AlphaCROWN analysis bounds
    BoundedTensor<torch::Tensor> _finalAnalysisBounds;             // Final analysis output bounds
    bool _hasFinalAnalysisBounds;                  // Whether final bounds are set

    // ANALYSIS CONFIGURATION (mirrors auto_LiRPA bound_opts)
    AnalysisConfig _analysisConfig;                // Analysis configuration
    
    // error checking
    void validateNodeIndex(unsigned nodeIndex) const;
};

} // namespace NLR

#endif // __TorchModel_h__ 