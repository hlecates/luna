#ifndef __AlphaCROWNAnalysis_h__
#define __AlphaCROWNAnalysis_h__

#include "CROWNAnalysis.h"
#include "BoundedTensor.h"
#include "Map.h"
#include "Vector.h"
#include "MString.h"

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <unordered_map>

namespace NLR {

// Forward declarations
class BoundedAlphaOptimizeNode;
class TorchModel;

// BoundSide enum for binary optimization selection (following auto-LiRPA)
enum class BoundSide {
    Lower,
    Upper
};

// Configuration structure for Alpha-CROWN optimization (following auto_LiRPA approach)
struct AlphaCROWNConfig {
    bool enable_alpha_crown = true;      // Enable alpha optimization
    unsigned iteration = 20;            // Number of optimization iterations (auto_LiRPA default)
    float lr_alpha = 0.5;              // Learning rate for alpha parameters (realistic range)
    bool keep_best = true;               // Keep best bounds during optimization
    bool use_shared_alpha = false;       // Share alpha variables to save memory
    float lr_decay = 0.98;             // Learning rate decay factor
    unsigned early_stop_patience = 10;   // Early stop patience
    bool fix_interm_bounds = true;       // Only optimize final layer bounds
    std::string optimizer = "adam";      // Optimizer type
    float start_save_best = 0.5f;       // Start saving best bounds at this fraction

    // Bound computation control (following auto_LiRPA approach)
    bool optimize_bound_args = true;     // Enable bound optimization (main switch)
    bool compute_lower_bounds = true;    // Compute and optimize lower bounds
    bool compute_upper_bounds = false;    // Compute and optimize upper bounds

    // Optimization side control - binary selection following auto-LiRPA
    BoundSide bound_side = BoundSide::Lower; // Binary choice: Lower or Upper

    AlphaCROWNConfig() = default;
};

// Structure to hold alpha parameters for a single layer and start
// similar to auto_LiRPA's alpha tensor format
struct AlphaParameters {
    torch::Tensor alpha;        // Shape: [2, spec_dim, batch_size, output_size]
    int specDim{0};             // Number of specifications being verified
    int batchDim{1};            // Batch dimension (typically 1)
    int outDim{0};              // Number of neurons in the layer
    bool requiresGrad{true};    // Whether gradients are enabled
};

class AlphaCROWNAnalysis
{
public:
    AlphaCROWNAnalysis(TorchModel* torchModel);
    
    ~AlphaCROWNAnalysis();
    
    void initializeAlphaParameters();

    torch::Tensor getAlphaForNode(unsigned nodeIndex, bool isLowerBound, unsigned specIndex = 0, unsigned batchIndex = 0) const;

    // Fetch alpha slice for ALL specs at once for a specific start.
    // Returns [spec, out] tensor for use at multiply time
    torch::Tensor getAlphaForNodeAllSpecs(
        unsigned nodeIndex,
        bool isLower,
        const std::string& startKey,
        int specDim,
        int outDim,
        const torch::Tensor& input_lb,
        const torch::Tensor& input_ub);
    
    void setOptimizationStage(const std::string& stage);

    std::string getOptimizationStage() const { return _optimizationStage; }
    
    bool hasAlphaParameters(unsigned nodeIndex) const;
    
    unsigned getNumOptimizableNodes() const;
    
    std::vector<unsigned> getOptimizableNodeIndices() const;
    
    void resetForOptimization();

    // Reset alpha parameters from CROWN slopes for fresh optimization pass
    void resetAlphasFromCROWNSlopes(bool isLower);

    CROWNAnalysis* getCROWNAnalysis() const;
    TorchModel* getTorchModel() const;

    torch::Tensor getAllAlphaParameters() const;

    // Bounds access methods
    std::pair<torch::Tensor, torch::Tensor> getAllCROWNBounds() const;
    std::pair<torch::Tensor, torch::Tensor> getCROWNBoundsForLayer(unsigned nodeIndex) const;
    std::pair<torch::Tensor, torch::Tensor> getAllConcreteBounds() const;
    std::pair<torch::Tensor, torch::Tensor> getConcreteBoundsForLayer(unsigned nodeIndex) const;
    
    // Alpha parameter access methods
    std::unordered_map<unsigned, AlphaParameters> getAllAlphaParametersByLayer() const;
    AlphaParameters getAlphaParametersForLayer(unsigned nodeIndex) const;
    std::vector<torch::Tensor> getAllAlphaTensors() const;

    // NEW REFACTORED ENTRY METHOD - Returns optimized bounds for specified side
    // This method handles the complete optimization loop internally
    torch::Tensor computeOptimizedBounds(BoundSide side);

    /* DEPRECATED - OLD IMPLEMENTATIONS
    std::pair<torch::Tensor, torch::Tensor> computeBoundsWithAlpha(BoundSide side);

    std::pair<torch::Tensor, torch::Tensor> run();

    std::pair<torch::Tensor, torch::Tensor> run(
        const torch::Tensor* input,
        const torch::Tensor* specificationMatrix,
        bool computeLowerBounds = true,
        bool computeUpperBounds = true
    );
    */


    void clipAlphaParameters();

    // Alpha parameter collection (needed by TorchModel for optimizer creation)
    std::vector<torch::Tensor> collectAlphaParameters(bool isLower);

    // Configuration parameters
    bool isAlphaEnabled() const { return _alphaEnabled; }
    void setAlphaEnabled(bool enabled) { _alphaEnabled = enabled; }
    
    bool isInitialized() const { return _initialized; }
    
    unsigned getIterations() const { return _iteration; }
    void setIterations(unsigned iterations) { _iteration = iterations; }
    
    float getLearningRate() const { return _learningRate; }
    void setLearningRate(float lr) { _learningRate = lr; }
    void decayLearningRate() { _learningRate *= _config.lr_decay; }
    
    // Configuration access (following auto_LiRPA approach)
    const AlphaCROWNConfig& getConfig() const { return _config; }
    void setConfig(const AlphaCROWNConfig& config) { _config = config; updateFromConfig(); }
    
    // Set individual config options (mirroring auto_LiRPA's interface)
    void setIteration(unsigned iteration) { _config.iteration = iteration; _iteration = iteration; }
    void setLrAlpha(float lr) { _config.lr_alpha = lr; _learningRate = lr; }
    void setKeepBest(bool keep) { _config.keep_best = keep; }
    void setOptimizer(const std::string& opt) { _config.optimizer = opt; }
    void setBoundSide(BoundSide side) { _config.bound_side = side; }

    // Optimization side queries
    BoundSide getBoundSide() const { return _config.bound_side; }
    bool isOptimizingLower() const { return _config.bound_side == BoundSide::Lower; }
    bool isOptimizingUpper() const { return _config.bound_side == BoundSide::Upper; }

private:

    TorchModel* _torchModel;
    std::unique_ptr<CROWNAnalysis> _crownAnalysis;

    // Alpha parameter storage: separate for lower and upper bounds
    // nodeIndex -> startKey -> AlphaParameters
    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> _alphaByNodeStartLower;
    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> _alphaByNodeStartUpper;

    // old storage (kept during migration for compatibility)
    std::unordered_map<unsigned, AlphaParameters> _alphaParameters;

    // Optimizable activation nodes
    std::vector<std::pair<unsigned, std::shared_ptr<BoundedAlphaOptimizeNode>>> _optimizableNodes;
    
    // Configuration (following auto_LiRPA approach)
    AlphaCROWNConfig _config;   // Alpha-CROWN configuration
    bool _alphaEnabled;         // Whether alpha optimization is enabled
    bool _initialized;          // Whether alpha parameters have been initialized
    unsigned _iteration;        // Number of optimization iterations (default: 20)
    float _learningRate;        // Learning rate for alpha optimization (default: 0.5)
    std::string _optimizationStage;  // Current optimization stage

    // Best alpha tracking for optimization: separate for lower and upper bounds
    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> _bestAlphaByNodeStartLower;
    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> _bestAlphaByNodeStartUpper;

    // Legacy best alpha storage
    std::unordered_map<unsigned, AlphaParameters> _bestAlphaParameters;

    /* DEPRECATED - Optimizers now managed by TorchModel
    std::shared_ptr<torch::optim::Optimizer> _optimizerLower;
    std::shared_ptr<torch::optim::Optimizer> _optimizerUpper;
    */

    // Best bounds tracking for dual optimization
    torch::Tensor _bestLowerBounds;
    torch::Tensor _bestUpperBounds;

    // Best intermediate bounds tracking (per-node concrete bounds)
    // Maps nodeIndex -> (best_lower, best_upper) for each layer
    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _bestIntermediateBounds;


    
    // Initialization phase methods
    
    void performForwardPass();
    void prepareOptimizableActivations();
    void performCROWNInitializationPass();
    void createAlphaParameters();
    
    void createAlphaForNode(unsigned nodeIndex,
                           std::shared_ptr<BoundedAlphaOptimizeNode> optimizableNode,
                           unsigned outputSize);

    void initializeAlphaWithCROWNSlopes(torch::Tensor& alpha,
                                       std::shared_ptr<BoundedAlphaOptimizeNode> optimizableNode,
                                       unsigned nodeIndex);

    // Ensure alpha exists for (node, start) pair with correct shape and initialization
    AlphaParameters& ensureAlphaFor(
        unsigned nodeIndex,
        const std::string& startKey,
        int specDim, int outDim,
        const torch::Tensor& input_lb,
        const torch::Tensor& input_ub);
    
    // Optimization phase methods
    
    /* DEPRECATED - Optimization loop now in TorchModel
    std::pair<torch::Tensor, torch::Tensor> runOptimizationLoop(
        const torch::Tensor& input,
        const torch::Tensor& specificationMatrix,
        bool computeLowerBounds,
        bool computeUpperBounds
    );

    std::shared_ptr<torch::optim::Optimizer> createOptimizerLower();
    std::shared_ptr<torch::optim::Optimizer> createOptimizerUpper();

    void updateOptimizerLearningRate(std::shared_ptr<torch::optim::Optimizer>& optimizer);
    */
    
    // Update best alphas for improvements (for specific bound side)
    void updateBestAlphas(const std::vector<int>& improvedIndices, bool isLower);

    // Reset alpha to best known values (for specific bound side)
    void restoreBestAlphas(bool isLower);

    /* DEPRECATED - Best alpha tracking simplified
    void setupBestAlphaTracking();
    */

    // Best intermediate bounds tracking (following auto-LiRPA)
    void snapshotBestIntermediateBounds();
    void restoreBestIntermediateBounds();
    
    torch::Tensor computeLoss(const torch::Tensor& lowerBounds,
                             const torch::Tensor& upperBounds,
                             bool optimizeLower,
                             c10::optional<torch::Tensor> stop_mask_opt = c10::nullopt);

    /* DEPRECATED - Gradient step now in TorchModel
    void performGradientStep(std::shared_ptr<torch::optim::Optimizer>& optimizer,
                           const torch::Tensor& loss);
    */
  
    std::vector<int> findImprovedIndices(const torch::Tensor& currentBounds,
                                        const torch::Tensor& bestBounds,
                                        bool isLowerBound);

    // Helper method
    bool shouldPerformOptimization() const;
    
    // Bound extraction methods (following auto_LiRPA approach)
    torch::Tensor extractLowerBoundsFromCROWN();
    torch::Tensor extractUpperBoundsFromCROWN();
    
    // Configuration synchronization
    void updateFromConfig();

    // Utility 
    void log(const String& message);
};

} // namespace NLR

#endif // __AlphaCROWNAnalysis_h__