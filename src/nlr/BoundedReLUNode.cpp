#include "BoundedReLUNode.h"
#include "AlphaCROWNAnalysis.h"
#include "GlobalConfiguration.h"
#include <algorithm>

namespace NLR {

BoundedReLUNode::BoundedReLUNode(const torch::nn::ReLU& reluModule, const String& name)
    : BoundedAlphaOptimizeNode()
    , _reluModule(std::make_shared<torch::nn::ReLU>(reluModule)) {
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
}

// Forward pass through the ReLU layer
torch::Tensor BoundedReLUNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel();
    }
    
    // Apply ReLU transformation
    return (*_reluModule)(input);
}

// Auto-LiRPA style boundBackward method
void BoundedReLUNode::boundBackward(
    const torch::Tensor& last_lA,
    const torch::Tensor& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<torch::Tensor, torch::Tensor>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedReLUNode expects at least one input");
    }

    const auto& inputBound = inputBounds[0];
    torch::Tensor input_lower = inputBound.lower();
    torch::Tensor input_upper = inputBound.upper();

    // Extract live spec dimension from last_lA or last_uA for per-spec alpha
    int specDim = 1;
    if (last_lA.defined() && last_lA.dim() >= 2) {
        specDim = (int)last_lA.size(1);  // Shape: [1, spec, out]
    } else if (last_uA.defined() && last_uA.dim() >= 2) {
        specDim = (int)last_uA.size(1);  // Shape: [1, spec, out]
    }
    _currentSpecDim = specDim;  // Store for use in _maskAlpha

    // Call the unified backward relaxation method (following auto_LiRPA approach)
    // This method handles both 'init' and 'opt' stages internally
    auto relaxation_result = _backwardRelaxation(last_lA, last_uA, input_lower, input_upper);

    // LOWER objective uses (aL, aU) = (lb_lower_d, lb_upper_d) with (bL, bU) = (bias_lower, bias_upper)
    // UPPER objective uses (aU, aL) = (ub_upper_d, lb_lower_d) with (bU, bL) = (bias_upper, bias_lower)

    auto expand_like = [](torch::Tensor v, const torch::Tensor& A) {
        if (!v.defined()) return v;
        int add_dims = A.dim() - v.dim();
        while (add_dims-- > 0) v = v.unsqueeze(0);  // add leading singleton dims
        return v.expand_as(A);
    };

    auto reduce_bias = [&](const torch::Tensor& term, int in_ndim) {
        // sum over the last `in_ndim` dims
        std::vector<int64_t> dims;
        for (int i = term.dim() - in_ndim; i < term.dim(); ++i) dims.push_back(i);
        return term.sum(dims);  // keeps leading batch/spec dims (if any)
    };

    torch::Tensor new_lA, new_uA;

    // ----- LOWER path -----
    if (last_lA.defined()) {
        auto Apos = torch::clamp_min(last_lA, 0);
        auto Aneg = torch::clamp_max(last_lA, 0);

        // Use separate slopes if available (optimization mode), otherwise fall back to unified slopes (init mode)
        auto aL_l = relaxation_result.lb_lower_d.defined() ? relaxation_result.lb_lower_d : relaxation_result.d_lower;
        auto aU_l = relaxation_result.lb_upper_d.defined() ? relaxation_result.lb_upper_d : relaxation_result.d_upper;
        auto bL_l = relaxation_result.bias_lower.defined() ? relaxation_result.bias_lower : torch::zeros_like(input_lower);
        auto bU_l = relaxation_result.bias_upper.defined() ? relaxation_result.bias_upper : torch::zeros_like(input_lower);

        auto A_l = Apos * expand_like(aL_l, last_lA) + Aneg * expand_like(aU_l, last_lA);
        auto b_l = Apos * expand_like(bL_l, last_lA) + Aneg * expand_like(bU_l, last_lA);

        new_lA = A_l;
        auto add_lbias = reduce_bias(b_l, /*in_ndim=*/input_lower.dim());

        // accumulate instead of overwrite
        lbias = lbias.defined() ? (lbias + add_lbias) : add_lbias;

        if (outputA_matrices.size() == 0) outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(torch::Tensor(), torch::Tensor()));
        outputA_matrices[0].first() = new_lA;
    }

    // ----- UPPER path -----
    if (last_uA.defined()) {
        auto Apos = torch::clamp_min(last_uA, 0);
        auto Aneg = torch::clamp_max(last_uA, 0);

        // Use separate slopes if available (optimization mode), otherwise fall back to unified slopes (init mode)
        auto aU_u = relaxation_result.ub_upper_d.defined()
                  ? relaxation_result.ub_upper_d
                  : relaxation_result.d_upper;
        auto aL_u = relaxation_result.lb_lower_d.defined()
                  ? relaxation_result.lb_lower_d
                  : relaxation_result.d_lower;
        auto bU_u = relaxation_result.bias_upper.defined() ? relaxation_result.bias_upper : torch::zeros_like(input_lower);
        auto bL_u = relaxation_result.bias_lower.defined() ? relaxation_result.bias_lower : torch::zeros_like(input_lower);

        auto A_u = Apos * expand_like(aU_u, last_uA) + Aneg * expand_like(aL_u, last_uA);
        auto b_u = Apos * expand_like(bU_u, last_uA) + Aneg * expand_like(bL_u, last_uA);

        auto add_ubias = reduce_bias(b_u, /*in_ndim=*/input_lower.dim());

        // accumulate instead of overwrite
        ubias = ubias.defined() ? (ubias + add_ubias) : add_ubias;

        if (outputA_matrices.size() == 0) outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(torch::Tensor(), torch::Tensor()));
        outputA_matrices[0].second() = A_u;
    }

    // Ensure outputA_matrices has the right structure if not already set
    if (outputA_matrices.size() == 0) {
        outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(new_lA, new_uA));
    }
}

// IBP (Interval Bound Propagation): Fast interval-based bound computation for ReLU
BoundedTensor<torch::Tensor> BoundedReLUNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("ReLU module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();
    
    // Set input size from the input tensor during IBP
    if (_input_size == 0 && inputLowerBound.defined()) {
        _input_size = inputLowerBound.numel();
    }
    
    // ReLU: y = max(0, x)
    torch::Tensor lowerBound = torch::clamp_min(inputLowerBound, 0);  // max(0, lower)
    torch::Tensor upperBound = torch::clamp_min(inputUpperBound, 0);  // max(0, upper)
    
    // Set output size from the computed bounds during IBP
    if (_output_size == 0 && lowerBound.defined()) {
        _output_size = lowerBound.numel();
    }

    return BoundedTensor<torch::Tensor>(lowerBound, upperBound);
}

// Node information
unsigned BoundedReLUNode::getInputSize() const {
    return _input_size;
}

unsigned BoundedReLUNode::getOutputSize() const {
    // If output size is not set, try to infer from input size
    if (_output_size == 0 && _input_size > 0) {
        return _input_size; // ReLU preserves input size
    }
    // If still 0, return a reasonable default for testing
    if (_output_size == 0) {
        return 2; // Default size for testing
    }
    return _output_size;
}

void BoundedReLUNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedReLUNode::setOutputSize(unsigned size) {
    _output_size = size;
}

// Unified backward relaxation method (following auto_LiRPA approach)
BoundedReLUNode::RelaxationResult BoundedReLUNode::_backwardRelaxation(
    const torch::Tensor& last_lA, const torch::Tensor& last_uA,
    const torch::Tensor& input_lower, const torch::Tensor& input_upper)
{
    (void)last_lA;  // Unused in simple implementation
    (void)last_uA;  // Unused in simple implementation

    RelaxationResult result;

    // Compute standard CROWN upper bound relaxation (secant line for unstable neurons)
    auto [upper_d, upper_b] = _reluUpperBound(input_lower, input_upper);

    // Compute standard CROWN lower bound
    torch::Tensor lower_d = _computeStandardCROWNLowerBound(input_lower, input_upper);

    // Store slopes for alpha initialization if needed
    init_d = lower_d.detach().clone();
    init_upper_d = upper_d.detach().clone();

    // Set basic relaxation slopes and biases
    result.d_lower = lower_d;
    result.d_upper = upper_d;
    // For lower bound: bias_lower must be zero (all ReLU lower relaxations pass through origin: y ≥ α·x + 0)
    result.bias_lower = torch::zeros_like(input_lower);
    result.bias_upper = upper_b;

    // Apply alpha masking if enabled
    _maskAlpha(input_lower, input_upper, upper_d, result);


    return result;
}

// Helper method: CROWN upper bound computation
std::pair<torch::Tensor, torch::Tensor> BoundedReLUNode::_reluUpperBound(const torch::Tensor& lb, const torch::Tensor& ub)
{
    // Compute standard CROWN upper bound relaxation slopes
    torch::Tensor lb_r = torch::clamp_max(lb, 0);  // Negative part of lower bound
    torch::Tensor ub_r = torch::clamp_min(ub, 0);  // Positive part of upper bound
    ub_r = torch::max(ub_r, lb_r + 1e-8);  // Avoid division by zero

    // Standard CROWN upper bound slope formula: upper_bound / (upper_bound - lower_bound)
    torch::Tensor upper_d = ub_r / (ub_r - lb_r);
    torch::Tensor upper_b = -lb_r * upper_d;

    return std::make_pair(upper_d, upper_b);
}

// Helper method: Standard CROWN lower bound computation
torch::Tensor BoundedReLUNode::_computeStandardCROWNLowerBound(const torch::Tensor& input_lower, const torch::Tensor& input_upper)
{
    // Initialize slopes for the three ReLU cases
    torch::Tensor slopes_lower = torch::zeros_like(input_lower);

    // Case 1: input_lower >= 0 (always active) - slope = 1
    auto always_active_mask = input_lower >= 0;
    slopes_lower = torch::where(always_active_mask, torch::ones_like(slopes_lower), slopes_lower);

    // Case 2: input_upper <= 0 (always inactive) - slope = 0 (already initialized)
    // Case 3: uncertain neurons - use adaptive approach
    auto uncertain_mask = (input_lower < 0) & (input_upper > 0);
    if ( uncertain_mask.any().item<bool>() )
    {
        // TODO: Avoid recomputing this and instead pass through by reference
        // Compute upper slope for adaptive decision
        torch::Tensor ub_r = torch::clamp_min(input_upper, 0);
        torch::Tensor lb_r = torch::clamp_max(input_lower, 0);
        ub_r = torch::max(ub_r, lb_r + 1e-8);
        torch::Tensor upper_slope = ub_r / (ub_r - lb_r);

        // if upper_slope > 0.5, use slope = 1, otherwise use slope = 0
        auto adaptive_mask = upper_slope > 0.5;
        torch::Tensor lower_slope = torch::where(adaptive_mask,
                                                torch::ones_like(input_lower),
                                                torch::zeros_like(input_lower));
        slopes_lower = torch::where(uncertain_mask, lower_slope, slopes_lower);
    }

    return slopes_lower;
}

// Helper method: Get alpha parameters for specific bound type
// NOTE: This method is deprecated - alpha is now fetched directly in boundBackward at multiply time
torch::Tensor BoundedReLUNode::getAlphaForBound(bool isLowerBound, int boundType) const
{
    (void)boundType;
    (void)isLowerBound;

    // Alpha is now fetched in boundBackward with proper start context
    // This method is kept for compatibility but returns empty tensor
    return torch::Tensor();
}

// Apply masking for stable/unstable neurons (following auto_LiRPA approach)
void BoundedReLUNode::_maskAlpha(const torch::Tensor& input_lower, const torch::Tensor& input_upper, const torch::Tensor& upper_d, RelaxationResult& result)
{
    auto unstable = (input_lower < 0) & (input_upper > 0);

    // For alpha optimization - alpha is the optimizable slope
    if (isAlphaOptimizationEnabled() && _alphaCrownAnalysis && _currentSpecDim > 0) {
        // Fetch alpha with LIVE spec dimensions from last_lA
        auto* crown = _alphaCrownAnalysis->getCROWNAnalysis();
        std::string startKey = crown->currentStartKey();
        if (startKey.empty()) startKey = "default";

        // Get the actual output dimension from input_lower
        // input_lower can be either [neurons] or [batch, neurons] or higher dimensional
        // We want the total number of neurons, which is the product of all dimensions
        int outDim = (int)input_lower.numel();
        int specDim = _currentSpecDim; // Use the live spec dimension from last_lA

        // DEBUG: Print shapes to understand the mismatch (disabled for clean output)
        // printf("[DEBUG _maskAlpha] input_lower shape:");
        // for (int i = 0; i < input_lower.dim(); ++i) {
        //     printf(" %lld", (long long)input_lower.size(i));
        // }
        // printf(", outDim=%d, specDim=%d\n", outDim, specDim);

        auto alpha_tensor = _alphaCrownAnalysis->getAlphaForNodeAllSpecs(
            getNodeIndex(), /*isLower=*/true,
            startKey, specDim, outDim,
            input_lower, input_upper); // Returns [spec, out]

        if (alpha_tensor.defined() && alpha_tensor.numel() > 0) {
            // Clamp alpha to [0,1], keep per-spec
            auto alpha_clamped = torch::clamp(alpha_tensor, 0.0, 1.0); // [spec, out]

            // Broadcast unstable mask to [spec, out]
            auto unstable_spec = unstable.reshape({1, -1}).expand({specDim, outDim});

            // Per-spec α on unstable, 0 elsewhere
            auto alpha_spec = torch::where(unstable_spec, alpha_clamped, torch::zeros_like(alpha_clamped)); // [spec, out]

            // Per-spec "upper" slope for the A<0 branch
            auto k_upper_spec = upper_d.reshape({1, -1}).expand_as(alpha_spec); // [spec, out]

            // Write per-spec lower-path choices
            result.lb_lower_d = alpha_spec;      // used when A ≥ 0
            result.lb_upper_d = k_upper_spec;    // used when A < 0

            // Write per-spec upper-path choices
            result.ub_upper_d = k_upper_spec;    // used when A ≥ 0
            result.ub_lower_d = alpha_spec;      // used when A < 0

            // Biases:
            // For alpha-CROWN lower bound:
            //   - When A ≥ 0: use (alpha, 0) -> bias_lower = 0
            //   - When A < 0: use (k_upper, bias_upper) -> need the secant bias
            // So bias_lower is used for Apos branch (always 0), bias_upper for Aneg branch
            auto b_upper = -input_lower * upper_d; // [out] - secant bias
            result.bias_lower = torch::zeros_like(input_lower); // [out] - for Apos (alpha) branch
            result.bias_upper = b_upper;                         // [out] - for Aneg (secant) branch
        }
    }

    // Apply upper bound slopes (secant line for unstable neurons)
    if (!isAlphaOptimizationEnabled() || !_alphaCrownAnalysis) {
        result.d_upper = torch::where(unstable, upper_d, result.d_upper);
        auto bU_fallback = -input_lower * upper_d;
        result.bias_upper = torch::where(unstable, bU_fallback, result.bias_upper);
    } else {
        result.d_upper = torch::where(unstable, upper_d, result.d_upper);
        auto bU_alpha = -input_lower * upper_d;
        result.bias_upper = torch::where(unstable, bU_alpha, result.bias_upper);
    }

    // Apply masking for stable neurons (always active/inactive)
    auto always_active_mask = input_lower >= 0;
    auto always_inactive_mask = input_upper <= 0;

    result.d_lower = torch::where(always_active_mask, torch::ones_like(result.d_lower), result.d_lower);
    result.d_lower = torch::where(always_inactive_mask, torch::zeros_like(result.d_lower), result.d_lower);
    result.bias_lower = torch::where(always_active_mask | always_inactive_mask,
                                   torch::zeros_like(result.bias_lower), result.bias_lower);

    result.d_upper = torch::where(always_active_mask, torch::ones_like(result.d_upper), result.d_upper);
    result.d_upper = torch::where(always_inactive_mask, torch::zeros_like(result.d_upper), result.d_upper);
    result.bias_upper = torch::where(always_active_mask | always_inactive_mask,
                                   torch::zeros_like(result.bias_upper), result.bias_upper);

    // Also override lb_lower_d and lb_upper_d for stable neurons
    if (result.lb_lower_d.defined()) {
        result.lb_lower_d = torch::where(always_active_mask, torch::ones_like(result.lb_lower_d), result.lb_lower_d);
        result.lb_lower_d = torch::where(always_inactive_mask, torch::zeros_like(result.lb_lower_d), result.lb_lower_d);
    }
    if (result.lb_upper_d.defined()) {
        result.lb_upper_d = torch::where(always_active_mask, torch::ones_like(result.lb_upper_d), result.lb_upper_d);
        result.lb_upper_d = torch::where(always_inactive_mask, torch::zeros_like(result.lb_upper_d), result.lb_upper_d);
    }
}

// Alpha-aware relaxation computation (implementing BoundedAlphaOptimizeNode interface)
void BoundedReLUNode::computeAlphaRelaxation(
    const torch::Tensor& last_lA,
    const torch::Tensor& last_uA,
    const torch::Tensor& input_lower,
    const torch::Tensor& input_upper,
    torch::Tensor& d_lower,
    torch::Tensor& d_upper,
    torch::Tensor& bias_lower,
    torch::Tensor& bias_upper) {
    
    // Use the unified backward relaxation method
    auto result = _backwardRelaxation(last_lA, last_uA, input_lower, input_upper);
    
    // Extract the computed slopes and biases
    d_lower = result.d_lower;
    d_upper = result.d_upper;
    bias_lower = result.bias_lower;
    bias_upper = result.bias_upper;
}

// Get CROWN slopes for alpha initialization (following auto_LiRPA approach)
torch::Tensor BoundedReLUNode::getCROWNSlope(bool isLowerBound) const
{
    if (isLowerBound) {
        if (!hasInitD()) {
            // Return default lower slopes if CROWN slopes not available
            return torch::full({getOutputSize()}, 0.5f, torch::kFloat32);
        }
        return init_d;  // Return actual lower bound slopes
    } else {
        if (!init_upper_d.defined() || init_upper_d.numel() == 0) {
            // Return default upper slopes if CROWN slopes not available
            return torch::full({getOutputSize()}, 1.0f, torch::kFloat32);
        }
        return init_upper_d;  // Return actual upper bound slopes
    }
}

} // namespace NLR