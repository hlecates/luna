#include "BoundedConvTransposeNode.h"
#include "conv/MatrixConvolution.h"
#include "conv/Patches.h"
#include <torch/nn/functional.h>
#include <stdexcept>
#include <cmath>

namespace NLR {

// Constructor for ConvTranspose2d
BoundedConvTransposeNode::BoundedConvTransposeNode(const torch::nn::ConvTranspose2d& convTransposeModule,
                                                   ConvMode mode,
                                                   const String& name)
    : convtranspose2d(convTransposeModule), mode(mode) {

    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;

    initializeFromConvTranspose2d(convTransposeModule);
}

void BoundedConvTransposeNode::initializeFromConvTranspose2d(const torch::nn::ConvTranspose2d& convTransposeModule) {
    if (!convTransposeModule) {
        throw std::runtime_error("ConvTranspose2d module is null");
    }

    // Extract convolution parameters
    auto options = convTransposeModule->options;

    // Set padding (ensure same padding on both sides for now)
    if (std::holds_alternative<torch::ExpandingArray<2>>(options.padding())) {
        auto pad_array = std::get<torch::ExpandingArray<2>>(options.padding());
        padding = {static_cast<int>((*pad_array)[0]),
                   static_cast<int>((*pad_array)[1])};
    } else {
        // Handle other padding types (kValid, kSame) for now default to 0
        padding = {0, 0};
    }

    // Set stride
    stride = {static_cast<int>((*options.stride())[0]),
              static_cast<int>((*options.stride())[1])};

    // Set dilation
    dilation = {static_cast<int>((*options.dilation())[0]),
               static_cast<int>((*options.dilation())[1])};

    // Set output_padding
    auto out_pad_array = *options.output_padding();
    output_padding = {static_cast<int>(out_pad_array[0]),
                     static_cast<int>(out_pad_array[1])};

    // Set groups
    groups = options.groups();

    // Check for bias
    has_bias = options.bias();

    // Apply assertions from auto_LiRPA implementation
    if (output_padding[0] != 0 || output_padding[1] != 0) {
        throw std::runtime_error("BoundedConvTransposeNode: output_padding must be [0, 0]");
    }
    if (dilation[0] != 1 || dilation[1] != 1) {
        throw std::runtime_error("BoundedConvTransposeNode: dilation must be [1, 1]");
    }
    if (stride[0] != stride[1]) {
        throw std::runtime_error("BoundedConvTransposeNode: stride[0] must equal stride[1]");
    }
    if (groups != 1) {
        throw std::runtime_error("BoundedConvTransposeNode: groups must be 1");
    }

    // Fix weight tensor properties for Alpha-CROWN compatibility
    if (convtranspose2d && convtranspose2d->weight.defined()) {
        auto weight = convtranspose2d->weight;
        bool weightNeedsFix = false;
        if (!weight.requires_grad() || !weight.is_contiguous() || weight.dtype() != torch::kFloat32) {
            weightNeedsFix = true;
            if (!weight.requires_grad()) {
            }
            if (!weight.is_contiguous()) {
            }
            if (weight.dtype() != torch::kFloat32) {
            }
        }
        
        if (weightNeedsFix) {
            convtranspose2d->weight = weight.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network weights are constants;
        }
        
        // Similar checks and fixes for bias if it exists
        if (has_bias && convtranspose2d->bias.defined()) {
            auto bias = convtranspose2d->bias;
            bool biasNeedsFix = false;
            if (!bias.requires_grad() || !bias.is_contiguous() || bias.dtype() != torch::kFloat32) {
                biasNeedsFix = true;
                if (!bias.requires_grad()) {
                }
                if (!bias.is_contiguous()) {
                }
                if (bias.dtype() != torch::kFloat32) {
                }
            }
            
            if (biasNeedsFix) {
                convtranspose2d->bias = bias.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network biases are constants;
            }
        }
    }
}

// Forward pass
torch::Tensor BoundedConvTransposeNode::forward(const torch::Tensor& input) {
    // Convert input to float32 and ensure contiguous
    torch::Tensor inputFloat = input.to(torch::kFloat32).contiguous();

    // Update input/output shapes
    input_shape.clear();
    for (int i = 0; i < input.dim(); ++i) {
        input_shape.push_back(input.size(i));
    }

    if (!convtranspose2d) {
        throw std::runtime_error("ConvTranspose2d module not initialized");
    }

    // Get weight and bias
    torch::Tensor weight = convtranspose2d->weight.to(torch::kFloat32).contiguous();
    torch::Tensor bias = has_bias ? convtranspose2d->bias.to(torch::kFloat32) : torch::Tensor();

    // Convert to int64_t
    std::vector<int64_t> stride_64(stride.begin(), stride.end());
    std::vector<int64_t> padding_64(padding.begin(), padding.end());
    std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());
    std::vector<int64_t> output_padding_64(output_padding.begin(), output_padding.end());

    // Apply transposed convolution
    c10::optional<torch::Tensor> bias_opt = bias.defined() ? c10::optional<torch::Tensor>(bias) : c10::nullopt;
    torch::Tensor output = at::conv_transpose2d(
        inputFloat, weight, bias_opt,
        stride_64, padding_64, output_padding_64, groups, dilation_64
    );

    // Update output shape
    output_shape.clear();
    for (int i = 0; i < output.dim(); ++i) {
        output_shape.push_back(output.size(i));
    }

    // Update sizes for getInputSize/getOutputSize
    _input_size = input.numel() / input.size(0);  // Size per batch
    _output_size = output.numel() / output.size(0);  // Size per batch

    return output;
}

void BoundedConvTransposeNode::moveToDevice(const torch::Device& device)
{
    BoundedTorchNode::moveToDevice(device);
    if (convtranspose2d) {
        convtranspose2d->to(device);
    }
}

// Backward bound propagation
void BoundedConvTransposeNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {
        
    // Fix for missing input_shape - Forward pass is not always run before doing CROWN
    if (input_shape.empty()) {
        if (inputBounds.size() < 1) {
             throw std::runtime_error("BoundedConvTransposeNode: input_shape empty and no input bounds provided");
        }
        
        auto& lb = inputBounds[0].lower();
        // Infer shape from total size and weights
        // weight: [in_c, out_c, k_h, k_w] for ConvTranspose
        torch::Tensor weight = convtranspose2d->weight;
        int64_t in_channels = weight.size(0);
        int64_t total_input_size = lb.numel();
        
        // Assume [Batch, C, H, W]
        int64_t spatial_dim_sq = total_input_size / in_channels; // H * W
        int64_t H = static_cast<int64_t>(std::sqrt(spatial_dim_sq));
        int64_t W = H;
        
        if (in_channels * H * W != total_input_size) {
        }
        
        input_shape = {1, static_cast<int>(in_channels), static_cast<int>(H), static_cast<int>(W)};
        
        // Compute output shape for ConvTranspose
        int64_t out_channels = weight.size(1);
        int64_t kernel_h = weight.size(2);
        int64_t kernel_w = weight.size(3);
        
        // ConvTranspose output formula: (H-1)*stride - 2*pad + kernel + output_padding
        int64_t out_h = (H - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0];
        int64_t out_w = (W - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1];
        
        output_shape = {1, static_cast<int>(out_channels), static_cast<int>(out_h), static_cast<int>(out_w)};
        
        _input_size = total_input_size;
        _output_size = out_channels * out_h * out_w;
    }

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedConvTransposeNode expects at least one input");
    }

    // Get weight and bias from ConvTranspose2d
    if (!convtranspose2d) {
        throw std::runtime_error("ConvTranspose2d module not initialized");
    }
    torch::Tensor weight = convtranspose2d->weight.to(torch::kFloat32);
    torch::Tensor bias = has_bias ? convtranspose2d->bias.to(torch::kFloat32) : torch::Tensor();

    // Compute bounds for lower and upper
    torch::Tensor lA_bias_contrib, uA_bias_contrib;
    BoundA lA_x = boundOneSide(last_lA, weight, bias, lA_bias_contrib);
    BoundA uA_x = boundOneSide(last_uA, weight, bias, uA_bias_contrib);

    // Flatten output matrices if input bounds are flat
    if (inputBounds.size() > 0 && inputBounds[0].lower().dim() == 1) {
        if (lA_x.isTensor()) {
            torch::Tensor t = lA_x.asTensor();
            if (t.dim() == 4) { // [B, C, H, W]
                 lA_x = BoundA(t.reshape({t.size(0), -1}));
            } else if (t.dim() == 5) { // [B, S, C, H, W]
                 lA_x = BoundA(t.reshape({t.size(0), t.size(1), -1}));
            }
        }
        if (uA_x.isTensor()) {
             torch::Tensor t = uA_x.asTensor();
             if (t.dim() == 4) {
                 uA_x = BoundA(t.reshape({t.size(0), -1}));
            } else if (t.dim() == 5) {
                 uA_x = BoundA(t.reshape({t.size(0), t.size(1), -1}));
            }
        }
    }

    // Prepare output matrices
    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(lA_x, uA_x));

    // Handle bias accumulation
    lbias = lA_bias_contrib;
    ubias = uA_bias_contrib;
}

BoundA BoundedConvTransposeNode::boundOneSide(const BoundA& last_A,
                                               const torch::Tensor& weight,
                                               const torch::Tensor& bias,
                                               torch::Tensor& sum_bias) {
    if (!last_A.defined()) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
        sum_bias = torch::zeros({1}, options);
        return BoundA();
    }

    if (last_A.isTensor()) {
        // Tensor mode: Use regular F.conv2d to propagate backward
        torch::Tensor last_A_tensor = last_A.asTensor();
        
        // Reshape last_A for conv2d
        auto shape = last_A_tensor.sizes().vec();
        
        torch::Tensor reshaped_last_A;
        
        if (shape.size() == 5) {
             reshaped_last_A = last_A_tensor.reshape({shape[0] * shape[1], shape[2], shape[3], shape[4]});
        } else if (shape.size() == 3 && output_shape.size() >= 4) {
             reshaped_last_A = last_A_tensor.reshape({shape[0] * shape[1], output_shape[1], output_shape[2], output_shape[3]});
        } else if (shape.size() == 2 && output_shape.size() >= 4) {
             reshaped_last_A = last_A_tensor.reshape({shape[0], output_shape[1], output_shape[2], output_shape[3]});
        } else {
             reshaped_last_A = last_A_tensor;
        }

        // Convert to int64_t
        std::vector<int64_t> stride_64(stride.begin(), stride.end());
        std::vector<int64_t> padding_64(padding.begin(), padding.end());
        std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());

        // Apply regular conv2d for backward pass (note: no output_padding for conv2d)
        torch::Tensor next_A = torch::nn::functional::conv2d(
            reshaped_last_A, weight,
            torch::nn::functional::Conv2dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        );

        // Reshape back
        if (shape.size() == 5) {
            next_A = next_A.view({shape[0], shape[1], next_A.size(1), next_A.size(2), next_A.size(3)});
        } else if (shape.size() == 3 && output_shape.size() >= 4) {
            next_A = next_A.view({shape[0], shape[1], next_A.size(1), next_A.size(2), next_A.size(3)});
        } else if (shape.size() == 2 && output_shape.size() >= 4) {
            next_A = next_A.view({shape[0], next_A.size(1), next_A.size(2), next_A.size(3)});
        }
        
        // Handle bias: sum_bias = (last_A.sum((3, 4)) * x[2].lower).sum(2)
        if (has_bias && bias.defined()) {
            if (shape.size() == 5) {
                // [S, B, C, H, W]
                torch::Tensor sum_spatial = last_A_tensor.sum({3, 4}); // [S, B, C]
                torch::Tensor bias_expanded = bias.unsqueeze(0).unsqueeze(0); // [1, 1, C]
                torch::Tensor product = sum_spatial * bias_expanded; // [S, B, C]
                sum_bias = product.sum(-1); // [S, B]
            } else if (shape.size() == 4) {
                // [B, C, H, W]
                torch::Tensor sum_spatial = last_A_tensor.sum({2, 3}); // [B, C]
                torch::Tensor bias_expanded = bias.unsqueeze(0); // [1, C]
                torch::Tensor product = sum_spatial * bias_expanded; // [B, C]
                sum_bias = product.sum(-1); // [B]
            } else {
                sum_bias = torch::zeros({1}, last_A_tensor.options());
            }
        } else {
            // No bias
            if (shape.size() == 5) {
                sum_bias = torch::zeros({shape[0], shape[1]}, last_A_tensor.options());
            } else if (shape.size() == 4) {
                sum_bias = torch::zeros({shape[0]}, last_A_tensor.options());
            } else {
                sum_bias = torch::zeros({1}, last_A_tensor.options());
            }
        }

        return BoundA(next_A);
    } else {
        // Patches mode
        auto last_patches = last_A.asPatches();
        
        torch::Tensor pieces;
        
        if (last_patches->identity == 0) {
            torch::Tensor patches_tensor = last_patches->patches;
            
            if (has_bias && bias.defined()) {
                torch::Tensor bias_view = bias.view({-1, 1, 1}); // [c, 1, 1]
                sum_bias = (patches_tensor * bias_view).sum({-3, -2, -1});
            } else {
                sum_bias = torch::zeros({1}, patches_tensor.options()); 
            }
            
            // Flatten patches
            int64_t C = patches_tensor.size(-3);
            int64_t H = patches_tensor.size(-2);
            int64_t W = patches_tensor.size(-1);
            
            torch::Tensor flattened = patches_tensor.reshape({-1, C, H, W});
            
            // For ConvTranspose backward: transpose and flip weights
            torch::Tensor weight_processed = weight.transpose(0, 1).flip({-1, -2});
            weight_processed = insert_zeros(weight_processed, last_patches->inserted_zeros);
            
            std::vector<int64_t> stride_64(stride.begin(), stride.end());
            pieces = at::conv_transpose2d(
                flattened, weight_processed, c10::nullopt,
                stride_64, {0, 0}, {0, 0}, 1, {1, 1}
            );
            
            // Reshape pieces back
            std::vector<int64_t> new_shape;
            for(int i=0; i<patches_tensor.dim()-3; ++i) new_shape.push_back(patches_tensor.size(i));
            new_shape.push_back(pieces.size(-3));
            new_shape.push_back(pieces.size(-2));
            new_shape.push_back(pieces.size(-1));
            
            pieces = pieces.view(new_shape);
            
        } else if (last_patches->identity == 1) {
            // Identity patches
            if (last_patches->unstable_idx.has_value()) {
                throw std::runtime_error("BoundedConvTransposeNode: Sparse identity patches not implemented");
            } else {
                pieces = weight.view({weight.size(0), 1, 1, 1, weight.size(1), weight.size(2), weight.size(3)});
                // Expand
                std::vector<int64_t> expand_dims = {
                    weight.size(0), 
                    last_patches->output_shape[1], // batch
                    last_patches->output_shape[2], // out_h
                    last_patches->output_shape[3], // out_w
                    weight.size(1), weight.size(2), weight.size(3)
                };
                pieces = pieces.expand(expand_dims);
                
                // Bias
                if (has_bias) {
                    sum_bias = bias.view({-1, 1, 1, 1}).expand({
                        weight.size(0), last_patches->output_shape[1], last_patches->output_shape[2], last_patches->output_shape[3]
                    });
                } else {
                    sum_bias = torch::zeros({1}, weight.options());
                }
            }
        } else {
            throw std::runtime_error("BoundedConvTransposeNode: Unknown patches identity value");
        }
        
        // Compute new padding/stride/output_padding
        std::vector<int64_t> new_padding_vec, new_stride_vec, new_output_padding_vec;
        
        std::vector<int64_t> patches_padding = unify_shape(last_patches->padding);
        std::vector<int64_t> patches_output_padding = unify_shape(last_patches->output_padding);
        std::vector<int64_t> this_stride = {static_cast<int64_t>(stride[0]), static_cast<int64_t>(stride[1])};
        std::vector<int64_t> this_padding = {static_cast<int64_t>(padding[0]), static_cast<int64_t>(padding[1]), 
                                             static_cast<int64_t>(padding[0]), static_cast<int64_t>(padding[1])};
        
        int64_t inserted_zeros = last_patches->inserted_zeros;
        
        // Compute new padding - need to account for weight dimensions
        new_padding_vec.resize(4);
        for (int j = 0; j < 4; ++j) {
            new_padding_vec[j] = patches_padding[j] * (inserted_zeros + 1) + (weight.size(3 - j/2) - 1);
        }
        
        // Compute new output padding
        new_output_padding_vec.resize(4);
        for (int j = 0; j < 4; ++j) {
            new_output_padding_vec[j] = patches_output_padding[j] * (inserted_zeros + 1) + this_padding[j];
        }
        
        // Update inserted_zeros
        inserted_zeros = (inserted_zeros + 1) * this_stride[0] - 1;
        
        // Check if we should convert to matrix mode
        if (inserted_zeros == 0 && !is_shape_used(new_output_padding_vec) && 
            pieces.size(-1) > input_shape[3]) {
        }
        
        return BoundA(last_patches->create_similar(
            pieces,
            last_patches->stride,  // stride doesn't change
            new_padding_vec,
            new_output_padding_vec,
            inserted_zeros,
            std::nullopt, // identity
            std::nullopt  // input_shape
        ));
    }
}

// IBP computation (L-infinity norm only)
BoundedTensor<torch::Tensor> BoundedConvTransposeNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.empty()) {
        throw std::runtime_error("No input bounds provided for IBP");
    }

    const BoundedTensor<torch::Tensor>& input = inputBounds[0];

    // Get weight and bias from ConvTranspose2d
    torch::Tensor weight = convtranspose2d->weight.to(torch::kFloat32);
    torch::Tensor bias = has_bias ? convtranspose2d->bias.to(torch::kFloat32) : torch::Tensor();

    torch::Tensor input_lower = input.lower();
    torch::Tensor input_upper = input.upper();

    // Reshape if needed
    if (input_lower.dim() == 1 || (input_lower.dim() == 2 && input_lower.size(0) == 1)) {
        if (!input_shape.empty() && input_shape.size() >= 3) {
            if (input_shape.size() == 4) {
                input_lower = input_lower.reshape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
                input_upper = input_upper.reshape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
            } else if (input_shape.size() == 3) {
                input_lower = input_lower.reshape({input_shape[0], input_shape[1], input_shape[2]});
                input_upper = input_upper.reshape({input_shape[0], input_shape[1], input_shape[2]});
            }
        } else {
            // Infer shape
            int in_channels = weight.size(0);
            int total_elements = input_lower.numel();
            
            if (total_elements % in_channels == 0) {
                int spatial_size = total_elements / in_channels;
                int H = static_cast<int>(std::sqrt(spatial_size));
                int W = H;
                
                if (H * W == spatial_size) {
                    input_lower = input_lower.reshape({in_channels, H, W});
                    input_upper = input_upper.reshape({in_channels, H, W});
                } else {
                    throw std::runtime_error("Cannot infer proper input shape for ConvTranspose");
                }
            } else {
                throw std::runtime_error("Input size incompatible with ConvTranspose weight dimensions");
            }
        }
    }

    // Add batch dimension if needed
    if (input_lower.dim() == 3) {
        input_lower = input_lower.unsqueeze(0);
        input_upper = input_upper.unsqueeze(0);
    }

    // L-infinity norm IBP
    torch::Tensor mid = (input_upper + input_lower) / 2.0;
    torch::Tensor diff = (input_upper - input_lower) / 2.0;
    torch::Tensor weight_abs = weight.abs();

    std::vector<int64_t> stride_64(stride.begin(), stride.end());
    std::vector<int64_t> padding_64(padding.begin(), padding.end());
    std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());
    std::vector<int64_t> output_padding_64(output_padding.begin(), output_padding.end());

    torch::Tensor deviation = at::conv_transpose2d(
        diff, weight_abs, c10::nullopt,
        stride_64, padding_64, output_padding_64, groups, dilation_64
    );

    // Use at::conv_transpose2d which accepts bias as an optional parameter
    c10::optional<torch::Tensor> bias_opt = bias.defined() ? c10::optional<torch::Tensor>(bias) : c10::nullopt;
    torch::Tensor center = at::conv_transpose2d(
        mid, weight, bias_opt,
        stride_64, padding_64, output_padding_64, groups, dilation_64
    );

    torch::Tensor upper_bound = center + deviation;
    torch::Tensor lower_bound = center - deviation;

    // Flatten the output if the original input was flattened
    if (input.lower().dim() == 1 || (input.lower().dim() == 2 && input.lower().size(0) == 1)) {
        lower_bound = lower_bound.flatten();
        upper_bound = upper_bound.flatten();
    }

    return BoundedTensor<torch::Tensor>(lower_bound, upper_bound);
}

// Size getters and setters
unsigned BoundedConvTransposeNode::getInputSize() const {
    return _input_size;
}

unsigned BoundedConvTransposeNode::getOutputSize() const {
    return _output_size;
}

void BoundedConvTransposeNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedConvTransposeNode::setOutputSize(unsigned size) {
    _output_size = size;
}

unsigned BoundedConvTransposeNode::inferOutputSize(unsigned inputSize) const {
    if (!convtranspose2d) return 0;
    torch::Tensor weight = convtranspose2d->weight;
    
    int64_t in_channels = weight.size(0);
    int64_t out_channels = weight.size(1);
    
    if (in_channels == 0) return 0;
    
    int64_t spatial_dim_sq = inputSize / in_channels;
    if (spatial_dim_sq <= 0) return 0;
    
    int64_t H = static_cast<int64_t>(std::sqrt(spatial_dim_sq));
    int64_t W = H;
    
    if (in_channels * H * W != inputSize) {
    }
    
    int64_t kernel_h = weight.size(2);
    int64_t kernel_w = weight.size(3);
    
    // ConvTranspose output formula
    int64_t out_h = (H - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0];
    int64_t out_w = (W - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1];
    
    return static_cast<unsigned>(out_channels * out_h * out_w);
}

} // namespace NLR
