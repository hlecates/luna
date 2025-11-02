#ifndef __OPERATIONS_H__
#define __OPERATIONS_H__

#include <torch/torch.h>

namespace Operations {

class ReshapeImpl : public torch::nn::Module {
public:
    ReshapeImpl() {}
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& shape_tensor);
};
TORCH_MODULE(Reshape);

class ReshapeWrapper : public torch::nn::Module {
private:
    torch::Tensor shape_tensor;
public:
    ReshapeWrapper(torch::Tensor shape) : shape_tensor(shape) {
        register_buffer("shape", this->shape_tensor);
    }
    torch::Tensor forward(const torch::Tensor& input) {
        // Simple reshape implementation
        torch::Tensor flattened_shape = shape_tensor.flatten();
        std::vector<int64_t> new_shape;
        for (int64_t i = 0; i < flattened_shape.numel(); ++i) {
            new_shape.push_back(flattened_shape[i].item<int64_t>());
        }

        // Handle batch dimension properly
        // If input has batch dimension (first dim = 1), preserve it
        if (input.dim() > 0 && input.size(0) == 1) {
            // Insert batch size as first dimension
            new_shape.insert(new_shape.begin(), 1);
        }

        return input.reshape(new_shape);
    }
};

class FlattenWrapper : public torch::nn::Module {
private:
    int64_t axis;
public:
    FlattenWrapper(int64_t axis_val = 1) : axis(axis_val) {}

    torch::Tensor forward(const torch::Tensor& input) {
        // Flatten from axis onward
        // ONNX Flatten spec: Flattens the input tensor into a 2D matrix
        // Dimensions [0, axis) are flattened to the outer dimension
        // Dimensions [axis, rank) are flattened to the inner dimension

        int64_t actual_axis = axis;
        if (actual_axis < 0) {
            actual_axis = input.dim() + actual_axis;
        }

        // Clamp axis to valid range
        actual_axis = std::max(int64_t(0), std::min(actual_axis, int64_t(input.dim())));

        // Compute the two dimensions for the flattened output
        int64_t dim1 = 1;
        for (int64_t i = 0; i < actual_axis; ++i) {
            dim1 *= input.size(i);
        }

        int64_t dim2 = 1;
        for (int64_t i = actual_axis; i < input.dim(); ++i) {
            dim2 *= input.size(i);
        }

        // If one dimension is 1, just return a 1D tensor to match typical usage
        if (dim1 == 1) {
            return input.reshape({dim2});
        } else if (dim2 == 1) {
            return input.reshape({dim1});
        }

        // ONNX Flatten produces a 2D tensor
        return input.reshape({dim1, dim2});
    }
};

class Constant : public torch::nn::Module {
    torch::Tensor value;
public:
    Constant(torch::Tensor value) : value(value) {
        register_buffer("value", this->value);
    }
    torch::Tensor forward();
};

} // namespace Operations

#endif // __OPERATIONS_H__ 