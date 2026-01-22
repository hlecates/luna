#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedReLUNode.h"
#include "src/common/BoundedTensor.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class ReLUNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        _reluModule = torch::nn::ReLU();
        _reluNode = std::make_shared<BoundedReLUNode>(_reluModule, "test_relu");
        _reluNode->setNodeIndex(0);
        _reluNode->setInputSize(10);
        _reluNode->setOutputSize(10);
    }

    torch::nn::ReLU _reluModule;
    std::shared_ptr<BoundedReLUNode> _reluNode;
};

// Forward pass tests
TEST_F(ReLUNodeTest, ForwardPassPositiveInput) {
    torch::Tensor input = torch::ones({1, 10});
    torch::Tensor output = _reluNode->forward(input);
    EXPECT_TENSORS_CLOSE(output, input, 1e-5, 1e-4);
}

TEST_F(ReLUNodeTest, ForwardPassNegativeInput) {
    torch::Tensor input = -torch::ones({1, 10});
    torch::Tensor output = _reluNode->forward(input);
    EXPECT_TENSORS_CLOSE(output, torch::zeros({1, 10}), 1e-5, 1e-4);
}

TEST_F(ReLUNodeTest, ForwardPassMixedInput) {
    torch::Tensor input = torch::tensor({-2.0, -1.0, 0.0, 1.0, 2.0}).unsqueeze(0);
    torch::Tensor expected = torch::tensor({0.0, 0.0, 0.0, 1.0, 2.0}).unsqueeze(0);
    torch::Tensor output = _reluNode->forward(input);
    EXPECT_TENSORS_CLOSE(output, expected, 1e-5, 1e-4);
}

// IBP tests
TEST_F(ReLUNodeTest, IBPAlwaysActiveRegion) {
    torch::Tensor lb = torch::ones({1, 10}) * 1.0f;
    torch::Tensor ub = torch::ones({1, 10}) * 2.0f;

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append(BoundedTensor<torch::Tensor>(lb, ub));

    auto result = _reluNode->computeIntervalBoundPropagation(inputBounds);

    EXPECT_TENSORS_CLOSE(result.lower(), lb, 1e-5, 1e-4);
    EXPECT_TENSORS_CLOSE(result.upper(), ub, 1e-5, 1e-4);
}

TEST_F(ReLUNodeTest, IBPAlwaysInactiveRegion) {
    torch::Tensor lb = -torch::ones({1, 10}) * 2.0f;
    torch::Tensor ub = -torch::ones({1, 10}) * 1.0f;

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append(BoundedTensor<torch::Tensor>(lb, ub));

    auto result = _reluNode->computeIntervalBoundPropagation(inputBounds);

    EXPECT_TENSORS_CLOSE(result.lower(), torch::zeros({1, 10}), 1e-5, 1e-4);
    EXPECT_TENSORS_CLOSE(result.upper(), torch::zeros({1, 10}), 1e-5, 1e-4);
}

TEST_F(ReLUNodeTest, IBPUnstableRegion) {
    // Input spans negative and positive: [-1, 1]
    torch::Tensor lb = -torch::ones({1, 10});
    torch::Tensor ub = torch::ones({1, 10});

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append(BoundedTensor<torch::Tensor>(lb, ub));

    auto result = _reluNode->computeIntervalBoundPropagation(inputBounds);

    // Lower bound should be 0 (inactive part clipped)
    EXPECT_TENSORS_CLOSE(result.lower(), torch::zeros({1, 10}), 1e-5, 1e-4);
    // Upper bound should be ub (active part preserved)
    EXPECT_TENSORS_CLOSE(result.upper(), ub, 1e-5, 1e-4);
}

TEST_F(ReLUNodeTest, IBPBoundWidth) {
    // Test that bound width doesn't increase (soundness property)
    torch::Tensor lb = torch::tensor({-0.5, -1.0, 0.0}).unsqueeze(0);
    torch::Tensor ub = torch::tensor({0.5, 1.0, 2.0}).unsqueeze(0);

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append(BoundedTensor<torch::Tensor>(lb, ub));

    auto result = _reluNode->computeIntervalBoundPropagation(inputBounds);
    
    torch::Tensor inputWidth = ub - lb;
    torch::Tensor outputWidth = result.upper() - result.lower();
    
    // Output width should be <= input width for each element
    auto flatInputWidth = inputWidth.flatten();
    auto flatOutputWidth = outputWidth.flatten();
    for (int64_t i = 0; i < flatInputWidth.numel(); ++i) {
        EXPECT_LE(flatOutputWidth[i].item<float>(), flatInputWidth[i].item<float>() + 1e-5);
    }
}
