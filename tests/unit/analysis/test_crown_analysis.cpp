#include <gtest/gtest.h>
#include "src/engine/CROWNAnalysis.h"
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class CROWNAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use deterministic seed for reproducibility
        torch::manual_seed(42);
        static bool threadsConfigured = false;
        if (!threadsConfigured) {
            at::set_num_threads(1);
            at::set_num_interop_threads(1);
            threadsConfigured = true;
        }
    }
};

TEST_F(CROWNAnalysisTest, IBPForwardBasic) {
    // Create a simple MLP: Input(2) -> Linear(2,3) -> ReLU -> Linear(3,2) -> Output
    auto model = ModelBuilder::createMLP(2, {3}, 2, true, false);
    
    auto inputBounds = BoundGenerator::epsilonBall(
        torch::zeros({1, 2}), 0.1);
    
    model->setInputBounds(inputBounds);
    
    NLR::CROWNAnalysis analysis(model.get());
    analysis.setInputBounds(inputBounds);
    analysis.computeIBPBounds();
    
    auto ibpBounds = analysis.getOutputIBPBounds();
    
    // Verify bounds are computed
    EXPECT_GT(ibpBounds.lower().numel(), 0);
    EXPECT_GT(ibpBounds.upper().numel(), 0);
    
    // Verify upper >= lower
    torch::Tensor width = ibpBounds.upper() - ibpBounds.lower();
    EXPECT_TRUE(torch::all(width >= 0).item<bool>());
}

TEST_F(CROWNAnalysisTest, CROWNTighterThanIBP) {
    auto model = ModelBuilder::createMLP(5, {10, 10}, 3, true, false);
    
    auto inputBounds = BoundGenerator::randomBounds({1, 5}, 0.0, 1.0, 0.2);
    model->setInputBounds(inputBounds);
    
    NLR::CROWNAnalysis analysis(model.get());
    analysis.setInputBounds(inputBounds);
    analysis.computeIBPBounds();
    auto ibpBounds = analysis.getOutputIBPBounds();
    
    analysis.run();
    auto crownBounds = analysis.getOutputBounds();
    
    // CROWN should be at least as tight as IBP
    EXPECT_TRUE(SoundnessChecker::crownTighterThanIBP(ibpBounds, crownBounds));
}

TEST_F(CROWNAnalysisTest, CROWNSoundnessSimple) {
    auto model = ModelBuilder::createMLP(5, {10}, 3, true, false);
    
    auto inputBounds = BoundGenerator::epsilonBall(
        torch::randn({1, 5}), 0.1);
    
    model->setInputBounds(inputBounds);
    
    NLR::CROWNAnalysis analysis(model.get());
    analysis.setInputBounds(inputBounds);
    analysis.run();
    
    auto outputBounds = analysis.getOutputBounds();
    
    // Verify soundness by checking center point
    EXPECT_TRUE(SoundnessChecker::centerContainedInBounds(
        inputBounds, outputBounds, *model));
}

TEST_F(CROWNAnalysisTest, BackwardPropagation) {
    auto model = ModelBuilder::createMLP(3, {5}, 2, true, false);
    
    auto inputBounds = BoundGenerator::wideBounds({1, 3}, 0.0, 1.0);
    model->setInputBounds(inputBounds);
    
    NLR::CROWNAnalysis analysis(model.get());
    analysis.setInputBounds(inputBounds);
    analysis.run();
    
    // Verify bounds computed successfully
    auto bounds = analysis.getOutputBounds();
    EXPECT_GT(bounds.lower().numel(), 0);
    EXPECT_GT(bounds.upper().numel(), 0);
}
