#include <gtest/gtest.h>
#include "src/engine/AlphaCROWNAnalysis.h"
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class AlphaCROWNAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        static bool threadsConfigured = false;
        if (!threadsConfigured) {
            at::set_num_threads(1);
            at::set_num_interop_threads(1);
            threadsConfigured = true;
        }
        
        // Configure AlphaCROWN
        LunaConfiguration::ALPHA_ITERATIONS = 5;
        LunaConfiguration::OPTIMIZE_LOWER = true;
        LunaConfiguration::OPTIMIZE_UPPER = false;
    }
};

TEST_F(AlphaCROWNAnalysisTest, BasicOptimization) {
    auto model = ModelBuilder::createMLP(3, {5, 5}, 2, true, false);
    
    auto inputBounds = BoundGenerator::epsilonBall(
        torch::zeros({1, 3}), 0.1);
    model->setInputBounds(inputBounds);
    
    NLR::AlphaCROWNAnalysis analysis(model.get());
    analysis.getCROWNAnalysis()->setInputBounds(inputBounds);
    
    // Test lower bound optimization
    torch::Tensor lowerBounds = analysis.computeOptimizedBounds(
        LunaConfiguration::BoundSide::Lower);
    
    EXPECT_GT(lowerBounds.numel(), 0);
}

TEST_F(AlphaCROWNAnalysisTest, AlphaCROWNTighterThanCROWN) {
    auto model = ModelBuilder::createMLP(4, {6}, 3, true, false);
    
    auto inputBounds = BoundGenerator::randomBounds({1, 4}, 0.0, 1.0, 0.15);
    model->setInputBounds(inputBounds);
    
    // Get CROWN bounds
    NLR::CROWNAnalysis crownAnalysis(model.get());
    crownAnalysis.setInputBounds(inputBounds);
    crownAnalysis.run();
    auto crownBounds = crownAnalysis.getOutputBounds();
    
    // Get AlphaCROWN bounds
    NLR::AlphaCROWNAnalysis alphaAnalysis(model.get());
    alphaAnalysis.getCROWNAnalysis()->setInputBounds(inputBounds);
    torch::Tensor alphaLower = alphaAnalysis.computeOptimizedBounds(
        LunaConfiguration::BoundSide::Lower);
    torch::Tensor alphaUpper = alphaAnalysis.getCROWNAnalysis()->getOutputBounds().upper();
    
    BoundedTensor<torch::Tensor> alphaBounds(alphaLower, alphaUpper);
    
    // AlphaCROWN should be at least as tight as CROWN
    EXPECT_TRUE(SoundnessChecker::alphaCrownTighterThanCrown(crownBounds, alphaBounds));
}
