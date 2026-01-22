#ifndef __VnnLibInputParser_h__
#define __VnnLibInputParser_h__

#include "MString.h"
#include "Vector.h"
#include "BoundedTensor.h"
#include "OutputConstraint.h"

// Undefine macros to avoid conflicts with PyTorch
#ifdef Warning
#undef Warning
#endif
#ifdef LOG
#undef LOG
#endif

#include <torch/torch.h>
#include <utility>

class VnnLibInputParser
{
public:
    /**
     * Parse input bounds from a VNN-LIB file.
     *
     * @param vnnlibFilePath Path to the .vnnlib file
     * @param expectedInputSize Expected number of input variables (for validation)
     * @return BoundedTensor containing lower and upper bounds for each input
     *
     * The parser extracts only input constraints (X_i variables) and ignores
     * output constraints (Y_i variables). It handles:
     * - (declare-const X_i Real)
     * - (assert (>= X_i value)) -> sets lower bound
     * - (assert (<= X_i value)) -> sets upper bound
     * - (assert (and ...)) -> multiple constraints
     *
     * Inputs without explicit bounds default to [-inf, +inf].
     */
    static BoundedTensor<torch::Tensor> parseInputBounds(const String &vnnlibFilePath,
                                                          unsigned expectedInputSize);

    /**
     * Parse output constraints from a VNN-LIB file.
     *
     * @param vnnlibFilePath Path to the .vnnlib file
     * @param expectedOutputSize Expected number of output variables
     * @return OutputConstraintSet containing parsed output constraints
     *
     * The parser extracts output constraints (Y_i variables) and converts them
     * to a format suitable for specification matrix generation. It handles:
     * - Simple bounds: (assert (>= Y_0 3.5))
     * - Comparisons: (assert (<= Y_0 Y_1)) -> Y_0 - Y_1 <= 0
     * - Linear combinations: (assert (>= (+ Y_0 (* -1 Y_1)) 0)) -> Y_0 - Y_1 >= 0
     */
    static NLR::OutputConstraintSet parseOutputConstraints(const String &vnnlibFilePath,
                                                           unsigned expectedOutputSize);

    /**
     * Parse both input bounds and output constraints from a VNN-LIB file.
     *
     * @param vnnlibFilePath Path to the .vnnlib file
     * @param expectedInputSize Expected number of input variables
     * @param expectedOutputSize Expected number of output variables
     * @return Pair of (input bounds, output constraints)
     *
     * This is more efficient than calling parseInputBounds and parseOutputConstraints
     * separately as it only reads and tokenizes the file once.
     */
    static std::pair<BoundedTensor<torch::Tensor>, NLR::OutputConstraintSet>
    parseInputAndOutputConstraints(const String &vnnlibFilePath,
                                   unsigned expectedInputSize,
                                   unsigned expectedOutputSize);

private:
    /**
     * Internal structure for tracking input bounds during parsing
     */
    struct InputBoundInfo {
        bool hasLowerBound;
        bool hasUpperBound;
        double lowerBound;  // Keep as double for accurate parsing
        double upperBound;  // Keep as double for accurate parsing

        InputBoundInfo()
            : hasLowerBound(false), hasUpperBound(false),
              lowerBound(-std::numeric_limits<double>::infinity()),
              upperBound(std::numeric_limits<double>::infinity()) {}
    };

    /**
     * Read and clean the VNN-LIB file contents
     */
    static String readVnnlibFile(const String &vnnlibFilePath);

    /**
     * Tokenize the VNN-LIB content
     */
    static Vector<String> tokenize(const String &vnnlibContent);

    /**
     * Parse the token stream and extract input bounds
     */
    static void parseTokens(const Vector<String> &tokens,
                           Vector<InputBoundInfo> &inputBounds);

    /**
     * Parse a single command (declare-const or assert)
     */
    static int parseCommand(int index,
                           const Vector<String> &tokens,
                           Vector<InputBoundInfo> &inputBounds);

    /**
     * Parse a declare-const command to identify input variables
     */
    static int parseDeclareConst(int index,
                                 const Vector<String> &tokens,
                                 Vector<InputBoundInfo> &inputBounds);

    /**
     * Parse an assert command to extract input bounds
     */
    static int parseAssert(int index,
                          const Vector<String> &tokens,
                          Vector<InputBoundInfo> &inputBounds);

    /**
     * Parse a condition (<=, >=, and)
     */
    static int parseCondition(int index,
                             const Vector<String> &tokens,
                             Vector<InputBoundInfo> &inputBounds);

    /**
     * Extract variable index from name (e.g., "X_0" -> 0)
     */
    static int extractVariableIndex(const String &varName);

    /**
     * Check if a variable name represents an input (X_i)
     */
    static bool isInputVariable(const String &varName);

    /**
     * Check if a variable name represents an output (Y_i)
     */
    static bool isOutputVariable(const String &varName);

    /**
     * Extract scalar value from string
     */
    static double extractScalar(const String &token);

    /**
     * Parse the token stream and extract output constraints.
     * Similar to parseTokens but focuses on Y_i variables.
     */
    static void parseOutputTokens(const Vector<String> &tokens,
                                  NLR::OutputConstraintSet &outputConstraints);

    /**
     * Parse a single command for output constraints (assert only)
     */
    static int parseOutputCommand(int index,
                                  const Vector<String> &tokens,
                                  NLR::OutputConstraintSet &outputConstraints);

    /**
     * Parse an assert command for output constraints
     */
    static int parseOutputAssert(int index,
                                 const Vector<String> &tokens,
                                 NLR::OutputConstraintSet &outputConstraints);

    /**
     * Parse an output condition (<=, >=, and, or) and add constraints
     */
    static int parseOutputCondition(int index,
                                    const Vector<String> &tokens,
                                    NLR::OutputConstraintSet &outputConstraints);

    /**
     * Parse a single OR branch (which may be an AND conjunction or a single constraint)
     * and return the constraints from that branch.
     */
    static int parseOutputBranch(int index,
                                  const Vector<String> &tokens,
                                  Vector<NLR::OutputConstraint> &branchConstraints,
                                  unsigned outputDim);

    /**
     * Parse a linear expression of output variables.
     * Handles forms like: Y_i, (+ Y_i Y_j), (+ Y_i 2.0), (* coeff Y_i)
     *
     * @param index Current token index (pointing to start of expression)
     * @param tokens Token stream
     * @param terms Output vector to receive parsed terms
     * @param scalarSum Output parameter to accumulate scalar values found in expression
     * @return Updated token index after parsing
     */
    static int parseLinearExpression(int index,
                                     const Vector<String> &tokens,
                                     Vector<NLR::OutputTerm> &terms,
                                     double &scalarSum);

    /**
     * Check if a token is a scalar value (number)
     */
    static bool isScalar(const String &token);

    /**
     * Parse tokens for both input bounds and output constraints simultaneously.
     */
    static void parseTokensBoth(const Vector<String> &tokens,
                                Vector<InputBoundInfo> &inputBounds,
                                NLR::OutputConstraintSet &outputConstraints);

    /**
     * Parse a command for both inputs and outputs
     */
    static int parseCommandBoth(int index,
                                const Vector<String> &tokens,
                                Vector<InputBoundInfo> &inputBounds,
                                NLR::OutputConstraintSet &outputConstraints);

    /**
     * Parse an assert command for both inputs and outputs
     */
    static int parseAssertBoth(int index,
                               const Vector<String> &tokens,
                               Vector<InputBoundInfo> &inputBounds,
                               NLR::OutputConstraintSet &outputConstraints);

    /**
     * Parse a condition for both inputs and outputs
     */
    static int parseConditionBoth(int index,
                                  const Vector<String> &tokens,
                                  Vector<InputBoundInfo> &inputBounds,
                                  NLR::OutputConstraintSet &outputConstraints);
};

#endif // __VnnLibInputParser_h__
