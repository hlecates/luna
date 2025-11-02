/*********************                                                        */
/*! \file LirpaError.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** Error class for the standalone LIRPA (auto-LiRPA) pipeline.
 ** This replaces NLRError for LIRPA-specific components to decouple
 ** from the Marabou NLR engine.
 **
 **/

#ifndef __LirpaError_h__
#define __LirpaError_h__

#include "Error.h"

class LirpaError : public Error
{
public:
    enum Code {
        // Analysis errors
        UNEXPECTED_TENSOR_SHAPE = 0,
        INVALID_NODE_TYPE = 1,
        INVALID_BOUND_COMPUTATION = 2,
        ALPHA_OPTIMIZATION_FAILED = 3,

        // Model construction errors
        ONNX_PARSING_ERROR = 4,
        UNSUPPORTED_OPERATION = 5,
        INVALID_MODEL_STRUCTURE = 6,

        // Bound propagation errors
        IBP_COMPUTATION_FAILED = 7,
        CROWN_BACKWARD_FAILED = 8,
        CONCRETE_BOUNDS_FAILED = 9,

        // Node errors
        NODE_NOT_FOUND = 10,
        INVALID_NODE_INDEX = 11,
        UNINITIALIZED_NODE = 12,

        // Generic errors
        INTERNAL_ERROR = 99
    };

    LirpaError( LirpaError::Code code )
        : Error( "LirpaError", (int)code )
    {
    }

    LirpaError( LirpaError::Code code, const char *userMessage )
        : Error( "LirpaError", (int)code, userMessage )
    {
    }
};

#endif // __LirpaError_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
