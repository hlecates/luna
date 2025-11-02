/*********************                                                        */
/*! \file stdlib.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief Stdlib wrapper for LIRPA (mock-free version)
 **
 ** This is a simplified version without CxxTest mocks for standalone LIRPA.
 **/

#ifndef __T__Stdlib_h__
#define __T__Stdlib_h__

// Direct passthrough to standard library - no mocking needed for LIRPA
#include <cstdlib>

namespace T {
    // Memory management - direct passthrough
    inline void* malloc(size_t size) {
        return ::malloc(size);
    }

    inline void free(void* ptr) {
        ::free(ptr);
    }

    inline void* realloc(void* ptr, size_t size) {
        return ::realloc(ptr, size);
    }

    // Random number generation
    inline void srand(unsigned seed) {
        ::srand(seed);
    }

    inline int rand() {
        return ::rand();
    }
}

#endif // __T__Stdlib_h__

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
