/*********************                                                        */
/*! \file Errno.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief Errno wrapper for LIRPA (mock-free version)
 **
 ** This is a simplified version without CxxTest mocks for standalone LIRPA.
 **/

#ifndef __T__Errno_h__
#define __T__Errno_h__

#include <cerrno>

namespace T {
    // Direct passthrough to errno - no mocking needed for LIRPA
    inline int errorNumber() {
        return errno;
    }
}

#endif // __T__Errno_h__

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//
