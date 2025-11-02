/*********************                                                        */
/*! \file stat.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief sys/stat wrapper for LIRPA (mock-free version)
 **
 ** This is a simplified version without CxxTest mocks for standalone LIRPA.
 **/

#ifndef __T__sys__stat_h__
#define __T__sys__stat_h__

// Direct passthrough to standard library - no mocking needed for LIRPA
#include <sys/stat.h>

namespace T {
    // stat function - direct passthrough
    inline int stat(const char *pathname, struct ::stat *statbuf) {
        return ::stat(pathname, statbuf);
    }
}

#endif // __T__sys__stat_h__
