/*********************                                                        */
/*! \file unistd.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief unistd wrapper for LIRPA (mock-free version)
 **
 ** This is a simplified version without CxxTest mocks for standalone LIRPA.
 **/

#ifndef __T__unistd_h__
#define __T__unistd_h__

// Direct passthrough to standard library - no mocking needed for LIRPA
#include <unistd.h>
#include <fcntl.h>

namespace T {
    // File operations - direct passthrough
    inline int open(const char *pathname, int flags) {
        return ::open(pathname, flags);
    }

    inline int open(const char *pathname, int flags, mode_t mode) {
        return ::open(pathname, flags, mode);
    }

    inline ssize_t read(int fd, void *buf, size_t count) {
        return ::read(fd, buf, count);
    }

    inline ssize_t write(int fd, const void *buf, size_t count) {
        return ::write(fd, buf, count);
    }

    inline int close(int fd) {
        return ::close(fd);
    }
}

#endif // __T__unistd_h__
