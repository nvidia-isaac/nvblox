
#pragma once

#include "glog/logging.h"

#ifdef PRE_CXX11_ABI_LINKABLE
/*
This section redefines some GLog macros with simpler variant.

This is needed so that pre-CXX11 ABI code can use and link againt nvblox.
Otherwise the use of these macros in header files (e.g. *_impl.h) injects
implementations into the pre-CXX11 ABI code of the nvblox user, which then
requires linking against GLog symbols with std::string arguments.
Since std::string has changed it's binary ABI in the CXX11 ABI, this results in
a linker error.

Currently PyTorch and other PyPi packages are built with manylinux1 standard
that forces using the old ABI. In the next year or two, they should switch to
manylinux_2_28 with the new ABI, at which point this file will be no longer
required.
*/

class NullBuffer : public std::streambuf {
 public:
  int overflow(int c) { return c; }
};

static NullBuffer null_buffer;
static std::ostream nullstr(&null_buffer);

#undef CHECK_EQ
#undef CHECK_GT
#undef CHECK_GE
#undef CHECK_LT
#undef CHECK_LE
#undef CHECK_NOTNULL
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_NEAR

#include <iostream>

#define CHECK_EQ(A, B)                                                   \
  (A == B) ? nullstr                                                     \
           : std::cerr << __FILE__ << ":" << __LINE__ << "CHECK_EQ" << A \
                       << " " << B << std::endl
#define CHECK_NE(A, B) \
  (A != B)             \
      ? nullstr        \
      : std::cerr << __FILE__ << ":" << __LINE__ << "CHECK_NE" << std::endl
#define CHECK_GT(A, B)                                                         \
  (A > B) ? nullstr                                                            \
          : std::cerr << __FILE__ << ":" << __LINE__ << "CHECK_GT" << A << " " \
                      << B << std::endl
#define CHECK_GE(A, B)                                                   \
  (A >= B) ? nullstr                                                     \
           : std::cerr << __FILE__ << ":" << __LINE__ << "CHECK_GE" << A \
                       << " " << B << std::endl
#define CHECK_LT(A, B)                                                         \
  (A < B) ? nullstr                                                            \
          : std::cerr << __FILE__ << ":" << __LINE__ << "CHECK_LT" << A << " " \
                      << B << std::endl
#define CHECK_LE(A, B)                                                   \
  (A <= B) ? nullstr                                                     \
           : std::cerr << __FILE__ << ":" << __LINE__ << "CHECK_LE" << A \
                       << " " << B << std::endl
#define CHECK_NOTNULL(X)                                                       \
  (X != nullptr) ? nullstr                                                     \
                 : std::cerr << __FILE__ << ":" << __LINE__ << "CHECK_NOTNULL" \
                             << std::endl
#define CHECK_NEAR(val1, val2, margin)   \
  do {                                   \
    CHECK_LE((val1), (val2) + (margin)); \
    CHECK_GE((val1), (val2) - (margin)); \
  } while (0)

#endif