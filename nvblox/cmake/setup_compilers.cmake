# ##############################################################################
# Handle user options
# ##############################################################################

# GPU architectures By default this flag is NOT set. Cmake then detects the
# architecture of the build computer and compiles for that architecture only.
# This can be an issue if you're building on one machine, and running on
# machines with a different GPU achitecture. In this case, set the flag. The
# penalty for doing this is increased build times.
option(BUILD_FOR_ALL_ARCHS "Build for all GPU architectures" OFF)

# This option avoids any implementations using std::string in their signature in
# header files Useful for Nvblox PyTorch wrapper, which requires the old
# Pre-CXX11 ABI
option(PRE_CXX11_ABI_LINKABLE "Better support pre-C++11 ABI library users" OFF)

# Treat warnings as errors on an opt-in basis. This flag should be enabled in CI
# and is also recommended for developers. Reason for opt-in is to avoid
# nuisances for users with compilers different from the one the lib was tested
# on.
option(WARNING_AS_ERROR "Treat compiler warnings as errors")

option(USE_SANITIZER "Enable gcc sanitizer")


# Set default build type if not provided by user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE
        STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo"
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# ##############################################################################
# Specify the C++ standard and general options
# ##############################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# This link flag replaces "runpath" with "rpath" in executables and shared
# objects. This is important because it means the search paths are passed down
# the shared object tree.
# https://stackoverflow.com/questions/58997230/cmake-project-fails-to-find-shared-library
set(nvblox_link_options "-Wl,--disable-new-dtags")

# Cmake -fPIC flag.
# NOTE(alexmillane): I needed to add this when I changed to linking against a
# static version of stdgpu. Without is we get an error from thrust/cub linking.
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# ##############################################################################
# Setup compiler definitions
# ##############################################################################
add_compile_definitions(
  "$<$<BOOL:${PRE_CXX11_ABI_LINKABLE}>:PRE_CXX11_ABI_LINKABLE>")

# Change namespace cub:: into nvblox::cub. This is to avoid conflicts when other modules calls non
# thread safe functions in the cub namespace. Appending nvblox:: ensures an unique symbol that is
# only accesed by this library.
add_compile_definitions(CUB_WRAPPED_NAMESPACE=nvblox)

# ##############################################################################
# Setup c++ compiler flags
# ##############################################################################

# The common flags are used for both g++ and nvcc when compiling host code
set(CXX_FLAGS_COMMON
   # Enable more warnings
   "-Wall"
   "-Wextra"
   "-Wshadow"
   # Facilitate stack-trace debugging
   "-fno-omit-frame-pointer")

if (USE_SANITIZER)
  set(CXX_FLAGS_COMMON "${CXX_FLAGS_COMMON}" "-fsanitize=address")
  set(nvblox_link_options "${nvblox_link_options}" "-fsanitize=address")
endif(USE_SANITIZER)

set(CXX_FLAGS_DEBUG "${CXX_FLAGS_COMMON}" "-O0")
set(CXX_FLAGS_RELWITHDEBINFO "${CXX_FLAGS_COMMON}" "-O2" "-g")
set(CXX_FLAGS_RELEASE "${CXX_FLAGS_COMMON}" "-O3" "-DNDEBUG")

add_compile_options(
  "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${WARNING_AS_ERROR}>>:-Werror>")
add_compile_options(
  "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:${CXX_FLAGS_DEBUG}>")
add_compile_options(
  "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:${CXX_FLAGS_RELWITHDEBINFO}>"
)
add_compile_options(
  "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:${CXX_FLAGS_RELEASE}>")

# ##############################################################################
# Setup cuda compiler flags
# ##############################################################################

# Only used if the BUILD_FOR_ALL_ARCHS flag above is true.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  set(CUDA_ARCHITECTURE_FLAGS "87;86;80;75;72;70;61;60")
else()
  set(CUDA_ARCHITECTURE_FLAGS
      " -gencode=arch=compute_87,code=sm_87 \
        -gencode=arch=compute_86,code=sm_86 \
        -gencode=arch=compute_80,code=sm_80 \
        -gencode=arch=compute_75,code=sm_75 \
        -gencode=arch=compute_72,code=sm_72 \
        -gencode=arch=compute_70,code=sm_70 \
        -gencode=arch=compute_61,code=sm_61 \
        -gencode=arch=compute_60,code=sm_60")
endif()

# When nvcc passes args to the native c++ compiler, it requires a comma
# separated list.
string(REPLACE ";" "," CXX_FLAGS_COMMON_COMMA_SEPARATED "${CXX_FLAGS_COMMON}")

set(CUDA_FLAGS_COMMON
    "${CMAKE_CUDA_FLAGS}"
    # Allow __host__, __device__ annotations in lambda declarations
    "--extended-lambda"
    # Allows sharing constexpr between host and device code
    "--expt-relaxed-constexpr"
    # Display warning numbers
    "-Xcudafe=--display_error_number"
    # Increased visibility of symbols
    "-Xcompiler=${CXX_FLAGS_COMMON_COMMA_SEPARATED}"
    # Suppress "dynamic initialization is not supported for a function-scope static __shared__
    # variable within a __device__/__global__ function". We cannot call the constructor in these
    # cases due to race condition. To my understanding, the variables are left un-constructed which
    # is still OK for our use case.
    "--diag-suppress=20054"
    # Suppress "a __constant__ variable cannot be directly read in a host function". We share
    # __constant__ between host and device in the marching cubes implementation.
    "--diag-suppress=20091"
  )

set(CUDA_FLAGS_DEBUG "${CUDA_FLAGS_COMMON}" "--debug" "--device-debug" "-O0")
set(CUDA_FLAGS_RELWITHDEBINFO "${CUDA_FLAGS_COMMON}" "-O2"
                              "--debug" "--generate-line-info")
set(CUDA_FLAGS_RELEASE "${CUDA_FLAGS_COMMON}" "-O3" "-DNDEBUG")
add_compile_options(
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${WARNING_AS_ERROR}>>:-Xcompiler=-Werror>")
add_compile_options(
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUDA_FLAGS_DEBUG}>")
add_compile_options(
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:${CUDA_FLAGS_RELWITHDEBINFO}>")
add_compile_options(
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:${CUDA_FLAGS_RELEASE}>")
