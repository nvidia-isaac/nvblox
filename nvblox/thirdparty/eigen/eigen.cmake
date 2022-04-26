include(ExternalProject)

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)

add_library(nvblox_eigen INTERFACE)

target_include_directories(nvblox_eigen INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/eigen/include/eigen3/>
    $<INSTALL_INTERFACE:include/eigen3>
)
add_dependencies(nvblox_eigen ext_eigen)
