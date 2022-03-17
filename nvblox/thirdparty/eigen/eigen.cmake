include(ExternalProject)

# If the caller wants Eigen installed somewhere, do it, otherwise don't configure and don't install (eigen is header only).
if(EIGEN_INCLUDE_DESTINATION)
    set(EIGEN_INCLUDE_DIRS "${EIGEN_INCLUDE_DESTINATION}/include/eigen3")
    file(MAKE_DIRECTORY ${EIGEN_INCLUDE_DIRS})
    set(EIGEN_CONFIGURE_CMD)
    set(EIGEN_INSTALL_CMD make install)
else()
    set(EIGEN_CONFIGURE_CMD echo 'not installing eigen')
    set(EIGEN_INSTALL_CMD echo 'not installing eigen')
endif()

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${EIGEN_CONFIGURE_CMD}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EIGEN_INCLUDE_DESTINATION}
    BUILD_COMMAND ""
    INSTALL_COMMAND ${EIGEN_INSTALL_CMD}
)

if(NOT EIGEN_INCLUDE_DESTINATION)
    ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
    set(EIGEN_INCLUDE_DIRS ${SOURCE_DIR})
endif()
