include_directories(../../lib/)
add_library(pca_lib pca.h pca.cpp)

target_include_directories(pca_lib
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

