add_library(trees_lib treeNode.h tree.h tree.cpp)

target_include_directories(glcm_lib
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )


