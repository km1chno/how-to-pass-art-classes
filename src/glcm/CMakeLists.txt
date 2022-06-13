add_library(glcm_lib glcm.h glcm.cpp)

target_include_directories(glcm_lib
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )


