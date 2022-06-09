cmake_minimum_required(VERSION 3.22)
project(images_helper)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES imagesHelper.cpp datasetManager.cpp datasetManager.h)
include_directories(../../lib/CImg-3.1.3)
add_library(images_helper ${SOURCE_FILES})

target_include_directories(images_helper
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )


