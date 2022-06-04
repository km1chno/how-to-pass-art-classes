cmake_minimum_required(VERSION 3.22)
project(images_helper)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES imagesHelper.cpp ${PROJECT_SOURCE_DIR}/../../lib/CImg-3.1.3/CImg.h)
add_library(images_helper ${SOURCE_FILES})
include_directories(../../lib/CImg-3.1.3)
link_directories(../../lib/CImg-3.1.3)


