cmake_minimum_required(VERSION 3.22)
project(how_to_pass_art_classes)

set(CMAKE_CXX_STANDARD 20)

set(GCC_COVERAGE_LINK_FLAGS "-lm -lpthread -lX11 -ljpeg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

include_directories(images)
add_subdirectory(images)
set(SOURCE_FILES app/main.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} images_helper)



